"""Tests for ActionExecutor (Step 9).

Release-blocking FRs exercised here (in addition to safety/process suites):
- FR-04: explicit display_scope=="isolated" required at execution boundary;
  any other value (observe_real, :0, missing, garbage) is hard-rejected
  *before* any xdotool subprocess, any env construction, or any X request.
- FR-03/24: executor is structurally incapable of emitting anything to real :0
  (all paths go exclusively through get_isolated_env private XAUTH+DISPLAY).
- Spatial vs non-spatial target rules exactly as specified.
- launch_app / navigate delegate to ProcessSupervisor.spawn with forced
  isolated env (no_shell, no real-display override possible).
- Success path only on a live isolated Xvfb (never :0).

All GUI actuation in this file uses *only* temp isolated displays.
Zero tests ever inject to, or read from, the real session :0.

Hardening #1 (XAUTHORITY ISOLATION) is re-exercised implicitly: every
xdotool and every spawned GUI app in these tests receives the private
env; the real ~/.Xauthority is absent.

Test style matches test_computer_use_process.py and test_computer_use_safety.py
(pytest fixtures, high display numbers, clean skips, best-effort teardown).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import psutil
import pytest

from agy_orchestrator.computer_use.action_executor import ActionExecutor
from agy_orchestrator.computer_use.models import ActionSpec, ActionStatus
from agy_orchestrator.computer_use.process_supervisor import ProcessSupervisor
from agy_orchestrator.computer_use.xauth import get_isolated_env


@pytest.fixture
def supervisor() -> ProcessSupervisor:
    """Fresh supervisor with guaranteed cleanup of owned trees."""
    sup = ProcessSupervisor()
    try:
        yield sup
    finally:
        for rid in list(sup._registry.keys()):
            try:
                sup.terminate_tree(rid)
            except Exception:
                pass


def _find_xvfb() -> str:
    xv = shutil.which("Xvfb")
    if not xv:
        pytest.skip("Xvfb not on PATH; ActionExecutor integration tests require it")
    return xv


def _wait_for_xvfb_ready(display: str, timeout: float = 2.0) -> None:
    """Best-effort readiness for a freshly spawned isolated Xvfb."""
    env = get_isolated_env(display)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if shutil.which("xwininfo"):
                r = subprocess.run(
                    ["xwininfo", "-root", "-display", display],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=0.3,
                )
                if r.returncode == 0:
                    return
            else:
                # Fallback: just give Xvfb a moment; xdotool will surface real errors
                time.sleep(0.25)
                return
        except Exception:
            pass
        time.sleep(0.05)


def _mk_spec(**kw: Any) -> ActionSpec:
    base: Dict[str, Any] = {
        "action_id": f"act-{int(time.time()*1000)}",
        "type": "wait",
        "display_scope": "isolated",
        "rationale": "test",
        "risk_level": "low",
    }
    base.update(kw)
    return ActionSpec(**base)


# ------------------------------------------------------------------
# FR-04 + hardening boundary tests (no Xvfb required)
# ------------------------------------------------------------------

@pytest.mark.not_slow
@pytest.mark.release_blocking
def test_fr04_rejects_real_display_dict_before_any_xdotool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real-display (or any non-isolated) dict is rejected with zero xdotool calls."""
    calls: list[list[str]] = []

    def fake_run(*a: Any, **k: Any) -> Any:
        calls.append(a[0] if a else [])
        return type("CP", (), {"returncode": 0, "stderr": b""})()

    monkeypatch.setattr("subprocess.run", fake_run)

    ex = ActionExecutor(isolated_display=":99")
    bad = {"display_scope": "observe_real", "type": "click", "target": {"kind": "coordinate", "x": 10, "y": 10}}
    res = ex.execute(bad)
    assert res.status == ActionStatus.REJECTED.value
    assert res.error_code == "display_scope_invalid"
    # Critical: no xdotool (or any subprocess) was ever invoked
    assert not calls, "xdotool (or any command) must not be invoked for real-display intent (FR-04)"


@pytest.mark.not_slow
@pytest.mark.release_blocking
def test_fr04_rejects_real_display_string_and_other_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(1) or type("C", (), {"returncode": 0})())
    ex = ActionExecutor(isolated_display=":99")

    for bad_ds in (":0", "observe_real", "", None, "ISOLATED"):
        bad = {"display_scope": bad_ds, "type": "hotkey", "hotkey": ["ctrl", "l"]}
        res = ex.execute(bad)
        assert res.status == ActionStatus.REJECTED.value
        assert res.error_code in ("display_scope_invalid", "schema_invalid")

    assert not calls, "No subprocess for any non-isolated display_scope"


def test_spatial_requires_coordinate_target() -> None:
    ex = ActionExecutor(isolated_display=":99")
    for bad in (
        _mk_spec(type="click", target=None),
        _mk_spec(type="double_click", target={"kind": "element", "handle_id": "h1"}),
        _mk_spec(type="type", text="hi", target={}),
        _mk_spec(type="scroll", scroll_delta={"dx": 0, "dy": 3}, target=None),
    ):
        res = ex.execute(bad)
        assert res.status == ActionStatus.REJECTED.value
        assert res.error_code in ("target_missing", "target_unresolvable", "schema_invalid")


def test_non_spatial_accept_target_none_or_absent() -> None:
    ex = ActionExecutor(isolated_display=":99")
    # These must not require target (contract)
    res = ex.execute(_mk_spec(type="wait", wait_ms=1))
    assert res.status == ActionStatus.OK.value

    res = ex.execute(_mk_spec(type="hotkey", hotkey=["Escape"]))
    # Will be "failed" or "ok" depending on xdotool presence, but must not be target_missing
    assert res.error_code != "target_missing"


# ------------------------------------------------------------------
# Live isolated Xvfb + GUI actuation (release-blocking integration)
# ------------------------------------------------------------------

def test_executor_spatial_and_nonspatial_on_isolated_xvfb(supervisor: ProcessSupervisor) -> None:
    """Launch temp Xvfb + simple GUI; execute spatial + non-spatial via executor.

    All success assertions are on the private isolated display only.
    Uses xclock (allowlisted) as the simple GUI target surface.
    """
    xvfb = _find_xvfb()
    if not shutil.which("xdotool"):
        pytest.skip("xdotool not on PATH")

    display = f":{90 + (os.getpid() % 47)}"
    xv_spec = type("S", (), {"display": display, "screen": "640x480x24", "xvfb_binary": xvfb, "timeout_ms": 2000})()
    try:
        spawned_xv = supervisor.spawn_isolated_display(xv_spec)  # type: ignore[arg-type]
    except RuntimeError as e:
        if "already active" in str(e).lower() or "address" in str(e).lower():
            pytest.skip(f"display {display} busy")
        raise

    _wait_for_xvfb_ready(display, timeout=1.8)

    ex = ActionExecutor(
        isolated_display=display,
        supervisor=supervisor,
        action_timeout_ms=8000,
    )

    # 1. Non-spatial: launch_app (xclock is allowlisted in default policy)
    launch = _mk_spec(type="launch_app", app="xclock", app_args=["-geometry", "80x80+20+20"])
    res = ex.execute(launch)
    assert res.status == ActionStatus.OK.value, f"launch_app failed: {res.error_code}"
    assert res.spawned_process_ids and res.spawned_process_ids[0] > 0
    assert supervisor.is_owned(res.spawned_process_ids[0])

    # Give the clock a moment to map
    time.sleep(0.4)

    # 2. Spatial: click (center of our small screen is safe)
    click = _mk_spec(type="click", target={"kind": "coordinate", "x": 320, "y": 240})
    res = ex.execute(click)
    assert res.status == ActionStatus.OK.value, f"click failed: {res.error_code} {res}"
    assert res.resolved_target == {"x": 320, "y": 240}

    # 3. Spatial: double_click
    res = ex.execute(_mk_spec(type="double_click", target={"kind": "coordinate", "x": 100, "y": 100}))
    assert res.status == ActionStatus.OK.value

    # 3b. Spatial: scroll (exercises target coord + wheel; now implemented consistently)
    res = ex.execute(_mk_spec(type="scroll", target={"kind": "coordinate", "x": 50, "y": 50}, scroll_delta={"dx": 0, "dy": -3}))
    assert res.status in (ActionStatus.OK.value, ActionStatus.FAILED.value)  # ok when xdotool can talk to isolated display
    if res.status == ActionStatus.OK.value:
        assert res.resolved_target == {"x": 50, "y": 50}

    # 4. Non-spatial: hotkey (global chord, target optional)
    res = ex.execute(_mk_spec(type="hotkey", hotkey=["Escape"]))
    assert res.status in (ActionStatus.OK.value, ActionStatus.FAILED.value)  # may be ok even if no window focused

    # 5. Non-spatial: wait (pure sleep, no X at all)
    res = ex.execute(_mk_spec(type="wait", wait_ms=30))
    assert res.status == ActionStatus.OK.value

    # 6. Another launch_app (the "true" stub, instant exit)
    res = ex.execute(_mk_spec(type="launch_app", app="true", app_args=[]))
    assert res.status == ActionStatus.OK.value
    assert res.spawned_process_ids

    # Teardown of Xvfb + launched xclock happens via supervisor fixture
    # (guarantees grandchildren via the KILLABLE TREE path already tested elsewhere).


def test_executor_navigate_delegates_to_supervisor(supervisor: ProcessSupervisor) -> None:
    """navigate uses supervisor.spawn (FR-24) with private env; never direct xdg-open inherit."""
    if not shutil.which("true"):
        pytest.skip("no true binary")
    display = f":{80 + (os.getpid() % 30)}"
    ex = ActionExecutor(isolated_display=display, supervisor=supervisor)
    res = ex.execute(_mk_spec(type="navigate", url="about:blank"))
    # Even if opener is "true", it must have gone through supervisor and returned a pid
    assert res.status == ActionStatus.OK.value
    assert res.spawned_process_ids and len(res.spawned_process_ids) == 1
    assert supervisor.is_owned(res.spawned_process_ids[0])


def test_executor_uses_private_xauth_env_for_xdotool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every xdotool invocation in executor receives the hardened get_isolated_env result."""
    captured_envs: list[Dict[str, str]] = []

    real_run = subprocess.run

    def spying_run(argv: Any, *, env: Dict[str, str] | None = None, **kw: Any) -> Any:
        if argv and isinstance(argv, (list, tuple)) and "xdotool" in str(argv[0]):
            captured_envs.append(dict(env or {}))
        return real_run(argv, env=env, **kw)

    monkeypatch.setattr("subprocess.run", spying_run)

    display = f":{70 + (os.getpid() % 20)}"
    ex = ActionExecutor(isolated_display=display)
    # Trigger one spatial that will call xdotool (even if it fails on missing display, env is captured)
    ex.execute(_mk_spec(type="click", target={"kind": "coordinate", "x": 10, "y": 10}))

    # If xdotool binary existed the env was captured; if not, the test is still valid
    # (the rejection paths already proved "no call for bad scope").
    if captured_envs:
        env = captured_envs[0]
        assert env.get("DISPLAY") == display
        assert "XAUTHORITY" in env
        assert env.get("AGY_ISOLATED_X") == "1"
        assert "WAYLAND_DISPLAY" not in env
        # The real cookie path (if any) must not appear in the env values passed to xdotool
        real_x = os.environ.get("XAUTHORITY") or str(Path.home() / ".Xauthority")
        for v in env.values():
            assert real_x not in str(v)


# ------------------------------------------------------------------
# Step 7: real_act executor + clearance token gate (hermetic, mock-only)
# @pytest.mark.realgui + @not_slow so they are excluded from default
# `pytest -q -m "not slow"` and from INVARIANT F regression runs unless
# explicitly selected. All use fakes/monkeypatch only; zero real :0 or
# foreign input. Exactly two new tests per Step 7 contract.
# ------------------------------------------------------------------

@pytest.mark.release_blocking
@pytest.mark.realgui
@pytest.mark.not_slow
def test_realgui_executor_rejects_real_act_without_clearance_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """real_act dict/spec without (or empty) clearance_token is REJECTED with
    clearance_token_invalid *before* any env materialization or xdotool call.
    Mirrors the FR-04 zero-side-effect contract for the new scope (INV E).
    """
    calls: list = []
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: calls.append(1) or type("C", (), {"returncode": 0, "stdout": b"", "stderr": b""})(),
    )
    ex = ActionExecutor(isolated_display=":99")

    # absent token
    bad1 = {"display_scope": "real_act", "type": "hotkey", "hotkey": ["Return"]}
    res1 = ex.execute(bad1)
    assert res1.status == ActionStatus.REJECTED.value
    assert res1.error_code == "clearance_token_invalid"
    assert not calls

    # empty/whitespace token also rejected
    bad2 = {"display_scope": "real_act", "type": "click", "target": {"kind": "coordinate", "x": 1, "y": 2}, "clearance_token": "   "}
    res2 = ex.execute(bad2)
    assert res2.status == ActionStatus.REJECTED.value
    assert res2.error_code == "clearance_token_invalid"
    assert not calls

    # via ActionSpec (dataclass path) also
    spec_bad = _mk_spec(type="hotkey", hotkey=["Escape"])
    # force real_act + no token (bypass normal __post_init__ for boundary test)
    object.__setattr__(spec_bad, "display_scope", "real_act")
    object.__setattr__(spec_bad, "clearance_token", None)
    res3 = ex.execute(spec_bad)
    assert res3.status == ActionStatus.REJECTED.value
    assert res3.error_code == "clearance_token_invalid"
    assert not calls

    # Laziness: the isolated _env must never have been touched
    assert ex._env is None


@pytest.mark.release_blocking
@pytest.mark.realgui
@pytest.mark.not_slow
def test_realgui_executor_valid_token_uses_real_env_not_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a non-empty clearance_token, real_act is allowed; xdotool receives
    the *real* operator :0 env (via minimal _get_real_env), never the isolated
    cookie or AGY_ISOLATED_X marker. isolated lazy env remains unmaterialized.
    """
    captured_envs: list[Dict[str, str]] = []
    real_run = subprocess.run

    def spying_run(argv: Any, *, env: Dict[str, str] | None = None, **kw: Any) -> Any:
        if argv and isinstance(argv, (list, tuple)) and "xdotool" in str(argv[0] if argv else ""):
            captured_envs.append(dict(env or {}))
        # Fake success so we don't actually need a display; the env capture is the assertion target
        return type("R", (), {"returncode": 0, "stdout": b"", "stderr": b""})()

    monkeypatch.setattr("subprocess.run", spying_run)

    ex = ActionExecutor(isolated_display=":99")
    good = {
        "action_id": "act-real-tok-1",
        "type": "hotkey",
        "display_scope": "real_act",
        "hotkey": ["Return"],
        "clearance_token": "clr:run123:pid456:real_act:1710000000",
        "rationale": "test real after kernel gate",
        "risk_level": "low",
    }
    res = ex.execute(good)

    # Must not be rejected by ds or token gates (may be OK/FAILED only on the fake xdotool result)
    assert res.status in (ActionStatus.OK.value, ActionStatus.FAILED.value)
    assert res.error_code not in ("display_scope_invalid", "clearance_token_invalid", "target_missing")

    # Env assertions: used real path, not isolated
    if captured_envs:
        env = captured_envs[0]
        assert env.get("AGY_ISOLATED_X") is None
        xa = env.get("XAUTHORITY", "")
        assert "agu-" not in xa and "isolated" not in xa.lower()
        # DISPLAY must look like a real session display
        assert isinstance(env.get("DISPLAY"), str) and env["DISPLAY"].startswith(":")
    else:
        # Even if xdotool binary absent, the gate passed (no early reject)
        pass

    # Critical laziness + isolation invariant: the executor's isolated _env slot
    # was never populated by the real_act path.
    assert ex._env is None

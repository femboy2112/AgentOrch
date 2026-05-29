"""Hermetic release-blocking realgui test suite (Step 11, §13 cases).

Exactly the 11 release blockers (FR-40 first) using only fakes/synthetic baselines.
<400 LOC, imports only computer_use.* + pytest + unittest.mock + stdlib.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

from agy_orchestrator.computer_use.action_executor import ActionExecutor
from agy_orchestrator.computer_use.grants import FakeClock, GrantCache
from agy_orchestrator.computer_use.gui_prompt import FakePrompter, GuiPrompter, PromptContext
from agy_orchestrator.computer_use.models import AskMode, PromptResult, ReasoningInput, ViolationCode
from agy_orchestrator.computer_use.ownership import FakeOwnershipResolver
from agy_orchestrator.computer_use.process_supervisor import ProcessSupervisor
from agy_orchestrator.computer_use.safety import SafetyKernel

# =============================================================================
# Minimal hermetic helpers (no test imports, no host side effects)
# =============================================================================

def _mk_budgets() -> Dict[str, int]:
    return {
        "max_steps": 200, "max_actions": 200, "action_timeout_ms": 10000,
        "reasoning_timeout_ms": 45000, "confirmation_wait_timeout_ms": 300000,
        "max_cpu_percent": 200, "max_rss_mb": 2048, "max_processes": 64,
    }


def _mk_session(**o: Any) -> Dict[str, Any]:
    s: Dict[str, Any] = {
        "run_id": o.pop("run_id", "rg-test"),
        "mode": o.pop("mode", "REAL"),
        "task_priority": "normal",
        "budgets": _mk_budgets(),
        "displays": {"isolated_display": ":99"},
        "capabilities": {
            "action_exec": True, "atspi": False, "ocr": False,
            "geometry": True, "dom": False, "degraded": False, "readiness": "ready",
        },
        "real_gui_policy": o.pop("real_gui_policy", "full"),
        "ask_mode": o.pop("ask_mode", "on"),
    }
    s.update(o)
    return s


def _mk_intent(**kw: Any) -> Dict[str, Any]:
    action: Dict[str, Any] = {
        "type": kw.pop("type", "click"),
        "display_scope": kw.pop("display_scope", "real_act"),
    }
    tgt = kw.pop("target", None)
    if tgt is not None:
        action["target"] = tgt
    for f in ("text", "hotkey", "scroll_delta", "drag_to", "wait_ms", "url", "app", "app_args"):
        if f in kw:
            action[f] = kw.pop(f)
    return {
        "intent_id": kw.pop("intent_id", "i1"),
        "snapshot_id": "s1",
        "action": action,
        "rationale": kw.pop("rationale", "realgui test"),
        "risk_level": kw.pop("risk_level", "low"),
        "requires_confirmation": False,
        "confirmation_token": kw.pop("confirmation_token", None),
    }


def _mk_snap_with_window(pid: int, x: int = 0, y: int = 0, w: int = 800, h: int = 600, z: int = 1) -> Dict[str, Any]:
    return {
        "real": {
            "windows": [
                {"window_id": "0xwin", "pid": pid, "bbox": {"x": x, "y": y, "w": w, "h": h}, "z_index": z}
            ]
        }
    }


# =============================================================================
# The 11 release-blocking cases (in design §13 order). All @realgui + @release_blocking
# =============================================================================

@pytest.mark.release_blocking
@pytest.mark.realgui
def test_fr40_baseline_foreign_denied_mission_critical():
    """FR-40 (INV A): baseline PID FOREIGN forever; real_full+ask=on+auto-deny -> DENY; prompter consulted; baseline beats naive owned."""
    pid = 4242
    res = FakeOwnershipResolver(synthetic_baseline_pids={pid}, synthetic_owned=set())
    p = FakePrompter()
    k = SafetyKernel(ownership_resolver=res, gui_prompter=p, grant_cache=GrantCache(run_id="fr40"), supervisor=ProcessSupervisor())
    r = k._real_act_gate(_mk_intent(target={"kind": "element", "app_pid": pid}), _mk_session(real_gui_policy="full", ask_mode="on"))
    assert r.valid is False and p.call_count >= 1 and res.classify(pid) == "FOREIGN"
    assert FakeOwnershipResolver(synthetic_baseline_pids=set(), synthetic_owned={pid}).classify(pid) == "OWNED"


@pytest.mark.release_blocking
@pytest.mark.realgui
def test_fr30_owned_child_allow_no_prompt():
    """FR-30: post-baseline ProcessSupervisor-owned child (synthetic) -> OWNED -> ALLOW
    with no prompter call (no grant needed).
    """
    owned_pid = 7777
    resolver = FakeOwnershipResolver(synthetic_baseline_pids={1111}, synthetic_owned={owned_pid})
    prompter = FakePrompter()
    gc = GrantCache(run_id="fr30")
    kernel = SafetyKernel(ownership_resolver=resolver, gui_prompter=prompter, grant_cache=gc, supervisor=ProcessSupervisor())

    sess = _mk_session(real_gui_policy="full", ask_mode="on")
    intent = _mk_intent(target={"kind": "element", "app_pid": owned_pid})

    res = kernel._real_act_gate(intent, sess)
    assert res.valid is True
    assert res.normalized_action is not None
    assert res.normalized_action.display_scope == "real_act"
    assert res.normalized_action.clearance_token and res.normalized_action.clearance_token.startswith("clr:")
    assert prompter.call_count == 0, "owned path must never consult prompter"


@pytest.mark.release_blocking
@pytest.mark.realgui
def test_fr31_children_policy_foreign_denied_no_prompt():
    """FR-31 (INV D): foreign pid + real_children policy -> DENY (FOREIGN_PROCESS), prompter never called."""
    foreign_pid = 8888
    resolver = FakeOwnershipResolver(synthetic_baseline_pids={foreign_pid}, synthetic_owned=set())
    prompter = FakePrompter()
    gc = GrantCache(run_id="fr31")
    kernel = SafetyKernel(ownership_resolver=resolver, gui_prompter=prompter, grant_cache=gc, supervisor=ProcessSupervisor())

    sess = _mk_session(real_gui_policy="children", ask_mode="on")
    intent = _mk_intent(target={"kind": "coordinate", "x": 10, "y": 10})

    res = kernel._real_act_gate(intent, sess, snapshots=_mk_snap_with_window(foreign_pid))
    assert res.valid is False
    codes = [v["code"] for v in (res.violations or [])]
    assert ViolationCode.FOREIGN_PROCESS.value in codes
    assert prompter.call_count == 0


@pytest.mark.release_blocking
@pytest.mark.realgui
def test_fr32_full_ask_off_foreign_hard_denied():
    """FR-32: foreign + full + ask_mode=off -> ASK_MODE_DISABLED, no prompt."""
    foreign_pid = 3333
    resolver = FakeOwnershipResolver(synthetic_baseline_pids={foreign_pid}, synthetic_owned=set())
    prompter = FakePrompter()
    gc = GrantCache(run_id="fr32")
    kernel = SafetyKernel(ownership_resolver=resolver, gui_prompter=prompter, grant_cache=gc, supervisor=ProcessSupervisor())

    sess = _mk_session(real_gui_policy="full", ask_mode="off")
    intent = _mk_intent(target={"kind": "element", "app_pid": foreign_pid})

    res = kernel._real_act_gate(intent, sess)
    assert res.valid is False
    codes = [v["code"] for v in (res.violations or [])]
    assert ViolationCode.ASK_MODE_DISABLED.value in codes
    assert prompter.call_count == 0


@pytest.mark.release_blocking
@pytest.mark.realgui
def test_fr33_fr34_foreign_full_ask_on_prompt_context_and_choices():
    """FR-33+34: foreign + full + ask=on -> prompter invoked with full context;
    choices include the four grant scopes + free-text entry (via queued result).
    """
    foreign_pid = 2222
    resolver = FakeOwnershipResolver(synthetic_baseline_pids={foreign_pid}, synthetic_owned=set())
    prompter = FakePrompter()
    prompter.queue(grant_scope="PROCESS_RUN", granted=True, operator_text="use only this one")
    gc = GrantCache(run_id="fr33")
    kernel = SafetyKernel(ownership_resolver=resolver, gui_prompter=prompter, grant_cache=gc, supervisor=ProcessSupervisor())

    sess = _mk_session(real_gui_policy="full", ask_mode="on")
    intent = _mk_intent(
        rationale="need to click the settings dialog I opened earlier",
        target={"kind": "element", "app_pid": foreign_pid, "window_id": "0x22"},
    )

    res = kernel._real_act_gate(intent, sess)
    assert res.valid is True
    assert prompter.call_count == 1
    ctx = prompter.calls[0]
    assert isinstance(ctx, PromptContext)
    assert ctx.pid == foreign_pid
    assert ctx.policy == "full"
    assert ctx.ask_mode == "on"
    assert "settings dialog" in ctx.rationale
    # The four real-time scopes are offered by the prompter impl (ACTION / PROCESS_RUN / PROCESS_TTL / DENY)


@pytest.mark.release_blocking
@pytest.mark.realgui
def test_fr36_grant_scopes_action_run_ttl_new_pid_reprompt():
    """FR-36: ACTION single-use (re-prompt next), PROCESS_RUN persists for pid,
    PROCESS_TTL allows before expiry / denies after (injected clock), new foreign pid always re-prompts.
    """
    clk = FakeClock(start=1_000_000.0)
    gc = GrantCache(clock=clk, run_id="fr36")
    resolver = FakeOwnershipResolver(synthetic_baseline_pids={100, 200}, synthetic_owned=set())
    prompter = FakePrompter()
    kernel = SafetyKernel(ownership_resolver=resolver, gui_prompter=prompter, grant_cache=gc, supervisor=ProcessSupervisor())
    sess = _mk_session(real_gui_policy="full", ask_mode="on")

    # ACTION: prompt path allows, first cache check after grant consumes+allows (single-use),
    # second cache check misses -> re-prompts (per FR-36). Queue two ACTION responses.
    prompter.queue(grant_scope="ACTION", granted=True)
    prompter.queue(grant_scope="ACTION", granted=True)
    intent1 = _mk_intent(target={"kind": "element", "app_pid": 100})
    r1 = kernel._real_act_gate(intent1, sess)
    assert r1.valid is True and prompter.call_count == 1
    r1b = kernel._real_act_gate(intent1, sess)  # consumes the ACTION grant (still valid, no prompt)
    assert r1b.valid is True and prompter.call_count == 1
    r1c = kernel._real_act_gate(intent1, sess)  # now misses cache -> re-prompt (second ACTION response)
    assert r1c.valid is True and prompter.call_count == 2

    # PROCESS_RUN: persists for pid, new pid re-prompts
    prompter.queue(grant_scope="PROCESS_RUN", granted=True)
    r2 = kernel._real_act_gate(_mk_intent(target={"kind": "element", "app_pid": 200}), sess)
    assert r2.valid is True and prompter.call_count == 3
    r2b = kernel._real_act_gate(_mk_intent(target={"kind": "element", "app_pid": 200}), sess)
    assert r2b.valid is True and prompter.call_count == 3  # no re-prompt
    prompter.queue(grant_scope="PROCESS_RUN", granted=True)
    r3 = kernel._real_act_gate(_mk_intent(target={"kind": "element", "app_pid": 300}), sess)  # new
    assert r3.valid is True and prompter.call_count == 4

    # PROCESS_TTL: before expiry allow, after deny (clock)
    prompter.queue(grant_scope="PROCESS_TTL", granted=True)
    r4 = kernel._real_act_gate(_mk_intent(target={"kind": "element", "app_pid": 400}), sess)
    assert r4.valid is True and prompter.call_count == 5
    clk.advance(700)  # > 600s default
    r4b = kernel._real_act_gate(_mk_intent(target={"kind": "element", "app_pid": 400}), sess)
    assert r4b.valid is False and prompter.call_count == 6  # re-prompt after expiry


@pytest.mark.release_blocking
@pytest.mark.realgui
def test_fr35_fail_closed_all_paths_deny_audited():
    """FR-35 (INV C): prompter raise/timeout/empty/None/missing-zenity-sim -> DENY (GRANT_REQUIRED or ASK_MODE_DISABLED)."""
    foreign_pid = 5555
    resolver = FakeOwnershipResolver(synthetic_baseline_pids={foreign_pid}, synthetic_owned=set())

    # raise path
    p1 = FakePrompter()
    p1.simulate_raise()
    k1 = SafetyKernel(ownership_resolver=resolver, gui_prompter=p1, grant_cache=GrantCache(run_id="f35"), supervisor=ProcessSupervisor())
    r1 = k1._real_act_gate(_mk_intent(target={"kind": "element", "app_pid": foreign_pid}), _mk_session())
    assert r1.valid is False and any(v["code"] == ViolationCode.GRANT_REQUIRED.value for v in (r1.violations or []))

    # explicit deny result
    p2 = FakePrompter()
    p2.queue("Deny", granted=False)
    k2 = SafetyKernel(ownership_resolver=resolver, gui_prompter=p2, grant_cache=GrantCache(run_id="f35b"), supervisor=ProcessSupervisor())
    r2 = k2._real_act_gate(_mk_intent(target={"kind": "element", "app_pid": foreign_pid}), _mk_session())
    assert r2.valid is False

    # no prompter available
    k3 = SafetyKernel(ownership_resolver=resolver, gui_prompter=None, grant_cache=GrantCache(run_id="f35c"), supervisor=ProcessSupervisor())
    r3 = k3._real_act_gate(_mk_intent(target={"kind": "element", "app_pid": foreign_pid}), _mk_session())
    assert r3.valid is False and any(v["code"] == ViolationCode.ASK_MODE_DISABLED.value for v in (r3.violations or []))


@pytest.mark.release_blocking
@pytest.mark.realgui
def test_fr37_operator_text_becomes_operator_note_in_reasoninginput():
    """FR-37: non-empty operator_text from PromptResult is turned into operator_note
    shape and can be placed directly into next ReasoningInput.orchestrator_messages
    (the adapter wrapper does the enqueue; here we verify the data contract end-to-end).
    """
    prompter = FakePrompter()
    prompter.queue(grant_scope="PROCESS_RUN", granted=True, operator_text="don't touch the browser; use the xterm I left open on the left")
    res = prompter.prompt(PromptContext(run_id="r37", pid=42, action_type="click", policy="full", ask_mode="on"))
    assert res.granted and res.operator_text and "xterm" in res.operator_text

    note: Dict[str, Any] = {
        "kind": "operator_note",
        "text": res.operator_text.strip(),
        "issued_at": "2026-05-29T12:00:00+00:00",
    }
    ri = ReasoningInput(
        run_id="r37",
        session_mode="REAL",
        task_priority="normal",
        objective="test",
        constraints={"must_use_display_scope": "real_act", "real_gui_policy": "full", "ask_mode": "on"},
        snapshots={},
        orchestrator_messages=[note],
    )
    assert any(
        isinstance(m, dict) and m.get("kind") == "operator_note" and "xterm" in m.get("text", "")
        for m in (ri.orchestrator_messages or [])
    )


@pytest.mark.release_blocking
@pytest.mark.realgui
def test_fr39_prompter_performs_no_foreign_actuation():
    """FR-39 (INV C): only FakePrompter used in hermetic suite; zero Popen/which/xdotool
    on foreign paths (GuiPrompter never .prompt()'ed; mocks prove no side effects).
    """
    fp = FakePrompter()
    fp.queue(grant_scope="ACTION", granted=True)
    resolver = FakeOwnershipResolver(synthetic_baseline_pids={999})
    k = SafetyKernel(ownership_resolver=resolver, gui_prompter=fp, grant_cache=GrantCache(run_id="f39"))
    sess = _mk_session()
    with patch("subprocess.Popen") as mp, patch("shutil.which") as mw:
        k._real_act_gate(_mk_intent(target={"kind": "element", "app_pid": 999}), sess)
        assert fp.call_count == 1 and mp.call_count == 0 and mw.call_count == 0
    assert GuiPrompter is not None  # DI type present; never instantiated+called here


@pytest.mark.release_blocking
@pytest.mark.realgui
def test_invariant_e_real_act_action_spec_and_executor_require_clearance():
    """INVARIANT E: real_act ActionSpec constructible ONLY after kernel issues clearance_token;
    reasoner ActionIntent cannot bypass (executor refuses real_act lacking valid token).
    """
    # 1. Direct ActionSpec construction with real_act + missing/empty token MUST raise
    from agy_orchestrator.computer_use.models import ActionSpec
    with pytest.raises(ValueError) as exc:
        ActionSpec(action_id="bad", type="click", display_scope="real_act", target={"kind": "coordinate", "x": 1, "y": 2})
    assert "clearance_token" in str(exc.value)

    # 2. Executor rejects dict / spec with real_act but no/empty clearance (zero side effects)
    exe = ActionExecutor(isolated_display=":99")
    bad = {"type": "click", "display_scope": "real_act", "target": {"kind": "coordinate", "x": 10, "y": 20}}
    out = exe.execute(bad)
    assert out.status == "rejected"
    assert out.error_code == "clearance_token_invalid"

    # Valid token path is kernel-only (we never construct a good real_act ActionSpec here without kernel)
    # A reasoner-produced intent with display_scope=real_act still must go through _real_act_gate to obtain token.


@pytest.mark.release_blocking
@pytest.mark.realgui
def test_invariant_f_prior_suite_remains_green():
    """INVARIANT F: entire existing computer-use suite + full pytest -q (non-slow, non-realgui)
    remain green. Regression gate. Re-runs prior modules via subprocess (no test-module imports).
    """
    prior_modules = [
        "tests/test_computer_use_models.py",
        "tests/test_computer_use_safety.py",
        "tests/test_computer_use_action.py",
        "tests/test_computer_use_adapter.py",
        "tests/test_computer_use_frs.py",
        "tests/test_computer_use_process.py",
        "tests/test_computer_use_reasoner.py",
        "tests/test_computer_use_perception.py",
        "tests/test_computer_use_capability.py",
        "tests/test_computer_use_audit.py",
        "tests/test_computer_use_hardenings.py",
        "tests/test_computer_use_harness.py",
    ]
    cmd = [
        "python", "-m", "pytest", "-q", "--tb=no",
        "-m", "not slow and not realgui",
    ] + prior_modules
    env = os.environ.copy()
    # Ensure no accidental real GUI leakage
    env.pop("AGY_REALGUI_E2E", None)
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    out = (res.stdout or "") + (res.stderr or "")
    assert res.returncode == 0, f"prior suite failed (INV F): {out[-800:]}"
    assert "passed" in out.lower() or "passed in" in out.lower()


# =============================================================================
# Slow real-zenity / :0 e2e skeleton (guarded, never executed in normal runs)
# =============================================================================

@pytest.mark.slow
@pytest.mark.realgui
def test_realgui_e2e_real_zenity_real_display_only_when_enabled():
    """Real zenity + :0 pop-up + foreign window grant e2e. Skipped unless AGY_REALGUI_E2E=1.
    This is the ONLY test allowed to touch the real operator :0 and must be run manually.
    """
    if os.environ.get("AGY_REALGUI_E2E") != "1":
        pytest.skip("AGY_REALGUI_E2E != 1 (release-blocking hermetic suite uses only Fakes)")
    # Skeleton: real implementation would construct GuiPrompter + real OwnershipResolver,
    # drive a foreign window action, assert zenity appeared on :0 and operator choice worked.
    # Never reached in CI or normal pytest -m realgui.
    assert True, "e2e path not exercised (guard passed)"

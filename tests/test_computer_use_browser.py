"""Hermetic release-blocking browser/DOM suite (Phase 1.x Step 10, design §7).

All tests in this module use FakeBrowserController or pure mocks only.
No test imports playwright, launches a real browser, or touches :0.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from agy_orchestrator.computer_use.action_executor import ActionExecutor
from agy_orchestrator.computer_use.browser import BrowserController, FakeBrowserController
from agy_orchestrator.computer_use.gui_prompt import FakePrompter
from agy_orchestrator.computer_use.models import (
    ActionStatus,
    RunRequest,
    ViolationCode,
)
from agy_orchestrator.computer_use.ownership import FakeOwnershipResolver
from agy_orchestrator.computer_use.process_supervisor import ProcessSupervisor
from agy_orchestrator.computer_use.safety import SafetyKernel
from harness import cli as harness_cli

pytestmark = [pytest.mark.release_blocking, pytest.mark.browser]


def _mk_dom_action(kind: str, *, app_pid: int, selector: str = "#search h3 a", index: int = 1) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "type": kind,
        "display_scope": "real_act",
        "target": {
            "kind": "dom",
            "selector": selector,
            "index": index,
            "app_pid": app_pid,
        },
        "rationale": "browser dom test",
        "risk_level": "low",
    }
    if kind == "type":
        base["text"] = "hello"
    return {"action": base, "intent_id": "intent-dom", "snapshot_id": "snap-dom", "rationale": "dom act"}


def test_browser_navigate_returns_landed_url_and_owned_pid() -> None:
    bc = FakeBrowserController(display=":88", engine="bing")
    ex = ActionExecutor(isolated_display=":88", supervisor=ProcessSupervisor(), browser_controller=bc)

    result = ex.execute({"type": "navigate", "display_scope": "isolated", "url": "https://bing.com/search?q=agentorch"})

    assert result.status == ActionStatus.OK.value
    assert result.spawned_process_ids == [bc.browser_pid]
    assert result.artifacts == ["landed_url:https://example.com/search?q=test"]
    assert bc.calls and bc.calls[-1]["op"] == "navigate"


def test_dom_click_and_type_route_to_browser_controller_with_new_tab_follow_and_one_based_index() -> None:
    bc = FakeBrowserController(display=":77")
    ex = ActionExecutor(isolated_display=":77", supervisor=ProcessSupervisor(), browser_controller=bc)
    ex.execute({"type": "navigate", "display_scope": "isolated", "url": "https://example.test"})

    click = ex.execute(
        {
            "type": "click",
            "display_scope": "isolated",
            "target": {"kind": "dom", "selector": "#search h3 a", "index": 3},
        }
    )
    assert click.status == ActionStatus.OK.value
    assert click.resolved_target == {"index": 3}
    assert click.artifacts is not None and "child_page_url:https://result3.example.com/" in click.artifacts

    typed = ex.execute(
        {
            "type": "type",
            "display_scope": "isolated",
            "text": "hello world",
            "target": {"kind": "dom", "selector": "input[name=q]", "index": 1},
        }
    )
    assert typed.status == ActionStatus.OK.value
    assert typed.resolved_target == {"index": 1}

    assert bc.calls[1] == {"op": "click_dom", "selector": "#search h3 a", "index": 3}
    assert bc.calls[2]["op"] == "type_dom"
    assert bc.calls[2]["index"] == 1


def test_dom_target_without_browser_controller_is_rejected_browser_not_open() -> None:
    ex = ActionExecutor(isolated_display=":66", supervisor=ProcessSupervisor(), browser_controller=None)

    click = ex.execute(
        {"type": "click", "display_scope": "isolated", "target": {"kind": "dom", "selector": "a", "index": 1}}
    )
    assert click.status == ActionStatus.REJECTED.value
    assert click.error_code == ViolationCode.BROWSER_NOT_OPEN.value

    typed = ex.execute(
        {
            "type": "type",
            "display_scope": "isolated",
            "text": "x",
            "target": {"kind": "dom", "selector": "input", "index": 1},
        }
    )
    assert typed.status == ActionStatus.REJECTED.value
    assert typed.error_code == ViolationCode.BROWSER_NOT_OPEN.value


def test_dom_target_with_controller_but_without_open_browser_is_rejected_browser_not_open() -> None:
    bc = FakeBrowserController(display=":66")
    ex = ActionExecutor(isolated_display=":66", supervisor=ProcessSupervisor(), browser_controller=bc)

    click = ex.execute(
        {"type": "click", "display_scope": "isolated", "target": {"kind": "dom", "selector": "a", "index": 1}}
    )
    assert click.status == ActionStatus.REJECTED.value
    assert click.error_code == ViolationCode.BROWSER_NOT_OPEN.value

    typed = ex.execute(
        {
            "type": "type",
            "display_scope": "isolated",
            "text": "x",
            "target": {"kind": "dom", "selector": "input", "index": 1},
        }
    )
    assert typed.status == ActionStatus.REJECTED.value
    assert typed.error_code == ViolationCode.BROWSER_NOT_OPEN.value


def test_b1_owned_browser_dom_real_act_allows_without_prompt_under_full_and_children() -> None:
    bc = FakeBrowserController()
    resolver = FakeOwnershipResolver(synthetic_baseline_pids=set(), synthetic_owned={bc.browser_pid})
    prompter = FakePrompter()
    kernel = SafetyKernel(
        ownership_resolver=resolver,
        gui_prompter=prompter,
        supervisor=ProcessSupervisor(),
    )

    for policy in ("full", "children"):
        gate = kernel._real_act_gate(
            _mk_dom_action("click", app_pid=bc.browser_pid, index=2),
            {"run_id": "b1", "mode": "REAL", "real_gui_policy": policy, "ask_mode": "on"},
        )
        assert gate.valid is True
        assert gate.normalized_action is not None
        assert gate.normalized_action.display_scope == "real_act"

    assert prompter.call_count == 0


def test_b2_foreign_dom_target_is_blocked_and_performs_zero_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    foreign_pid = 90909
    bc = FakeBrowserController()
    resolver = FakeOwnershipResolver(synthetic_baseline_pids={foreign_pid}, synthetic_owned={bc.browser_pid})
    kernel = SafetyKernel(
        ownership_resolver=resolver,
        gui_prompter=FakePrompter(),
        supervisor=ProcessSupervisor(),
    )

    subprocess_calls: list[Any] = []

    def _fail_if_called(*a: Any, **k: Any) -> Any:
        subprocess_calls.append((a, k))
        raise AssertionError("subprocess.run must not be used for dom controller path")

    monkeypatch.setattr("subprocess.run", _fail_if_called)

    denied = kernel._real_act_gate(
        _mk_dom_action("click", app_pid=foreign_pid, index=1),
        {"run_id": "b2", "mode": "REAL", "real_gui_policy": "children", "ask_mode": "on"},
    )
    assert denied.valid is False
    codes = [v["code"] for v in (denied.violations or [])]
    assert ViolationCode.FOREIGN_PROCESS.value in codes

    assert not subprocess_calls
    assert all(call["op"] != "click_dom" and call["op"] != "type_dom" for call in bc.calls)


def test_b3_close_and_stop_paths_reap_browser_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    sup = MagicMock()
    sup.terminate_tree = MagicMock()

    # Direct BrowserController.close() reaping path
    bc = BrowserController(supervisor=sup, display=":99")
    bc._root_id = "browser-root-1"  # type: ignore[attr-defined]
    bc._browser = MagicMock()  # type: ignore[attr-defined]
    bc._playwright = MagicMock()  # type: ignore[attr-defined]
    bc.close()
    sup.terminate_tree.assert_called_with("browser-root-1")

    # Session close path invokes controller.close() (stop/timeout teardown path)
    class _ReapingFakeBrowser:
        cdp_endpoint = "ws://fake"
        browser_pid = 31337

        def __init__(self, supervisor: Any) -> None:
            self.supervisor = supervisor
            self.closed = False

        def close(self) -> None:
            self.closed = True
            self.supervisor.terminate_tree("browser-root-2")

    from agy_orchestrator.computer_use.session import SessionController

    fake_bc = _ReapingFakeBrowser(sup)
    ctrl = SessionController(supervisor=sup, browser_controller=fake_bc)
    ctrl.create_session(RunRequest(run_id="b3-run", objective="teardown"))
    ctrl.close_session("b3-run")
    assert fake_bc.closed is True
    assert any(call.args and call.args[0] == "browser-root-2" for call in sup.terminate_tree.mock_calls)

    # Timeout/watchdog path: over-budget enforce_limits must reap adopted browser tree.
    sup_timeout = ProcessSupervisor()
    sup_timeout.adopt_owned_process(2222, root_id="browser-root-timeout")
    killed: List[str] = []
    monkeypatch.setattr(sup_timeout, "terminate_tree", lambda rid: killed.append(rid))

    class _FakeProc:
        def children(self, recursive: bool = True) -> List[Any]:
            return []

        def memory_info(self) -> Any:
            return SimpleNamespace(rss=1)

        def cpu_percent(self, interval: float = 0.0) -> float:
            return 0.0

    monkeypatch.setattr("psutil.pid_exists", lambda pid: True)
    monkeypatch.setattr("psutil.Process", lambda pid: _FakeProc())
    sup_timeout.enforce_limits({"budgets": {"max_processes": 0, "max_rss_mb": 2048, "max_cpu_percent": 200}})
    assert "browser-root-timeout" in killed


def test_cli_flags_roundtrip_to_runrequest(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def _fake_dispatch(instruction: str, **kwargs: Any) -> Any:
        captured["instruction"] = instruction
        captured.update(kwargs)
        return SimpleNamespace(success=True)

    monkeypatch.setattr(harness_cli, "dispatch", _fake_dispatch)
    monkeypatch.setattr(harness_cli, "_print_result", lambda _: None)

    rc = harness_cli.main(
        [
            "do",
            "search and open result 3",
            "--generator",
            "computer-use",
            "--computer-use-mode",
            "REAL",
            "--real-gui-policy",
            "children",
            "--ask-mode",
            "off",
            "--browser-engine",
            "bing",
            "--browser-display",
            ":0",
        ]
    )
    assert rc == 0

    req = RunRequest(
        run_id="rr-1",
        objective="obj",
        mode=captured["computer_use_mode"],
        real_gui_policy=captured["real_gui_policy"],
        ask_mode=captured["ask_mode"],
        browser_engine=captured["browser_engine"],
        browser_display=captured["browser_display"],
    )
    assert req.mode == "REAL"
    assert req.real_gui_policy == "children"
    assert req.ask_mode == "off"
    assert req.browser_engine == "bing"
    assert req.browser_display == ":0"


class _FakeLocator:
    """A locator whose click tiers fail until a configured one succeeds.

    `fail_through` lists the tiers ("normal", "force") that raise, simulating a
    JS-only / ungeometried control on a software-GL display where the Playwright
    locator path stalls. The "evaluate" tier is on the page, not the locator.
    """

    def __init__(self, fail_through: set, calls: List[str]) -> None:
        self.fail_through = fail_through
        self.calls = calls

    def scroll_into_view_if_needed(self, timeout: int = 0) -> None:
        self.calls.append("scroll")

    def click(self, timeout: int = 0, force: bool = False) -> None:
        tier = "force" if force else "normal"
        self.calls.append(f"click:{tier}")
        if tier in self.fail_through:
            raise RuntimeError(f"{tier} click failed (locator path stalled)")


class _FakePage:
    def __init__(self, locator: _FakeLocator, calls: List[str], url: str = "http://localhost/#/runs/x") -> None:
        self._locator = locator
        self._calls = calls
        self.url = url

    def locator(self, sel: str) -> "_FakeLocatorNth":
        return _FakeLocatorNth(self._locator)

    def evaluate(self, expr: str, arg: Any = None) -> bool:
        # Mirrors the in-page click tier: returns True (element found + clicked)
        # unless this tier is configured to fail.
        self._calls.append("evaluate:click")
        if "evaluate" in self._locator.fail_through:
            raise RuntimeError("evaluate click failed")
        return True

    def wait_for_timeout(self, ms: int) -> None:
        pass

    def wait_for_load_state(self, state: str, timeout: int = 0) -> None:
        pass


class _FakeLocatorNth:
    def __init__(self, locator: _FakeLocator) -> None:
        self._locator = locator

    def nth(self, idx: int) -> _FakeLocator:
        return self._locator


class _FakeContext:
    def __init__(self, page: _FakePage) -> None:
        self.pages = [page]


def _mk_controller_with_page(fail_through: set) -> "tuple[BrowserController, List[str]]":
    """A BrowserController whose Playwright handles are pure fakes (no real browser)."""
    calls: List[str] = []
    locator = _FakeLocator(fail_through, calls)
    bc = object.__new__(BrowserController)
    page = _FakePage(locator, calls)
    bc._context = _FakeContext(page)  # type: ignore[attr-defined]
    bc._page = page  # type: ignore[attr-defined]
    bc._closed = False  # type: ignore[attr-defined]
    bc._cdp_timeout = 30000.0  # type: ignore[attr-defined]
    return bc, calls


def test_click_dom_leads_with_in_page_evaluate_click() -> None:
    # The evaluate tier is tried FIRST: a direct in-page el.click() that lands on
    # any display (and is the only path that works on software-GL). When it
    # succeeds the trusted locator fallbacks are never touched.
    bc, calls = _mk_controller_with_page(fail_through=set())
    out = bc._click_dom_impl(selector="#presentation-toggle", index=1)
    assert out["ok"] is True
    assert out["click_method"] == "evaluate"
    assert calls == ["evaluate:click"]


def test_click_dom_falls_back_to_force_when_evaluate_cannot_click() -> None:
    # If the in-page click can't act (element rejects synthetic click / not found),
    # fall back to a trusted forced input click.
    bc, calls = _mk_controller_with_page(fail_through={"evaluate"})
    out = bc._click_dom_impl(selector="#x", index=1)
    assert out["ok"] is True
    assert out["click_method"] == "force"
    assert calls == ["evaluate:click", "click:force"]


def test_click_dom_falls_back_to_normal_scrolled_click_last() -> None:
    bc, calls = _mk_controller_with_page(fail_through={"evaluate", "force"})
    out = bc._click_dom_impl(selector="#x", index=1)
    assert out["ok"] is True
    assert out["click_method"] == "normal"
    assert calls == ["evaluate:click", "click:force", "scroll", "click:normal"]


def test_click_dom_reports_actuation_failure_when_all_tiers_fail() -> None:
    bc, _calls = _mk_controller_with_page(fail_through={"normal", "force", "evaluate"})
    out = bc._click_dom_impl(selector="#x", index=1)
    assert out["ok"] is False
    assert out["error_code"] == "dom_actuation_failed"


def _mk_controller_for_collect(browser_pid: int) -> "tuple[BrowserController, List[str]]":
    """A BrowserController with fake Playwright handles + a recording scan, used to
    prove the perceive-path liveness gate (no real browser / pid probe stubbed)."""
    calls: List[str] = []
    bc = object.__new__(BrowserController)
    page = _FakePage(_FakeLocator(set(), calls), calls)
    bc._context = _FakeContext(page)  # type: ignore[attr-defined]
    bc._page = page  # type: ignore[attr-defined]
    bc._closed = False  # type: ignore[attr-defined]
    bc._owner_ident = None  # type: ignore[attr-defined]
    bc._browser_pid = browser_pid  # type: ignore[attr-defined]
    return bc, calls


def test_collect_dom_skips_scan_when_owned_browser_is_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    # A crashed Chromium pid must NOT reach the no-timeout page.evaluate (that wedges
    # the run loop). The gate fails closed: perceive degrades to [].
    bc, calls = _mk_controller_for_collect(browser_pid=999_999)
    monkeypatch.setattr("psutil.pid_exists", lambda pid: False)
    scanned: List[str] = []

    def scan(page: Any) -> List[Any]:
        scanned.append("ran")
        return [SimpleNamespace(name="x")]

    out = bc.collect_dom_elements(scan)
    assert out == []
    assert scanned == []  # scan never invoked against the dead transport


def test_collect_dom_runs_scan_when_owned_browser_is_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    bc, _calls = _mk_controller_for_collect(browser_pid=4242)

    class _Proc:
        def status(self) -> str:
            return "running"

    monkeypatch.setattr("psutil.pid_exists", lambda pid: True)
    monkeypatch.setattr("psutil.Process", lambda pid: _Proc())

    def scan(page: Any) -> List[Any]:
        return [SimpleNamespace(name="ok")]

    out = bc.collect_dom_elements(scan)
    assert [e.name for e in out] == ["ok"]


def test_collect_dom_does_not_gate_when_pid_unknown() -> None:
    # No owned pid (extraction failed / fake-page path): never block on a probe.
    bc, _calls = _mk_controller_for_collect(browser_pid=0)
    out = bc.collect_dom_elements(lambda page: [SimpleNamespace(name="ok")])
    assert [e.name for e in out] == ["ok"]


def test_b4_prior_computer_use_suite_pass_count_stays_exact() -> None:
    root = Path(__file__).resolve().parent.parent
    prior_files = [
        "tests/test_computer_use_action.py",
        "tests/test_computer_use_adapter.py",
        "tests/test_computer_use_audit.py",
        "tests/test_computer_use_capability.py",
        "tests/test_computer_use_frs.py",
        "tests/test_computer_use_hardenings.py",
        "tests/test_computer_use_harness.py",
        "tests/test_computer_use_models.py",
        "tests/test_computer_use_perception.py",
        "tests/test_computer_use_process.py",
        "tests/test_computer_use_realgui.py",
        "tests/test_computer_use_reasoner.py",
        "tests/test_computer_use_safety.py",
        "tests/test_computer_use_session.py",
    ]
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=no", *prior_files]
    completed = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)

    output = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, output[-4000:]

    m = re.search(r"(\d+)\s+passed,\s+(\d+)\s+skipped", output)
    assert m is not None, output
    assert (int(m.group(1)), int(m.group(2))) == (179, 2)

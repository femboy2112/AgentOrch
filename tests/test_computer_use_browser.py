"""Hermetic release-blocking browser/DOM suite (Phase 1.x Step 10, design §7).

All tests in this module use FakeBrowserController or pure mocks only.
No test imports playwright, launches a real browser, or touches :0.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from agy_orchestrator.computer_use import AuditEventSink, ComputerUseWorkerAdapter
from agy_orchestrator.computer_use.action_executor import ActionExecutor
from agy_orchestrator.computer_use.browser import BrowserController, FakeBrowserController
from agy_orchestrator.computer_use.gui_prompt import FakePrompter
from agy_orchestrator.computer_use.models import ActionIntent, ActionStatus, RunRequest, ViolationCode
from agy_orchestrator.computer_use.ownership import FakeOwnershipResolver
from agy_orchestrator.computer_use.process_supervisor import ProcessSupervisor
from agy_orchestrator.computer_use.session import _pick_isolated_display
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

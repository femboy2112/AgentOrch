"""Tiny integration smoke for Step 12 harness wiring of computer-use as standard worker.

- Constructs ComputerUseWorkerAdapter directly (with full mocks for reasoner/perception/safety/executor
  so no Xvfb, no real :0, no CLI spawns, no tesseract).
- Starts a trivial objective run; verifies events.jsonl is written under runs/ and contains lifecycle.
- Confirms is_available() returns a well-formed CapabilityReport.
- Confirms harness.roles recognizes 'computer-use' token (describe + build_role_agent shim path).

All per the exact Step-12 instruction. No real-:0 actions. Release-blocking FR coverage is in the
dedicated computer_use_* test modules.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from agy_orchestrator.computer_use import (
    ComputerUseWorkerAdapter,
    RunRequest,
    is_available,
)
from agy_orchestrator.computer_use.audit import AuditEventSink
from agy_orchestrator.computer_use.models import (
    ActionIntent,
    ActionStatus,
    ConfirmationOutcome,
    ConfirmationOutcomeType,
    ValidationResult,
    WorkerEventType,
)
from harness import roles
from harness import cli as harness_cli
from harness import computer_use_role


def _mk_intent() -> ActionIntent:
    return ActionIntent(
        intent_id="i-smoke",
        snapshot_id="s-smoke",
        action={"type": "wait", "display_scope": "isolated", "wait_ms": 1},
        rationale="smoke test done",
        risk_level="low",
        requires_confirmation=False,
        confidence=0.99,
    )


def _mk_val() -> ValidationResult:
    from agy_orchestrator.computer_use.models import ActionSpec
    spec = ActionSpec(
        action_id="a-smoke",
        type="wait",
        display_scope="isolated",
        wait_ms=1,
        rationale="smoke",
    )
    return ValidationResult(valid=True, normalized_action=spec)


class _FakeReasoner:
    def __init__(self, intents: Optional[List[ActionIntent]] = None):
        self.intents = intents or [_mk_intent()]
        self.calls: List[Dict[str, Any]] = []

    def decide(self, ri: Any, timeout_ms: int = 45000) -> ActionIntent:
        self.calls.append({"objective": getattr(ri, "objective", None)})
        return self.intents.pop(0) if self.intents else _mk_intent()

    def engine_status(self):
        return MagicMock(claude="unavailable", codex="unavailable")


class _FakePerception:
    def snapshot(self, scope: str):
        from agy_orchestrator.computer_use.models import PerceptionSnapshot
        return PerceptionSnapshot(
            snapshot_id=f"s-{scope}", run_id="smoke", mode="ISOLATED", scope=scope,
            captured_at="2026-05-29T00:00:00Z", windows=[], elements=[], raw_text_blocks=[],
        )

    def snapshot_set(self, scopes):
        return {s: self.snapshot(s) for s in scopes}


class _FakeSafety:
    def validate(self, intent, session):
        return _mk_val()

    def classify_risk(self, intent):
        return "low"

    def require_confirmation(self, intent):
        from agy_orchestrator.computer_use.models import GateDecision
        return GateDecision(gate="allow", reason="smoke")

    def resolve_target(self, intent, snapshots):
        return None

    def validate_launch_app(self, payload, policy):
        return MagicMock(valid=True)


class _FakeExecutor:
    def execute(self, action):
        return MagicMock(status=ActionStatus.OK, executed_at="2026-05-29T00:00:01Z")


class _FakeSupervisor:
    def is_owned(self, pid):
        return True

    def terminate_tree(self, root_id):
        pass

    def enforce_limits(self, sess):
        pass


class _FakeSessionController:
    """Hermetic stand-in: no Xvfb spawn, no real cap probe side effects, no real supervisor."""

    def __init__(self):
        self._sessions: Dict[str, Any] = {}

    def create_session(self, config: Any) -> Any:
        if isinstance(config, dict):
            rid = config.get("run_id") or "cu-fake"
        else:
            rid = getattr(config, "run_id", None) or "cu-fake"
        ws = SimpleNamespace(
            budgets={
                "max_steps": 3,
                "max_actions": 3,
                "action_timeout_ms": 2000,
                "reasoning_timeout_ms": 5000,
                "confirmation_wait_timeout_ms": 5000,
                "max_cpu_percent": 50,
                "max_rss_mb": 128,
                "max_processes": 4,
            },
            mode="ISOLATED",
            task_priority="normal",
        )
        sess = SimpleNamespace(
            run_id=rid,
            worker_session=ws,
            supervisor=_FakeSupervisor(),
            isolated_display=":99",
            observe_display=None,
            current_step=0,
            current_actions=0,
            pending_messages=[],
            status="active",
            xvfb_root_id=None,
        )
        self._sessions[rid] = sess
        return sess

    def close_session(self, session_id: str) -> Any:
        self._sessions.pop(session_id, None)
        return SimpleNamespace(run_id=session_id, status="stopped", terminated=True)

    def enforce_session_limits(self, run_id: str) -> None:
        pass

    def await_confirmation(self, run_id: str, intent_id: str, timeout_ms: int) -> ConfirmationOutcome:
        return ConfirmationOutcome(
            outcome=ConfirmationOutcomeType.ACCEPTED.value,
            token="test-token",
            intent_id=intent_id,
            received_at="2026-01-01T00:00:00Z",
        )


def test_computer_use_harness_smoke_constructs_adapter_and_writes_events(tmp_path):
    """Direct adapter construction + start with full mocks; events.jsonl under controlled runs/ + is_available.

    Fully hermetic: injected fake SessionController prevents any Xvfb spawn, real capability
    probing side effects, or host process creation. Uses explicit audit_sink_factory so events
    land exactly where the test asserts (no chdir, no pollution of real runs/ or cwd).
    Exercises the adapter start path + FR-13 event sink contract.
    """
    runs_dir = tmp_path / "runs"
    # Controlled sink factory forces the exact events.jsonl location the test will assert on.
    def _sink_factory(rid: str) -> AuditEventSink:
        return AuditEventSink(run_id=rid, runs_root=runs_dir)

    reasoner = _FakeReasoner()
    perception = _FakePerception()
    safety = _FakeSafety()
    executor = _FakeExecutor()
    supervisor = _FakeSupervisor()
    controller = _FakeSessionController()

    adapter = ComputerUseWorkerAdapter(
        session_controller=controller,
        reasoner=reasoner,
        perception=perception,
        safety=safety,
        executor=executor,
        supervisor=supervisor,
        audit_sink_factory=_sink_factory,
    )

    # Trivial objective (rationale contains "done" so fake loop exits after one safe action)
    h = adapter.start({"run_id": "cu-harness-smoke-001", "objective": "trivial smoke objective"})

    assert h.run_id == "cu-harness-smoke-001"
    assert h.status in ("completed", "errored")

    ev_path = runs_dir / "cu-harness-smoke-001" / "events.jsonl"
    assert ev_path.exists(), "events.jsonl must be written under runs/<id>/ per FR-13 wiring"
    lines = [ln for ln in ev_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 1, "at least the action_executed (or safety) event must be present"
    first = json.loads(lines[0])
    assert first["run_id"] == "cu-harness-smoke-001"
    assert "event_type" in first

    # is_available reports correctly (CapabilityReport shape) — real probe is fine here
    # (no GUI, no spawn, just binary/import checks; shape is part of the contract).
    rep = adapter.is_available()
    assert hasattr(rep, "readiness")
    assert rep.readiness in ("ready", "degraded", "unavailable")
    assert isinstance(rep.atspi, bool)
    assert isinstance(rep.action_exec, bool)


def test_roles_recognizes_computer_use_token():
    """'computer-use' is a first-class selectable worker token via roles (Step 12)."""
    # describe_chain works (used for logging in dispatch)
    desc = roles.describe_chain(["computer-use"], fallback=False)
    assert "computer-use" in desc.lower()

    # build_role_agent returns the shim (used if anyone constructs directly)
    shim = roles.build_role_agent(["computer-use"], prompt="smoke objective", fallback=False)
    assert shim is not None
    assert "shim" in type(shim).__name__.lower()

    # Token constant present
    assert roles.COMPUTER_USE_TOKEN == "computer-use"


def test_cli_forwards_real_gui_and_browser_flags(monkeypatch):
    captured: Dict[str, Any] = {}

    def _fake_dispatch(instruction: str, **kwargs: Any) -> Any:
        captured["instruction"] = instruction
        captured.update(kwargs)
        return SimpleNamespace(
            success=True,
            run_id="r-cli",
            mode=kwargs.get("mode", "adversarial"),
            generator="computer-use",
            critic=None,
            duration_s=0.0,
            quality=None,
            error=None,
            changed_files=[],
            added=[],
            modified=[],
            deleted=[],
            run_dir="/tmp/r-cli",
        )

    monkeypatch.setattr(harness_cli, "dispatch", _fake_dispatch)
    monkeypatch.setattr(harness_cli, "_print_result", lambda _: None)

    rc = harness_cli.main(
        [
            "do",
            "search and click",
            "--generator",
            "computer-use",
            "--computer-use-mode",
            "REAL",
            "--real-gui-policy",
            "children",
            "--ask-mode",
            "off",
            "--browser-engine",
            "duckduckgo",
            "--browser-display",
            ":0",
        ]
    )
    assert rc == 0
    assert captured["computer_use_mode"] == "REAL"
    assert captured["real_gui_policy"] == "children"
    assert captured["ask_mode"] == "off"
    assert captured["browser_engine"] == "duckduckgo"
    assert captured["browser_display"] == ":0"


def test_computer_use_shim_forwards_browser_config(monkeypatch):
    captured: Dict[str, Any] = {}

    class _FakeAuditSink:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

    class _FakeAdapter:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def start(self, req: Dict[str, Any]) -> Any:
            captured.update(req)
            return SimpleNamespace(run_id=req["run_id"], status="completed", events_path=None)

    monkeypatch.setattr(
        computer_use_role,
        "_get_cu_imports",
        lambda: (_FakeAdapter, object, _FakeAuditSink),
    )

    shim = computer_use_role.ComputerUseShim(
        prompt="search objective",
        harness_run_id="cu-shim-browser",
        computer_use_config={
            "mode": "REAL",
            "real_gui_policy": "children",
            "ask_mode": "off",
            "browser_engine": "bing",
            "browser_display": ":0",
        },
    )
    asyncio.run(shim.run_async())

    assert captured["mode"] == "REAL"
    assert captured["real_gui_policy"] == "children"
    assert captured["ask_mode"] == "off"
    assert captured["browser_engine"] == "bing"
    assert captured["browser_display"] == ":0"

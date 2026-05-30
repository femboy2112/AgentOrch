"""AuditEventSink + WorkerEvent streaming tests (Step 6).

Covers the exact contract requested:
- Safe append-only JSONL (one line per event) to runs/<run_id>/events.jsonl
- Creates parent directories on demand
- Atomic append implementation that *never truncates* on error (full line is staged to a temp sibling + fsync'd before any mutation of the target)
- Supports every WorkerEventType value from the authoritative spec
- emit() also invokes registered callbacks (harness EVENT_BUS integration point) with plain dicts after durable write
- Every persisted shape matches the WorkerEvent dataclass exactly (ts, run_id, event_type, payload) and roundtrips via from_json
- Crash-safety test: when a write error is injected "mid-event", the on-disk file contains only complete prior lines (no partial/truncated JSON)

These events are the *only* audit trail (FR-13). Later components (perception, reasoner, action executor, session controller, adapter) are required to emit through this sink for every lifecycle transition.

All tests are hermetic: they use a temp runs_root and never touch the real runs/ tree or real :0.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agy_orchestrator.computer_use import AuditEventSink
from agy_orchestrator.computer_use.models import WorkerEvent, WorkerEventType


def _mk_event(run_id: str, etype: str, payload: Dict[str, Any] | None = None) -> WorkerEvent:
    return WorkerEvent(
        ts="2026-05-29T12:00:00.000000Z",
        run_id=run_id,
        event_type=etype,
        payload=payload or {"note": "test"},
    )


def test_sink_creates_dirs_and_appends_lines(tmp_path: Path) -> None:
    """Basic append contract + directory creation."""
    run_id = "20260529-120000-001"
    sink = AuditEventSink(run_id, runs_root=tmp_path)

    # Directory should have been created
    assert sink.events_path.parent.exists()
    assert sink.events_path.exists()

    ev1 = _mk_event(run_id, "capability.probe", {"atspi": False, "ocr": True})
    sink.emit(ev1)

    ev2 = _mk_event(run_id, "perception.snapshot", {"scope": "isolated", "windows": 3})
    sink.emit(ev2)

    lines = sink.events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    d1 = json.loads(lines[0])
    d2 = json.loads(lines[1])
    assert d1["event_type"] == "capability.probe"
    assert d2["event_type"] == "perception.snapshot"
    assert d1["run_id"] == run_id
    assert d2["run_id"] == run_id


@pytest.mark.parametrize("etype", [e.value for e in WorkerEventType])
def test_every_worker_event_type_is_supported_and_roundtrips(tmp_path: Path, etype: str) -> None:
    """Every event_type literal from the spec must be accepted and persist in canonical shape."""
    run_id = "etype-roundtrip"
    sink = AuditEventSink(run_id, runs_root=tmp_path)

    payload = {
        "intent_id": "i1",
        "risk_level": "high",
        "display_scope": "isolated",
        "rationale": "test step 6",
    }
    ev = _mk_event(run_id, etype, payload)
    sink.emit(ev)

    # Read back and reconstruct
    stored = sink.read_all()
    assert len(stored) == 1
    assert stored[0]["event_type"] == etype

    # Exact roundtrip through the dataclass validator
    ev2 = WorkerEvent.from_json(json.dumps(stored[0]))
    assert ev2.event_type == etype
    assert ev2.run_id == run_id
    assert ev2.payload.get("rationale") == "test step 6"


def test_callbacks_are_invoked_after_successful_write(tmp_path: Path) -> None:
    """emit() fans out to registered callbacks with the dict form (harness bus contract)."""
    run_id = "cb-test"
    sink = AuditEventSink(run_id, runs_root=tmp_path)

    received: List[Dict[str, Any]] = []
    sink.add_callback(lambda d: received.append(d))

    ev = _mk_event(run_id, "action.dry_run", {"requires_confirmation": True})
    sink.emit(ev)

    assert len(received) == 1
    assert received[0]["event_type"] == "action.dry_run"
    assert received[0]["payload"]["requires_confirmation"] is True

    # Second callback also works
    received2: List[Dict[str, Any]] = []
    sink.add_callback(lambda d: received2.append(d["event_type"]))

    ev2 = _mk_event(run_id, "confirmation.received", {"token": "tok-xyz"})
    sink.emit(ev2)

    assert len(received) == 2  # first list still appended to
    assert received2 == ["confirmation.received"]


def test_append_only_behavior(tmp_path: Path) -> None:
    """Multiple emits never overwrite; order and count preserved; file always ends with \n."""
    run_id = "append-only"
    sink = AuditEventSink(run_id, runs_root=tmp_path)

    for i, et in enumerate(["reasoner.intent", "action.executed", "session.terminated"]):
        sink.emit(_mk_event(run_id, et, {"step": i}))

    content = sink.events_path.read_text(encoding="utf-8")
    assert content.endswith("\n")
    lines = [ln for ln in content.splitlines() if ln.strip()]
    assert len(lines) == 3
    assert json.loads(lines[0])["payload"]["step"] == 0
    assert json.loads(lines[2])["event_type"] == "session.terminated"


def test_crash_safety_simulated_write_error_leaves_no_partial_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Release-blocking crash-safety guarantee.

    When an I/O error occurs "mid-event" (during the target append), the on-disk
    events.jsonl must contain *only* complete, previously committed lines.
    No truncated JSON, no half-written line, and the line count must be unchanged.
    """
    run_id = "crash-safety"
    sink = AuditEventSink(run_id, runs_root=tmp_path)

    # Seed two good events
    sink.emit(_mk_event(run_id, "capability.probe", {"phase": "start"}))
    sink.emit(_mk_event(run_id, "perception.snapshot", {"phase": "snap"}))
    before_text = sink.events_path.read_text(encoding="utf-8")
    before_lines = [ln for ln in before_text.splitlines() if ln.strip()]
    assert len(before_lines) == 2
    # last line must be valid JSON
    json.loads(before_lines[-1])

    # Inject failure on the *target* append path only (the "ab" write)
    # Our _safe_append_line stages to temp first, so the temp always has the full line.
    boom_count = {"n": 0}

    real_open = builtins.open

    def crashing_open(path: Any, mode: str = "r", *args: Any, **kwargs: Any):
        p = str(path)
        if "events.jsonl" in p and "b" in mode:  # the critical "ab" append of the staged line
            boom_count["n"] += 1
            if boom_count["n"] == 1:
                # Simulate the write error exactly as a mid-event disk failure would appear
                raise OSError(28, "No space left on device (simulated mid-event)")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", crashing_open)

    bad_ev = _mk_event(run_id, "reasoner.timeout", {"timeout_ms": 45000})
    with pytest.raises(OSError):
        sink.emit(bad_ev)

    after_text = sink.events_path.read_text(encoding="utf-8")
    after_lines = [ln for ln in after_text.splitlines() if ln.strip()]

    # Invariant (the heart of the release-blocking guarantee): no new lines were added,
    # the file contains only the exact prior complete events.
    assert len(after_lines) == len(before_lines)
    assert after_text == before_text or after_text.endswith("\n")

    # Every persisted line is still valid JSON (no truncation or half-written objects)
    for ln in after_lines:
        json.loads(ln)

    # The staged temp for the failed write must have been cleaned up (finally block)
    for f in sink.events_path.parent.iterdir():
        assert not f.name.endswith(".jsonl.tmp"), f"leftover temp after crash: {f}"


def test_worker_event_shapes_match_spec_exactly(tmp_path: Path) -> None:
    """Persisted records contain precisely the four fields the spec defines for WorkerEvent."""
    run_id = "shape-fidelity"
    sink = AuditEventSink(run_id, runs_root=tmp_path)

    ev = WorkerEvent(
        ts="2026-05-29T12:34:56.789012Z",
        run_id=run_id,
        event_type="safety.violation",
        payload={
            "code": "display_scope_invalid",
            "message": "FR-04 hard reject",
            "field": "display_scope",
        },
    )
    sink.emit(ev)

    stored = sink.read_all()[0]
    assert set(stored.keys()) == {"ts", "run_id", "event_type", "payload"}
    assert stored["event_type"] in {e.value for e in WorkerEventType}

    # Reconstruct validates the enum
    round = WorkerEvent.from_json(json.dumps(stored))
    assert round.event_type == "safety.violation"
    assert round.payload["code"] == "display_scope_invalid"


def test_explicit_events_path_overrides_runs_root(tmp_path: Path) -> None:
    """When events_path= is supplied it is used verbatim (used by adapter in later steps)."""
    custom = tmp_path / "custom" / "my-run" / "events.jsonl"
    sink = AuditEventSink("explicit-run", events_path=custom)
    assert sink.events_path == custom.resolve()
    assert custom.parent.exists()

    sink.emit(_mk_event("explicit-run", "resource.limit", {}))
    assert custom.exists()
    assert len(sink.read_all()) == 1


@pytest.mark.realgui
def test_realgui_real_run_one_foreign_prompt_produces_new_event_kinds(tmp_path: Path) -> None:
    """5-line: REAL foreign prompt emits the new WorkerEventTypes (FR-38)."""
    from agy_orchestrator.computer_use.grants import GrantCache
    from agy_orchestrator.computer_use.gui_prompt import FakePrompter
    from agy_orchestrator.computer_use.ownership import FakeOwnershipResolver
    from agy_orchestrator.computer_use.process_supervisor import ProcessSupervisor
    from agy_orchestrator.computer_use.safety import SafetyKernel
    sink = AuditEventSink("rg-audit", runs_root=tmp_path)
    k = SafetyKernel(ownership_resolver=FakeOwnershipResolver(synthetic_baseline_pids={42}, synthetic_owned=set()), gui_prompter=FakePrompter().queue(grant_scope="PROCESS_RUN", granted=True, operator_text="n"), grant_cache=GrantCache(run_id="rg"), supervisor=ProcessSupervisor(), audit_sink=sink)
    k._real_act_gate({"action": {"type": "click", "target": {"kind": "element", "app_pid": 42}}}, {"run_id": "rg", "mode": "REAL", "real_gui_policy": "full", "ask_mode": "on"})
    kinds = {e["event_type"] for e in sink.read_all()}
    assert {"permission.prompt_shown", "permission.granted", "operator_note.received"} <= kinds

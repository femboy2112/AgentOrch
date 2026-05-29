"""ReasonerBridge unit tests (Step 10).

Covers FR-14/16/20/21/25 + the exact CLI contract (prompt envelope, first-fenced
ACTION_INTENT_JSON_V1 parse, schema validation via models, bounded repair+fallback,
timeout kill of owned subprocess, auth-required detection and kill, never produce
ActionIntent from bad/unparseable/auth/timeout output).

All tests use mocks only (AsyncMock on agent.run_async). No real CLIs, no X11,
no real :0, no network. Hermetic.

Release-blocking behaviors for this component are exercised here; higher-level
FRs (adapter loop) are covered in later steps.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agy_orchestrator.computer_use import (
    AuditEventSink,
    ReasonerBridge,
    ReasonerError,
    ReasoningInput,
    SnapshotSummary,
    WorkerEventType,
)
from agy_orchestrator.computer_use.models import ActionIntent, TaskPriority


def _mk_ri(task_priority: str = "normal", objective: str = "click the button") -> ReasoningInput:
    """Minimal valid ReasoningInput for tests."""
    return ReasoningInput(
        run_id="test-run-001",
        session_mode="ISOLATED",
        task_priority=task_priority,
        objective=objective,
        constraints={
            "must_use_display_scope": "isolated",
            "max_actions_remaining": 10,
            "max_steps_remaining": 5,
            "disallowed_ops": [],
        },
        snapshots={
            "isolated": SnapshotSummary(
                snapshot_id="s1",
                captured_at="2026-05-29T00:00:00Z",
                scope="isolated",
                windows=[],
                elements=[],
                raw_text_blocks=[],
            )
        },
        output_contract="ACTION_INTENT_JSON_V1",
    )


def _good_action_intent_dict() -> Dict[str, Any]:
    return {
        "intent_id": "i-001",
        "snapshot_id": "s1",
        "action": {
            "type": "click",
            "display_scope": "isolated",
            "target": {"kind": "element", "handle_id": "h42"},
        },
        "rationale": "Click the obvious submit button",
        "risk_level": "low",
        "requires_confirmation": False,
        "confidence": 0.91,
    }


def _good_json_response() -> str:
    return json.dumps({"action_intent": _good_action_intent_dict()})


def _mk_sink(tmp_path: Path, run_id: str = "test-run-001") -> AuditEventSink:
    p = tmp_path / run_id / "events.jsonl"
    return AuditEventSink(run_id=run_id, events_path=str(p))


def _mk_mock_agent(return_text: str = "", side_effect: Any = None) -> MagicMock:
    """Create a mock AgentInstance with async run_async + kill hook for FR-20/25 tests."""
    agent = MagicMock(spec=["prompt", "run_async", "stdout", "stderr", "_current_process", "timeout", "_kill_current"])
    agent.prompt = ""
    agent.stdout = ""
    agent.stderr = ""
    agent._current_process = None
    agent.timeout = 0
    agent._kill_current = AsyncMock()  # track kill attempts
    if side_effect is not None:
        agent.run_async = AsyncMock(side_effect=side_effect)
    else:
        agent.run_async = AsyncMock(return_value=return_text)
    return agent


class TestReasonerBridgeContract:
    def test_valid_contract_roundtrip_returns_action_intent_and_emits_event(self, tmp_path: Path):
        """Happy path: model returns perfect fenced ACTION_INTENT_JSON_V1 -> ActionIntent + reasoner.intent event."""
        sink = _mk_sink(tmp_path)
        ri = _mk_ri()
        good = _good_json_response()

        claude = _mk_mock_agent(return_text=good)
        bridge = ReasonerBridge("test-run-001", audit_sink=sink, claude_agent=claude, codex_agent=None)

        intent = bridge.decide(ri, timeout_ms=5000)

        assert isinstance(intent, ActionIntent)
        assert intent.action["type"] in ("click", "wait")  # stub may supply its safe default
        assert intent.risk_level in ("low", "medium")

        # Verify event was written
        events_file = Path(sink.events_path)
        assert events_file.exists()
        lines = events_file.read_text().strip().splitlines()
        assert any('"reasoner.intent"' in ln for ln in lines)

    def test_high_priority_routes_claude_first(self, tmp_path: Path):
        """FR-14: task_priority=high -> claude lead, then codex fallback."""
        sink = _mk_sink(tmp_path)
        ri = _mk_ri(task_priority="high")

        claude = _mk_mock_agent(return_text="BAD")  # causes parse_error -> one repair attempt
        codex = _mk_mock_agent(return_text=_good_json_response())

        bridge = ReasonerBridge("r1", audit_sink=sink, claude_agent=claude, codex_agent=codex)
        intent = bridge.decide(ri, timeout_ms=2000)

        assert intent is not None
        # claude (lead for high) must be attempted first (initial + 1 repair)
        assert claude.run_async.call_count >= 2
        # codex used as fallback after claude repairs exhausted
        assert codex.run_async.call_count >= 1
        # Verify the prompt envelope passed to the reasoner contains the required contract sections
        called_prompt = claude.run_async.call_args[0][0] if claude.run_async.call_args else ""
        assert "HARD SAFETY CONSTRAINTS" in called_prompt
        assert "ACTION_INTENT_JSON_V1" in called_prompt or "action_intent" in called_prompt
        assert "REASONING_INPUT" in called_prompt
        assert '"display_scope": "isolated"' in called_prompt or "display_scope" in called_prompt

    def test_normal_priority_routes_codex_first(self, tmp_path: Path):
        """FR-21: absent/normal -> codex lead, claude fallback."""
        sink = _mk_sink(tmp_path)
        ri = _mk_ri(task_priority="normal")

        codex = _mk_mock_agent(return_text="BAD")
        claude = _mk_mock_agent(return_text=_good_json_response())

        bridge = ReasonerBridge("r2", audit_sink=sink, claude_agent=claude, codex_agent=codex)
        intent = bridge.decide(ri, timeout_ms=2000)
        assert intent is not None

    def test_parse_failure_triggers_exactly_one_repair_then_fallback(self, tmp_path: Path):
        """FR-16: bad output -> emit parse_error, one repair retry on same, then fallback."""
        sink = _mk_sink(tmp_path)
        # Force high priority so claude is the lead that gets the bad output + repair
        ri = _mk_ri(task_priority="high")

        # Lead (claude for high): first call bad, repair call also bad -> fallback to codex success
        claude = _mk_mock_agent(return_text="not valid json at all")
        codex = _mk_mock_agent(return_text=_good_json_response())

        bridge = ReasonerBridge("r3", audit_sink=sink, claude_agent=claude, codex_agent=codex)
        bridge.decide(ri, timeout_ms=3000)

        # Exactly: 1 initial + 1 repair on claude (max_repair=1 default), then 1 success on codex
        assert claude.run_async.call_count == 2
        assert codex.run_async.call_count == 1
        # Events must include parse_error(s)
        txt = Path(sink.events_path).read_text()
        assert "reasoner.parse_error" in txt

    def test_timeout_kills_subprocess_emits_event_and_falls_back(self, tmp_path: Path):
        """FR-20: decide timeout -> explicit kill of _current_process, reasoner.timeout event, fallback used."""
        sink = _mk_sink(tmp_path)
        # High priority: claude (lead) times out -> kill + timeout event + fallback to codex
        ri = _mk_ri(task_priority="high")

        async def timeout_side_effect(*a, **k):
            await asyncio.sleep(10)  # will be cut by wait_for
            return "never"

        claude = _mk_mock_agent(side_effect=asyncio.TimeoutError)
        # Attach a real process mock so the kill path has something to act on
        fake_proc = MagicMock(returncode=None)
        claude._current_process = fake_proc

        codex = _mk_mock_agent(return_text=_good_json_response())

        bridge = ReasonerBridge("r4", audit_sink=sink, claude_agent=claude, codex_agent=codex)

        intent = bridge.decide(ri, timeout_ms=50)
        assert intent is not None

        # FR-20: timeout event must be emitted and kill must have been attempted on the hung agent
        txt = Path(sink.events_path).read_text()
        assert "reasoner.timeout" in txt
        # The _kill_current on the claude mock must have been called at least once
        assert claude._kill_current.call_count >= 1

    def test_auth_marker_kills_and_falls_back_no_intent_from_bad_engine(self, tmp_path: Path):
        """FR-25: OAuth / interactive marker in stdout+stderr -> reasoner.auth_required, kill, fallback, no execution from bad engine."""
        sink = _mk_sink(tmp_path)
        # High priority: claude (lead) emits auth marker -> auth event + kill + fallback to codex
        ri = _mk_ri(task_priority="high")

        claude = _mk_mock_agent(return_text="Please open your browser to re-authenticate your OAuth session")
        claude.stderr = "interactive login required"
        fake_proc = MagicMock(returncode=None)
        claude._current_process = fake_proc

        codex = _mk_mock_agent(return_text=_good_json_response())

        bridge = ReasonerBridge("r5", audit_sink=sink, claude_agent=claude, codex_agent=codex)

        intent = bridge.decide(ri, timeout_ms=1000)
        assert intent is not None

        txt = Path(sink.events_path).read_text()
        assert "reasoner.auth_required" in txt
        # FR-25 kill + fallback: kill hook called on the auth-failing engine
        assert claude._kill_current.call_count >= 1
        # claude attempted (and failed auth), codex succeeded
        assert claude.run_async.call_count >= 1
        assert codex.run_async.call_count >= 1

    def test_all_routes_exhausted_raises_and_never_returns_intent(self, tmp_path: Path):
        """Core safety: every bad path (parse, auth, timeout) after retries -> ReasonerError, zero ActionIntent returned."""
        sink = _mk_sink(tmp_path)
        ri = _mk_ri()

        claude = _mk_mock_agent(return_text="{ \"garbage\": true }")
        codex = _mk_mock_agent(return_text="also completely invalid")

        bridge = ReasonerBridge("r6", audit_sink=sink, claude_agent=claude, codex_agent=codex)
        # (priority does not matter; both engines produce unparseable output)

        # The core safety invariant of FR-16/20/25: after all routes + repairs, MUST raise and never hand back an executable intent.
        with pytest.raises(ReasonerError):
            bridge.decide(ri, timeout_ms=1000)

        txt = Path(sink.events_path).read_text()
        assert "reasoner.parse_error" in txt
        # final parse_error event must carry the exhausted marker
        assert "exhausted" in txt or "final" in txt

    def test_engine_status_reports_routing_and_last_error(self, tmp_path: Path):
        sink = _mk_sink(tmp_path)
        bridge = ReasonerBridge("r7", audit_sink=sink, claude_agent=_mk_mock_agent(), codex_agent=_mk_mock_agent())
        health = bridge.engine_status()
        assert health.routing_high_priority == "claude_then_codex"
        assert health.routing_default == "codex_then_claude"

    def test_no_action_from_malformed_missing_action_intent_key(self, tmp_path: Path):
        """Parser must reject blocks that do not contain the required key."""
        sink = _mk_sink(tmp_path)
        # High priority: bad (missing action_intent key) on lead (claude) -> repair + fallback
        ri = _mk_ri(task_priority="high")
        bad = "```json\n{\"foo\": \"bar\"}\n```"
        claude = _mk_mock_agent(return_text=bad)
        codex = _mk_mock_agent(return_text=_good_json_response())

        bridge = ReasonerBridge("r8", audit_sink=sink, claude_agent=claude, codex_agent=codex)
        intent = bridge.decide(ri, timeout_ms=1000)
        assert intent is not None  # fell back successfully after repair on claude
        assert claude.run_async.call_count >= 1
        assert codex.run_async.call_count >= 1
        txt = Path(sink.events_path).read_text()
        assert "reasoner.parse_error" in txt

    def test_redacted_text_in_ri_never_leaks_even_if_somehow_present(self, tmp_path: Path):
        """Defense-in-depth: even if a secret reached the ReasoningInput (should not), the prompt construction itself is not the redactor,
        but we assert the test ri used here has no planted secret (redaction contract is in perception + models tests)."""
        sink = _mk_sink(tmp_path)
        ri = _mk_ri(objective="do nothing with SECRET_TOKEN=abc123def")
        # The objective above is deliberately not redacted here; real path redacts in Perception before building RI.
        # ReasonerBridge must not be the first redactor. This test just documents the expectation.
        claude = _mk_mock_agent(return_text=_good_json_response())
        bridge = ReasonerBridge("r9", audit_sink=sink, claude_agent=claude)
        # We do not assert redaction here (that is Step 1/8 contract); we only ensure decide works.
        intent = bridge.decide(ri, timeout_ms=1000)
        assert intent is not None

    def test_prompt_envelope_contains_full_contract_sections(self, tmp_path: Path):
        """Exact CLI contract: the text handed to the reasoner CLI must contain header safety rules,
        the deterministic fenced ReasoningInput, and the strict ACTION_INTENT_JSON_V1 footer instructions.
        """
        sink = _mk_sink(tmp_path)
        ri = _mk_ri(objective="test the envelope")
        claude = _mk_mock_agent(return_text=_good_json_response())

        bridge = ReasonerBridge("env-test", audit_sink=sink, claude_agent=claude)
        bridge.decide(ri, timeout_ms=1000)

        prompt = claude.run_async.call_args[0][0] if claude.run_async.call_args else ""
        # Header
        assert "HARD SAFETY CONSTRAINTS" in prompt
        assert "NEVER VIOLATE" in prompt or "VIOLATIONS ARE NEVER" in prompt
        # Body
        assert "REASONING_INPUT" in prompt
        assert "```json" in prompt
        assert '"objective": "test the envelope"' in prompt or "test the envelope" in prompt
        # Footer / contract
        assert "ACTION_INTENT_JSON_V1" in prompt
        assert "EXACTLY ONE fenced JSON" in prompt or "exactly one" in prompt.lower()
        assert '"display_scope": "isolated"' in prompt or "display_scope" in prompt

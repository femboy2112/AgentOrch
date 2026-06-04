"""Issue #74: a reconcile reply that embeds a RAW control character (an unescaped
newline/tab) inside a JSON string value must be REPAIRED and parsed, not bailed as
hollow. Strict ``json.loads`` rejects such output with "Invalid control character
at ...", which on a --mission-critical run failed the fail-disposition gate on a
COSMETIC formatting glitch — so the anti-hollow-win (dead-wiring) check never ran,
exactly when an independent cert found a real binding dead-wiring defect.

These contracts assert the NEW behavior: the verdict is recovered (parse_error is
None) and the dead-wiring finding is examined. Genuinely malformed structure must
STILL read as parse_error (repair never blesses corruption).
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.workflows.reconcile import ReconciliationReview


class _SeqAgent(AgentInstance):
    """Returns successive replies from a list; records how many calls were made."""

    def __init__(self, replies):
        super().__init__(prompt="")
        self._replies = list(replies)
        self.calls = 0

    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input=None):
        i = min(self.calls, len(self._replies) - 1)
        self.calls += 1
        return self._replies[i]


def _dead_wiring_reply_with_raw_newline() -> str:
    """A verdict object carrying a real dead-wiring finding, but with a LITERAL
    newline (0x0A) inside the ``evidence`` string value — the exact
    "Invalid control character at" failure from the incident. Hand-built (not
    json.dumps, which would escape the newline) so the control char is raw."""
    return (
        '{\n'
        '  "findings": [{\n'
        '    "name": "per_level_world_richness",\n'
        '    "classification": "exists_not_load_bearing",\n'
        '    "sub_kind": "stub_constant",\n'
        '    "location": "world.py:30",\n'
        '    "evidence": "set in config but never consumed\n'   # <- RAW newline in string
        'by the live agent path",\n'
        '    "witness": {"signal": 0.0}\n'
        '  }]\n'
        '}'
    )


def _dead_wiring_reply_with_raw_tab() -> str:
    """Same defect, but the stray control char is a literal TAB (0x09)."""
    return (
        '{"findings": [{'
        '"name": "richness", "classification": "exists_not_load_bearing", '
        '"sub_kind": "stub_constant", "location": "w.py:1", '
        '"evidence": "dead\tcode", "witness": {"signal": 0.0}}]}'
    )


def _sanity_raw_newline_is_strict_unparseable():
    """Guard: confirm the fixture really is strict-unparseable (control char), so
    the contract is honestly about the repair path, not a no-op."""
    with pytest.raises(json.JSONDecodeError):
        json.loads(_dead_wiring_reply_with_raw_newline())


@pytest.mark.not_slow
def test_raw_newline_in_string_is_repaired_and_finding_examined():
    _sanity_raw_newline_is_strict_unparseable()
    station = ReconciliationReview(agent=_SeqAgent(["x"]), goal="g", working_directory=".")
    result = station.parse(_dead_wiring_reply_with_raw_newline())

    assert result.parse_error is None                 # recovered, not bailed hollow
    assert result.substance_status() != "ran:parse_error"
    assert result.is_substantive                      # a real mechanism was examined
    assert result.reconciled is False                 # the dead wiring was CAUGHT
    assert any(f.is_dead for f in result.findings_list)
    assert result.findings_list[0].name == "per_level_world_richness"


@pytest.mark.not_slow
def test_raw_tab_in_string_is_repaired():
    station = ReconciliationReview(agent=_SeqAgent(["x"]), goal="g", working_directory=".")
    result = station.parse(_dead_wiring_reply_with_raw_tab())
    assert result.parse_error is None
    assert result.reconciled is False
    assert any(f.is_dead for f in result.findings_list)


@pytest.mark.not_slow
def test_reprompt_reply_with_control_char_recovers_end_to_end():
    """Mirror the incident: first reply has no JSON (triggers the #68 re-prompt),
    the re-prompt reply carries the control char. The station must recover the
    verdict from the re-prompt reply rather than declaring parse_error/hollow."""
    agent = _SeqAgent(["I traced it and it all looks wired.",
                       _dead_wiring_reply_with_raw_newline()])
    station = ReconciliationReview(agent=agent, goal="g", working_directory=".")
    result = asyncio.run(station.execute())

    assert agent.calls == 2                            # re-prompted exactly once
    assert result.parse_error is None                  # re-prompt reply recovered
    assert result.substance_status() != "ran:parse_error"
    assert result.reconciled is False                  # dead wiring caught
    assert result.is_substantive


@pytest.mark.not_slow
def test_genuinely_malformed_json_still_parse_errors():
    """No-false-positive: a structurally broken reply (not just a stray control
    char) must STILL fail to parse — the repair pass never blesses corruption."""
    station = ReconciliationReview(agent=_SeqAgent(["x"]), goal="g", working_directory=".")
    # Balanced-brace span exists (so a blob is extracted) but the contents are not
    # valid JSON by any leniency.
    result = station.parse('{"findings": [ {name: per_level, this is not json } ] }')
    assert result.parse_error is not None
    assert result.substance_status() == "ran:parse_error"
    assert result.reconciled is False

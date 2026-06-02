"""Issue #68: when the reconcile agent replies with no parseable JSON object, the
station must RE-PROMPT once for JSON-only output before declaring a parse_error —
otherwise the anti-hollow-win check silently no-ops while the run still reports
"verified" from the unit verifier alone.
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


def _verdict_json():
    return json.dumps({
        "findings": [{
            "name": "thing", "sub_kind": "wired", "location": "mod.py:10",
            "witness": {"signal": 1.0},
        }],
    })


@pytest.mark.not_slow
def test_reprompt_recovers_parse_error():
    # First reply is pure prose (no JSON); the JSON-only re-prompt yields a verdict.
    agent = _SeqAgent(["Sure! I traced everything and it all looks wired up.",
                       _verdict_json()])
    station = ReconciliationReview(agent=agent, goal="g", working_directory=".")
    result = asyncio.run(station.execute())

    assert agent.calls == 2                 # re-prompted exactly once
    assert result.parse_error is None       # recovered
    assert result.is_substantive            # property: a real mechanism was examined
    assert result.substance_status() != "ran:parse_error"


@pytest.mark.not_slow
def test_reprompt_failure_keeps_parse_error():
    # Both replies are unparseable prose: the parse_error stands (no silent pass).
    agent = _SeqAgent(["no json here", "still no json, sorry"])
    station = ReconciliationReview(agent=agent, goal="g", working_directory=".")
    result = asyncio.run(station.execute())

    assert agent.calls == 2
    assert result.parse_error is not None
    assert result.substance_status() == "ran:parse_error"
    assert result.reconciled is False       # hollow never blesses the run


@pytest.mark.not_slow
def test_good_first_reply_does_not_reprompt():
    agent = _SeqAgent([_verdict_json()])
    station = ReconciliationReview(agent=agent, goal="g", working_directory=".")
    result = asyncio.run(station.execute())
    assert agent.calls == 1                 # no wasteful re-prompt
    assert result.parse_error is None

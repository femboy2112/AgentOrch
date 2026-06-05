"""Issue #79: the reconcile parse_error path re-prompts the SAME provider (#68) but
never CYCLES to the next provider in the chain. A parse_error is a SUCCESSFUL
produce (prose, not JSON), so the FallbackAgent — which only cycles on a produce
FAILURE — never advances; one provider's JSON-formatting flakiness then false-fails
an otherwise fully-verified run under disposition=fail, without ever asking the
other providers.

Fix: after the #68 same-provider re-prompt still fails, rotate the chain's LEAD
through every remaining provider (via rotate_offset, the #65 lever) until one
yields a parseable verdict.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.workflows.reconcile import ReconciliationReview


class _P1:  # dummy provider classes — only __name__ is read (chain membership)
    pass


class _P2:
    pass


class _P3:
    pass


def _verdict_json():
    return json.dumps({
        "findings": [{
            "name": "thing", "sub_kind": "wired", "location": "mod.py:10",
            "witness": {"signal": 1.0},
        }],
    })


class _ChainAgent(AgentInstance):
    """Mimics a FallbackAgent: exposes ``_chain`` + honours ``rotate_offset``.

    ``json_at_offsets`` is the set of rotate_offset values for which the (rotated)
    lead provider returns parseable JSON; every other call returns prose. Records
    the offsets it was asked at."""

    def __init__(self, chain, json_at_offsets):
        super().__init__(prompt="")
        self._chain = list(chain)
        self._json_at_offsets = set(json_at_offsets)
        self.rotate_offset = 0
        self.offsets_seen = []
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
        off = int(getattr(self, "rotate_offset", 0) or 0)
        self.offsets_seen.append(off)
        self.calls += 1
        return _verdict_json() if off in self._json_at_offsets else "prose, no json here"


@pytest.mark.not_slow
def test_cycles_to_next_provider_and_recovers():
    # Lead (offset 0) always emits prose; provider 2 (offset 1) emits JSON.
    agent = _ChainAgent([_P1, _P2, _P3], json_at_offsets={1})
    station = ReconciliationReview(agent=agent, goal="g", working_directory=".")
    result = asyncio.run(station.execute())

    assert result.parse_error is None, "must recover via the next provider (#79)"
    assert result.substance_status() != "ran:parse_error"
    # Saw the lead twice (#68 re-prompt) then rotated to offset 1.
    assert 1 in agent.offsets_seen, agent.offsets_seen
    # rotate_offset is restored after cycling (no leak into later calls).
    assert agent.rotate_offset == 0


@pytest.mark.not_slow
def test_all_providers_unparseable_stays_hollow():
    # No offset yields JSON: every provider emits prose -> verdict stays hollow.
    agent = _ChainAgent([_P1, _P2, _P3], json_at_offsets=set())
    station = ReconciliationReview(agent=agent, goal="g", working_directory=".")
    result = asyncio.run(station.execute())

    assert result.parse_error is not None
    assert result.substance_status() == "ran:parse_error"
    assert result.reconciled is False
    # Tried the lead (x2 for #68) AND cycled through providers 2 and 3.
    assert set(agent.offsets_seen) >= {0, 1, 2}, agent.offsets_seen
    assert agent.rotate_offset == 0


@pytest.mark.not_slow
def test_single_provider_chain_does_not_cycle():
    # A one-provider chain has no sibling: behaviour is exactly the #68 path.
    agent = _ChainAgent([_P1], json_at_offsets=set())
    station = ReconciliationReview(agent=agent, goal="g", working_directory=".")
    result = asyncio.run(station.execute())

    assert result.parse_error is not None
    assert agent.offsets_seen == [0, 0], agent.offsets_seen  # lead only, twice (#68)


@pytest.mark.not_slow
def test_recovery_at_last_provider():
    # Only the final provider (offset 2) emits JSON -> still recovers.
    agent = _ChainAgent([_P1, _P2, _P3], json_at_offsets={2})
    station = ReconciliationReview(agent=agent, goal="g", working_directory=".")
    result = asyncio.run(station.execute())

    assert result.parse_error is None
    assert 2 in agent.offsets_seen
    assert agent.rotate_offset == 0

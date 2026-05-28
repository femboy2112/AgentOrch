"""When a sub trips the watchdog, FallbackAgent must consult the rules table
and re-route the NEXT attempt to a target chosen by trip reason — not just
march through the normal sequence. This is the rule the design memo cares about:
verbose runs route to a terse worker; stalled runs escalate.
"""
from __future__ import annotations

import asyncio

from agy_orchestrator.core.agent import (
    WATCHDOG_MARKER,
    WATCHDOG_VERBOSE,
    AgentInstance,
)
from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent


class _BaseStub(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]


class _VerboseTrip(_BaseStub):
    """Pretends to have been killed by the verbose watchdog."""
    async def run_async(self, piped_input=None):
        self.stderr = f"{WATCHDOG_MARKER}{WATCHDOG_VERBOSE}] bytes>budget"
        raise RuntimeError(self.stderr)


class _TerseTarget(_BaseStub):
    """The rule's verbose-trip target. Always succeeds."""
    invoked = False

    async def run_async(self, piped_input=None):
        type(self).invoked = True
        self.stdout = "terse-ok"
        self.returncode = 0
        return self.stdout


class _NormalNext(_BaseStub):
    """The next provider in the normal chain — must NOT be reached first when a
    verbose rule pre-empts. Tracks whether it was invoked at all."""
    invoked = False

    async def run_async(self, piped_input=None):
        type(self).invoked = True
        self.stdout = "normal-ok"
        self.returncode = 0
        return self.stdout


def test_verbose_trip_reroutes_to_rule_target_before_normal_next():
    _TerseTarget.invoked = False
    _NormalNext.invoked = False

    Agent = make_fallback_agent(
        chain=[_VerboseTrip, _NormalNext],
        cycles=1,
        watchdog_rules={WATCHDOG_VERBOSE: [_TerseTarget]},
    )

    out = asyncio.run(Agent(prompt="x").run_async())
    assert out == "terse-ok"
    assert _TerseTarget.invoked is True
    # The rule target succeeded, so the normal-next never had to run.
    assert _NormalNext.invoked is False


def test_no_rule_falls_through_to_normal_sequence():
    _NormalNext.invoked = False
    Agent = make_fallback_agent(
        chain=[_VerboseTrip, _NormalNext],
        cycles=1,
        watchdog_rules=None,
    )

    out = asyncio.run(Agent(prompt="x").run_async())
    # No rule for the watchdog trip -> just continue to the next provider normally.
    assert out == "normal-ok"
    assert _NormalNext.invoked is True

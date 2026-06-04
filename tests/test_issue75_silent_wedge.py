"""Issue #75 — a SILENTLY transport-wedged provider must be detected and killed.

Root cause: on a master/--mission-critical run, a codex generator went SILENT
(emitted NOTHING for 30 min) at step 9/19. The per-call stall watchdog's
silence/STALL trip is gated on stall_seconds>0 (agent.py line 833), and when
stall_seconds is 0/unarmed (env unset, calibration miss, or the FallbackAgent
sub never armed by the post-construct hook), a SILENT call (any_output stays
False, no transport errors) trips NOTHING. The call runs to the idle ceiling
(AGY_TIMEOUT) / absolute / the 1800s run backstop, which discards healthy
completed steps. The #72/#73 fix handles a CHATTY wedge but NOT a SILENT one.

Fix: a DEFAULT-ON silence ceiling (AGY_SILENCE_MAX_SECONDS, default 600) so a
call that has emitted NOTHING trips WATCHDOG_STALLED regardless of whether
stall_seconds was armed. The watchdog already loops by default via the transport
bound, and the sub reads the env default in __init__, so this works on the
fallback path with no propagation. CRITICAL: the guard fires ONLY when NO output
was EVER produced (not any_output[0]); an emit-then-quiet call must NOT trip it.
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from agy_orchestrator.core.agent import (
    WATCHDOG_MARKER,
    WATCHDOG_STALLED,
    AgentInstance,
)
from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent


class _Base(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]


# --------------------------------------------------------------------------- #
# (1) UNIT: a SILENT worker (emits nothing) trips WATCHDOG_STALLED on the
#     default-on silence ceiling, even with stall_seconds==0.
# --------------------------------------------------------------------------- #
def test_silent_worker_trips_stalled_via_silence_ceiling():
    """A worker that emits NOTHING at all (sleeps then exits) must trip
    WATCHDOG_STALLED via the silence ceiling within the silence bound, even
    though stall_seconds==0 (unarmed).

    BASELINE (no silence guard): stall_seconds==0 gates the only STALLED trip,
    so nothing fires; the call sleeps the full 2s and exits 0 -> RED.
    """

    class _SilentWedge(_Base):
        def build_command(self, piped_input=None):
            # Emits NOTHING, then exits after 2s. A silence ceiling of 0.3s must
            # kill it within ~a couple of 2s watchdog slices... but the watchdog
            # wakes every 2s; use a sub-slice-friendly assertion below.
            return ["bash", "-c", "sleep 2"]

    agent = _SilentWedge(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 0          # UNARMED — the #75 root condition.
    agent.silence_max_seconds = 0.3  # tiny silence ceiling
    agent.timeout = 60
    agent.max_retries = 1

    with pytest.raises(RuntimeError):
        asyncio.run(agent.run_async())

    assert agent._watchdog_reason == WATCHDOG_STALLED
    assert WATCHDOG_MARKER in agent.stderr
    assert WATCHDOG_STALLED in agent.stderr


# --------------------------------------------------------------------------- #
# (2) NON-REGRESSION: an emit-then-quiet worker must NOT trip the silence
#     guard (any_output True); and silence_max_seconds<=0 disables it.
# --------------------------------------------------------------------------- #
def test_emit_then_quiet_does_not_trip_silence_guard():
    """A worker that EMITS output then goes briefly quiet (reasoning between
    bursts) must NOT trip the silence guard — any_output is True so the guard is
    inert; only stall_seconds / the idle timeout govern that case."""

    class _EmitThenQuiet(_Base):
        def build_command(self, piped_input=None):
            # One early output line, then ~1.5s of silence, then exit 0.
            return ["bash", "-c", "echo 'starting work'; sleep 1.5"]

    agent = _EmitThenQuiet(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 0
    agent.silence_max_seconds = 0.3  # would trip if it ignored any_output
    agent.timeout = 60
    agent.max_retries = 1

    asyncio.run(agent.run_async())
    assert agent.returncode == 0
    assert agent._watchdog_reason is None


def test_silence_guard_disabled_when_zero():
    """silence_max_seconds<=0 disables the guard: a fully silent call that exits
    on its own is NOT killed."""

    class _SilentShort(_Base):
        def build_command(self, piped_input=None):
            return ["bash", "-c", "sleep 1"]

    agent = _SilentShort(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 0
    agent.silence_max_seconds = 0  # disabled
    # transport bound default-on keeps the watchdog looping, but with no silence
    # ceiling and no transport errors nothing trips.
    agent.timeout = 60
    agent.max_retries = 1

    asyncio.run(agent.run_async())
    assert agent.returncode == 0
    assert agent._watchdog_reason is None


def test_silence_default_is_on_and_conservative():
    """The silence ceiling defaults ON (600s) so an unarmed silent wedge is still
    bounded, while being far above any healthy call's first-output latency."""
    a = _Base(prompt="x")
    assert a.silence_max_seconds == 600


# --------------------------------------------------------------------------- #
# (3) FALLBACK PATH: a FallbackAgent over [silent-codex-like, healthy-agy-like]
#     must CYCLE to the healthy provider when the first is silently wedged.
# --------------------------------------------------------------------------- #
def test_silent_wedge_cycles_to_healthy_provider():
    """End-to-end through the REAL agent path: provider 1 emits NOTHING and would
    run effectively forever; provider 2 is healthy. The silence ceiling kills
    provider 1's call (WATCHDOG_STALLED -> RuntimeError), FallbackAgent advances,
    and provider 2's result is returned.

    BASELINE-RED: without the silence guard, provider 1 sleeps ~past the run
    ceiling with stall_seconds==0 and no transport errors, so nothing trips and
    the run hangs / times out -> never reaches provider 2.
    """

    class _SilentProvider(_Base):
        def build_command(self, piped_input=None):
            # Emits NOTHING for ~100s (past the 90s run ceiling below). On
            # baseline this neither trips a watchdog nor exits within budget.
            return ["bash", "-c", "sleep 100"]

    class _HealthyProvider(_Base):
        invoked = False

        async def run_async(self, piped_input=None):
            type(self).invoked = True
            self.stdout = "healthy-provider-2-ok"
            self.returncode = 0
            return self.stdout

    _HealthyProvider.invoked = False

    def _arm(sub, agent_cls):
        # Tighten the silence ceiling on the wedge so the test runs fast. We do
        # NOT set stall_seconds (proving the silence guard fires when unarmed).
        sub.stall_seconds = 0
        sub.silence_max_seconds = 0.3
        sub.timeout = 60
        sub.max_retries = 1

    prev = os.environ.get("AGY_TRANSPORT_RETRIES")
    os.environ["AGY_TRANSPORT_RETRIES"] = "0"
    try:
        Agent = make_fallback_agent(
            chain=[_SilentProvider, _HealthyProvider],
            cycles=1,
            post_construct_hook=_arm,
        )

        async def _run_bounded():
            return await asyncio.wait_for(Agent(prompt="x").run_async(), timeout=90)

        out = asyncio.run(_run_bounded())
    finally:
        if prev is None:
            os.environ.pop("AGY_TRANSPORT_RETRIES", None)
        else:
            os.environ["AGY_TRANSPORT_RETRIES"] = prev

    assert out == "healthy-provider-2-ok"
    assert _HealthyProvider.invoked is True


def test_silent_wedge_cycles_via_env_default_no_hook():
    """Belt-and-suspenders: even with NO post-construct hook arming the sub, the
    sub reads AGY_SILENCE_MAX_SECONDS from the env in __init__, so a silent wedge
    is bounded and the fallback advances. Proves the env-default reaches the
    fallback path without propagation."""

    class _SilentProvider(_Base):
        def build_command(self, piped_input=None):
            return ["bash", "-c", "sleep 100"]

    class _HealthyProvider(_Base):
        invoked = False

        async def run_async(self, piped_input=None):
            type(self).invoked = True
            self.stdout = "ok2"
            self.returncode = 0
            return self.stdout

    _HealthyProvider.invoked = False

    prev_sil = os.environ.get("AGY_SILENCE_MAX_SECONDS")
    prev_to = os.environ.get("AGY_TIMEOUT")
    prev_retries = os.environ.get("AGY_TRANSPORT_RETRIES")
    os.environ["AGY_SILENCE_MAX_SECONDS"] = "0.3"
    os.environ["AGY_TIMEOUT"] = "60"
    os.environ["AGY_TRANSPORT_RETRIES"] = "0"
    try:
        Agent = make_fallback_agent(chain=[_SilentProvider, _HealthyProvider], cycles=1)

        async def _run_bounded():
            return await asyncio.wait_for(Agent(prompt="x").run_async(), timeout=90)

        out = asyncio.run(_run_bounded())
    finally:
        for k, v in (
            ("AGY_SILENCE_MAX_SECONDS", prev_sil),
            ("AGY_TIMEOUT", prev_to),
            ("AGY_TRANSPORT_RETRIES", prev_retries),
        ):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    assert out == "ok2"
    assert _HealthyProvider.invoked is True

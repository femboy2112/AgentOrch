"""Issues #72 + #73 — a chatty-but-wedged provider must be detected and killed.

Root cause: run 20260603-143500-685 step 13/19 wedged ~72 min. codex hit a
sustained websocket transport failure ("Connection reset by peer (os error 104)")
and the subprocess stayed ALIVE retrying internally at ~28% CPU. The existing
transport-stall watchdog (#53) is gated on transport_noise_only[0] / _last_progress,
both of which are RESET by any interleaved ordinary stdout chatter — and a
busy-failing codex INTERLEAVES stdout between the websocket-reset stderr lines.
So neither the TRANSPORT_STALL nor the plain STALL trip ever fires, the call never
terminates, FallbackAgent never advances to provider 2 (#73) and AGY_STALL_SECONDS
never fires (#72).

Fix (Layer 1): a transport-degradation tracker decoupled from _last_progress /
stdout resets. It counts genuine transport-error stderr lines across a "spell" that
survives interleaved progress, and trips WATCHDOG_TRANSPORT_STALL when too many
errors accumulate or the spell runs too long — while a recovery window of clean
output clears the spell (no false positive on a transient blip).
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from agy_orchestrator.core.agent import (
    WATCHDOG_MARKER,
    WATCHDOG_TRANSPORT_STALL,
    AgentInstance,
    is_transport_error,
)
from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent


# The literal codex failure line from the wedged run — interleaved with ordinary
# stdout chatter so the #53 transport_noise_only / _last_progress heuristics get
# reset every loop and never trip (that is the whole bug).
_RESET_LINE = (
    "codex_api::endpoint::responses_websocket: failed to connect to websocket: "
    "IO error: Connection reset by peer (os error 104)"
)


class _Base(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]


def test_reset_line_is_a_transport_error():
    """Sanity: the wedged-run failure line is counted as a genuine transport error
    by is_transport_error (the COUNTING helper), not merely generic noise."""
    assert is_transport_error(_RESET_LINE)


# --------------------------------------------------------------------------- #
# (a) UNIT: chatty-but-wedged provider trips WATCHDOG_TRANSPORT_STALL.
# --------------------------------------------------------------------------- #
def test_interleaved_stdout_and_transport_errors_trips_by_error_count():
    """A provider that interleaves ordinary stdout chatter with repeated websocket-
    reset stderr lines (never completing) must trip WATCHDOG_TRANSPORT_STALL on the
    cumulative transport-error count — even though every stdout line resets the
    #53 transport_noise_only / _last_progress heuristics.

    BASELINE (no tracker): the interleaved stdout keeps _last_progress fresh and
    transport_noise_only False, so neither stall trip fires; the loop runs to its
    hard wall (here proven by the absence of a watchdog reason) -> RED.
    """

    class _ChattyWedge(_Base):
        def build_command(self, piped_input=None):
            # Each iteration: one ordinary stdout progress line, one websocket-reset
            # stderr line. ~40 iterations over ~4s -> >25 transport errors, while
            # stdout never lets the #53 heuristics trip.
            return [
                "bash", "-c",
                "for i in $(seq 1 60); do "
                "echo \"working on chunk $i\"; "
                "echo '" + _RESET_LINE + "' >&2; "
                "sleep 0.1; done",
            ]

    agent = _ChattyWedge(prompt="x")
    agent.max_output_bytes = 0
    # Stall guard set generously high so it is NOT what trips — proving the new
    # transport tracker is independent of stall_seconds (issue #72 root cause).
    agent.stall_seconds = 0
    agent.timeout = 60
    agent.max_retries = 1
    # Tighten the transport bounds so the test is fast: trip after a handful of
    # cumulative transport errors. Seconds bound left at default (300) so the
    # error-count bound is what fires here.
    agent.transport_max_errors = 5
    agent.transport_max_seconds = 300
    agent.transport_recovery_window = 60

    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run_async())
    elapsed = time.monotonic() - t0

    assert agent._watchdog_reason == WATCHDOG_TRANSPORT_STALL
    assert WATCHDOG_MARKER in agent.stderr
    assert WATCHDOG_TRANSPORT_STALL in agent.stderr
    # Trips quickly (a few 2s slices), not the 60s hard timeout.
    assert elapsed < 15


def test_interleaved_transport_errors_trips_by_spell_seconds():
    """When the error COUNT bound is disabled, a degraded spell that persists past
    transport_max_seconds must still trip — proving the wall-clock bound on a
    long-running degraded spell (the ~72-min real wedge)."""

    class _SlowChattyWedge(_Base):
        def build_command(self, piped_input=None):
            return [
                "bash", "-c",
                "for i in $(seq 1 60); do "
                "echo \"chunk $i\"; "
                "echo '" + _RESET_LINE + "' >&2; "
                "sleep 0.2; done",
            ]

    agent = _SlowChattyWedge(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 0
    agent.timeout = 60
    agent.max_retries = 1
    agent.transport_max_errors = 0          # disable the count bound
    agent.transport_max_seconds = 3.0       # spell may persist at most 3s
    agent.transport_recovery_window = 60

    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run_async())
    elapsed = time.monotonic() - t0

    assert agent._watchdog_reason == WATCHDOG_TRANSPORT_STALL
    assert elapsed < 15


# --------------------------------------------------------------------------- #
# (b) RECOVERY / no false positive.
# --------------------------------------------------------------------------- #
def test_recovery_window_clears_spell_no_trip():
    """A short burst of transport errors followed by a recovery window of clean
    output (no further transport errors) must NOT trip — the spell clears."""

    class _BlipThenRecover(_Base):
        def build_command(self, piped_input=None):
            # 3 transport errors quickly, then ~3s of clean stdout. With a 1s
            # recovery window the spell clears well before any bound would fire.
            return [
                "bash", "-c",
                "for i in 1 2 3; do echo '" + _RESET_LINE + "' >&2; sleep 0.1; done; "
                "for i in $(seq 1 15); do echo \"clean work $i\"; sleep 0.2; done",
            ]

    agent = _BlipThenRecover(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 0
    agent.timeout = 60
    agent.max_retries = 1
    agent.transport_max_errors = 5
    agent.transport_max_seconds = 300
    agent.transport_recovery_window = 1.0

    asyncio.run(agent.run_async())
    assert agent.returncode == 0
    assert agent._watchdog_reason is None


def test_healthy_call_never_trips():
    """A completely healthy call (only stdout, zero transport errors) must never
    arm or trip the transport tracker."""

    class _Healthy(_Base):
        def build_command(self, piped_input=None):
            return ["bash", "-c",
                    "for i in $(seq 1 8); do echo \"build step $i\"; sleep 0.1; done"]

    agent = _Healthy(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 0
    agent.timeout = 60
    agent.max_retries = 1
    # Default-on conservative bounds; a healthy call emits ZERO transport errors
    # so it can never trip.
    asyncio.run(agent.run_async())
    assert agent.returncode == 0
    assert agent._watchdog_reason is None


def test_default_knobs_are_on_and_conservative():
    """The new knobs default ON (non-zero) with conservative values so a healthy
    call cannot trip but a sustained wedge is bounded."""
    a = _Base(prompt="x")
    assert a.transport_max_errors == 25
    assert a.transport_max_seconds == 300
    assert a.transport_recovery_window == 60


# --------------------------------------------------------------------------- #
# (c) INTEGRATION (#72 + #73 end-to-end): the wedge terminates, so FallbackAgent
#     advances to a healthy provider 2.
# --------------------------------------------------------------------------- #
def test_wedged_provider_cycles_to_healthy_provider():
    """End-to-end through the REAL agent path: provider 1 is a chatty-but-wedged
    codex-like worker (interleaved stdout + websocket resets, never completes);
    provider 2 is healthy. With Layer 1 the wedged call terminates with
    WATCHDOG_TRANSPORT_STALL, FallbackAgent exhausts its bounded same-provider
    transport retries, advances to provider 2, and returns provider 2's result.

    BASELINE-RED: without the tracker, provider 1's call NEVER terminates (the
    interleaved stdout keeps both #53 heuristics from tripping and the hard
    wall-clock is minutes away), so run_async() blocks inside provider 1 forever
    and never reaches provider 2 — the test times out / hangs. We make the wedge
    faithful (a real subprocess that genuinely never exits on its own within the
    test budget) so the contract is honestly RED on the pre-Layer-1 world.
    """

    class _WedgeProvider(_Base):
        def build_command(self, piped_input=None):
            # Interleaved stdout chatter + websocket-reset stderr, effectively
            # forever (1000 iters * 0.1s = ~100s, past the 90s run ceiling). On
            # baseline this neither trips a watchdog nor exits within the test
            # budget, so the call wedges -> the run hits the 90s timeout -> RED.
            # The tracker kills it after a few errors (seconds).
            return [
                "bash", "-c",
                "for i in $(seq 1 1000); do "
                "echo \"retrying request $i\"; "
                "echo '" + _RESET_LINE + "' >&2; "
                "sleep 0.1; done",
            ]

    class _HealthyProvider(_Base):
        invoked = False

        async def run_async(self, piped_input=None):
            type(self).invoked = True
            self.stdout = "healthy-provider-2-ok"
            self.returncode = 0
            return self.stdout

    _HealthyProvider.invoked = False

    def _arm_transport_bounds(agent_cls, sub):
        # Arm the watchdog + tighten the transport bounds on the wedge provider so
        # the test runs fast. (The harness arms these in production via env.)
        sub.stall_seconds = 0
        sub.timeout = 60
        sub.max_retries = 1
        sub.transport_max_errors = 5
        sub.transport_max_seconds = 300
        sub.transport_recovery_window = 60

    # Keep same-provider transport retries small but non-zero so we still prove the
    # bounded-retry-then-advance path (#73), not just an immediate jump.
    prev = os.environ.get("AGY_TRANSPORT_RETRIES")
    os.environ["AGY_TRANSPORT_RETRIES"] = "1"
    try:
        Agent = make_fallback_agent(
            chain=[_WedgeProvider, _HealthyProvider],
            cycles=1,
            post_construct_hook=_arm_transport_bounds,
        )

        async def _run_bounded():
            # Hard ceiling on the whole run so a baseline regression (never-
            # terminating wedge) surfaces as a clear timeout failure rather than an
            # indefinite hang.
            return await asyncio.wait_for(Agent(prompt="x").run_async(), timeout=90)

        out = asyncio.run(_run_bounded())
    finally:
        if prev is None:
            os.environ.pop("AGY_TRANSPORT_RETRIES", None)
        else:
            os.environ["AGY_TRANSPORT_RETRIES"] = prev

    assert out == "healthy-provider-2-ok"
    assert _HealthyProvider.invoked is True

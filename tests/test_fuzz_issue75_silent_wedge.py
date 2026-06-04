"""FUZZ probes for the issue #75 SILENCE ceiling and adjacent stall-guard edges.

Context: #75 added a DEFAULT-ON silence ceiling (``AGY_SILENCE_MAX_SECONDS``,
default 600) in ``agy_orchestrator/core/agent.py`` so a SILENTLY wedged worker —
one that emits NOTHING (no stdout, no transport errors) with ``stall_seconds``
unarmed — trips ``WATCHDOG_STALLED`` instead of running to the idle ceiling /
absolute cap / run backstop and discarding healthy completed steps. The watchdog
loop (agent.py ~L754) checks, in order: verbose-bytes -> transport spell ->
cumulative transport -> stall_seconds (transport-noise-only) -> silence ceiling
(``silence_bound = stall_seconds if stall_seconds > 0 else silence_max_seconds``,
gated on ``not any_output[0]``).

This file FUZZES that area. Each probe is hermetic and fast (sleep-based silent
subprocesses, fake fallback providers, tiny bounds) and is classified
resilient | bug. NONE of these modify source — they pin documented behavior or
encode a RED contract (``@pytest.mark.xfail(strict=True)``) for a behavior the
code does not yet satisfy (a fix is a later phase).

Probe map (verdicts in the structured report):
  P1  silent-from-start, stall=0          -> trips STALLED in silence bound   [resilient]
  P2  emit-then-quiet, stall=0             -> silence guard inert (any_output) [resilient]
  P3  emit-then-quiet, stall ARMED         -> documented: NOT tripped, idle    [resilient]
  P4  silence=0 AND stall=0 (the OLD gap)  -> no watchdog trip; idle-bounded   [resilient]
  P5  fallback: first silent-wedged        -> cycles to healthy, bounded       [resilient]
  P6  fallback: ALL silent-wedged          -> chain exhausts (raises), bounded [resilient]
  P7  one transport-line then silent,
      transport bound DISABLED             -> silence guard still trips         [resilient]
  P8  one transport-line then silent,
      BOTH bounds armed                    -> transport trips first, bounded    [resilient]
  P9  silence trip races teardown          -> single kill, no wedge/double-kill [resilient]
  P10 DESIRED: silence=0/stall=0 silent
      wedge gets a DISTINCT stalled signal -> RED contract (#75 residual gap)   [bug]
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from agy_orchestrator.core.agent import (
    WATCHDOG_MARKER,
    WATCHDOG_STALLED,
    WATCHDOG_TRANSPORT_STALL,
    AgentInstance,
)
from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent

pytestmark = pytest.mark.not_slow


class _Base(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]


def _disarm_transport(agent) -> None:
    """Turn OFF the default-on transport bounds so a probe isolates the silence
    guard (otherwise an interleaved transport line could trip transport first)."""
    agent.transport_max_errors = 0
    agent.transport_max_seconds = 0


# --------------------------------------------------------------------------- #
# P1  silent from the start (no output ever), stall_seconds==0  ->  the silence
#     ceiling trips WATCHDOG_STALLED within the silence bound (well under the
#     idle ceiling). This is the core #75 guarantee.
# --------------------------------------------------------------------------- #
def test_p1_silent_from_start_trips_stalled_within_silence_bound():
    class _Silent(_Base):
        def build_command(self, piped_input=None):
            return ["bash", "-c", "sleep 30"]  # silent, far longer than the bound

    agent = _Silent(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 0           # UNARMED — the #75 root condition
    agent.silence_max_seconds = 0.3   # tiny ceiling
    _disarm_transport(agent)
    agent.timeout = 60                # idle ceiling far above the silence bound
    agent.absolute_timeout = 0
    agent.max_retries = 1

    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run_async())
    elapsed = time.monotonic() - t0

    assert agent._watchdog_reason == WATCHDOG_STALLED
    assert WATCHDOG_MARKER in agent.stderr
    assert WATCHDOG_STALLED in agent.stderr
    # Bounded by the silence ceiling + at most one 2s watchdog slice — NOT the
    # 60s idle ceiling. (The watchdog wakes every 2s.)
    assert elapsed < 8, f"silence trip took {elapsed:.1f}s, expected ~<4s"


# --------------------------------------------------------------------------- #
# P2  emit-then-quiet with stall_seconds==0: the silence guard must NOT mis-fire
#     because any_output is True. Documented behavior: the call is governed by
#     stall_seconds / the idle timeout, not the silence guard. Here neither is
#     armed tightly, so the call completes normally.
# --------------------------------------------------------------------------- #
def test_p2_emit_then_silent_does_not_misfire_silence_guard():
    class _EmitThenQuiet(_Base):
        def build_command(self, piped_input=None):
            # one early line, then ~1.5s of dead silence, then exit 0
            return ["bash", "-c", "echo working; sleep 1.5"]

    agent = _EmitThenQuiet(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 0
    agent.silence_max_seconds = 0.3   # would kill if it ignored any_output
    _disarm_transport(agent)
    agent.timeout = 60
    agent.max_retries = 1

    asyncio.run(agent.run_async())
    # any_output True -> silence guard inert; the brief silence is allowed.
    assert agent.returncode == 0
    assert agent._watchdog_reason is None


# --------------------------------------------------------------------------- #
# P3  emit-then-quiet with stall_seconds ARMED. Per the agent.py docstring, an
#     emit-then-quiet call is DELIBERATELY not stall-tripped (the worker may be
#     reasoning between bursts) — only the hard idle/wall timeout catches a
#     genuine post-emit hang. Assert that DOCUMENTED behavior: a brief post-emit
#     silence SHORTER than the idle window completes without a watchdog trip,
#     even though stall_seconds is tiny.
# --------------------------------------------------------------------------- #
def test_p3_emit_then_quiet_with_stall_armed_is_not_tripped():
    class _EmitThenQuiet(_Base):
        def build_command(self, piped_input=None):
            return ["bash", "-c", "echo working; sleep 1.2"]

    agent = _EmitThenQuiet(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 0.2   # ARMED and tiny, but emit-then-quiet is exempt
    agent.silence_max_seconds = 0.2
    _disarm_transport(agent)
    agent.timeout = 60          # idle window comfortably above the 1.2s gap
    agent.max_retries = 1

    asyncio.run(agent.run_async())
    # The silence guard's any_output gate AND the stall branch's emit-then-quiet
    # exemption both leave this untouched: returncode 0, no watchdog reason.
    assert agent.returncode == 0
    assert agent._watchdog_reason is None


# --------------------------------------------------------------------------- #
# P4  silence_max_seconds disabled (0) AND stall_seconds==0: the OLD gap. No
#     silence/stall WATCHDOG branch arms, so the silent wedge is bounded by the
#     idle ceiling (AGY_TIMEOUT). Assert it is bounded (not an infinite hang) and
#     that the idle-timeout handler now classifies a ZERO-OUTPUT wedge distinctly
#     as WATCHDOG_STALLED (the #75 residual-gap fix — see P10) so fallback/telemetry
#     can route it on the stall rule rather than fold it into a generic idle timeout.
# --------------------------------------------------------------------------- #
def test_p4_old_gap_silence0_stall0_is_idle_bounded_not_hung():
    class _Silent(_Base):
        def build_command(self, piped_input=None):
            return ["bash", "-c", "sleep 30"]  # would hang if nothing bounded it

    agent = _Silent(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 0
    agent.silence_max_seconds = 0     # DISABLED -> the old gap
    _disarm_transport(agent)
    agent.timeout = 1.0               # idle ceiling is the only bound left
    agent.absolute_timeout = 0
    agent.max_retries = 1

    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run_async())
    elapsed = time.monotonic() - t0

    # Idle-bounded, NOT infinite — and a zero-output wedge is now tagged STALLED.
    assert elapsed < 6, f"old gap not idle-bounded; took {elapsed:.1f}s"
    assert agent._watchdog_reason == WATCHDOG_STALLED
    assert "timed out" in (agent.stderr or "")


# --------------------------------------------------------------------------- #
# P5  FallbackAgent: first provider silently wedged -> the silence ceiling kills
#     it and the chain cycles to a healthy provider, in bounded time.
# --------------------------------------------------------------------------- #
def test_p5_fallback_first_silent_wedge_cycles_to_healthy():
    class _SilentProvider(_Base):
        def build_command(self, piped_input=None):
            return ["bash", "-c", "sleep 60"]

    class _HealthyProvider(_Base):
        invoked = False

        async def run_async(self, piped_input=None):
            type(self).invoked = True
            self.stdout = "healthy-ok"
            self.returncode = 0
            return self.stdout

    _HealthyProvider.invoked = False

    def _arm(sub, agent_cls):
        sub.stall_seconds = 0
        sub.silence_max_seconds = 0.3
        sub.transport_max_errors = 0
        sub.transport_max_seconds = 0
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
            return await asyncio.wait_for(Agent(prompt="x").run_async(), timeout=30)

        t0 = time.monotonic()
        out = asyncio.run(_run_bounded())
        elapsed = time.monotonic() - t0
    finally:
        if prev is None:
            os.environ.pop("AGY_TRANSPORT_RETRIES", None)
        else:
            os.environ["AGY_TRANSPORT_RETRIES"] = prev

    assert out == "healthy-ok"
    assert _HealthyProvider.invoked is True
    assert elapsed < 20, f"cycle to healthy took {elapsed:.1f}s"


# --------------------------------------------------------------------------- #
# P6  FallbackAgent: ALL providers silently wedged -> the chain EXHAUSTS (raises
#     RuntimeError), in bounded wall-clock time, instead of hanging forever. The
#     silence ceiling makes each wedged sub fail fast so the chain drains.
# --------------------------------------------------------------------------- #
def test_p6_fallback_all_silent_wedged_exhausts_bounded():
    class _SilentProvider(_Base):
        def build_command(self, piped_input=None):
            return ["bash", "-c", "sleep 60"]

    class _SilentProviderB(_SilentProvider):
        pass

    def _arm(sub, agent_cls):
        sub.stall_seconds = 0
        sub.silence_max_seconds = 0.3
        sub.transport_max_errors = 0
        sub.transport_max_seconds = 0
        sub.timeout = 60
        sub.max_retries = 1

    prev = os.environ.get("AGY_TRANSPORT_RETRIES")
    os.environ["AGY_TRANSPORT_RETRIES"] = "0"
    try:
        Agent = make_fallback_agent(
            chain=[_SilentProvider, _SilentProviderB],
            cycles=2,  # 2 providers x 2 cycles = 4 silent attempts, each ~<2s
            post_construct_hook=_arm,
        )

        async def _run_bounded():
            # Outer wait_for is a SAFETY net: if the chain hangs (regression) this
            # raises TimeoutError, not RuntimeError, distinguishing hang from the
            # correct bounded-exhaustion path below.
            return await asyncio.wait_for(Agent(prompt="x").run_async(), timeout=40)

        t0 = time.monotonic()
        with pytest.raises(RuntimeError) as ei:
            asyncio.run(_run_bounded())
        elapsed = time.monotonic() - t0
    finally:
        if prev is None:
            os.environ.pop("AGY_TRANSPORT_RETRIES", None)
        else:
            os.environ["AGY_TRANSPORT_RETRIES"] = prev

    # The exhaustion message (not an asyncio timeout) — proves bounded drain, not
    # a hang the safety net caught.
    assert "exhausted" in str(ei.value).lower()
    assert elapsed < 30, f"all-wedged exhaustion took {elapsed:.1f}s"


# --------------------------------------------------------------------------- #
# P7  Interaction with the #72/#73 transport bound: a provider emits ONE
#     transport-error line then goes silent, with the transport bound DISABLED.
#     A transport-noise stderr line is NOT counted as any_output (it sets only
#     transport_noise_only, never _mark_progress), so the silence guard remains
#     eligible and trips WATCHDOG_STALLED — bounded, not hung.
# --------------------------------------------------------------------------- #
def test_p7_one_transport_line_then_silent_silence_trips_when_transport_off():
    class _BlipThenSilent(_Base):
        def build_command(self, piped_input=None):
            return ["bash", "-c",
                    "echo 'connection reset by peer' 1>&2; sleep 30"]

    agent = _BlipThenSilent(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 0
    agent.silence_max_seconds = 0.3
    _disarm_transport(agent)          # ONLY the silence guard is eligible
    agent.timeout = 60
    agent.absolute_timeout = 0
    agent.max_retries = 1

    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run_async())
    elapsed = time.monotonic() - t0

    assert agent._watchdog_reason == WATCHDOG_STALLED
    assert elapsed < 8, f"silence trip after a transport line took {elapsed:.1f}s"


# --------------------------------------------------------------------------- #
# P8  Same shape as P7 but with BOTH the silence ceiling AND the transport bound
#     armed tight. The watchdog checks transport BEFORE silence, so the single
#     transport error (which opens a spell) trips WATCHDOG_TRANSPORT_STALL first.
#     Either way it is WALL-CLOCK BOUNDED and distinctly classified — the key
#     fuzz guarantee is "one of them trips, fast, with a sensible reason".
# --------------------------------------------------------------------------- #
def test_p8_transport_then_silent_both_armed_trips_bounded_and_classified():
    class _BlipThenSilent(_Base):
        def build_command(self, piped_input=None):
            return ["bash", "-c",
                    "echo 'connection reset by peer' 1>&2; sleep 30"]

    agent = _BlipThenSilent(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 0
    agent.silence_max_seconds = 0.4
    agent.transport_max_errors = 1     # one error opens+exceeds the spell
    agent.transport_max_seconds = 0.4
    agent.transport_recovery_window = 30  # long enough not to clear the spell
    agent.timeout = 60
    agent.absolute_timeout = 0
    agent.max_retries = 1

    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run_async())
    elapsed = time.monotonic() - t0

    # Bounded, and classified as one of the two legitimate stall reasons.
    assert agent._watchdog_reason in (WATCHDOG_TRANSPORT_STALL, WATCHDOG_STALLED)
    assert WATCHDOG_MARKER in agent.stderr
    assert elapsed < 8, f"both-armed trip took {elapsed:.1f}s"


# --------------------------------------------------------------------------- #
# P9  The silence trip races teardown: a silent wedge tripped by the watchdog
#     (killpg) must wind down cleanly through _await_with_liveness's teardown —
#     no double-kill wedge, no hung gather, single bounded return. We run the
#     SAME call many times back-to-back; any teardown wedge/leak would make the
#     aggregate wall-clock blow past the per-call silence bound x N, or leave a
#     dangling process. Assert all N return promptly with the stall reason.
# --------------------------------------------------------------------------- #
def test_p9_silence_trip_teardown_no_double_kill_or_wedge():
    class _Silent(_Base):
        def build_command(self, piped_input=None):
            return ["bash", "-c", "sleep 30"]

    def _one():
        agent = _Silent(prompt="x")
        agent.max_output_bytes = 0
        agent.stall_seconds = 0
        agent.silence_max_seconds = 0.3
        _disarm_transport(agent)
        agent.timeout = 60
        agent.absolute_timeout = 0
        agent.max_retries = 1
        with pytest.raises(RuntimeError):
            asyncio.run(agent.run_async())
        assert agent._watchdog_reason == WATCHDOG_STALLED
        # The tracked process handle is cleared after a kill -> no leaked handle.
        assert getattr(agent, "_current_process", None) is None

    t0 = time.monotonic()
    for _ in range(4):
        _one()
    elapsed = time.monotonic() - t0
    # 4 calls, each bounded ~<4s; a teardown wedge would balloon this. Generous
    # ceiling keeps it non-flaky while still catching an unbounded hang.
    assert elapsed < 30, f"repeated silence-trip teardown took {elapsed:.1f}s"


# --------------------------------------------------------------------------- #
# P10  REGRESSION (residual #75 gap, NOW FIXED). When BOTH the silence ceiling and
#      stall_seconds are disabled (the OLD gap, P4), a silent (zero-output) wedge
#      that hits the idle ceiling is classified distinctly as a STALL (a dedicated
#      _watchdog_reason == WATCHDOG_STALLED) so the fallback layer can route it on
#      the stall rule and telemetry doesn't fold it into a generic idle timeout.
#      The fix tags a zero-output idle/ceiling timeout (not a wall-clock-cap overrun,
#      not an emit-then-hang) as WATCHDOG_STALLED in the run_async timeout handler.
# --------------------------------------------------------------------------- #
def test_p10_old_gap_silent_wedge_should_be_classified_stalled():
    class _Silent(_Base):
        def build_command(self, piped_input=None):
            return ["bash", "-c", "sleep 30"]

    agent = _Silent(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 0
    agent.silence_max_seconds = 0     # DISABLED -> old gap
    _disarm_transport(agent)
    agent.timeout = 1.0
    agent.absolute_timeout = 0
    agent.max_retries = 1

    with pytest.raises(RuntimeError):
        asyncio.run(agent.run_async())

    # DESIRED: a silent (zero-output) wedge that hit the idle ceiling is tagged
    # as a stall so it can be routed/telemetered as one. Currently None -> fails.
    assert agent._watchdog_reason == WATCHDOG_STALLED

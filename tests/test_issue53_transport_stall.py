"""Issue #53 — the stall watchdog must trip on a transport stall: a worker that
keeps emitting "connection reset / retrying" noise on stderr with zero forward
progress used to ride that noise as liveness and hang until the hard timeout.
Transport-noise stderr lines must NOT count as progress; the trip is labelled
WATCHDOG_TRANSPORT_STALL so the fallback layer can re-route immediately.
"""
from __future__ import annotations

import asyncio
import time

from agy_orchestrator.core.agent import (
    WATCHDOG_MARKER,
    WATCHDOG_STALLED,
    WATCHDOG_TRANSPORT_STALL,
    AgentInstance,
    _is_transport_noise_line,
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


def test_transport_noise_line_classifier():
    assert _is_transport_noise_line("websocket connection reset; retrying")
    assert _is_transport_noise_line("network timeout, retrying request")
    assert _is_transport_noise_line("stream reset by peer")
    # Real worker chatter is NOT transport noise -> stays liveness.
    assert not _is_transport_noise_line("applied patch to foo.py")
    assert not _is_transport_noise_line("thinking about the problem")


def test_transport_spam_trips_transport_stall():
    """stderr that only emits reset/retry noise (no forward progress) trips the
    transport-stall watchdog rather than holding the worker alive to the hard cap."""

    class _TransportSpam(_Base):
        def build_command(self, piped_input=None):
            # Many transport-noise stderr lines over ~3s, no stdout, no real
            # stderr progress. With stall_seconds=1.0 the watchdog must trip.
            return [
                "bash", "-c",
                "for i in $(seq 1 30); do echo 'connection reset, retrying' >&2; "
                "sleep 0.1; done",
            ]

    agent = _TransportSpam(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 1.0
    agent.timeout = 30
    agent.max_retries = 1

    t0 = time.monotonic()
    import pytest
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run_async())
    elapsed = time.monotonic() - t0

    assert agent._watchdog_reason == WATCHDOG_TRANSPORT_STALL
    assert WATCHDOG_MARKER in agent.stderr
    assert WATCHDOG_TRANSPORT_STALL in agent.stderr
    # Trips within a few 2s poll slices, not the 30s hard timeout.
    assert elapsed < 10


def test_real_progress_then_transport_spam_still_trips():
    """A worker that makes REAL progress first (so any_output is True), then falls
    into transport-noise-only spam, must still trip — the old `not any_output`
    gate would have let this hang."""

    class _ProgressThenSpam(_Base):
        def build_command(self, piped_input=None):
            return [
                "bash", "-c",
                # one real progress line, then nothing but transport noise.
                "echo 'building project' >&2; "
                "for i in $(seq 1 30); do echo 'websocket reset; retrying' >&2; "
                "sleep 0.1; done",
            ]

    agent = _ProgressThenSpam(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 1.0
    agent.timeout = 30
    agent.max_retries = 1

    import pytest
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run_async())
    assert agent._watchdog_reason == WATCHDOG_TRANSPORT_STALL


def test_genuinely_silent_worker_still_plain_stall():
    """No output on either stream -> classic STALLED, not transport-stall."""

    class _Silent(_Base):
        def build_command(self, piped_input=None):
            return ["bash", "-c", "sleep 10"]

    agent = _Silent(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 1.0
    agent.timeout = 30
    agent.max_retries = 1

    import pytest
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run_async())
    assert agent._watchdog_reason == WATCHDOG_STALLED


def test_real_stderr_progress_does_not_trip():
    """Non-noise stderr work must still reset the stall clock (no false trip) —
    backward-compat with the existing stderr-progress behaviour."""

    class _RealWork(_Base):
        def build_command(self, piped_input=None):
            return ["bash", "-c",
                    "for i in $(seq 1 6); do echo 'compiling module' >&2; sleep 0.3; done"]

    agent = _RealWork(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 1.0
    agent.timeout = 30
    agent.max_retries = 1
    asyncio.run(agent.run_async())
    assert agent.returncode == 0
    assert agent._watchdog_reason is None


def test_transport_stall_marker_reroutes_via_watchdog_rules():
    """The WATCHDOG_TRANSPORT_STALL marker on stderr must be routable by the
    existing _watchdog_rules mechanism in FallbackAgent (issue #53)."""

    class _StallTrip(_Base):
        async def run_async(self, piped_input=None):
            self.stderr = f"{WATCHDOG_MARKER}{WATCHDOG_TRANSPORT_STALL}] transport"
            raise RuntimeError(self.stderr)

    class _Escalated(_Base):
        invoked = False

        async def run_async(self, piped_input=None):
            type(self).invoked = True
            self.stdout = "escalated-ok"
            self.returncode = 0
            return self.stdout

    _Escalated.invoked = False
    # Disable same-provider transport retries so the watchdog rule is what routes.
    import os
    prev = os.environ.get("AGY_TRANSPORT_RETRIES")
    os.environ["AGY_TRANSPORT_RETRIES"] = "0"
    try:
        Agent = make_fallback_agent(
            chain=[_StallTrip],
            cycles=1,
            watchdog_rules={WATCHDOG_TRANSPORT_STALL: [_Escalated]},
        )
        out = asyncio.run(Agent(prompt="x").run_async())
    finally:
        if prev is None:
            os.environ.pop("AGY_TRANSPORT_RETRIES", None)
        else:
            os.environ["AGY_TRANSPORT_RETRIES"] = prev

    assert out == "escalated-ok"
    assert _Escalated.invoked is True

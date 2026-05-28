"""Streaming watchdog must kill a child whose output exceeds the byte budget,
record the trip reason, and surface a [watchdog:reason] marker on self.stderr so
the FallbackAgent's rule table can re-route the next attempt deterministically.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from agy_orchestrator.core.agent import (
    WATCHDOG_MARKER,
    WATCHDOG_STALLED,
    WATCHDOG_VERBOSE,
    AgentInstance,
)


class _SpewAgent(AgentInstance):
    """Floods stdout fast enough to blow any sensible byte budget within ~1s."""

    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        # 'yes' emits 'y\n' as fast as the kernel will schedule it. Pipe through
        # head to bound the test in case the watchdog regresses (we still expect
        # the watchdog to trip well before head finishes).
        return ["bash", "-c", "yes | head -c 5000000"]


def test_watchdog_verbose_kills_and_records_reason():
    agent = _SpewAgent(prompt="x")
    agent.max_output_bytes = 32_000      # ~32 KB; 'yes' clears that in <100ms
    agent.stall_seconds = 0
    agent.timeout = 30                   # well above the 2s watchdog poll
    agent.max_retries = 1

    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run_async())
    elapsed = time.monotonic() - t0

    assert agent._watchdog_reason == WATCHDOG_VERBOSE
    assert WATCHDOG_MARKER in agent.stderr
    assert "verbose" in agent.stderr
    # Watchdog polls every 2s, so worst-case ~4s. 10s ceiling = plenty of margin
    # without making the test pass for the wrong reason (hard timeout was 30s).
    assert elapsed < 10


def test_watchdog_disabled_lets_normal_runs_complete():
    """With NO budgets, the existing fast-path still works (no false trips)."""

    class _Echo(_SpewAgent):
        def build_command(self, piped_input=None):
            return ["true"]

    agent = _Echo(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 0
    agent.max_retries = 1
    asyncio.run(agent.run_async())
    assert agent.returncode == 0
    assert agent._watchdog_reason is None


def test_stderr_progress_resets_stall_clock():
    """Stderr activity must reset the stall clock — codex/agy/grok/claude all
    emit their work tracking to stderr while keeping stdout silent until the
    final reply. Before this fix, a busy worker that hadn't yet written to
    stdout looked stalled at 180s and got killed mid-reasoning."""

    class _StderrBusyAgent(_SpewAgent):
        def build_command(self, piped_input=None):
            # Six stderr writes over ~1.8s, no stdout. With stall_seconds=1.0
            # and progress only counting stdout (the old behaviour), this would
            # trip near 1s — well before the loop finishes. With the fix it
            # rides through to the natural exit.
            return ["bash", "-c", "for i in $(seq 1 6); do echo work >&2; sleep 0.3; done"]

    agent = _StderrBusyAgent(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 1.0
    agent.timeout = 30
    agent.max_retries = 1
    asyncio.run(agent.run_async())
    assert agent.returncode == 0
    assert agent._watchdog_reason is None


def test_watchdog_still_kills_truly_silent_worker():
    """A worker silent on BOTH streams past stall_seconds is genuinely hung
    and must still be killed — the stderr-progress fix must not turn the
    stall watchdog into a no-op."""

    class _SilentAgent(_SpewAgent):
        def build_command(self, piped_input=None):
            return ["bash", "-c", "sleep 10"]

    agent = _SilentAgent(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 1.0
    agent.timeout = 30
    agent.max_retries = 1

    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run_async())
    elapsed = time.monotonic() - t0

    assert agent._watchdog_reason == WATCHDOG_STALLED
    # Watchdog polls every 2s, so worst-case ~3-4s with a 1s stall budget.
    # 8s ceiling = comfortable margin without masking a regression.
    assert elapsed < 8

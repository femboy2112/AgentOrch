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

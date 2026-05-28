"""The base agent must kill and fail a subprocess that hangs past its timeout,
so a stalled worker fails over instead of hanging the dispatch forever.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from agy_orchestrator.core.agent import AgentInstance


class _SleepAgent(AgentInstance):
    """Runs `sleep` far longer than the timeout — stands in for a hung CLI."""

    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["sleep", "30"]


def test_run_async_times_out_and_kills():
    agent = _SleepAgent(prompt="x")
    agent.timeout = 0.5          # force a fast ceiling
    agent.max_retries = 1

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        asyncio.run(agent.run_async())
    elapsed = time.monotonic() - started

    # Failed fast (well under the 30s sleep), and recorded why.
    assert elapsed < 10
    assert "timed out" in agent.stderr


def test_timeout_disabled_when_zero(monkeypatch):
    # timeout=0 must mean "no ceiling": a quick command still completes normally.
    class _Echo(_SleepAgent):
        def build_command(self, piped_input=None):
            return ["true"]

    agent = _Echo(prompt="x")
    agent.timeout = 0
    asyncio.run(agent.run_async())
    assert agent.returncode == 0

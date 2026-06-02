"""Issue #69: after a worker call completes, its asyncio subprocess transport must
be closed IN-loop so it isn't finalized after the event loop is torn down — the
source of the "RuntimeError: Event loop is closed" BaseSubprocessTransport.__del__
noise at master-run teardown. (Child reaping itself is covered by #66.)
"""
from __future__ import annotations

import asyncio
import gc
import warnings

import pytest

from agy_orchestrator.core.agent import AgentInstance


class _EchoAgent(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["sh", "-c", "echo hello"]


@pytest.mark.not_slow
def test_transport_closed_after_stream_call():
    agent = _EchoAgent(prompt="x")
    agent.stall_seconds = 30          # arm the streaming path
    agent.max_retries = 1
    out = asyncio.run(agent.run_async())
    assert "hello" in out
    # The tracked process is cleared and its transport closed (no lingering handle).
    assert agent._current_process is None


@pytest.mark.not_slow
def test_close_transport_helper_is_safe_on_junk():
    # Best-effort: never raises on a missing/odd transport.
    class _NoTransport:
        pass
    AgentInstance._close_transport(_NoTransport())  # no _transport attr -> no-op


@pytest.mark.not_slow
def test_no_event_loop_closed_warning_after_run():
    # Run a streaming worker call, then force GC and assert no "Event loop is
    # closed" ResourceWarning/RuntimeError leaks from an unfinalized transport.
    agent = _EchoAgent(prompt="x")
    agent.stall_seconds = 30
    agent.max_retries = 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(agent.run_async())
        gc.collect()
    msgs = [str(w.message) for w in caught]
    assert not any("Event loop is closed" in m for m in msgs), msgs

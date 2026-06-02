"""Issue #66: a worker that exits while a child it spawned still holds the
inherited stdout/stderr pipe must NOT wedge the orchestrator forever in ep_poll.
After the worker exits we reap its process group (closing the inherited fds) and
bound the post-exit drain, so the call returns promptly instead of blocking until
the orphan happens to finish (~30 min in the live report).
"""
from __future__ import annotations

import asyncio
import shutil
import time

import pytest

from agy_orchestrator.core.agent import AgentInstance


class _OrphanPipeAgent(AgentInstance):
    """Worker that backgrounds a long-lived child holding the stdout pipe, then
    exits 0 itself — standing in for a worker whose self-check outlives it."""

    cmd_str = ""

    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["sh", "-c", self.cmd_str]


@pytest.mark.not_slow
def test_worker_exit_with_child_holding_pipe_does_not_hang():
    agent = _OrphanPipeAgent(prompt="x")
    # `sleep 30 &` inherits stdout (the pipe) and outlives the worker, which exits
    # immediately. Without the fix the drain blocks ~30s waiting for EOF; with it,
    # reaping the worker's process group kills the child and the pipe EOFs at once.
    agent.cmd_str = "sleep 30 & echo started; exit 0"
    agent.stall_seconds = 60   # arm the streaming path (won't trip; worker exits fast)
    agent.max_retries = 1

    started = time.monotonic()
    out = asyncio.run(agent.run_async())
    elapsed = time.monotonic() - started

    assert elapsed < 10, f"worker call hung {elapsed:.1f}s on an orphan-held pipe"
    assert "started" in out


@pytest.mark.not_slow
@pytest.mark.skipif(shutil.which("setsid") is None, reason="setsid not available")
def test_escaped_grandchild_pipe_bounded_by_grace(monkeypatch):
    # `setsid sleep 30 &` runs the child in its OWN session, escaping the worker's
    # process group, so killpg can't reap it. The post-exit drain grace must still
    # bound the wait so the call returns instead of hanging the full 30s.
    monkeypatch.setenv("AGY_WORKER_DRAIN_GRACE", "2")
    agent = _OrphanPipeAgent(prompt="x")
    agent.cmd_str = "setsid sleep 30 & echo started; exit 0"
    agent.stall_seconds = 60
    agent.max_retries = 1

    started = time.monotonic()
    asyncio.run(agent.run_async())
    elapsed = time.monotonic() - started

    # Bounded by the 2s grace (plus scheduling slack), nowhere near the 30s sleep.
    assert elapsed < 10, f"escaped-grandchild pipe not bounded: {elapsed:.1f}s"


@pytest.mark.not_slow
def test_normal_worker_still_returns_full_output():
    # The fix must not truncate a normal worker's output: a worker that prints and
    # exits cleanly (no lingering child) returns everything.
    agent = _OrphanPipeAgent(prompt="x")
    agent.cmd_str = "echo line1; echo line2; echo line3"
    agent.stall_seconds = 60
    agent.max_retries = 1
    out = asyncio.run(agent.run_async())
    assert "line1" in out and "line2" in out and "line3" in out

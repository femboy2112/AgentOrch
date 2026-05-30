"""The base agent must kill and fail a subprocess that hangs past its timeout,
so a stalled worker fails over instead of hanging the dispatch forever.
"""
from __future__ import annotations

import asyncio
import os
import sys
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


class _GrandchildAgent(AgentInstance):
    """Spawns a grandchild `sleep`, records its pid, then hangs.

    Stands in for a worker CLI (e.g. `codex`) whose heavy child (`codex exec`
    running `make check`) must NOT survive a timeout kill by escaping to init.
    """

    pidfile = ""

    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        code = (
            "import subprocess, sys, time; "
            "g = subprocess.Popen(['sleep', '30']); "
            "open(sys.argv[1], 'w').write(str(g.pid)); "
            "time.sleep(30)"
        )
        return [sys.executable, "-c", code, self.pidfile]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_timeout_reaps_whole_process_group_not_just_child(tmp_path):
    """Runaway-codex regression: killing on timeout must reap the grandchild too.

    Without start_new_session + killpg, the grandchild `sleep` would reparent to
    init and keep running. The group kill must take the entire tree down.
    """
    pidfile = tmp_path / "grand.pid"
    agent = _GrandchildAgent(prompt="x")
    agent.pidfile = str(pidfile)
    agent.timeout = 0.8
    agent.max_retries = 1

    with pytest.raises(RuntimeError, match="timed out"):
        asyncio.run(agent.run_async())

    # The grandchild recorded its pid before the parent hung.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not pidfile.exists():
        time.sleep(0.05)
    assert pidfile.exists(), "grandchild never recorded its pid"
    gpid = int(pidfile.read_text().strip())

    # After the timeout kill, the grandchild must be gone (poll for reap).
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and _alive(gpid):
        time.sleep(0.05)
    assert not _alive(gpid), "grandchild sleep survived the timeout kill — tree not reaped"


def test_timeout_disabled_when_zero(monkeypatch):
    # timeout=0 must mean "no ceiling": a quick command still completes normally.
    class _Echo(_SleepAgent):
        def build_command(self, piped_input=None):
            return ["true"]

    agent = _Echo(prompt="x")
    agent.timeout = 0
    asyncio.run(agent.run_async())
    assert agent.returncode == 0

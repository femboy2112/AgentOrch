"""Issue #49 — no wall-clock bound on a long-but-active worker-spawned command.

A worker ran ``pytest -m slow`` directly for 44+ min inside one master step. The
idle watchdog (AGY_STALL_SECONDS) never fired because the test emitted periodic
output ("active", not "stalled"), and the default hard cap (4x the idle window =
160 min) was far too lax to stop it. The orchestrator can't wrap the worker's
*internal* command in ``timeout``, but it CAN bound the worker call that contains
it, and it can propagate the loop's ``-m 'not slow'`` selection policy into the
worker env. These tests pin both knobs (both opt-in, default-off).
"""

import asyncio
import time
from typing import List, Optional

import pytest

from agy_orchestrator.core.agent import (
    ABSOLUTE_TIMEOUT_FACTOR,
    AgentInstance,
    apply_worker_resource_bounds,
)


class _StubAgent(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input: Optional[str] = None) -> List[str]:
        return ["true"]


# --- proposal #2: test-selection scope propagation -------------------------------

def test_marker_scope_not_injected_by_default(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.delenv("AGY_WORKER_PYTEST_MARKERS", raising=False)
    monkeypatch.delenv("AGY_WORKER_PYTEST_XDIST", raising=False)
    # Default-off: only the #48 xdist bound, no -m (slow is a per-repo convention).
    assert apply_worker_resource_bounds({})["PYTEST_ADDOPTS"] == "-n 2"


def test_marker_scope_injected_when_set(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.delenv("AGY_WORKER_PYTEST_XDIST", raising=False)
    monkeypatch.setenv("AGY_WORKER_PYTEST_MARKERS", "not slow")
    out = apply_worker_resource_bounds({})
    # shlex-quoted so pytest parses "not slow" as a single -m expression token.
    assert out["PYTEST_ADDOPTS"] == "-n 2 -m 'not slow'"


def test_marker_scope_preserves_existing_addopts(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.delenv("AGY_WORKER_PYTEST_XDIST", raising=False)
    monkeypatch.setenv("AGY_WORKER_PYTEST_MARKERS", "not slow")
    out = apply_worker_resource_bounds({"PYTEST_ADDOPTS": "-q"})
    assert out["PYTEST_ADDOPTS"] == "-q -n 2 -m 'not slow'"


# --- proposal #1/#3: hard per-worker-call wall-clock cap -------------------------

def test_absolute_cap_default_is_factor_of_timeout(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_CMD_TIMEOUT", raising=False)
    monkeypatch.delenv("AGY_ABSOLUTE_TIMEOUT", raising=False)
    monkeypatch.delenv("AGY_TIMEOUT", raising=False)
    agent = _StubAgent(prompt="x")
    # Unchanged default behavior (no #28 regression): 4 x the 2400s idle window.
    assert agent._absolute_cap() == pytest.approx(2400.0 * ABSOLUTE_TIMEOUT_FACTOR)


def test_worker_cmd_timeout_tightens_the_cap(monkeypatch):
    monkeypatch.delenv("AGY_ABSOLUTE_TIMEOUT", raising=False)
    monkeypatch.setenv("AGY_WORKER_CMD_TIMEOUT", "1200")
    agent = _StubAgent(prompt="x")
    assert agent._absolute_cap() == 1200.0


def test_tightest_of_both_caps_wins(monkeypatch):
    monkeypatch.setenv("AGY_ABSOLUTE_TIMEOUT", "7200")
    monkeypatch.setenv("AGY_WORKER_CMD_TIMEOUT", "1200")
    agent = _StubAgent(prompt="x")
    assert agent._absolute_cap() == 1200.0  # min, not last-set
    monkeypatch.setenv("AGY_WORKER_CMD_TIMEOUT", "9000")
    agent2 = _StubAgent(prompt="x")
    assert agent2._absolute_cap() == 7200.0  # absolute_timeout now the tighter


def test_liveness_wall_clock_kill_fires_on_active_output(monkeypatch):
    """The crux of #49: a child that keeps emitting output must still be killed
    once it exceeds the hard cap, even though the idle window never elapses."""
    monkeypatch.setenv("AGY_WORKER_CMD_TIMEOUT", "0.3")
    agent = _StubAgent(prompt="x")
    agent.timeout = 100.0  # huge idle window -> idle watchdog would never fire

    async def _never_returns():
        # Simulate a long-but-active command: bump _last_progress constantly so
        # the idle path can't trip; only the wall-clock cap can stop us.
        while True:
            agent._last_progress = time.monotonic()
            await asyncio.sleep(0.02)

    async def _drive():
        start = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await agent._await_with_liveness(_never_returns(), start)
        return time.monotonic() - start

    elapsed = asyncio.run(_drive())
    assert agent._timeout_kind == "wallclock"
    assert elapsed < 5.0  # killed near the 0.3s cap, not left to run unbounded

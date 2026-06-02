"""Regression tests for core/agents/fallback_agent.py — cross-provider fallback
chain (chain/cycles/rotation/transport/watchdog-reroute).

fallback-1 (blocker): a self- or mutually-referencing watchdog_rules table made
run_async() re-splice a fresh re-route target onto ``pending`` on every failure,
so ``pending`` never drained and the loop spun forever (CPU-bound busy loop, no
await point on the failing path -> not even asyncio.wait_for could cancel it).
The documented invariant "a bad chain order can't get stuck in an infinite
rules-table ping-pong" was therefore false.

Fix: a global re-route budget caps the total number of rule-splice applications
so the chain always drains to the exhaustion RuntimeError. These tests run fully
hermetic (no subprocess, no network); each must terminate fast (the bug would
otherwise hang, so the implicit oracle is "returns at all").
"""
from __future__ import annotations

import asyncio

import pytest

from agy_orchestrator.core.agent import (
    WATCHDOG_MARKER,
    WATCHDOG_TRANSPORT_STALL,
    WATCHDOG_VERBOSE,
    AgentInstance,
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


def _run_bounded(coro_factory, timeout=8.0):
    """Run a coroutine under a hard wall-clock cap so a regression that re-hangs
    surfaces as a failure rather than wedging the suite. The failing path between
    awaits is synchronous, so wait_for cannot cancel a true busy-loop hang; we run
    the coroutine in a worker thread and join with a timeout instead."""
    import threading

    box = {}

    def _worker():
        try:
            box["result"] = asyncio.run(coro_factory())
        except BaseException as exc:  # noqa: BLE001 - capture for the caller
            box["error"] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise AssertionError(
            "run_async did not terminate within the bound — fallback-1 regression "
            "(unbounded watchdog-rule re-route loop)"
        )
    if "error" in box:
        raise box["error"]
    return box.get("result")


def test_mutually_referencing_rules_drain_to_exhaustion():
    """A trips 'verbose'->[B], B trips 'stalled'->[A]: the original bug looped
    forever. The chain must instead drain to the exhaustion RuntimeError."""

    class A(_Base):
        async def run_async(self, piped_input=None):
            self.stderr = f"{WATCHDOG_MARKER}{WATCHDOG_VERBOSE}] x"
            raise RuntimeError("a")

    class B(_Base):
        async def run_async(self, piped_input=None):
            self.stderr = f"{WATCHDOG_MARKER}stalled] x"
            raise RuntimeError("b")

    Agent = make_fallback_agent(
        chain=[A],
        cycles=1,
        watchdog_rules={WATCHDOG_VERBOSE: [B], "stalled": [A]},
    )

    with pytest.raises(RuntimeError, match="exhausted"):
        _run_bounded(lambda: Agent(prompt="x").run_async())


def test_self_referencing_rule_drains_to_exhaustion():
    """A single self-referencing rule (V trips 'verbose'->[V]) must terminate."""

    class V(_Base):
        async def run_async(self, piped_input=None):
            self.stderr = f"{WATCHDOG_MARKER}{WATCHDOG_VERBOSE}] x"
            raise RuntimeError("v")

    Agent = make_fallback_agent(
        chain=[V],
        cycles=1,
        watchdog_rules={WATCHDOG_VERBOSE: [V]},
    )

    with pytest.raises(RuntimeError, match="exhausted"):
        _run_bounded(lambda: Agent(prompt="x").run_async())


def test_self_referencing_transport_stall_rule_drains(monkeypatch):
    """transport_stall is also affected: with same-provider transport retries
    disabled, a {WATCHDOG_TRANSPORT_STALL:[T]} self-rule used to re-splice forever
    once the bounded retries were spent. It must now drain."""

    monkeypatch.setenv("AGY_TRANSPORT_RETRIES", "0")

    class T(_Base):
        async def run_async(self, piped_input=None):
            self.stderr = f"{WATCHDOG_MARKER}{WATCHDOG_TRANSPORT_STALL}] transport"
            raise RuntimeError("t")

    Agent = make_fallback_agent(
        chain=[T],
        cycles=1,
        watchdog_rules={WATCHDOG_TRANSPORT_STALL: [T]},
    )

    with pytest.raises(RuntimeError, match="exhausted"):
        _run_bounded(lambda: Agent(prompt="x").run_async())


def test_legitimate_single_reroute_still_fires():
    """The fix must NOT regress the intended behaviour: a verbose trip still routes
    to its rule target before the normal-next provider, and the target succeeds."""

    class _VerboseTrip(_Base):
        async def run_async(self, piped_input=None):
            self.stderr = f"{WATCHDOG_MARKER}{WATCHDOG_VERBOSE}] x"
            raise RuntimeError(self.stderr)

    class _TerseTarget(_Base):
        invoked = False

        async def run_async(self, piped_input=None):
            type(self).invoked = True
            self.stdout = "terse-ok"
            self.returncode = 0
            return self.stdout

    class _NormalNext(_Base):
        invoked = False

        async def run_async(self, piped_input=None):
            type(self).invoked = True
            self.stdout = "normal-ok"
            self.returncode = 0
            return self.stdout

    _TerseTarget.invoked = False
    _NormalNext.invoked = False

    Agent = make_fallback_agent(
        chain=[_VerboseTrip, _NormalNext],
        cycles=1,
        watchdog_rules={WATCHDOG_VERBOSE: [_TerseTarget]},
    )

    out = _run_bounded(lambda: Agent(prompt="x").run_async())
    assert out == "terse-ok"
    assert _TerseTarget.invoked is True
    assert _NormalNext.invoked is False


def test_reroute_fires_across_multiple_cycles_before_budget_caps():
    """The budget is generous enough that a legitimate per-trip re-route fires on
    every cycle of a normal chain — the cap only bites pathological self-loops.
    Here a verbose-tripping lead routes to a target that itself fails; across 3
    cycles the target must be reached every cycle (not starved by the budget) yet
    the run still terminates."""

    class _Verbose(_Base):
        async def run_async(self, piped_input=None):
            self.stderr = f"{WATCHDOG_MARKER}{WATCHDOG_VERBOSE}] x"
            raise RuntimeError("verbose")

    class _Target(_Base):
        count = 0

        async def run_async(self, piped_input=None):
            type(self).count += 1
            self.stderr = "plain failure"
            raise RuntimeError("target-fail")

    _Target.count = 0

    Agent = make_fallback_agent(
        chain=[_Verbose],
        cycles=3,
        watchdog_rules={WATCHDOG_VERBOSE: [_Target]},
    )

    with pytest.raises(RuntimeError, match="exhausted"):
        _run_bounded(lambda: Agent(prompt="x").run_async())
    # The re-route target was reached at least once per cycle (legitimate fan-out
    # preserved), proving the budget did not strangle normal re-routing.
    assert _Target.count >= 3

"""Issue #57 — FallbackAgent must classify a transport/network blip distinctly
from a quota wall, and prefer a BOUNDED same-provider backoff-retry before
advancing to the next provider (the connection failed, not the provider).
"""
from __future__ import annotations

import asyncio

import pytest

from agy_orchestrator.core.agent import (
    WATCHDOG_MARKER,
    WATCHDOG_TRANSPORT_STALL,
    AgentInstance,
    is_context_overflow,
    is_transport_error,
    is_usage_wall,
)
from agy_orchestrator.core.agents.fallback_agent import (
    _reason_category,
    make_fallback_agent,
)


# --- the shared classifier --------------------------------------------------

def test_is_transport_error_matches_network_blips():
    for blip in (
        "websocket connection reset by peer",
        "Error: read ECONNRESET",
        "stream error: stream ID 5",
        "service temporarily unavailable",
        "HTTP 503 Service Unavailable",
        "upstream returned 502",
    ):
        assert is_transport_error(blip), blip


def test_is_transport_error_conservative_about_quota_and_overflow():
    # A quota wall is NEVER a transport error, even if a future marker overlaps.
    assert is_usage_wall("usage limit reached")
    assert not is_transport_error("usage limit reached")
    # Context overflow is its own category, not transport.
    assert is_context_overflow("context_length_exceeded")
    assert not is_transport_error("context_length_exceeded")
    # Plain unrelated error is not transport.
    assert not is_transport_error("SyntaxError: invalid token")


def test_reason_category_transport():
    assert _reason_category(looked_like_usage=False, watchdog_reason=None,
                            transport=True) == "transport"
    # The watchdog transport-stall trip funnels into the transport category too.
    assert _reason_category(looked_like_usage=False,
                            watchdog_reason=WATCHDOG_TRANSPORT_STALL) == "transport"
    # A real quota wall still wins over a (false) transport flag.
    assert _reason_category(looked_like_usage=True, watchdog_reason=None,
                            transport=True) == "usage"


# --- bounded same-provider retry --------------------------------------------

class _BaseStub(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]


def test_transport_blip_retries_same_provider_then_succeeds(monkeypatch):
    # No backoff sleep in the test.
    monkeypatch.setenv("AGY_TRANSPORT_BACKOFF", "0")
    monkeypatch.setenv("AGY_TRANSPORT_RETRIES", "2")

    class _FlakyThenOK(_BaseStub):
        calls = 0

        async def run_async(self, piped_input=None):
            type(self).calls += 1
            if type(self).calls == 1:
                self.stderr = "websocket connection reset; retrying"
                raise RuntimeError(self.stderr)
            self.stdout = "recovered"
            self.returncode = 0
            return self.stdout

    class _NextProvider(_BaseStub):
        invoked = False

        async def run_async(self, piped_input=None):
            type(self).invoked = True
            self.stdout = "next"
            self.returncode = 0
            return self.stdout

    _FlakyThenOK.calls = 0
    _NextProvider.invoked = False
    Agent = make_fallback_agent(chain=[_FlakyThenOK, _NextProvider], cycles=1)
    out = asyncio.run(Agent(prompt="x").run_async())

    assert out == "recovered"
    assert _FlakyThenOK.calls == 2          # one blip + one same-provider retry
    assert _NextProvider.invoked is False    # never had to advance


def test_transport_retries_are_bounded_then_advances(monkeypatch):
    monkeypatch.setenv("AGY_TRANSPORT_BACKOFF", "0")
    monkeypatch.setenv("AGY_TRANSPORT_RETRIES", "2")

    class _AlwaysReset(_BaseStub):
        calls = 0

        async def run_async(self, piped_input=None):
            type(self).calls += 1
            self.stderr = "Error: read ECONNRESET"
            raise RuntimeError(self.stderr)

    class _NextProvider(_BaseStub):
        invoked = False

        async def run_async(self, piped_input=None):
            type(self).invoked = True
            self.stdout = "next-ok"
            self.returncode = 0
            return self.stdout

    _AlwaysReset.calls = 0
    _NextProvider.invoked = False
    Agent = make_fallback_agent(chain=[_AlwaysReset, _NextProvider], cycles=1)
    out = asyncio.run(Agent(prompt="x").run_async())

    assert out == "next-ok"
    # 1 initial attempt + 2 bounded same-provider retries, THEN advance.
    assert _AlwaysReset.calls == 3
    assert _NextProvider.invoked is True


def test_usage_wall_does_not_get_transport_retry(monkeypatch):
    monkeypatch.setenv("AGY_TRANSPORT_BACKOFF", "0")

    class _Walled(_BaseStub):
        calls = 0

        async def run_async(self, piped_input=None):
            type(self).calls += 1
            self.stderr = "usage limit reached for this plan"
            raise RuntimeError(self.stderr)

    class _NextProvider(_BaseStub):
        async def run_async(self, piped_input=None):
            self.stdout = "next-ok"
            self.returncode = 0
            return self.stdout

    _Walled.calls = 0
    Agent = make_fallback_agent(chain=[_Walled, _NextProvider], cycles=1)
    out = asyncio.run(Agent(prompt="x").run_async())

    assert out == "next-ok"
    # A quota wall advances immediately — exactly ONE call, no same-provider retry.
    assert _Walled.calls == 1


def test_transport_retry_emits_distinct_telemetry(monkeypatch):
    monkeypatch.setenv("AGY_TRANSPORT_BACKOFF", "0")
    monkeypatch.setenv("AGY_TRANSPORT_RETRIES", "1")

    events = []

    class _Flaky(_BaseStub):
        calls = 0

        async def run_async(self, piped_input=None):
            type(self).calls += 1
            if type(self).calls == 1:
                self.stderr = "stream error: server reset"
                raise RuntimeError(self.stderr)
            self.stdout = "ok"
            self.returncode = 0
            return self.stdout

    _Flaky.calls = 0
    Agent = make_fallback_agent(chain=[_Flaky], cycles=1)
    inst = Agent(prompt="x")
    inst.event_callback = lambda ev: events.append(ev)
    asyncio.run(inst.run_async())

    transitions = [
        e["data"]["orchestration"]
        for e in events
        if e.get("kind") == "lifecycle"
        and e.get("data", {}).get("event") == "orchestration_transition"
    ]
    retries = [t for t in transitions if t.get("action") == "transport_retry"]
    assert retries, transitions
    assert retries[0]["reason_category"] == "transport"
    assert retries[0]["from_worker"] == retries[0]["to_worker"]  # same provider

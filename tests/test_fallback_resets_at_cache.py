"""Issue #82 — FallbackAgent must not re-probe a rate-limited (provider, model) on
every call. When a usage/quota wall reports a reset time (codex stashes the rollout
reset epoch on the sub as ``usage_wall_resets_at``), the (provider, model) is cached
in a process-level TIME-BOUNDED set and skipped on subsequent calls until the window
clears — instead of being reconstructed, re-run, and re-walled every call.

Hermetic: stub agents only, no network. ``fallback_agent._now`` is monkeypatched to
drive wall-clock deterministically. AGY_USAGE_WALL_BACKOFF pinned to 0.
"""
from __future__ import annotations

import asyncio

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.core.agents import fallback_agent
from agy_orchestrator.core.agents.fallback_agent import (
    _RATE_LIMITED_UNTIL,
    clear_rate_limited,
    is_rate_limited,
    make_fallback_agent,
    record_rate_limited,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_rate_limited()
    yield
    clear_rate_limited()


class _Base(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]


def _run(coro_factory, timeout=8.0):
    import threading
    box = {}

    def _worker():
        try:
            box["result"] = asyncio.run(coro_factory())
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise AssertionError("run_async did not terminate within the bound")
    if "error" in box:
        raise box["error"]
    return box.get("result")


# --- (1) record + is_rate_limited with a future window -> True --------------------
def test_record_then_rate_limited_in_future():
    record_rate_limited("Codex", "spark", 2000.0)
    assert is_rate_limited("Codex", "spark", now=1000.0) is True
    # a DIFFERENT model on the same provider is unaffected.
    assert is_rate_limited("Codex", "gpt-5.4", now=1000.0) is False
    # a falsy model is never rate-limited.
    assert is_rate_limited("Codex", None, now=1000.0) is False


# --- (2) past resets_at -> False AND entry auto-popped (re-enable) -----------------
def test_past_reset_auto_reenables_and_pops():
    record_rate_limited("Codex", "spark", 2000.0)
    assert ("Codex", "spark") in _RATE_LIMITED_UNTIL
    assert is_rate_limited("Codex", "spark", now=2500.0) is False
    # window elapsed -> the entry is removed so the model is re-enabled.
    assert ("Codex", "spark") not in _RATE_LIMITED_UNTIL


# --- (3) malformed resets_at -> never cached --------------------------------------
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"),
                                 float("-inf"), 0, 0.0, -1, "abc", object()])
def test_malformed_resets_at_not_cached(bad):
    record_rate_limited("Codex", "spark", bad)
    assert ("Codex", "spark") not in _RATE_LIMITED_UNTIL
    assert is_rate_limited("Codex", "spark", now=1.0) is False


def test_falsy_model_never_cached():
    record_rate_limited("Codex", None, 2000.0)
    record_rate_limited("Codex", "", 2000.0)
    assert _RATE_LIMITED_UNTIL == {}


# --- (4) CROSS-CALL: a walled (provider, model) with a future reset is SKIPPED -----
def test_cross_call_skips_cached_walled_provider(monkeypatch):
    monkeypatch.setenv("AGY_USAGE_WALL_BACKOFF", "0")
    monkeypatch.setattr(fallback_agent, "_now", lambda: 1000.0)
    seen_codex = []
    seen_agy = []

    class Codex(_Base):
        async def run_async(self, piped_input=None):
            seen_codex.append(self.model)
            # spark walled, and the wall reports a reset 5000s out (future).
            self.usage_wall_resets_at = 6000.0
            self.stderr = "usage limit reached (used_percent=100.0)"
            raise RuntimeError("usage wall")

    class Agy(_Base):
        async def run_async(self, piped_input=None):
            seen_agy.append(self.model)
            return "built-by-agy"

    Agent = make_fallback_agent(
        chain=[Codex, Agy], cycles=1,
        configs={Codex: {"model": "spark"}, Agy: {"model": "pro"}},
    )

    # Call 1: codex@spark walls (and caches resets_at), then agy serves.
    r1 = _run(lambda: Agent(prompt="p", model="spark").run_async())
    assert r1 == "built-by-agy"
    assert seen_codex == ["spark"], seen_codex
    assert ("Codex", "spark") in _RATE_LIMITED_UNTIL

    # Call 2: codex@spark is cache-walled -> its run_async is NOT invoked; straight to agy.
    r2 = _run(lambda: Agent(prompt="p", model="spark").run_async())
    assert r2 == "built-by-agy"
    assert seen_codex == ["spark"], "codex must NOT be re-probed while cache-walled"
    assert seen_agy == ["pro", "pro"]


# --- (5) ALL providers cache-walled -> still >=1 run_async invoked (no zero-attempt) -
def test_all_walled_still_attempts_at_least_one(monkeypatch):
    monkeypatch.setenv("AGY_USAGE_WALL_BACKOFF", "0")
    monkeypatch.setattr(fallback_agent, "_now", lambda: 1000.0)
    seen_codex = []
    seen_agy = []

    class Codex(_Base):
        async def run_async(self, piped_input=None):
            seen_codex.append(self.model)
            return "built-by-codex"  # succeeds if it's ever attempted

    class Agy(_Base):
        async def run_async(self, piped_input=None):
            seen_agy.append(self.model)
            return "built-by-agy"

    # Pre-populate the cache so BOTH providers look rate-limited this call.
    record_rate_limited("Codex", "spark", 9000.0)
    record_rate_limited("Agy", "pro", 9000.0)

    Agent = make_fallback_agent(
        chain=[Codex, Agy], cycles=1,
        configs={Codex: {"model": "spark"}, Agy: {"model": "pro"}},
    )
    result = _run(lambda: Agent(prompt="p", model="spark").run_async())
    # Nothing viable left AND nothing ran for the lead -> it is attempted anyway.
    assert result == "built-by-codex", result
    assert seen_codex == ["spark"], "the lead must be attempted, not skipped to zero"


# --- (6) wall with resets_at=None -> NOT cached -> re-probed next call -------------
def test_wall_without_resets_at_is_reprobed(monkeypatch):
    monkeypatch.setenv("AGY_USAGE_WALL_BACKOFF", "0")
    monkeypatch.setattr(fallback_agent, "_now", lambda: 1000.0)
    seen_codex = []

    class Codex(_Base):
        async def run_async(self, piped_input=None):
            seen_codex.append(self.model)
            # wall reports NO reset time -> must not be cached.
            self.usage_wall_resets_at = None
            self.stderr = "usage limit reached (used_percent=100.0)"
            raise RuntimeError("usage wall")

    class Agy(_Base):
        async def run_async(self, piped_input=None):
            return "built-by-agy"

    Agent = make_fallback_agent(
        chain=[Codex, Agy], cycles=1,
        configs={Codex: {"model": "spark"}, Agy: {"model": "pro"}},
    )

    r1 = _run(lambda: Agent(prompt="p", model="spark").run_async())
    assert r1 == "built-by-agy"
    assert ("Codex", "spark") not in _RATE_LIMITED_UNTIL

    # Call 2: codex@spark was NOT cached -> it is re-probed exactly as before.
    r2 = _run(lambda: Agent(prompt="p", model="spark").run_async())
    assert r2 == "built-by-agy"
    assert seen_codex == ["spark", "spark"], "no resets_at -> codex must be re-probed"


# --- (7) #78 interaction: a cached lead-model wall must NOT abandon the provider's
#         working sibling model on later calls (regression for the fuzz finding) ----
def test_cached_lead_wall_still_tries_working_sibling_model(monkeypatch):
    monkeypatch.setenv("AGY_USAGE_WALL_BACKOFF", "0")
    monkeypatch.setattr(fallback_agent, "_now", lambda: 1000.0)
    seen_codex = []
    agy_built = []

    class Codex(_Base):
        async def run_async(self, piped_input=None):
            seen_codex.append(self.model)
            if self.model == "spark":  # only spark is walled; gpt-5.5 has headroom
                self.usage_wall_resets_at = 9000.0  # future
                self.stderr = "usage limit reached (used_percent=100.0)"
                raise RuntimeError("usage wall")
            return f"built-by-codex@{self.model}"

    class Agy(_Base):
        async def run_async(self, piped_input=None):
            agy_built.append(self.model)
            return "built-by-agy"

    Agent = make_fallback_agent(
        chain=[Codex, Agy], cycles=1,
        configs={Codex: {"model": "spark"}, Agy: {"model": "pro"}},
        model_fallbacks={Codex: ["gpt-5.5", "gpt-5.4"]},
    )

    # Call 1: spark walls -> #78 reactively tries codex@gpt-5.5, which serves.
    r1 = _run(lambda: Agent(prompt="p", model="spark").run_async())
    assert r1 == "built-by-codex@gpt-5.5", r1
    assert seen_codex == ["spark", "gpt-5.5"]
    assert ("Codex", "spark") in _RATE_LIMITED_UNTIL  # spark cached, sibling not
    assert agy_built == []

    seen_codex.clear()
    # Call 2: spark is cache-walled, but #82's skip must NOT jump to agy — it must
    # PROACTIVELY try the working sibling gpt-5.5 (no re-probe of spark).
    r2 = _run(lambda: Agent(prompt="p", model="spark").run_async())
    assert r2 == "built-by-codex@gpt-5.5", r2
    assert "spark" not in seen_codex, "walled lead must not be re-probed"
    assert seen_codex == ["gpt-5.5"], "must stay on the provider's working sibling"
    assert agy_built == [], "must NOT switch providers while a sibling model works"

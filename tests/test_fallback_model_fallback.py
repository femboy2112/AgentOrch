"""Operator request / issue #78 — on a usage/quota wall, FallbackAgent must try the
provider's OTHER models before switching providers.

codex's spark window can be exhausted while gpt-5.5/gpt-5.4 still have headroom
(separate rate-limit windows). The chain ``codex -> agy`` with codex
model_fallbacks ``[gpt-5.5, gpt-5.4]`` must, on a spark wall, try codex@gpt-5.5
(then @gpt-5.4) BEFORE ever constructing agy.

Hermetic: stub agents that wall on named models and succeed on others; no network.
"""
from __future__ import annotations

import asyncio

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.core.agents.fallback_agent import (
    is_model_known_inaccessible,
    make_fallback_agent,
    reset_inaccessible_models,
)


@pytest.fixture(autouse=True)
def _clear_inaccessible_cache():
    reset_inaccessible_models()
    yield
    reset_inaccessible_models()


class _Base(AgentInstance):
    constructed: list = []

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


def test_usage_wall_tries_other_models_before_next_provider(monkeypatch):
    monkeypatch.setenv("AGY_USAGE_WALL_BACKOFF", "0")
    seen_codex_models = []
    agy_built = []

    class Codex(_Base):
        async def run_async(self, piped_input=None):
            seen_codex_models.append(self.model)
            if self.model in ("spark", "gpt-5.5"):  # both walled
                self.stderr = "usage limit reached (used_percent=100.0)"
                raise RuntimeError("usage wall")
            return f"built-by-codex@{self.model}"  # gpt-5.4 succeeds

    class Agy(_Base):
        async def run_async(self, piped_input=None):
            agy_built.append(self.model)
            return "built-by-agy"

    Agent = make_fallback_agent(
        chain=[Codex, Agy],
        cycles=1,
        configs={Codex: {"model": "spark"}, Agy: {"model": "pro"}},
        model_fallbacks={Codex: ["gpt-5.5", "gpt-5.4"]},
    )
    result = _run(lambda: Agent(prompt="p", model="spark").run_async())

    assert result == "built-by-codex@gpt-5.4", result
    # spark walled -> gpt-5.5 walled -> gpt-5.4 succeeded; agy NEVER ran.
    assert seen_codex_models == ["spark", "gpt-5.5", "gpt-5.4"], seen_codex_models
    assert agy_built == [], "agy must not run while codex has untried models"


def test_all_codex_models_walled_then_advances_to_next_provider(monkeypatch):
    monkeypatch.setenv("AGY_USAGE_WALL_BACKOFF", "0")
    seen_codex_models = []

    class Codex(_Base):
        async def run_async(self, piped_input=None):
            seen_codex_models.append(self.model)
            self.stderr = "usage limit reached (used_percent=100.0)"
            raise RuntimeError("usage wall")

    class Agy(_Base):
        async def run_async(self, piped_input=None):
            return "built-by-agy"

    Agent = make_fallback_agent(
        chain=[Codex, Agy],
        cycles=1,
        configs={Codex: {"model": "spark"}, Agy: {"model": "pro"}},
        model_fallbacks={Codex: ["gpt-5.5", "gpt-5.4"]},
    )
    result = _run(lambda: Agent(prompt="p", model="spark").run_async())

    assert result == "built-by-agy"
    # Each alternate tried exactly once, then provider switch.
    assert seen_codex_models == ["spark", "gpt-5.5", "gpt-5.4"], seen_codex_models


def test_model_fallback_only_on_usage_wall_not_generic_error(monkeypatch):
    """A generic (non-usage) failure must NOT trigger intra-provider model swap —
    it advances to the next provider as before."""
    monkeypatch.setenv("AGY_USAGE_WALL_BACKOFF", "0")
    seen_codex_models = []

    class Codex(_Base):
        async def run_async(self, piped_input=None):
            seen_codex_models.append(self.model)
            self.stderr = "some unrelated build error"
            raise RuntimeError("boom")

    class Agy(_Base):
        async def run_async(self, piped_input=None):
            return "built-by-agy"

    Agent = make_fallback_agent(
        chain=[Codex, Agy],
        cycles=1,
        configs={Codex: {"model": "spark"}, Agy: {"model": "pro"}},
        model_fallbacks={Codex: ["gpt-5.5", "gpt-5.4"]},
    )
    result = _run(lambda: Agent(prompt="p", model="spark").run_async())
    assert result == "built-by-agy"
    assert seen_codex_models == ["spark"], "generic error must not swap codex models"


def test_no_model_fallbacks_is_unchanged(monkeypatch):
    """Without a model_fallbacks map the behavior is exactly the prior one:
    a usage wall advances straight to the next provider."""
    monkeypatch.setenv("AGY_USAGE_WALL_BACKOFF", "0")
    seen = []

    class Codex(_Base):
        async def run_async(self, piped_input=None):
            seen.append(("codex", self.model))
            self.stderr = "usage limit reached"
            raise RuntimeError("usage wall")

    class Agy(_Base):
        async def run_async(self, piped_input=None):
            seen.append(("agy", self.model))
            return "ok"

    Agent = make_fallback_agent(
        chain=[Codex, Agy], cycles=1,
        configs={Codex: {"model": "spark"}, Agy: {"model": "pro"}},
    )
    result = _run(lambda: Agent(prompt="p", model="spark").run_async())
    assert result == "ok"
    assert seen == [("codex", "spark"), ("agy", "pro")], seen


def test_model_unavailable_prunes_and_tries_next_then_caches(monkeypatch):
    """A model that's inaccessible to the account is pruned dynamically: the
    fallback tries the next codex model, and the bad model is cached process-wide
    so a SECOND run never offers it again."""
    monkeypatch.setenv("AGY_USAGE_WALL_BACKOFF", "0")
    seen = []

    class Codex(_Base):
        async def run_async(self, piped_input=None):
            seen.append(self.model)
            if self.model in ("spark", "gpt-5.2"):
                # spark walls; gpt-5.2 is inaccessible to this account.
                if self.model == "spark":
                    self.stderr = "usage limit reached (used_percent=100.0)"
                else:
                    self.stderr = ("The 'gpt-5.2' model is not supported when using "
                                   "Codex with a ChatGPT account.")
                raise RuntimeError("fail")
            return f"ok@{self.model}"

    class Agy(_Base):
        async def run_async(self, piped_input=None):
            return "built-by-agy"

    Agent = make_fallback_agent(
        chain=[Codex, Agy], cycles=1,
        configs={Codex: {"model": "spark"}, Agy: {"model": "pro"}},
        # gpt-5.2 first so it's reached, then gpt-5.4 succeeds.
        model_fallbacks={Codex: ["gpt-5.2", "gpt-5.4"]},
    )
    result = _run(lambda: Agent(prompt="p", model="spark").run_async())
    assert result == "ok@gpt-5.4", result
    assert seen == ["spark", "gpt-5.2", "gpt-5.4"], seen
    assert is_model_known_inaccessible("Codex", "gpt-5.2")

    # Second run: gpt-5.2 is cached-inaccessible, so it's skipped entirely.
    seen.clear()
    result2 = _run(lambda: Agent(prompt="p", model="spark").run_async())
    assert result2 == "ok@gpt-5.4", result2
    assert seen == ["spark", "gpt-5.4"], seen


def test_roles_wires_codex_model_fallbacks(monkeypatch):
    """harness.roles builds a codex model-fallback map from codex's discovered
    roster (candidate_models(): cache best-first, static fallback), honouring the
    env override/disable. Runtime pruning still drops inaccessible models."""
    from harness.roles import _codex_model_fallbacks, _class_for
    from agy_orchestrator.core.agents.codex_agent import CodexAgent

    codex_cls = _class_for("codex")
    configs = {codex_cls: {"model": "standard"}}

    monkeypatch.delenv("AGY_CODEX_MODEL_FALLBACKS", raising=False)
    fb = _codex_model_fallbacks(configs)
    assert codex_cls in fb
    # Universe is the discovered roster minus the walled lead ('standard'->spark).
    expected = [m for m in CodexAgent.candidate_models()
                if m not in ("standard", "gpt-5.3-codex-spark")]
    assert fb[codex_cls] == expected
    assert "gpt-5.3-codex-spark" not in fb[codex_cls]
    # Codex's own priority order puts the strongest model first.
    assert fb[codex_cls][0] == "gpt-5.5"

    monkeypatch.setenv("AGY_CODEX_MODEL_FALLBACKS", "gpt-5.4, gpt-5.2")
    assert _codex_model_fallbacks(configs)[codex_cls] == ["gpt-5.4", "gpt-5.2"]

    # An explicit operator override is AUTHORITATIVE — names NOT in the discovered
    # roster pass through unfiltered (model names are advisory per #46; a genuinely
    # bad one is pruned at runtime when it 400s). Only the env path trusts the
    # operator like this; the default (roster-sourced) path can't contain typos.
    monkeypatch.setenv("AGY_CODEX_MODEL_FALLBACKS", "gpt-5.4, gpt-9.9-bogus")
    assert _codex_model_fallbacks(configs)[codex_cls] == ["gpt-5.4", "gpt-9.9-bogus"]

    monkeypatch.setenv("AGY_CODEX_MODEL_FALLBACKS", "")
    assert _codex_model_fallbacks(configs) == {}

    # No codex in the chain -> empty map.
    monkeypatch.delenv("AGY_CODEX_MODEL_FALLBACKS", raising=False)
    assert _codex_model_fallbacks({}) == {}

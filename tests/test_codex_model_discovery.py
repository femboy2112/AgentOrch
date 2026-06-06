"""Codex models come from codex's OWN server-fetched roster, not a hardcoded list.

codex caches its account's selectable models (with a ``priority`` rank and per-model
supported reasoning efforts) to ``~/.codex/models_cache.json`` — the same roster its
picker shows. Reading it is true dynamic detection: retired/ungranted models (gpt-5.2,
bare gpt-5.3-codex) simply aren't present, and the order is codex's own. These tests
pin the parse, the static fallback, and that a SINGLE-provider chain still rolls models.
"""
import json

import pytest

from agy_orchestrator.core.agents.agy_agent import AgyAgent
from agy_orchestrator.core.agents.codex_agent import CodexAgent, codex_models_from_cache
from agy_orchestrator.core.agents.grok_agent import GrokAgent
from harness import roles


def _write_cache(tmp_path, models):
    p = tmp_path / "models_cache.json"
    p.write_text(json.dumps({"fetched_at": "now", "models": models}), encoding="utf-8")
    return p


@pytest.fixture
def real_cache_like():
    # Mirrors the real ~/.codex/models_cache.json shape (subset of fields used).
    return [
        {"slug": "gpt-5.4", "visibility": "list", "priority": 16},
        {"slug": "gpt-5.5", "visibility": "list", "priority": 9},
        {"slug": "codex-auto-review", "visibility": "hide", "priority": 43},
        {"slug": "gpt-5.4-mini", "visibility": "list", "priority": 23},
        {"slug": "gpt-5.3-codex-spark", "visibility": "list", "priority": 26},
    ]


def test_cache_parsed_in_priority_order_excluding_hidden(tmp_path, monkeypatch, real_cache_like):
    monkeypatch.setenv("AGY_CODEX_MODELS_CACHE", str(_write_cache(tmp_path, real_cache_like)))
    # Sorted by codex's priority; the hidden helper is excluded.
    assert codex_models_from_cache() == [
        "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex-spark",
    ]
    assert "codex-auto-review" not in codex_models_from_cache()


def test_cache_absent_yields_empty_then_static_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_CODEX_MODELS_CACHE", str(tmp_path / "nope.json"))
    assert codex_models_from_cache() == []
    # candidate_models falls back to the static roster (never empty).
    assert CodexAgent.candidate_models() == list(CodexAgent.AVAILABLE_MODELS)


def test_cache_corrupt_is_swallowed(tmp_path, monkeypatch):
    p = tmp_path / "models_cache.json"
    p.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("AGY_CODEX_MODELS_CACHE", str(p))
    assert codex_models_from_cache() == []  # never raises


def test_cache_model_without_priority_sinks_to_end(tmp_path, monkeypatch):
    models = [
        {"slug": "gpt-5.5", "visibility": "list", "priority": 9},
        {"slug": "weird-no-priority", "visibility": "list"},
        {"slug": "gpt-5.4", "visibility": "list", "priority": 16},
    ]
    monkeypatch.setenv("AGY_CODEX_MODELS_CACHE", str(_write_cache(tmp_path, models)))
    assert codex_models_from_cache() == ["gpt-5.5", "gpt-5.4", "weird-no-priority"]


def test_static_fallback_has_no_ungranted_models():
    # gpt-5.2 and bare gpt-5.3-codex are not granted on the account and must not
    # sit in the last-resort static list either.
    assert "gpt-5.2" not in CodexAgent.AVAILABLE_MODELS
    assert "gpt-5.3-codex" not in CodexAgent.AVAILABLE_MODELS
    assert CodexAgent.AVAILABLE_MODELS[0] == "gpt-5.5"


def test_candidate_models_uses_cache_when_present(tmp_path, monkeypatch, real_cache_like):
    monkeypatch.setenv("AGY_CODEX_MODELS_CACHE", str(_write_cache(tmp_path, real_cache_like)))
    assert CodexAgent.candidate_models()[0] == "gpt-5.5"


# --------------------------------------------------------------------------- #
# Single-provider chains must still roll the provider's models (issue #78).
# --------------------------------------------------------------------------- #
def test_single_codex_chain_wraps_for_model_rolling():
    g = roles.build_role_agent(["codex"], prompt="x", fallback=True)
    assert "FallbackAgent" in type(g).__name__
    fb = getattr(type(g), "_model_fallbacks", {})
    codex_cls = roles._class_for("codex")
    assert fb.get(codex_cls), "codex single-token chain must carry a model-fallback list"
    assert fb[codex_cls][0] == "gpt-5.5"  # rolls to the best model first


def test_single_codex_with_override_excludes_current_model():
    g = roles.build_role_agent(["codex"], prompt="x", fallback=True,
                               overrides={"codex": {"model": "gpt-5.5"}})
    fb = getattr(type(g), "_model_fallbacks", {})[roles._class_for("codex")]
    assert "gpt-5.5" not in fb  # never re-pick the walled current model


def test_single_non_codex_chains_stay_direct():
    # agy/grok have no model-fallback list -> byte-identical direct path (no wrap).
    assert type(roles.build_role_agent(["agy"], prompt="x", fallback=True)) is AgyAgent
    assert type(roles.build_role_agent(["grok"], prompt="x", fallback=True)) is GrokAgent


def test_fallback_off_stays_direct_even_for_codex():
    assert type(roles.build_role_agent(["codex"], prompt="x", fallback=False)) is CodexAgent


def test_master_single_codex_chain_wraps_for_rolling():
    cls, _model, _effort = roles.build_master_agent_class(["codex"], fallback=True)
    assert "FallbackAgent" in cls.__name__
    assert getattr(cls, "_model_fallbacks", {}).get(roles._class_for("codex"))


def test_master_single_agy_chain_stays_direct():
    cls, _m, _e = roles.build_master_agent_class(["agy"], fallback=True)
    # No model-fallback list for agy -> bare class (or hooked), never a FallbackAgent.
    assert "FallbackAgent" not in cls.__name__

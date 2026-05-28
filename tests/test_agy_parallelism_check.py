"""Agy parallelism guard for vote/tot modes.

Warn-only helper: detect when high-parallel dispatch would place agy in
multiple concurrent candidate slots, which serializes on agy's global
settings.json lock and erases the expected speedup.
"""
from __future__ import annotations

from harness import roles


def test_default_chain_single_agy_slot_silent(monkeypatch):
    monkeypatch.delenv("AGY_PARALLELISM_CHECK", raising=False)
    msg = roles.check_agy_parallelism_warning("vote", ["codex", "agy", "grok"], 3)
    assert msg is None


def test_homogeneous_agy_chain_warns(monkeypatch):
    monkeypatch.delenv("AGY_PARALLELISM_CHECK", raising=False)
    msg = roles.check_agy_parallelism_warning("vote", ["agy"], 3)
    assert msg is not None
    assert "agy-parallelism" in msg


def test_partial_agy_chain_warns_at_two(monkeypatch):
    monkeypatch.delenv("AGY_PARALLELISM_CHECK", raising=False)
    msg = roles.check_agy_parallelism_warning("vote", ["agy", "codex", "agy"], 4)
    assert msg is not None
    assert "agy-parallelism" in msg


def test_non_vote_mode_silent(monkeypatch):
    monkeypatch.delenv("AGY_PARALLELISM_CHECK", raising=False)
    msg = roles.check_agy_parallelism_warning("adversarial", ["agy"], 3)
    assert msg is None


def test_branches_one_silent(monkeypatch):
    monkeypatch.delenv("AGY_PARALLELISM_CHECK", raising=False)
    msg = roles.check_agy_parallelism_warning("vote", ["agy", "agy"], 1)
    assert msg is None


def test_env_var_disables_check(monkeypatch):
    monkeypatch.setenv("AGY_PARALLELISM_CHECK", "off")
    msg = roles.check_agy_parallelism_warning("vote", ["agy"], 3)
    assert msg is None

"""Cross-family verifier guard — emit a warning when generator and critic
chains lead with the same provider family.

Paper anchor: "When Does Verification Pay Off?" (arxiv 2512.02304) —
self-verification substantially underperforms cross-family critique, so we
nudge operators away from accidentally configuring it. Warning-only; the
operator can override via AGY_CRITIC_FAMILY_CHECK=off.
"""
from __future__ import annotations

from harness import roles


def test_cross_family_default_no_warning():
    """The shipped defaults (codex generator, agy critic) are cross-family.
    No warning should fire — this regression-guards the happy path."""
    msg = roles.check_chains_cross_family(roles.GENERATOR_CHAIN, roles.CRITIC_CHAIN)
    assert msg is None


def test_same_family_emits_warning():
    """codex generator + codex critic = self-verification. Warn."""
    msg = roles.check_chains_cross_family(["codex"], ["codex"])
    assert msg is not None
    assert "openai" in msg
    assert "self-verification" in msg
    # Operator needs an escape hatch — the message must tell them how.
    assert "AGY_CRITIC_FAMILY_CHECK=off" in msg


def test_all_cross_family_pairs_silent():
    """Spot-check the diagonal: every distinct (gen, crit) pair across our
    four workers should be cross-family and silent."""
    workers = list(roles.WORKER_FAMILY.keys())
    for gen in workers:
        for crit in workers:
            msg = roles.check_chains_cross_family([gen], [crit])
            same_family = roles.WORKER_FAMILY[gen] == roles.WORKER_FAMILY[crit]
            if same_family:
                assert msg is not None, f"expected warning for {gen}+{crit}"
            else:
                assert msg is None, f"unexpected warning for {gen}+{crit}: {msg}"


def test_chain_lead_is_what_counts():
    """Only the LEAD provider on each chain matters — fallback tail can
    overlap without triggering. The first provider is the one the run
    actually starts with; the rest are usage-wall fallbacks that may never
    fire."""
    # codex (openai) → agy (google): cross-family even though both chains
    # include codex somewhere.
    msg = roles.check_chains_cross_family(["codex", "agy"], ["agy", "codex"])
    assert msg is None


def test_unknown_worker_falls_open():
    """If we don't know the worker's family, stay silent rather than
    false-positive. Better to miss a warning than to nag operators about
    workers we haven't mapped."""
    msg = roles.check_chains_cross_family(["mystery-worker"], ["codex"])
    assert msg is None
    msg = roles.check_chains_cross_family(["codex"], ["another-mystery"])
    assert msg is None


def test_env_var_disables_check(monkeypatch):
    """AGY_CRITIC_FAMILY_CHECK=off must short-circuit cleanly even for the
    case that would normally warn — operator's explicit opt-out."""
    monkeypatch.setenv("AGY_CRITIC_FAMILY_CHECK", "off")
    msg = roles.check_chains_cross_family(["codex"], ["codex"])
    assert msg is None


def test_empty_chains_silent():
    """Edge: empty chains should not crash, should not warn."""
    assert roles.check_chains_cross_family([], ["codex"]) is None
    assert roles.check_chains_cross_family(["codex"], []) is None
    assert roles.check_chains_cross_family([], []) is None

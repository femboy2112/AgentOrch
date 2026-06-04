"""Fuzz #77: watchdog-aborted run must not inherit verified/high confidence.

build_ledger(..., run_aborted=True) downgrades any in-loop verified/approved
signal to an honest terminal label, because a run killed mid-build by the
run-level watchdog only verified the steps that actually ran — not the whole
build. This probe fuzzes the abort x signal cross-product and pins the
back-compat contract (omitting run_aborted is byte-identical to False).

Each test is a RED contract for the DESIRED behavior. A variant the code does
NOT satisfy is marked xfail(strict=True) (verdict=bug); a satisfied variant is
left plain (verdict=resilient). Source is NOT modified.
"""
from __future__ import annotations

import pytest

from agy_orchestrator.execution.ledger import build_ledger

pytestmark = pytest.mark.not_slow


class _WF:
    """Minimal workflow stub exposing the signals build_ledger reads."""

    def __init__(self, *, verified=False, approved=False, stalled=False,
                 iterations_used=None):
        self.verified = verified
        self.approved = approved
        self.stalled = stalled
        self.iterations_used = iterations_used


# ---------------------------------------------------------------------------
# Variant 1: aborted run with workflow.verified=True -> NOT verified/high + note
# ---------------------------------------------------------------------------
def test_aborted_verified_not_high_with_honest_note():
    wf = _WF(verified=True, iterations_used=9)
    row = build_ledger(wf, mode="master", had_verifier=True,
                       produced_output=True, run_aborted=True)
    assert row["confidence"] not in ("verified", "approved")
    note = row["note"].lower()
    assert "high confidence" not in note
    assert any(w in note for w in ("abort", "watchdog", "stall"))
    # raw signal preserved faithfully even though confidence is downgraded
    assert row["verified"] is True


# ---------------------------------------------------------------------------
# Variant 2: NORMAL verified completed run (run_aborted False) -> verified/high
# ---------------------------------------------------------------------------
def test_normal_verified_completed_stays_high():
    wf = _WF(verified=True, iterations_used=2)
    row = build_ledger(wf, mode="master", had_verifier=True,
                       produced_output=True, run_aborted=False)
    assert row["confidence"] == "verified"
    assert "high confidence" in row["note"].lower()


# ---------------------------------------------------------------------------
# Variant 3: aborted run ALREADY unverified -> stays unverified-ish, no double state
# ---------------------------------------------------------------------------
def test_aborted_already_unverified_no_double_state():
    # produced output, nothing verified/approved -> normally "unverified".
    wf = _WF(verified=False, approved=False)
    row = build_ledger(wf, mode="adversarial", had_verifier=True,
                       produced_output=True, run_aborted=True)
    # must not magically gain confidence by being aborted
    assert row["confidence"] not in ("verified", "approved")
    # terminal honest label, single coherent state
    assert row["confidence"] in ("stalled", "unverified", "failed")
    assert row["verified"] is False
    assert row["critic_approved"] is False


# ---------------------------------------------------------------------------
# Variant 4: aborted APPROVED-but-not-verified run -> not medium/high
# ---------------------------------------------------------------------------
def test_aborted_approved_not_medium_or_high():
    wf = _WF(verified=False, approved=True)
    row = build_ledger(wf, mode="adversarial", had_verifier=False,
                       produced_output=True, run_aborted=True)
    assert row["confidence"] not in ("verified", "approved")
    note = row["note"].lower()
    assert "medium confidence" not in note
    assert "high confidence" not in note


# ---------------------------------------------------------------------------
# Variant 5: meta.json confidence agrees with the printed summary note.
# dispatch.py uses quality["note"] for BOTH the logged summary and the meta row,
# and the note is keyed off the SAME confidence label -> they cannot disagree.
# ---------------------------------------------------------------------------
def test_meta_confidence_agrees_with_summary_note():
    from agy_orchestrator.execution.ledger import (
        _NOTE, _NOTE_UNVERIFIED_WITH_VERIFIER,
    )
    cases = [
        (_WF(verified=True), True, True, False),
        (_WF(verified=True), True, True, True),    # aborted -> stalled
        (_WF(approved=True), False, True, True),
        (_WF(), True, True, False),                # unverified w/ verifier
        (_WF(), True, False, False),               # failed (no output)
    ]
    for wf, had_v, produced, aborted in cases:
        row = build_ledger(wf, mode="master", had_verifier=had_v,
                           produced_output=produced, run_aborted=aborted)
        conf = row["confidence"]
        if conf == "unverified" and had_v:
            expected = _NOTE_UNVERIFIED_WITH_VERIFIER
        else:
            expected = _NOTE[conf]
        # The single source of truth: the note dispatch prints == note in meta,
        # and it is exactly the note for the confidence label in the same row.
        assert row["note"] == expected


# ---------------------------------------------------------------------------
# Variant 6: back-compat -> every existing build_ledger call (no run_aborted)
# behaves identically to passing run_aborted=False.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("wf,had_v,produced", [
    (_WF(verified=True, iterations_used=3), True, True),
    (_WF(approved=True), False, True),
    (_WF(), True, True),
    (_WF(), False, True),
    (_WF(), True, False),
])
def test_backcompat_default_byte_identical(wf, had_v, produced):
    omitted = build_ledger(wf, mode="direct", had_verifier=had_v,
                           produced_output=produced)
    explicit_false = build_ledger(wf, mode="direct", had_verifier=had_v,
                                  produced_output=produced, run_aborted=False)
    assert omitted == explicit_false

"""#77: an ABORTED (run-watchdog/stall) run must not report verified/high confidence.

build_ledger derives confidence purely from workflow.verified, ignoring that the
RUN was aborted mid-build by the run-level watchdog. A run wedged at step 9/19 and
then watchdog-aborted at 1800s reported confidence="verified" / "high confidence"
in meta.json + the final summary, contradicting its own stalled/FAILED outcome.

Contract:
  - build_ledger(..., run_aborted=True) must NOT read "verified"/"approved";
    it downgrades to an honest terminal label ("stalled") with an abort note.
  - run_aborted defaults to False -> every existing call is byte-identical.
  - A normal completed verified run (run_aborted=False) still reports
    "verified" / "high confidence" EXACTLY as before.
"""
from __future__ import annotations

from agy_orchestrator.execution.ledger import build_ledger


class _WF:
    """A minimal workflow stub exposing the signals build_ledger reads."""

    def __init__(self, *, verified=False, approved=False, stalled=False,
                 iterations_used=None):
        self.verified = verified
        self.approved = approved
        self.stalled = stalled
        self.iterations_used = iterations_used


def test_aborted_run_does_not_claim_verified():
    """A verified-looking workflow that was run-aborted must not read verified."""
    wf = _WF(verified=True, iterations_used=4)
    row = build_ledger(wf, mode="master", had_verifier=True,
                       produced_output=True, run_aborted=True)
    assert row["confidence"] not in ("verified", "approved")
    # honest terminal label
    assert row["confidence"] in ("stalled", "unverified")
    # note must own the abort, not claim a passing verifier
    note = row["note"].lower()
    assert "high confidence" not in note
    assert "abort" in note or "watchdog" in note or "stall" in note
    # the raw verified signal is still recorded faithfully
    assert row["verified"] is True


def test_aborted_run_with_approval_also_downgrades():
    wf = _WF(approved=True)
    row = build_ledger(wf, mode="adversarial", had_verifier=False,
                       produced_output=True, run_aborted=True)
    assert row["confidence"] not in ("verified", "approved")


def test_normal_verified_run_unchanged():
    """run_aborted defaults to False -> verified run still reads verified/high."""
    wf = _WF(verified=True, iterations_used=2)
    baseline = build_ledger(wf, mode="master", had_verifier=True,
                            produced_output=True)
    explicit = build_ledger(wf, mode="master", had_verifier=True,
                            produced_output=True, run_aborted=False)
    assert baseline == explicit
    assert baseline["confidence"] == "verified"
    assert baseline["note"] == "A programmatic verifier passed — high confidence."


def test_default_call_byte_identical():
    """Omitting run_aborted entirely keeps the row identical to passing False."""
    wf = _WF(verified=True)
    assert (build_ledger(wf, mode="direct", had_verifier=False, produced_output=True)
            == build_ledger(wf, mode="direct", had_verifier=False,
                            produced_output=True, run_aborted=False))


def test_aborted_no_output_still_failed():
    """No output + aborted stays failed (abort note doesn't manufacture confidence)."""
    wf = _WF()
    row = build_ledger(wf, mode="master", had_verifier=True,
                       produced_output=False, run_aborted=True)
    assert row["confidence"] not in ("verified", "approved")

"""Ledger note honesty when a verifier WAS configured (#45).

The "unverified" note hardcoded "no verifier", which is factually wrong when a
verifier ran but never confirmed the final output green. The note for the
unverified branch is now conditional on had_verifier; other branches are fixed.
"""
from agy_orchestrator.execution.ledger import build_ledger


class _WF:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_unverified_note_differs_by_had_verifier():
    wf = _WF(verified=False, approved=False, stalled=False, iterations_used=3)
    with_verifier = build_ledger(
        wf, mode="master", had_verifier=True, produced_output=True
    )
    without_verifier = build_ledger(
        wf, mode="master", had_verifier=False, produced_output=True
    )

    assert with_verifier["confidence"] == "unverified"
    assert without_verifier["confidence"] == "unverified"

    # The two notes must differ.
    assert with_verifier["note"] != without_verifier["note"]

    # The had_verifier=True note must NOT contain the false phrase.
    assert "no verifier" not in with_verifier["note"].lower()
    assert "configured" in with_verifier["note"].lower()

    # The had_verifier=False note keeps the honest "no verifier" wording.
    assert "no verifier" in without_verifier["note"].lower()


def test_verified_note_unchanged_regardless_of_had_verifier():
    wf = _WF(verified=True, approved=True, stalled=False, iterations_used=1)
    led = build_ledger(wf, mode="master", had_verifier=True, produced_output=True)
    assert led["confidence"] == "verified"
    assert led["note"] == "A programmatic verifier passed — high confidence."

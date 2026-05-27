"""Tests for the quality-cost ledger (task #9)."""
from agy_orchestrator.execution.ledger import build_ledger


class _WF:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_verified_is_highest_confidence():
    wf = _WF(verified=True, approved=True, stalled=False, iterations_used=2)
    led = build_ledger(wf, mode="feedback", had_verifier=True, produced_output=True)
    assert led["confidence"] == "verified"
    assert led["iterations_used"] == 2
    assert led["had_verifier"] is True


def test_critic_approved_without_verifier_is_medium():
    wf = _WF(verified=False, approved=True, stalled=False, iterations_used=1)
    led = build_ledger(wf, mode="adversarial", had_verifier=False, produced_output=True)
    assert led["confidence"] == "approved"


def test_stalled_no_approval_is_unverified():
    wf = _WF(verified=False, approved=False, stalled=True, iterations_used=2)
    led = build_ledger(wf, mode="adversarial", had_verifier=False, produced_output=True)
    assert led["confidence"] == "unverified"
    assert led["stalled"] is True


def test_no_output_is_failed():
    led = build_ledger(None, mode="direct", had_verifier=False, produced_output=False)
    assert led["confidence"] == "failed"


def test_direct_with_output_is_unverified():
    # direct mode has no workflow object; output present but unconfirmed.
    led = build_ledger(None, mode="direct", had_verifier=False, produced_output=True)
    assert led["confidence"] == "unverified"
    assert led["iterations_used"] is None

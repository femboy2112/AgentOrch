"""Quality-cost ledger — record per-run confidence signals (task #9).

The operator wants to "account for quality reductions in orchestration runs."
This derives a small, honest ledger from a workflow's
exposed signals (did a STRONG programmatic verifier pass? did the LLM critic
approve? did the loop stall or hit its cap? how many repair rounds?) and a coarse
``confidence`` label. It is descriptive, not magic: the strongest signal is a
passing test verifier; an LLM-critic approval is weaker; a stalled/maxed loop
with no approval is the weakest.

Confidence ladder (high -> low):
  verified   — a programmatic verifier (tests/lint/build) passed. Ground truth.
  approved   — no verifier, but the LLM critic explicitly approved.
  unverified — produced output, but nothing confirmed it (direct mode, or a loop
               that hit its cap / stalled without approval). Treat as suspect.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def build_ledger(workflow: Any, *, mode: str, had_verifier: bool,
                 produced_output: bool) -> Dict[str, Optional[object]]:
    """Read whatever signals a workflow exposes and derive a confidence label.

    Unknown workflows (master/direct) degrade gracefully: missing attributes just
    stay None and confidence falls back to output-presence."""
    verified = bool(getattr(workflow, "verified", False))
    approved = bool(getattr(workflow, "approved", False))
    stalled = bool(getattr(workflow, "stalled", False))
    iterations = getattr(workflow, "iterations_used", None)

    if verified:
        confidence = "verified"
    elif approved:
        confidence = "approved"
    elif produced_output:
        confidence = "unverified"
    else:
        confidence = "failed"

    return {
        "confidence": confidence,
        "verified": verified,
        "critic_approved": approved,
        "stalled": stalled,
        "iterations_used": iterations,
        "had_verifier": had_verifier,
        # A short, operator-facing note on how much to trust this run.
        "note": _NOTE[confidence],
    }


_NOTE = {
    "verified": "A programmatic verifier passed — high confidence.",
    "approved": "No verifier; the LLM critic approved — medium confidence, "
                "add a --test-cmd to harden.",
    "unverified": "Nothing confirmed this output (no verifier, loop stalled or "
                  "hit its cap). Treat as suspect; review or re-run with --test-cmd.",
    "failed": "The run produced no output.",
}

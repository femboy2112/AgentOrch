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
                 produced_output: bool,
                 telemetry: Optional[Dict[str, Any]] = None,
                 run_aborted: bool = False,
                 ) -> Dict[str, Optional[object]]:
    """Read whatever signals a workflow exposes and derive a confidence label.

    Unknown workflows (master/direct) degrade gracefully: missing attributes just
    stay None and confidence falls back to output-presence.

    ``telemetry`` is an optional dict of soft signals captured during execution
    (``wall_ms``, ``out_bytes``, ``watchdog_reason``, ``worker``, ``model``,
    ``effort``). These get written through to the ledger row verbatim so the
    next calibration cycle can re-fit per-config baselines (see
    core/calibration.py). None values are dropped from the output to keep ledger
    rows small for workflows that don't surface telemetry.

    ``run_aborted`` (#77): the RUN was killed mid-build by the run-level watchdog
    (a stall/abort). Any in-loop ``verified``/``approved`` signal then covers only
    the steps that ran, NOT the whole build, so the confidence MUST NOT read
    "verified"/"approved" — it is downgraded to the honest terminal label
    "stalled". Defaults False, so every existing call is byte-identical.
    """
    verified = bool(getattr(workflow, "verified", False))
    approved = bool(getattr(workflow, "approved", False))
    stalled = bool(getattr(workflow, "stalled", False))
    iterations = getattr(workflow, "iterations_used", None)

    if run_aborted:
        # #77: a watchdog-aborted run is terminal-stalled regardless of any
        # per-step verified/approved signal (which only covers the steps that
        # ran). "failed" still wins when nothing was produced.
        confidence = "stalled" if produced_output else "failed"
    elif verified:
        confidence = "verified"
    elif approved:
        confidence = "approved"
    elif produced_output:
        confidence = "unverified"
    else:
        confidence = "failed"

    # The "unverified" note is conditional on had_verifier (#45): the hardcoded
    # "no verifier" wording is factually wrong when a verifier WAS configured but
    # the final output simply wasn't confirmed green. Other branches are fixed.
    if confidence == "unverified" and had_verifier:
        note = _NOTE_UNVERIFIED_WITH_VERIFIER
    else:
        note = _NOTE[confidence]

    row: Dict[str, Optional[object]] = {
        "confidence": confidence,
        "verified": verified,
        "critic_approved": approved,
        "stalled": stalled,
        "iterations_used": iterations,
        "had_verifier": had_verifier,
        # A short, operator-facing note on how much to trust this run.
        "note": note,
    }
    if telemetry:
        for k in ("wall_ms", "out_bytes", "watchdog_reason",
                  "worker", "model", "effort",
                  "baseline_ok", "baseline_error_hash", "baseline_duration_ms",
                  "verifier_delta"):
            v = telemetry.get(k)
            if v is not None:
                row[k] = v
    return row


_NOTE = {
    "verified": "A programmatic verifier passed — high confidence.",
    "approved": "No verifier; the LLM critic approved — medium confidence, "
                "add a --test-cmd to harden.",
    "unverified": "Nothing confirmed this output (no verifier, loop stalled or "
                  "hit its cap). Treat as suspect; review or re-run with --test-cmd.",
    "stalled": "The run was ABORTED (watchdog / run-level stall) before completing "
               "— any in-loop verification covers only the steps that ran, not the "
               "whole build. Treat as failed; review or re-run.",
    "failed": "The run produced no output.",
}

# Honest "unverified" wording when a verifier WAS configured (#45): we can't
# claim "no verifier" — one ran but never confirmed the final output green.
_NOTE_UNVERIFIED_WITH_VERIFIER = (
    "A verifier was configured but the final output was not confirmed green "
    "(the loop hit its cap or the last attempt failed). Treat as suspect; "
    "review the run."
)

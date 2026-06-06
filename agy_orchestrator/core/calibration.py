"""Per-config latency/token baselines for the streaming watchdog.

The streaming watchdog needs to decide "is this run a runaway?" mid-flight. A flat
threshold (e.g. "kill at 50KB output") is wrong across configs — a brutal-tier
solution that legitimately wants 5KB looks indistinguishable from a `haiku:low`
ramble at 5KB but very different at 25KB. The honest answer is a PER-CONFIG
baseline: what does p95 output look like when this exact (worker, model, effort)
SUCCEEDS, and trip the watchdog only when the current run exceeds it by a safety
multiplier.

This module reads ``calibrate.jsonl`` (or any JSONL with the same row shape
emitted by scripts/calibrate.py / scripts/token_efficiency.py) and exposes a
``CalibrationTable`` with ``budget_for(worker, model, effort)`` -> (max_bytes,
stall_seconds). When the table has no data for a config we return CONSERVATIVE
defaults that won't false-positive on any successful run we've measured this
session — the watchdog stays a safety net, never a default-kill.

Schema expected per row (only fields used here):
  worker, model, effort, ok (bool), out_tokens (int|None), wall_ms (float|None)

Rows without out_tokens (grok/agy) are recorded for wall-time only — the
watchdog falls back to a wall-clock stall ceiling for those workers.
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from statistics import quantiles
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Safety multipliers: trip the watchdog only when output exceeds p95 by this much.
# 5x is the elbow our data shows between "rambling" and "legitimately long" — the
# haiku:low calc3 runaway was ~24x peer median; brutal-tier successes vary <2x.
DEFAULT_BUDGET_MULTIPLIER = 5.0
# Bytes per token (rough): 4 chars/token across most CLIs. We store the budget in
# BYTES because that's what the streaming watchdog can count without parsing JSON.
BYTES_PER_TOKEN = 4
# Conservative defaults when the table has no data for a config. Picked to NOT
# false-positive on any successful run measured this session.
DEFAULT_MAX_BYTES = 200_000        # ~50k tokens — far above any observed success
DEFAULT_STALL_SECONDS = 300.0      # 5 min of no progress on EITHER stream = stalled

# Per-worker stall ceilings (cold-start, no calibration data). codex has the
# longest legitimate quiet windows because of its in-CLI network-retry logic
# (an upstream API hiccup can leave it silent for several minutes before the
# CLI's own retry logic reconnects); agy/grok also reason in long bursts.
# claude's normal CLI path (--output-format json) is silent until the end of
# the call but the call itself is short, so a tighter ceiling is safe there.
# Override per call via AGY_STALL_SECONDS env var or by writing to the agent's
# stall_seconds attribute directly.
DEFAULT_STALL_SECONDS_BY_WORKER = {
    "codex": 600.0,
    "agy":   600.0,
    "grok":  600.0,
    "claude": 180.0,
}

def research_dir() -> Path:
    """Persistent directory for research/telemetry JSONL (offline calibration
    sweeps + the live ledger). Repo-relative and gitignored so the measured data
    SURVIVES across reboots — the prior ``/tmp/agentorch_research`` default got
    wiped by /tmp cleanup and lost the per-(task,model,effort) sweep results.
    Override with ``AGY_RESEARCH_DIR``."""
    override = os.environ.get("AGY_RESEARCH_DIR")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2] / "var" / "agentorch_research"


# Path the calibrate sweep writes to (persistent, see research_dir()).
DEFAULT_CALIBRATION_PATH = Path(
    os.environ.get("AGY_CALIBRATION_JSONL") or str(research_dir() / "calibrate.jsonl")
)

# Live ledger — appended after every verified dispatch by harness/dispatch.py.
# CalibrationTable.load() reads it alongside the offline sweep so per-config
# budgets continuously refine from real traffic. The contextual-bandit
# routing literature (arxiv 2510.07429, 2508.21141) backs this closing-the-
# loop pattern over static-only routing: online feedback over deployment
# trajectories outperforms offline-only calibration by ~12% on
# accuracy-cost frontiers. Disable via AGY_LIVE_LEDGER=off.
DEFAULT_LIVE_LEDGER_PATH = Path(
    os.environ.get("AGY_LIVE_LEDGER_JSONL") or str(research_dir() / "live_ledger.jsonl")
)


class CalibrationTable:
    """Per-config (max_output_bytes, stall_seconds) lookup, derived from prior runs.

    Empty table -> all configs return the conservative defaults. The watchdog
    works correctly with an empty table (it just never trips false positives);
    sweep data only tightens the budgets.
    """

    def __init__(self, multiplier: float = DEFAULT_BUDGET_MULTIPLIER):
        self.multiplier = multiplier
        # (worker, model, effort) -> list of (out_tokens, wall_ms) from successful runs
        self._success: dict[tuple[str, str, Optional[str]], list[tuple[int, float]]] = defaultdict(list)

    def ingest(self, rows: Iterable[dict]) -> int:
        """Add rows; return count actually used (passing rows with usable telemetry).

        A corrupt ledger row (non-numeric or non-finite ``out_tokens`` /
        ``wall_ms``) is skipped rather than allowed to crash or poison the
        table. ``load()`` promises never to error on a partially-written
        live ledger, and ``budget_for()`` must never see a string (TypeError
        on ``str > 0``) or an Infinity/NaN (which would silently disable the
        watchdog by inflating the stall budget to inf). Validate here, at the
        single point telemetry enters the table.
        """
        used = 0
        for r in rows:
            if not r.get("ok"):
                continue
            key = (r.get("worker", ""), r.get("model", ""), r.get("effort"))
            out_tok = r.get("out_tokens")
            wall_ms = r.get("wall_ms")
            if out_tok is None and wall_ms is None:
                continue
            out_val = self._coerce_finite(out_tok)
            wall_val = self._coerce_finite(wall_ms)
            # Drop a row only if BOTH telemetry fields are unusable; a row that
            # carries one valid metric (e.g. wall-only grok/agy rows) is still
            # recorded, with the unusable field zeroed.
            if out_val is None and wall_val is None:
                continue
            self._success[key].append((int(out_val or 0), float(wall_val or 0.0)))
            used += 1
        return used

    @staticmethod
    def _coerce_finite(value) -> Optional[float]:
        """Return ``value`` as a finite float, or None if it isn't usable.

        Handles corrupt ledger rows: non-numeric strings, bool (rejected —
        a stray ``true`` is not a token count), and Infinity/NaN literals
        (which json.loads accepts by default)."""
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
        f = float(value)
        if f != f or f in (float("inf"), float("-inf")):  # NaN or +/-inf
            return None
        return f

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        *,
        live_path: Optional[Path] = None,
    ) -> "CalibrationTable":
        """Build a table from the offline sweep + the live dispatch ledger.

        Missing files return an empty (or partial) table — never errors. The
        live ledger is what makes this a *closing-the-loop* router: every
        verified dispatch contributes a real observation, so budgets refine
        from production traffic rather than only the calibrate.py snapshot."""
        table = cls()
        sources: List[Tuple[str, Path]] = []
        offline_path = Path(path or DEFAULT_CALIBRATION_PATH)
        sources.append(("offline", offline_path))
        # Live ledger reads default unless the caller opted out by passing
        # an explicit live_path=Path('/dev/null') or env var disables.
        if live_path is not None:
            sources.append(("live", Path(live_path)))
        elif os.environ.get("AGY_LIVE_LEDGER", "").lower() != "off":
            sources.append(("live", DEFAULT_LIVE_LEDGER_PATH))
        for label, p in sources:
            if not p.exists():
                logger.debug("calibration: no %s file at %s", label, p)
                continue
            try:
                with open(p) as f:
                    rows = [json.loads(line) for line in f if line.strip()]
            except Exception as exc:
                logger.warning("calibration: failed to read %s %s: %s", label, p, exc)
                continue
            try:
                used = table.ingest(rows)
            except Exception as exc:  # defense in depth: load() must never error
                logger.warning("calibration: failed to ingest %s rows from %s: %s",
                               label, p, exc)
                continue
            logger.info("calibration: loaded %d/%d passing %s rows from %s",
                        used, len(rows), label, p)
        return table

    def budget_for(self, worker: str, model: str,
                   effort: Optional[str]) -> tuple[int, float]:
        """Return (max_output_bytes, stall_seconds) for a (worker, model, effort)."""
        # Per-worker default ceiling; falls back to the global default for unknown
        # workers. Empirical: codex's network-retry can leave it silent ~5 min.
        default_stall = DEFAULT_STALL_SECONDS_BY_WORKER.get(worker, DEFAULT_STALL_SECONDS)
        rows = self._success.get((worker, model, effort), [])
        if len(rows) < 3:
            return (DEFAULT_MAX_BYTES, default_stall)
        token_samples = [r[0] for r in rows if r[0] > 0]
        wall_samples = [r[1] for r in rows if r[1] > 0]
        max_bytes = DEFAULT_MAX_BYTES
        stall_seconds = default_stall
        # statistics.quantiles needs >=2 samples; for tiny n we still want some
        # tightening, so use the simple max-with-multiplier fallback.
        if len(token_samples) >= 2:
            p95_tokens = quantiles(token_samples, n=20)[-1] if len(token_samples) >= 20 else max(token_samples)
            max_bytes = max(int(p95_tokens * BYTES_PER_TOKEN * self.multiplier),
                            8_000)  # never go below 8KB; tiny runs still get headroom
        if len(wall_samples) >= 2:
            p95_wall_ms = quantiles(wall_samples, n=20)[-1] if len(wall_samples) >= 20 else max(wall_samples)
            # Stall ceiling: a successful run took p95 ms; if a new run has produced
            # NO output for >p95 wall, it's almost certainly stalled. Multiply for
            # safety. Floor at the per-worker cold-start default (NOT a hardcoded
            # 30s): a calibrated config must never get a *tighter* stall window than
            # an uncalibrated one, or gathering data would make the watchdog more
            # trigger-happy and kill silent high-effort reasoning workers (e.g. codex
            # effort=high) before their first token. See issue #29.
            stall_seconds = max(p95_wall_ms / 1000 * self.multiplier, default_stall)
        return (max_bytes, stall_seconds)

    def has_data_for(self, worker: str, model: str, effort: Optional[str]) -> bool:
        return len(self._success.get((worker, model, effort), [])) >= 3


def append_live_row(
    worker: str,
    model: str,
    effort: Optional[str],
    *,
    ok: bool,
    out_bytes: Optional[int] = None,
    wall_ms: Optional[float] = None,
    mode: Optional[str] = None,
    stage_used: Optional[int] = None,
    n_candidates: Optional[int] = None,
    n_passed: Optional[int] = None,
    winner_index: Optional[int] = None,
    verifier_delta: Optional[str] = None,
    verifier_failure_kind: Optional[str] = None,
    diff_files_added: Optional[int] = None,
    diff_files_modified: Optional[int] = None,
    diff_files_deleted: Optional[int] = None,
    path: Optional[Path] = None,
) -> None:
    """Append one dispatch's observation to the live ledger JSONL.

    Called from harness/dispatch.py after every verified run. Schema matches
    what CalibrationTable.ingest() consumes — ``ok`` gates whether the row
    contributes (matching the offline sweep), and out_bytes is converted to
    out_tokens via the same BYTES_PER_TOKEN rate the table uses.

    No-op when ``AGY_LIVE_LEDGER=off`` — defense in depth on top of the
    same env-var check in load(). Also no-op on rows missing both telemetry
    fields (nothing useful to record).

    Failures are swallowed: a disk error MUST NOT crash a successful
    dispatch — the worst case is a missed observation, not a regression
    in the operator's run.
    """
    if os.environ.get("AGY_LIVE_LEDGER", "").lower() == "off":
        return
    has_metric = any(
        v is not None
        for v in (
            out_bytes,
            wall_ms,
            mode,
            stage_used,
            n_candidates,
            n_passed,
            winner_index,
            verifier_delta,
            verifier_failure_kind,
            diff_files_added,
            diff_files_modified,
            diff_files_deleted,
        )
    )
    if not has_metric:
        return
    row = {
        "worker": worker,
        "model": model,
        "effort": effort,
        "ok": bool(ok),
    }
    optional_values = {
        "out_tokens": (int(out_bytes) // BYTES_PER_TOKEN) if out_bytes is not None else None,
        "wall_ms": float(wall_ms) if wall_ms is not None else None,
        "mode": mode,
        "stage_used": stage_used,
        "n_candidates": n_candidates,
        "n_passed": n_passed,
        "winner_index": winner_index,
        "verifier_delta": verifier_delta,
        "verifier_failure_kind": verifier_failure_kind,
        "diff_files_added": diff_files_added,
        "diff_files_modified": diff_files_modified,
        "diff_files_deleted": diff_files_deleted,
    }
    for key, value in optional_values.items():
        if value is not None:
            row[key] = value
    target = Path(path or DEFAULT_LIVE_LEDGER_PATH)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception as exc:
        logger.warning("calibration: failed to append live ledger row to %s: %s",
                       target, exc)

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
from typing import Iterable, Optional

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
DEFAULT_STALL_SECONDS = 180.0      # 3 min of no progress = stalled

# Path the calibrate sweep writes to; sweep results in /tmp by default.
DEFAULT_CALIBRATION_PATH = Path(
    os.environ.get("AGY_CALIBRATION_JSONL", "/tmp/agentorch_research/calibrate.jsonl")
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
        """Add rows; return count actually used (passing rows with usable telemetry)."""
        used = 0
        for r in rows:
            if not r.get("ok"):
                continue
            key = (r.get("worker", ""), r.get("model", ""), r.get("effort"))
            out_tok = r.get("out_tokens")
            wall_ms = r.get("wall_ms")
            if out_tok is None and wall_ms is None:
                continue
            self._success[key].append((out_tok or 0, float(wall_ms or 0)))
            used += 1
        return used

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CalibrationTable":
        """Build a table from a JSONL file. Missing file returns an empty table."""
        table = cls()
        path = Path(path or DEFAULT_CALIBRATION_PATH)
        if not path.exists():
            logger.debug("calibration: no file at %s; using defaults", path)
            return table
        try:
            with open(path) as f:
                rows = [json.loads(line) for line in f if line.strip()]
        except Exception as exc:
            logger.warning("calibration: failed to read %s: %s", path, exc)
            return table
        used = table.ingest(rows)
        logger.info("calibration: loaded %d/%d passing rows from %s",
                    used, len(rows), path)
        return table

    def budget_for(self, worker: str, model: str,
                   effort: Optional[str]) -> tuple[int, float]:
        """Return (max_output_bytes, stall_seconds) for a (worker, model, effort)."""
        rows = self._success.get((worker, model, effort), [])
        if len(rows) < 3:
            return (DEFAULT_MAX_BYTES, DEFAULT_STALL_SECONDS)
        token_samples = [r[0] for r in rows if r[0] > 0]
        wall_samples = [r[1] for r in rows if r[1] > 0]
        max_bytes = DEFAULT_MAX_BYTES
        stall_seconds = DEFAULT_STALL_SECONDS
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
            # safety; floor at 30s so tiny tasks don't ship a 5s stall budget.
            stall_seconds = max(p95_wall_ms / 1000 * self.multiplier, 30.0)
        return (max_bytes, stall_seconds)

    def has_data_for(self, worker: str, model: str, effort: Optional[str]) -> bool:
        return len(self._success.get((worker, model, effort), [])) >= 3

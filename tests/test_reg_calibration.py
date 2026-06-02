"""Regression tests for corrupt-row robustness in CalibrationTable.

Subsystem: core optimizer/profile/calibration/model_discovery + execution
ledger/selector.

Both bugs share one root cause: CalibrationTable.ingest() coerced telemetry
fields (out_tokens / wall_ms) into numbers with no validation, so a single
corrupt live-ledger JSONL row could crash or silently poison the table.
load() promises in its docstring to "never error" on a partially-written
ledger, and that file is appended to after every dispatch — a corrupt row
must degrade to a partial table, not abort the watchdog-arming hot path.

Hermetic: no workers, no network, no credentials. Pure on-disk JSONL files
in tmp_path.

calibration-1: load() must not crash on a non-numeric wall_ms (ValueError out
              of float()), and budget_for() must not crash on a non-numeric
              out_tokens (TypeError on `str > 0`) once >=3 rows accumulate.
calibration-2: an Infinity / NaN wall_ms (json.loads accepts these literals)
              must not inflate the stall budget to inf, which silently disables
              the watchdog.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from agy_orchestrator.core.calibration import (
    DEFAULT_MAX_BYTES,
    DEFAULT_STALL_SECONDS_BY_WORKER,
    CalibrationTable,
)


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# --------------------------------------------------------------------------
# calibration-1: corrupt-but-valid-JSON rows must not crash load()/budget_for()
# --------------------------------------------------------------------------

def test_load_does_not_crash_on_non_numeric_wall_ms(tmp_path):
    """A single JSON-valid row with a string wall_ms used to raise ValueError
    straight out of ingest() (outside load()'s try/except). load() promises
    never to error — it must return a (partial) table instead."""
    p = _write_jsonl(
        tmp_path / "live.jsonl",
        [{"ok": True, "worker": "codex", "model": "m", "effort": "high",
          "wall_ms": "oops"}],
    )
    # Must not raise.
    t = CalibrationTable.load(path=p, live_path=tmp_path / "no_live.jsonl")
    assert isinstance(t, CalibrationTable)
    # The corrupt row contributed nothing usable -> defaults stand.
    max_bytes, stall = t.budget_for("codex", "m", "high")
    assert max_bytes == DEFAULT_MAX_BYTES
    assert stall == DEFAULT_STALL_SECONDS_BY_WORKER["codex"]


def test_budget_for_does_not_crash_on_string_out_tokens(tmp_path):
    """>=3 rows with a string out_tokens used to survive ingest() (stored
    verbatim) then crash budget_for() on `str > 0` (TypeError)."""
    rows = [{"ok": True, "worker": "codex", "model": "m", "effort": "high",
             "out_tokens": "big", "wall_ms": 100} for _ in range(3)]
    p = _write_jsonl(tmp_path / "live.jsonl", rows)
    t = CalibrationTable.load(path=p, live_path=tmp_path / "no_live.jsonl")
    # Must not raise TypeError.
    max_bytes, stall = t.budget_for("codex", "m", "high")
    assert isinstance(max_bytes, int)
    # wall_ms (100ms) is valid, so the rows are still recorded for stall.
    assert math.isfinite(stall)


def test_corrupt_out_tokens_does_not_break_direct_ingest():
    t = CalibrationTable()
    rows = [{"ok": True, "worker": "codex", "model": "m", "effort": "high",
             "out_tokens": "big", "wall_ms": 100} for _ in range(3)]
    used = t.ingest(rows)
    # wall metric is valid -> rows still counted.
    assert used == 3
    # And budget_for stays crash-free with a finite stall.
    _, stall = t.budget_for("codex", "m", "high")
    assert math.isfinite(stall)


def test_row_with_both_fields_unusable_is_dropped():
    """If BOTH telemetry fields are non-numeric, the row carries nothing and is
    skipped entirely (so it can't pad the >=3 threshold with junk)."""
    t = CalibrationTable()
    rows = [{"ok": True, "worker": "codex", "model": "m", "effort": "high",
             "out_tokens": "big", "wall_ms": "oops"} for _ in range(5)]
    assert t.ingest(rows) == 0
    assert not t.has_data_for("codex", "m", "high")


# --------------------------------------------------------------------------
# calibration-2: Infinity / NaN must not poison the stall budget
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_wall_ms_does_not_poison_stall_budget(tmp_path, bad):
    """json.loads accepts Infinity/-Infinity/NaN. A non-finite wall_ms used to
    flow into stall_seconds = max(inf, default) = inf, silently disabling the
    watchdog. Such rows must be dropped, leaving a finite stall budget."""
    # Emit the literals via Python's json so the on-disk file matches what a
    # real corrupt ledger row would contain.
    rows = [{"ok": True, "worker": "codex", "model": "m", "effort": "high",
             "out_tokens": 100, "wall_ms": bad} for _ in range(3)]
    p = _write_jsonl(tmp_path / "live.jsonl", rows)
    t = CalibrationTable.load(path=p, live_path=tmp_path / "no_live.jsonl")
    _, stall = t.budget_for("codex", "m", "high")
    assert math.isfinite(stall)
    # The non-finite wall rows are dropped; out_tokens=100 alone (no valid
    # wall) -> stall falls back to the per-worker cold-start default.
    assert stall == DEFAULT_STALL_SECONDS_BY_WORKER["codex"]


def test_non_finite_does_not_disable_watchdog_via_ingest():
    t = CalibrationTable()
    rows = [{"ok": True, "worker": "agy", "model": "m", "effort": "high",
             "out_tokens": 100, "wall_ms": float("inf")} for _ in range(3)]
    t.ingest(rows)
    _, stall = t.budget_for("agy", "m", "high")
    assert math.isfinite(stall)


def test_load_never_errors_contract_on_mixed_corrupt_ledger(tmp_path):
    """End-to-end: a ledger holding good rows interleaved with every flavour of
    corruption must load cleanly and still use the good rows."""
    good = {"ok": True, "worker": "claude", "model": "sonnet", "effort": "low",
            "out_tokens": 200, "wall_ms": 5000}
    rows = [
        good,
        {"ok": True, "worker": "claude", "model": "sonnet", "effort": "low",
         "wall_ms": "oops"},
        {"ok": True, "worker": "claude", "model": "sonnet", "effort": "low",
         "out_tokens": "big", "wall_ms": float("inf")},
        good,
        {"ok": True, "worker": "claude", "model": "sonnet", "effort": "low",
         "out_tokens": float("nan"), "wall_ms": float("nan")},
        good,
    ]
    p = _write_jsonl(tmp_path / "live.jsonl", rows)
    # Must not raise.
    t = CalibrationTable.load(path=p, live_path=tmp_path / "no_live.jsonl")
    assert t.has_data_for("claude", "sonnet", "low")
    _, stall = t.budget_for("claude", "sonnet", "low")
    assert math.isfinite(stall)

"""CalibrationTable derives per-(worker,model,effort) byte/stall budgets from
prior successful runs. With no data it must return SAFE defaults so the watchdog
never false-positives; with data it tightens budgets but never below a floor.
"""
from __future__ import annotations

import json
from pathlib import Path

from agy_orchestrator.core.calibration import (
    BYTES_PER_TOKEN,
    DEFAULT_MAX_BYTES,
    DEFAULT_STALL_SECONDS,
    DEFAULT_STALL_SECONDS_BY_WORKER,
    CalibrationTable,
)


def test_empty_table_returns_safe_defaults():
    """No calibration data ⇒ codex gets its per-worker default (not the global
    floor), unknown workers fall back to DEFAULT_STALL_SECONDS."""
    t = CalibrationTable()
    max_bytes, stall = t.budget_for("codex", "gpt-5.4-mini", "low")
    assert max_bytes == DEFAULT_MAX_BYTES
    assert stall == DEFAULT_STALL_SECONDS_BY_WORKER["codex"]
    assert not t.has_data_for("codex", "gpt-5.4-mini", "low")


def test_per_worker_stall_defaults_diverge_from_global():
    """Known workers get tuned ceilings; unknown workers fall back to the
    global default. Regression guard for the bump we made after observing
    codex's normal network-retry windows trip the old 180s ceiling."""
    t = CalibrationTable()
    _, codex_stall = t.budget_for("codex", "gpt-5.5", "high")
    _, claude_stall = t.budget_for("claude", "sonnet", "low")
    _, mystery_stall = t.budget_for("mystery", "x", None)
    assert codex_stall == DEFAULT_STALL_SECONDS_BY_WORKER["codex"]
    assert claude_stall == DEFAULT_STALL_SECONDS_BY_WORKER["claude"]
    assert mystery_stall == DEFAULT_STALL_SECONDS
    # codex's default must be ≥ the global default — that's the whole point.
    assert codex_stall >= DEFAULT_STALL_SECONDS


def test_ingest_skips_failing_rows():
    t = CalibrationTable()
    rows = [
        {"worker": "claude", "model": "sonnet", "effort": "low",
         "ok": True, "out_tokens": 200, "wall_ms": 5000},
        {"worker": "claude", "model": "sonnet", "effort": "low",
         "ok": False, "out_tokens": 99999, "wall_ms": 1},  # ignored: !ok
    ]
    used = t.ingest(rows)
    assert used == 1
    # Only 1 sample -> falls back to defaults (we require >=3 to tighten).
    max_bytes, _ = t.budget_for("claude", "sonnet", "low")
    assert max_bytes == DEFAULT_MAX_BYTES


def test_budget_tightens_with_enough_passing_rows():
    t = CalibrationTable(multiplier=5.0)
    # 5 passing rows around ~200 tokens; budget should land at p95*5*BPT ~= 4000.
    # But the floor is 8 KB, so we should see exactly 8000.
    rows = [{"worker": "claude", "model": "sonnet", "effort": "low",
             "ok": True, "out_tokens": 180 + 5 * i, "wall_ms": 5000 + 100 * i}
            for i in range(5)]
    t.ingest(rows)
    assert t.has_data_for("claude", "sonnet", "low")
    max_bytes, stall = t.budget_for("claude", "sonnet", "low")
    # The byte floor is 8_000; expected = max(p95 * 4 * 5, 8000).
    p95_tokens = max(r["out_tokens"] for r in rows)
    expected = max(int(p95_tokens * BYTES_PER_TOKEN * 5.0), 8_000)
    assert max_bytes == expected
    # Stall budget = max(p95 wall_ms * multiplier, per-worker default). Here
    # p95 wall ~5.4s * 5 = 27s, well below claude's 180s floor, so it floors at
    # the per-worker default — NOT a hardcoded 30s. (Issue #29.)
    assert stall == DEFAULT_STALL_SECONDS_BY_WORKER["claude"]


def test_calibrated_stall_never_below_per_worker_default():
    """Regression for #29: a calibrated config must never get a *tighter* stall
    window than the uncalibrated cold-start default. A fast-returning codex
    config (tiny p95 wall) used to floor at a hardcoded 30s — far below codex's
    600s cold-start default — which killed silent high-effort reasoning workers
    before their first token. The floor is now the per-worker default."""
    t = CalibrationTable(multiplier=5.0)
    # 4 fast codex:high runs: p95 wall 2s * 5 = 10s, below the old 30s floor and
    # far below codex's 600s default.
    rows = [{"worker": "codex", "model": "standard", "effort": "high",
             "ok": True, "out_tokens": 50, "wall_ms": 2000} for _ in range(4)]
    t.ingest(rows)
    assert t.has_data_for("codex", "standard", "high")
    _, calibrated_stall = t.budget_for("codex", "standard", "high")
    # Must NOT collapse to 30s — floors at codex's cold-start default.
    assert calibrated_stall == DEFAULT_STALL_SECONDS_BY_WORKER["codex"]
    # Invariant: calibrated >= uncalibrated for the same config.
    _, uncalibrated_stall = CalibrationTable().budget_for("codex", "standard", "high")
    assert calibrated_stall >= uncalibrated_stall


def test_load_missing_file_returns_empty_table(tmp_path):
    t = CalibrationTable.load(tmp_path / "nonexistent.jsonl")
    assert isinstance(t, CalibrationTable)
    # "any" is not in the per-worker table → global default applies.
    max_bytes, stall = t.budget_for("any", "any", None)
    assert (max_bytes, stall) == (DEFAULT_MAX_BYTES, DEFAULT_STALL_SECONDS)


def test_load_reads_jsonl(tmp_path: Path):
    p = tmp_path / "cal.jsonl"
    rows = [{"worker": "codex", "model": "gpt-5.4-mini", "effort": "low",
             "ok": True, "out_tokens": 150 + i, "wall_ms": 6000 + 100 * i}
            for i in range(4)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    t = CalibrationTable.load(p)
    assert t.has_data_for("codex", "gpt-5.4-mini", "low")


def test_ledger_telemetry_passthrough():
    """Soft-signal telemetry must be carried through ledger rows so the next
    calibration cycle can re-fit baselines on the latest data."""
    from agy_orchestrator.execution.ledger import build_ledger

    class _WF:
        verified = True
        approved = True
        stalled = False
        iterations_used = 1

    led = build_ledger(
        _WF(), mode="feedback", had_verifier=True, produced_output=True,
        telemetry={"wall_ms": 5432, "out_bytes": 1024, "watchdog_reason": None,
                   "worker": "codex", "model": "gpt-5.4-mini", "effort": "low"},
    )
    # None values dropped; non-None carried through.
    assert led["wall_ms"] == 5432
    assert led["out_bytes"] == 1024
    assert led["worker"] == "codex"
    assert "watchdog_reason" not in led

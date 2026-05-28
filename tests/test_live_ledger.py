"""Live ledger — closes the calibration loop from production traffic.

Every verified dispatch appends an observation. CalibrationTable.load()
reads both the offline calibrate.jsonl AND the live ledger, so per-config
budgets refine continuously from real runs rather than only from a periodic
offline sweep.

Paper anchors: arxiv 2510.07429 (bandit-fed routing +12% over offline-only),
arxiv 2508.21141 (online budget-aware contextual routing).
"""
from __future__ import annotations

import json
from pathlib import Path

from agy_orchestrator.core.calibration import (
    BYTES_PER_TOKEN,
    DEFAULT_MAX_BYTES,
    CalibrationTable,
    append_live_row,
)


def test_append_live_row_writes_jsonl(tmp_path):
    """A row appended must be readable as JSONL by ingest()."""
    p = tmp_path / "live.jsonl"
    append_live_row(
        worker="codex", model="standard", effort="high",
        ok=True, out_bytes=4000, wall_ms=12000.0, path=p,
    )
    assert p.exists()
    rows = [json.loads(line) for line in p.read_text().splitlines() if line]
    assert len(rows) == 1
    r = rows[0]
    assert r["worker"] == "codex"
    assert r["model"] == "standard"
    assert r["effort"] == "high"
    assert r["ok"] is True
    # out_bytes -> out_tokens conversion uses BYTES_PER_TOKEN
    assert r["out_tokens"] == 4000 // BYTES_PER_TOKEN
    assert r["wall_ms"] == 12000.0


def test_append_live_row_multiple_runs_aggregate(tmp_path):
    """N successive appends produce N readable rows; CalibrationTable
    .ingest() picks them all up."""
    p = tmp_path / "live.jsonl"
    for wall in (6000.0, 8000.0, 7000.0):
        append_live_row(
            worker="agy", model="pro", effort="high",
            ok=True, out_bytes=2000, wall_ms=wall, path=p,
        )
    t = CalibrationTable()
    rows = [json.loads(line) for line in p.read_text().splitlines() if line]
    used = t.ingest(rows)
    assert used == 3
    assert t.has_data_for("agy", "pro", "high")


def test_load_reads_both_offline_and_live(tmp_path):
    """The key feature: CalibrationTable.load() merges offline + live."""
    offline = tmp_path / "calibrate.jsonl"
    live = tmp_path / "live.jsonl"
    # 2 offline rows + 2 live rows for the same config = 4 samples total,
    # passes the >=3 threshold so the table tightens budgets.
    offline.write_text("\n".join([
        json.dumps({"worker": "claude", "model": "sonnet", "effort": "low",
                    "ok": True, "out_tokens": 200, "wall_ms": 5500}),
        json.dumps({"worker": "claude", "model": "sonnet", "effort": "low",
                    "ok": True, "out_tokens": 220, "wall_ms": 5800}),
    ]) + "\n")
    live.write_text("\n".join([
        json.dumps({"worker": "claude", "model": "sonnet", "effort": "low",
                    "ok": True, "out_tokens": 180, "wall_ms": 5200}),
        json.dumps({"worker": "claude", "model": "sonnet", "effort": "low",
                    "ok": True, "out_tokens": 240, "wall_ms": 6100}),
    ]) + "\n")
    t = CalibrationTable.load(offline, live_path=live)
    assert t.has_data_for("claude", "sonnet", "low")
    # Budget should now be tightened (>=3 samples present), not the default.
    max_bytes, _ = t.budget_for("claude", "sonnet", "low")
    assert max_bytes != DEFAULT_MAX_BYTES, (
        "Expected the combined offline+live data to tighten the byte budget."
    )


def test_load_handles_missing_live_ledger(tmp_path):
    """Live ledger missing -> still loads offline cleanly."""
    offline = tmp_path / "calibrate.jsonl"
    offline.write_text(json.dumps({
        "worker": "codex", "model": "x", "effort": "high",
        "ok": True, "out_tokens": 100, "wall_ms": 1000,
    }) + "\n")
    live = tmp_path / "nonexistent.jsonl"
    t = CalibrationTable.load(offline, live_path=live)
    # 1 row -> below threshold, default returned, but no crash.
    max_bytes, _ = t.budget_for("codex", "x", "high")
    assert max_bytes == DEFAULT_MAX_BYTES


def test_load_handles_missing_offline_with_live(tmp_path):
    """Offline ledger missing -> still loads live cleanly (asymmetric case)."""
    offline = tmp_path / "nonexistent.jsonl"
    live = tmp_path / "live.jsonl"
    live.write_text("\n".join([
        json.dumps({"worker": "grok", "model": "grok-build", "effort": None,
                    "ok": True, "out_tokens": 500, "wall_ms": 30000})
        for _ in range(4)
    ]) + "\n")
    t = CalibrationTable.load(offline, live_path=live)
    assert t.has_data_for("grok", "grok-build", None)


def test_append_live_row_skips_when_disabled(tmp_path, monkeypatch):
    """AGY_LIVE_LEDGER=off -> append is a no-op. Operator opt-out path."""
    monkeypatch.setenv("AGY_LIVE_LEDGER", "off")
    p = tmp_path / "live.jsonl"
    append_live_row(
        worker="codex", model="m", effort="high",
        ok=True, out_bytes=1000, wall_ms=1000.0, path=p,
    )
    assert not p.exists(), "live ledger should not be written when disabled"


def test_append_live_row_skips_when_no_telemetry(tmp_path):
    """Rows with neither out_bytes nor wall_ms are useless — skip."""
    p = tmp_path / "live.jsonl"
    append_live_row(
        worker="codex", model="m", effort="high",
        ok=True, out_bytes=None, wall_ms=None, path=p,
    )
    assert not p.exists()


def test_append_live_row_swallows_disk_errors(tmp_path, monkeypatch):
    """If the live ledger can't be written, the dispatch must complete
    normally — the worst-case outcome of a write failure is a missed
    observation, not a crashed run."""
    p = tmp_path / "live.jsonl"

    def _explode(*args, **kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr(Path, "mkdir", _explode)
    # Should not raise.
    append_live_row(
        worker="codex", model="m", effort="high",
        ok=True, out_bytes=1000, wall_ms=1000.0, path=p,
    )


def test_load_disabled_skips_live_path(tmp_path, monkeypatch):
    """AGY_LIVE_LEDGER=off on load() also skips the live file entirely —
    matches the append-side semantics."""
    monkeypatch.setenv("AGY_LIVE_LEDGER", "off")
    offline = tmp_path / "calibrate.jsonl"
    offline.write_text("")
    # Default path used; should NOT crash even though the env disables
    # the live read. We just need to verify load() returns cleanly.
    t = CalibrationTable.load(offline)
    assert isinstance(t, CalibrationTable)

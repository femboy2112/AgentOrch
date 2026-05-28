from __future__ import annotations

import json

from agy_orchestrator.core.calibration import append_live_row


def test_append_live_row_writes_new_fields(tmp_path):
    p = tmp_path / "live.jsonl"
    append_live_row(
        worker="codex",
        model="standard",
        effort="high",
        ok=True,
        mode="pat",
        stage_used=0,
        n_candidates=3,
        verifier_delta="fixed",
        diff_files_modified=2,
        path=p,
    )
    rows = [json.loads(line) for line in p.read_text().splitlines() if line]
    assert len(rows) == 1
    row = rows[0]
    assert row["mode"] == "pat"
    assert row["stage_used"] == 0
    assert row["n_candidates"] == 3
    assert row["verifier_delta"] == "fixed"
    assert row["diff_files_modified"] == 2


def test_append_live_row_skips_none_new_fields(tmp_path):
    p = tmp_path / "live.jsonl"
    append_live_row(
        worker="codex",
        model="standard",
        effort="high",
        ok=True,
        wall_ms=100.0,
        path=p,
    )
    row = json.loads(p.read_text().strip())
    for key in (
        "mode",
        "stage_used",
        "n_candidates",
        "n_passed",
        "winner_index",
        "verifier_delta",
        "verifier_failure_kind",
        "diff_files_added",
        "diff_files_modified",
        "diff_files_deleted",
    ):
        assert key not in row


def test_append_live_row_negative_sentinels_passed_as_none(tmp_path):
    """Sentinel normalization happens at dispatch call sites, not here."""
    p = tmp_path / "live.jsonl"
    append_live_row(
        worker="codex",
        model="standard",
        effort="high",
        ok=True,
        wall_ms=100.0,
        stage_used=-1,
        path=p,
    )
    row = json.loads(p.read_text().strip())
    assert row["stage_used"] == -1


def test_append_live_row_disabled_skips_all_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_LIVE_LEDGER", "off")
    p = tmp_path / "live.jsonl"
    append_live_row(
        worker="codex",
        model="standard",
        effort="high",
        ok=True,
        mode="pat",
        stage_used=0,
        n_candidates=3,
        verifier_delta="fixed",
        diff_files_modified=2,
        path=p,
    )
    assert not p.exists()

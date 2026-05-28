from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient

from dashboard.routers.live import _CLI_RUN_STALE_SECONDS, _scan_filesystem_runs
from dashboard.server import create_app


def _make_run_dir(
    runs_dir: Path,
    run_id: str,
    *,
    with_meta: bool = False,
    events_mtime: float | None = None,
    prompt_text: str = "## Instruction\nrun from cli",
) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = run_dir / "prompt.txt"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    events_path = run_dir / "events.jsonl"
    events_path.write_text('{"kind":"message","text":"hi"}\n', encoding="utf-8")
    if events_mtime is not None:
        os.utime(events_path, (events_mtime, events_mtime))
    if with_meta:
        (run_dir / "meta.json").write_text("{}", encoding="utf-8")
    return run_dir


def test_fs_scan_finds_run_without_meta(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _make_run_dir(runs_dir, "20260528-120000-001")
    rows = _scan_filesystem_runs(runs_dir, set())
    assert any(row["run_id"] == "20260528-120000-001" for row in rows)


def test_fs_scan_skips_run_with_meta(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _make_run_dir(runs_dir, "20260528-120000-002", with_meta=True)
    rows = _scan_filesystem_runs(runs_dir, set())
    assert all(row["run_id"] != "20260528-120000-002" for row in rows)


def test_fs_scan_skips_stale_run(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    stale = time.time() - _CLI_RUN_STALE_SECONDS - 5.0
    _make_run_dir(runs_dir, "20260528-120000-003", events_mtime=stale)
    rows = _scan_filesystem_runs(runs_dir, set())
    assert all(row["run_id"] != "20260528-120000-003" for row in rows)


def test_fs_scan_skips_tracked_run(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_id = "20260528-120000-004"
    _make_run_dir(runs_dir, run_id)
    rows = _scan_filesystem_runs(runs_dir, {run_id})
    assert all(row["run_id"] != run_id for row in rows)


def test_list_live_merges_tracked_and_fs(tmp_path: Path) -> None:
    app = create_app()
    app.state.dashboard.runs_dir = tmp_path / "runs"
    app.state.dashboard.running = {
        "tracked-1": {
            "run_id": "tracked-1",
            "started_at": 100.0,
            "mode": "adversarial",
            "generator": ["codex"],
            "critic": ["agy"],
        }
    }

    now = time.time()
    _make_run_dir(
        app.state.dashboard.runs_dir,
        "cli-1",
        events_mtime=now,
    )
    os.utime(app.state.dashboard.runs_dir / "cli-1" / "prompt.txt", (200.0, 200.0))

    with TestClient(app) as client:
        resp = client.get("/api/live")
    assert resp.status_code == 200

    running = resp.json()["running"]
    ids = [row["run_id"] for row in running]
    assert "tracked-1" in ids
    assert "cli-1" in ids
    assert ids.index("cli-1") < ids.index("tracked-1")

"""A stale/aborted run dir (events.jsonl but no meta.json, untouched for a long
time) — or a non-timestamp test/dev leftover like 'r-red-1' — must NOT be
reported as a perpetual "in progress", and must not hijack "latest" by sorting
lexically above real timestamped runs.
"""
from __future__ import annotations

import os
import time

import pytest

from harness import telegram_bot as tb


def _mk_run(root, name, *, events=True, meta=False, age_s=0.0):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    if events:
        (d / "events.jsonl").write_text('{"event_type":"x","run_id":"%s"}\n' % name)
    if meta:
        (d / "meta.json").write_text('{"success": true, "mode": "direct"}')
    if age_s:
        old = time.time() - age_s
        for f in d.iterdir():
            os.utime(f, (old, old))
        os.utime(d, (old, old))
    return d


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "RUNS_DIR", tmp_path)
    monkeypatch.delenv("AGY_TELEGRAM_LIVE_STALE_S", raising=False)
    return tmp_path


@pytest.mark.not_slow
def test_stale_metaless_dir_is_not_live(runs):
    # 4 days old, events but no meta -> aborted, not live.
    d = _mk_run(runs, "r-red-1", events=True, meta=False, age_s=4 * 86400)
    assert tb.is_live_run(d) is False


@pytest.mark.not_slow
def test_fresh_metaless_dir_is_live(runs):
    d = _mk_run(runs, "20260602-120000-001", events=True, meta=False, age_s=0)
    assert tb.is_live_run(d) is True


@pytest.mark.not_slow
def test_run_dirs_ordered_by_mtime_not_name(runs):
    # 'r-red-1' sorts lexically above the timestamp name, but is much older.
    _mk_run(runs, "r-red-1", events=True, meta=False, age_s=4 * 86400)
    fresh = _mk_run(runs, "20260602-120000-001", events=True, meta=True, age_s=0)
    assert tb._run_dirs()[0].name == fresh.name  # recency wins, not the name


@pytest.mark.not_slow
def test_status_skips_stale_and_reports_completed(runs):
    # A stale leftover plus a real completed run: /status must report the completed
    # run, NOT "in progress · r-red-1".
    _mk_run(runs, "r-red-1", events=True, meta=False, age_s=4 * 86400)
    _mk_run(runs, "20260602-120000-001", events=True, meta=True, age_s=10)
    out = tb.summarize_latest()
    assert "in progress" not in out.lower()
    assert "r-red-1" not in out
    assert "20260602-120000-001" in out


@pytest.mark.not_slow
def test_status_reports_genuinely_live_run(runs):
    _mk_run(runs, "20260602-120000-002", events=True, meta=False, age_s=0)
    out = tb.summarize_latest()
    assert "in progress" in out.lower()
    assert "20260602-120000-002" in out


@pytest.mark.not_slow
def test_only_stale_leftovers_reports_no_completed(runs):
    _mk_run(runs, "r-red-1", events=True, meta=False, age_s=4 * 86400)
    _mk_run(runs, "cu-probe", events=True, meta=False, age_s=5 * 86400)
    out = tb.summarize_latest()
    assert "in progress" not in out.lower()
    assert "no completed runs" in out.lower()

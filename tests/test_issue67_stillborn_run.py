"""Issue #67: a run that registers but exits before writing events/meta (a
stillborn or killed dispatch) must NOT stay '• in progress' forever in /runs. The
dispatch records its PID; the tracker deregisters the run the instant that PID is
gone (with a recency fallback for old runs that have no PID).
"""
from __future__ import annotations

import os
import subprocess
import time

import pytest

from harness import telegram_bot as bot


def _dead_pid() -> int:
    """A PID that is guaranteed not to be running (spawned then reaped)."""
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def _mk(root, name, *, events=True, meta=False, pid=None, age_s=0.0):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    if events:
        (d / "events.jsonl").write_text("")  # 0-byte, as a stillborn run leaves it
    if meta:
        (d / "meta.json").write_text('{"success": true, "mode": "direct"}')
    if pid is not None:
        (d / "run.pid").write_text(str(pid))
    if age_s:
        old = time.time() - age_s
        for f in d.iterdir():
            os.utime(f, (old, old))
        os.utime(d, (old, old))
    return d


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "RUNS_DIR", tmp_path)
    monkeypatch.delenv("AGY_TELEGRAM_LIVE_STALE_S", raising=False)
    return tmp_path


# --------------------------------------------------------------------------- #
# _run_pid_alive
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_pid_alive_true_for_own_pid(runs):
    d = _mk(runs, "r1", pid=os.getpid())
    assert bot._run_pid_alive(d) is True


@pytest.mark.not_slow
def test_pid_alive_false_for_dead_pid(runs):
    d = _mk(runs, "r2", pid=_dead_pid())
    assert bot._run_pid_alive(d) is False


@pytest.mark.not_slow
def test_pid_alive_none_when_no_pidfile(runs):
    d = _mk(runs, "r3")  # no run.pid
    assert bot._run_pid_alive(d) is None


# --------------------------------------------------------------------------- #
# is_live_run
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_live_when_pid_alive_even_with_empty_events(runs):
    # A just-born run with a 0-byte events.jsonl but a LIVE process is in progress.
    d = _mk(runs, "20260602-000001", pid=os.getpid())
    assert bot.is_live_run(d) is True


@pytest.mark.not_slow
def test_not_live_when_pid_dead(runs):
    # Stillborn: registered (pid file) but the process is gone and no meta written.
    d = _mk(runs, "20260602-000002", pid=_dead_pid())
    assert bot.is_live_run(d) is False


@pytest.mark.not_slow
def test_meta_present_never_live(runs):
    d = _mk(runs, "20260602-000003", meta=True, pid=os.getpid())
    assert bot.is_live_run(d) is False


@pytest.mark.not_slow
def test_no_pid_falls_back_to_recency(runs):
    fresh = _mk(runs, "20260602-000004", events=True)  # no pid, recent
    (fresh / "events.jsonl").write_text('{"e":1}\n')
    assert bot.is_live_run(fresh) is True
    stale = _mk(runs, "20260602-000005", events=True, age_s=4 * 86400)  # no pid, old
    assert bot.is_live_run(stale) is False


# --------------------------------------------------------------------------- #
# summarize_runs labels
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_summarize_runs_marks_stillborn_aborted_not_in_progress(runs):
    _mk(runs, "20260602-130001", pid=_dead_pid())          # stillborn
    _mk(runs, "20260602-130002", pid=os.getpid())          # genuinely live
    out = bot.summarize_runs()
    assert "130001" in out and "aborted" in out
    # The live one is still in progress; the stillborn one is NOT counted as such.
    assert out.count("in progress") == 1
    assert "130002" in out
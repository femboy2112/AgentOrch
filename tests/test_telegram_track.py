"""Hermetic tests for the TEMPORARY cross-process live-run tracking commands.

The bot is a separate process from a running ``harness do`` and follows a build
by TAILING the on-disk ``runs/<id>/events.jsonl``. These tests cover discovery,
``/track`` (latest / explicit / all), bounded cursor-advancing tailing,
auto-untrack + final summary on ``meta.json`` appearing, ``/untrack``, and
resilience to garbage / missing event streams.

Everything is hermetic: no network, no real token, the runs dir is a tmp dir,
state is pinned to tmp via the autouse ``isolate_real_paths`` fixture that also
forbids touching the real default state/whitelist paths. Sends are captured.
"""
from __future__ import annotations

import json

import pytest

from harness import telegram as tg
from harness import telegram_bot as bot


FAKE_TOKEN = "123456:FAKE-TEST-TOKEN-not-real"
FAKE_USER_ID = 111222333
FAKE_CHAT_ID = 444555666


# Reuse the same hermetic isolation contract as tests/test_telegram.py: pin the
# state + whitelist to tmp and forbid opening the real default out-of-repo paths.
@pytest.fixture(autouse=True)
def isolate_real_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_TELEGRAM_STATE", str(tmp_path / "iso_bot_state.json"))
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(tmp_path / "iso_users.json"))

    real_paths = {tg.DEFAULT_USERS_PATH, bot.DEFAULT_STATE_PATH}
    real_opened: list = []
    real_open = open

    def guarded_open(file, *args, **kwargs):
        try:
            if str(file) in real_paths:
                real_opened.append(str(file))
        except Exception:
            pass
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded_open)
    yield
    assert real_opened == [], f"hermetic test touched real path(s): {real_opened}"


@pytest.fixture
def fake_key(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_KEY", FAKE_TOKEN)
    return FAKE_TOKEN


@pytest.fixture
def users_file(tmp_path, monkeypatch):
    p = tmp_path / "users.json"
    p.write_text(
        json.dumps(
            [{"id": FAKE_USER_ID, "username": "tester", "last_chat_id": FAKE_CHAT_ID}]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(p))
    return p


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    rd = tmp_path / "runs"
    rd.mkdir()
    monkeypatch.setattr(bot, "RUNS_DIR", rd)
    return rd


@pytest.fixture
def state_file(tmp_path):
    return str(tmp_path / "bot_state.json")


@pytest.fixture
def sends(monkeypatch):
    captured = []
    monkeypatch.setattr(
        tg.TelegramClient, "send_message",
        lambda self, cid, text, **k: captured.append((cid, text)) or {"ok": True},
    )
    return captured


# --------------------------------------------------------------------------- #
# Helpers to build run dirs
# --------------------------------------------------------------------------- #
def _make_run(runs_dir, run_id, *, events=None, meta=None):
    d = runs_dir / run_id
    d.mkdir()
    (d / "events.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in (events or [])),
        encoding="utf-8",
    )
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def _step_started(run_id, idx=1, total=2, title="do x"):
    return {
        "ts": 1.0, "run_id": run_id, "worker": "codex", "kind": "lifecycle",
        "data": {"orchestration": {"phase": "step", "action": "started",
                                   "step_index": idx, "step_total": total,
                                   "step_title": title}},
    }


def _lifecycle_started(run_id):
    return {"ts": 0.0, "run_id": run_id, "kind": "lifecycle",
            "data": {"event": "dispatch_started"}}


def _make_daemon(client=None):
    return bot.BotDaemon(client=client or tg.TelegramClient())


# --------------------------------------------------------------------------- #
# Live-run discovery
# --------------------------------------------------------------------------- #
def test_is_live_run_events_present_no_meta(runs_dir):
    d = _make_run(runs_dir, "20260602-000001", events=[_lifecycle_started("r")])
    assert bot.is_live_run(d) is True


def test_is_live_run_false_when_finished(runs_dir):
    d = _make_run(runs_dir, "20260602-000002",
                  events=[_lifecycle_started("r")], meta={"success": True})
    assert bot.is_live_run(d) is False


def test_is_live_run_false_without_events(runs_dir):
    d = runs_dir / "20260602-000003"
    d.mkdir()  # no events.jsonl at all
    assert bot.is_live_run(d) is False


def test_latest_live_run_picks_newest(runs_dir):
    _make_run(runs_dir, "20260602-000001", events=[_lifecycle_started("a")])
    _make_run(runs_dir, "20260602-000009", events=[_lifecycle_started("b")])
    # A finished run must be ignored even if it sorts newest.
    _make_run(runs_dir, "20260602-000099",
              events=[_lifecycle_started("c")], meta={"success": True})
    latest = bot.latest_live_run()
    assert latest is not None and latest.name == "20260602-000009"


# --------------------------------------------------------------------------- #
# /track command
# --------------------------------------------------------------------------- #
def test_track_latest_selects_newest(runs_dir, state_file):
    _make_run(runs_dir, "20260602-000001", events=[_lifecycle_started("a")])
    _make_run(runs_dir, "20260602-000005", events=[_lifecycle_started("b")])
    state = {}
    msg = bot.handle_command("/track latest", state=state, state_file=state_file,
                             chat_id=FAKE_CHAT_ID)
    assert "Tracking 1 live run" in msg
    tracked = state["tracking"][str(FAKE_CHAT_ID)]
    assert list(tracked.keys()) == ["20260602-000005"]


def test_track_default_is_latest(runs_dir, state_file):
    _make_run(runs_dir, "20260602-000005", events=[_lifecycle_started("b")])
    state = {}
    msg = bot.handle_command("/track", state=state, state_file=state_file,
                             chat_id=FAKE_CHAT_ID)
    assert "Tracking 1 live run" in msg
    assert "20260602-000005" in state["tracking"][str(FAKE_CHAT_ID)]


def test_track_all_tracks_multiple(runs_dir, state_file):
    _make_run(runs_dir, "20260602-000001", events=[_lifecycle_started("a")])
    _make_run(runs_dir, "20260602-000002", events=[_lifecycle_started("b")])
    _make_run(runs_dir, "20260602-000003",
              events=[_lifecycle_started("c")], meta={"success": True})  # finished
    state = {}
    msg = bot.handle_command("/track all", state=state, state_file=state_file,
                             chat_id=FAKE_CHAT_ID)
    assert "Tracking 2 live runs" in msg
    tracked = state["tracking"][str(FAKE_CHAT_ID)]
    assert set(tracked.keys()) == {"20260602-000001", "20260602-000002"}


def test_showliveall_alias(runs_dir, state_file):
    _make_run(runs_dir, "20260602-000001", events=[_lifecycle_started("a")])
    _make_run(runs_dir, "20260602-000002", events=[_lifecycle_started("b")])
    state = {}
    msg = bot.handle_command("/showliveall", state=state, state_file=state_file,
                             chat_id=FAKE_CHAT_ID)
    assert "Tracking 2 live runs" in msg


def test_track_no_live_runs(runs_dir, state_file):
    msg = bot.handle_command("/track latest", state={}, state_file=state_file,
                             chat_id=FAKE_CHAT_ID)
    assert "No live runs" in msg


def test_track_explicit_run_id(runs_dir, state_file):
    _make_run(runs_dir, "20260602-000007", events=[_lifecycle_started("a")])
    state = {}
    msg = bot.handle_command("/track 20260602-000007", state=state,
                             state_file=state_file, chat_id=FAKE_CHAT_ID)
    assert "Tracking 1 live run" in msg
    assert "20260602-000007" in state["tracking"][str(FAKE_CHAT_ID)]


def test_track_unknown_run_id(runs_dir, state_file):
    msg = bot.handle_command("/track nope-123", state={}, state_file=state_file,
                             chat_id=FAKE_CHAT_ID)
    assert "Unknown run" in msg


# --------------------------------------------------------------------------- #
# Tailing: only NEW lines per poll, cursor advances
# --------------------------------------------------------------------------- #
def test_tail_emits_only_new_lines(runs_dir, fake_key, sends, state_file):
    rid = "20260602-000010"
    d = _make_run(runs_dir, rid, events=[_step_started(rid, 1, 2, "first")])
    daemon = _make_daemon()
    daemon.state_file = state_file
    # Track it.
    state = bot.load_state(state_file)
    bot._chat_tracking(state, FAKE_CHAT_ID)[rid] = 0
    bot.save_state(state, state_file)

    n1 = daemon.tail_tracked()
    assert n1 == 1
    assert any("first" in t for _, t in sends)
    before = len(sends)

    # Second poll with no new lines -> nothing re-sent.
    n2 = daemon.tail_tracked()
    assert n2 == 0
    assert len(sends) == before

    # Append a new event -> only the new one is sent.
    with (d / "events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_step_started(rid, 2, 2, "second")) + "\n")
    n3 = daemon.tail_tracked()
    assert n3 == 1
    assert any("second" in t for _, t in sends)
    # The cursor advanced (no re-send of "first").
    assert sum(1 for _, t in sends if "first" in t) == 1


def test_tail_messages_carry_run_tag(runs_dir, fake_key, sends, state_file):
    rid = "20260602-abcdef"
    _make_run(runs_dir, rid, events=[_step_started(rid, 1, 2, "tagme")])
    daemon = _make_daemon()
    daemon.state_file = state_file
    state = bot.load_state(state_file)
    bot._chat_tracking(state, FAKE_CHAT_ID)[rid] = 0
    bot.save_state(state, state_file)
    daemon.tail_tracked()
    # Short tag = last 6 of the run id.
    assert any("abcdef" in t for _, t in sends)


def test_tail_multiple_runs_distinguishable(runs_dir, fake_key, sends, state_file):
    r1, r2 = "20260602-aaaaaa", "20260602-bbbbbb"
    _make_run(runs_dir, r1, events=[_step_started(r1, 1, 1, "alpha")])
    _make_run(runs_dir, r2, events=[_step_started(r2, 1, 1, "beta")])
    daemon = _make_daemon()
    daemon.state_file = state_file
    state = bot.load_state(state_file)
    entry = bot._chat_tracking(state, FAKE_CHAT_ID)
    entry[r1] = 0
    entry[r2] = 0
    bot.save_state(state, state_file)
    daemon.tail_tracked()
    texts = [t for _, t in sends]
    assert any("aaaaaa" in t and "alpha" in t for t in texts)
    assert any("bbbbbb" in t and "beta" in t for t in texts)


# --------------------------------------------------------------------------- #
# Auto-untrack + final summary when meta.json appears
# --------------------------------------------------------------------------- #
def test_auto_untrack_and_final_summary(runs_dir, fake_key, sends, state_file):
    rid = "20260602-000020"
    d = _make_run(runs_dir, rid, events=[_step_started(rid, 1, 1, "x")])
    daemon = _make_daemon()
    daemon.state_file = state_file
    state = bot.load_state(state_file)
    bot._chat_tracking(state, FAKE_CHAT_ID)[rid] = 0
    bot.save_state(state, state_file)

    daemon.tail_tracked()  # drains the live event
    # Run finishes: meta.json appears.
    (d / "meta.json").write_text(json.dumps({
        "success": True, "mode": "master", "duration_s": 12,
        "run_id": rid, "changed_files": ["a.py"],
        "quality": {"confidence": "verified"},
    }), encoding="utf-8")

    n = daemon.tail_tracked()
    assert n >= 1
    # A final summary card went out.
    assert any("Build verified" in t or "Build complete" in t for _, t in sends)
    # And the run was auto-untracked.
    state2 = bot.load_state(state_file)
    assert rid not in state2["tracking"].get(str(FAKE_CHAT_ID), {})


# --------------------------------------------------------------------------- #
# /untrack
# --------------------------------------------------------------------------- #
def test_untrack_all_clears(runs_dir, state_file):
    _make_run(runs_dir, "20260602-000030", events=[_lifecycle_started("a")])
    _make_run(runs_dir, "20260602-000031", events=[_lifecycle_started("b")])
    state = {}
    bot.handle_command("/track all", state=state, state_file=state_file,
                       chat_id=FAKE_CHAT_ID)
    assert state["tracking"][str(FAKE_CHAT_ID)]
    msg = bot.handle_command("/untrack", state=state, state_file=state_file,
                             chat_id=FAKE_CHAT_ID)
    assert "Stopped tracking all" in msg
    assert state["tracking"][str(FAKE_CHAT_ID)] == {}


def test_untrack_one_by_short_tag(runs_dir, state_file):
    r1, r2 = "20260602-aaaaaa", "20260602-bbbbbb"
    _make_run(runs_dir, r1, events=[_lifecycle_started("a")])
    _make_run(runs_dir, r2, events=[_lifecycle_started("b")])
    state = {}
    bot.handle_command("/track all", state=state, state_file=state_file,
                       chat_id=FAKE_CHAT_ID)
    msg = bot.handle_command("/untrack aaaaaa", state=state, state_file=state_file,
                             chat_id=FAKE_CHAT_ID)
    assert "Stopped tracking" in msg
    remaining = state["tracking"][str(FAKE_CHAT_ID)]
    assert r1 not in remaining and r2 in remaining


def test_untrack_not_tracked(state_file):
    msg = bot.handle_command("/untrack whatever", state={}, state_file=state_file,
                             chat_id=FAKE_CHAT_ID)
    assert "Not tracking" in msg


# --------------------------------------------------------------------------- #
# Resilience
# --------------------------------------------------------------------------- #
def test_tail_garbage_events_does_not_crash(runs_dir, fake_key, sends, state_file):
    rid = "20260602-000040"
    d = runs_dir / rid
    d.mkdir()
    (d / "events.jsonl").write_text("{not json\nalso bad}}}\n", encoding="utf-8")
    daemon = _make_daemon()
    daemon.state_file = state_file
    state = bot.load_state(state_file)
    bot._chat_tracking(state, FAKE_CHAT_ID)[rid] = 0
    bot.save_state(state, state_file)
    # Must not raise; garbage lines are skipped.
    daemon.tail_tracked()
    assert all("not json" not in t for _, t in sends)


def test_tail_missing_events_file_does_not_crash(runs_dir, fake_key, sends, state_file):
    rid = "20260602-000041"
    d = runs_dir / rid
    d.mkdir()  # dir exists but NO events.jsonl
    daemon = _make_daemon()
    daemon.state_file = state_file
    state = bot.load_state(state_file)
    bot._chat_tracking(state, FAKE_CHAT_ID)[rid] = 0
    bot.save_state(state, state_file)
    daemon.tail_tracked()  # must not raise


def test_tail_vanished_run_dir_untracks(runs_dir, fake_key, sends, state_file):
    rid = "20260602-000042"  # never created
    daemon = _make_daemon()
    daemon.state_file = state_file
    state = bot.load_state(state_file)
    bot._chat_tracking(state, FAKE_CHAT_ID)[rid] = 0
    bot.save_state(state, state_file)
    daemon.tail_tracked()
    state2 = bot.load_state(state_file)
    assert rid not in state2["tracking"].get(str(FAKE_CHAT_ID), {})


def test_tail_send_failure_does_not_crash(runs_dir, fake_key, monkeypatch, state_file):
    rid = "20260602-000043"
    _make_run(runs_dir, rid, events=[_step_started(rid, 1, 1, "x")])

    def boom(self, *a, **k):
        raise RuntimeError("telegram down")

    monkeypatch.setattr(tg.TelegramClient, "send_message", boom)
    daemon = _make_daemon()
    daemon.state_file = state_file
    state = bot.load_state(state_file)
    bot._chat_tracking(state, FAKE_CHAT_ID)[rid] = 0
    bot.save_state(state, state_file)
    daemon.tail_tracked()  # must not raise even though every send blows up


# --------------------------------------------------------------------------- #
# Whitelist enforcement on /track via the daemon
# --------------------------------------------------------------------------- #
def test_track_rejects_non_whitelisted_user(runs_dir, fake_key, users_file, sends):
    _make_run(runs_dir, "20260602-000050", events=[_lifecycle_started("a")])
    daemon = _make_daemon()
    daemon._process_update({
        "update_id": 1,
        "message": {"from": {"id": 999999}, "chat": {"id": 999999},
                    "text": "/track latest"},
    })
    assert sends == []  # non-whitelisted -> ignored


def test_track_accepts_whitelisted_user(runs_dir, fake_key, users_file, sends,
                                        tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_TELEGRAM_STATE", str(tmp_path / "wl_state.json"))
    _make_run(runs_dir, "20260602-000051", events=[_lifecycle_started("a")])
    daemon = _make_daemon()
    daemon._process_update({
        "update_id": 1,
        "message": {"from": {"id": FAKE_USER_ID}, "chat": {"id": FAKE_CHAT_ID},
                    "text": "/track latest"},
    })
    assert len(sends) == 1
    assert "Tracking 1 live run" in sends[0][1]


# --------------------------------------------------------------------------- #
# poll_once integration: tailing runs each iteration
# --------------------------------------------------------------------------- #
def test_poll_once_tails_tracked_runs(runs_dir, fake_key, sends, state_file, monkeypatch):
    rid = "20260602-000060"
    _make_run(runs_dir, rid, events=[_step_started(rid, 1, 1, "polled")])
    monkeypatch.setattr(
        tg.TelegramClient, "get_updates", lambda self, offset=None, timeout=25: [])
    daemon = _make_daemon()
    daemon.state_file = state_file
    state = bot.load_state(state_file)
    bot._chat_tracking(state, FAKE_CHAT_ID)[rid] = 0
    bot.save_state(state, state_file)
    daemon.poll_once()
    assert any("polled" in t for _, t in sends)

"""Acceptance CONTRACT for Telegram fast-follow F2 — /mute, /watch, quiet-hours.

Anti-no-op gate for the per-chat suppression feature (spec §F2). Asserts the
NEW observable surface + behavior, designed so it FAILS on the current code:

  * the public suppression seam exists: ``telegram._suppressed`` and the window
    parser ``telegram.parse_quiet_window``;
  * a MUTED chat suppresses progress chatter (a step-started message) but NEVER
    a build-failed or summary message;
  * inside a QUIET window the same policy holds (progress dropped, failure +
    summary always delivered);
  * ``/quiet 23:00-07:00`` parses + persists into ``state["chats"][cid]``;
  * a garbage quiet window is rejected WITHOUT raising;
  * ``_suppressed`` never raises — on garbage chat/state/now it returns False
    (deliver, fail-open) rather than blowing up the delivery path.

Time-of-day logic is deterministic: the helper accepts an injected ``now`` so
the test never touches the wall clock in an assertion path.

Hermetic: fake token + fake ids, _post captured, state pinned to tmp_path.
"""
from __future__ import annotations

import datetime as _dt

import pytest

from harness import telegram as tg
from harness import telegram_bot as bot


FAKE_TOKEN = "123456:FAKE-TEST-TOKEN-not-real"
FAKE_CHAT_ID = 444555666


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_TELEGRAM_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(tmp_path / "users.json"))
    monkeypatch.setenv("TELEGRAM_BOT_KEY", FAKE_TOKEN)


def _at(hh, mm=0):
    """A deterministic datetime at HH:MM (date is irrelevant to the window logic)."""
    return _dt.datetime(2026, 6, 3, hh, mm, 0)


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #
def test_suppression_surface_exists():
    assert hasattr(tg, "_suppressed"), "telegram._suppressed() missing"
    assert hasattr(tg, "parse_quiet_window"), "telegram.parse_quiet_window() missing"


# --------------------------------------------------------------------------- #
# Quiet-window parsing
# --------------------------------------------------------------------------- #
def test_quiet_window_parses_valid():
    win = tg.parse_quiet_window("23:00-07:00")
    assert win is not None, "valid window returned None"
    # canonical (start, end) in minutes-of-day: 23:00 -> 1380, 07:00 -> 420
    start, end = win
    assert start == 23 * 60
    assert end == 7 * 60


def test_quiet_window_rejects_garbage_without_raising():
    for junk in ("", "nonsense", "25:00-26:00", "23:00", "23:00-", "abc-def", "::-::"):
        # must NOT raise and must signal rejection (None / falsey)
        assert not tg.parse_quiet_window(junk), f"garbage window {junk!r} accepted"


# --------------------------------------------------------------------------- #
# /quiet command parses + persists into the per-chat prefs
# --------------------------------------------------------------------------- #
def test_quiet_command_persists_window():
    state = {}
    reply = bot.handle_command(
        "/quiet 23:00-07:00", state=state, state_file=None, chat_id=FAKE_CHAT_ID
    )
    assert reply  # acknowledges
    chats = state.get("chats") or {}
    prefs = chats.get(str(FAKE_CHAT_ID)) or {}
    assert prefs.get("quiet_window"), "quiet_window not persisted under chats[cid]"
    # the persisted window round-trips back through the parser
    assert tg.parse_quiet_window(prefs["quiet_window"]) is not None


def test_quiet_off_clears_window():
    state = {}
    bot.handle_command("/quiet 23:00-07:00", state=state, state_file=None, chat_id=FAKE_CHAT_ID)
    bot.handle_command("/quiet off", state=state, state_file=None, chat_id=FAKE_CHAT_ID)
    prefs = (state.get("chats") or {}).get(str(FAKE_CHAT_ID)) or {}
    assert not prefs.get("quiet_window"), "/quiet off did not clear the window"


def test_quiet_command_rejects_garbage_without_raising():
    state = {}
    reply = bot.handle_command(
        "/quiet not-a-window", state=state, state_file=None, chat_id=FAKE_CHAT_ID
    )
    assert reply  # responds with an error, does not raise
    prefs = (state.get("chats") or {}).get(str(FAKE_CHAT_ID)) or {}
    assert not prefs.get("quiet_window"), "garbage window should not persist"


# --------------------------------------------------------------------------- #
# /mute + /watch
# --------------------------------------------------------------------------- #
def test_mute_then_watch_round_trips():
    state = {}
    bot.handle_command("/mute on", state=state, state_file=None, chat_id=FAKE_CHAT_ID)
    prefs = (state.get("chats") or {}).get(str(FAKE_CHAT_ID)) or {}
    assert prefs.get("mute_until"), "/mute did not set mute_until"
    bot.handle_command("/watch", state=state, state_file=None, chat_id=FAKE_CHAT_ID)
    prefs = (state.get("chats") or {}).get(str(FAKE_CHAT_ID)) or {}
    assert not prefs.get("mute_until"), "/watch did not clear mute_until"


# --------------------------------------------------------------------------- #
# Core policy: muted chat drops PROGRESS but NEVER failure/summary
# --------------------------------------------------------------------------- #
def test_muted_suppresses_progress_but_not_failure_or_summary():
    state = {"chats": {str(FAKE_CHAT_ID): {"mute_until": "on"}}}
    # progress chatter is suppressed while muted
    assert tg._suppressed(FAKE_CHAT_ID, "progress", state) is True
    # failures + the final summary are ALWAYS delivered
    assert tg._suppressed(FAKE_CHAT_ID, "failure", state) is False
    assert tg._suppressed(FAKE_CHAT_ID, "summary", state) is False


def test_unmuted_chat_suppresses_nothing():
    state = {"chats": {str(FAKE_CHAT_ID): {}}}
    assert tg._suppressed(FAKE_CHAT_ID, "progress", state) is False
    assert tg._suppressed(FAKE_CHAT_ID, "failure", state) is False
    assert tg._suppressed(FAKE_CHAT_ID, "summary", state) is False


def test_mute_expires_with_injected_now():
    # mute_until is an epoch in the past -> no longer muted
    state = {"chats": {str(FAKE_CHAT_ID): {"mute_until": 1000.0}}}
    assert tg._suppressed(FAKE_CHAT_ID, "progress", state, now=2000.0) is False
    # ... but still in the future -> muted
    state2 = {"chats": {str(FAKE_CHAT_ID): {"mute_until": 5000.0}}}
    assert tg._suppressed(FAKE_CHAT_ID, "progress", state2, now=2000.0) is True


# --------------------------------------------------------------------------- #
# Core policy: quiet window drops PROGRESS but NEVER failure/summary
# --------------------------------------------------------------------------- #
def test_quiet_window_suppresses_progress_only():
    state = {"chats": {str(FAKE_CHAT_ID): {"quiet_window": "23:00-07:00"}}}
    # 02:00 is inside the overnight window -> progress dropped
    assert tg._suppressed(FAKE_CHAT_ID, "progress", state, now=_at(2)) is True
    # but failure + summary still go out
    assert tg._suppressed(FAKE_CHAT_ID, "failure", state, now=_at(2)) is False
    assert tg._suppressed(FAKE_CHAT_ID, "summary", state, now=_at(2)) is False
    # 12:00 is OUTSIDE the window -> progress delivered
    assert tg._suppressed(FAKE_CHAT_ID, "progress", state, now=_at(12)) is False


# --------------------------------------------------------------------------- #
# Resilience: the suppression read never raises
# --------------------------------------------------------------------------- #
def test_suppressed_never_raises_on_garbage():
    for chat in (None, "x", 42, object()):
        for st in (None, 42, "x", {}, {"chats": "nope"}, {"chats": {str(FAKE_CHAT_ID): "bad"}}):
            for nw in (None, "notnum", object(), -1):
                # must never raise; fail-open default is "deliver" (False)
                tg._suppressed(chat, "progress", st, now=nw)

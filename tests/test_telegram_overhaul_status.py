"""Acceptance CONTRACT for the Telegram overhaul — Pillar B (live status card).

Anti-no-op gate for the sink-side BuildState machine + the edit-in-place pinned
"live status" message. Drives events through the real notifier seam and asserts
OBSERVABLE behavior (a card is sent, then EDITED in place via editMessageText
reusing the sendMessage message_id) plus the derived build-state (step position,
progress, honest ETA) and the compaction guard. Loose on exact wording, strict
on: the public surface (BuildState + render_status_card), the reference point,
edit-in-place, ETA honesty, and that compaction never corrupts step geometry.

Hermetic: fake token + fake ids, _post captured, state pinned to tmp_path.
"""
from __future__ import annotations

import re

import pytest

from harness import telegram as tg


FAKE_TOKEN = "123456:FAKE-TEST-TOKEN-not-real"
FAKE_CHAT_ID = 444555666


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_TELEGRAM_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(tmp_path / "users.json"))
    monkeypatch.setenv("TELEGRAM_BOT_KEY", FAKE_TOKEN)


@pytest.fixture
def posts(monkeypatch):
    """Capture every API call; sendMessage returns an incrementing message_id."""
    sent = []
    state = {"mid": 1000}

    def fake_post(self, method, params, *, timeout=None):
        sent.append({"method": method, "params": dict(params)})
        if method == "sendMessage":
            state["mid"] += 1
            return {"ok": True, "result": {"message_id": state["mid"]}}
        if method == "getUpdates":
            return {"ok": True, "result": []}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(tg.TelegramClient, "_post", fake_post)
    return sent


def _notifier():
    return tg.TelegramNotifier(
        run_id="a3f9c1", mode="master", verbosity="normal",
        client=tg.TelegramClient(), chat_ids=[FAKE_CHAT_ID],
    )


def _ev(data, *, ts, **top):
    e = {"kind": "lifecycle", "run_id": "a3f9c1", "ts": ts, "data": data}
    e.update(top)
    return e


def _orch(orch, *, ts):
    return _ev({"event": "orchestration_transition", "orchestration": orch}, ts=ts)


def _feed_through_step2(n, t0=1000.0):
    """dispatch_started -> plan(4) -> step1 started+done(verified) -> step2 started.
    Returns the timeline end ts."""
    n(_ev({"event": "dispatch_started",
           "detail": {"mode": "master", "generator_chain": ["codex"], "critic_chain": ["agy"]}}, ts=t0))
    n(_orch({"workflow": "master", "phase": "plan", "action": "completed",
             "step_total": 4, "step_titles": ["A", "B", "C", "D"]}, ts=t0 + 1))
    n(_orch({"phase": "step", "action": "started", "step_index": 1, "step_total": 4,
             "step_title": "A"}, ts=t0 + 2))
    n(_orch({"phase": "step", "action": "completed", "step_index": 1, "step_total": 4,
             "step_title": "A", "outcome": "verified", "verified": True}, ts=t0 + 122))
    n(_orch({"phase": "step", "action": "started", "step_index": 2, "step_total": 4,
             "step_title": "B"}, ts=t0 + 123))
    return t0 + 123


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #
def test_buildstate_and_render_status_card_exist():
    assert hasattr(tg, "BuildState"), "BuildState class missing"
    assert hasattr(tg, "render_status_card"), "render_status_card() missing"


# --------------------------------------------------------------------------- #
# Build-state derivation: position + counts from the event stream alone
# --------------------------------------------------------------------------- #
def test_notifier_tracks_step_position_and_counts(posts):
    n = _notifier()
    _feed_through_step2(n)
    st = n._state
    assert st.step_total == 4
    assert st.cur_step_index == 2
    assert st.n_steps_done == 1


def test_status_card_shows_reference_point_and_progress_bar(posts):
    n = _notifier()
    _feed_through_step2(n)
    card = tg.render_status_card(n._state)
    assert "2/4" in card                       # the reference point
    # a progress bar made of block/▰▱-style cells
    assert re.search(r"[█▰▓░▱▮▯]", card), "no progress-bar glyph in status card"


def test_status_card_uses_real_newlines_not_literal_backslash_n(posts):
    # Guard against the f-string "\\n" escaping bug: the multi-line card MUST use
    # real newlines, else Telegram shows one line with visible "\n" text.
    n = _notifier()
    _feed_through_step2(n)
    card = tg.render_status_card(n._state)
    assert "\n" in card, "status card is not multi-line (no real newline)"
    assert "\\n" not in card, "status card contains a LITERAL backslash-n (escaping bug)"


# --------------------------------------------------------------------------- #
# Edit-in-place: the pinned card is sent once then EDITED as state advances
# --------------------------------------------------------------------------- #
def test_pinned_card_is_sent_then_edited_in_place(posts):
    n = _notifier()
    _feed_through_step2(n)
    n.flush(timeout=5.0)
    sends = [p for p in posts if p["method"] == "sendMessage"]
    edits = [p for p in posts if p["method"] == "editMessageText"]
    assert sends, "no sendMessage at all"
    assert edits, "status card never edited in place (editMessageText not used)"
    # the edit targets a message_id that a prior sendMessage returned (1001..100N)
    edited_ids = {p["params"].get("message_id") for p in edits}
    assert any(isinstance(m, int) and m >= 1001 for m in edited_ids)


# --------------------------------------------------------------------------- #
# ETA honesty: no numeric ETA before a step completes; estimate only once earned
# --------------------------------------------------------------------------- #
def test_eta_is_honest_before_any_step_completes(posts):
    n = _notifier()
    t0 = 1000.0
    n(_ev({"event": "dispatch_started",
           "detail": {"mode": "master", "generator_chain": ["codex"], "critic_chain": ["agy"]}}, ts=t0))
    n(_orch({"workflow": "master", "phase": "plan", "action": "completed",
             "step_total": 4, "step_titles": ["A", "B", "C", "D"]}, ts=t0 + 1))
    n(_orch({"phase": "step", "action": "started", "step_index": 1, "step_total": 4,
             "step_title": "A"}, ts=t0 + 2))
    card = tg.render_status_card(n._state)
    # No fabricated "~Nm left" before any duration is observed.
    assert not re.search(r"~\s*\d+\s*m", card), "ETA invented before any step completed"


def test_eta_appears_after_a_step_completes(posts):
    n = _notifier()
    _feed_through_step2(n)            # one step completed in ~120s
    card = tg.render_status_card(n._state)
    # Some honest estimate is now shown (prefixed ~, or an explicit ETA/left label).
    assert ("~" in card) or ("eta" in card.lower()) or ("left" in card.lower())


# --------------------------------------------------------------------------- #
# Compaction guard: a compact-role spin-up must NOT corrupt step geometry
# --------------------------------------------------------------------------- #
def test_compaction_does_not_reset_step_geometry(posts):
    n = _notifier()
    end = _feed_through_step2(n)
    before = n._state.cur_step_index
    # a context-compaction worker spins up between steps (role=compact)
    n(_ev({"event": "agent_started", "detail": {"role": "compact"}},
          ts=end + 1, worker="codex", model="m", effort="low"))
    assert n._state.cur_step_index == before == 2


# --------------------------------------------------------------------------- #
# Resilience: status path never raises, even on garbage
# --------------------------------------------------------------------------- #
def test_status_path_never_raises_on_garbage(posts):
    n = _notifier()
    for g in (None, 42, "x", {"kind": "lifecycle"}, {"data": {"orchestration": "no"}},
              {"ts": "notnum", "data": {"event": "dispatch_started"}}):
        n(g)  # must not raise
    n.flush(timeout=5.0)


def test_finished_renders_terminal_status(posts):
    n = _notifier()
    _feed_through_step2(n)
    n.finished({"success": True, "mode": "master", "duration_s": 600,
                "changed_files": ["a.py"], "quality": {"confidence": "verified"},
                "tokens": {"grand_total": {"total_tokens": 1000}}})
    n.flush(timeout=5.0)
    # the pinned card is finalized via a last edit (or a final send) — either way
    # at least one edit happened over the run.
    assert any(p["method"] == "editMessageText" for p in posts)

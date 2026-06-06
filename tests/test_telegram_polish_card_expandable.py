"""CONTRACT (RED-first) for the Telegram "Card + expandable detail" polish.

This file pins the NEW polished rendering chosen by the operator: an emoji-led,
compact, always-visible CARD for at-a-glance state, with secondary/long detail
(recent activity, the full step list, changed-files, the per-step recap) TUCKED
into a Telegram ``<blockquote expandable>…</blockquote>`` so each message stays
short until the operator taps to expand.

It is written BEFORE the implementation and is expected to FAIL on the current
(un-polished) code — the three render sites in ``harness/telegram.py`` do not yet
emit ``<blockquote expandable>`` blocks.

Scope discipline (do NOT regress the existing semantic contracts):
* These tests pin the NEW *formatting* (the expandable, the compact header split,
  HTML validity, emoji retention). They deliberately do NOT re-assert verbosity
  gating, glyph semantics, or which-event-surfaces-when — those live in the
  existing telegram test files and must keep passing unchanged.

Hermetic: fake token + fake ids, ``_post`` monkeypatched, state pinned to
tmp_path. Mirrors the fixtures/style of tests/test_telegram_overhaul_status.py.
"""
from __future__ import annotations

import re

import pytest

from harness import telegram as tg


FAKE_TOKEN = "123456:FAKE-TEST-TOKEN-not-real"
FAKE_CHAT_ID = 444555666

# A Telegram-HTML expandable opening tag, tolerant of the optional cite/extra
# attributes Telegram permits (``<blockquote expandable>`` is the canonical form).
_EXPANDABLE_OPEN = re.compile(r"<blockquote\s+expandable[^>]*>")
_EXPANDABLE_BLOCK = re.compile(
    r"<blockquote\s+expandable[^>]*>(.*?)</blockquote>", re.DOTALL
)


# --------------------------------------------------------------------------- #
# Hermetic fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_TELEGRAM_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(tmp_path / "users.json"))
    monkeypatch.setenv("TELEGRAM_BOT_KEY", FAKE_TOKEN)


@pytest.fixture
def posts(monkeypatch):
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


GOAL = "Add OAuth device flow to the login service"


def _notifier(goal=GOAL):
    return tg.TelegramNotifier(
        run_id="a3f9c1", mode="master", verbosity="normal",
        client=tg.TelegramClient(), chat_ids=[FAKE_CHAT_ID],
        instruction=goal,
    )


def _ev(data, *, ts, **top):
    e = {"kind": "lifecycle", "run_id": "a3f9c1", "ts": ts, "data": data}
    e.update(top)
    return e


def _orch(orch, *, ts):
    return _ev({"event": "orchestration_transition", "orchestration": orch}, ts=ts)


def _feed_through_step2(n, t0=1000.0):
    """dispatch_started -> plan(4) -> step1 done(verified) -> step2 started."""
    n(_ev({"event": "dispatch_started",
           "detail": {"mode": "master", "generator_chain": ["codex"],
                      "critic_chain": ["agy"]}}, ts=t0))
    n(_orch({"workflow": "master", "phase": "plan", "action": "completed",
             "step_total": 4, "step_titles": ["A", "B", "C", "D"]}, ts=t0 + 1))
    n(_orch({"phase": "step", "action": "started", "step_index": 1,
             "step_total": 4, "step_title": "A"}, ts=t0 + 2))
    n(_orch({"phase": "step", "action": "completed", "step_index": 1,
             "step_total": 4, "step_title": "A",
             "outcome": "verified", "verified": True}, ts=t0 + 122))
    n(_orch({"phase": "step", "action": "started", "step_index": 2,
             "step_total": 4, "step_title": "B"}, ts=t0 + 123))
    return t0 + 123


# --------------------------------------------------------------------------- #
# HTML-validity helper (shared by the validity test)
# --------------------------------------------------------------------------- #
_ALLOWED_TAGS = {
    "b", "i", "u", "s", "code", "pre", "a", "blockquote", "tg-spoiler",
}
_FORBIDDEN_TAGS = {"br", "div", "span", "tg-emoji"}

_TAG_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9-]*)(\s[^>]*)?>")


def _assert_html_valid(text: str):
    """Scan ``text`` for balanced/closed allowed tags and reject disallowed ones.

    * Every tag name must be in the Telegram-allowed set (``blockquote
      expandable`` parses as the ``blockquote`` tag here).
    * Opening/closing tags must nest like a stack (balanced + properly closed).
    * No ``<blockquote>`` may be nested inside another ``<blockquote>`` (Telegram
      forbids nested blockquotes).
    """
    stack = []
    for m in _TAG_RE.finditer(text):
        closing = bool(m.group(1))
        name = m.group(2).lower()
        assert name not in _FORBIDDEN_TAGS, f"forbidden tag <{name}> in: {text!r}"
        assert name in _ALLOWED_TAGS, f"non-allowed tag <{name}> in: {text!r}"
        if closing:
            assert stack, f"closing </{name}> with empty stack in: {text!r}"
            top = stack.pop()
            assert top == name, f"mismatched </{name}> (open <{top}>) in: {text!r}"
        else:
            if name == "blockquote":
                assert "blockquote" not in stack, (
                    f"nested <blockquote> not allowed in: {text!r}"
                )
            stack.append(name)
    assert not stack, f"unclosed tags {stack} in: {text!r}"


def _expandable_inner(text: str) -> str:
    m = _EXPANDABLE_BLOCK.search(text)
    assert m is not None, f"no <blockquote expandable> block in: {text!r}"
    return m.group(1)


def _outside_expandable(text: str) -> str:
    """The card text with every expandable block stripped (the always-visible part)."""
    return _EXPANDABLE_BLOCK.sub("", text)


# --------------------------------------------------------------------------- #
# 1. render_status_card: compact header always visible + detail in expandable
# --------------------------------------------------------------------------- #
def test_status_card_header_compact_and_detail_in_expandable(posts):
    n = _notifier()
    _feed_through_step2(n)
    st = n._state
    # A recognizable "recent activity" detail to look for inside the expandable.
    st.last_event_line = "Draft iter 2 codex drafting endpoints handler"

    card = tg.render_status_card(st)

    # --- compact header: always-visible, OUTSIDE any expandable -------------- #
    head = _outside_expandable(card)
    assert "2/4" in head, "step reference point missing from the always-visible header"
    assert re.search(r"[█▰▓░▱▮▯]", head), "progress bar missing from the visible header"
    assert ("elapsed" in head.lower()) or re.search(r"\d+m\d", head), \
        "elapsed/time missing from the visible header"

    # --- secondary detail tucked into a <blockquote expandable> -------------- #
    assert _EXPANDABLE_OPEN.search(card), \
        "status card does not use <blockquote expandable> for secondary detail"
    inner = _expandable_inner(card)
    # the recent-activity / last-event detail lives INSIDE the expandable, not
    # on the always-visible header.
    assert "endpoints handler" in inner, \
        "recent-activity detail not tucked inside the expandable"
    assert "endpoints handler" not in head, \
        "recent-activity detail leaked onto the always-visible header"


# --------------------------------------------------------------------------- #
# 2. render_status_card: no EMPTY expandable when there is no secondary detail
# --------------------------------------------------------------------------- #
def test_status_card_no_empty_expandable_when_no_detail(posts):
    n = _notifier(goal=None)
    # Right after dispatch start: no completed steps, no recent-activity line.
    n(_ev({"event": "dispatch_started",
           "detail": {"mode": "master", "generator_chain": ["codex"]}}, ts=1000.0))
    st = n._state
    st.last_event_line = ""  # nothing to tuck

    card = tg.render_status_card(st)

    # If an expandable is emitted at all, it must carry real content — never an
    # empty / whitespace-only shell.
    for inner in _EXPANDABLE_BLOCK.findall(card):
        assert inner.strip(), f"emitted an EMPTY <blockquote expandable> in: {card!r}"
    # The blunt literal-empty form must never appear.
    assert "<blockquote expandable></blockquote>" not in card.replace("\n", "")


# --------------------------------------------------------------------------- #
# 3. render_event: PLAN-ready headline visible, step list tucked into expandable
# --------------------------------------------------------------------------- #
def test_plan_event_tucks_step_list_into_expandable():
    titles = [
        "Scaffold module", "Wire the endpoint", "Add migrations",
        "Integration tests", "Docs + changelog",
    ]
    msg = tg.render_event(
        _ev({"event": "orchestration_transition",
             "orchestration": {"workflow": "master", "phase": "plan",
                               "action": "completed", "step_total": len(titles),
                               "step_titles": titles}}, ts=1000.0),
        verbosity="normal", mode="master", run_id="a3f9c1",
    )
    assert msg is not None

    head = _outside_expandable(msg)
    # The "Plan ready · N steps" headline stays OUTSIDE the expandable.
    assert "Plan ready" in head, "plan headline missing from the always-visible part"
    assert str(len(titles)) in head, "step count missing from the plan headline"

    # The full step list is tucked INSIDE the expandable.
    assert _EXPANDABLE_OPEN.search(msg), \
        "plan event does not tuck its step list into <blockquote expandable>"
    inner = _expandable_inner(msg)
    for t in titles:
        assert t in inner, f"step title {t!r} not inside the expandable step list"
    # The step titles should not be duplicated onto the always-visible headline.
    assert "Scaffold module" not in head, "step list leaked onto the visible headline"


# --------------------------------------------------------------------------- #
# 4. _summary_card: headline+stats visible, files + recap tucked into expandable
# --------------------------------------------------------------------------- #
def test_summary_card_tucks_files_and_recap_into_expandable(posts):
    n = _notifier()
    _feed_through_step2(n)
    st = n._state
    # Populate a multi-step recap + a roomy changed-files list deterministically.
    st.step_total = 4
    st.n_steps_done = 4
    st.step_records = [
        {"index": 1, "title": "Schema", "duration": 120, "outcome": "verified", "rounds": 1},
        {"index": 2, "title": "Endpoints", "duration": 134, "outcome": "verified", "rounds": 2},
        {"index": 3, "title": "Tests", "duration": 30, "outcome": "approved", "rounds": 1},
        {"index": 4, "title": "Docs", "duration": 12, "outcome": "approved", "rounds": 1},
    ]
    changed = [f"src/pkg/file_{i}.py" for i in range(12)]

    card = n._summary_card({
        "success": True, "mode": "master", "duration_s": 296,
        "changed_files": changed,
        "quality": {"confidence": "verified"},
        "tokens": {"grand_total": {"total_tokens": 184200}},
    })

    head = _outside_expandable(card)
    # Always-visible: headline + the "N/N steps · status · duration" line.
    assert "4/4 steps" in head, "steps line missing from the always-visible header"
    assert "verified" in head.lower(), "status word missing from the visible header"

    # Tucked: the changed-files list AND the <pre> per-step recap.
    assert _EXPANDABLE_OPEN.search(card), \
        "summary card does not tuck detail into <blockquote expandable>"
    inner = _expandable_inner(card)
    assert "file_0.py" in inner, "changed-files list not inside the expandable"
    assert "<pre>" in inner and "</pre>" in inner, \
        "per-step recap <pre> not inside the expandable"
    for title in ("Schema", "Endpoints", "Tests", "Docs"):
        assert title in inner, f"step '{title}' missing from the recap inside expandable"

    # The big file list must NOT be dumped onto the always-visible header.
    assert "file_0.py" not in head, "changed-files list leaked onto the visible header"


# --------------------------------------------------------------------------- #
# 5. HTML validity across all three render sites
# --------------------------------------------------------------------------- #
def test_status_card_html_is_valid_and_escapes_dynamic_text(posts):
    n = _notifier(goal="patch <b>auth</b> & retry <flow>")
    _feed_through_step2(n)
    st = n._state
    st.cur_step_title = "fix <handler> & <route>"
    st.last_event_line = "drafting <module> & wiring"
    card = tg.render_status_card(st)

    _assert_html_valid(card)
    # dynamic text is escaped (the raw injected markup never appears verbatim).
    assert "<b>auth</b>" not in card, "dynamic goal text was not escaped"
    assert "<handler>" not in card, "dynamic step title was not escaped"
    assert "&lt;" in card, "escaping helper (_e) not applied to dynamic text"


def test_plan_event_html_is_valid(posts):
    titles = ["Scaffold <core>", "Wire & test", "Docs"]
    msg = tg.render_event(
        _ev({"event": "orchestration_transition",
             "orchestration": {"workflow": "master", "phase": "plan",
                               "action": "completed", "step_total": len(titles),
                               "step_titles": titles}}, ts=1000.0),
        verbosity="normal", mode="master", run_id="a3f9c1",
    )
    assert msg is not None
    _assert_html_valid(msg)
    assert "<core>" not in msg, "plan step title not escaped"


def test_summary_card_html_is_valid(posts):
    n = _notifier()
    _feed_through_step2(n)
    st = n._state
    st.step_total = 2
    st.n_steps_done = 2
    st.step_records = [
        {"index": 1, "title": "Schema <x>", "duration": 120, "outcome": "verified", "rounds": 1},
        {"index": 2, "title": "Endpoints & more", "duration": 30, "outcome": "approved", "rounds": 1},
    ]
    card = n._summary_card({
        "success": True, "mode": "master", "duration_s": 150,
        "changed_files": [f"src/file_{i}.py" for i in range(10)],
        "quality": {"confidence": "verified"},
        "tokens": {"grand_total": {"total_tokens": 9000}},
    })
    _assert_html_valid(card)


# --------------------------------------------------------------------------- #
# 6. Emoji retained on all three sites
# --------------------------------------------------------------------------- #
def test_status_card_keeps_lightning_emoji(posts):
    n = _notifier()
    _feed_through_step2(n)
    card = tg.render_status_card(n._state)
    assert "⚡" in card, "live status card lost its ⚡ emoji"


def test_plan_event_keeps_clipboard_emoji():
    msg = tg.render_event(
        _ev({"event": "orchestration_transition",
             "orchestration": {"workflow": "master", "phase": "plan",
                               "action": "completed", "step_total": 2,
                               "step_titles": ["A", "B"]}}, ts=1000.0),
        verbosity="normal", mode="master", run_id="a3f9c1",
    )
    assert msg is not None
    assert "📋" in msg, "plan event lost its 📋 emoji"


def test_summary_card_keeps_outcome_emoji(posts):
    n = _notifier()
    _feed_through_step2(n)
    verified = n._summary_card({
        "success": True, "mode": "master", "duration_s": 100,
        "changed_files": ["a.py"], "quality": {"confidence": "verified"},
        "tokens": {"grand_total": {"total_tokens": 100}},
    })
    failed = n._summary_card({
        "success": False, "mode": "master", "duration_s": 100,
        "error": "verifier red", "changed_files": [], "tokens": {},
    })
    assert any(g in verified for g in ("✅", "☑️")), "summary lost its success emoji"
    assert "🛑" in failed, "failed summary lost its 🛑 emoji"


# --------------------------------------------------------------------------- #
# 7. Length caps: long runs MUST stay well under Telegram's 4096-char limit.
#    The live status card is editMessageText'd in place, so an over-limit render
#    400s ("message is too long") and silently freezes the pinned card.
# --------------------------------------------------------------------------- #
_LONG_TITLE = "Refactor the authentication middleware and wire the device-flow handler"


def _many_records(n):
    return [
        {"index": i, "title": f"{_LONG_TITLE} #{i}",
         "duration": 90 + i, "outcome": "verified", "rounds": 1}
        for i in range(1, n + 1)
    ]


def test_status_card_caps_steps_and_stays_under_telegram_limit(posts):
    n = _notifier()
    _feed_through_step2(n)
    st = n._state
    # A pathological run: far more completed steps than the cap, each with a
    # long title — uncapped this crossed 4096 at ~37 steps and 400'd the edit.
    st.step_records = _many_records(60)
    st.last_event_line = "drafting endpoints handler"

    card = tg.render_status_card(st)

    assert len(card) < 4096, (
        f"live status card crossed Telegram's 4096-char limit ({len(card)} chars) "
        "— editMessageText would 400 and freeze the pinned card"
    )
    inner = _expandable_inner(card)
    # only the most-recent cap-worth of steps are rendered (tail kept)...
    assert f"#{60}" in inner, "most-recent step missing from the live card detail"
    assert f"#{1}" not in inner, "oldest step should be elided, not rendered, on the live card"
    # ...and the elision is surfaced honestly.
    assert "earlier" in inner, "elided-step count not surfaced (no '+K earlier')"
    _assert_html_valid(card)


def test_summary_card_caps_recap_and_stays_under_telegram_limit(posts):
    n = _notifier()
    _feed_through_step2(n)
    st = n._state
    st.step_total = 300
    st.n_steps_done = 300
    st.step_records = _many_records(300)

    card = n._summary_card({
        "success": True, "mode": "master", "duration_s": 9000,
        "changed_files": [f"src/pkg/module_{i}/file_{i}.py" for i in range(60)],
        "quality": {"confidence": "verified"},
        "tokens": {"grand_total": {"total_tokens": 184200}},
    })

    assert len(card) < 4096, (
        f"end-of-run summary crossed Telegram's 4096-char limit ({len(card)} chars)"
    )
    inner = _expandable_inner(card)
    assert "earlier" in inner, "elided step-recap count not surfaced (no '+K earlier')"
    assert "<pre>" in inner and "</pre>" in inner, "recap <pre> dropped"
    _assert_html_valid(card)

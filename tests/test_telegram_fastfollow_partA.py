"""Acceptance CONTRACT for the Telegram fast-follow — Part A (honest gaps).

Anti-no-op gate for G1/G2/G3 (docs/telegram-fastfollow-spec.md §"Part A"):

  G1  GOAL IN HEADER + STATUS CARD — the dispatch instruction is threaded to the
      notifier (BuildState.goal) and rendered as the bold lead of the header card
      AND the <i>title</i> lead of the status card. Absent goal => no crash, no
      empty bold line.
  G2  STEP-DONE DURATION + ROUNDS — the step-completed line carries the step's
      wall-clock duration + adversarial round count, e.g.
      "✅ Step 3/8 done · <i>title</i> · 2m14s · 2 rounds · verified".
  G3  RICH END-OF-RUN SUMMARY CARD — finished()/_summary_card renders a multi-line
      card: headline (✅ verified / ☑️ complete / 🛑 failed), the goal,
      "N/N steps · status · duration", a changed-files block (first ~8 + "… +K
      more"), a <pre> per-step recap (one line per step), tokens + mode. REAL
      newlines, never the literal two-char backslash-n.

Hermetic: fake token + fake ids, _post captured (no network), state pinned to
tmp_path. Mirrors tests/test_telegram_overhaul_status.py.
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


def _feed_three_steps(n, t0=1000.0):
    """dispatch_started -> plan(3) -> 3 steps, step 2 with two adversarial rounds.
    Step 1 spans 120s, step 2 spans 134s w/ 2 iterations, step 3 spans 30s."""
    n(_ev({"event": "dispatch_started",
           "detail": {"mode": "master", "generator_chain": ["codex"], "critic_chain": ["agy"]}}, ts=t0))
    n(_orch({"workflow": "master", "phase": "plan", "action": "completed",
             "step_total": 3, "step_titles": ["Schema", "Endpoints", "Tests"]}, ts=t0 + 1))
    # step 1
    n(_orch({"phase": "step", "action": "started", "step_index": 1, "step_total": 3,
             "step_title": "Schema"}, ts=t0 + 2))
    n(_orch({"phase": "step", "action": "completed", "step_index": 1, "step_total": 3,
             "step_title": "Schema", "outcome": "verified", "verified": True}, ts=t0 + 122))
    # step 2 — two adversarial iterations
    n(_orch({"phase": "step", "action": "started", "step_index": 2, "step_total": 3,
             "step_title": "Endpoints"}, ts=t0 + 123))
    n(_orch({"phase": "adversarial", "action": "iteration_started", "iteration": 1,
             "iteration_total": 5}, ts=t0 + 124))
    n(_orch({"phase": "adversarial", "action": "iteration_started", "iteration": 2,
             "iteration_total": 5}, ts=t0 + 200))
    n(_orch({"phase": "step", "action": "completed", "step_index": 2, "step_total": 3,
             "step_title": "Endpoints", "outcome": "verified", "verified": True}, ts=t0 + 257))
    # step 3
    n(_orch({"phase": "step", "action": "started", "step_index": 3, "step_total": 3,
             "step_title": "Tests"}, ts=t0 + 258))
    n(_orch({"phase": "step", "action": "completed", "step_index": 3, "step_total": 3,
             "step_title": "Tests", "outcome": "approved", "approved": True}, ts=t0 + 288))
    return t0 + 288


# --------------------------------------------------------------------------- #
# G1 — goal in header + status card
# --------------------------------------------------------------------------- #
def test_buildstate_has_goal_field(posts):
    st = tg.BuildState("a3f9c1", "master")
    assert hasattr(st, "goal"), "BuildState.goal field missing"


def test_notifier_accepts_instruction_and_stores_on_state(posts):
    n = _notifier()
    assert n._state.goal == GOAL, "instruction not stored on BuildState.goal"


def test_header_card_leads_with_goal(posts):
    n = _notifier()
    t0 = 1000.0
    header = tg.render_event(
        _ev({"event": "dispatch_started",
             "detail": {"mode": "master", "generator_chain": ["codex"]}}, ts=t0),
        verbosity="normal", mode="master", run_id="a3f9c1",
        context={"goal": GOAL},
    )
    assert header is not None
    assert "OAuth device flow" in header, "goal not in header card"
    # rendered as bold lead
    assert "<b>" in header


def test_status_card_leads_with_goal(posts):
    n = _notifier()
    _feed_three_steps(n)
    card = tg.render_status_card(n._state)
    assert "OAuth device flow" in card, "goal not in status card"


def test_absent_goal_no_crash_no_empty_bold(posts):
    n = _notifier(goal=None)
    _feed_three_steps(n)
    card = tg.render_status_card(n._state)
    # no empty bold lead line like "<b></b>"
    assert "<b></b>" not in card
    # header still renders without a goal
    header = tg.render_event(
        _ev({"event": "dispatch_started", "detail": {"generator_chain": ["codex"]}}, ts=1000.0),
        verbosity="normal", mode="master", run_id="a3f9c1", context={},
    )
    assert header is not None
    assert "<b></b>" not in header


# --------------------------------------------------------------------------- #
# G2 — step-done line carries duration + rounds
# --------------------------------------------------------------------------- #
def test_step_done_line_has_duration_and_rounds(posts):
    n = _notifier()
    _feed_three_steps(n)
    # The step-2 completed line is rendered through the notifier; capture the
    # broadcast text for the step-2 completion. Re-drive just that completion and
    # read what was rendered.
    n.flush(timeout=5.0)
    texts = [p["params"].get("text", "") for p in posts
             if p["method"] in ("sendMessage", "editMessageText")]
    blob = "\n".join(texts)
    # a step-done line that contains a duration token (minutes "m") AND "round"
    done_lines = [t for t in texts if "done" in t.lower() and "Step" in t]
    assert done_lines, "no step-done line was broadcast"
    joined = "\n".join(done_lines)
    assert re.search(r"\d+m", joined), "step-done line missing a duration like '2m14s'"
    assert "round" in joined.lower(), "step-done line missing the adversarial round count"


# --------------------------------------------------------------------------- #
# G3 — rich multi-line end-of-run summary card
# --------------------------------------------------------------------------- #
def _finish(n, *, success=True, confidence="verified",
            changed=None, t_end=1288.0):
    if changed is None:
        changed = [f"src/file_{i}.py" for i in range(12)]
    n.finished({
        "success": success, "mode": "master", "duration_s": 288,
        "changed_files": changed,
        "quality": {"confidence": confidence},
        "tokens": {"grand_total": {"total_tokens": 42000}},
        "run_id": "a3f9c1",
    })


def test_summary_card_is_multiline_with_real_newlines(posts):
    n = _notifier()
    _feed_three_steps(n)
    card = n._summary_card({
        "success": True, "mode": "master", "duration_s": 288,
        "changed_files": [f"src/file_{i}.py" for i in range(12)],
        "quality": {"confidence": "verified"},
        "tokens": {"grand_total": {"total_tokens": 42000}},
    })
    assert card.count("\n") >= 3, "summary card is not multi-line"
    assert "\\n" not in card, "summary card contains a LITERAL backslash-n"


def test_summary_card_verified_headline_and_goal(posts):
    n = _notifier()
    _feed_three_steps(n)
    card = n._summary_card({
        "success": True, "mode": "master", "duration_s": 288,
        "changed_files": ["a.py"], "quality": {"confidence": "verified"},
        "tokens": {"grand_total": {"total_tokens": 1000}},
    })
    assert "verified" in card.lower()
    assert "OAuth device flow" in card, "goal not in summary card"


def test_summary_card_shows_file_count_and_truncates(posts):
    n = _notifier()
    _feed_three_steps(n)
    changed = [f"src/file_{i}.py" for i in range(12)]
    card = n._summary_card({
        "success": True, "mode": "master", "duration_s": 288,
        "changed_files": changed, "quality": {"confidence": "verified"},
        "tokens": {"grand_total": {"total_tokens": 42000}},
    })
    # the total file count appears
    assert "12" in card, "file count not shown"
    # changed-files block truncates with a "+K more" tail (first ~8 then rest)
    assert "more" in card.lower(), "changed-files block did not truncate with '+K more'"


def test_summary_card_has_pre_step_recap_with_each_step(posts):
    n = _notifier()
    _feed_three_steps(n)
    card = n._summary_card({
        "success": True, "mode": "master", "duration_s": 288,
        "changed_files": ["a.py"], "quality": {"confidence": "verified"},
        "tokens": {"grand_total": {"total_tokens": 42000}},
    })
    assert "<pre>" in card and "</pre>" in card, "no <pre> recap block"
    # each tracked step title appears in the recap
    for title in ("Schema", "Endpoints", "Tests"):
        assert title in card, f"step '{title}' missing from recap"


def test_summary_card_failed_headline(posts):
    n = _notifier()
    _feed_three_steps(n)
    card = n._summary_card({
        "success": False, "mode": "master", "duration_s": 100,
        "error": "verifier red", "changed_files": [],
        "tokens": {}, "run_id": "a3f9c1",
    })
    # a failure headline glyph (🛑 or ❌) + the goal still shown
    assert ("🛑" in card) or ("❌" in card), "no failure headline glyph"
    assert "OAuth device flow" in card


def test_summary_card_no_literal_backslash_n_on_failure(posts):
    n = _notifier()
    _feed_three_steps(n)
    card = n._summary_card({
        "success": False, "mode": "master", "duration_s": 100,
        "error": "boom", "changed_files": ["x.py"], "tokens": {},
    })
    assert "\\n" not in card

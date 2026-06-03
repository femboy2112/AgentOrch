"""Acceptance CONTRACT for the Telegram overhaul — Pillar A (rich output + commands).

This file is an anti-no-op gate: it asserts the NEW public surface and the
STRUCTURAL shape of the richer output actually exist, so a build that doesn't
implement the feature fails the verifier instead of false-greening on unchanged
files. It is intentionally LOOSE about exact wording/emoji (the implementer owns
the visual design) but STRICT about: the new client methods, the run-header
card, the plan tree, step cards carrying a reference point + outcome, and the
/summary //files //why commands returning real content from a fixture meta.json.

Hermetic: fake token + fake ids, _post monkeypatched, state pinned to tmp_path.
Mirrors the fixtures/contract style of tests/test_telegram.py.
"""
from __future__ import annotations

import json

import pytest

from harness import telegram as tg
from harness import telegram_bot as bot


FAKE_TOKEN = "123456:FAKE-TEST-TOKEN-not-real"
FAKE_CHAT_ID = 444555666


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_TELEGRAM_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(tmp_path / "users.json"))


@pytest.fixture
def captured_posts(monkeypatch):
    sent = []

    def fake_post(self, method, params, *, timeout=None):
        sent.append({"method": method, "params": dict(params)})
        if method == "sendMessage":
            return {"ok": True, "result": {"message_id": 4242}}
        if method == "getUpdates":
            return {"ok": True, "result": []}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(tg.TelegramClient, "_post", fake_post)
    return sent


# --------------------------------------------------------------------------- #
# New TelegramClient methods (support the pinned live card; pure client surface)
# --------------------------------------------------------------------------- #
def test_client_has_edit_and_pin_methods():
    for name in ("edit_message_text", "pin_chat_message", "unpin_chat_message"):
        assert hasattr(tg.TelegramClient, name), f"TelegramClient.{name} missing"


def test_edit_message_text_posts_editMessageText(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_KEY", FAKE_TOKEN)
    calls = []
    monkeypatch.setattr(
        tg.TelegramClient, "_post",
        lambda self, method, params, *, timeout=None: calls.append((method, params)) or {"ok": True, "result": {}},
    )
    c = tg.TelegramClient()
    c.edit_message_text(FAKE_CHAT_ID, 99, "<b>hi</b>")
    assert calls and calls[0][0] == "editMessageText"
    assert calls[0][1].get("message_id") == 99
    assert calls[0][1].get("parse_mode") == "HTML"


def test_client_methods_are_best_effort_when_unconfigured(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_KEY", raising=False)
    c = tg.TelegramClient(token=None)
    # No token -> no network, return None, never raise.
    assert c.edit_message_text(1, 2, "x") is None
    assert c.pin_chat_message(1, 2) is None
    assert c.unpin_chat_message(1, 2) is None


# --------------------------------------------------------------------------- #
# Richer stream output: run-header card, plan tree, step reference point
# --------------------------------------------------------------------------- #
def _ev(data, **top):
    e = {"kind": "lifecycle", "run_id": "abc123", "data": data}
    e.update(top)
    return e


def test_dispatch_started_renders_a_header_card_with_mode_and_run():
    msg = tg.render_event(
        _ev({"event": "dispatch_started",
             "detail": {"mode": "master", "generator_chain": ["codex", "agy"],
                        "critic_chain": ["agy"]}}),
        verbosity="normal", mode="master", run_id="abc123",
    )
    assert msg is not None
    # A header card names the mode and the run; richer than a bare token.
    assert "master" in msg
    assert "abc123" in msg or "abc123"[-6:] in msg


def test_plan_completed_renders_a_step_tree_with_titles():
    titles = ["Scaffold module", "Wire the endpoint", "Integration tests"]
    msg = tg.render_event(
        _ev({"event": "orchestration_transition",
             "orchestration": {"workflow": "master", "phase": "plan",
                               "action": "completed", "step_total": 3,
                               "step_titles": titles}}),
        verbosity="normal", mode="master", run_id="abc123",
    )
    assert msg is not None
    # The plan surfaces the actual step list, not just "plan completed".
    for t in titles:
        assert t in msg, f"plan tree missing step title {t!r}"


def test_step_completed_carries_reference_point_and_outcome():
    msg = tg.render_event(
        _ev({"event": "orchestration_transition",
             "orchestration": {"workflow": "master", "phase": "step",
                               "action": "completed", "step_index": 3,
                               "step_total": 8, "step_title": "Wire the endpoint",
                               "outcome": "verified", "verified": True}}),
        verbosity="normal", mode="master", run_id="abc123",
    )
    assert msg is not None
    assert "3" in msg and "8" in msg          # reference point Step 3/8
    assert "Wire the endpoint" in msg          # the human anchor
    assert "verified" in msg.lower()           # the outcome


# --------------------------------------------------------------------------- #
# Rich end-of-run summary card (verified vs approved honesty + file count)
# --------------------------------------------------------------------------- #
def test_summary_card_is_rich_and_distinguishes_verified():
    n = tg.TelegramNotifier(run_id="abc123", mode="master", verbosity="normal",
                            client=tg.TelegramClient(token=FAKE_TOKEN),
                            chat_ids=[FAKE_CHAT_ID])
    card = n._summary_card({
        "success": True, "mode": "master", "duration_s": 1024,
        "changed_files": [f"path/file{i}.py" for i in range(11)],
        "quality": {"confidence": "verified"},
        "tokens": {"grand_total": {"total_tokens": 184200}},
    })
    assert "11" in card                        # file count surfaced
    assert "verified" in card.lower()


# --------------------------------------------------------------------------- #
# Commands: /summary, /files, /why, richer /runs
# --------------------------------------------------------------------------- #
@pytest.fixture
def fixture_run(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    rid = "20260603-120000-001"
    d = runs / rid
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({
        "success": True, "mode": "master", "duration_s": 252,
        "changed_files": ["harness/telegram.py", "tests/test_x.py"],
        "added": ["tests/test_x.py"], "modified": ["harness/telegram.py"], "deleted": [],
        "quality": {"confidence": "verified", "verified": True, "critic_approved": False,
                    "iterations_used": 2, "note": "A programmatic verifier passed."},
        "reconciliation": {"verdict": "not_reconciled", "disposition": "warn",
                           "findings": [{"name": "surprise", "classification": "exists_not_load_bearing",
                                         "sub_kind": "stub_constant"}]},
        "tokens": {"grand_total": {"total_tokens": 48210}},
    }), encoding="utf-8")
    (d / "changed-files.diff").write_text(
        "--- a/harness/telegram.py\n+++ b/harness/telegram.py\n@@ -1 +1,2 @@\n+new line\n", encoding="utf-8")
    monkeypatch.setattr(bot, "RUNS_DIR", runs)
    return rid


def test_help_lists_new_commands():
    for cmd in ("/summary", "/files", "/why"):
        assert cmd in bot.HELP_TEXT, f"{cmd} not in HELP_TEXT"


def test_summarize_run_helper_exists_and_is_rich(fixture_run):
    assert hasattr(bot, "summarize_run"), "summarize_run helper missing"
    out = bot.summarize_run(fixture_run)
    assert isinstance(out, str) and out
    assert "master" in out
    # surfaces the verdict/confidence and a duration
    assert "verified" in out.lower()


def test_command_summary_returns_rich_reply(fixture_run):
    reply = bot.handle_command(f"/summary {fixture_run}", state={}, chat_id=FAKE_CHAT_ID)
    assert reply and "master" in reply and "verified" in reply.lower()


def test_command_files_lists_changed_files(fixture_run):
    reply = bot.handle_command(f"/files {fixture_run}", state={}, chat_id=FAKE_CHAT_ID)
    assert reply
    assert "telegram.py" in reply


def test_command_why_surfaces_verdict_and_reconcile(fixture_run):
    reply = bot.handle_command(f"/why {fixture_run}", state={}, chat_id=FAKE_CHAT_ID)
    assert reply
    # the critic/verifier stance and the reconcile finding both appear
    assert "verifier" in reply.lower() or "verified" in reply.lower()
    assert "surprise" in reply or "load" in reply.lower()


def test_runs_is_richer_than_bare(fixture_run):
    reply = bot.handle_command("/runs 5", state={}, chat_id=FAKE_CHAT_ID)
    assert reply
    # richer /runs surfaces a confidence chip and/or relative age
    assert ("verified" in reply.lower()) or ("ago" in reply.lower())

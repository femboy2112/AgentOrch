"""Acceptance CONTRACT for Telegram fast-follow F1 — inline-keyboard buttons +
callback_query dispatch (spec §F1).

Anti-no-op gate. Drives the REAL ``BotDaemon._process_update`` seam with a fake
``callback_query`` update dict and asserts OBSERVABLE behavior:

  * a whitelisted ``why:<rid>`` callback yields a reply containing the verdict
    AND an ``answerCallbackQuery`` API call (clears the client spinner);
  * a NON-whitelisted callback is ignored — no reply, no answer;
  * ``data`` > 64 bytes / an unknown verb is a safe no-op (still answered when
    whitelisted, never a reply);
  * ``TelegramClient.answer_callback_query`` posts ``answerCallbackQuery`` and
    ``send_message`` passes ``reply_markup`` through to ``_post``.

Hermetic: fake token "123456:FAKE-TEST-TOKEN-not-real", FAKE ids only, every
API call captured via a monkeypatched ``TelegramClient._post`` (no real
network), state + whitelist pinned to tmp_path.
"""
from __future__ import annotations

import json

import pytest

from harness import telegram as tg
from harness import telegram_bot as bot


FAKE_TOKEN = "123456:FAKE-TEST-TOKEN-not-real"
FAKE_CHAT_ID = 444555666
FAKE_USER_ID = 111222333
OTHER_USER_ID = 999888777  # never whitelisted


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_TELEGRAM_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(tmp_path / "users.json"))
    monkeypatch.setenv("TELEGRAM_BOT_KEY", FAKE_TOKEN)


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
def posts(monkeypatch):
    """Capture every API call; sendMessage returns an incrementing message_id."""
    sent = []
    state = {"mid": 2000}

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


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    rd = tmp_path / "runs"
    rd.mkdir()
    monkeypatch.setattr(bot, "RUNS_DIR", rd)
    return rd


VERDICT_TEXT = "load-bearing-and-wired"


@pytest.fixture
def fixture_run(runs_dir):
    """A finished run whose /why output contains a recognizable verdict string."""
    rid = "20260603-120000-f1why"
    d = runs_dir / rid
    d.mkdir()
    (d / "events.jsonl").write_text("", encoding="utf-8")
    meta = {
        "success": True,
        "mode": "master",
        "changed_files": ["harness/telegram.py", "harness/telegram_bot.py"],
        "added": [],
        "quality": {"confidence": "verified", "verified": True,
                    "critic_approved": True, "note": "all checks passed"},
        "reconciliation": {"verdict": VERDICT_TEXT, "findings": []},
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return rid


def _short(rid: str) -> str:
    return rid[-6:] if len(rid) > 6 else rid


def _callback_update(*, data, user_id, query_id="cbq-1"):
    return {
        "update_id": 5000,
        "callback_query": {
            "id": query_id,
            "from": {"id": user_id},
            "message": {"chat": {"id": FAKE_CHAT_ID}, "message_id": 4242},
            "data": data,
        },
    }


def _daemon():
    return bot.BotDaemon(client=tg.TelegramClient())


# --------------------------------------------------------------------------- #
# Client surface
# --------------------------------------------------------------------------- #
def test_answer_callback_query_posts_method(posts):
    client = tg.TelegramClient()
    client.answer_callback_query("cbq-1", text="ok")
    methods = [p["method"] for p in posts]
    assert "answerCallbackQuery" in methods
    p = next(p for p in posts if p["method"] == "answerCallbackQuery")
    assert str(p["params"].get("callback_query_id")) == "cbq-1"


def test_send_message_passes_reply_markup_through(posts):
    client = tg.TelegramClient()
    markup = {"inline_keyboard": [[{"text": "📂 Files", "callback_data": "files:abc"}]]}
    client.send_message(FAKE_CHAT_ID, "hi", reply_markup=markup)
    p = next(p for p in posts if p["method"] == "sendMessage")
    assert "reply_markup" in p["params"]
    # serialized JSON, round-trips back to the markup
    assert json.loads(p["params"]["reply_markup"]) == markup


# --------------------------------------------------------------------------- #
# Callback dispatch (the F1 core)
# --------------------------------------------------------------------------- #
def test_whitelisted_why_callback_replies_with_verdict_and_answers(
    posts, users_file, fixture_run
):
    daemon = _daemon()
    daemon._process_update(
        _callback_update(data=f"why:{_short(fixture_run)}", user_id=FAKE_USER_ID)
    )

    sends = [p for p in posts if p["method"] == "sendMessage"]
    answers = [p for p in posts if p["method"] == "answerCallbackQuery"]

    assert sends, "whitelisted callback must produce a reply send"
    assert any(VERDICT_TEXT in (p["params"].get("text") or "") for p in sends), \
        "the why reply must contain the verdict"
    assert answers, "callback must ALWAYS be answered (clears the spinner)"
    assert str(answers[0]["params"].get("callback_query_id")) == "cbq-1"


def test_non_whitelisted_callback_is_ignored(posts, users_file, fixture_run):
    daemon = _daemon()
    daemon._process_update(
        _callback_update(data=f"why:{_short(fixture_run)}", user_id=OTHER_USER_ID)
    )
    assert [p for p in posts if p["method"] == "sendMessage"] == []
    assert [p for p in posts if p["method"] == "answerCallbackQuery"] == []


def test_oversized_data_is_safe_no_op_but_still_answered(posts, users_file):
    daemon = _daemon()
    big = "why:" + ("x" * 80)  # > 64 bytes
    assert len(big.encode("utf-8")) > 64
    daemon._process_update(_callback_update(data=big, user_id=FAKE_USER_ID))
    assert [p for p in posts if p["method"] == "sendMessage"] == []
    # still answered so the client spinner clears
    assert [p for p in posts if p["method"] == "answerCallbackQuery"]


def test_unknown_verb_is_safe_no_op_but_still_answered(posts, users_file, fixture_run):
    daemon = _daemon()
    daemon._process_update(
        _callback_update(data=f"explode:{_short(fixture_run)}", user_id=FAKE_USER_ID)
    )
    assert [p for p in posts if p["method"] == "sendMessage"] == []
    assert [p for p in posts if p["method"] == "answerCallbackQuery"]


def test_files_callback_replies(posts, users_file, fixture_run):
    daemon = _daemon()
    daemon._process_update(
        _callback_update(data=f"files:{_short(fixture_run)}", user_id=FAKE_USER_ID)
    )
    sends = [p for p in posts if p["method"] == "sendMessage"]
    assert sends, "files callback must reply"
    assert any("telegram.py" in (p["params"].get("text") or "") for p in sends)
    assert [p for p in posts if p["method"] == "answerCallbackQuery"]

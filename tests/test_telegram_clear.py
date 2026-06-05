"""Hermetic contract for /clear — delete this chat's bot-known messages.

Telegram has no wipe-history API: a bot can only delete individual messages
< 48h old whose ids it tracked while running. Both the daemon (incoming + its
sends) and the dispatch-side TelegramNotifier (a separate process) append ids to
ONE shared JSONL log next to the state file; /clear reads it, deletes everything
in-window (batch + per-message fallback), purges the log, and confirms honestly.

Fully hermetic: NO network (FakeClient captures delete/send calls), FAKE ids
only, state/log pinned to tmp_path, ``_clock`` monkeypatched for the age window.
"""
from __future__ import annotations

import json

import pytest

from harness import telegram as tg
from harness import telegram_bot as bot


FAKE_TOKEN = "123456:FAKE-TEST-TOKEN-not-real"
FAKE_CHAT_ID = 444555666
OTHER_CHAT_ID = 222111000
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
        json.dumps([{"id": FAKE_USER_ID, "username": "t", "last_chat_id": FAKE_CHAT_ID}]),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(p))
    return p


class FakeClient:
    """Captures delete/send calls; never touches the network."""

    def __init__(self, *, batch_ok=True, per_ok=True):
        self.configured = True
        self.deleted_batches = []
        self.deleted_singles = []
        self.sent = []
        self._batch_ok = batch_ok
        self._per_ok = per_ok  # bool or a predicate(mid)->bool

    def delete_messages(self, chat_id, ids):
        self.deleted_batches.append((chat_id, list(ids)))
        return {"ok": bool(self._batch_ok)}

    def delete_message(self, chat_id, mid):
        self.deleted_singles.append((chat_id, mid))
        ok = self._per_ok(mid) if callable(self._per_ok) else bool(self._per_ok)
        return {"ok": ok} if ok else {"ok": False, "description": "too old"}

    def send_message(self, chat_id, text, reply_markup=None):
        self.sent.append({"chat_id": chat_id, "text": text})
        return {"ok": True, "result": {"message_id": 9000 + len(self.sent)}}


def _msg_update(text, *, user_id=FAKE_USER_ID, message_id=100):
    return {
        "update_id": 1,
        "message": {
            "from": {"id": user_id},
            "chat": {"id": FAKE_CHAT_ID},
            "message_id": message_id,
            "text": text,
        },
    }


# =========================================================================== #
# A. shared id-log (record / read / purge)
# =========================================================================== #
def test_record_then_read_roundtrip():
    for mid in (10, 11, 12):
        tg.record_sent_message(FAKE_CHAT_ID, mid)
    assert tg.read_clear_targets(FAKE_CHAT_ID) == [10, 11, 12]


def test_read_dedupes_and_filters_by_chat():
    tg.record_sent_message(FAKE_CHAT_ID, 10)
    tg.record_sent_message(FAKE_CHAT_ID, 10)  # dup
    tg.record_sent_message(OTHER_CHAT_ID, 99)  # other chat
    assert tg.read_clear_targets(FAKE_CHAT_ID) == [10]
    assert tg.read_clear_targets(OTHER_CHAT_ID) == [99]


def test_age_window_excludes_old_entries(monkeypatch):
    # Record at t=0; read far in the future -> beyond the 48h window -> excluded.
    tg.record_sent_message(FAKE_CHAT_ID, 5, now=0.0)
    assert tg.read_clear_targets(FAKE_CHAT_ID, now=10.0) == [5]
    way_later = tg.CLEAR_MAX_AGE_SECONDS + 100.0
    assert tg.read_clear_targets(FAKE_CHAT_ID, now=way_later) == []


def test_record_extracts_message_id_from_send_result():
    tg.record_sent_message(FAKE_CHAT_ID, {"ok": True, "result": {"message_id": 77}})
    assert tg.read_clear_targets(FAKE_CHAT_ID) == [77]


@pytest.mark.parametrize("bad", [
    None,
    {"ok": False},                       # no result
    {"ok": True, "result": {}},          # no message_id
    {"ok": True, "result": {"message_id": None}},
    True,                                # bool is not a message id
    "not-an-int",
])
def test_record_ignores_unusable_ids(bad):
    tg.record_sent_message(FAKE_CHAT_ID, bad)
    assert tg.read_clear_targets(FAKE_CHAT_ID) == []


def test_record_never_raises_on_bad_chat():
    # No chat id -> silent no-op, never raises.
    tg.record_sent_message(None, 5)
    assert tg.read_clear_targets(None) == []


def test_purge_drops_chat_keeps_others():
    tg.record_sent_message(FAKE_CHAT_ID, 1)
    tg.record_sent_message(OTHER_CHAT_ID, 2)
    tg.purge_clear_log(FAKE_CHAT_ID)
    assert tg.read_clear_targets(FAKE_CHAT_ID) == []
    assert tg.read_clear_targets(OTHER_CHAT_ID) == [2]


def test_read_missing_log_is_empty():
    assert tg.read_clear_targets(FAKE_CHAT_ID) == []


def test_read_tolerates_garbage_lines(tmp_path):
    path = tg.clearlog_path()
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("not json\n")
        fh.write(json.dumps({"c": str(FAKE_CHAT_ID), "m": 42, "t": 1.0}) + "\n")
        fh.write("{broken\n")
    assert tg.read_clear_targets(FAKE_CHAT_ID, now=2.0) == [42]


# =========================================================================== #
# B. client delete methods (params shape, guards)
# =========================================================================== #
def test_delete_messages_sends_json_array(monkeypatch):
    captured = {}

    def fake_post(self, method, params, *, timeout=None):
        captured["method"] = method
        captured["params"] = dict(params)
        return {"ok": True, "result": True}

    monkeypatch.setattr(tg.TelegramClient, "_post", fake_post)
    tg.TelegramClient().delete_messages(FAKE_CHAT_ID, [1, 2, 3])
    assert captured["method"] == "deleteMessages"
    assert json.loads(captured["params"]["message_ids"]) == [1, 2, 3]


def test_delete_message_params(monkeypatch):
    captured = {}
    monkeypatch.setattr(tg.TelegramClient, "_post",
                        lambda self, m, p, *, timeout=None: captured.update(method=m, params=dict(p)) or {"ok": True})
    tg.TelegramClient().delete_message(FAKE_CHAT_ID, 55)
    assert captured["method"] == "deleteMessage"
    assert captured["params"] == {"chat_id": FAKE_CHAT_ID, "message_id": 55}


def test_delete_messages_empty_is_noop(monkeypatch):
    monkeypatch.setattr(tg.TelegramClient, "_post",
                        lambda *a, **k: pytest.fail("must not POST for empty ids"))
    assert tg.TelegramClient().delete_messages(FAKE_CHAT_ID, []) is None


# =========================================================================== #
# C. daemon _handle_clear
# =========================================================================== #
def test_handle_clear_batch_deletes_purges_and_confirms():
    client = FakeClient(batch_ok=True)
    daemon = bot.BotDaemon(client=client)
    for mid in (1, 2, 3):
        tg.record_sent_message(FAKE_CHAT_ID, mid)

    daemon._handle_clear(FAKE_CHAT_ID)

    assert client.deleted_batches == [(FAKE_CHAT_ID, [1, 2, 3])]
    assert client.deleted_singles == []  # batch succeeded -> no per-message fallback
    # Log purged for this chat (the new confirmation message is the only new entry).
    assert tg.read_clear_targets(FAKE_CHAT_ID) == [9001]
    assert client.sent and "Cleared 3 messages" in client.sent[0]["text"]


def test_handle_clear_per_message_fallback_when_batch_fails():
    client = FakeClient(batch_ok=False, per_ok=True)
    daemon = bot.BotDaemon(client=client)
    for mid in (4, 5):
        tg.record_sent_message(FAKE_CHAT_ID, mid)

    daemon._handle_clear(FAKE_CHAT_ID)

    assert client.deleted_batches == [(FAKE_CHAT_ID, [4, 5])]
    # Batch returned not-ok -> each id retried individually.
    assert client.deleted_singles == [(FAKE_CHAT_ID, 4), (FAKE_CHAT_ID, 5)]
    assert "Cleared 2 messages" in client.sent[0]["text"]


def test_handle_clear_reports_partial_failures():
    # Batch fails; per-message succeeds only for even ids -> 1 of 2 removed.
    client = FakeClient(batch_ok=False, per_ok=lambda mid: mid % 2 == 0)
    daemon = bot.BotDaemon(client=client)
    for mid in (6, 7):
        tg.record_sent_message(FAKE_CHAT_ID, mid)

    daemon._handle_clear(FAKE_CHAT_ID)

    text = client.sent[0]["text"]
    assert "Cleared 1 message." in text
    assert "1 couldn't be removed" in text


def test_handle_clear_empty_log_is_friendly():
    client = FakeClient()
    daemon = bot.BotDaemon(client=client)
    daemon._handle_clear(FAKE_CHAT_ID)
    assert client.deleted_batches == [] and client.deleted_singles == []
    assert "Nothing to clear" in client.sent[0]["text"]


def test_handle_clear_chunks_over_100(monkeypatch):
    client = FakeClient(batch_ok=True)
    daemon = bot.BotDaemon(client=client)
    for mid in range(1, 251):  # 250 ids -> 100 + 100 + 50
        tg.record_sent_message(FAKE_CHAT_ID, mid)

    daemon._handle_clear(FAKE_CHAT_ID)

    sizes = [len(ids) for _cid, ids in client.deleted_batches]
    assert sizes == [100, 100, 50]
    assert "Cleared 250 messages" in client.sent[0]["text"]


def test_handle_clear_never_raises_on_client_explosion():
    class Boom(FakeClient):
        def delete_messages(self, chat_id, ids):
            raise RuntimeError("network down")

        def delete_message(self, chat_id, mid):
            raise RuntimeError("network down")

    client = Boom()
    daemon = bot.BotDaemon(client=client)
    tg.record_sent_message(FAKE_CHAT_ID, 1)
    # Must not raise even when every delete throws.
    daemon._handle_clear(FAKE_CHAT_ID)
    assert client.sent  # still sent an (honest) confirmation


# =========================================================================== #
# D. _process_update routing + incoming-id tracking
# =========================================================================== #
def test_clear_command_routes_and_tracks_incoming(users_file):
    client = FakeClient(batch_ok=True)
    daemon = bot.BotDaemon(client=client)
    # The /clear message itself (id 555) is tracked BEFORE routing, so it's a
    # delete target — proves incoming-id tracking + routing in one shot.
    daemon._process_update(_msg_update("/clear", message_id=555))
    assert client.deleted_batches == [(FAKE_CHAT_ID, [555])]
    assert client.sent  # confirmation sent


def test_clear_not_handled_via_pure_handle_command():
    # /clear is an action, not a string reply -> handle_command ignores it.
    assert bot.handle_command("/clear", state=bot.load_state(), chat_id=FAKE_CHAT_ID) is None


def test_clear_from_non_whitelisted_user_ignored(users_file):
    client = FakeClient()
    daemon = bot.BotDaemon(client=client)
    daemon._process_update(_msg_update("/clear", user_id=OTHER_USER_ID, message_id=1))
    assert client.deleted_batches == [] and client.deleted_singles == []
    assert client.sent == []
    assert tg.read_clear_targets(FAKE_CHAT_ID) == []  # nothing even tracked


def test_clear_registered_in_menu_and_help():
    assert any(c["command"] == "clear" for c in bot.BOT_COMMANDS)
    assert "/clear" in bot.HELP_TEXT

"""Issue #63: telegram commands must (a) register with Telegram (setMyCommands so
the client shows a command menu) and (b) be answerable WHILE a build/dispatch runs
via a singleton-guarded embedded poller — without a separately-launched daemon.

All hermetic: a fake client (no network), tmp whitelist + state/lock paths.
"""
from __future__ import annotations

import json
import time

import pytest

from harness import telegram_bot as tb
from harness.telegram_bot import (
    BOT_COMMANDS,
    BotDaemon,
    EmbeddedCommandPoller,
    poller_lock_path,
)


class _FakeClient:
    """Configured Telegram client that records calls and never hits the network."""

    def __init__(self):
        self.configured = True
        self.commands_set = None
        self.updates_calls = 0

    def set_my_commands(self, commands):
        self.commands_set = commands
        return {"ok": True}

    def get_updates(self, offset=None, timeout=25):
        self.updates_calls += 1
        return []

    def send_message(self, *a, **k):
        return {"ok": True}


@pytest.fixture
def whitelist(tmp_path, monkeypatch):
    users = tmp_path / "users.json"
    users.write_text(json.dumps([{"id": 111, "last_chat_id": 111}]))
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(users))
    state = tmp_path / "bot_state.json"
    monkeypatch.setenv("AGY_TELEGRAM_STATE", str(state))
    return {"users": str(users), "state": str(state)}


# --------------------------------------------------------------------------- #
# (a) command registration
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_bot_commands_shape():
    # Telegram requires lowercase names (no slash) + a non-empty description.
    assert BOT_COMMANDS
    for c in BOT_COMMANDS:
        assert set(c) == {"command", "description"}
        assert c["command"] == c["command"].lower() and "/" not in c["command"]
        assert c["description"]


@pytest.mark.not_slow
def test_register_commands_calls_set_my_commands():
    fake = _FakeClient()
    BotDaemon(client=fake).register_commands()
    assert fake.commands_set == BOT_COMMANDS


@pytest.mark.not_slow
def test_register_commands_never_raises():
    class _Boom(_FakeClient):
        def set_my_commands(self, commands):
            raise RuntimeError("telegram down")

    # Best-effort: a Telegram outage must not propagate.
    BotDaemon(client=_Boom()).register_commands()


# --------------------------------------------------------------------------- #
# (b) embedded poller + singleton lock
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_poller_lock_path_next_to_state(whitelist):
    p = poller_lock_path()
    assert p.endswith("telegram_poller.lock")
    assert p.rsplit("/", 1)[0] == whitelist["state"].rsplit("/", 1)[0]


@pytest.mark.not_slow
def test_embedded_poller_starts_registers_and_polls(whitelist):
    fake = _FakeClient()
    poller = EmbeddedCommandPoller(client=fake, poll_timeout=1)
    assert poller.start() is True
    try:
        assert fake.commands_set == BOT_COMMANDS  # registered on start
        # The loop polls; give it a tick.
        deadline = time.monotonic() + 2.0
        while fake.updates_calls == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert fake.updates_calls >= 1
    finally:
        poller.stop()


@pytest.mark.not_slow
def test_only_one_poller_runs_at_a_time(whitelist):
    """A second poller must stand down while the first holds the getUpdates lock
    (prevents Telegram 409 Conflict), then succeed once the first stops."""
    first = EmbeddedCommandPoller(client=_FakeClient(), poll_timeout=1)
    assert first.start() is True
    try:
        second = EmbeddedCommandPoller(client=_FakeClient(), poll_timeout=1)
        assert second.start() is False  # lock held by `first`
    finally:
        first.stop()

    # Once the first releases, a new poller can acquire the lock.
    third = EmbeddedCommandPoller(client=_FakeClient(), poll_timeout=1)
    assert third.start() is True
    third.stop()


@pytest.mark.not_slow
def test_poller_no_op_without_whitelist(tmp_path, monkeypatch):
    # No whitelisted users -> nothing to serve -> poller declines to start.
    users = tmp_path / "users.json"
    users.write_text("[]")
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(users))
    monkeypatch.setenv("AGY_TELEGRAM_STATE", str(tmp_path / "state.json"))
    poller = EmbeddedCommandPoller(client=_FakeClient())
    assert poller.start() is False


@pytest.mark.not_slow
def test_poller_no_op_when_client_unconfigured(whitelist):
    class _Unconfigured(_FakeClient):
        def __init__(self):
            super().__init__()
            self.configured = False

    assert EmbeddedCommandPoller(client=_Unconfigured()).start() is False

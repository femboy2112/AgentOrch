from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time

import pytest

from agy_orchestrator.core.agent import AgentInstance
from harness import cli, dispatch, passthrough, roles, telegram_bot as bot


FAKE_CHAT_ID = 12345


@pytest.fixture(autouse=True)
def _isolate_passthrough(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(dispatch, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(passthrough, "SESSIONS_DIR", runs_dir / ".sessions")
    monkeypatch.setenv("AGY_BENCH_MOCK", "1")
    monkeypatch.setenv("AGY_BENCH_MOCK_SLEEP", "0")
    monkeypatch.setenv("AGY_BENCH_MOCK_OUTPUT", "mock passthrough reply")
    monkeypatch.setenv("AGY_WATCHDOG", "off")
    monkeypatch.setenv("AGY_USAGE_WALL_BACKOFF", "0")
    monkeypatch.setenv("AGY_TELEGRAM_STATE", str(tmp_path / "telegram_state.json"))
    monkeypatch.setenv("TELEGRAM_BOT_KEY", "123456:FAKE-TEST-TOKEN")

    def fake_atomic_write_text(path, text, *, encoding="utf-8"):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding=encoding)
        tmp.replace(path)

    monkeypatch.setattr(dispatch, "_atomic_write_text", fake_atomic_write_text)
    monkeypatch.setattr(cli, "version_string", lambda _name: "harness-test")

    def forbidden_popen(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("passthrough tests must not spawn real subprocesses")

    async def forbidden_async_exec(*_args, **_kwargs):  # pragma: no cover
        raise AssertionError("passthrough tests must not spawn real subprocesses")

    monkeypatch.setattr(subprocess, "Popen", forbidden_popen)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_async_exec)
    monkeypatch.setattr(roles, "_codex_model_fallbacks", lambda _configs: {})
    return runs_dir


class _FakeTelegramClient:
    instances = []

    def __init__(self):
        self.messages = []
        _FakeTelegramClient.instances.append(self)

    def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text, kwargs))
        return {"ok": True}


def _install_fake_telegram(monkeypatch):
    from harness import telegram

    _FakeTelegramClient.instances = []
    monkeypatch.setattr(telegram, "TelegramClient", _FakeTelegramClient)


@pytest.fixture(autouse=True)
def _fake_telegram_client(monkeypatch):
    _install_fake_telegram(monkeypatch)


def _run(coro):
    return asyncio.run(coro)


class _BaseStubAgent(AgentInstance):
    constructed = []
    reply = "ok"
    fail = False
    stderr_text = ""
    produced_session_id = None

    def __init__(self, prompt: str, model=None, additional_flags=None, **kwargs):
        super().__init__(
            prompt=prompt, model=model, additional_flags=additional_flags, **kwargs
        )
        self.kwargs = dict(kwargs)
        self.effort = kwargs.get("effort")
        type(self).constructed.append(self)

    @classmethod
    async def get_available_models(cls):
        return ["stub-model"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input=None):
        if self.fail:
            self.stderr = self.stderr_text or "usage limit reached"
            raise RuntimeError(self.stderr)
        if self.produced_session_id:
            self.session_id = self.produced_session_id
        self.stdout = self.reply
        self.returncode = 0
        return self.reply


def _agent_class(name, *, reply="ok", fail=False, stderr_text="", session_id=None):
    class Stub(_BaseStubAgent):
        constructed = []

    Stub.__name__ = name
    Stub.reply = reply
    Stub.fail = fail
    Stub.stderr_text = stderr_text
    Stub.produced_session_id = session_id
    return Stub


def _install_agent_classes(monkeypatch, **classes):
    merged = {**roles.AGENT_CLASSES, **classes}
    monkeypatch.setattr(roles, "AGENT_CLASSES", merged)
    monkeypatch.setattr(roles, "_class_for", lambda name: merged[name])


def test_ask_one_shot_returns_reply_prints_stdout_exit_zero(monkeypatch, capsys):
    async def fake_run_turn(prompt, **kwargs):
        assert prompt == "hello worker"
        assert kwargs["provider_chain"] == ["codex"]
        assert kwargs["capture"] is False
        return passthrough.PassthroughResult(reply="worker reply", provider_used="codex")

    monkeypatch.setattr(passthrough, "run_turn", fake_run_turn)

    rc = cli.main(["ask", "hello worker", "--no-capture"])

    assert rc == 0
    assert capsys.readouterr().out == "worker reply\n"


def test_session_create_resume_native_raw_prompt_and_codex_transcript(monkeypatch):
    calls = []

    class FakeAgent:
        def __init__(self, reply, session_id):
            self.reply = reply
            self.session_id = session_id

        async def run_async(self):
            return self.reply

    def fake_build_role_agent(chain, **kwargs):
        calls.append((list(chain), dict(kwargs)))
        session_id = "native-1" if chain[0] == "claude" else None
        return FakeAgent(f"reply-{len(calls)}", session_id)

    monkeypatch.setattr(roles, "build_role_agent", fake_build_role_agent)
    session = passthrough.PassthroughSession(
        session_id="native-chat",
        provider="claude",
        provider_chain=["claude"],
        model=None,
        effort=None,
        provider_session_id=None,
        turns=[],
        created_at=1.0,
        updated_at=1.0,
    )

    first = _run(
        passthrough.run_turn(
            "first prompt",
            provider_chain=["claude"],
            session=session,
            capture=False,
        )
    )
    assert first.reply == "reply-1"
    assert calls[-1][1]["prompt"] == "first prompt"
    assert calls[-1][1]["session_id"] is None
    assert session.provider_session_id == "native-1"

    loaded = passthrough.SessionStore.load_session("native-chat")
    second = _run(
        passthrough.run_turn(
            "second prompt",
            provider_chain=["claude"],
            session=loaded,
            capture=False,
        )
    )
    assert second.reply == "reply-2"
    assert calls[-1][1]["prompt"] == "second prompt"
    assert calls[-1][1]["session_id"] == "native-1"

    codex_session = passthrough.SessionStore.load_session("native-chat")
    _run(
        passthrough.run_turn(
            "codex prompt",
            provider_chain=["codex"],
            session=codex_session,
            capture=False,
        )
    )
    sent = calls[-1][1]["prompt"]
    assert sent.startswith("Continue this conversation:\n")
    assert "User: first prompt" in sent
    assert "Assistant: reply-1" in sent
    assert sent.endswith("\ncodex prompt")
    assert calls[-1][1]["session_id"] is None
    assert codex_session.provider == "codex"
    assert codex_session.provider_session_id is None


def test_session_round_trip_persists_provider_session_id_atomically():
    session = passthrough.PassthroughSession(
        session_id="My Session",
        provider="grok",
        provider_chain=["grok"],
        model="grok-build",
        effort=None,
        provider_session_id="grok-native-7",
        turns=[{"role": "user", "text": "hi", "ts": 2.0}],
        created_at=1.0,
        updated_at=2.0,
    )

    passthrough.SessionStore.save_session(session)
    path = passthrough.SessionStore.session_file("My Session")

    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["provider_session_id"] == "grok-native-7"
    loaded = passthrough.SessionStore.load_session("my session")
    assert loaded is not None
    assert loaded.provider_session_id == "grok-native-7"
    assert loaded.turns == session.turns


def test_build_role_agent_session_none_matches_omitted_direct_and_fallback(monkeypatch):
    ClaudeAgent = _agent_class("ClaudeAgent", reply="claude")
    GrokAgent = _agent_class("GrokAgent", reply="grok")
    _install_agent_classes(monkeypatch, claude=ClaudeAgent, grok=GrokAgent)

    omitted = roles.build_role_agent(["claude"], prompt="p", fallback=False)
    explicit_none = roles.build_role_agent(
        ["claude"], prompt="p", fallback=False, session_id=None
    )
    assert type(omitted) is type(explicit_none)
    assert omitted.prompt == explicit_none.prompt == "p"
    assert omitted.model == explicit_none.model
    assert omitted.kwargs == explicit_none.kwargs
    assert "session_id" not in omitted.kwargs
    assert "fork_session" not in omitted.kwargs

    fb_omitted = roles.build_role_agent(["claude", "grok"], prompt="p")
    fb_none = roles.build_role_agent(["claude", "grok"], prompt="p", session_id=None)
    assert type(fb_omitted).__name__ == type(fb_none).__name__
    assert fb_omitted.__dict__ == fb_none.__dict__
    assert not hasattr(fb_none, "_session_owner")


def test_fallback_wraps_and_rolls_from_walled_codex_to_agy(monkeypatch):
    CodexAgent = _agent_class(
        "CodexAgent",
        fail=True,
        stderr_text="usage limit reached (used_percent=100.0)",
    )
    AgyAgent = _agent_class("AgyAgent", reply="agy-ok")
    _install_agent_classes(monkeypatch, codex=CodexAgent, agy=AgyAgent)

    agent = roles.build_role_agent(["codex", "agy"], prompt="p")

    assert type(agent).__name__.startswith("FallbackAgent")
    assert _run(agent.run_async()) == "agy-ok"
    assert len(CodexAgent.constructed) == 1
    assert len(AgyAgent.constructed) == 1
    assert getattr(agent, "last_provider") == "AgyAgent"


def test_render_transcript_respects_turn_and_char_bounds():
    turns = [
        {"role": "user", "text": f"user-{i}-" + ("x" * 20), "ts": float(i)}
        if i % 2 == 0
        else {
            "role": "assistant",
            "text": f"assistant-{i}-" + ("y" * 20),
            "ts": float(i),
        }
        for i in range(10)
    ]

    by_turn = passthrough.render_transcript(turns, max_turns=4, max_chars=1000)
    assert "user-0" not in by_turn
    assert "assistant-7" in by_turn
    assert "user-8" in by_turn
    assert "assistant-9" in by_turn

    by_chars = passthrough.render_transcript(turns, max_turns=10, max_chars=95)
    assert len(by_chars) <= 95
    assert by_chars.startswith("Continue this conversation:")
    assert "user-0" not in by_chars

    tiny = passthrough.render_transcript(turns, max_turns=10, max_chars=10)
    assert tiny == "Continue t"
    assert len(tiny) <= 10


def test_sessions_cli_list_show_rm_and_missing_show(capsys):
    now = time.time()
    passthrough.SessionStore.save_session(
        passthrough.PassthroughSession(
            session_id="alpha",
            provider="codex",
            provider_chain=["codex"],
            model="m1",
            effort="low",
            provider_session_id=None,
            turns=[
                {"role": "user", "text": "hello", "ts": now},
                {"role": "assistant", "text": "hi", "ts": now},
            ],
            created_at=now,
            updated_at=now,
        )
    )

    assert cli.main(["sessions", "list"]) == 0
    out = capsys.readouterr().out
    assert "alpha\tprovider=codex\tmodel=m1\tturns=2\tupdated-at=" in out

    assert cli.main(["sessions", "show", "alpha"]) == 0
    assert capsys.readouterr().out == "User: hello\nAssistant: hi\n"

    assert cli.main(["sessions", "show", "missing"]) == 1
    err = capsys.readouterr().err
    assert "session not found: missing" in err

    assert cli.main(["sessions", "rm", "alpha"]) == 0
    capsys.readouterr()
    assert passthrough.SessionStore.load_session("alpha") is None


@pytest.mark.parametrize("bad", ["", "   ", ".", "..", "../x", "x/../y", "a/b", r"a\b"])
def test_session_slug_rejects_empty_traversal_and_path_separators(bad):
    with pytest.raises(ValueError):
        passthrough.SessionStore.session_file(bad)


def test_session_slug_sanitizes_and_bounds_plain_names():
    path = passthrough.SessionStore.session_file("My Weird Session!!!")
    assert path.name == "my-weird-session---.json"

    long_path = passthrough.SessionStore.session_file("A" * 200)
    assert long_path.stem == "a" * passthrough.SESSION_ID_MAX_LEN


def test_telegram_ask_and_chat_build_detached_argv_lists(monkeypatch):
    calls = []
    state = {"chats": {str(FAKE_CHAT_ID): {"mute_until": "on"}}}
    monkeypatch.delenv("AGY_PASSTHROUGH_PROVIDER", raising=False)
    monkeypatch.setattr(bot, "_spawn_dispatch", lambda argv: calls.append(list(argv)) or 222)

    ask_reply = bot.handle_command(
        "/ask --provider agy summarize this", state=state, chat_id=FAKE_CHAT_ID
    )
    ask_argv = calls[-1]
    assert ask_reply == "🤔 asking agy…"
    assert isinstance(ask_argv, list)
    assert ask_argv[:4] == [sys.executable, "-m", "harness", "ask"]
    assert ask_argv[4] == "summarize this"
    assert ask_argv[5:] == [
        "--provider",
        "agy",
        "--telegram",
        "--telegram-chat",
        str(FAKE_CHAT_ID),
    ]

    chat_reply = bot.handle_command(
        "/chat --provider grok remember this", state=state, chat_id=FAKE_CHAT_ID
    )
    chat_argv = calls[-1]
    assert chat_reply == "🤔 asking grok…"
    assert chat_argv[:4] == [sys.executable, "-m", "harness", "ask"]
    assert chat_argv[4] == "remember this"
    assert chat_argv[chat_argv.index("--provider") + 1] == "grok"
    assert chat_argv[chat_argv.index("--session") + 1] == f"tg-{FAKE_CHAT_ID}"
    assert chat_argv[chat_argv.index("--telegram-chat") + 1] == str(FAKE_CHAT_ID)
    assert "--telegram" in chat_argv
    assert state["chats"][str(FAKE_CHAT_ID)]["passthrough_provider"] == "grok"
    assert state["chats"][str(FAKE_CHAT_ID)]["mute_until"] == "on"


def test_telegram_invalid_provider_rejected_and_chat_new_resets(monkeypatch):
    calls = []
    deleted = []
    state = {}
    monkeypatch.setattr(bot, "_spawn_dispatch", lambda argv: calls.append(list(argv)) or 222)
    monkeypatch.setattr(
        passthrough.SessionStore,
        "delete_session",
        staticmethod(lambda session_id: deleted.append(session_id)),
    )

    bad = bot.handle_command(
        "/ask --provider nope prompt", state=state, chat_id=FAKE_CHAT_ID
    )
    assert "Unknown provider" in bad
    assert calls == []

    reset = bot.handle_command("/chat new", state=state, chat_id=FAKE_CHAT_ID)
    assert reset == "🧹 Chat session reset."
    assert deleted == [f"tg-{FAKE_CHAT_ID}"]
    assert calls == []


def test_telegram_passthrough_commands_inherit_whitelist_gate(tmp_path, monkeypatch):
    users_path = tmp_path / "users.json"
    users_path.write_text('[{"id": 111, "username": "allowed"}]', encoding="utf-8")
    calls = []

    class FakeClient:
        def __init__(self):
            self.messages = []

        def send_message(self, chat_id, text, reply_markup=None):
            self.messages.append((chat_id, text, reply_markup))
            return {"ok": True}

    monkeypatch.setattr(bot, "_spawn_dispatch", lambda argv: calls.append(list(argv)) or 222)
    client = FakeClient()
    daemon = bot.BotDaemon(
        client=client,
        users_path=str(users_path),
        state_file=str(tmp_path / "state.json"),
    )

    daemon._process_update(
        {
            "update_id": 1,
            "message": {
                "from": {"id": 999},
                "chat": {"id": FAKE_CHAT_ID},
                "message_id": 77,
                "text": "/ask should not spawn",
            },
        }
    )

    assert calls == []
    assert client.messages == []


def test_ask_telegram_sends_escaped_reply_once(monkeypatch, capsys):
    _install_fake_telegram(monkeypatch)

    async def fake_run_turn(*_args, **_kwargs):
        return passthrough.PassthroughResult(
            reply="<b>hello</b> & goodbye",
            provider_used="codex",
        )

    monkeypatch.setattr(passthrough, "run_turn", fake_run_turn)

    rc = cli.main(
        [
            "ask",
            "prompt",
            "--telegram",
            "--telegram-chat",
            str(FAKE_CHAT_ID),
            "--no-capture",
        ]
    )

    assert rc == 0
    assert capsys.readouterr().out == "<b>hello</b> & goodbye\n"
    assert len(_FakeTelegramClient.instances) == 1
    assert _FakeTelegramClient.instances[0].messages == [
        (str(FAKE_CHAT_ID), "&lt;b&gt;hello&lt;/b&gt; &amp; goodbye", {})
    ]


def test_ask_telegram_splits_long_escaped_reply(monkeypatch):
    _install_fake_telegram(monkeypatch)
    reply = "<" + ("x" * 5000)

    async def fake_run_turn(*_args, **_kwargs):
        return passthrough.PassthroughResult(reply=reply, provider_used="codex")

    monkeypatch.setattr(passthrough, "run_turn", fake_run_turn)

    rc = cli.main(
        [
            "ask",
            "prompt",
            "--telegram",
            "--telegram-chat",
            str(FAKE_CHAT_ID),
            "--no-capture",
        ]
    )

    assert rc == 0
    messages = _FakeTelegramClient.instances[0].messages
    assert len(messages) == 2
    assert all(chat_id == str(FAKE_CHAT_ID) for chat_id, _text, _kwargs in messages)
    assert all(len(text) <= 4096 for _chat_id, text, _kwargs in messages)
    assert messages[0][1].startswith("&lt;")


def test_ask_telegram_error_sends_short_notice_no_traceback(monkeypatch, capsys):
    _install_fake_telegram(monkeypatch)

    async def fake_run_turn(*_args, **_kwargs):
        return passthrough.PassthroughResult(
            reply="",
            provider_used="codex",
            error='Traceback (most recent call last):\n  File "x.py", line 1\nRuntimeError: boom',
        )

    monkeypatch.setattr(passthrough, "run_turn", fake_run_turn)

    rc = cli.main(
        [
            "ask",
            "prompt",
            "--telegram",
            "--telegram-chat",
            str(FAKE_CHAT_ID),
            "--no-capture",
        ]
    )

    assert rc == 1
    capsys.readouterr()
    assert len(_FakeTelegramClient.instances) == 1
    [(chat_id, text, _kwargs)] = _FakeTelegramClient.instances[0].messages
    assert chat_id == str(FAKE_CHAT_ID)
    assert "Ask failed" in text
    assert "RuntimeError: boom" in text
    assert "Traceback" not in text
    assert "File " not in text


def test_ask_telegram_client_setup_failure_does_not_crash_or_retry(monkeypatch):
    from harness import telegram

    attempts = []

    class RaisingTelegramClient:
        def __init__(self):
            attempts.append("init")
            raise RuntimeError("telegram unavailable")

    async def fake_run_turn(*_args, **_kwargs):
        return passthrough.PassthroughResult(reply="ok", provider_used="codex")

    monkeypatch.setattr(telegram, "TelegramClient", RaisingTelegramClient)
    monkeypatch.setattr(passthrough, "run_turn", fake_run_turn)

    rc = cli.main(
        [
            "ask",
            "prompt",
            "--telegram",
            "--telegram-chat",
            str(FAKE_CHAT_ID),
            "--no-capture",
        ]
    )

    assert rc == 0
    assert attempts == ["init"]


def test_capture_default_writes_minimal_run_metadata_and_no_capture_writes_none(
    monkeypatch, _isolate_passthrough
):
    class FakeAgent:
        session_id = None

        async def run_async(self):
            return "captured reply"

    monkeypatch.setattr(
        roles, "build_role_agent", lambda *_args, **_kwargs: FakeAgent()
    )

    first = _run(
        passthrough.run_turn(
            "capture prompt",
            provider_chain=["codex"],
            model="m",
            effort="low",
        )
    )
    assert first.error is None

    run_dirs = [p for p in _isolate_passthrough.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "prompt.txt").read_text(encoding="utf-8") == "capture prompt"
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "ask"
    assert meta["provider"] == "codex"
    assert meta["model"] == "m"
    assert meta["effort"] == "low"
    assert meta["success"] is True
    assert meta["reply_chars"] == len("captured reply")
    assert isinstance(meta["duration_s"], float)

    for child in list(_isolate_passthrough.iterdir()):
        if child.is_dir():
            for nested in child.iterdir():
                nested.unlink()
            child.rmdir()

    second = _run(
        passthrough.run_turn(
            "no capture prompt",
            provider_chain=["codex"],
            capture=False,
        )
    )
    assert second.error is None
    assert [p for p in _isolate_passthrough.iterdir() if p.is_dir()] == []


# --------------------------------------------------------------------------- #
# Fuzz-driven regression tests (passthrough hardening, 2026-06-11)
# --------------------------------------------------------------------------- #
def test_concurrent_same_session_saves_do_not_crash_and_leave_valid_file():
    """A fixed .tmp name raced when the SAME session was saved concurrently
    (e.g. two rapid Telegram /chat turns); a unique per-write tmp fixes it."""
    import threading

    errors = []

    def _save():
        try:
            passthrough.SessionStore.save_session(
                passthrough.PassthroughSession(
                    session_id="race",
                    provider="codex",
                    provider_chain=["codex"],
                )
            )
        except Exception as exc:  # pragma: no cover - failure is the assertion target
            errors.append(repr(exc))

    threads = [threading.Thread(target=_save) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent same-session saves raced: {errors}"
    assert list(passthrough.SESSIONS_DIR.glob("*.tmp*")) == []
    loaded = passthrough.SessionStore.load_session("race")
    assert loaded is not None and loaded.provider == "codex"


def test_load_session_with_missing_optional_fields_loads_with_none_defaults():
    """Optional fields default to None so a partial/older-schema session still
    loads instead of silently reading as None (missing required defaults)."""
    passthrough.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    (passthrough.SESSIONS_DIR / "old.json").write_text(
        json.dumps(
            {"session_id": "old", "provider": "claude", "provider_chain": ["claude"]}
        ),
        encoding="utf-8",
    )
    loaded = passthrough.SessionStore.load_session("old")
    assert loaded is not None
    assert loaded.provider == "claude"
    assert loaded.model is None
    assert loaded.effort is None
    assert loaded.provider_session_id is None
    assert loaded.turns == []


def test_from_dict_rejects_non_dict_cleanly_and_load_returns_none():
    with pytest.raises(ValueError):
        passthrough.PassthroughSession.from_dict(None)
    with pytest.raises(ValueError):
        passthrough.PassthroughSession.from_dict(123)
    passthrough.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    (passthrough.SESSIONS_DIR / "scalar.json").write_text("123", encoding="utf-8")
    assert passthrough.SessionStore.load_session("scalar") is None
    (passthrough.SESSIONS_DIR / "listy.json").write_text("[]", encoding="utf-8")
    assert passthrough.SessionStore.load_session("listy") is None


@pytest.mark.parametrize("punct", ["!@#", "---", "***", "  $$  "])
def test_session_slug_rejects_punctuation_only_names(punct):
    """Punctuation-only names previously all collapsed to dashes and COLLIDED on
    one file; they must now be rejected cleanly instead of silently aliasing."""
    with pytest.raises(ValueError):
        passthrough._sanitize_session_id(punct)


def test_session_slug_allows_names_with_alphanumerics():
    assert passthrough._sanitize_session_id("My Session 1") == "my-session-1"
    assert passthrough._sanitize_session_id("tg-12345") == "tg-12345"
    assert passthrough._sanitize_session_id("a_b-c") == "a_b-c"

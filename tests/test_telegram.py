"""Hermetic tests for the Telegram build-progress bot (no network, no real token).

The HTTP layer is monkeypatched so no real request is ever made; outgoing
payloads are captured in a list. The whitelist + state files are tmp fixtures
with FAKE ids — never the real /home/leah/tgbot/data/users.json.
"""
from __future__ import annotations

import json

import pytest

from harness import telegram as tg
from harness import telegram_bot as bot


FAKE_TOKEN = "123456:FAKE-TEST-TOKEN-not-real"
FAKE_USER_ID = 111222333
FAKE_CHAT_ID = 444555666


@pytest.fixture(autouse=True)
def isolate_real_paths(tmp_path, monkeypatch):
    """Hermetic isolation: every test resolves telegram state to a tmp file (never
    the real out-of-repo DEFAULT_STATE_PATH) and asserts neither real default path
    is ever opened. Whitelist defaults are also redirected to a non-existent tmp
    file so a test without ``users_file`` can't read the real users.json.
    """
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
    # The whitelisted-user tests run /status (read-only) but stay one command away
    # from save_state(); pin state to a tmp file so no real path is reachable.
    monkeypatch.setenv("AGY_TELEGRAM_STATE", str(tmp_path / "bot_state.json"))
    return p


@pytest.fixture
def capture_client(monkeypatch):
    """A TelegramClient whose _post is replaced to capture payloads."""
    sent = []

    def fake_post(self, method, params, *, timeout=None):
        sent.append({"method": method, "params": dict(params)})
        if method == "getUpdates":
            return {"ok": True, "result": []}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(tg.TelegramClient, "_post", fake_post)
    return sent


# --------------------------------------------------------------------------- #
# Whitelist loading
# --------------------------------------------------------------------------- #
def test_whitelist_present(users_file):
    entries = tg.load_whitelist()
    assert len(entries) == 1
    assert tg.whitelist_chat_ids(entries) == [FAKE_CHAT_ID]
    assert tg.whitelist_user_ids(entries) == {FAKE_USER_ID}


def test_whitelist_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(tmp_path / "nope.json"))
    assert tg.load_whitelist() == []


def test_whitelist_garbage(tmp_path, monkeypatch):
    p = tmp_path / "users.json"
    p.write_text("{not json at all", encoding="utf-8")
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(p))
    assert tg.load_whitelist() == []


def test_whitelist_falls_back_to_id(tmp_path, monkeypatch):
    p = tmp_path / "users.json"
    p.write_text(json.dumps([{"id": 999, "username": "x"}]), encoding="utf-8")
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(p))
    entries = tg.load_whitelist()
    assert tg.whitelist_chat_ids(entries) == [999]


# --------------------------------------------------------------------------- #
# Verbosity gating
# --------------------------------------------------------------------------- #
def _event_stream():
    # Outcome strings are the EXACT ones master/adversarial emit on the
    # orchestration stream (verified / approved / verifier_timeout / stalled /
    # continue) — NOT synthetic placeholders.
    return [
        {"kind": "lifecycle", "run_id": "R", "data": {"event": "dispatch_started"}},
        {"kind": "lifecycle", "run_id": "R",
         "data": {"orchestration": {"phase": "step", "action": "started",
                                    "step_index": 1, "step_total": 3, "step_title": "do x"}}},
        {"kind": "lifecycle", "run_id": "R",
         "data": {"orchestration": {"phase": "step", "action": "completed",
                                    "step_index": 1, "step_total": 3, "step_title": "do x",
                                    "outcome": "verified", "verified": True}}},
        {"kind": "lifecycle", "run_id": "R",
         "data": {"orchestration": {"phase": "step", "action": "completed",
                                    "step_index": 2, "step_total": 3,
                                    "outcome": "verifier_timeout"}}},
        {"kind": "lifecycle", "run_id": "R",
         "data": {"orchestration": {"phase": "adversarial", "action": "iteration_completed",
                                    "iteration": 1, "iteration_total": 5, "approved": True}}},
        {"kind": "lifecycle", "run_id": "R",
         "data": {"orchestration": {"phase": "plan", "action": "completed"}}},
        {"kind": "heartbeat", "run_id": "R", "data": {"step": "2/3", "elapsed_s": 42}},
        {"kind": "usage", "run_id": "R", "worker": "codex",
         "data": {"usage_kind": "call", "total_tokens": 1234}},
        {"kind": "stderr", "run_id": "R", "text": "noise"},
        {"kind": "lifecycle", "run_id": "R", "data": {"event": "dispatch_finished"}},
    ]


def _rendered_count(verbosity):
    return sum(
        1 for ev in _event_stream()
        if tg.render_event(ev, verbosity=verbosity, mode="master", run_id="R") is not None
    )


def test_gating_quiet_only_lifecycle():
    # dispatch_started only (dispatch_finished is owned by the summary card now)
    assert _rendered_count("quiet") == 1


def test_gating_normal_adds_steps_and_failures():
    # quiet 1 + step started + step completed(verified) + verifier_timeout = 4
    assert _rendered_count("normal") == 4


def test_gating_verbose_adds_iteration_and_plan():
    # normal 4 + adversarial iteration + plan transition = 6
    assert _rendered_count("verbose") == 6


def test_gating_debug_adds_heartbeat_and_usage():
    # verbose 6 + heartbeat + usage = 8 (stderr never forwarded)
    assert _rendered_count("debug") == 8
    # stderr is never rendered at any level
    stderr_ev = {"kind": "stderr", "data": {}, "text": "x"}
    assert tg.render_event(stderr_ev, verbosity="debug", mode="m", run_id="R") is None


def test_dispatch_finished_is_silent():
    # The terse stream marker was dropped — the summary card owns end-of-run.
    ev = {"kind": "lifecycle", "run_id": "R", "data": {"event": "dispatch_finished"}}
    assert tg.render_event(ev, verbosity="quiet", mode="m", run_id="R") is None


def test_infra_outcome_renders_distinct_glyph():
    # An infra failure (#50) must NOT read as a benign step success.
    for outcome, needle in (
        ("verifier_timeout", "timed out"),
        ("verifier_resource_exceeded", "OOM"),
        ("verifier_infra_failed", "infra"),
    ):
        ev = {"kind": "lifecycle", "run_id": "R",
              "data": {"orchestration": {"phase": "step", "action": "completed",
                                         "step_index": 2, "step_total": 3,
                                         "outcome": outcome}}}
        msg = tg.render_event(ev, verbosity="normal", mode="master", run_id="R")
        assert msg is not None and needle in msg
        assert "Step 2 done" not in msg  # routed to the failure banner, not ✅


def test_step_completed_without_outcome_is_neutral():
    # A completed step with no terminal signal must not claim a false ✅.
    ev = {"kind": "lifecycle", "run_id": "R",
          "data": {"orchestration": {"phase": "step", "action": "completed",
                                     "step_index": 1, "step_total": 2,
                                     "step_title": "x"}}}
    msg = tg.render_event(ev, verbosity="normal", mode="master", run_id="R")
    assert msg is not None
    assert "✅" not in msg and "▪" in msg


def test_progress_lines_carry_run_tag():
    # step/adversarial/plan lines must be attributable to a run.
    started = {"kind": "lifecycle", "run_id": "abcdef123456",
               "data": {"orchestration": {"phase": "step", "action": "started",
                                          "step_index": 1, "step_total": 2}}}
    msg = tg.render_event(started, verbosity="normal", mode="m", run_id="abcdef123456")
    assert "123456" in msg  # last-6 run tag present


# --------------------------------------------------------------------------- #
# HTML escaping
# --------------------------------------------------------------------------- #
def test_html_escaping_of_titles():
    ev = {"kind": "lifecycle", "data": {"orchestration": {
        "phase": "step", "action": "started", "step_index": 1, "step_total": 2,
        "step_title": "<script>alert('x')</script> & friends"}}}
    msg = tg.render_event(ev, verbosity="normal", mode="m", run_id="R")
    assert "<script>" not in msg
    assert "&lt;script&gt;" in msg
    assert "&amp; friends" in msg


# --------------------------------------------------------------------------- #
# Notifier resilience + summary
# --------------------------------------------------------------------------- #
def test_notifier_never_raises_when_send_fails(fake_key, monkeypatch):
    def boom(self, *a, **k):
        raise RuntimeError("telegram down")

    monkeypatch.setattr(tg.TelegramClient, "send_message", boom)
    client = tg.TelegramClient()
    notifier = tg.TelegramNotifier(
        run_id="R", mode="master", verbosity="debug",
        client=client, chat_ids=[FAKE_CHAT_ID],
    )
    # Must not raise even though every send blows up.
    notifier({"kind": "lifecycle", "data": {"event": "dispatch_started"}})
    notifier.finished({"success": True, "duration_s": 5, "mode": "master", "run_id": "R"})


def test_notifier_broadcasts_to_all_chats(fake_key, capture_client):
    client = tg.TelegramClient()
    notifier = tg.TelegramNotifier(
        run_id="R", mode="direct", verbosity="quiet",
        client=client, chat_ids=[1, 2],
    )
    notifier({"kind": "lifecycle", "data": {"event": "dispatch_started"}})
    notifier.flush(timeout=5.0)  # delivery is off-loop now
    sends = [s for s in capture_client if s["method"] == "sendMessage"]
    assert len(sends) == 2
    assert {s["params"]["chat_id"] for s in sends} == {1, 2}
    assert "Build started" in sends[0]["params"]["text"]


def test_final_summary_success_formatting(fake_key, capture_client):
    client = tg.TelegramClient()
    notifier = tg.TelegramNotifier(
        run_id="R", mode="master", verbosity="quiet",
        client=client, chat_ids=[FAKE_CHAT_ID],
    )
    meta = {
        "success": True, "mode": "master", "duration_s": 125,
        "run_id": "R", "changed_files": ["a.py", "b.py"],
        "quality": {"confidence": "verified"},
        "tokens": {"grand_total": {"total_tokens": 4242}},
    }
    notifier.finished(meta)
    text = capture_client[-1]["params"]["text"]
    assert "Build verified" in text  # verified confidence -> distinct headline
    assert "verified" in text
    assert "2 files" in text
    assert "4,242 tok" in text


def test_final_summary_failure_formatting(fake_key, capture_client):
    client = tg.TelegramClient()
    notifier = tg.TelegramNotifier(
        run_id="R", mode="direct", verbosity="quiet",
        client=client, chat_ids=[FAKE_CHAT_ID],
    )
    notifier.finished({"success": False, "mode": "direct", "duration_s": 9,
                       "run_id": "R", "error": "verifier failed <hard>"})
    text = capture_client[-1]["params"]["text"]
    assert "Build failed" in text
    assert "&lt;hard&gt;" in text  # escaped


def test_notifier_inactive_without_chats(fake_key, capture_client):
    client = tg.TelegramClient()
    notifier = tg.TelegramNotifier(
        run_id="R", mode="m", verbosity="debug", client=client, chat_ids=[],
    )
    assert notifier.active is False
    notifier({"kind": "lifecycle", "data": {"event": "dispatch_started"}})
    assert capture_client == []


# --------------------------------------------------------------------------- #
# Bot command handlers
# --------------------------------------------------------------------------- #
def test_handle_status_in_progress(monkeypatch, tmp_path):
    rd = tmp_path / "runs"
    rd.mkdir()
    run = rd / "20260602_000001"
    run.mkdir()
    # A real in-progress run streams events.jsonl (touched at dispatch start) and
    # has no meta.json yet; liveness now requires a recent event stream so a stale
    # leftover dir can't be reported as a perpetual "in progress" (issue: phantom
    # /status). A freshly-written events file is live.
    (run / "events.jsonl").write_text('{"event_type":"x","run_id":"20260602_000001"}\n')
    monkeypatch.setattr(bot, "RUNS_DIR", rd)
    msg = bot.handle_command("/status", state={})
    assert "in progress" in msg.lower()


def test_handle_status_with_meta(monkeypatch, tmp_path):
    rd = tmp_path / "runs"
    rd.mkdir()
    run = rd / "20260602_000002"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({
        "success": True, "mode": "master", "duration_s": 30,
        "changed_files": ["x.py"], "quality": {"confidence": "verified"},
    }), encoding="utf-8")
    monkeypatch.setattr(bot, "RUNS_DIR", rd)
    msg = bot.handle_command("/status", state={})
    assert "OK" in msg
    assert "verified" in msg


def test_handle_runs(monkeypatch, tmp_path):
    rd = tmp_path / "runs"
    rd.mkdir()
    for i in range(3):
        run = rd / f"2026060200000{i}"
        run.mkdir()
        (run / "meta.json").write_text(json.dumps({
            "success": i % 2 == 0, "mode": "direct", "duration_s": i,
        }), encoding="utf-8")
    monkeypatch.setattr(bot, "RUNS_DIR", rd)
    msg = bot.handle_command("/runs 2", state={})
    assert "Recent runs" in msg
    assert msg.count("\n") == 2  # header + 2 rows


def test_handle_verbosity_show_and_set(tmp_path):
    state_file = str(tmp_path / "state.json")
    # show default
    msg = bot.handle_command("/verbosity", state={}, state_file=state_file)
    assert "normal" in msg
    # set
    st = {}
    msg = bot.handle_command("/verbosity verbose", state=st, state_file=state_file)
    assert "verbose" in msg
    assert json.loads(open(state_file).read())["verbosity"] == "verbose"
    # invalid
    msg = bot.handle_command("/verbosity bogus", state={}, state_file=state_file)
    assert "Unknown" in msg


def test_handle_start_and_help():
    assert "Connected" in bot.handle_command("/start", state={})
    assert "/status" in bot.handle_command("/help", state={})


def test_non_command_ignored():
    assert bot.handle_command("just chatting", state={}) is None


def test_bot_rejects_non_whitelisted_user(fake_key, users_file, capture_client):
    daemon = bot.BotDaemon(client=tg.TelegramClient())
    # Sender id is NOT in the whitelist -> no reply sent.
    daemon._process_update({
        "update_id": 1,
        "message": {"from": {"id": 999999}, "chat": {"id": 999999}, "text": "/status"},
    })
    assert [s for s in capture_client if s["method"] == "sendMessage"] == []


def test_bot_accepts_whitelisted_user(fake_key, users_file, capture_client, monkeypatch, tmp_path):
    rd = tmp_path / "runs"
    rd.mkdir()
    monkeypatch.setattr(bot, "RUNS_DIR", rd)
    daemon = bot.BotDaemon(client=tg.TelegramClient())
    daemon._process_update({
        "update_id": 1,
        "message": {"from": {"id": FAKE_USER_ID}, "chat": {"id": FAKE_CHAT_ID},
                    "text": "/status"},
    })
    sends = [s for s in capture_client if s["method"] == "sendMessage"]
    assert len(sends) == 1
    assert sends[0]["params"]["chat_id"] == FAKE_CHAT_ID


def test_client_send_swallows_failure(fake_key, monkeypatch):
    # Even if urlopen explodes, send_message returns None (never raises).
    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(tg.urllib.request, "urlopen", boom)
    client = tg.TelegramClient()
    assert client.send_message(1, "hi") is None


# --------------------------------------------------------------------------- #
# Whitelist wrong-shape (valid JSON, wrong type) coverage
# --------------------------------------------------------------------------- #
def test_whitelist_dict_not_list(tmp_path, monkeypatch):
    p = tmp_path / "users.json"
    p.write_text(json.dumps({"id": 1, "last_chat_id": 10}), encoding="utf-8")
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(p))
    assert tg.load_whitelist() == []


def test_whitelist_mixed_entries_filtered(tmp_path, monkeypatch):
    p = tmp_path / "users.json"
    p.write_text(
        json.dumps([{"id": 1, "last_chat_id": 10}, "x", 42, {"no_ids": True}]),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(p))
    entries = tg.load_whitelist()
    assert len(entries) == 1
    assert tg.whitelist_chat_ids(entries) == [10]
    assert tg.whitelist_user_ids(entries) == {1}


# --------------------------------------------------------------------------- #
# Per-call token formatting (debug)
# --------------------------------------------------------------------------- #
def test_usage_token_thousands_separator():
    ev = {"kind": "usage", "run_id": "R", "worker": "codex",
          "data": {"usage_kind": "call", "total_tokens": 184231}}
    msg = tg.render_event(ev, verbosity="debug", mode="m", run_id="R")
    assert "184,231 tok" in msg


# --------------------------------------------------------------------------- #
# BotDaemon.poll_once offset advancement + end-to-end whitelisted command
# --------------------------------------------------------------------------- #
def test_poll_once_advances_offset(fake_key, users_file, monkeypatch):
    batches = [
        [{"update_id": 7, "message": {"from": {"id": FAKE_USER_ID},
                                      "chat": {"id": FAKE_CHAT_ID}, "text": "/help"}}],
        [],
    ]
    seen_offsets = []

    def fake_get_updates(self, offset=None, timeout=25):
        seen_offsets.append(offset)
        return batches.pop(0) if batches else []

    sends = []
    monkeypatch.setattr(tg.TelegramClient, "get_updates", fake_get_updates)
    monkeypatch.setattr(
        tg.TelegramClient, "send_message",
        lambda self, cid, text, **k: sends.append((cid, text)),
    )
    daemon = bot.BotDaemon(client=tg.TelegramClient())
    assert daemon.poll_once() == 1
    assert daemon._offset == 8  # update_id + 1
    assert daemon.poll_once() == 0  # empty batch -> nothing
    assert seen_offsets == [None, 8]
    assert len(sends) == 1  # the /help reply only once (no reprocessing)


def test_verbosity_set_end_to_end_for_whitelisted_user(fake_key, users_file, tmp_path,
                                                       monkeypatch):
    state_file = str(tmp_path / "bot_state.json")
    sends = []
    monkeypatch.setattr(
        tg.TelegramClient, "send_message",
        lambda self, cid, text, **k: sends.append((cid, text)),
    )
    daemon = bot.BotDaemon(client=tg.TelegramClient(), state_file=state_file)
    daemon._process_update({
        "update_id": 1,
        "message": {"from": {"id": FAKE_USER_ID}, "chat": {"id": FAKE_CHAT_ID},
                    "text": "/verbosity verbose"},
    })
    assert len(sends) == 1
    assert json.loads(open(state_file).read())["verbosity"] == "verbose"


# --------------------------------------------------------------------------- #
# Dispatch wiring: _build_telegram_notifier tri-state policy
# --------------------------------------------------------------------------- #
def test_build_notifier_disabled_returns_none(monkeypatch):
    from harness import dispatch
    monkeypatch.setenv("TELEGRAM_BOT_KEY", FAKE_TOKEN)
    n = dispatch._build_telegram_notifier(
        run_id="R", mode="m", enabled=False, verbosity=None)
    assert n is None


def test_build_notifier_auto_on_with_key_and_whitelist(fake_key, users_file):
    from harness import dispatch
    n = dispatch._build_telegram_notifier(
        run_id="R", mode="m", enabled=None, verbosity=None)
    assert n is not None
    assert n.active is True


def test_build_notifier_auto_off_without_key(monkeypatch):
    from harness import dispatch
    monkeypatch.delenv("TELEGRAM_BOT_KEY", raising=False)
    n = dispatch._build_telegram_notifier(
        run_id="R", mode="m", enabled=None, verbosity=None)
    assert n is None


def test_build_notifier_forced_missing_warns_and_stays_off(monkeypatch, caplog):
    from harness import dispatch
    monkeypatch.delenv("TELEGRAM_BOT_KEY", raising=False)
    with caplog.at_level("WARNING"):
        n = dispatch._build_telegram_notifier(
            run_id="R", mode="m", enabled=True, verbosity=None)
    assert n is None
    assert any("telegram" in r.message.lower() for r in caplog.records)

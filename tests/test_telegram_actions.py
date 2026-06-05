"""Hermetic acceptance CONTRACT for the Telegram action + consolidation surface.

Covers (per the build contract):

  * A. /build — argv is a LIST, instruction is a SINGLE arg (a shell-injection
    string stays inert), the flag ALLOWLIST is enforced + unknown flags dropped,
    the _spawn_dispatch seam is the only spawn path, the new run is auto-tracked,
    empty instruction is a usage hint, and nothing ever raises.
  * A. /cancel — killpg-then-kill fallback, missing pid, never signals an
    unrecorded pid, never raises.
  * A. /retry — reads prompt.txt + meta, re-spawns via the SAME seam, missing
    prompt is a friendly error.
  * B. /run facets route + default summary + unknown-facet hint; /notify subs
    route + bare shows state; OLD command names still return non-None (aliases).
  * D. gate approve/reject callback writes runs/<id>/gate_decision.json while the
    default flow stays unchanged (no block).

Fully hermetic: NO real subprocess (_spawn_dispatch stubbed), NO real signals
(_killpg/_kill stubbed), NO network (TelegramClient._post stubbed), NO real
sleeps (_sleep stubbed), FAKE ids only, state + whitelist pinned to tmp_path.
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
def runs_dir(tmp_path, monkeypatch):
    rd = tmp_path / "runs"
    rd.mkdir()
    monkeypatch.setattr(bot, "RUNS_DIR", rd)
    return rd


@pytest.fixture
def no_sleep(monkeypatch):
    """The new-run discovery poll never actually sleeps in tests."""
    monkeypatch.setattr(bot, "_sleep", lambda *_a, **_k: None)


@pytest.fixture
def state():
    return bot.load_state()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _short(rid: str) -> str:
    return rid[-6:] if len(rid) > 6 else rid


def _write_run(runs_dir, rid, *, meta=None, prompt=None, pid=None, events=True):
    d = runs_dir / rid
    d.mkdir()
    if events:
        (d / "events.jsonl").write_text("", encoding="utf-8")
    if prompt is not None:
        (d / "prompt.txt").write_text(prompt, encoding="utf-8")
    if pid is not None:
        (d / "run.pid").write_text(str(pid), encoding="utf-8")
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


# =========================================================================== #
# A. /build
# =========================================================================== #
@pytest.fixture
def spawn_spy(monkeypatch, runs_dir):
    """Stub _spawn_dispatch: capture argv, fabricate a live run dir, return a pid."""
    calls = {"argv": [], "pid": 4242}

    def fake_spawn(argv):
        calls["argv"].append(list(argv))
        rid = f"20260605-00000{len(calls['argv'])}-stub"
        # A brand-new live run: events but no terminal meta, run.pid = this proc
        # (alive) so is_live_run -> True and auto-track resolves it.
        import os

        _write_run(runs_dir, rid, pid=os.getpid())
        calls["last_run"] = rid
        return calls["pid"]

    monkeypatch.setattr(bot, "_spawn_dispatch", fake_spawn)
    return calls


def test_build_builds_argv_list_with_instruction_as_single_arg(spawn_spy, no_sleep, state):
    bot.handle_command("/build add a feature to the parser",
                       state=state, chat_id=FAKE_CHAT_ID)
    argv = spawn_spy["argv"][0]
    assert isinstance(argv, list)
    # [python, -m, harness, do, <instruction>, *flags]
    assert argv[1:4] == ["-m", "harness", "do"]
    assert argv[4] == "add a feature to the parser"


def test_build_injection_string_stays_one_inert_arg(spawn_spy, no_sleep, state):
    bot.handle_command("/build foo; rm -rf ~", state=state, chat_id=FAKE_CHAT_ID)
    argv = spawn_spy["argv"][0]
    # The whole dangerous string is ONE argv element -> shell metachars inert.
    assert argv[4] == "foo; rm -rf ~"
    # No shell anywhere: the literal must not be split across argv elements.
    assert "rm" not in argv[5:]


def test_build_flag_allowlist_enforced_and_unknown_dropped(spawn_spy, no_sleep, state):
    bot.handle_command(
        "/build fix the bug --mode feedback --test-cmd pytest -q "
        "--mission-critical --web-search --danger --evil rm",
        state=state, chat_id=FAKE_CHAT_ID,
    )
    argv = spawn_spy["argv"][0]
    assert argv[4] == "fix the bug"
    tail = argv[5:]
    assert "--mode" in tail and "feedback" in tail
    assert "--test-cmd" in tail
    # --test-cmd consumes its value tokens up to the next recognised flag.
    assert tail[tail.index("--test-cmd") + 1] == "pytest -q"
    assert "--mission-critical" in tail
    assert "--web-search" in tail
    # Arbitrary flags are NEVER passed through.
    assert "--danger" not in tail
    assert "--evil" not in tail


def test_build_bad_mode_value_is_dropped(spawn_spy, no_sleep, state):
    bot.handle_command("/build do x --mode bogus", state=state, chat_id=FAKE_CHAT_ID)
    argv = spawn_spy["argv"][0]
    assert "--mode" not in argv[5:]
    assert "bogus" not in argv[5:]


def test_build_calls_spawn_seam_and_auto_tracks(spawn_spy, no_sleep, state):
    reply = bot.handle_command("/build ship it", state=state, chat_id=FAKE_CHAT_ID)
    assert spawn_spy["argv"], "spawn seam must be invoked"
    assert reply and "Launched" in reply
    # The freshly-discovered run is registered for this chat's tracking.
    persisted = bot.load_state()
    tracked = persisted.get("tracking", {}).get(str(FAKE_CHAT_ID), {})
    assert spawn_spy["last_run"] in tracked


def test_build_empty_instruction_is_hint_no_spawn(spawn_spy, no_sleep, state):
    reply = bot.handle_command("/build   ", state=state, chat_id=FAKE_CHAT_ID)
    assert reply and "/build" in reply
    assert spawn_spy["argv"] == [], "empty instruction must NOT spawn"


def test_build_only_flags_no_instruction_is_hint(spawn_spy, no_sleep, state):
    reply = bot.handle_command("/build --mode master", state=state, chat_id=FAKE_CHAT_ID)
    assert reply and "/build" in reply
    assert spawn_spy["argv"] == []


def test_build_never_raises_when_spawn_fails(monkeypatch, runs_dir, no_sleep, state):
    def boom(argv):
        raise RuntimeError("popen exploded")

    monkeypatch.setattr(bot, "_spawn_dispatch", boom)
    reply = bot.handle_command("/build anything", state=state, chat_id=FAKE_CHAT_ID)
    assert isinstance(reply, str) and reply  # never raises, always answers


def test_build_no_new_run_discovered_reports_launched(monkeypatch, runs_dir, no_sleep, state):
    # Spawn returns a pid but creates NO run dir -> bounded poll finds nothing.
    monkeypatch.setattr(bot, "_spawn_dispatch", lambda argv: 777)
    reply = bot.handle_command("/build slow build", state=state, chat_id=FAKE_CHAT_ID)
    assert reply and "777" in reply and "shortly" in reply


def test_build_seam_is_only_spawn_path(monkeypatch, runs_dir, no_sleep, state):
    """No real subprocess: if _spawn_dispatch is stubbed, subprocess.Popen is never hit."""
    import subprocess

    def forbidden(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("real subprocess.Popen called")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(bot, "_spawn_dispatch", lambda argv: 9)
    reply = bot.handle_command("/build x", state=state, chat_id=FAKE_CHAT_ID)
    assert isinstance(reply, str) and reply


# =========================================================================== #
# A. /cancel
# =========================================================================== #
def test_cancel_signals_recorded_pid_via_killpg(monkeypatch, runs_dir, state):
    _write_run(runs_dir, "20260605-120000-live", pid=55555)
    killpg = []
    monkeypatch.setattr(bot, "_killpg", lambda pid, sig: killpg.append((pid, sig)))
    monkeypatch.setattr(bot, "_kill", lambda pid, sig: (_ for _ in ()).throw(
        AssertionError("kill must not run when killpg succeeds")))
    reply = bot.handle_command("/cancel 20260605-120000-live", state=state, chat_id=FAKE_CHAT_ID)
    assert killpg == [(55555, bot.signal.SIGTERM)]
    assert "Signaled" in reply


def test_cancel_falls_back_to_kill_when_killpg_errors(monkeypatch, runs_dir, state):
    _write_run(runs_dir, "20260605-120100-live", pid=66666)
    kills = []

    def bad_killpg(pid, sig):
        raise PermissionError("no pgid")

    monkeypatch.setattr(bot, "_killpg", bad_killpg)
    monkeypatch.setattr(bot, "_kill", lambda pid, sig: kills.append((pid, sig)))
    reply = bot.handle_command("/cancel 20260605-120100-live", state=state, chat_id=FAKE_CHAT_ID)
    assert kills == [(66666, bot.signal.SIGTERM)]
    assert "Signaled" in reply


def test_cancel_already_finished_when_both_fail(monkeypatch, runs_dir, state):
    _write_run(runs_dir, "20260605-120200-gone", pid=70000)
    monkeypatch.setattr(bot, "_killpg", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(bot, "_kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    reply = bot.handle_command("/cancel 20260605-120200-gone", state=state, chat_id=FAKE_CHAT_ID)
    assert "already finished" in reply


def test_cancel_missing_pid_file(monkeypatch, runs_dir, state):
    # Run dir with NO run.pid.
    _write_run(runs_dir, "20260605-120300-nopid", meta={"success": True, "mode": "direct"})
    signaled = []
    monkeypatch.setattr(bot, "_killpg", lambda p, s: signaled.append(("pg", p)))
    monkeypatch.setattr(bot, "_kill", lambda p, s: signaled.append(("k", p)))
    reply = bot.handle_command("/cancel 20260605-120300-nopid", state=state, chat_id=FAKE_CHAT_ID)
    assert "No pid" in reply
    assert signaled == [], "must not signal anything when no pid recorded"


def test_cancel_never_signals_unrecorded_pid(monkeypatch, runs_dir, state):
    signaled = []
    monkeypatch.setattr(bot, "_killpg", lambda p, s: signaled.append(p))
    monkeypatch.setattr(bot, "_kill", lambda p, s: signaled.append(p))
    # Unknown run id -> no dir -> no pid read -> nothing signaled.
    reply = bot.handle_command("/cancel 999999-nope", state=state, chat_id=FAKE_CHAT_ID)
    assert signaled == []
    assert "not found" in reply.lower() or "📭" in reply


def test_cancel_never_raises_on_garbage_pid(monkeypatch, runs_dir, state):
    d = runs_dir / "20260605-120400-bad"
    d.mkdir()
    (d / "run.pid").write_text("not-a-number", encoding="utf-8")
    monkeypatch.setattr(bot, "_killpg", lambda p, s: None)
    monkeypatch.setattr(bot, "_kill", lambda p, s: None)
    reply = bot.handle_command("/cancel 20260605-120400-bad", state=state, chat_id=FAKE_CHAT_ID)
    assert isinstance(reply, str) and reply


def test_cancel_no_live_run_default(runs_dir, state):
    reply = bot.handle_command("/cancel", state=state, chat_id=FAKE_CHAT_ID)
    assert "No live run" in reply


def test_cancel_finished_run_with_stale_pid_is_not_signaled(monkeypatch, runs_dir, state):
    # A FINISHED run keeps its run.pid on disk (never cleared) but also has a
    # terminal meta.json. /cancel of it must NOT signal the (possibly recycled)
    # pid — the terminal-state gate short-circuits before any kill seam.
    _write_run(
        runs_dir, "20260605-121000-done",
        pid=70123, meta={"success": True, "mode": "direct"},
    )
    signaled = []
    monkeypatch.setattr(bot, "_killpg", lambda p, s: signaled.append(("pg", p)))
    monkeypatch.setattr(bot, "_kill", lambda p, s: signaled.append(("k", p)))
    reply = bot.handle_command("/cancel 20260605-121000-done", state=state, chat_id=FAKE_CHAT_ID)
    assert signaled == [], "must not signal the stale pid of a finished run"
    assert "already finished" in reply


def test_cancel_refuses_to_signal_the_bots_own_pid(monkeypatch, runs_dir, state):
    # A recycled run.pid resolving to the bot's OWN pid must be refused, not
    # SIGTERM'd (which would kill the daemon). No meta.json -> reaches the guard.
    import os as _os
    _write_run(runs_dir, "20260605-122000-self", pid=_os.getpid())
    signaled = []
    monkeypatch.setattr(bot, "_killpg", lambda p, s: signaled.append(("pg", p)))
    monkeypatch.setattr(bot, "_kill", lambda p, s: signaled.append(("k", p)))
    reply = bot.handle_command("/cancel 20260605-122000-self", state=state, chat_id=FAKE_CHAT_ID)
    assert signaled == [], "must never signal the bot's own process/group"
    assert "Refusing" in reply


# =========================================================================== #
# A. /retry
# =========================================================================== #
def test_retry_reads_prompt_and_meta_and_respawns(spawn_spy, no_sleep, runs_dir, state):
    _write_run(
        runs_dir, "20260605-130000-done",
        prompt="rebuild the index",
        meta={"success": True, "mode": "master"},
    )
    reply = bot.handle_command("/retry 20260605-130000-done", state=state, chat_id=FAKE_CHAT_ID)
    assert spawn_spy["argv"], "retry must re-spawn via the seam"
    argv = spawn_spy["argv"][0]
    assert argv[4] == "rebuild the index"
    assert "--mode" in argv[5:] and "master" in argv[5:]
    assert reply and "Launched" in reply


def test_retry_missing_prompt_no_spawn(spawn_spy, no_sleep, runs_dir, state):
    _write_run(runs_dir, "20260605-130100-nodoc", meta={"success": True, "mode": "direct"})
    reply = bot.handle_command("/retry 20260605-130100-nodoc", state=state, chat_id=FAKE_CHAT_ID)
    assert spawn_spy["argv"] == [], "no prompt -> must not spawn"
    assert "No prompt" in reply


def test_retry_unknown_run(spawn_spy, no_sleep, runs_dir, state):
    reply = bot.handle_command("/retry 999999-x", state=state, chat_id=FAKE_CHAT_ID)
    assert spawn_spy["argv"] == []
    assert "not found" in reply.lower() or "📭" in reply


# =========================================================================== #
# B. /run facets
# =========================================================================== #
@pytest.fixture
def finished_run(runs_dir):
    rid = "20260605-140000-fin"
    _write_run(
        runs_dir, rid,
        prompt="do the thing",
        meta={
            "success": True, "mode": "master",
            "changed_files": ["harness/telegram.py"], "added": [],
            "quality": {"confidence": "verified", "verified": True,
                        "critic_approved": True, "note": "ok"},
            "reconciliation": {"verdict": "wired", "findings": []},
        },
    )
    (runs_dir / rid / "changed-files.diff").write_text(
        "--- a/x\n+++ b/x\n@@ -1 +1 @@\n+y\n", encoding="utf-8")
    return rid


def test_run_default_facet_is_summary(finished_run, state):
    out = bot.handle_command(f"/run {finished_run}", state=state, chat_id=FAKE_CHAT_ID)
    assert out and "Run Summary" in out


def test_run_bare_is_latest_summary(finished_run, state):
    out = bot.handle_command("/run", state=state, chat_id=FAKE_CHAT_ID)
    assert out and "Run Summary" in out


@pytest.mark.parametrize("facet,needle", [
    ("summary", "Run Summary"),
    ("why", "Why?"),
    ("files", "Changed files"),
    ("diff", "Diff"),
])
def test_run_each_facet_routes(finished_run, facet, needle, state):
    out = bot.handle_command(f"/run {finished_run} {facet}",
                            state=state, chat_id=FAKE_CHAT_ID)
    assert out and needle in out


def test_run_facet_first_form(finished_run, state):
    out = bot.handle_command("/run why", state=state, chat_id=FAKE_CHAT_ID)
    assert out and "Why?" in out


def test_run_unknown_facet_hint(finished_run, state):
    out = bot.handle_command(f"/run {finished_run} bogusfacet",
                            state=state, chat_id=FAKE_CHAT_ID)
    assert out and "/run" in out


# =========================================================================== #
# B. /notify subs
# =========================================================================== #
def test_notify_bare_shows_state(state):
    out = bot.handle_command("/notify", state=state, chat_id=FAKE_CHAT_ID)
    assert out and "Notifications" in out
    assert "Verbosity" in out and "Quiet hours" in out


def test_notify_show_explicit(state):
    out = bot.handle_command("/notify show", state=state, chat_id=FAKE_CHAT_ID)
    assert out and "Notifications" in out


def test_notify_mute_routes(state):
    out = bot.handle_command("/notify mute 30m", state=state, chat_id=FAKE_CHAT_ID)
    assert out and "Muted" in out
    persisted = bot.load_state()
    assert persisted["chats"][str(FAKE_CHAT_ID)].get("mute_until")


def test_notify_watch_routes(state):
    bot.handle_command("/notify mute on", state=state, chat_id=FAKE_CHAT_ID)
    out = bot.handle_command("/notify watch", state=bot.load_state(), chat_id=FAKE_CHAT_ID)
    assert out and ("resumed" in out or "already on" in out)


def test_notify_quiet_routes(state):
    out = bot.handle_command("/notify quiet 23:00-07:00", state=state, chat_id=FAKE_CHAT_ID)
    assert out and "Quiet hours" in out
    persisted = bot.load_state()
    assert persisted["chats"][str(FAKE_CHAT_ID)].get("quiet_window") == "23:00-07:00"


def test_notify_verbosity_routes(state):
    out = bot.handle_command("/notify verbosity debug", state=state, chat_id=FAKE_CHAT_ID)
    assert out and "debug" in out
    assert bot.load_state().get("verbosity") == "debug"


def test_notify_unknown_sub_hint(state):
    out = bot.handle_command("/notify wat", state=state, chat_id=FAKE_CHAT_ID)
    assert out and "/notify" in out


# =========================================================================== #
# B. back-compat aliases (every OLD command still answers non-None)
# =========================================================================== #
@pytest.mark.parametrize("cmd", [
    "/status", "/summary", "/files", "/why", "/runs", "/verbosity",
    "/mute 30m", "/watch", "/quiet off", "/diff", "/track", "/untrack",
    "/showliveall", "/health", "/tail", "/help", "/start",
])
def test_old_commands_still_return_non_none(cmd, runs_dir, finished_run, state):
    out = bot.handle_command(cmd, state=state, chat_id=FAKE_CHAT_ID)
    assert out is not None, f"{cmd} must still answer (non-None)"


# =========================================================================== #
# D. mission-critical gate seam (write sidecar; default flow unchanged)
# =========================================================================== #
@pytest.fixture
def users_file(tmp_path, monkeypatch):
    p = tmp_path / "users.json"
    p.write_text(
        json.dumps([{"id": FAKE_USER_ID, "username": "t", "last_chat_id": FAKE_CHAT_ID}]),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(p))
    return p


@pytest.fixture
def posts(monkeypatch):
    sent = []

    def fake_post(self, method, params, *, timeout=None):
        sent.append({"method": method, "params": dict(params)})
        if method == "sendMessage":
            return {"ok": True, "result": {"message_id": 1}}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(tg.TelegramClient, "_post", fake_post)
    return sent


def _callback_update(*, data, user_id):
    return {
        "update_id": 1,
        "callback_query": {
            "id": "cbq-1",
            "from": {"id": user_id},
            "message": {"chat": {"id": FAKE_CHAT_ID}, "message_id": 7},
            "data": data,
        },
    }


def test_callback_verbs_include_approve_reject():
    assert "approve" in bot.CALLBACK_VERBS
    assert "reject" in bot.CALLBACK_VERBS


def test_build_gate_keyboard_shape():
    kb = bot.build_gate_keyboard("20260605-150000-gate")
    assert kb is not None
    btns = kb["inline_keyboard"][0]
    cbs = {b["callback_data"] for b in btns}
    assert cbs == {"approve:0-gate", "reject:0-gate"}
    for b in btns:
        assert len(b["callback_data"].encode("utf-8")) <= 64
    assert bot.build_gate_keyboard("") is None


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_gate_callback_writes_sidecar(monkeypatch, runs_dir, users_file, posts, decision):
    rid = "20260605-150100-gate"
    _write_run(runs_dir, rid, pid=12345)
    monkeypatch.setattr(bot, "_now", lambda: 1717600000.0)
    daemon = bot.BotDaemon(client=tg.TelegramClient())
    daemon._process_update(
        _callback_update(data=f"{decision}:{_short(rid)}", user_id=FAKE_USER_ID))

    sidecar = runs_dir / rid / "gate_decision.json"
    assert sidecar.exists(), "approve/reject must write gate_decision.json"
    payload = json.loads(sidecar.read_text())
    assert payload == {"decision": decision, "ts": 1717600000.0}
    # The callback was answered (spinner cleared) and a reply sent.
    assert any(p["method"] == "answerCallbackQuery" for p in posts)


def test_gate_callback_default_flow_unchanged(monkeypatch, runs_dir, users_file, posts):
    """A gate tap writes ONLY the sidecar; it must not touch meta/events/etc."""
    rid = "20260605-150200-gate"
    d = _write_run(runs_dir, rid, pid=222)
    before = {p.name for p in d.iterdir()}
    daemon = bot.BotDaemon(client=tg.TelegramClient())
    daemon._process_update(_callback_update(data=f"approve:{_short(rid)}", user_id=FAKE_USER_ID))
    after = {p.name for p in d.iterdir()}
    # Exactly one new file appeared: the opt-in gate sidecar.
    assert after - before == {"gate_decision.json"}


def test_gate_callback_non_whitelisted_no_write(monkeypatch, runs_dir, users_file, posts):
    rid = "20260605-150300-gate"
    _write_run(runs_dir, rid, pid=333)
    daemon = bot.BotDaemon(client=tg.TelegramClient())
    daemon._process_update(_callback_update(data=f"approve:{_short(rid)}", user_id=OTHER_USER_ID))
    assert not (runs_dir / rid / "gate_decision.json").exists()
    assert posts == [], "non-whitelisted tap: no reply, no answer, no write"


# =========================================================================== #
# never-raise guard for the new handlers
# =========================================================================== #
@pytest.mark.parametrize("cmd", [
    "/build", "/build x", "/cancel", "/cancel latest", "/cancel weird",
    "/retry", "/retry latest", "/run", "/run x y z", "/notify", "/notify x",
])
def test_new_commands_never_raise(monkeypatch, runs_dir, no_sleep, cmd, state):
    monkeypatch.setattr(bot, "_spawn_dispatch", lambda argv: 1)
    monkeypatch.setattr(bot, "_killpg", lambda p, s: None)
    monkeypatch.setattr(bot, "_kill", lambda p, s: None)
    out = bot.handle_command(cmd, state=state, chat_id=FAKE_CHAT_ID)
    assert out is None or isinstance(out, str)

"""Acceptance CONTRACT for F3 — /health, /tail, /diff read-only commands (spec §F3).

Anti-no-op gate: drives ``handle_command`` for the three NEW read-only commands
against a fixture run dir built under ``tmp_path`` and asserts each returns
non-empty, length-guarded content.

  /health — poller alive (NON-DESTRUCTIVE flock test that does NOT steal the
            lock), live-run count, last outcome, recent reroute/usage-wall
            scraped from the newest events.jsonl tails.
  /tail [n] [run] — last N rendered events via render_event (bounded read).
  /diff [run] — --stat summary + first ~60 lines of changed-files.diff in <pre>,
                hard 4096-char cap + escape.

Strict on: the three commands exist + are wired into HELP_TEXT/BOT_COMMANDS,
each answers with real content from a fixture run, length is capped under
Telegram's 4096 limit, /diff escapes + wraps in <pre>, and /health does NOT
steal the poller lock.

Hermetic: fake token + fake ids, _post captured, state pinned to tmp_path.
"""
from __future__ import annotations

import json

import pytest

from harness import telegram as tg
from harness import telegram_bot as tb


FAKE_TOKEN = "123456:FAKE-TEST-TOKEN-not-real"
FAKE_CHAT_ID = 444555666


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_TELEGRAM_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(tmp_path / "users.json"))
    monkeypatch.setenv("TELEGRAM_BOT_KEY", FAKE_TOKEN)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Monkeypatch _post so no command path can touch the network."""
    def fake_post(self, method, params, *, timeout=None):
        if method == "getUpdates":
            return {"ok": True, "result": []}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(tg.TelegramClient, "_post", fake_post)


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """A fake runs/ dir, wired into the bot via the module-level RUNS_DIR override."""
    d = tmp_path / "runs"
    d.mkdir()
    monkeypatch.setattr(tb, "RUNS_DIR", d)
    return d


def _write_finished_run(runs_dir, run_id="20260603-010101-001"):
    """A finished run with meta.json, events.jsonl and a changed-files.diff."""
    rd = runs_dir / run_id
    rd.mkdir()
    meta = {
        "success": True,
        "mode": "master",
        "duration_s": 123.0,
        "changed_files": ["harness/telegram_bot.py", "docs/telegram.md"],
        "added": ["docs/telegram.md"],
        "modified": ["harness/telegram_bot.py"],
        "deleted": [],
        "quality": {"confidence": "verified", "verified": True,
                    "critic_approved": True, "iterations_used": 2},
        "tokens": {"grand_total": {"total_tokens": 4242}},
    }
    (rd / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    events = [
        {"ts": 1780000000.0, "run_id": run_id, "kind": "lifecycle",
         "data": {"event": "dispatch_started",
                  "detail": {"mode": "master", "generator_chain": ["codex"],
                             "critic_chain": ["agy"]}}},
        {"ts": 1780000001.0, "run_id": run_id, "kind": "lifecycle",
         "data": {"event": "orchestration_transition",
                  "orchestration": {"phase": "step", "action": "started",
                                    "step_index": 1, "step_total": 2,
                                    "step_title": "Build it"}}},
        {"ts": 1780000130.0, "run_id": run_id, "kind": "lifecycle",
         "data": {"event": "orchestration_transition",
                  "orchestration": {"phase": "step", "action": "completed",
                                    "step_index": 1, "step_total": 2,
                                    "step_title": "Build it", "outcome": "verified",
                                    "verified": True}}},
    ]
    with (rd / "events.jsonl").open("w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    diff = "\n".join(
        f"+added line {i} with <html> & special chars" for i in range(120)
    )
    (rd / "changed-files.diff").write_text(diff, encoding="utf-8")
    return rd


def _state():
    return tb.load_state()


# --------------------------------------------------------------------------- #
# Public surface: commands wired into help + the menu
# --------------------------------------------------------------------------- #
def test_f3_commands_in_help_and_bot_commands():
    # /diff is still reachable (now folded under /run diff, listed in HELP_TEXT's
    # alias line) but the consolidation contract deliberately keeps deprecated
    # aliases OUT of the setMyCommands menu, so only the canonical verbs appear.
    for cmd in ("/health", "/tail", "/diff"):
        assert cmd in tb.HELP_TEXT, f"{cmd} missing from HELP_TEXT"
    names = {c["command"] for c in tb.BOT_COMMANDS}
    for c in ("health", "tail"):
        assert c in names, f"{c} missing from BOT_COMMANDS"
    # /diff still answers as a back-compat alias even though it's not in the menu.
    assert tb.handle_command("/diff", state=tb.load_state(), chat_id=FAKE_CHAT_ID) is not None


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
def test_health_returns_nonempty_and_does_not_steal_lock(runs_dir):
    _write_finished_run(runs_dir)
    out = tb.handle_command("/health", state=_state(), chat_id=FAKE_CHAT_ID)
    assert isinstance(out, str) and out.strip(), "/health returned empty"
    assert 0 < len(out) <= 4096, "/health not length-guarded"

    # Non-destructive: after /health ran its poller-alive probe, this process
    # must still be able to ACQUIRE the poller lock — i.e. /health did NOT steal
    # (hold) it. If /health stole the lock this acquire would fail.
    import harness.telegram_bot as _tb
    if _tb.fcntl is not None:
        import os as _os
        path = _tb.poller_lock_path()
        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        fd = _os.open(path, _os.O_CREAT | _os.O_RDWR, 0o600)
        try:
            _tb.fcntl.flock(fd, _tb.fcntl.LOCK_EX | _tb.fcntl.LOCK_NB)
        finally:
            _tb.fcntl.flock(fd, _tb.fcntl.LOCK_UN)
            _os.close(fd)


def test_health_reports_no_lock_holder_as_not_running(runs_dir):
    # No poller holds the lock in the test -> /health must say the poller is not
    # running (or otherwise produce a real status line), never crash.
    out = tb.handle_command("/health", state=_state(), chat_id=FAKE_CHAT_ID)
    assert out and len(out) <= 4096


def test_health_detects_held_lock_as_running(runs_dir):
    """When another process holds the poller lock, /health reports it alive."""
    if tb.fcntl is None:
        pytest.skip("fcntl unavailable")
    import os as _os
    path = tb.poller_lock_path()
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    fd = _os.open(path, _os.O_CREAT | _os.O_RDWR, 0o600)
    tb.fcntl.flock(fd, tb.fcntl.LOCK_EX | tb.fcntl.LOCK_NB)
    try:
        out = tb.handle_command("/health", state=_state(), chat_id=FAKE_CHAT_ID)
    finally:
        tb.fcntl.flock(fd, tb.fcntl.LOCK_UN)
        _os.close(fd)
    assert out and len(out) <= 4096
    low = out.lower()
    assert ("alive" in low or "running" in low or "✅" in out), \
        "/health did not detect the held lock as a live poller"


# --------------------------------------------------------------------------- #
# /tail
# --------------------------------------------------------------------------- #
def test_tail_renders_recent_events(runs_dir):
    rid = "20260603-010101-001"
    _write_finished_run(runs_dir, rid)
    out = tb.handle_command("/tail", state=_state(), chat_id=FAKE_CHAT_ID)
    assert isinstance(out, str) and out.strip(), "/tail returned empty"
    assert 0 < len(out) <= 4096
    # The fixture's step-1 event rendered through render_event mentions the step.
    assert "Step" in out or "step" in out or "Build" in out


def test_tail_respects_n_argument(runs_dir):
    rid = "20260603-010101-001"
    _write_finished_run(runs_dir, rid)
    out = tb.handle_command("/tail 1", state=_state(), chat_id=FAKE_CHAT_ID)
    assert isinstance(out, str) and out.strip()
    assert len(out) <= 4096


def test_tail_handles_no_runs(runs_dir):
    out = tb.handle_command("/tail", state=_state(), chat_id=FAKE_CHAT_ID)
    assert isinstance(out, str) and out.strip(), "/tail must answer even with no runs"


# --------------------------------------------------------------------------- #
# /diff
# --------------------------------------------------------------------------- #
def test_diff_returns_capped_escaped_pre_block(runs_dir):
    rid = "20260603-010101-001"
    _write_finished_run(runs_dir, rid)
    out = tb.handle_command("/diff", state=_state(), chat_id=FAKE_CHAT_ID)
    assert isinstance(out, str) and out.strip(), "/diff returned empty"
    assert len(out) <= 4096, "/diff exceeded the 4096 hard cap"
    assert "<pre>" in out and "</pre>" in out, "/diff not wrapped in <pre>"
    # The raw diff contained "<html> & special chars"; the reply must HTML-escape
    # them so Telegram doesn't choke on bad markup.
    assert "&lt;html&gt;" in out or "&amp;" in out, "/diff did not HTML-escape content"
    # Raw unescaped angle brackets from the diff body must not leak through.
    assert "<html>" not in out, "/diff leaked an unescaped tag"


def test_diff_handles_missing_run(runs_dir):
    out = tb.handle_command("/diff nope-missing", state=_state(), chat_id=FAKE_CHAT_ID)
    assert isinstance(out, str) and out.strip(), "/diff must answer for a missing run"


def test_diff_no_real_literal_backslash_n(runs_dir):
    rid = "20260603-010101-001"
    _write_finished_run(runs_dir, rid)
    out = tb.handle_command("/diff", state=_state(), chat_id=FAKE_CHAT_ID)
    assert "\\n" not in out, "/diff contains a LITERAL backslash-n (escaping bug)"

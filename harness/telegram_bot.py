"""Minimal Telegram long-poll command daemon for AgentOrch.

Run it with::

    python -m harness.telegram_bot

It polls ``getUpdates``, enforces the whitelist by ``message.from.id`` (silently
ignoring anyone not listed), and answers a deliberately small command set:

    /start            greet + confirm this chat will receive build updates
    /help             list commands
    /status           summarize the most recent run (or "in progress")
    /runs [N=5]       compact list of recent runs
    /verbosity [lvl]  show or set the persisted default verbosity
    /track <target>   follow a live run's progress (latest | <run_id> | all)
    /untrack [target] stop following one run (or all)

The loop is resilient: one bad update or a failed send never crashes the
daemon, and SIGINT shuts down cleanly. Everything is stdlib only and the bot
token is read from ``TELEGRAM_BOT_KEY`` (never hardcoded). The persisted state
file lives OUTSIDE the repo (default ``/home/leah/tgbot/data/bot_state.json``;
override via ``AGY_TELEGRAM_STATE``).

TEMPORARY cross-process live-run tracking
-----------------------------------------
The bot is a SEPARATE process from any running ``harness do``, so it cannot
subscribe to that run's in-memory EventBus. To still follow a build live it
TAILS the on-disk ``runs/<id>/events.jsonl`` (the persisted live stream) on
every poll iteration, advancing a per-run byte cursor so only NEW lines are
emitted. This is EXPLICITLY TEMPORARY: once the Phase 3 singleton makes the
orchestrator resident in memory, any caller can subscribe to a live run
directly and ``/track`` becomes obsolete.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import fcntl  # POSIX-only; the singleton poller lock degrades gracefully without it
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

from harness.telegram import (
    DEFAULT_STATE_PATH,
    DEFAULT_VERBOSITY,
    VERBOSITY_ORDER,
    TelegramClient,
    TelegramNotifier,
    _e,
    _fmt_duration,
    _run_tag,
    _suppressed,
    load_whitelist,
    normalize_verbosity,
    parse_quiet_window,
    render_event,
    state_path,
    whitelist_user_ids,
)

logger = logging.getLogger(__name__)

# Command list registered with Telegram via setMyCommands so the client shows a
# command menu/autocomplete (issue #63: "it's not registering my commands"). Kept
# in sync with handle_command + HELP_TEXT.
BOT_COMMANDS: List[Dict[str, str]] = [
    {"command": "status", "description": "most recent run (or in-progress)"},
    {"command": "summary", "description": "rich recap card for a run"},
    {"command": "files", "description": "list files changed by a run"},
    {"command": "why", "description": "explain the verdict/reconciliation"},
    {"command": "runs", "description": "recent runs (default 5)"},
    {"command": "track", "description": "follow a live run (latest | <id> | all)"},
    {"command": "untrack", "description": "stop following a live run"},
    {"command": "verbosity", "description": "show or set notification verbosity"},
    {"command": "mute", "description": "silence progress updates (30m|2h|on)"},
    {"command": "watch", "description": "resume updates (unmute)"},
    {"command": "quiet", "description": "quiet hours HH:MM-HH:MM (or off)"},
    {"command": "health", "description": "poller status, live runs, last outcome"},
    {"command": "tail", "description": "last N rendered events (default 10)"},
    {"command": "diff", "description": "changed-files diff summary"},
    {"command": "help", "description": "list commands"},
]

# Resolved lazily from harness.dispatch on first use (keeps the bot's import
# graph light and avoids any import cycle). Tests override this attribute.
RUNS_DIR: Optional[Path] = None


# --------------------------------------------------------------------------- #
# State (persisted OUTSIDE the repo)
# --------------------------------------------------------------------------- #
def poller_lock_path(state_file: Optional[str] = None) -> str:
    """Path to the cross-process getUpdates singleton lock (next to the state file,
    OUTSIDE the repo). Telegram returns 409 Conflict if two processes long-poll
    getUpdates at once, so only ONE command poller — a standalone daemon OR a
    dispatch's embedded poller — may run; the flock on this file is the gate (#63).
    """
    return str(Path(state_path(state_file)).with_name("telegram_poller.lock"))


def load_state(path: Optional[str] = None) -> dict:
    p = state_path(path)
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(state: dict, path: Optional[str] = None) -> None:
    p = state_path(path)
    try:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
    except Exception as exc:
        logger.debug("telegram bot state save failed: %s", exc)


def get_verbosity(state: dict) -> str:
    return normalize_verbosity(state.get("verbosity") or DEFAULT_VERBOSITY)


def _fmt_age(timestamp: float) -> str:
    import time
    diff = time.time() - timestamp
    if diff < 60:
        return f"{int(diff)}s ago"
    m = int(diff / 60)
    if m < 60:
        return f"{m}m ago"
    h = int(m / 60)
    if h < 24:
        return f"{h}h ago"
    return f"{int(h / 24)}d ago"


# --------------------------------------------------------------------------- #
# Run summaries (read runs/<id>/meta.json)
# --------------------------------------------------------------------------- #
def _runs_dir() -> Path:
    """Resolve the runs directory, importing dispatch lazily.

    Kept light so importing the bot daemon does not eagerly pull in the whole
    dispatch module. Tests may override the module-level ``RUNS_DIR`` attribute;
    when set we honour it, otherwise we resolve it on demand.
    """
    override = globals().get("RUNS_DIR")
    if override is not None:
        return override
    from harness.dispatch import RUNS_DIR as _RUNS_DIR  # lazy, avoids import cost

    return _RUNS_DIR


def _run_dirs() -> List[Path]:
    runs_dir = _runs_dir()
    if not runs_dir.exists():
        return []
    dirs = [d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    # Order by most-recent activity (mtime), NOT lexically by name. A non-timestamp
    # run-id — a test/dev leftover like "r-red-1" or "cu-probe" — sorts lexically
    # ABOVE every real "YYYYMMDD-HHMMSS-…" run and would hijack "latest", so
    # /status reported a months-old meta-less dir as a perpetual "in progress".
    # mtime tracks the last event/heartbeat write, so a genuinely active run sorts
    # first and a stale dir sinks regardless of its name.
    def _mtime(d: Path) -> float:
        try:
            return d.stat().st_mtime
        except Exception:
            return 0.0

    return sorted(dirs, key=_mtime, reverse=True)


def _read_meta(run_dir: Path) -> Optional[dict]:
    meta = run_dir / "meta.json"
    if not meta.exists():
        return None
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# TEMPORARY cross-process live-run tracking (tail runs/<id>/events.jsonl)
#
# The bot is a SEPARATE process from any running `harness do`, so it cannot
# subscribe to that run's in-memory EventBus. It follows runs by tailing the
# persisted on-disk event stream. OBSOLETE once the Phase 3 singleton makes the
# orchestrator resident in memory (any caller can then subscribe directly).
# --------------------------------------------------------------------------- #
def _events_path(run_dir: Path) -> Path:
    return run_dir / "events.jsonl"


def _live_stale_seconds() -> float:
    """Max age (s) of the last event write for a run to still count as live.

    A real in-progress dispatch streams events and periodic heartbeats, so its
    events.jsonl mtime stays fresh; an aborted/abandoned run goes silent. Generous
    default (30 min) so a slow-but-alive step (e.g. a long pytest between
    heartbeats) is never dropped; tunable via AGY_TELEGRAM_LIVE_STALE_S (0 = off)."""
    try:
        return float(os.environ.get("AGY_TELEGRAM_LIVE_STALE_S", "1800") or 0)
    except Exception:
        return 1800.0


def _run_pid_alive(run_dir: Path) -> Optional[bool]:
    """Liveness of the dispatch that owns ``run_dir``, from its recorded run.pid.

    Returns True (process alive), False (recorded but dead — a stillborn/killed
    run), or None (no pid recorded, or it can't be checked → caller falls back to
    the recency gate). Same-host check (os.kill(pid, 0)); good enough for the
    single-workstation bot, conservative on any ambiguity (#67)."""
    try:
        raw = (run_dir / "run.pid").read_text(encoding="utf-8").strip()
    except Exception:
        return None  # no pid file (old run, or unreadable) -> unknown
    try:
        pid = int(raw)
    except Exception:
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
        return True            # signal 0 delivered -> alive
    except ProcessLookupError:
        return False           # no such process -> dead (deregister now)
    except PermissionError:
        return True            # exists but not ours -> alive
    except Exception:
        return None            # anything else -> unknown, fall back to recency


def is_live_run(run_dir: Path) -> bool:
    """True iff the run is ACTIVELY in progress: no terminal meta.json, and either
    its recorded dispatch PID is still alive (#67) or — when no PID is recorded —
    its event stream was written recently.

    ``meta.json`` is written only when a run FINISHES, so events-without-meta is
    how the bot discovers runs it never started. The PID check deregisters a
    stillborn/early-exiting/killed run the instant its process is gone; the recency
    gate is the fallback (and the fix for a stale dir reported as perpetually "in
    progress" — a dead run is not a live one).
    """
    try:
        if not run_dir.is_dir():
            return False
        if (run_dir / "meta.json").exists():
            return False  # finished -> terminal, never "in progress"
        # Definitive signal first: the owning process's liveness.
        pid_state = _run_pid_alive(run_dir)
        if pid_state is True:
            return True
        if pid_state is False:
            return False  # process gone, no terminal meta -> aborted/stillborn
        # No usable PID (old run): fall back to "events exist AND are recent".
        ev = _events_path(run_dir)
        if not ev.exists():
            return False
        stale = _live_stale_seconds()
        if stale > 0:
            import time as _t
            if (_t.time() - ev.stat().st_mtime) > stale:
                return False  # silent for too long -> aborted, not live
        return True
    except Exception:
        return False


def live_run_dirs() -> List[Path]:
    """All currently-live run dirs, newest first (by run-id / dir name sort)."""
    return [d for d in _run_dirs() if is_live_run(d)]


def latest_live_run() -> Optional[Path]:
    dirs = live_run_dirs()
    return dirs[0] if dirs else None


def _run_dir_by_id(run_id: str) -> Optional[Path]:
    if not run_id:
        return None
    d = _runs_dir() / run_id
    return d if d.is_dir() else None


def _short_tag(run_id: str) -> str:
    s = str(run_id or "").strip()
    return s[-6:] if len(s) > 6 else s


# Verbs the inline-keyboard buttons (and their callback_data) carry. callback_data
# is "verb:short_run_id" (<=64 bytes); the dispatch routes each verb to an
# existing read-only helper. Unknown verbs are a safe no-op.
CALLBACK_VERBS = ("files", "why", "diff")


def build_run_keyboard(run_id: str) -> Optional[dict]:
    """Inline-keyboard reply_markup for a run: [📂 Files] [❓ Why] [📊 Diff].

    callback_data is ``"verb:short_run_id"`` (the short tag keeps it well under
    Telegram's 64-byte cap). Returns None when there's no run id to attach to.
    """
    short = _short_tag(run_id)
    if not short:
        return None
    return {
        "inline_keyboard": [[
            {"text": "📂 Files", "callback_data": f"files:{short}"},
            {"text": "❓ Why", "callback_data": f"why:{short}"},
            {"text": "📊 Diff", "callback_data": f"diff:{short}"},
        ]]
    }


def _resolve_short_tag(tag: str) -> Optional[str]:
    """Map a short run tag (or full id) back to a full run id, or None.

    Buttons carry ``short_run_id`` (the last 6 chars). Match it against live and
    recent run dirs by full id or by short tag; prefer the most-recent on a tie.
    A full id that exists also resolves to itself. Never raises.
    """
    try:
        tag = str(tag or "").strip()
        if not tag:
            return None
        # Exact full-id hit first.
        if _run_dir_by_id(tag) is not None:
            return tag
        for d in _run_dirs():  # most-recent first
            if _short_tag(d.name) == tag:
                return d.name
    except Exception:
        return None
    return None


def _handle_diff(run_id: str) -> str:
    """Lightweight diff summary for the [📊 Diff] button.

    F1 only needs a routable, length-bounded reply here; the full /diff command
    is F3. Reuse the run summary as a stand-in so the button always answers with
    real content for an existing run.
    """
    return summarize_run(run_id)


def _tracking_state(state: dict) -> dict:
    """The per-chat tracking map: ``{chat_id_str: {run_id: byte_cursor}}``."""
    tracking = state.get("tracking")
    if not isinstance(tracking, dict):
        tracking = {}
        state["tracking"] = tracking
    return tracking


def _chat_tracking(state: dict, chat_id: Any) -> dict:
    tracking = _tracking_state(state)
    key = str(chat_id)
    entry = tracking.get(key)
    if not isinstance(entry, dict):
        entry = {}
        tracking[key] = entry
    return entry


def _event_class(ev: Any) -> str:
    """Coarse class of a raw event for F2 suppression: 'failure' for a failed /
    stalled step or a failed dispatch_finished (always delivered), else
    'progress'. Mirrors TelegramNotifier._event_kind. Never raises."""
    try:
        from harness.telegram import _orchestration, _FAILURE_OUTCOMES
        data = ev.get("data") if isinstance(ev, dict) else None
        data = data if isinstance(data, dict) else {}
        orch = _orchestration(ev) or {}
        if str(orch.get("outcome")) in _FAILURE_OUTCOMES:
            return "failure"
        if data.get("event") == "dispatch_finished" and data.get("success") is False:
            return "failure"
        return "progress"
    except Exception:
        return "progress"


def _chat_prefs(state: dict, chat_id: Any) -> dict:
    """The MUTABLE per-chat preference dict ``state["chats"][str(chat_id)]``.

    Creates the ``chats`` map / per-chat entry as needed (F2: ``mute_until`` +
    ``quiet_window`` live here). The dispatch-side reader uses
    :func:`harness.telegram.chat_prefs` (read-only) against the same shape."""
    chats = state.get("chats")
    if not isinstance(chats, dict):
        chats = {}
        state["chats"] = chats
    key = str(chat_id)
    prefs = chats.get(key)
    if not isinstance(prefs, dict):
        prefs = {}
        chats[key] = prefs
    return prefs


def _parse_mute_duration(spec: str) -> Optional[object]:
    """Map a /mute argument to a ``mute_until`` value, or None on garbage.

    ``on`` / ``""`` → indefinite mute (the literal string ``"on"``). A duration
    like ``30m`` / ``2h`` / ``90s`` / ``45`` (bare = minutes) → an absolute epoch
    ``now + duration``. Never raises."""
    try:
        import time as _t
        s = str(spec or "").strip().lower()
        if s in ("", "on"):
            return "on"
        unit = s[-1]
        if unit in ("s", "m", "h"):
            num = float(s[:-1])
            secs = num * {"s": 1, "m": 60, "h": 3600}[unit]
        else:
            num = float(s)  # bare number = minutes
            secs = num * 60
        if secs <= 0:
            return None
        return _t.time() + secs
    except Exception:
        return None


def _handle_mute(args: List[str], *, state: dict, chat_id: Any,
                 state_file: Optional[str]) -> str:
    """/mute [30m|2h|on] — silence progress chatter (failures + summary still go)."""
    spec = args[0] if args else "on"
    mute_until = _parse_mute_duration(spec)
    if mute_until is None:
        return ("⚠️ Bad duration. Use <code>/mute 30m</code>, <code>/mute 2h</code>, "
                "or <code>/mute on</code> (indefinite).")
    prefs = _chat_prefs(state, chat_id)
    prefs["mute_until"] = mute_until
    save_state(state, state_file)
    if mute_until == "on":
        when = "until you <code>/watch</code>"
    else:
        when = f"for {_e(str(spec).strip())}"
    return f"🔕 Muted {when}. Failures + the final summary still come through."


def _handle_watch(args: List[str], *, state: dict, chat_id: Any,
                  state_file: Optional[str]) -> str:
    """/watch — clear any mute (resume progress updates)."""
    prefs = _chat_prefs(state, chat_id)
    had = bool(prefs.get("mute_until"))
    prefs.pop("mute_until", None)
    save_state(state, state_file)
    return "🔔 Updates resumed." if had else "🔔 Not muted — updates already on."


def _handle_quiet(args: List[str], *, state: dict, chat_id: Any,
                  state_file: Optional[str]) -> str:
    """/quiet HH:MM-HH:MM (DND window) | /quiet off."""
    arg = (args[0].strip() if args else "")
    prefs = _chat_prefs(state, chat_id)
    if arg.lower() in ("off", ""):
        had = bool(prefs.get("quiet_window"))
        prefs.pop("quiet_window", None)
        save_state(state, state_file)
        return "🔔 Quiet hours cleared." if had else "🔔 No quiet hours set."
    if parse_quiet_window(arg) is None:
        return ("⚠️ Bad window. Use <code>/quiet 23:00-07:00</code> "
                "or <code>/quiet off</code>.")
    prefs["quiet_window"] = arg
    save_state(state, state_file)
    return (f"🌙 Quiet hours <b>{_e(arg)}</b>. Progress is held during this window; "
            "failures + the final summary still come through.")


def _read_new_events(events_file: Path, cursor: int) -> tuple:
    """Read events appended to ``events_file`` since byte offset ``cursor``.

    Returns ``(events, new_cursor)``. Bounded — only the tail past ``cursor`` is
    read, so the cursor advances and history is never re-sent. A missing /
    rotated / garbage file never raises: it yields ``([], cursor)`` (rotation,
    i.e. file shrank below the cursor, resets to read from 0). Individual
    malformed JSON lines are skipped.
    """
    try:
        size = events_file.stat().st_size
    except Exception:
        return [], cursor
    start = cursor
    if size < cursor:  # truncated / rotated -> re-read from the top
        start = 0
    if size <= start:
        return [], size
    events: List[dict] = []
    try:
        with events_file.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(start)
            chunk = fh.read()
            new_cursor = fh.tell()
    except Exception:
        return [], cursor
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue  # skip a single garbage line, keep tailing
        if isinstance(ev, dict):
            events.append(ev)
    return events, new_cursor


def summarize_latest() -> str:
    dirs = _run_dirs()
    if not dirs:
        return "📭 <b>No runs yet</b>"
    # Skip stale-incomplete leftovers (meta-less AND not live) so a dead dir can't
    # be reported as a perpetual "in progress". Report the first run (most recent
    # by mtime) that is either actively live or has finished (has meta.json).
    latest = None
    meta = None
    for d in dirs:
        m = _read_meta(d)
        if m is None:
            if is_live_run(d):
                return f"⏳ <b>Run in progress</b> · <code>{_e(d.name)}</code>"
            continue  # stale/aborted incomplete run — skip
        latest, meta = d, m
        break
    if meta is None:
        return "📭 <b>No completed runs yet</b>"
    success = bool(meta.get("success"))
    icon = "✅" if success else "❌"
    mode = _e(meta.get("mode") or "")
    dur = _fmt_duration(meta.get("duration_s"))
    changed = meta.get("changed_files")
    n = len(changed) if isinstance(changed, list) else 0
    quality = meta.get("quality") if isinstance(meta.get("quality"), dict) else {}
    conf = _e(quality.get("confidence") or "n/a")
    lines = [
        f"{icon} <b>{'OK' if success else 'FAIL'}</b> · <code>{_e(latest.name)}</code>",
        f"mode <code>{mode}</code> · {conf} · {_e(dur)} · {n} file{'s' if n != 1 else ''}",
    ]
    err = meta.get("error")
    if not success and err:
        lines.append(f"<i>{_e(err)}</i>")
    return "\n".join(lines)


def summarize_run(run_id: str) -> str:
    if run_id.lower() == "latest":
        d = latest_live_run() or (_run_dirs()[0] if _run_dirs() else None)
    else:
        d = _run_dir_by_id(run_id)
    if not d:
        return "📭 <b>Run not found</b>"
    meta = _read_meta(d)
    if not meta:
        return "⏳ <b>Run in progress</b> (no summary yet)"
    
    success = bool(meta.get("success"))
    mode = _e(meta.get("mode") or "")
    dur = _fmt_duration(meta.get("duration_s"))
    changed = meta.get("changed_files")
    n_files = len(changed) if isinstance(changed, list) else 0
    added = len(meta.get("added") or [])
    modified = len(meta.get("modified") or [])
    deleted = len(meta.get("deleted") or [])
    
    quality = meta.get("quality") or {}
    conf = _e(quality.get("confidence") or "n/a")
    verified = quality.get("verified")
    critic = quality.get("critic_approved")
    iters = quality.get("iterations_used")
    
    tokens = meta.get("tokens") or {}
    grand = tokens.get("grand_total") or {}
    total_tokens = grand.get("total_tokens") or 0
    
    recon = meta.get("reconciliation") or {}
    verdict = recon.get("verdict")
    
    icon = "✅" if success else "❌"
    
    lines = [
        f"{icon} <b>Run Summary</b> · <code>{_e(d.name)}</code>",
        f"mode <code>{mode}</code> · {conf} · {_e(dur)}",
        f"<b>Files</b>: {n_files} (+{added}/~{modified}/-{deleted})",
        f"<b>Quality</b>: verifier {'✅' if verified else '❌'} · critic {'✓' if critic else '✗'} · iters: {iters}",
    ]
    if verdict:
        lines.append(f"<b>Reconcile</b>: {_e(verdict)}")
    lines.append(f"<b>Tokens</b>: {total_tokens:,}")
    return "\n".join(lines)


def _handle_files(run_id: str) -> str:
    if run_id.lower() == "latest":
        d = latest_live_run() or (_run_dirs()[0] if _run_dirs() else None)
    else:
        d = _run_dir_by_id(run_id)
    if not d:
        return "📭 <b>Run not found</b>"
    meta = _read_meta(d)
    if not meta:
        return "⏳ <b>Run in progress</b> (no files yet)"
    changed = meta.get("changed_files") or []
    if not changed:
        return "No files changed."
    added = set(meta.get("added") or [])
    deleted = set(meta.get("deleted") or [])
    
    out = [f"📁 <b>Changed files</b> ({len(changed)})"]
    for i, f in enumerate(changed):
        if i >= 30:
            out.append(f"… +{len(changed) - 30} more")
            break
        glyph = "➕" if f in added else ("➖" if f in deleted else "📝")
        out.append(f"{glyph} <code>{_e(f)}</code>")
    return "\n".join(out)


def _handle_why(run_id: str) -> str:
    if run_id.lower() == "latest":
        d = latest_live_run() or (_run_dirs()[0] if _run_dirs() else None)
    else:
        d = _run_dir_by_id(run_id)
    if not d:
        return "📭 <b>Run not found</b>"
    meta = _read_meta(d)
    if not meta:
        return "⏳ <b>Run in progress</b>"
    
    quality = meta.get("quality") or {}
    note = quality.get("note") or ""
    verified = quality.get("verified")
    critic = quality.get("critic_approved")
    stalled = quality.get("stalled")
    
    recon = meta.get("reconciliation") or {}
    findings = recon.get("findings") or []
    verdict = recon.get("verdict")

    out = [f"🤔 <b>Why?</b> · <code>{_e(d.name)}</code>", ""]
    if note:
        out.append(f"<i>{_e(note)}</i>\n")

    v_str = "✅ verified" if verified else "❌ not verified"
    c_str = "✅ critic approved" if critic else "❌ critic rejected"
    s_str = "⚠️ stalled" if stalled else ""
    out.append(f"<b>Status</b>: {v_str} · {c_str} {s_str}")

    if verdict:
        out.append(f"<b>Reconcile</b>: {_e(verdict)}")
    
    if findings:
        out.append("\n<b>Findings:</b>")
        for f in findings:
            name = f.get("name")
            cls = f.get("classification")
            sk = f.get("sub_kind")
            out.append(f"• {_e(name)} · {_e(cls)} · {_e(sk)}")
            
    return "\n".join(out)


def summarize_runs(n: int = 5) -> str:
    dirs = _run_dirs()
    if not dirs:
        return "📭 <b>No runs yet</b>"
    try:
        n = max(1, min(int(n), 20))
    except Exception:
        n = 5
    out: List[str] = ["📜 <b>Recent runs</b>"]
    for d in dirs[:n]:
        meta = _read_meta(d)
        if meta is None:
            # #67: only call it "in progress" when the run is actually live (its
            # dispatch PID is alive, or — for an old run with no PID — its events
            # are recent). A stillborn / early-exited / killed run that never wrote
            # a terminal meta is shown as aborted, so it stops inflating the
            # open-run count forever.
            if is_live_run(d):
                try:
                    age = _fmt_age(d.stat().st_mtime)
                except Exception:
                    age = ""
                out.append(f"• <code>{_e(d.name)}</code> · in progress · {age}")
            else:
                out.append(f"⚠️ <code>{_e(d.name)}</code> · aborted (no result)")
            continue
        icon = "✅" if meta.get("success") else "❌"
        mode = _e(meta.get("mode") or "")
        dur = _fmt_duration(meta.get("duration_s"))
        quality = meta.get("quality") or {}
        raw_conf = quality.get("confidence")
        conf_str = f" · {_e(str(raw_conf).strip())}" if raw_conf else ""
        try:
            age = _fmt_age(d.stat().st_mtime)
        except Exception:
            age = ""
        out.append(f"{icon} <code>{_e(d.name)}</code> · {mode} · {_e(dur)}{conf_str} · {age}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Command handling (pure-ish: returns a reply string or None)
# --------------------------------------------------------------------------- #
HELP_TEXT = (
    "🤖 <b>AgentOrch build bot</b>\n"
    "/status — most recent run\n"
    "/summary [latest|<run_id>] — rich recap card\n"
    "/files [latest|<run_id>] — list changed files\n"
    "/why [latest|<run_id>] — explain the verdict\n"
    "/runs [N] — recent runs (default 5)\n"
    "/verbosity [level] — show or set default ("
    + " / ".join(VERBOSITY_ORDER)
    + ")\n"
    "/track [latest|&lt;run_id&gt;|all] — follow a live run (default: latest)\n"
    "/untrack [&lt;run_id&gt;|all] — stop following (default: all)\n"
    "/mute [30m|2h|on] — silence progress (failures + summary still come)\n"
    "/watch — resume updates (unmute)\n"
    "/quiet HH:MM-HH:MM — quiet hours (DND); /quiet off to clear\n"
    "/health — poller status, live runs, last outcome, recent signals\n"
    "/tail [N] [run] — last N rendered events (default 10, latest run)\n"
    "/diff [run] — changed-files diff summary + first lines\n"
    "/help — this message"
)


def _handle_track(args: List[str], *, state: dict, chat_id: Any,
                  state_file: Optional[str]) -> str:
    """/track latest | <run_id> | all — follow a live run's persisted stream."""
    target = (args[0].strip().lower() if args else "latest")
    entry = _chat_tracking(state, chat_id)
    added: List[str] = []

    if target in ("all", ""):
        for d in live_run_dirs():
            if d.name not in entry:
                entry[d.name] = 0
                added.append(d.name)
        if not entry:
            return "📭 No live runs right now."
    elif target == "latest":
        d = latest_live_run()
        if d is None:
            return "📭 No live runs right now."
        if d.name not in entry:
            entry[d.name] = 0
            added.append(d.name)
    else:
        # An explicit run id — allow tracking even a not-yet-discovered dir, but
        # only if the dir exists (avoids tracking a typo forever).
        rid = args[0].strip()
        d = _run_dir_by_id(rid)
        if d is None:
            return f"⚠️ Unknown run <code>{_e(rid)}</code>."
        if not is_live_run(d):
            return f"ℹ️ Run <code>{_e(rid)}</code> is not live (already finished?)."
        if rid not in entry:
            entry[rid] = 0
            added.append(rid)

    save_state(state, state_file)
    tracked = sorted(entry.keys())
    tags = ", ".join(f"<code>{_e(_short_tag(r))}</code>" for r in tracked)
    n = len(tracked)
    return f"▶️ Tracking {n} live run{'s' if n != 1 else ''}: {tags}"


def _handle_untrack(args: List[str], *, state: dict, chat_id: Any,
                    state_file: Optional[str]) -> str:
    """/untrack [<run_id>|all] — stop following one run (bare = all)."""
    entry = _chat_tracking(state, chat_id)
    target = (args[0].strip() if args else "all")
    if target.lower() == "all" or not target:
        had = len(entry)
        entry.clear()
        save_state(state, state_file)
        if not had:
            return "⏹ Not tracking any runs."
        return f"⏹ Stopped tracking all {had} run{'s' if had != 1 else ''}."
    # Match by full id or short tag.
    match = None
    if target in entry:
        match = target
    else:
        for rid in list(entry):
            if _short_tag(rid) == target:
                match = rid
                break
    if match is None:
        return f"ℹ️ Not tracking <code>{_e(target)}</code>."
    entry.pop(match, None)
    save_state(state, state_file)
    return f"⏹ Stopped tracking <code>{_e(_short_tag(match))}</code>."


# --------------------------------------------------------------------------- #
# F3 read-only commands: /health, /tail, /diff
# --------------------------------------------------------------------------- #
TELEGRAM_MSG_CAP = 4096  # Telegram's hard per-message limit.


def _poller_alive(state_file: Optional[str] = None) -> Optional[bool]:
    """NON-DESTRUCTIVE liveness probe for the singleton getUpdates poller.

    Tries to take the flock NON-BLOCKING: if the lock is FREE we grab it,
    immediately RELEASE it, and report the poller as NOT running (False). If the
    lock is already HELD by a daemon / embedded poller the acquire fails with
    EWOULDBLOCK and we report it ALIVE (True) — without ever stealing or holding
    the lock ourselves. Returns None when fcntl is unavailable (can't tell).
    Never raises.
    """
    if fcntl is None:
        return None
    path = poller_lock_path(state_file)
    fd = None
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True  # held by someone else -> a poller is alive
        # We acquired it -> nobody else held it -> no poller running. Release
        # IMMEDIATELY so we never hold (steal) the singleton lock.
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        return False
    except Exception:
        return None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass


def _scan_recent_signals(limit_runs: int = 5) -> List[str]:
    """Scrape the newest events.jsonl tails for reroute / usage-wall signals.

    Bounded: only the most-recent ``limit_runs`` run dirs, only the tail of each
    file. Returns a short de-duplicated list of human notes (possibly empty).
    Never raises."""
    notes: List[str] = []
    seen: set = set()
    try:
        for d in _run_dirs()[:limit_runs]:
            ev = _events_path(d)
            if not ev.exists():
                continue
            try:
                size = ev.stat().st_size
                with ev.open("r", encoding="utf-8", errors="replace") as fh:
                    if size > 16384:
                        fh.seek(size - 16384)
                        fh.readline()  # drop a partial line
                    tail = fh.read()
            except Exception:
                continue
            for line in tail.splitlines():
                low = line.lower()
                tag = None
                if "usage_wall" in low or "usage wall" in low:
                    tag = "usage-wall"
                elif "reroute" in low or "fallback" in low or "rotation" in low:
                    tag = "reroute/fallback"
                if tag and (d.name, tag) not in seen:
                    seen.add((d.name, tag))
                    notes.append(f"{tag} · <code>{_e(_short_tag(d.name))}</code>")
    except Exception:
        return notes
    return notes


def _handle_health(state_file: Optional[str] = None) -> str:
    """/health — poller liveness (non-destructive), live-run count, last outcome,
    recent reroute/usage-wall signals. Best-effort; never raises a crash out."""
    lines: List[str] = ["🩺 <b>Health</b>"]
    alive = _poller_alive(state_file)
    if alive is True:
        lines.append("Poller: ✅ running")
    elif alive is False:
        lines.append("Poller: ⚠️ not running")
    else:
        lines.append("Poller: ❔ unknown")

    try:
        live = live_run_dirs()
    except Exception:
        live = []
    n_live = len(live)
    lines.append(f"Live runs: <b>{n_live}</b>")

    # Last finished outcome (most-recent run that has a terminal meta.json).
    last_line = "Last outcome: <i>none yet</i>"
    try:
        for d in _run_dirs():
            m = _read_meta(d)
            if m is None:
                continue
            ok = bool(m.get("success"))
            icon = "✅" if ok else "❌"
            mode = _e(m.get("mode") or "")
            last_line = (f"Last outcome: {icon} {'OK' if ok else 'FAIL'} · "
                         f"<code>{_e(_short_tag(d.name))}</code> · {mode}")
            break
    except Exception:
        pass
    lines.append(last_line)

    signals = _scan_recent_signals()
    if signals:
        lines.append("Recent: " + ", ".join(signals[:6]))

    out = "\n".join(lines)
    return out[:TELEGRAM_MSG_CAP]


def _tail_events(run_dir: Path, n: int) -> List[dict]:
    """Last ``n`` events from a run's events.jsonl (bounded tail read). Never raises."""
    ev = _events_path(run_dir)
    if not ev.exists():
        return []
    try:
        size = ev.stat().st_size
        with ev.open("r", encoding="utf-8", errors="replace") as fh:
            # Read only a bounded tail (generous: ~64 KiB covers many events).
            if size > 65536:
                fh.seek(size - 65536)
                fh.readline()  # drop the partial first line
            chunk = fh.read()
    except Exception:
        return []
    out: List[dict] = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out[-n:] if n > 0 else out


def _handle_tail(args: List[str], *, state: dict) -> str:
    """/tail [n=10] [run] — last N rendered events via render_event (bounded)."""
    n = 10
    target = "latest"
    rest = list(args)
    # First numeric arg = N; first non-numeric = run id.
    consumed_n = False
    leftover: List[str] = []
    for a in rest:
        s = a.strip()
        if not consumed_n and s.isdigit():
            try:
                n = max(1, min(int(s), 30))
            except Exception:
                n = 10
            consumed_n = True
        else:
            leftover.append(s)
    if leftover:
        target = leftover[0]

    if target.lower() == "latest":
        d = latest_live_run() or (_run_dirs()[0] if _run_dirs() else None)
    else:
        d = _run_dir_by_id(target)
    if d is None:
        return "📭 <b>No runs to tail</b>"

    events = _tail_events(d, n)
    if not events:
        return f"📭 <b>No events yet</b> · <code>{_e(_short_tag(d.name))}</code>"

    verbosity = get_verbosity(state)
    out: List[str] = [f"📰 <b>Last {len(events)} events</b> · <code>{_e(_short_tag(d.name))}</code>"]
    for ev in events:
        try:
            msg = render_event(
                ev, verbosity=verbosity,
                mode=str(ev.get("data", {}).get("mode") or "")
                if isinstance(ev.get("data"), dict) else "",
                run_id=d.name,
            )
        except Exception:
            continue
        if msg:
            out.append(msg)
    if len(out) == 1:
        # Nothing rendered at this verbosity — still answer with a stub line.
        out.append("<i>(no events render at the current verbosity)</i>")
    text = "\n".join(out)
    if len(text) > TELEGRAM_MSG_CAP:
        text = text[:TELEGRAM_MSG_CAP - 1] + "…"
    return text


def _handle_diff_command(run_id: str, *, max_lines: int = 60) -> str:
    """/diff [run] — --stat-style summary + first ~60 lines of changed-files.diff
    in a <pre> block, HTML-escaped, hard-capped at 4096 chars. Never raises."""
    rid = (run_id or "latest").strip()
    if rid.lower() == "latest":
        d = latest_live_run() or (_run_dirs()[0] if _run_dirs() else None)
    else:
        d = _run_dir_by_id(rid)
    if d is None:
        return "📭 <b>Run not found</b>"

    diff_path = d / "changed-files.diff"
    if not diff_path.exists():
        return f"📭 <b>No diff</b> · <code>{_e(_short_tag(d.name))}</code>"

    try:
        text = diff_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return f"⚠️ <b>Could not read diff</b> · <code>{_e(_short_tag(d.name))}</code>"

    lines = text.splitlines()
    total = len(lines)
    # Lightweight --stat: count added/removed/changed-file markers.
    files = sum(1 for ln in lines if ln.startswith("diff --git") or ln.startswith("+++ "))
    added = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))

    head = (f"📊 <b>Diff</b> · <code>{_e(_short_tag(d.name))}</code>\n"
            f"~{files} file(s) · +{added}/-{removed} · {total} lines")
    body_lines = lines[:max_lines]
    truncated = total > max_lines
    body = "\n".join(_e(ln) for ln in body_lines)
    if truncated:
        body += f"\n… +{total - max_lines} more lines"

    out = f"{head}\n<pre>{body}</pre>"
    if len(out) > TELEGRAM_MSG_CAP:
        # Trim the <pre> body to fit, keeping the closing tag.
        budget = TELEGRAM_MSG_CAP - len(head) - len("\n<pre>") - len("</pre>") - 1
        if budget < 0:
            return head[:TELEGRAM_MSG_CAP]
        out = f"{head}\n<pre>{body[:budget]}…</pre>"
    return out


def handle_command(
    text: str,
    *,
    state: dict,
    state_file: Optional[str] = None,
    chat_id: Any = None,
) -> Optional[str]:
    """Map a whitelisted user's message to a reply. Returns None to ignore.

    ``chat_id`` is required for the per-chat /track and /untrack commands (the
    daemon passes the originating chat); other commands ignore it.
    """
    text = (text or "").strip()
    if not text.startswith("/"):
        return None
    parts = text.split()
    cmd = parts[0].split("@", 1)[0].lower()  # tolerate /cmd@BotName
    args = parts[1:]

    if cmd == "/start":
        return (
            "🟢 <b>Connected</b>\n"
            "This chat will receive AgentOrch build updates.\n\n" + HELP_TEXT
        )
    if cmd == "/help":
        return HELP_TEXT
    if cmd == "/status":
        return summarize_latest()
    if cmd == "/summary":
        return summarize_run(args[0] if args else "latest")
    if cmd == "/files":
        return _handle_files(args[0] if args else "latest")
    if cmd == "/why":
        return _handle_why(args[0] if args else "latest")
    if cmd == "/runs":
        return summarize_runs(args[0] if args else 5)
    if cmd == "/verbosity":
        if not args:
            return f"🔧 Default verbosity: <b>{_e(get_verbosity(state))}</b>"
        requested = args[0].strip().lower()
        if requested not in VERBOSITY_ORDER:
            return (
                f"⚠️ Unknown level <code>{_e(requested)}</code>. "
                f"Choose: {', '.join(VERBOSITY_ORDER)}"
            )
        state["verbosity"] = requested
        save_state(state, state_file)
        return f"✅ Default verbosity set to <b>{_e(requested)}</b>"
    if cmd == "/track":
        return _handle_track(args, state=state, chat_id=chat_id, state_file=state_file)
    if cmd == "/showliveall":  # alias for /track all
        return _handle_track(["all"], state=state, chat_id=chat_id, state_file=state_file)
    if cmd == "/untrack":
        return _handle_untrack(args, state=state, chat_id=chat_id, state_file=state_file)
    if cmd == "/mute":
        return _handle_mute(args, state=state, chat_id=chat_id, state_file=state_file)
    if cmd == "/watch":
        return _handle_watch(args, state=state, chat_id=chat_id, state_file=state_file)
    if cmd == "/quiet":
        return _handle_quiet(args, state=state, chat_id=chat_id, state_file=state_file)
    if cmd == "/health":
        return _handle_health(state_file)
    if cmd == "/tail":
        return _handle_tail(args, state=state)
    if cmd == "/diff":
        return _handle_diff_command(args[0] if args else "latest")
    return None


# --------------------------------------------------------------------------- #
# Long-poll daemon
# --------------------------------------------------------------------------- #
class BotDaemon:
    def __init__(
        self,
        *,
        client: Optional[TelegramClient] = None,
        users_path: Optional[str] = None,
        state_file: Optional[str] = None,
        poll_timeout: int = 25,
    ):
        self.client = client or TelegramClient()
        self.users_path = users_path
        self.state_file = state_file
        self.poll_timeout = int(poll_timeout)
        self._running = False
        self._offset: Optional[int] = None

    def allowed_user_ids(self) -> set:
        return whitelist_user_ids(load_whitelist(self.users_path))

    def register_commands(self) -> None:
        """Register the command menu with Telegram (best-effort, never raises)."""
        try:
            self.client.set_my_commands(BOT_COMMANDS)
        except Exception as exc:
            logger.debug("telegram setMyCommands failed: %s", exc)

    def _handle_callback_query(self, cbq: dict) -> None:
        """Dispatch one inline-button callback_query (F1); never raises.

        Whitelist-gate by ``from.id`` (a non-whitelisted tap is silently dropped
        — no reply, no answer). Parse ``data`` = ``"verb:short_run_id"`` (>64
        bytes or an unknown verb => safe no-op). Route files/why/diff to the
        existing read-only helpers, reply to the chat, then ALWAYS
        ``answer_callback_query`` so the client's button spinner clears.
        """
        try:
            sender = cbq.get("from") or {}
            user_id = sender.get("id")
            if user_id is None:
                return
            try:
                user_id = int(user_id)
            except Exception:
                return
            if user_id not in self.allowed_user_ids():
                logger.debug("ignoring callback from non-whitelisted user %s", user_id)
                return  # not whitelisted -> no reply AND no answer

            query_id = cbq.get("id")
            message = cbq.get("message") or {}
            chat = message.get("chat") or {}
            chat_id = chat.get("id")
            data = cbq.get("data")

            reply: Optional[str] = None
            # Parse + route only well-formed, in-bounds, known-verb callbacks; any
            # deviation is a safe no-op (still answered below).
            if isinstance(data, str) and len(data.encode("utf-8")) <= 64:
                verb, _, tag = data.partition(":")
                verb = verb.strip().lower()
                if verb in CALLBACK_VERBS and tag:
                    run_id = _resolve_short_tag(tag.strip())
                    if run_id:
                        if verb == "files":
                            reply = _handle_files(run_id)
                        elif verb == "why":
                            reply = _handle_why(run_id)
                        elif verb == "diff":
                            reply = _handle_diff(run_id)

            if reply and chat_id is not None:
                try:
                    self.client.send_message(chat_id, reply)
                except Exception as exc:
                    logger.debug("telegram callback reply failed: %s", exc)
            # ALWAYS answer (whitelisted) so the client spinner clears, even on a no-op.
            try:
                self.client.answer_callback_query(query_id)
            except Exception as exc:
                logger.debug("telegram answerCallbackQuery failed: %s", exc)
        except Exception as exc:
            logger.debug("telegram callback processing failed: %s", exc)

    def _process_update(self, update: dict) -> None:
        """Handle one update; never raises."""
        try:
            cbq = update.get("callback_query")
            if isinstance(cbq, dict):
                self._handle_callback_query(cbq)
                return
            message = update.get("message") or update.get("edited_message")
            if not isinstance(message, dict):
                return
            sender = message.get("from") or {}
            chat = message.get("chat") or {}
            user_id = sender.get("id")
            chat_id = chat.get("id")
            if user_id is None or chat_id is None:
                return
            try:
                user_id = int(user_id)
            except Exception:
                return
            if user_id not in self.allowed_user_ids():
                logger.debug("ignoring non-whitelisted user %s", user_id)
                return
            text = message.get("text") or ""
            reply = handle_command(
                text,
                state=load_state(self.state_file),
                state_file=self.state_file,
                chat_id=chat_id,
            )
            if reply:
                # Attach the [📂 Files] [❓ Why] [📊 Diff] inline keyboard to
                # /status so the most-recent run is one tap away (F1).
                markup = None
                try:
                    cmd = text.strip().split()[0].split("@", 1)[0].lower() if text.strip() else ""
                    if cmd == "/status":
                        dirs = _run_dirs()
                        if dirs:
                            markup = build_run_keyboard(dirs[0].name)
                except Exception:
                    markup = None
                self.client.send_message(chat_id, reply, reply_markup=markup)
        except Exception as exc:
            logger.debug("telegram update processing failed: %s", exc)

    def poll_once(self) -> int:
        """Fetch + process one batch of updates. Returns count processed.

        Each iteration also tails any tracked live runs (cheap when none are
        tracked). Tailing runs alongside command handling without blocking the
        long-poll: a tracking error is swallowed per run and never bubbles out.
        """
        updates = self.client.get_updates(offset=self._offset, timeout=self.poll_timeout)
        processed = 0
        for update in updates:
            if not isinstance(update, dict):
                continue
            uid = update.get("update_id")
            if isinstance(uid, int):
                self._offset = uid + 1
            self._process_update(update)
            processed += 1
        # TEMPORARY cross-process live-run tail (see module docstring).
        try:
            self.tail_tracked()
        except Exception as exc:  # never let tailing crash the poll loop
            logger.debug("telegram live-run tail failed: %s", exc)
        return processed

    def _send_tail(self, chat_id: Any, text: str,
                   reply_markup: Optional[dict] = None) -> None:
        try:
            self.client.send_message(chat_id, text, reply_markup=reply_markup)
        except Exception as exc:  # client already swallows; belt-and-suspenders
            logger.debug("telegram tail send failed: %s", exc)

    def _final_summary_card(self, run_id: str, meta: Optional[dict]) -> str:
        """Render meta.json into the same end-of-run card style as the notifier.

        Reuses :meth:`TelegramNotifier._summary_card` so a tracked run's final
        message matches the dispatch-side summary exactly (success/fail card).
        """
        try:
            stub = TelegramNotifier(
                run_id=run_id, mode=str((meta or {}).get("mode") or ""),
                verbosity=DEFAULT_VERBOSITY, client=self.client, chat_ids=[],
            )
            return stub._summary_card(meta or {})
        except Exception:
            return f"🏁 <b>Run finished</b> · <code>{_e(_short_tag(run_id))}</code>"

    def tail_tracked(self) -> int:
        """Emit NEW events for every tracked run across all chats.

        For each tracked run: read lines appended to events.jsonl since the
        stored byte cursor, render each through ``render_event`` at the chat's
        current verbosity (prefixed with a short run tag so concurrent runs
        don't blur), and advance the cursor (history is never re-sent). When a
        tracked run finishes (meta.json appears) a final summary card is sent
        and the run is AUTO-UNTRACKED. Returns the number of messages sent.

        Fully resilient: a missing/rotated/garbage events.jsonl, a vanished run
        dir, or a send failure is swallowed+logged per run; the loop keeps going.
        """
        state = load_state(self.state_file)
        tracking = _tracking_state(state)
        if not tracking:
            return 0
        verbosity = get_verbosity(state)
        sent = 0
        dirty = False
        for chat_key, runs in list(tracking.items()):
            if not isinstance(runs, dict):
                tracking[chat_key] = {}
                dirty = True
                continue
            for run_id in list(runs.keys()):
                try:
                    run_dir = _runs_dir() / run_id
                    cursor = runs.get(run_id) or 0
                    try:
                        cursor = int(cursor)
                    except Exception:
                        cursor = 0
                    if not run_dir.is_dir():
                        # Run dir vanished — stop tracking it cleanly.
                        runs.pop(run_id, None)
                        dirty = True
                        continue
                    # Drain any events appended since the last cursor first, so
                    # the final lines are delivered before the summary card.
                    events, new_cursor = _read_new_events(_events_path(run_dir), cursor)
                    if new_cursor != cursor:
                        runs[run_id] = new_cursor
                        dirty = True
                    short = _short_tag(run_id)
                    prefix = f"<code>{_e(short)}</code> "
                    for ev in events:
                        try:
                            msg = render_event(
                                ev, verbosity=verbosity,
                                mode=str(ev.get("data", {}).get("mode") or "")
                                if isinstance(ev.get("data"), dict) else "",
                                run_id=run_id,
                            )
                        except Exception:
                            continue
                        if not msg:
                            continue
                        # F2: drop progress chatter for a muted / quiet-hours chat.
                        # Failures (a failed step / failed dispatch) are classified
                        # non-progress and always pass through; the run's final
                        # summary card below is sent unconditionally.
                        if _suppressed(chat_key, _event_class(ev), state):
                            continue
                        # Label every message with a leading short run tag so
                        # concurrent tracked runs are distinguishable in a shared
                        # chat. render_event already appends its own ' · <code>'
                        # suffix on progress lines; the leading tag is the stable
                        # per-message label the spec asks for.
                        self._send_tail(chat_key, f"{prefix}{msg}")
                        sent += 1
                    # Auto-untrack on finish (meta.json present).
                    meta = _read_meta(run_dir)
                    if meta is not None:
                        card = self._final_summary_card(run_id, meta)
                        self._send_tail(
                            chat_key, f"<code>{_e(short)}</code> {card}",
                            reply_markup=build_run_keyboard(run_id),
                        )
                        sent += 1
                        runs.pop(run_id, None)
                        dirty = True
                except Exception as exc:  # per-run isolation
                    logger.debug("telegram tail run %s failed: %s", run_id, exc)
        if dirty:
            save_state(state, self.state_file)
        return sent

    def stop(self, *_args: Any) -> None:
        self._running = False

    def run(self) -> int:
        if not self.client.configured:
            print("TELEGRAM_BOT_KEY not set — bot cannot start.")
            return 2
        self._running = True
        try:
            signal.signal(signal.SIGINT, self.stop)
            signal.signal(signal.SIGTERM, self.stop)
        except Exception:
            pass
        # Register the command menu so the client shows autocomplete (#63).
        self.register_commands()
        logger.info("telegram bot started (long-poll)")
        while self._running:
            try:
                self.poll_once()
            except KeyboardInterrupt:
                break
            except Exception as exc:  # one bad batch never crashes the daemon
                logger.debug("telegram poll loop error: %s", exc)
        logger.info("telegram bot stopped")
        return 0


class EmbeddedCommandPoller:
    """Run the command long-poll in a background thread for the lifetime of a
    dispatch, so /status, /track, etc. work WHILE a build runs (issue #63) without
    a separately-launched daemon. The dispatch already pushes notifications; this
    adds the inbound half.

    Cross-process singleton-guarded by an flock (:func:`poller_lock_path`): if a
    standalone daemon — or a sibling dispatch — already holds it, :meth:`start`
    is a no-op (returns False), so concurrent processes never run rival
    ``getUpdates`` loops (Telegram returns 409 Conflict on concurrent polls).
    Fully best-effort: any failure leaves the dispatch untouched.
    """

    def __init__(self, *, client: Optional[TelegramClient] = None,
                 users_path: Optional[str] = None, state_file: Optional[str] = None,
                 poll_timeout: int = 5):
        self.client = client or TelegramClient()
        self.users_path = users_path
        self.state_file = state_file
        self.poll_timeout = int(poll_timeout)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock_fd: Optional[int] = None

    def _acquire_lock(self) -> bool:
        """Non-blocking flock; True iff THIS process becomes the sole poller."""
        if fcntl is None:
            return False
        path = poller_lock_path(self.state_file)
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        except Exception as exc:
            logger.debug("telegram poller lock open failed: %s", exc)
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)  # someone else already polls — stand down
            return False
        self._lock_fd = fd
        return True

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
            except Exception:
                pass
            self._lock_fd = None

    def start(self) -> bool:
        """Begin polling if possible. Returns True iff a poller thread started."""
        if not self.client.configured:
            return False
        if not whitelist_user_ids(load_whitelist(self.users_path)):
            return False  # no whitelisted users -> nothing to serve
        if not self._acquire_lock():
            return False  # another poller owns the getUpdates stream
        daemon = BotDaemon(client=self.client, users_path=self.users_path,
                           state_file=self.state_file, poll_timeout=self.poll_timeout)
        daemon.register_commands()

        def _loop() -> None:
            try:
                while not self._stop.is_set():
                    try:
                        daemon.poll_once()
                    except Exception as exc:  # one bad batch never kills the loop
                        logger.debug("embedded telegram poll error: %s", exc)
            finally:
                self._release_lock()

        t = threading.Thread(target=_loop, name="telegram-cmd-poller", daemon=True)
        self._thread = t
        t.start()
        return True

    def stop(self, timeout: float = 6.0) -> None:
        """Signal the loop to stop and release the lock. Best-effort, never raises.

        The in-flight long-poll returns within ``poll_timeout`` (default 5s), so
        the thread exits promptly; the lock is released in the loop's finally."""
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        # Belt-and-suspenders: if the thread didn't release in time, release here.
        if self._lock_fd is not None:
            self._release_lock()


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    return BotDaemon().run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

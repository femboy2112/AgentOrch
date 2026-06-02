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
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.telegram import (
    DEFAULT_VERBOSITY,
    VERBOSITY_ORDER,
    TelegramClient,
    TelegramNotifier,
    _e,
    _fmt_duration,
    _run_tag,
    load_whitelist,
    normalize_verbosity,
    render_event,
    whitelist_user_ids,
)

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = "/home/leah/tgbot/data/bot_state.json"

# Resolved lazily from harness.dispatch on first use (keeps the bot's import
# graph light and avoids any import cycle). Tests override this attribute.
RUNS_DIR: Optional[Path] = None


# --------------------------------------------------------------------------- #
# State (persisted OUTSIDE the repo)
# --------------------------------------------------------------------------- #
def state_path(path: Optional[str] = None) -> str:
    return path or os.environ.get("AGY_TELEGRAM_STATE") or DEFAULT_STATE_PATH


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
    return sorted(
        (d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")),
        reverse=True,
    )


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


def is_live_run(run_dir: Path) -> bool:
    """True iff the run has started streaming events but has no terminal meta.

    A run is "live" when ``events.jsonl`` exists (the dispatch is streaming) and
    ``meta.json`` (written only when the run FINISHES) is absent. This is how the
    bot catches runs it never started — discovery is pure filesystem scan.
    """
    try:
        if not run_dir.is_dir():
            return False
        return _events_path(run_dir).exists() and not (run_dir / "meta.json").exists()
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
    latest = dirs[0]
    meta = _read_meta(latest)
    if meta is None:
        return f"⏳ <b>Run in progress</b> · <code>{_e(latest.name)}</code>"
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
            out.append(f"• <code>{_e(d.name)}</code> · in progress")
            continue
        icon = "✅" if meta.get("success") else "❌"
        mode = _e(meta.get("mode") or "")
        dur = _fmt_duration(meta.get("duration_s"))
        out.append(f"{icon} <code>{_e(d.name)}</code> · {mode} · {_e(dur)}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Command handling (pure-ish: returns a reply string or None)
# --------------------------------------------------------------------------- #
HELP_TEXT = (
    "🤖 <b>AgentOrch build bot</b>\n"
    "/status — most recent run\n"
    "/runs [N] — recent runs (default 5)\n"
    "/verbosity [level] — show or set default ("
    + " / ".join(VERBOSITY_ORDER)
    + ")\n"
    "/track [latest|&lt;run_id&gt;|all] — follow a live run (default: latest)\n"
    "/untrack [&lt;run_id&gt;|all] — stop following (default: all)\n"
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

    def _process_update(self, update: dict) -> None:
        """Handle one update; never raises."""
        try:
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
            reply = handle_command(
                message.get("text") or "",
                state=load_state(self.state_file),
                state_file=self.state_file,
                chat_id=chat_id,
            )
            if reply:
                self.client.send_message(chat_id, reply)
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

    def _send_tail(self, chat_id: Any, text: str) -> None:
        try:
            self.client.send_message(chat_id, text)
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
                        self._send_tail(chat_key, f"<code>{_e(short)}</code> {card}")
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


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    return BotDaemon().run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

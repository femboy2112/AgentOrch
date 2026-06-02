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

The loop is resilient: one bad update or a failed send never crashes the
daemon, and SIGINT shuts down cleanly. Everything is stdlib only and the bot
token is read from ``TELEGRAM_BOT_KEY`` (never hardcoded). The persisted state
file lives OUTSIDE the repo (default ``/home/leah/tgbot/data/bot_state.json``;
override via ``AGY_TELEGRAM_STATE``).
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
    _e,
    _fmt_duration,
    load_whitelist,
    normalize_verbosity,
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
    "/help — this message"
)


def handle_command(text: str, *, state: dict, state_file: Optional[str] = None) -> Optional[str]:
    """Map a whitelisted user's message to a reply. Returns None to ignore."""
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
            )
            if reply:
                self.client.send_message(chat_id, reply)
        except Exception as exc:
            logger.debug("telegram update processing failed: %s", exc)

    def poll_once(self) -> int:
        """Fetch + process one batch of updates. Returns count processed."""
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
        return processed

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

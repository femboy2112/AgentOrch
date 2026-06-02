"""Telegram build-progress notifier + thin client (stdlib only).

This module turns a dispatch's EventBus stream into concise, professional
Telegram messages and provides a small outbound client. Every network call is
**best-effort and exception-isolated** — modelled on
:class:`harness.run_monitor.Notifier`: a Telegram outage (or a missing/garbage
whitelist, or no token) can never block, slow, or fail a dispatch.

Design notes
------------
* stdlib only — HTTP via ``urllib.request`` (no third-party deps).
* The bot token is read from ``TELEGRAM_BOT_KEY`` in the environment; it is
  NEVER hardcoded and never persisted by this module.
* The recipient whitelist lives OUTSIDE the repo (default
  ``/home/leah/tgbot/data/users.json``; override via ``AGY_TELEGRAM_USERS``).
* :func:`render_event` is a pure function: ``(event, verbosity, mode, run_id)``
  -> an HTML message string for events that should be sent at that verbosity,
  else ``None``. This makes the gating policy unit-testable without any I/O.
"""
from __future__ import annotations

import html
import json
import logging
import os
import queue
import threading
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org/bot{token}/{method}"

DEFAULT_USERS_PATH = "/home/leah/tgbot/data/users.json"

# Verbosity ladder (low -> high). Each level is a strict superset of the prior.
VERBOSITY_ORDER = ["quiet", "normal", "verbose", "debug"]
DEFAULT_VERBOSITY = "normal"

# Outcomes that signal a failure/stall worth surfacing at ``normal`` and above.
# These are the exact strings the workflows emit on the orchestration stream:
#   adversarial._iteration_outcome / master step-completed emit
#     -> verified | approved | verifier_timeout | verifier_resource_exceeded
#        | verifier_infra_failed | stalled | continue
# Infra failures (#50) are a slow/OOM verifier, NOT a code defect — they get a
# distinct glyph so an OOM never reads as a normal revision.
_INFRA_OUTCOMES = frozenset(
    {"verifier_timeout", "verifier_resource_exceeded", "verifier_infra_failed"}
)
_FAILURE_OUTCOMES = frozenset({"stalled"}) | _INFRA_OUTCOMES

# Pretty labels for the failure/stall banner.
_OUTCOME_LABEL = {
    "verifier_timeout": "⏱ verifier timed out",
    "verifier_resource_exceeded": "💥 verifier OOM",
    "verifier_infra_failed": "💥 verifier infra failure",
    "stalled": "⚠️ stalled",
}


def _verbosity_index(verbosity: Optional[str]) -> int:
    try:
        return VERBOSITY_ORDER.index(str(verbosity))
    except ValueError:
        return VERBOSITY_ORDER.index(DEFAULT_VERBOSITY)


def normalize_verbosity(verbosity: Optional[str]) -> str:
    v = str(verbosity or "").strip().lower()
    return v if v in VERBOSITY_ORDER else DEFAULT_VERBOSITY


def _e(value: Any) -> str:
    """HTML-escape any interpolated text (titles, run ids, reasons)."""
    return html.escape(str(value if value is not None else ""), quote=False)


def _run_tag(rid: Any) -> str:
    """A compact, already-escaped ' · <code>…</code>' run suffix (or '').

    Lets progress lines (step/adversarial/reroute/plan) be attributed to a run
    when several dispatches share one chat. ``rid`` is expected pre-escaped.
    """
    s = str(rid or "").strip()
    if not s:
        return ""
    short = s[-6:] if len(s) > 6 else s
    return f" · <code>{short}</code>"


def _fmt_tokens(total: Any) -> str:
    """Thousands-separated token count when integral, else escaped as-is."""
    if isinstance(total, bool):  # bool is an int subclass — treat as text
        return _e(total)
    if isinstance(total, int):
        return f"{total:,}"
    try:
        return f"{int(total):,}"
    except Exception:
        return _e(total)


# --------------------------------------------------------------------------- #
# Thin client
# --------------------------------------------------------------------------- #
class TelegramClient:
    """Minimal, best-effort Telegram Bot API client (stdlib only).

    ``send_message`` swallows and logs on any failure and NEVER raises into the
    caller (matching :class:`harness.run_monitor.Notifier`). ``get_updates`` is
    used by the long-poll daemon and returns ``[]`` on any error.
    """

    def __init__(self, token: Optional[str] = None, *, timeout: float = 5.0):
        self.token = token if token is not None else (os.environ.get("TELEGRAM_BOT_KEY") or None)
        self.timeout = float(timeout)

    @property
    def configured(self) -> bool:
        return bool(self.token)

    def _url(self, method: str) -> str:
        return API_BASE.format(token=self.token, method=method)

    def _post(self, method: str, params: Dict[str, Any], *, timeout: Optional[float] = None) -> Optional[dict]:
        """POST form-encoded params; return parsed JSON dict or None on failure."""
        if not self.configured:
            return None
        data = urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None}
        ).encode("utf-8")
        req = urllib.request.Request(
            self._url(method),
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:  # best-effort — never raise into the caller
            logger.debug("telegram %s failed: %s", method, exc)
            return None

    def send_message(
        self,
        chat_id: Any,
        text: str,
        *,
        parse_mode: str = "HTML",
        disable_web_page_preview: bool = True,
    ) -> Optional[dict]:
        """Send one message. Best-effort; returns the API result or None."""
        if not self.configured or chat_id is None:
            return None
        return self._post(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": "true" if disable_web_page_preview else "false",
            },
        )

    def get_updates(self, offset: Optional[int] = None, timeout: int = 25) -> List[dict]:
        """Long-poll for updates. Returns the ``result`` list, or [] on failure."""
        if not self.configured:
            return []
        # Network read must outlast the server-side long-poll window.
        body = self._post(
            "getUpdates",
            {"offset": offset, "timeout": int(timeout)},
            timeout=self.timeout + float(timeout),
        )
        if isinstance(body, dict) and isinstance(body.get("result"), list):
            return body["result"]
        return []


# --------------------------------------------------------------------------- #
# Whitelist
# --------------------------------------------------------------------------- #
def whitelist_path(path: Optional[str] = None) -> str:
    return path or os.environ.get("AGY_TELEGRAM_USERS") or DEFAULT_USERS_PATH


def load_whitelist(path: Optional[str] = None) -> List[dict]:
    """Load the recipient whitelist; tolerate a missing/garbage file -> [].

    Each returned entry exposes a chat id (``last_chat_id`` preferred, else
    ``id``), a user id, and a username (when present).
    """
    p = whitelist_path(path)
    try:
        with open(p, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as exc:
        logger.debug("telegram whitelist load failed (%s): %s", p, exc)
        return []
    if not isinstance(raw, list):
        return []
    out: List[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        uid = entry.get("id")
        chat_id = entry.get("last_chat_id")
        if chat_id is None:
            chat_id = uid
        if chat_id is None and uid is None:
            continue
        out.append(
            {
                "id": uid,
                "chat_id": chat_id,
                "username": entry.get("username"),
            }
        )
    return out


def whitelist_chat_ids(entries: List[dict]) -> List[Any]:
    seen: List[Any] = []
    for e in entries or []:
        cid = e.get("chat_id")
        if cid is not None and cid not in seen:
            seen.append(cid)
    return seen


def whitelist_user_ids(entries: List[dict]) -> set:
    out = set()
    for e in entries or []:
        uid = e.get("id")
        if uid is not None:
            try:
                out.add(int(uid))
            except Exception:
                continue
    return out


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _fmt_duration(seconds: Any) -> str:
    try:
        s = float(seconds)
    except Exception:
        return "?"
    if s < 60:
        return f"{s:.0f}s"
    m, sec = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _orchestration(event: dict) -> Optional[dict]:
    data = event.get("data")
    if not isinstance(data, dict):
        return None
    orch = data.get("orchestration")
    return orch if isinstance(orch, dict) else None


# --------------------------------------------------------------------------- #
# Pure gating + rendering
# --------------------------------------------------------------------------- #
def render_event(
    event: dict,
    *,
    verbosity: str = DEFAULT_VERBOSITY,
    mode: str = "",
    run_id: str = "",
) -> Optional[str]:
    """Return an HTML message for ``event`` at ``verbosity``, else None.

    Gating (each level is a superset of the prior):
      quiet   : dispatch start + dispatch finish only.
      normal  : + step started/completed, + failures/stalls.
      verbose : + adversarial iteration_completed, fallback reroutes,
                reconcile/plan transitions.
      debug   : + heartbeats + per-call token usage.
    """
    if not isinstance(event, dict):
        return None
    level = _verbosity_index(verbosity)
    kind = event.get("kind")
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    rid = _e(run_id or event.get("run_id") or "")
    safe_mode = _e(mode)

    # ---- quiet (level 0): dispatch lifecycle only ------------------------- #
    if kind == "lifecycle":
        ev = data.get("event")
        if ev == "dispatch_started":
            return f"🟢 <b>Build started</b> · <code>{safe_mode}</code> · run <code>{rid}</code>"
        if ev == "dispatch_finished":
            # The polished end-of-run card is owned by TelegramNotifier.finished()
            # (success/fail, duration, files, tokens). A bare marker here would be
            # a redundant second message for the same event, so we stay silent and
            # let the summary card be the single end-of-run notification.
            return None

    # ---- normal (level 1): steps + failures/stalls ------------------------ #
    orch = _orchestration(event)
    if orch is not None:
        phase = orch.get("phase")
        action = orch.get("action")
        outcome = orch.get("outcome")

        # Failures/stalls — normal and above, regardless of phase.
        if level >= 1 and str(outcome) in _FAILURE_OUTCOMES:
            title = _e(orch.get("step_title") or phase or "step")
            label = _OUTCOME_LABEL.get(str(outcome), str(outcome).replace("_", " "))
            return f"<b>{_e(label)}</b> · <i>{title}</i>{_run_tag(rid)}"

        if phase == "step":
            idx = orch.get("step_index")
            total = orch.get("step_total")
            title = _e(orch.get("step_title") or "step")
            pos = f"{idx}/{total}" if total else (str(idx) if idx is not None else "?")
            if level >= 1 and action == "started":
                return f"▸ <b>Step {pos}</b> · <i>{title}</i>{_run_tag(rid)}"
            if level >= 1 and action == "completed":
                # The completed emit carries the step's real outcome (master
                # attaches verified/approved/infra/stalled). A green check only
                # when we have a positive signal; neutral when none is present
                # (avoids a misleading ✅ for a step with no verdict).
                if str(outcome) in _FAILURE_OUTCOMES:
                    ok = "❌"
                elif outcome in ("verified", "approved"):
                    ok = "✅"
                else:
                    ok = "▪"
                return f"{ok} <b>Step {pos} done</b> · <i>{title}</i>{_run_tag(rid)}"

        # ---- verbose (level 2): iteration / fallback / plan / reconcile --- #
        if level >= 2:
            if phase == "adversarial" and action in ("iteration_completed", "completed"):
                it = orch.get("iteration")
                itn = orch.get("iteration_total")
                pos = f"{it}/{itn}" if itn else (str(it) if it is not None else "")
                verdict = orch.get("verified")
                tag = "approved" if (orch.get("approved") or verdict) else "revised"
                extra = f" · iter {pos}" if pos else ""
                return f"🔁 <b>Adversarial {_e(tag)}</b>{_e(extra)}{_run_tag(rid)}"
            if phase == "fallback":
                return f"↪️ <b>Reroute</b> · <i>{_e(orch.get('action') or 'fallback')}</i>{_run_tag(rid)}"
            if phase in ("plan", "reconcile"):
                act = _e(action or phase)
                return f"🗂 <b>{_e(str(phase).title())}</b> · <i>{act}</i>{_run_tag(rid)}"

    # ---- debug (level 3): heartbeats + per-call usage --------------------- #
    if level >= 3:
        if kind == "heartbeat":
            step = data.get("step")
            elapsed = data.get("elapsed_s")
            stepf = f" · step {_e(step)}" if step else ""
            return f"💓 <b>Heartbeat</b>{stepf} · {_e(_fmt_duration(elapsed))}"
        if kind == "usage" and data.get("usage_kind") == "call":
            worker = _e(data.get("worker") or event.get("worker") or "worker")
            total = data.get("total_tokens")
            if total is None:
                inp = data.get("input_tokens") or 0
                out = data.get("output_tokens") or 0
                total = (inp or 0) + (out or 0)
            return f"🔢 <b>{worker}</b> · {_fmt_tokens(total)} tok"

    return None


# --------------------------------------------------------------------------- #
# EventBus sink
# --------------------------------------------------------------------------- #
class TelegramNotifier:
    """An EventBus sink that forwards rendered build-progress to Telegram.

    Called as ``sink(event_dict)`` for every event of the run. Fully
    exception-isolated: a Telegram failure (or a malformed event) can never
    propagate into the dispatch coroutine. Raw stderr chatter is never
    forwarded (``render_event`` already declines it).
    """

    def __init__(
        self,
        *,
        run_id: str,
        mode: str,
        verbosity: str,
        client: TelegramClient,
        chat_ids: List[Any],
        max_queue: int = 1000,
    ):
        self.run_id = run_id
        self.mode = mode
        self.verbosity = normalize_verbosity(verbosity)
        self.client = client
        self.chat_ids = list(chat_ids or [])
        self.sent = 0
        # Delivery runs on a single background daemon thread so a slow/hung
        # Telegram endpoint never blocks the event-loop thread (mirrors
        # harness.run_monitor.Notifier). Rendered messages are pushed onto a
        # bounded queue and sent in order; a full queue drops oldest rather than
        # block (best-effort, never back-pressures _drain).
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=int(max_queue))
        self._worker: Optional[threading.Thread] = None
        self._worker_lock = threading.Lock()

    @property
    def active(self) -> bool:
        return bool(self.client and self.client.configured and self.chat_ids)

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            t = threading.Thread(
                target=self._drain_queue,
                name=f"telegram-notify-{self.run_id}",
                daemon=True,
            )
            self._worker = t
            t.start()

    def _drain_queue(self) -> None:
        while True:
            text = self._queue.get()
            try:
                if text is None:  # poison pill -> stop the worker
                    return
                self._deliver(text)
            except Exception as exc:  # never let the worker thread die noisily
                logger.debug("telegram worker error: %s", exc)
            finally:
                self._queue.task_done()

    def _deliver(self, text: str) -> None:
        """Synchronous send to every chat. Runs ONLY on the worker thread."""
        for chat_id in self.chat_ids:
            try:
                self.client.send_message(chat_id, text)
                self.sent += 1
            except Exception as exc:  # belt-and-suspenders; client already swallows
                logger.debug("telegram broadcast failed: %s", exc)

    def _broadcast(self, text: str) -> None:
        """Enqueue a rendered message for off-loop delivery. Never blocks."""
        if not text:
            return
        self._ensure_worker()
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            # Drop the oldest queued message to make room — progress chatter is
            # disposable and we must never block the event loop.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Exception:
                pass
            try:
                self._queue.put_nowait(text)
            except Exception:
                pass

    def flush(self, timeout: Optional[float] = None) -> None:
        """Block until queued messages are delivered. Best-effort; never raises.

        Used at end-of-run (so the summary card actually goes out before the
        process exits) and by tests for determinism.
        """
        try:
            if timeout is None:
                self._queue.join()
            else:
                # queue.join has no timeout; poll unfinished_tasks instead.
                import time as _time

                deadline = _time.monotonic() + float(timeout)
                while self._queue.unfinished_tasks and _time.monotonic() < deadline:
                    _time.sleep(0.01)
        except Exception:
            pass

    def __call__(self, event: dict) -> None:
        """EventBus sink entrypoint — never raises."""
        if not self.active:
            return
        try:
            text = render_event(
                event,
                verbosity=self.verbosity,
                mode=self.mode,
                run_id=self.run_id,
            )
        except Exception as exc:
            logger.debug("telegram render failed: %s", exc)
            return
        if text:
            self._broadcast(text)

    def finished(self, meta: Optional[dict]) -> None:
        """Send the polished final-summary card, then drain. Never raises."""
        if not self.active:
            return
        try:
            self._broadcast(self._summary_card(meta or {}))
        except Exception as exc:
            logger.debug("telegram summary failed: %s", exc)
        # Drain so the card actually leaves the process before dispatch returns.
        # Bounded so a hung endpoint can't stall teardown.
        self.flush(timeout=6.0)

    def _summary_card(self, meta: dict) -> str:
        success = bool(meta.get("success"))
        mode = _e(meta.get("mode") or self.mode)
        dur = _fmt_duration(meta.get("duration_s"))
        rid = _e(meta.get("run_id") or self.run_id)
        changed = meta.get("changed_files")
        n_files = len(changed) if isinstance(changed, list) else 0
        quality = meta.get("quality") if isinstance(meta.get("quality"), dict) else {}
        raw_conf = quality.get("confidence")
        confidence = str(raw_conf).strip() if raw_conf is not None else ""
        verified = confidence.lower() == "verified"
        # Omit the confidence chip entirely when it's absent / "n/a" (a literal
        # "n/a" next to a green check reads oddly).
        conf_chip = (
            f" · {_e(confidence)}"
            if confidence and confidence.lower() not in ("n/a", "na", "none")
            else ""
        )

        tokens = meta.get("tokens") if isinstance(meta.get("tokens"), dict) else {}
        grand = tokens.get("grand_total") if isinstance(tokens.get("grand_total"), dict) else {}
        grand_total = grand.get("total_tokens")
        token_str = f" · {grand_total:,} tok" if isinstance(grand_total, int) else ""

        if success:
            # Distinct headline for a verified run vs a completed-but-unverified
            # one (the only prior difference was a bare confidence word).
            headline = "✅ <b>Build verified</b>" if verified else "☑️ <b>Build complete</b>"
            return (
                f"{headline}{_e(conf_chip)} · {_e(dur)} · "
                f"{n_files} file{'s' if n_files != 1 else ''}{_e(token_str)}\n"
                f"<code>{mode}</code> · run <code>{rid}</code>"
            )
        reason = _e(meta.get("error") or meta.get("run_outcome") or "failed")
        return (
            f"❌ <b>Build failed</b> · {reason} · {_e(dur)}\n"
            f"<code>{mode}</code> · run <code>{rid}</code>"
        )


# --------------------------------------------------------------------------- #
# Enable logic (shared by dispatch + bot)
# --------------------------------------------------------------------------- #
def resolve_verbosity(explicit: Optional[str] = None) -> str:
    """Verbosity from explicit flag, else AGY_TELEGRAM_VERBOSITY, else default."""
    if explicit:
        return normalize_verbosity(explicit)
    return normalize_verbosity(os.environ.get("AGY_TELEGRAM_VERBOSITY"))

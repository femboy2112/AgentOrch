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
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org/bot{token}/{method}"

DEFAULT_USERS_PATH = "/home/leah/tgbot/data/users.json"
# Persisted bot state (the /verbosity default, /track cursors) lives OUTSIDE the
# repo. This module owns the path so both the dispatch-side notifier and the bot
# daemon read the SAME file (telegram_bot re-imports these).
DEFAULT_STATE_PATH = "/home/leah/tgbot/data/bot_state.json"

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

# Render caps to keep messages well under Telegram's 4096-char limit even on
# long (30+ step) runs. The live status card is editMessageText'd in place on
# every event, so an over-limit render 400s and silently freezes the pinned
# card — hence the tight tail cap on the live "Steps so far" list. The
# end-of-run recap is sent once, so it can carry a larger cap.
_STATUS_STEP_CAP = 12
_SUMMARY_STEP_CAP = 40

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
        reply_markup: Optional[dict] = None,
    ) -> Optional[dict]:
        """Send one message. Best-effort; returns the API result or None.

        ``reply_markup`` (e.g. an ``{"inline_keyboard": [...]}`` dict) is
        JSON-serialized and passed through to the API so the message can carry
        inline-keyboard buttons (F1). Omitted when None.
        """
        if not self.configured or chat_id is None:
            return None
        params: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": "true" if disable_web_page_preview else "false",
        }
        if reply_markup is not None:
            params["reply_markup"] = json.dumps(reply_markup)
        return self._post("sendMessage", params)

    def answer_callback_query(
        self, callback_query_id: Any, text: str = ""
    ) -> Optional[dict]:
        """Answer a callback_query (clears the client's button spinner).

        Best-effort; returns the API result or None. ``text``, when given,
        surfaces as a brief toast on the client. Telegram requires every
        callback_query to be answered within a short window — the bot's callback
        dispatch ALWAYS calls this, even for an ignored/no-op verb (F1).
        """
        if not self.configured or callback_query_id is None:
            return None
        params: Dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            params["text"] = text
        return self._post("answerCallbackQuery", params)

    def edit_message_text(
        self,
        chat_id: Any,
        message_id: int,
        text: str,
        *,
        reply_markup: Optional[dict] = None,
    ) -> Optional[dict]:
        """Edit an existing message. Best-effort; returns the API result or None."""
        if not self.configured or chat_id is None:
            return None
        params: Dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }
        if reply_markup is not None:
            params["reply_markup"] = json.dumps(reply_markup)
        return self._post("editMessageText", params)

    def pin_chat_message(self, chat_id: Any, message_id: int) -> Optional[dict]:
        """Pin a message. Best-effort; returns the API result or None."""
        if not self.configured or chat_id is None:
            return None
        return self._post("pinChatMessage", {"chat_id": chat_id, "message_id": message_id})

    def unpin_chat_message(self, chat_id: Any, message_id: int) -> Optional[dict]:
        """Unpin a message. Best-effort; returns the API result or None."""
        if not self.configured or chat_id is None:
            return None
        return self._post("unpinChatMessage", {"chat_id": chat_id, "message_id": message_id})

    def delete_message(self, chat_id: Any, message_id: Any) -> Optional[dict]:
        """Delete one message. Best-effort; returns the API result or None.

        Telegram only allows this for messages < 48h old; in a PRIVATE chat a bot
        may delete both its own outgoing and the user's incoming messages. An
        un-deletable/already-gone message returns a non-ok result (the caller
        treats that as "couldn't remove" rather than an error).
        """
        if not self.configured or chat_id is None or message_id is None:
            return None
        return self._post(
            "deleteMessage", {"chat_id": chat_id, "message_id": message_id})

    def delete_messages(self, chat_id: Any, message_ids: List[Any]) -> Optional[dict]:
        """Bulk-delete up to 100 messages at once (Bot API 7.0 ``deleteMessages``).

        Best-effort; returns the API result or None. ``message_ids`` is sent as a
        JSON array. The whole batch fails if ANY id is un-deletable, so the caller
        falls back to per-message :meth:`delete_message` on a non-ok result.
        """
        if not self.configured or chat_id is None:
            return None
        ids = [m for m in (message_ids or []) if m is not None]
        if not ids:
            return None
        return self._post(
            "deleteMessages",
            {"chat_id": chat_id, "message_ids": json.dumps(ids)},
        )

    def set_my_commands(self, commands: List[Dict[str, str]]) -> Optional[dict]:
        """Register the bot's command list with Telegram (setMyCommands).

        Best-effort; returns the API result or None. ``commands`` is a list of
        ``{"command": "status", "description": "..."}`` dicts. Without this call
        Telegram shows no command menu/autocomplete — the bot's commands look
        "unregistered" even though the daemon would answer them (issue #63).
        """
        if not self.configured or not commands:
            return None
        return self._post("setMyCommands", {"commands": json.dumps(commands)})

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


# Phase letters for a compact build-context suffix ("Phase B" etc.). Maps the
# orchestration ``phase`` string to a short human label so a spin-up/round line
# can say which kind of work it belongs to without a long word.
_PHASE_LABEL = {
    # "step" is intentionally absent: the step_pos chip ("Step 2/5") already
    # conveys it, so a "Step phase" label would read redundantly.
    "adversarial": "Adversarial",
    "fallback": "Fallback",
    "plan": "Plan",
    "reconcile": "Reconcile",
    "tot": "ToT",
}


# What a worker spin-up is DOING, from the agent's role (stamped on agent_started
# by the orchestrator). Lets the operator tell apart the several distinct codex
# calls a master step makes — a per-step summarizer + the next step's generator
# are both codex and otherwise render as two identical, confusing "spun up" lines.
_ROLE_LABEL = {
    "draft": "drafting",
    "critic": "reviewing",
    "summary": "summarizing",
    "compact": "compacting context",
    "plan": "planning",
    "tot": "exploring",
    "tot-judge": "scoring branches",
    "reconcile": "reconciling",
}


def _chip(value: Any) -> str:
    """Normalize a model/effort chip value; '' when absent/'n/a'."""
    s = str(value or "").strip()
    if not s or s.lower() in ("n/a", "na", "none"):
        return ""
    return s


def _build_context_suffix(context: Optional[dict]) -> str:
    """Format a build-context dict into a compact, escaped HTML suffix.

    ``context`` carries ``{step_pos, step_title, phase}`` (any subset). Returns
    a leading-separator suffix like ``"  · <i>Step 2/5</i>"`` or
    ``"  · <i>Step 2/5 · Adversarial</i>"`` — or ``""`` when there's nothing to
    say. All dynamic text is escaped via :func:`_e` (parse_mode is HTML).
    """
    if not isinstance(context, dict):
        return ""
    bits: List[str] = []
    pos = context.get("step_pos")
    if pos:
        bits.append(f"Step {_e(pos)}")
    phase = context.get("phase")
    plabel = _PHASE_LABEL.get(str(phase)) if phase else None
    if plabel:
        bits.append(_e(plabel))
    if not bits:
        return ""
    return f" · <i>{' · '.join(bits)}</i>"



class BuildState:
    def __init__(self, run_id: str, mode: str):
        self.run_id = run_id
        self.mode = mode
        self.goal = ""
        self.run_start_ts = None
        self.finished = False
        
        self.step_total = None
        self.is_graph = False
        
        self.cur_step_index = None
        self.cur_step_title = ""
        self.cur_phase = ""
        self.cur_step_start_ts = None
        self.cur_iter = None
        self.cur_iter_total = None
        
        self.active_role = ""
        
        self.completed_steps = set()
        self.step_durations = []
        self._step_started_at = {}
        # Per-step recap rows (ordered by completion) for the end-of-run card:
        # {"index", "title", "duration", "outcome", "rounds"}.
        self.step_records = []
        # adversarial iterations observed for the in-flight step (reset on start).
        self._cur_step_rounds = 0
        # how many rounds the most-recently-completed step ran (for the step-done
        # line, which render_event reads via the context channel).
        self.last_step_rounds = 0
        self.last_step_duration = None
        
        self.n_steps_done = 0
        self.n_verified = 0
        self.n_approved = 0
        
        self.last_event_ts = None
        self.since_progress_s = 0.0
        self.pinned_message_id = None
        self.last_render_ts = 0.0
        self.last_render_signature = None
        self.last_event_line = ""

    @property
    def elapsed(self):
        if self.run_start_ts is None:
            return 0.0
        return max(0.0, (self.last_event_ts or self.run_start_ts) - self.run_start_ts)

    def update(self, event: dict):
        ts = event.get("ts", 0.0)
        if ts:
            self.last_event_ts = ts

        data = event.get("data", {})
        if not isinstance(data, dict):
            data = {}
        kind = event.get("kind")

        if kind == "heartbeat":
            self.since_progress_s = data.get("since_progress_s", 0)
            return

        if kind == "lifecycle":
            ev = data.get("event")
            if ev == "dispatch_started":
                self.run_start_ts = ts
            elif ev == "agent_started":
                detail = data.get("detail", {})
                if isinstance(detail, dict) and "role" in detail:
                    # Do not move step geometry on compact spin-ups
                    self.active_role = detail.get("role", "")
            # Removed the return here to allow orchestration transitions to be processed

        orch = data.get("orchestration")
        if orch and isinstance(orch, dict):
            phase = orch.get("phase")
            action = orch.get("action")
            
            if phase:
                self.cur_phase = phase

            if phase == "plan" and action == "completed":
                self.step_total = orch.get("step_total")
                if orch.get("is_graph"):
                    self.is_graph = True

            elif phase == "step":
                idx = orch.get("step_index")
                if action == "started":
                    if self.step_total is None:
                        self.step_total = orch.get("step_total")
                    if self.cur_step_index and idx and idx != self.cur_step_index + 1:
                        self.is_graph = True
                    if idx and idx > 1 and not self.completed_steps:
                        for i in range(1, idx):
                            self.completed_steps.add(i)
                        self.n_steps_done = len(self.completed_steps)
                    
                    self.cur_step_index = idx
                    self.cur_step_title = orch.get("step_title", "")
                    self.cur_step_start_ts = ts
                    if idx:
                        self._step_started_at[idx] = ts
                    self.cur_iter = None
                    self.cur_iter_total = None
                    self._cur_step_rounds = 0

                elif action == "completed":
                    dur = None
                    if idx:
                        self.completed_steps.add(idx)
                        self.n_steps_done = len(self.completed_steps)
                        if idx in self._step_started_at:
                            dur = max(0.0, float(ts) - float(self._step_started_at[idx]))
                            self.step_durations.append(dur)
                    outcome = orch.get("outcome")
                    if outcome == "verified":
                        self.n_verified += 1
                    elif outcome == "approved":
                        self.n_approved += 1
                    rounds = self._cur_step_rounds
                    self.last_step_rounds = rounds
                    self.last_step_duration = dur
                    self.step_records.append({
                        "index": idx,
                        "title": orch.get("step_title", "") or self.cur_step_title,
                        "duration": dur,
                        "outcome": outcome,
                        "rounds": rounds,
                    })

            elif phase == "adversarial":
                if action == "iteration_started":
                    self.cur_iter = orch.get("iteration")
                    self.cur_iter_total = orch.get("iteration_total")
                    self._cur_step_rounds += 1

def _card_signature(state: BuildState) -> tuple:
    el = round(state.elapsed / 5.0) * 5
    return (
        state.cur_step_index,
        state.n_steps_done,
        state.cur_phase,
        state.cur_iter,
        state.active_role,
        el,
        state.last_event_line
    )

def render_status_card(state: BuildState) -> str:
    rid_short = state.run_id[-6:] if len(state.run_id) > 6 else state.run_id
    
    if state.step_total and state.step_total > 0:
        pct = min(10, round(state.n_steps_done / state.step_total * 10))
        bar = "█" * pct + "░" * (10 - pct)
        step_ref = f"Step {state.cur_step_index or '?'}/{state.step_total}"
    else:
        idx = (state.cur_step_index or 0) % 10
        cells = ["░"] * 10
        cells[idx] = "█"
        bar = "".join(cells)
        if state.cur_step_index:
            step_ref = f"Step {state.cur_step_index} (of ?)"
        else:
            step_ref = "Planning…"

    title = state.cur_step_title or "Initializing"
    
    phase_glyph = "⚙️"
    phase_label = str(state.cur_phase).title() if state.cur_phase else "Orchestrating"
    iter_str = ""
    if state.cur_iter:
        iter_str = f" · <b>iter {state.cur_iter}/{state.cur_iter_total or '?'}</b>"
    
    if state.last_event_ts:
        spinner_glyphs = ["◐", "◓", "◑", "◒"]
        spinner = spinner_glyphs[int(state.last_event_ts) % 4]
        if not iter_str:
            phase_glyph = spinner

    elapsed_str = _fmt_duration(state.elapsed)

    if state.n_steps_done >= 1 and state.step_total and state.step_total > state.n_steps_done:
        durs = state.step_durations
        rem = state.step_total - state.n_steps_done
        if state.is_graph:
            rate = state.elapsed / max(1, state.n_steps_done)
            eta_sec = rate * rem
            in_flight = 0
            if state.cur_step_start_ts and state.last_event_ts:
                in_flight = min(state.last_event_ts - state.cur_step_start_ts, rate)
            eta_sec = max(0.0, float(eta_sec - in_flight))
            eta_str = f"~{_fmt_duration(eta_sec)} left (parallel)"
        else:
            if len(durs) >= 3:
                s_durs = sorted(durs)
                d_hat = s_durs[len(durs)//2]
            elif len(durs) > 0:
                d_hat = sum(durs) / len(durs)
            else:
                d_hat = 0
            in_flight = 0
            if state.cur_step_start_ts and state.last_event_ts:
                in_flight = min(state.last_event_ts - state.cur_step_start_ts, d_hat)
            eta_sec = max(0.0, float((d_hat * rem) - in_flight))
            if len(durs) >= 3:
                low = eta_sec * 0.8
                high = eta_sec * 1.2
            else:
                low = eta_sec * 0.5
                high = eta_sec * 2.0
            eta_str = f"~{_fmt_duration(low)}–{_fmt_duration(high)} (≈{_fmt_duration(eta_sec)}) left"
    else:
        if state.step_total and state.n_steps_done >= state.step_total:
            eta_str = "finishing…"
        else:
            eta_str = "estimating…"

    stall_flag = ""
    if state.since_progress_s > 60:
        stall_flag = f"  ⚠️ <b>no progress {_fmt_duration(state.since_progress_s)}</b>"

    worker_state = _e(state.active_role or "running")

    goal = str(getattr(state, "goal", "") or "").strip()
    goal_line = f"<i>{_e(goal[:80])}</i>\n" if goal else ""

    # ---- always-visible compact card --------------------------------------- #
    # Status · bar+step · phase/worker/title · elapsed+ETA stay on the surface;
    # everything else is tucked into the expandable so the live card never grows.
    header = (
        f"⚡ <b>Building</b> · <code>{rid_short}</code>{stall_flag}\n"
        f"{goal_line}"
        f"<code>{bar}</code>  <b>{step_ref}</b>\n"
        f"{phase_glyph} <b>{phase_label}</b> · {worker_state}{iter_str} · <i>{_e(title)}</i>\n"
        f"⏱ <b>{elapsed_str}</b> elapsed · {eta_str}"
    )

    # ---- expandable secondary detail (recent activity + per-step progress) -- #
    detail: List[str] = []
    records = list(getattr(state, "step_records", []) or [])
    if records:
        # Cap to the most-recent N so the in-place card can never grow past
        # Telegram's 4096-char limit (which would 400 the editMessageText and
        # freeze the pinned live card). Recent steps matter most here, so we
        # keep the tail and note "+K earlier" at the top.
        shown = records[-_STATUS_STEP_CAP:]
        earlier = len(records) - len(shown)
        detail.append("<b>Steps so far</b>")
        if earlier > 0:
            detail.append(f"  … +{earlier} earlier")
        for rec in shown:
            outcome = str(rec.get("outcome") or "")
            if outcome == "verified":
                glyph = "✅"
            elif outcome == "approved":
                glyph = "☑️"
            elif outcome in _FAILURE_OUTCOMES:
                glyph = "❌"
            else:
                glyph = "▪"
            idx_s = _e(rec.get("index") if rec.get("index") is not None else "?")
            rtitle = _e(str(rec.get("title") or ""))
            d = rec.get("duration")
            dstr = f" · {_fmt_duration(d)}" if d is not None else ""
            detail.append(f"{glyph} <b>{idx_s}</b> · {rtitle}{dstr}")
    if state.last_event_line:
        if detail:
            detail.append("")
        detail.append(f"💬 <i>{_e(state.last_event_line)}</i>")

    if detail:
        body = "\n".join(detail)
        return f"{header}\n<blockquote expandable>{body}</blockquote>"
    return header

# --------------------------------------------------------------------------- #
# Pure gating + rendering

# --------------------------------------------------------------------------- #
def render_event(
    event: dict,
    *,
    verbosity: str = DEFAULT_VERBOSITY,
    mode: str = "",
    run_id: str = "",
    context: Optional[dict] = None,
) -> Optional[str]:
    """Return an HTML message for ``event`` at ``verbosity``, else None.

    Pure function (no I/O, no global state). ``context`` is an OPTIONAL
    build-context dict ``{step_pos, step_title, phase}`` used only to annotate
    worker spin-up and adversarial-round lines with where they sit in the build
    ("· Step 2/5 · Adversarial"). The caller (:class:`TelegramNotifier`) owns
    that state; render stays stateless.

    Gating (each level is a superset of the prior). This tiering changed by
    operator request: spin-ups + adversarial rounds now surface at ``normal``
    so the operator sees models come up and critic verdicts, not just bare
    steps.
      quiet   : dispatch start + end-of-run summary card only.
      normal  : + step started/completed, + failures/stalls,
                + WORKER SPIN-UPS (every agent_started),
                + ADVERSARIAL ROUNDS (draft / verdict / generator rotation),
                + plan / reconcile / fallback transitions.
      verbose : + ToT branch activity + finer per-iteration detail.
      debug   : + heartbeats + per-call token usage.
    """
    if not isinstance(event, dict):
        return None
    level = _verbosity_index(verbosity)
    kind = event.get("kind")
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    rid = _e(run_id or event.get("run_id") or "")
    safe_mode = _e(mode)
    ctx_suffix = _build_context_suffix(context)

    # ---- quiet (level 0): dispatch lifecycle only ------------------------- #
    if kind == "lifecycle":
        ev = data.get("event")
        if ev == "dispatch_started":
            detail = data.get("detail") or {}
            gen_chain = detail.get("generator_chain") or []
            orch = " · ".join(str(w) for w in gen_chain)
            orch_line = f"\n<i>orchestrating {_e(orch)}</i>\n" if orch else ""
            # The goal/instruction leads the header card in bold (when known —
            # the notifier passes it via the context channel; render stays pure).
            goal = str((context or {}).get("goal") or "").strip() if isinstance(context, dict) else ""
            goal_line = f"<b>{_e(goal[:80])}</b>\n" if goal else ""
            return f"🟢 <b>Build started</b>\n{goal_line}{orch_line}\nmode <code>{safe_mode}</code> · run <code>{rid}</code>"
        if ev == "dispatch_finished":
            # The polished end-of-run card is owned by TelegramNotifier.finished()
            # (success/fail, duration, files, tokens). A bare marker here would be
            # a redundant second message for the same event, so we stay silent and
            # let the summary card be the single end-of-run notification.
            return None

        # ---- normal (level 1): WORKER SPIN-UPS ---------------------------- #
        # Every worker CLI call emits agent_started at the start of run_async;
        # worker/model/effort are stamped on the event TOP LEVEL by the bus.
        # This is the "a model spun up" signal the operator asked to see.
        if level >= 1 and ev == "agent_started":
            # agent_started is emitted at TWO layers: the canonical once-per-CLI-call
            # spin-up (agy_orchestrator/core/agent.py, detail={}) and per-internal-turn
            # events from the dashboard stream adapters (codex detail="turn.started",
            # claude detail="message_start"). Render ONLY the canonical spin-up — a
            # non-empty string detail is per-turn adapter noise that would emit many
            # duplicate "spun up" lines for a single multi-turn call (the exact spam
            # the one-message-per-spin-up intent must avoid).
            detail = data.get("detail")
            if isinstance(detail, str) and detail.strip():
                return None
            # The orchestrator stamps the agent's role in the (dict) detail; turn
            # it into a human verb so two same-provider calls (e.g. a step's
            # summarizer vs the next step's generator) are distinguishable rather
            # than two identical "spun up" lines.
            role = detail.get("role") if isinstance(detail, dict) else None
            role_label = _ROLE_LABEL.get(str(role)) if role else None
            role_str = f" · {_e(role_label)}" if role_label else ""
            worker = _e(event.get("worker") or "worker")
            model = _chip(event.get("model"))
            effort = _chip(event.get("effort"))
            chips = " · ".join(c for c in (model, effort) if c)
            chip_str = f" · <code>{_e(chips)}</code>" if chips else ""
            return f"🤖 <b>{worker}</b> spun up{role_str}{chip_str}{ctx_suffix}"

    # ---- orchestration transitions (steps, rounds, plan, …) --------------- #
    orch = _orchestration(event)
    if orch is not None:
        phase = orch.get("phase")
        action = orch.get("action")
        outcome = orch.get("outcome")

        # Failures/stalls — normal and above. Routed to the banner EXCEPT inside
        # an adversarial round, where the verdict line below keys on the same
        # outcomes (so an infra OOM mid-round reads as ⚠️ Verdict, with the
        # round position, rather than a context-free banner).
        if (
            level >= 1
            and str(outcome) in _FAILURE_OUTCOMES
            and phase != "adversarial"
        ):
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
                    ok = "✅" if outcome == "verified" else "☑️"
                else:
                    ok = "▪"
                # Duration + adversarial-round count come from BuildState via the
                # context channel (render stays pure). e.g. " · 2m14s · 2 rounds".
                ctx = context if isinstance(context, dict) else {}
                meta_bits: List[str] = []
                dur = ctx.get("step_duration")
                if dur is not None:
                    try:
                        meta_bits.append(_fmt_duration(dur))
                    except Exception:
                        pass
                rounds = ctx.get("step_rounds")
                if rounds:
                    try:
                        r = int(rounds)
                        meta_bits.append(f"{r} round{'s' if r != 1 else ''}")
                    except Exception:
                        pass
                meta_str = (" · " + " · ".join(meta_bits)) if meta_bits else ""
                return (
                    f"{ok} <b>Step {pos} done</b> · <i>{title}</i>"
                    f"{meta_str} · {_e(outcome)}{_run_tag(rid)}"
                )

        # ---- ADVERSARIAL ROUNDS — normal and above (operator request) ----- #
        # The operator was "not seeing adversarial rounds"; surface the draft
        # (generator spinning up for a round), the critic verdict, and any
        # generator rotation at the DEFAULT level.
        if level >= 1 and phase == "adversarial":
            it = orch.get("iteration")
            itn = orch.get("iteration_total")
            pos = f"{it}/{itn}" if itn else (str(it) if it is not None else "")
            it_str = f" · iter {_e(pos)}" if pos else ""
            if action == "generator_rotation":
                to_worker = _e(orch.get("to_worker") or "next worker")
                return f"↪️ <b>Rotation</b> → <b>{to_worker}</b>{it_str}{ctx_suffix}"
            if action == "iteration_started":
                # Generator drafting this round (carries its model+effort).
                model = _chip(orch.get("model"))
                mchip = f" · <code>{_e(model)}</code>" if model else ""
                return f"✍️ <b>Draft</b>{it_str}{mchip}{ctx_suffix}"
            if action in ("iteration_completed", "completed"):
                # Critic verdict, keyed on outcome so an infra OOM never reads
                # as a benign revision.
                if str(outcome) in _INFRA_OUTCOMES:
                    glyph, tag = "⚠️", _OUTCOME_LABEL.get(
                        str(outcome), "verifier infra"
                    )
                elif str(outcome) == "stalled":
                    glyph, tag = "⏹", "stalled"
                elif orch.get("approved") or orch.get("verified") or outcome in (
                    "verified", "approved"
                ):
                    glyph, tag = "✅", "approved"
                elif str(outcome) == "continue":
                    glyph, tag = "♻️", "revised"
                else:
                    glyph, tag = "♻️", "revised"
                return f"{glyph} <b>Verdict</b> · {_e(tag)}{it_str}{ctx_suffix}"

        # ---- plan / reconcile / fallback — normal and above --------------- #
        if level >= 1:
            if phase == "fallback":
                return f"↪️ <b>Reroute</b> · <i>{_e(orch.get('action') or 'fallback')}</i>{_run_tag(rid)}"
            if phase == "plan" and action == "completed":
                total = orch.get("step_total") or 0
                titles = orch.get("step_titles") or []
                head = f"📋 <b>Plan ready</b> · <b>{total} steps</b>{_run_tag(rid)}"
                if not titles:
                    return head
                # Tuck the full step list into an expandable so a 12-step plan
                # collapses to a one-line headline until the operator taps it.
                shown = titles[:40]
                extra = len(titles) - len(shown)
                n = len(shown)
                rows: List[str] = []
                for i, t in enumerate(shown, 1):
                    prefix = "└" if (i == n and extra <= 0) else "├"
                    rows.append(f"{prefix} <b>{i}</b> · {_e(t)}")
                if extra > 0:
                    rows.append(f"└ … +{extra} more")
                body = "\n".join(rows)
                return f"{head}\n<blockquote expandable>{body}</blockquote>"
            if phase in ("plan", "reconcile"):
                act = _e(action or phase)
                return f"🗂 <b>{_e(str(phase).title())}</b> · <i>{act}</i>{_run_tag(rid)}"

        # ---- verbose (level 2): ToT branch activity ----------------------- #
        if level >= 2 and phase == "tot":
            act = _e(action or "branch")
            branch = _e(event.get("branch") or orch.get("branch") or "")
            btag = f" · <code>{branch}</code>" if branch else ""
            return f"🌿 <b>ToT</b> · <i>{act}</i>{btag}{_run_tag(rid)}"

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
        dynamic_verbosity: Optional["Callable[[], Optional[str]]"] = None,
        instruction: Optional[str] = None,
    ):
        self.run_id = run_id
        self.mode = mode
        self.verbosity = normalize_verbosity(verbosity)
        # When the dispatch did NOT pin a level (no --telegram-verbosity flag),
        # follow the operator's persisted /verbosity default LIVE: a callable that
        # returns the current level (or None to mean "use the constructed
        # default"), re-read on EVERY event in __call__. This is the fix for
        # "setting /verbosity doesn't change anything" — the running dispatch's
        # push notifications now track a mid-build /verbosity change instead of
        # being frozen at construction. A pinned dispatch passes None and stays
        # fixed. Never trusted blindly: _effective_verbosity normalizes + isolates.
        self._dynamic_verbosity = dynamic_verbosity
        self.client = client
        self.chat_ids = list(chat_ids or [])
        self.sent = 0
        # Build-context tracked across events so a worker spin-up / adversarial
        # round can be annotated with the step + phase it belongs to. Updated
        # from orchestration events in __call__ BEFORE a non-orchestration event
        # is rendered, so a spin-up shows the step that just started. render_event
        # stays pure — this state lives only here.
        self._ctx: Dict[str, Any] = {"step_pos": "", "step_title": "", "phase": ""}
        # Delivery runs on a single background daemon thread so a slow/hung
        # Telegram endpoint never blocks the event-loop thread (mirrors
        # harness.run_monitor.Notifier). Rendered messages are pushed onto a
        # bounded queue and sent in order; a full queue drops oldest rather than
        # block (best-effort, never back-pressures _drain).
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=int(max_queue))
        self._worker: Optional[threading.Thread] = None
        self._worker_lock = threading.Lock()
        # TTL/mtime cache for the persisted bot_state.json read on the publish
        # hot path (see _live_state) — avoids per-event blocking file I/O.
        self._state_cache: dict = {}
        self._state_cache_ts: Optional[float] = None
        self._state_cache_mtime: Optional[float] = None
        # Set once stop()/close() has poison-pilled the worker so a late event
        # doesn't silently re-spin a daemon thread after teardown.
        self._stopped = False
        self._state = BuildState(run_id, mode)
        # The dispatch goal/instruction (truncated) — rendered as the bold lead of
        # the header card and the <i>title</i> lead of the status/summary card.
        goal = str(instruction or "").strip()
        if goal:
            self._state.goal = goal[:80]
            self._ctx["goal"] = self._state.goal
        self._pinned_card_sent = False

    @property
    def active(self) -> bool:
        return bool(self.client and self.client.configured and self.chat_ids)

    def _effective_verbosity(self) -> str:
        """The verbosity to gate on right now.

        Consults the live persisted level (``_dynamic_verbosity``) when the
        dispatch left the level unpinned, so a mid-build ``/verbosity`` change
        is honored immediately; otherwise the constructed level. Best-effort —
        a failing reader falls back to the constructed default, never raises.
        """
        reader = self._dynamic_verbosity
        if reader is not None:
            try:
                live = reader()
            except Exception:
                live = None
            if live:
                return normalize_verbosity(live)
        return self.verbosity

    def _live_state(self) -> dict:
        """Read the persisted bot_state.json live (same pattern as the dynamic
        /verbosity reader) so a mid-run /mute or /quiet takes effect immediately.
        Best-effort — returns {} on any failure, never raises.

        Cached with a short TTL (and an mtime short-circuit) so an event flood
        does NOT pay a blocking open()+json.load() per event on the synchronous
        publish path — that serialized disk I/O starved the producing coroutine
        on a slow/contended/NFS mount. Live /mute//quiet still take effect, just
        bounded by the TTL (a sub-second lag is harmless). The TTL is tunable via
        AGY_TELEGRAM_STATE_TTL (seconds; <=0 disables caching = always re-read).
        """
        import time as _time

        try:
            ttl = float(os.environ.get("AGY_TELEGRAM_STATE_TTL", "1.0"))
        except Exception:
            ttl = 1.0
        now = _time.monotonic()
        # Within the TTL window, reuse the cached parse — no syscall at all.
        if ttl > 0 and self._state_cache_ts is not None and (now - self._state_cache_ts) < ttl:
            return self._state_cache
        try:
            p = state_path()
            # An mtime short-circuit keeps a refresh cheap (one stat, no parse)
            # when the file is unchanged across TTL windows; we still re-stat at
            # most once per TTL so a fresh /mute lands within the window.
            try:
                mtime = os.path.getmtime(p)
            except Exception:
                mtime = None
            if (
                ttl > 0
                and mtime is not None
                and self._state_cache_mtime is not None
                and mtime == self._state_cache_mtime
            ):
                self._state_cache_ts = now
                return self._state_cache
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._state_cache = data if isinstance(data, dict) else {}
            self._state_cache_mtime = mtime
            self._state_cache_ts = now
            return self._state_cache
        except Exception:
            # Cache the empty result too so a missing/garbage file isn't reopened
            # on every event during the flood.
            self._state_cache = {}
            self._state_cache_mtime = None
            self._state_cache_ts = now
            return self._state_cache

    def _event_kind(self, event: Any) -> str:
        """Classify an event for suppression: 'failure' (a failed/stalled step or
        a failed dispatch_finished) is always delivered; everything else is
        suppressible 'progress'. The end-of-run summary card is classified
        'summary' at its own call site (finished()). Never raises."""
        try:
            data = event.get("data") if isinstance(event, dict) else None
            data = data if isinstance(data, dict) else {}
            orch = _orchestration(event) or {}
            if str(orch.get("outcome")) in _FAILURE_OUTCOMES:
                return "failure"
            if data.get("event") == "dispatch_finished" and data.get("success") is False:
                return "failure"
            return "progress"
        except Exception:
            return "progress"

    def _suppressed_for(self, event: Any) -> bool:
        """True iff THIS event should be dropped for every chat per F2 prefs.

        Consults :func:`_suppressed` against the live bot_state for each chat;
        only suppresses when ALL chats agree to drop it (a shared-chat broadcast
        is one message). Failure/summary kinds are never suppressed. Never raises."""
        try:
            kind = self._event_kind(event)
            if kind in _ALWAYS_DELIVER_KINDS:
                return False
            state = self._live_state()
            if not state.get("chats"):
                return False
            return all(
                _suppressed(cid, kind, state) for cid in self.chat_ids
            ) if self.chat_ids else False
        except Exception:
            return False

    def _ensure_worker(self) -> None:
        # Don't re-spin a daemon after the notifier was stopped/closed — a stray
        # late event must not leak a fresh thread post-teardown.
        if self._stopped:
            return
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
            item = self._queue.get()
            try:
                if item is None:  # poison pill -> stop the worker
                    return
                if isinstance(item, str):
                    self._deliver(item)
                elif isinstance(item, tuple):
                    action = item[0]
                    text = item[1] if len(item) > 1 else ""
                    markup = item[2] if len(item) > 2 else None
                    if action == "send_pinned":
                        for chat_id in self.chat_ids:
                            try:
                                res = self.client.send_message(chat_id, text)
                                record_sent_message(chat_id, res)
                                if res and res.get("result"):
                                    mid = res["result"].get("message_id")
                                    if mid:
                                        self._state.pinned_message_id = mid
                                        self.client.pin_chat_message(chat_id, mid)
                                self.sent += 1
                            except Exception as exc:
                                logger.debug("telegram send_pinned failed: %s", exc)
                    elif action == "send":
                        # A plain (optionally button-bearing) broadcast — the
                        # end-of-run summary card carries the inline keyboard.
                        for chat_id in self.chat_ids:
                            try:
                                res = self.client.send_message(
                                    chat_id, text, reply_markup=markup)
                                record_sent_message(chat_id, res)
                                self.sent += 1
                            except Exception as exc:
                                logger.debug("telegram send failed: %s", exc)
                    elif action == "edit_pinned":
                        mid = self._state.pinned_message_id
                        if mid:
                            for chat_id in self.chat_ids:
                                try:
                                    self.client.edit_message_text(
                                        chat_id, mid, text, reply_markup=markup)
                                except Exception as exc:
                                    logger.debug("telegram edit_pinned failed: %s", exc)
            except Exception as exc:  # never let the worker thread die noisily
                logger.debug("telegram worker error: %s", exc)
            finally:
                self._queue.task_done()

    def _deliver(self, text: str) -> None:
        """Synchronous send to every chat. Runs ONLY on the worker thread."""
        for chat_id in self.chat_ids:
            try:
                res = self.client.send_message(chat_id, text)
                record_sent_message(chat_id, res)
                self.sent += 1
            except Exception as exc:  # belt-and-suspenders; client already swallows
                logger.debug("telegram broadcast failed: %s", exc)

    def _enqueue_action(self, action: Any) -> None:
        if not action:
            return
        self._ensure_worker()
        try:
            self._queue.put_nowait(action)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Exception:
                pass
            try:
                self._queue.put_nowait(action)
            except Exception:
                pass

    def _broadcast(self, text: str) -> None:
        """Enqueue a rendered message for off-loop delivery. Never blocks."""
        if text:
            self._enqueue_action(text)

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

    def _update_context(self, event: dict) -> None:
        """Track current step + phase from orchestration events. Never raises."""
        try:
            orch = _orchestration(event)
            if not orch:
                return
            phase = orch.get("phase")
            if phase:
                self._ctx["phase"] = phase
            # Only a real "step" transition carries step position/title; keep the
            # last known step so an interleaved spin-up/round shows its step.
            if phase == "step":
                idx = orch.get("step_index")
                total = orch.get("step_total")
                if idx is not None or total is not None:
                    self._ctx["step_pos"] = (
                        f"{idx}/{total}" if total else (str(idx) if idx is not None else "")
                    )
                title = orch.get("step_title")
                if title is not None:
                    self._ctx["step_title"] = str(title)
        except Exception as exc:
            logger.debug("telegram context update failed: %s", exc)

    def __call__(self, event: dict) -> None:
        """EventBus sink entrypoint — never raises."""
        if not self.active:
            return
        # Update build-context FIRST so a spin-up/round event that arrives right
        # after a step transition is annotated with that step.
        self._update_context(event)
        
        try:
            self._state.update(event)
        except Exception as exc:
            logger.debug("BuildState update failed: %s", exc)

        # After the state update, surface the just-completed step's duration +
        # adversarial round count to render_event via the context channel, so the
        # step-done line can read "✅ Step 3/8 done · <i>title</i> · 2m14s · 2
        # rounds · verified". render_event stays pure. Cleared for non-completion
        # events so a later spin-up line doesn't inherit a stale duration.
        try:
            orch = _orchestration(event) or {}
            if orch.get("phase") == "step" and orch.get("action") == "completed":
                self._ctx["step_duration"] = self._state.last_step_duration
                self._ctx["step_rounds"] = self._state.last_step_rounds
            else:
                self._ctx.pop("step_duration", None)
                self._ctx.pop("step_rounds", None)
        except Exception as exc:
            logger.debug("telegram step-done annotate failed: %s", exc)

        try:
            text = render_event(
                event,
                verbosity=self._effective_verbosity(),
                mode=self.mode,
                run_id=self.run_id,
                context=self._ctx,
            )
        except Exception as exc:
            logger.debug("telegram render failed: %s", exc)
            return
            
        # F2: drop progress chatter while muted / in quiet hours (failures + the
        # final summary are classified non-progress and always pass through).
        suppressed = self._suppressed_for(event)

        if text:
            try:
                clean = str(text).replace("\n", " ").strip()
                import re
                clean = re.sub(r'<[^>]+>', '', clean)
                self._state.last_event_line = (clean[:80] + "…") if len(clean) > 80 else clean
            except Exception:
                pass
            if not suppressed:
                self._broadcast(text)

        try:
            kind = event.get("kind")
            is_meaningful = kind not in ("heartbeat",) and event.get("data", {}).get("event") != "dispatch_started"
            if suppressed:
                is_meaningful = False  # the live status card is progress chatter too
            if is_meaningful and not self._pinned_card_sent:
                self._pinned_card_sent = True
                card_text = render_status_card(self._state)
                self._state.last_render_signature = _card_signature(self._state)
                self._state.last_render_ts = float(event.get("ts", 0.0))
                self._enqueue_action(("send_pinned", card_text))
            elif self._pinned_card_sent and not suppressed:
                now = float(event.get("ts", 0.0))
                sig = _card_signature(self._state)
                orch = _orchestration(event) or {}
                action = orch.get("action")
                is_force = action in ("started", "completed", "iteration_completed", "generator_rotation")
                if event.get("data", {}).get("event") == "dispatch_finished":
                    is_force = True
                time_since = now - self._state.last_render_ts
                if (is_force and time_since >= 1.2) or (time_since >= 1.2 and sig != self._state.last_render_signature):
                    self._state.last_render_signature = sig
                    self._state.last_render_ts = now
                    card_text = render_status_card(self._state)
                    self._enqueue_action(("edit_pinned", card_text))
        except Exception as exc:
            logger.debug("Status card edit failed: %s", exc)

    def _run_keyboard(self) -> Optional[dict]:
        """Inline-keyboard reply_markup for THIS run's summary card (F1).

        [📂 Files] [❓ Why] [📊 Diff]; callback_data is ``"verb:short_run_id"``
        (the short tag keeps it under Telegram's 64-byte cap). The bot's callback
        dispatch routes a tap back to the read-only helpers. Never raises."""
        try:
            rid = str(self.run_id or "").strip()
            short = rid[-6:] if len(rid) > 6 else rid
            if not short:
                return None
            return {
                "inline_keyboard": [[
                    {"text": "📂 Files", "callback_data": f"files:{short}"},
                    {"text": "❓ Why", "callback_data": f"why:{short}"},
                    {"text": "📊 Diff", "callback_data": f"diff:{short}"},
                ]]
            }
        except Exception:
            return None

    def finished(self, meta: Optional[dict]) -> None:
        """Send the polished final-summary card, then drain. Never raises."""
        if not self.active:
            return
        try:
            card = self._summary_card(meta or {})
            markup = self._run_keyboard()
            if getattr(self, '_pinned_card_sent', False):
                self._enqueue_action(("edit_pinned", card, markup))
            else:
                self._enqueue_action(("send", card, markup))
        except Exception as exc:
            logger.debug("telegram summary failed: %s", exc)
        # Drain so the card actually leaves the process before dispatch returns.
        # Bounded so a hung endpoint can't stall teardown.
        self.flush(timeout=6.0)
        # Reap the daemon delivery thread now that the guaranteed final-summary
        # card has been flushed — otherwise the worker blocks on queue.get()
        # forever and a long-lived broker leaks one live thread (+ its
        # BuildState/queue closure) per dispatch. Idempotent.
        self.stop()

    def stop(self, timeout: float = 2.0) -> None:
        """Poison-pill the delivery worker and join it (bounded). Idempotent.

        The worker thread blocks on ``queue.get()`` forever waiting for work; a
        long-lived broker that builds one notifier per dispatch would leak a
        live thread each time. ``stop()`` enqueues the ``None`` poison pill (which
        :meth:`_drain_queue` treats as "return") and joins the worker with a
        bounded timeout so teardown can never hang. Safe to call more than once
        and safe to call when no worker was ever started. Never raises.

        ``close()``/``shutdown()`` are aliases. :meth:`finished` calls this after
        the final-summary flush so the guaranteed end-of-run card still goes out
        before the thread exits.
        """
        self._stopped = True
        worker = self._worker
        try:
            # Only enqueue a pill (and pay a join) if a worker actually exists;
            # otherwise there's nothing to reap.
            if worker is not None:
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    # Make room for the pill so the worker is guaranteed to see
                    # it and exit rather than block forever on a full queue.
                    try:
                        self._queue.get_nowait()
                        self._queue.task_done()
                    except Exception:
                        pass
                    try:
                        self._queue.put_nowait(None)
                    except Exception:
                        pass
                worker.join(timeout=max(0.0, float(timeout)))
        except Exception as exc:  # best-effort teardown — never raise
            logger.debug("telegram notifier stop failed: %s", exc)

    # Aliases so a caller (or a broker) can reap the worker via any common name.
    def close(self, timeout: float = 2.0) -> None:
        self.stop(timeout=timeout)

    def shutdown(self, timeout: float = 2.0) -> None:
        self.stop(timeout=timeout)

    def _summary_card(self, meta: dict) -> str:
        """Rich multi-line end-of-run card from BuildState + meta (spec §8).

        Lines: a headline (✅ verified / ☑️ complete / 🛑 failed), the goal, a
        "N/N steps · status · duration" line, a changed-files block (first ~8 +
        "… +K more"), a <pre> per-step recap, then tokens + mode + run. REAL
        newlines only.
        """
        state = self._state
        success = bool(meta.get("success"))
        mode = _e(meta.get("mode") or self.mode)
        dur = _fmt_duration(meta.get("duration_s"))
        rid = _e(meta.get("run_id") or self.run_id)
        changed = meta.get("changed_files")
        files = [str(f) for f in changed] if isinstance(changed, list) else []
        n_files = len(files)

        quality = meta.get("quality") if isinstance(meta.get("quality"), dict) else {}
        raw_conf = quality.get("confidence")
        confidence = str(raw_conf).strip() if raw_conf is not None else ""
        verified = confidence.lower() == "verified"

        tokens = meta.get("tokens") if isinstance(meta.get("tokens"), dict) else {}
        grand = tokens.get("grand_total") if isinstance(tokens.get("grand_total"), dict) else {}
        grand_total = grand.get("total_tokens")
        token_str = f"{grand_total:,} tok" if isinstance(grand_total, int) else ""

        # ---- headline ---------------------------------------------------- #
        if success:
            if verified:
                headline = "✅ <b>Build verified</b>"
                status_word = "verified"
            else:
                headline = "☑️ <b>Build complete</b>"
                status_word = confidence if confidence and confidence.lower() not in (
                    "n/a", "na", "none") else "complete"
        else:
            headline = "🛑 <b>Build failed</b>"
            status_word = str(meta.get("error") or meta.get("run_outcome") or "failed")

        lines: List[str] = [headline]

        # ---- goal -------------------------------------------------------- #
        goal = str(getattr(state, "goal", "") or "").strip()
        if goal:
            lines.append(f"<i>{_e(goal[:80])}</i>")

        # ---- N/N steps · status · duration ------------------------------- #
        total = state.step_total
        done = state.n_steps_done
        if total:
            steps_str = f"{done}/{total} steps"
        elif done:
            steps_str = f"{done} steps"
        else:
            steps_str = ""
        meta_bits = [b for b in (steps_str, _e(status_word), _e(dur)) if b]
        if meta_bits:
            lines.append(" · ".join(meta_bits))

        # ---- expandable detail: changed-files list + <pre> per-step recap -- #
        # The headline + key stats stay on the surface; the long lists collapse
        # into a single tap-to-expand block so a 40-file / 30-step run is a tidy
        # card rather than a wall of text.
        detail: List[str] = []
        if n_files:
            detail.append(f"📂 <b>{n_files} file{'s' if n_files != 1 else ''} changed</b>")
            shown = files[:20]
            for f in shown:
                detail.append(f"  • <code>{_e(f)}</code>")
            extra = n_files - len(shown)
            if extra > 0:
                detail.append(f"  … +{extra} more")

        records = list(getattr(state, "step_records", []) or [])
        if records:
            # Cap the recap so a long run stays well under 4096 chars even with
            # the changed-files list above it. Keep the most-recent steps and
            # note "+K earlier" at the top.
            shown_recs = records[-_SUMMARY_STEP_CAP:]
            earlier = len(records) - len(shown_recs)
            recap_rows = []
            for rec in shown_recs:
                idx = rec.get("index")
                title = str(rec.get("title") or "")
                outcome = str(rec.get("outcome") or "")
                if outcome == "verified":
                    glyph = "✅"
                elif outcome == "approved":
                    glyph = "☑️"
                elif outcome in _FAILURE_OUTCOMES:
                    glyph = "❌"
                else:
                    glyph = "▪"
                d = rec.get("duration")
                dstr = _fmt_duration(d) if d is not None else ""
                # roughly aligned columns: index, glyph, padded title, duration.
                idx_s = str(idx if idx is not None else "?")
                title_col = title[:22].ljust(22)
                recap_rows.append(f"{idx_s:>2} {glyph} {title_col} {dstr}")
            if earlier > 0:
                recap_rows.insert(0, f"… +{earlier} earlier")
            recap = "\n".join(_e(r) for r in recap_rows)
            if detail:
                detail.append("")
            detail.append(f"<b>Step recap</b>\n<pre>{recap}</pre>")

        if detail:
            body = "\n".join(detail)
            lines.append(f"<blockquote expandable>{body}</blockquote>")

        # ---- tokens + mode + run ----------------------------------------- #
        footer_bits = [b for b in (token_str, f"<code>{mode}</code>") if b]
        footer = " · ".join(footer_bits)
        lines.append(f"{footer} · run <code>{rid}</code>")

        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Enable logic (shared by dispatch + bot)
# --------------------------------------------------------------------------- #
def resolve_verbosity(explicit: Optional[str] = None) -> str:
    """Verbosity from explicit flag, else AGY_TELEGRAM_VERBOSITY, else default."""
    if explicit:
        return normalize_verbosity(explicit)
    return normalize_verbosity(os.environ.get("AGY_TELEGRAM_VERBOSITY"))


def state_path(path: Optional[str] = None) -> str:
    """Path to the bot's persisted state file (verbosity, /track cursors).

    Lives OUTSIDE the repo (default ``/home/leah/tgbot/data/bot_state.json``;
    override via ``AGY_TELEGRAM_STATE``). Owned here so the dispatch-side notifier
    and the bot daemon agree on one file.
    """
    return path or os.environ.get("AGY_TELEGRAM_STATE") or DEFAULT_STATE_PATH


# --------------------------------------------------------------------------- #
# /clear — shared sent-message-id log
#
# Telegram has no "wipe chat history" API. A bot CAN delete individual messages
# (< 48h old; in a private chat, both its own and the user's), but only if it
# knows their message_ids. The daemon sees incoming ids and its own sends, while
# the DISPATCH process sends most of the build chatter via TelegramNotifier — a
# different process. So both append (chat_id, message_id, ts) to ONE shared JSONL
# log next to the state file; /clear reads it, deletes everything in-window, then
# purges the chat's entries. Best-effort throughout: a missing/garbage/unwritable
# log degrades to "clear what we can" and never raises into a send path.
# --------------------------------------------------------------------------- #
# Telegram only deletes messages < 48h old; stay safely under the boundary.
CLEAR_MAX_AGE_SECONDS = 47 * 3600
# Compact stale lines when the log grows past this (keeps it bounded between
# /clear calls without reading the whole file on every append).
_CLEARLOG_COMPACT_BYTES = 1_000_000


def _clock() -> float:
    """Wall-clock seam so tests can drive the /clear age window deterministically."""
    import time as _t
    return _t.time()


def clearlog_path(state_file: Optional[str] = None) -> str:
    """Path to the shared sent-message-id log (``sent_messages.jsonl``).

    Lives next to the bot state file (outside the repo) so the dispatch notifier
    and the bot daemon append to the same log.
    """
    base = state_path(state_file)
    return os.path.join(os.path.dirname(base) or ".", "sent_messages.jsonl")


def _extract_message_id(message_id: Any) -> Optional[int]:
    """Coerce an int message_id, or pull result.message_id from a sendMessage dict."""
    mid = message_id
    if isinstance(mid, dict):
        result = mid.get("result")
        mid = result.get("message_id") if isinstance(result, dict) else None
    if mid is None or isinstance(mid, bool):
        return None
    try:
        return int(mid)
    except (TypeError, ValueError):
        return None


def record_sent_message(
    chat_id: Any, message_id: Any, *,
    state_file: Optional[str] = None, now: Optional[float] = None,
) -> None:
    """Append a (chat_id, message_id) to the shared id-log so /clear can delete it.

    ``message_id`` may be an int OR a sendMessage API result dict (its
    ``result.message_id`` is used; a None/non-ok result is silently ignored).
    Best-effort: any failure (falsy id, unwritable dir) is swallowed.
    """
    if chat_id is None:
        return
    mid = _extract_message_id(message_id)
    if mid is None:
        return
    ts = float(now) if now is not None else _clock()
    path = clearlog_path(state_file)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"c": str(chat_id), "m": mid, "t": ts}) + "\n")
    except Exception as exc:
        logger.debug("clearlog append failed: %s", exc)
        return
    try:
        if os.path.getsize(path) > _CLEARLOG_COMPACT_BYTES:
            _rewrite_clearlog(path, drop_chat=None, now=ts)
    except Exception:
        pass


def read_clear_targets(
    chat_id: Any, *, state_file: Optional[str] = None,
    now: Optional[float] = None, max_age: float = CLEAR_MAX_AGE_SECONDS,
) -> List[int]:
    """De-duped, within-window message ids logged for a chat (insertion order)."""
    path = clearlog_path(state_file)
    cutoff = (float(now) if now is not None else _clock()) - float(max_age)
    target = str(chat_id)
    seen: set = set()
    ids: List[int] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if str(obj.get("c")) != target:
                    continue
                try:
                    ts = float(obj.get("t"))
                    mid = int(obj.get("m"))
                except (TypeError, ValueError):
                    continue
                if ts < cutoff or mid in seen:
                    continue
                seen.add(mid)
                ids.append(mid)
    except FileNotFoundError:
        return []
    except Exception as exc:
        logger.debug("clearlog read failed: %s", exc)
        return []
    return ids


def _rewrite_clearlog(
    path: str, *, drop_chat: Optional[str],
    now: Optional[float] = None, max_age: float = CLEAR_MAX_AGE_SECONDS,
) -> None:
    """Atomically rewrite the log, dropping ``drop_chat``'s lines + all stale lines."""
    cutoff = (float(now) if now is not None else _clock()) - float(max_age)
    kept: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if drop_chat is not None and str(obj.get("c")) == drop_chat:
                    continue
                try:
                    if float(obj.get("t")) < cutoff:
                        continue
                except (TypeError, ValueError):
                    continue
                kept.append(s)
    except FileNotFoundError:
        return
    except Exception as exc:
        logger.debug("clearlog rewrite read failed: %s", exc)
        return
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            if kept:
                fh.write("\n".join(kept) + "\n")
        os.replace(tmp, path)
    except Exception as exc:
        logger.debug("clearlog rewrite write failed: %s", exc)


def purge_clear_log(
    chat_id: Any, *, state_file: Optional[str] = None,
    now: Optional[float] = None, max_age: float = CLEAR_MAX_AGE_SECONDS,
) -> None:
    """Drop a chat's entries (and any globally-stale entries) from the id-log."""
    _rewrite_clearlog(
        clearlog_path(state_file), drop_chat=str(chat_id), now=now, max_age=max_age)


def load_persisted_verbosity(path: Optional[str] = None) -> Optional[str]:
    """The operator's ``/verbosity``-set default from the bot state file, or None.

    Returns the normalized level when the state file records one, else ``None``
    (file absent / unreadable / garbage / no ``verbosity`` key). Best-effort and
    never raises — a missing file just means "no persisted preference", and the
    caller falls back to its own default. This is what lets a ``/verbosity``
    change take effect on a LIVE dispatch's push notifications: the notifier
    re-reads this each event rather than freezing the level at construction.
    """
    try:
        with open(state_path(path), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    if isinstance(data, dict):
        v = data.get("verbosity")
        if v:
            return normalize_verbosity(v)
    return None


# --------------------------------------------------------------------------- #
# F2 — per-chat /mute, /watch, quiet-hours suppression
#
# Per-chat preferences live under ``state["chats"][str(chat_id)]`` =
# ``{"mute_until": <epoch|"on"|None>, "quiet_window": "HH:MM-HH:MM"|None}``.
# The single ``_suppressed`` predicate is consulted on BOTH delivery paths:
# the dispatch-side TelegramNotifier and the bot's tail. It reads the SAME live
# bot_state.json (just like load_persisted_verbosity), so toggling /mute mid-run
# takes effect immediately. Policy: while muted or inside the quiet window, drop
# PROGRESS chatter but ALWAYS deliver "failure" + "summary" events.
# --------------------------------------------------------------------------- #

# Event classes that are NEVER suppressed (failures must always reach the
# operator; the end-of-run summary card is the one guaranteed message). Anything
# not in this set is treated as suppressible progress chatter.
_ALWAYS_DELIVER_KINDS = frozenset({"failure", "summary"})


def parse_quiet_window(spec: Optional[str]):
    """Parse a ``HH:MM-HH:MM`` quiet-hours window into ``(start, end)`` minutes.

    Returns a ``(start_minutes, end_minutes)`` tuple of ints in ``[0, 1439]`` on
    success, or ``None`` for any garbage (empty, missing dash, out-of-range, or
    non-numeric). NEVER raises — a bad window is rejected, not an exception.
    An overnight window (start >= end, e.g. 23:00-07:00) is valid and wraps.
    """
    try:
        s = str(spec or "").strip()
        if "-" not in s:
            return None
        lo, hi = s.split("-", 1)
        start = _parse_hhmm(lo)
        end = _parse_hhmm(hi)
        if start is None or end is None:
            return None
        return (start, end)
    except Exception:
        return None


def _parse_hhmm(token: str):
    """``HH:MM`` -> minutes-of-day int in [0,1439], or None. Never raises."""
    try:
        t = str(token or "").strip()
        if ":" not in t:
            return None
        hh_s, mm_s = t.split(":", 1)
        hh = int(hh_s)
        mm = int(mm_s)
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return None
        return hh * 60 + mm
    except Exception:
        return None


def _now_minute_of_day(now) -> Optional[int]:
    """Minutes-of-day for an injected ``now`` (datetime, epoch float/int), or the
    real local clock when ``now`` is None. Never raises; returns None on garbage."""
    try:
        if now is None:
            import datetime as _dt
            n = _dt.datetime.now()
            return n.hour * 60 + n.minute
        # datetime instance
        if hasattr(now, "hour") and hasattr(now, "minute"):
            return int(now.hour) * 60 + int(now.minute)
        # epoch seconds (int/float)
        import datetime as _dt
        n = _dt.datetime.fromtimestamp(float(now))
        return n.hour * 60 + n.minute
    except Exception:
        return None


def _now_epoch(now) -> Optional[float]:
    """An epoch-seconds float for an injected ``now`` (datetime/epoch) or the real
    clock when None. Used to compare against ``mute_until``. Never raises."""
    try:
        if now is None:
            import time as _t
            return _t.time()
        if hasattr(now, "timestamp"):
            return float(now.timestamp())
        return float(now)
    except Exception:
        return None


def _in_quiet_window(window, minute: Optional[int]) -> bool:
    """True iff ``minute`` (minutes-of-day) falls inside ``window`` (start,end).

    Handles overnight wrap (start > end). Never raises."""
    try:
        if window is None or minute is None:
            return False
        start, end = window
        if start == end:
            return False  # zero-width window means "never quiet"
        if start < end:
            return start <= minute < end
        # overnight: e.g. 23:00-07:00 -> [23:00, 24:00) U [00:00, 07:00)
        return minute >= start or minute < end
    except Exception:
        return False


def chat_prefs(state: Any, chat_id: Any) -> dict:
    """The per-chat preference dict from ``state["chats"][str(chat_id)]``.

    Returns ``{}`` for any missing/garbage state or chat id. Never raises."""
    try:
        if not isinstance(state, dict):
            return {}
        chats = state.get("chats")
        if not isinstance(chats, dict):
            return {}
        prefs = chats.get(str(chat_id))
        return prefs if isinstance(prefs, dict) else {}
    except Exception:
        return {}


def _is_muted(prefs: dict, now) -> bool:
    """True iff the chat is currently muted per ``prefs['mute_until']``.

    ``"on"`` (or any non-numeric truthy string) means muted indefinitely; an
    epoch float/int means muted until that time; absent/falsey means not muted.
    Never raises."""
    try:
        mu = prefs.get("mute_until")
        if not mu:
            return False
        # Indefinite mute (e.g. "on").
        if isinstance(mu, str):
            try:
                mu_epoch = float(mu)
            except Exception:
                return True  # "on" / non-numeric string -> muted indefinitely
        else:
            mu_epoch = float(mu)
        now_epoch = _now_epoch(now)
        if now_epoch is None:
            return True  # can't tell time -> respect the mute conservatively
        return now_epoch < mu_epoch
    except Exception:
        return False


def _suppressed(chat_id: Any, event_kind: Any, state: Any, *, now: Any = None) -> bool:
    """Should this chat NOT receive an event of ``event_kind`` right now?

    The single suppression predicate for F2, consulted on BOTH delivery paths
    (TelegramNotifier + bot tail). ``event_kind`` is a coarse class:
    ``"failure"`` and ``"summary"`` are ALWAYS delivered (return False); anything
    else is suppressible progress chatter, dropped while the chat is muted or
    inside its quiet window.

    Deterministic + testable: ``now`` may be injected (a ``datetime`` or epoch
    seconds); when None the real local clock is used (only outside assertions).
    Fail-open + NEVER raises — on any garbage it returns False (deliver), so a
    bad pref can never silence a failure or crash the delivery thread.
    """
    try:
        if str(event_kind) in _ALWAYS_DELIVER_KINDS:
            return False
        prefs = chat_prefs(state, chat_id)
        if not prefs:
            return False
        if _is_muted(prefs, now):
            return True
        window = parse_quiet_window(prefs.get("quiet_window"))
        if window is not None and _in_quiet_window(window, _now_minute_of_day(now)):
            return True
        return False
    except Exception:
        return False  # fail-open: never suppress on an error

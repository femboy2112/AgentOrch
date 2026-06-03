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
                    
                elif action == "completed":
                    if idx:
                        self.completed_steps.add(idx)
                        self.n_steps_done = len(self.completed_steps)
                        if idx in self._step_started_at:
                            self.step_durations.append(max(0.0, float(ts) - float(self._step_started_at[idx])))
                    outcome = orch.get("outcome")
                    if outcome == "verified":
                        self.n_verified += 1
                    elif outcome == "approved":
                        self.n_approved += 1

            elif phase == "adversarial":
                if action == "iteration_started":
                    self.cur_iter = orch.get("iteration")
                    self.cur_iter_total = orch.get("iteration_total")

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
        stall_flag = f" ⚠️ no progress {_fmt_duration(state.since_progress_s)}"

    last_block = f"\n└ last: {_e(state.last_event_line)}" if state.last_event_line else ""

    worker_state = _e(state.active_role or "running")

    return (
        f"⚡ <b>Building</b> · <code>{rid_short}</code>\n"
        f"<code>{bar}</code>  <b>{step_ref}</b>\n"
        f"<i>{_e(title)}</i>\n\n"
        f"{phase_glyph} {phase_label} · {worker_state}{iter_str}\n"
        f"⏱ {elapsed_str} elapsed · {eta_str}{stall_flag}"
        f"{last_block}"
    )

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
            return f"🟢 <b>Build started</b>\n{orch_line}\nmode <code>{safe_mode}</code> · run <code>{rid}</code>"
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
                return f"{ok} <b>Step {pos} done</b> · <i>{title}</i>\n   {_e(outcome)}{_run_tag(rid)}"

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
                lines = [f"📋 <b>Plan ready</b> · <b>{total} steps</b>{_run_tag(rid)}"]
                for i, t in enumerate(titles, 1):
                    prefix = "└" if i == len(titles) else "├"
                    lines.append(f"{prefix} {i} · {_e(t)}")
                    if i >= 12 and len(titles) > 13:
                        lines.append(f"└ … +{len(titles) - 12} more")
                        break
                return "\n".join(lines)
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
        self._state = BuildState(run_id, mode)
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
            item = self._queue.get()
            try:
                if item is None:  # poison pill -> stop the worker
                    return
                if isinstance(item, str):
                    self._deliver(item)
                elif isinstance(item, tuple):
                    action, text = item
                    if action == "send_pinned":
                        for chat_id in self.chat_ids:
                            try:
                                res = self.client.send_message(chat_id, text)
                                if res and res.get("result"):
                                    mid = res["result"].get("message_id")
                                    if mid:
                                        self._state.pinned_message_id = mid
                                        self.client.pin_chat_message(chat_id, mid)
                                self.sent += 1
                            except Exception as exc:
                                logger.debug("telegram send_pinned failed: %s", exc)
                    elif action == "edit_pinned":
                        mid = self._state.pinned_message_id
                        if mid:
                            for chat_id in self.chat_ids:
                                try:
                                    self.client.edit_message_text(chat_id, mid, text)
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
                self.client.send_message(chat_id, text)
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
            
        if text:
            try:
                clean = str(text).replace("\n", " ").strip()
                import re
                clean = re.sub(r'<[^>]+>', '', clean)
                self._state.last_event_line = (clean[:80] + "…") if len(clean) > 80 else clean
            except Exception:
                pass
            self._broadcast(text)

        try:
            kind = event.get("kind")
            is_meaningful = kind not in ("heartbeat",) and event.get("data", {}).get("event") != "dispatch_started"
            if is_meaningful and not self._pinned_card_sent:
                self._pinned_card_sent = True
                card_text = render_status_card(self._state)
                self._state.last_render_signature = _card_signature(self._state)
                self._state.last_render_ts = float(event.get("ts", 0.0))
                self._enqueue_action(("send_pinned", card_text))
            elif self._pinned_card_sent:
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

    def finished(self, meta: Optional[dict]) -> None:
        """Send the polished final-summary card, then drain. Never raises."""
        if not self.active:
            return
        try:
            card = self._summary_card(meta or {})
            if getattr(self, '_pinned_card_sent', False):
                self._enqueue_action(("edit_pinned", card))
            else:
                self._broadcast(card)
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


def state_path(path: Optional[str] = None) -> str:
    """Path to the bot's persisted state file (verbosity, /track cursors).

    Lives OUTSIDE the repo (default ``/home/leah/tgbot/data/bot_state.json``;
    override via ``AGY_TELEGRAM_STATE``). Owned here so the dispatch-side notifier
    and the bot daemon agree on one file.
    """
    return path or os.environ.get("AGY_TELEGRAM_STATE") or DEFAULT_STATE_PATH


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

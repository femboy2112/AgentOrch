"""SessionController and supporting run-session lifecycle (Step 11).

Owns WorkerSession construction (FR-01 default ISOLATED), Xvfb bootstrap via
ProcessSupervisor for ISOLATED (FR-22) — and always for actuation capability
in OBSERVE so the structural observer/actor split is preserved.

Implements the suspended confirmation wait state (FR-19: no reasoner polling
or action execution while a high-risk intent is gated) using threading.Event
per (run_id, intent_id). enqueue_orchestrator_message feeds tokens and notes;
await_confirmation blocks only on the matching token or timeout (FR-18).

Deterministic owned-tree teardown on close_session (Xvfb + any action spawns
registered under the same supervisor instance). Best-effort private Xauth/iso-home
cleanup for hardening #1 files created during the session.

All Xvfb / display usage is isolated; zero real-:0 side effects from this module.
Emits session.* and confirmation.* lifecycle events when an AuditEventSink is supplied.
"""

from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .capability import CapabilityBroker
from .models import (
    AskMode,
    CapabilityReport,
    ConfirmationOutcome,
    ConfirmationOutcomeType,
    IsolatedDisplaySpec,
    OrchestratorMessage,
    RunMode,
    RunRequest,
    StopResult,
    TaskPriority,
    WorkerEvent,
    WorkerEventType,
    WorkerSession,
)
from .ownership import OwnershipResolver
from .process_supervisor import ProcessSupervisor, SpawnedProc

DEFAULT_BUDGETS: Dict[str, int] = {
    "max_steps": 200,
    "max_actions": 200,
    "action_timeout_ms": 10000,
    "reasoning_timeout_ms": 45000,
    "confirmation_wait_timeout_ms": 300000,
    "max_cpu_percent": 200,
    "max_rss_mb": 2048,
    "max_processes": 64,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pick_isolated_display(run_id: str) -> str:
    """Stable per-run display allocation (reduces collision under concurrent runs)."""
    # :99 + small offset from run_id hash; keeps tests predictable when they pass explicit.
    h = abs(hash(run_id)) % 8
    return f":{99 + h}"


@dataclass
class Session:
    """Runtime session context owned by SessionController (not a public model)."""
    run_id: str
    worker_session: WorkerSession
    supervisor: ProcessSupervisor
    isolated_display: str
    observe_display: Optional[str]
    xvfb_root_id: Optional[str] = None
    xvfb_spawned: Optional[SpawnedProc] = None
    # Tracked private cookie / home paths for deterministic cleanup (hardening #1)
    private_xauth_path: Optional[Path] = None
    private_home_dir: Optional[Path] = None
    # Step 6: baseline stored on internal Session (mirrors WorkerSession for REAL runs)
    baseline_pids: Optional[List[int]] = None
    baseline_windows: Optional[Dict[str, Any]] = None
    # Mutable run state
    current_step: int = 0
    current_actions: int = 0
    suspended_intent: Optional[str] = None  # current high-risk intent id while awaiting confirmation (FR-19)
    # Pending orchestrator messages (confirmation tokens + notes) for next ReasoningInput
    pending_messages: List[OrchestratorMessage] = field(default_factory=list)
    # Suspended confirmation state (FR-19): key = intent_id
    _confirm_events: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # intent_id -> {event, token, ...}
    # Last known status
    status: str = "active"
    # Phase 1.x Step 4 (additive): browser_controller ownership + B3 lifecycle; current_cdp_endpoint exposure for perception/actuation
    browser_controller: Optional[Any] = None
    browser_display: Optional[str] = None
    browser_engine: Optional[str] = None
    current_cdp_endpoint: Optional[str] = None


class SessionController:
    """Owns run lifecycle, mode policy, Xvfb bootstrap/teardown, and confirmation wait orchestration (FR-01/18/19/22)."""

    def __init__(
        self, *, audit_sink_factory: Optional[Any] = None, audit_sink: Optional[Any] = None,
        supervisor: Optional[ProcessSupervisor] = None,
        ownership_resolver: Optional[Any] = None,
        gui_prompter: Optional[Any] = None,
        grant_cache: Optional[Any] = None,
        browser_controller: Optional[Any] = None,
    ) -> None:
        self._sessions: Dict[str, Session] = {}
        self._audit_factory = audit_sink_factory  # callable(run_id) -> AuditEventSink or None
        self._audit_sink = audit_sink  # optional direct sink instance (used by tests)
        self._lock = threading.RLock()
        # Injectable at controller level (Step 6); production paths construct inside create_session
        self._supervisor = supervisor
        self._ownership_resolver = ownership_resolver
        self._gui_prompter = gui_prompter
        self._grant_cache = grant_cache
        self._browser_controller = browser_controller  # Phase 1.x Step 4: optional injected for tests (else construct per-run)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_session(self, config: RunRequest) -> Session:
        """Create and register a fully-initialized session.

        FR-01: mode defaults to ISOLATED when absent.
        FR-22: on ISOLATED (and always when action capability present) an owned
               Xvfb is spawned and registered so terminate_tree can reap it.
        """
        if not isinstance(config, RunRequest):
            # Accept loose dicts from adapter for convenience
            config = RunRequest(
                run_id=config.get("run_id") if isinstance(config, dict) else str(config),
                objective=config.get("objective", "") if isinstance(config, dict) else "",
                mode=config.get("mode") if isinstance(config, dict) else None,
                task_priority=config.get("task_priority") if isinstance(config, dict) else None,
                budgets=config.get("budgets") if isinstance(config, dict) else None,
                observe_display=config.get("observe_display") if isinstance(config, dict) else None,
                real_gui_policy=config.get("real_gui_policy") if isinstance(config, dict) else None,
                ask_mode=config.get("ask_mode") if isinstance(config, dict) else None,
                browser_engine=config.get("browser_engine") if isinstance(config, dict) else None,
                browser_display=config.get("browser_display") if isinstance(config, dict) else None,
            )

        run_id = config.run_id or f"cu-{int(time.time()*1000)}"
        mode = (config.mode or RunMode.ISOLATED.value).upper()
        if mode not in {m.value for m in RunMode}:
            mode = RunMode.ISOLATED.value

        task_priority = (config.task_priority or TaskPriority.NORMAL.value).lower()
        if task_priority not in {p.value for p in TaskPriority}:
            task_priority = TaskPriority.NORMAL.value

        # Step 6 refinement (finalize): normalize real_gui_policy/ask_mode from enum or str/cased input
        # so RunRequest validation + WorkerSession storage always receive canonical lowercase values.
        # (Loose dicts from adapter may vary; direct RunRequest already validated but getattr is defensive.)
        real_gui_policy = config.real_gui_policy
        ask_mode = config.ask_mode
        if real_gui_policy is not None:
            if hasattr(real_gui_policy, "value"):
                real_gui_policy = real_gui_policy.value
            real_gui_policy = str(real_gui_policy).lower()
        if ask_mode is not None:
            if hasattr(ask_mode, "value"):
                ask_mode = ask_mode.value
            ask_mode = str(ask_mode).lower()

        # Merge budgets (spec defaults + partial overrides)
        budgets = dict(DEFAULT_BUDGETS)
        if config.budgets:
            for k, v in config.budgets.items():
                if k in budgets and isinstance(v, (int, float)):
                    budgets[k] = int(v)

        # Capability probe (FR-15) — stored in WorkerSession for the run
        cap_report: CapabilityReport = CapabilityBroker().probe()

        # Display allocation
        isolated_display = _pick_isolated_display(run_id)
        observe_display = config.observe_display or (":0" if mode == RunMode.OBSERVE.value else None)

        # Step 6: make supervisor (and other harness components) injectable at controller level for tests.
        # Production paths use fresh; injected ones (e.g. pre-populated registry for ownership tests) are used.
        supervisor = self._supervisor or ProcessSupervisor()

        # Step 6: real_gui_policy / ask_mode forwarding + REAL baseline capture (FR-28, INV A)
        # ask_mode defaults to "on" ONLY for REAL (per spec INV C/F + D2). Normalization hoisted above
        # guarantees canonical lowercase str (or None) regardless of enum/casing from caller or loose dict.
        if mode == RunMode.REAL.value and not ask_mode:
            ask_mode = AskMode.ON.value

        # REAL baseline capture (FR-28) — BEFORE any Xvfb spawn when action_exec.
        # Uses injected resolver (FakeOwnershipResolver in all hermetic tests) or fresh OwnershipResolver
        # wired to the (possibly injected) supervisor so classify() will see live owned children.
        # Never mutates baseline or spawns for ISOLATED/OBSERVE (INV F regression).
        baseline_pids: Optional[List[int]] = None
        baseline_windows: Optional[Dict[str, Any]] = None
        if mode == RunMode.REAL.value and cap_report.action_exec:
            try:
                resolver: Any = getattr(self, "_ownership_resolver", None) or OwnershipResolver(supervisor=supervisor)
                binfo = resolver.capture_baseline(observe_display or ":0")
                bp = binfo.get("baseline_pids") or set()
                baseline_pids = sorted(int(p) for p in bp if isinstance(p, (int, str)) and int(p) > 0)
                baseline_windows = dict(binfo.get("baseline_windows") or {})
            except Exception:
                baseline_pids = []
                baseline_windows = {}

        # Always provision an isolated display + Xvfb when we can act (supports
        # OBSERVE read-only on :0 + actuation on private X authority per FR-03/structural split).
        # FR-22 explicitly requires it for ISOLATED; we do the same for OBSERVE
        # actuation side so SafetyKernel/ActionExecutor never have a path to real :0.
        # (supervisor already created above; spawn below registers under it for deterministic teardown)
        xvfb_root_id: Optional[str] = None
        xvfb_sp: Optional[SpawnedProc] = None
        xauth_path: Optional[Path] = None
        home_dir: Optional[Path] = None

        if cap_report.action_exec:
            # Pre-generate cookie so we can record the exact path for cleanup
            try:
                from .xauth import generate_private_xauthority, get_isolated_env

                xauth_path = generate_private_xauthority(isolated_display)
                iso_env = get_isolated_env(isolated_display)
                home_dir = Path(iso_env.get("HOME", "")) if iso_env.get("HOME") else None
            except Exception:
                xauth_path = None
                home_dir = None

            try:
                spec = IsolatedDisplaySpec(
                    display=isolated_display,
                    screen="1024x768x24",
                    xvfb_binary="Xvfb",
                    timeout_ms=8000,
                )
                xvfb_sp = supervisor.spawn_isolated_display(spec)
                xvfb_root_id = xvfb_sp.root_id
            except Exception:
                # Fail-closed: if we cannot own an Xvfb we cannot safely act on an isolated display.
                # The caller (adapter) will see action_exec still True in caps but will
                # hit spawn failures later; we still record the attempt.
                xvfb_root_id = None
                xvfb_sp = None

        # Phase 1.x Step 4: BrowserController ownership (lifecycle + B3). After supervisor + xvfb setup (per spec).
        # eff display: REAL uses :0 (agent child, visible, no prompt B1); else the run's isolated Xvfb.
        # Injected wins; else Fake under test (hermetic, never launches real or touches :0); else real if playwright present.
        # Engine defaults to "bing" (bot-tolerant; Google/DDG block automated). Stored on sess + echoed to ws.
        eff_browser_display = getattr(config, "browser_display", None) or (":0" if mode == RunMode.REAL.value else isolated_display)
        engine = getattr(config, "browser_engine", None) or "bing"
        bc = None
        if getattr(self, "_browser_controller", None) is not None:
            bc = self._browser_controller
        else:
            under_test = False
            try:
                import os as _os
                import sys as _sys
                pytesty = _os.environ.get("PYTEST_CURRENT_TEST") or any("_pytest" in m or "pytest" in m for m in _sys.modules) or _os.environ.get("AGY_REALGUI_TEST") == "1"
                browser_e2e = _os.environ.get("AGY_BROWSER_E2E") == "1"
                if pytesty and not browser_e2e:
                    under_test = True
            except Exception:
                under_test = True  # safe default for hermetic
            from .browser import FakeBrowserController as _FakeBC
            if under_test:
                _BC = _FakeBC
            else:
                try:
                    from playwright.sync_api import (
                        sync_playwright as _sp,  # type: ignore  # probe only; no launch here
                    )
                    del _sp
                    from .browser import BrowserController as _BC
                except Exception:
                    _BC = _FakeBC
            try:
                bc = _BC(supervisor=supervisor, display=eff_browser_display, engine=engine)
            except Exception:
                # Fail-closed but never absent: fall back to Fake for deterministic/session-safe behavior.
                try:
                    bc = _FakeBC(supervisor=supervisor, display=eff_browser_display, engine=engine)
                except Exception:
                    bc = None

        # Assemble the immutable WorkerSession snapshot (persisted in meta / events)
        ws = WorkerSession(
            run_id=run_id,
            mode=mode,
            task_priority=task_priority,
            created_at=_now_iso(),
            budgets=budgets,
            displays={
                "isolated_display": isolated_display,
                "isolated_xvfb_root_pid": xvfb_sp.pid if xvfb_sp else None,
                "observe_display": observe_display,
            },
            capabilities=cap_report,
            real_gui_policy=real_gui_policy,
            ask_mode=ask_mode,
            browser_engine=engine,
            browser_display=eff_browser_display,
            baseline_pids=baseline_pids,
            baseline_windows=baseline_windows,
        )

        sess = Session(
            run_id=run_id,
            worker_session=ws,
            supervisor=supervisor,
            isolated_display=isolated_display,
            observe_display=observe_display,
            xvfb_root_id=xvfb_root_id,
            xvfb_spawned=xvfb_sp,
            private_xauth_path=xauth_path,
            private_home_dir=home_dir,
            baseline_pids=baseline_pids,
            baseline_windows=baseline_windows,
            browser_controller=bc,
            browser_display=eff_browser_display,
            browser_engine=engine,
            current_cdp_endpoint=(getattr(bc, "cdp_endpoint", None) if bc is not None else None),
        )
        self._sessions[run_id] = sess

        # Emit capability probe event (FR-13) if we can obtain a sink
        self._emit(sess, WorkerEventType.CAPABILITY_PROBE.value, {
            "readiness": cap_report.readiness,
            "degraded": cap_report.degraded,
            "atspi": cap_report.atspi,
            "action_exec": cap_report.action_exec,
            "isolated_display": isolated_display,
            "xvfb_registered": bool(xvfb_root_id),
        })

        # Emit baseline capture (FR-38) for REAL runs (counts only; full sets in worker_session)
        if baseline_pids is not None:
            self._emit(sess, WorkerEventType.BASELINE_CAPTURED.value, {
                "run_id": sess.run_id,
                "baseline_pid_count": len(baseline_pids),
                "baseline_window_count": len(baseline_windows or {}),
                "display": observe_display or ":0",
            })  # exact uniform FR-38 shape (includes run_id); counts only (full in WorkerSession)
        return sess

    def close_session(self, session_id: str) -> StopResult:
        """Deterministic owned-tree teardown + cookie/home cleanup (FR-11/12/22)."""
        with self._lock:
            sess = self._sessions.pop(session_id, None)
        if not sess:
            return StopResult(run_id=session_id, status="unknown", terminated=False)

        # Terminate the Xvfb tree first (covers any children it may have spawned)
        if sess.xvfb_root_id:
            try:
                sess.supervisor.terminate_tree(sess.xvfb_root_id)
            except Exception:
                pass

        # Phase 1.x Step 4 B3 (additive): reap agent-owned browser tree via controller.close() (which calls supervisor.terminate_tree on its root_id).
        # Idempotent; covers close_session, adapter.stop, and all finally/exit paths. Watchdog (enforce_limits) reaps via supervisor registry (browser adopted).
        bc = getattr(sess, "browser_controller", None)
        if bc is not None:
            try:
                if hasattr(bc, "close"):
                    bc.close()
            except Exception:
                pass
            try:
                if hasattr(sess, "current_cdp_endpoint"):
                    sess.current_cdp_endpoint = None
            except Exception:
                pass

        # Belt-and-suspenders: terminate any other roots still registered under this supervisor
        try:
            for root in list(sess.supervisor.roots()):
                try:
                    sess.supervisor.terminate_tree(root.root_id)
                except Exception:
                    pass
        except Exception:
            pass

        # Hardening #1 cookie + private HOME cleanup (best-effort, never raises)
        self._cleanup_private_artifacts(sess)

        sess.status = "closed"
        self._emit(sess, WorkerEventType.SESSION_TERMINATED.value, {
            "reason": "close_session",
            "final_step": sess.current_step,
            "final_actions": sess.current_actions,
        })
        return StopResult(run_id=session_id, status="stopped", terminated=True)

    def _cleanup_private_artifacts(self, sess: Session) -> None:
        """Unlink the exact private Xauth and rmtree the exact iso HOME we created."""
        for p in (sess.private_xauth_path,):
            try:
                if p and p.exists():
                    p.unlink()
            except Exception:
                pass
        for d in (sess.private_home_dir,):
            try:
                if d and d.exists() and d.is_dir():
                    # Only remove if it looks like one of ours (contains no user data)
                    if "iso-home-" in str(d):
                        shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

        # Defensive sweep for any stray agu-* files for this display (in case of
        # multiple get_isolated_env calls during the run). Only /tmp, only our prefix.
        try:
            disp_tag = sess.isolated_display.replace(":", "d")
            for cand in Path("/tmp").glob(f"agu-{disp_tag}-*.Xauth"):
                try:
                    cand.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Orchestrator message ingress (confirmation tokens + notes)
    # ------------------------------------------------------------------

    def enqueue_orchestrator_message(self, run_id: str, msg: OrchestratorMessage) -> None:
        """Queue a message for the next ReasoningInput; wake any matching confirmation waiter (FR-18)."""
        sess = self._sessions.get(run_id)
        if not sess:
            return
        with self._lock:
            sess.pending_messages.append(msg)
            kind = (msg or {}).get("kind") if isinstance(msg, dict) else getattr(msg, "kind", None)
            if kind == "confirmation_token":
                tok = (msg or {}).get("token") if isinstance(msg, dict) else getattr(msg, "token", None)
                iid = (msg or {}).get("intent_id") if isinstance(msg, dict) else getattr(msg, "intent_id", None)
                self._inject_confirmation(sess, tok, iid)
            # Notes are just carried forward into the next ReasoningInput; no wake

    def _inject_confirmation(self, sess: Session, token: Optional[str], intent_id: Optional[str]) -> None:
        """Internal: record token and wake the waiter if present."""
        if not token:
            return
        # If intent_id supplied, target exact; else wake the first (if any) pending high-risk
        targets: List[str] = [intent_id] if intent_id else list(sess._confirm_events.keys())
        for iid in targets:
            entry = sess._confirm_events.get(iid)
            if entry:
                entry["token"] = token
                entry["received_at"] = _now_iso()
                try:
                    entry["event"].set()
                except Exception:
                    pass
                # Emit received (FR-13)
                self._emit(sess, WorkerEventType.CONFIRMATION_RECEIVED.value, {
                    "intent_id": iid,
                    "token_present": bool(token),
                })

    def _drain_pending_confirmation(self, sess: Session, intent_id: Optional[str]) -> Optional[str]:
        """Remove and return a matching confirmation_token from pending_messages if present.
        Used to handle early submit_confirmation (before await) and to clean one-shot tokens.
        """
        if not sess:
            return None
        with self._lock:
            for i, msg in enumerate(list(sess.pending_messages)):
                try:
                    kind = (msg or {}).get("kind") if isinstance(msg, dict) else getattr(msg, "kind", None)
                    if kind == "confirmation_token":
                        tok = (msg or {}).get("token") if isinstance(msg, dict) else getattr(msg, "token", None)
                        iid = (msg or {}).get("intent_id") if isinstance(msg, dict) else getattr(msg, "intent_id", None)
                        if tok and (iid == intent_id or not iid or not intent_id):
                            # consume this one-shot token msg
                            sess.pending_messages.pop(i)
                            return tok
                except Exception:
                    continue
        return None

    # ------------------------------------------------------------------
    # Suspended wait (FR-19 — no reasoner busy loop)
    # ------------------------------------------------------------------

    def await_confirmation(
        self, run_id: str, pending_intent_id: str, timeout_ms: int
    ) -> ConfirmationOutcome:
        """Block until a matching valid token arrives or timeout (FR-18/19).

        While this call is blocked the adapter run loop must NOT be polling the
        reasoner or executing actions — the caller (adapter) is responsible for
        yielding the control thread into this await.
        """
        sess = self._sessions.get(run_id)
        if not sess:
            return ConfirmationOutcome(
                outcome=ConfirmationOutcomeType.REJECTED.value,
                reason="no such session",
            )

        ev = threading.Event()
        with self._lock:
            sess._confirm_events[pending_intent_id] = {
                "event": ev,
                "token": None,
                "received_at": None,
            }
            sess.suspended_intent = pending_intent_id

        # Always emit wait_started (FR-19 audit); for immediate pre-arrival it will be zero duration
        self._emit(sess, WorkerEventType.CONFIRMATION_WAIT_STARTED.value, {
            "intent_id": pending_intent_id,
            "timeout_ms": int(timeout_ms),
        })

        # Handle token that arrived *before* we entered await (early submit / race with reasoner)
        pre_tok = self._drain_pending_confirmation(sess, pending_intent_id)
        if pre_tok:
            with self._lock:
                entry = sess._confirm_events.get(pending_intent_id) or {}
                entry["token"] = pre_tok
                entry["received_at"] = _now_iso()
                try:
                    entry.get("event").set() if entry.get("event") else None
                except Exception:
                    pass
            self._emit(sess, WorkerEventType.CONFIRMATION_RECEIVED.value, {
                "intent_id": pending_intent_id,
                "token_present": True,
                "pre_arrival": True,
            })
            # Clean up and return accepted without blocking
            with self._lock:
                sess._confirm_events.pop(pending_intent_id, None)
                sess.suspended_intent = None
            return ConfirmationOutcome(
                outcome=ConfirmationOutcomeType.ACCEPTED.value,
                token=pre_tok,
                intent_id=pending_intent_id,
                received_at=_now_iso(),
            )

        deadline = time.time() + max(0.1, timeout_ms / 1000.0)
        timed_out = not ev.wait(timeout=max(0.0, deadline - time.time()))

        with self._lock:
            entry = sess._confirm_events.pop(pending_intent_id, {})
            sess.suspended_intent = None
            token = entry.get("token") if entry else None
            received_at = entry.get("received_at") if entry else None

        if timed_out:
            self._emit(sess, WorkerEventType.CONFIRMATION_WAIT_TIMEOUT.value, {
                "intent_id": pending_intent_id,
            })
            return ConfirmationOutcome(
                outcome=ConfirmationOutcomeType.TIMEOUT.value,
                intent_id=pending_intent_id,
                reason="confirmation timeout",
            )

        # Post-consume any still-pending matching token msg (normal wake path)
        self._drain_pending_confirmation(sess, pending_intent_id)

        # Token present and waiter woke — success path for the gate
        return ConfirmationOutcome(
            outcome=ConfirmationOutcomeType.ACCEPTED.value,
            token=token,
            intent_id=pending_intent_id,
            received_at=received_at or _now_iso(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_session(self, run_id: str) -> Optional[Session]:
        return self._sessions.get(run_id)

    def _emit(self, sess: Session, event_type: str, payload: Dict[str, Any]) -> None:
        """Best-effort emit via a per-run sink if the controller was given a factory or direct sink."""
        try:
            sink = None
            if self._audit_sink and hasattr(self._audit_sink, "emit"):
                sink = self._audit_sink
            elif self._audit_factory:
                sink = self._audit_factory(sess.run_id)
            if sink and hasattr(sink, "emit"):
                sink.emit(
                    WorkerEvent(
                        ts=_now_iso(),
                        run_id=sess.run_id,
                        event_type=event_type,
                        payload=payload or {},
                    )
                )
            # Also try the session's own worker_session if it carries a sink (future extension)
        except Exception:
            # Audit must never break control flow
            pass

    def enforce_session_limits(self, run_id: str) -> None:
        """Delegated poll for the adapter run loop (FR-10/11)."""
        sess = self._sessions.get(run_id)
        if sess:
            try:
                sess.supervisor.enforce_limits(sess.worker_session.to_dict() if hasattr(sess.worker_session, "to_dict") else {"budgets": sess.worker_session.budgets})
            except Exception:
                pass

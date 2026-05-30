"""ComputerUseWorkerAdapter (Step 11) — standard AgentOrch worker + core perception-reason-safety-execute loop.

Implements the public contract:
  is_available() -> CapabilityReport
  start(run_request) -> RunHandle
  stop(run_id) -> StopResult
  submit_confirmation(run_id, token, intent_id=None) -> Ack

Wires every component from the approved architecture (PerceptionPipeline, ReasonerBridge,
SafetyKernel, ActionExecutor, ProcessSupervisor, AuditEventSink, SessionController).

The run loop (inside _execute_run) is the single place that:
- Builds multi-scope ReasoningInput for OBSERVE (FR-17)
- Enforces must_use_display_scope="isolated" + full SafetyKernel gate (FR-03/04/09/12/23/24)
- Uses suspended wait (no busy-poll) for high-risk confirmations (FR-18/19)
- Emits every WorkerEventType required by FR-13
- Respects budgets + calls enforce_limits (FR-10/11)
- Tears down owned trees (including Xvfb) on every exit path (FR-22)

All actuation is performed exclusively against the worker-owned isolated Xvfb display.
Zero code path ever targets real :0 for input or spawn env overrides.

Adapter-level tests (see tests/test_computer_use_adapter.py) drive this with fully
mocked reasoner/perception/safety and assert the event stream + gate + teardown behavior.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Union

from .action_executor import ActionExecutor
from .audit import AuditEventSink
from .capability import is_available as cap_is_available
from .grants import GrantCache
from .gui_prompt import FakePrompter, GuiPrompter
from .models import (
    ACTION_INTENT_JSON_V1,
    Ack,
    ActionResult,
    ActionStatus,
    ConfirmationOutcomeType,
    GateType,
    RunHandle,
    RunRequest,
    StopResult,
    WorkerEvent,
    WorkerEventType,
    from_dict,
)
from .ownership import OwnershipResolver
from .perception import PerceptionPipeline
from .process_supervisor import ProcessSupervisor
from .reasoner import ReasonerBridge, ReasonerError
from .safety import SafetyKernel
from .session import SessionController


def _running_under_test() -> bool:
    """Detect pytest/CI so adapter REAL path auto-selects FakePrompter (never spawns zenity/:0 in hermetic tests)."""
    try:
        import os
        import sys

        if os.environ.get("PYTEST_CURRENT_TEST"):
            return True
        if any("pytest" in m or "_pytest" in m for m in sys.modules):
            return True
        return os.environ.get("AGY_REALGUI_TEST") == "1"
    except Exception:
        return False


class ComputerUseWorkerAdapter:
    """Standard AgentOrch computer-use worker adapter (Step 11).

    Can be used directly by harness/roles.py once registered, or exercised via
    python -m harness with the appropriate worker name (future integration).

    Example:
        adapter = ComputerUseWorkerAdapter()
        h = adapter.start({"run_id": "cu-1", "objective": "open xclock on isolated display"})
    """

    # Small usage example (as required for Step 14 final pass):
    #   from agy_orchestrator.computer_use.adapter import ComputerUseWorkerAdapter
    #   adapter = ComputerUseWorkerAdapter()
    #   h = adapter.start(RunRequest(run_id=..., objective=...))  # then adapter.submit_confirmation / stop as needed

    def __init__(
        self,
        *,
        # Optional injected collaborators (primary hook for unit tests)
        session_controller: Optional[SessionController] = None,
        reasoner: Optional[ReasonerBridge] = None,
        perception: Optional[PerceptionPipeline] = None,
        safety: Optional[SafetyKernel] = None,
        executor: Optional[ActionExecutor] = None,
        supervisor: Optional[ProcessSupervisor] = None,
        audit_sink_factory: Optional[Callable[[str], Optional[AuditEventSink]]] = None,
        # Step 9 real-GUI DI (tests inject Fakes here; production REAL auto-constructs in start())
        ownership_resolver: Optional[Any] = None,
        gui_prompter: Optional[Any] = None,
        grant_cache: Optional[Any] = None,
        browser_controller: Optional[Any] = None,
        **kw: Any,
    ) -> None:
        self._reasoner = reasoner
        self._perception = perception
        self._safety = safety
        self._executor = executor
        self._supervisor = supervisor
        self._ownership_resolver = ownership_resolver
        self._gui_prompter = gui_prompter
        self._grant_cache = grant_cache
        self._browser_controller = browser_controller
        self._audit_factory = audit_sink_factory or (lambda rid: AuditEventSink(run_id=rid))
        self._active_runs: Dict[str, Dict[str, Any]] = {}  # run_id -> {"thread": , "stop": Event, ...}
        self._lock = threading.RLock()
        self._user_supplied_controller = bool(session_controller)

        ctrl_kwargs: Dict[str, Any] = {"audit_sink_factory": audit_sink_factory}
        if ownership_resolver is not None:
            ctrl_kwargs["ownership_resolver"] = ownership_resolver
        if gui_prompter is not None:
            ctrl_kwargs["gui_prompter"] = gui_prompter
        if grant_cache is not None:
            ctrl_kwargs["grant_cache"] = grant_cache
        if browser_controller is not None:
            ctrl_kwargs["browser_controller"] = browser_controller
        self._controller = session_controller or SessionController(**ctrl_kwargs)

    # ------------------------------------------------------------------
    # Public AgentOrch worker surface (exact per spec §5)
    # ------------------------------------------------------------------

    def is_available(self) -> Any:
        """Return the full graded CapabilityReport (FR-05/15).

        Harness and operator tooling see the matrix (atspi/ocr/geometry/dom/action_exec/degraded/readiness).
        """
        return cap_is_available()

    def start(self, run_request: Union[RunRequest, Dict[str, Any], str]) -> RunHandle:
        """Create session, wire components, execute the perception-reason-act loop to completion or budget.

        Returns a RunHandle immediately describing the run; the loop runs to termination
        inside this call (synchronous control API). For long-running use the caller
        should invoke in a background thread or use the harness dispatch machinery.
        """
        if isinstance(run_request, str):
            run_request = {"run_id": run_request, "objective": ""}

        if isinstance(run_request, dict):
            req = from_dict(RunRequest, run_request) if "run_id" in run_request else RunRequest(**run_request)
        else:
            req = run_request

        run_id = req.run_id or f"cu-{int(time.time()*1000)}"
        objective = getattr(req, "objective", "") or (run_request.get("objective", "") if isinstance(run_request, dict) else "")

        # Step 9: wire REAL-mode harness components (ownership + grant cache + prompter) to
        # SafetyKernel (the gate) and SessionController (for baseline capture). Only for mode=REAL;
        # ISOLATED/OBSERVE paths unchanged (INVARIANT F). Prompter wrapper (here) routes free-text
        # results into orchestrator_messages as OperatorNoteMessageDict (FR-37). Perception remains
        # isolated (+ observe_real only for OBSERVE mode). Constraints for REAL advertise the contract.
        is_real_mode = False
        m = getattr(req, "mode", None) or (req.get("mode") if isinstance(req, dict) else None)
        if m is not None:
            if hasattr(m, "value"):
                m = m.value
            is_real_mode = str(m).upper() == "REAL"

        # Prefer ctor-injected (tests supply Fakes); construct only for REAL when absent.
        # Prompter choice: Fake under test (hermetic, no zenity/:0 ever), GuiPrompter in prod.
        ownership_resolver = self._ownership_resolver
        grant_cache = self._grant_cache
        prompter = self._gui_prompter
        if is_real_mode:
            if ownership_resolver is None:
                try:
                    ownership_resolver = OwnershipResolver()
                except Exception:
                    ownership_resolver = None
            if grant_cache is None:
                try:
                    grant_cache = GrantCache(run_id=run_id)
                except Exception:
                    grant_cache = None
            if prompter is None:
                if _running_under_test():
                    prompter = FakePrompter()
                else:
                    try:
                        prompter = GuiPrompter()
                    except Exception:
                        prompter = None

            # Route prompter free-text -> next orchestrator_messages as operator_note (FR-37).
            # Wrap exactly once (supports both injected and constructed; Fake and Gui).
            if prompter is not None and not getattr(prompter, "_note_routing_wrapped", False):
                orig = prompter.prompt

                def _note_routing_prompt(ctx: Any) -> Any:
                    res = orig(ctx)
                    try:
                        txt = ""
                        if isinstance(res, dict):
                            txt = (res or {}).get("operator_text") or ""
                        else:
                            txt = getattr(res, "operator_text", "") or ""
                        if txt and str(txt).strip():
                            note: Dict[str, Any] = {
                                "kind": "operator_note",
                                "text": str(txt).strip(),
                                "issued_at": datetime.now(timezone.utc).isoformat(),
                            }
                            self._controller.enqueue_orchestrator_message(run_id, note)
                    except Exception:
                        pass
                    return res

                prompter.prompt = _note_routing_prompt  # type: ignore[attr-defined]
                try:
                    setattr(prompter, "_note_routing_wrapped", True)
                except Exception:
                    pass

            # Thread to SessionController for REAL baseline capture (FR-28) inside create_session.
            # Save/restore keeps the (usually long-lived) controller instance clean for mixed runs.
            if not self._user_supplied_controller:
                old_own = getattr(self._controller, "_ownership_resolver", None)
                old_p = getattr(self._controller, "_gui_prompter", None)
                old_g = getattr(self._controller, "_grant_cache", None)
                try:
                    if ownership_resolver is not None:
                        setattr(self._controller, "_ownership_resolver", ownership_resolver)
                    if prompter is not None:
                        setattr(self._controller, "_gui_prompter", prompter)
                    if grant_cache is not None:
                        setattr(self._controller, "_grant_cache", grant_cache)
                    # Create the authoritative session (this also spawns Xvfb when action_exec possible)
                    sess = self._controller.create_session(req)
                finally:
                    # Restore prior controller state (no cross-run pollution of REAL-only DI objects)
                    try:
                        setattr(self._controller, "_ownership_resolver", old_own)
                        setattr(self._controller, "_gui_prompter", old_p)
                        setattr(self._controller, "_grant_cache", old_g)
                    except Exception:
                        pass
            else:
                # user-supplied controller (advanced tests); assume it already has what it needs
                sess = self._controller.create_session(req)
        else:
            # Create the authoritative session (this also spawns Xvfb when action_exec possible)
            sess = self._controller.create_session(req)

        # Per-run audit sink (FR-13) — everything from here on goes through it
        sink = self._audit_factory(run_id) or AuditEventSink(run_id=run_id)
        # Make sure the sink directory exists (harness contract)
        try:
            sink.events_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # FR-38 wiring: attach sink to prompter (for prompt_shown emit from gui_prompt site) and pass to SafetyKernel
        if prompter is not None:
            try:
                setattr(prompter, "_audit_sink", sink)
            except Exception:
                pass

        # Wire (or build) all components for this run, passing the sink for event emission
        sup = self._supervisor or sess.supervisor
        perception = self._perception or PerceptionPipeline(
            audit_sink=sink,
            redaction_enabled=None,  # defaults ON for OBSERVE hardening #4
            default_isolated_display=sess.isolated_display,
            default_observe_display=sess.observe_display or ":0",
        )
        reasoner = self._reasoner or ReasonerBridge(
            run_id=run_id,
            audit_sink=sink,
            auto_build_agents=False,  # tests and real harness always supply or let outer layer decide
        )

        # Step 9: SafetyKernel receives the (REAL-wired or None) components; non-REAL paths receive None
        # for the harness objects (preserves every byte of ISOLATED/OBSERVE behavior + all prior tests).
        safety = self._safety or SafetyKernel(
            supervisor=sup,
            ownership_resolver=ownership_resolver if is_real_mode else None,
            gui_prompter=prompter if is_real_mode else None,
            grant_cache=grant_cache if is_real_mode else None,
            audit_sink=sink,
        )
        executor = self._executor or ActionExecutor(
            isolated_display=sess.isolated_display,
            supervisor=sup,
            browser_controller=getattr(sess, "browser_controller", None) or self._browser_controller,
            action_timeout_ms=sess.worker_session.budgets.get("action_timeout_ms", 10000),
        )

        # Record for stop() / introspection
        with self._lock:
            self._active_runs[run_id] = {
                "session": sess,
                "sink": sink,
                "stop_event": threading.Event(),
                "started_at": time.time(),
            }

        # Execute the core loop (blocking until budgets, completion, or external stop)
        try:
            self._execute_run(
                run_id=run_id,
                objective=objective,
                sess=sess,
                sink=sink,
                perception=perception,
                reasoner=reasoner,
                safety=safety,
                executor=executor,
                stop_event=self._active_runs[run_id]["stop_event"],
            )
            final_status = "completed"
        except Exception:
            # Never leak exceptions that would destabilize the caller; always close owned trees.
            # The close_session path (finally) is the single source of the session.terminated event.
            try:
                # Mark for the close path to include error context if desired (best-effort)
                if hasattr(sess, "status"):
                    sess.status = "errored"
            except Exception:
                pass
            final_status = "errored"
        finally:
            # Always tear down (idempotent)
            try:
                self._controller.close_session(run_id)
            except Exception:
                pass
            with self._lock:
                self._active_runs.pop(run_id, None)

        events_path = str(sink.events_path) if hasattr(sink, "events_path") else None
        return RunHandle(
            run_id=run_id,
            status=final_status,
            session_id=run_id,
            events_path=events_path,
        )

    def stop(self, run_id: str) -> StopResult:
        """Request cooperative stop of an in-flight run and perform teardown."""
        with self._lock:
            rec = self._active_runs.get(run_id)
        if rec:
            try:
                rec["stop_event"].set()
            except Exception:
                pass
        # Force-close the session resources regardless
        res = self._controller.close_session(run_id)
        with self._lock:
            self._active_runs.pop(run_id, None)
        return res

    def submit_confirmation(self, run_id: str, token: str, intent_id: Optional[str] = None) -> Ack:
        """External (orchestrator) ingress for high-risk action confirmations (FR-18).

        The token is injected into the pending wait state; if a waiter is blocked in
        await_confirmation it will be woken. The next validate() call will see the token.
        """
        if not token or not isinstance(token, str):
            return Ack(ok=False, run_id=run_id, message="invalid token")
        try:
            msg: Dict[str, Any] = {
                "kind": "confirmation_token",
                "token": token,
                "intent_id": intent_id,
                "issued_at": datetime.now(timezone.utc).isoformat(),
            }
            self._controller.enqueue_orchestrator_message(run_id, msg)
            return Ack(ok=True, run_id=run_id, intent_id=intent_id, message="confirmation enqueued")
        except Exception as e:
            return Ack(ok=False, run_id=run_id, message=str(e)[:120])

    # ------------------------------------------------------------------
    # Core loop (the heart of Step 11)
    # ------------------------------------------------------------------

    def _execute_run(
        self,
        *,
        run_id: str,
        objective: str,
        sess: Any,
        sink: AuditEventSink,
        perception: PerceptionPipeline,
        reasoner: ReasonerBridge,
        safety: SafetyKernel,
        executor: ActionExecutor,
        stop_event: threading.Event,
    ) -> None:
        """Single-threaded perception → reason → safety → execute loop.

        All hard guardrails (display_scope, ownership, risk, budgets, spawn policy)
        are enforced on every iteration. The loop yields control into
        await_confirmation for gated actions (no busy reasoner polling).
        """
        budgets = dict(sess.worker_session.budgets)
        max_steps = int(budgets.get("max_steps", 200))
        max_actions = int(budgets.get("max_actions", 200))
        reasoning_to = int(budgets.get("reasoning_timeout_ms", 45000))
        confirm_to = int(budgets.get("confirmation_wait_timeout_ms", 300000))

        mode = sess.worker_session.mode
        task_priority = sess.worker_session.task_priority

        # Initial capability snapshot (already emitted by controller.create_session)
        # Budget-aware loop: check limits *before* each perception/reason/execute cycle (FR-10)
        for step in range(1, max_steps + 1):
            if stop_event.is_set():
                break
            if sess.current_actions >= max_actions:
                break
            sess.current_step = step

            # Resource backstop (FR-10/11) — psutil + the rlimits already installed at spawn time
            self._controller.enforce_session_limits(run_id)

            # ------------------------------------------------------------------
            # Perception (FR-17 multi-scope for OBSERVE)
            # ------------------------------------------------------------------
            scopes = ["isolated"]
            if mode == "OBSERVE":
                scopes.append("observe_real")

            displays_map = {
                "isolated": sess.isolated_display,
                "observe_real": sess.observe_display or ":0",
            }
            cdp = (
                getattr(sess, "browser_cdp_endpoint", None)
                or getattr(sess, "current_cdp_endpoint", None)
                or getattr(getattr(sess, "browser_controller", None), "cdp_endpoint", None)
                or getattr(self._browser_controller, "cdp_endpoint", None)
            )

            # Phase 1.x: hand perception the owned browser controller so DOM scanning
            # can be marshaled onto the Playwright owner thread when needed.
            dom_page = None
            _bc = getattr(sess, "browser_controller", None) or self._browser_controller
            if _bc is not None:
                dom_page = _bc

            try:
                raw_snaps = perception.snapshot_set(
                    scopes,
                    run_id=run_id,
                    displays=displays_map,
                    cdp_endpoint=cdp,
                    dom_page=dom_page,
                )
            except Exception as e:
                # Degrade gracefully; emit and continue with whatever we have (or empty)
                sink.emit(WorkerEvent(
                    ts=datetime.now(timezone.utc).isoformat(),
                    run_id=run_id,
                    event_type=WorkerEventType.SAFETY_VIOLATION.value,
                    payload={"reason": "perception_failure", "error": str(e)[:120]},
                ))
                raw_snaps = {}

            summaries: Dict[str, Any] = {}
            for sc, snap in raw_snaps.items():
                try:
                    # Dict-shaped so the in-process reasoner sees the SAME contract a
                    # subprocess reasoner gets after JSON serialization (snapshots[scope]
                    # is a JSON object with an "elements" list, never a dataclass).
                    summaries[sc] = asdict(perception.make_summary(snap))
                except Exception:
                    summaries[sc] = {"snapshot_id": getattr(snap, "snapshot_id", "s"), "scope": sc}

            # ------------------------------------------------------------------
            # Build ReasoningInput (constraints advertise real_act + policy/ask_mode for REAL per Step 9;
            # reasoner sees the contract; operator_notes from pending_messages already included)
            # ------------------------------------------------------------------
            constraints = {
                "must_use_display_scope": "real_act" if mode == "REAL" else "isolated",
                "max_actions_remaining": max(0, max_actions - sess.current_actions),
                "max_steps_remaining": max(0, max_steps - sess.current_step),
                "disallowed_ops": [],
            }
            if mode == "REAL":
                # Surface policy + ask mode so reasoner knows the exact harness contract (no prompt for owned, etc.)
                constraints["real_gui_policy"] = getattr(sess.worker_session, "real_gui_policy", None) or "full"
                constraints["ask_mode"] = getattr(sess.worker_session, "ask_mode", None) or "on"
            ri = {
                "run_id": run_id,
                "session_mode": mode,
                "task_priority": task_priority,
                "objective": objective,
                "constraints": constraints,
                "snapshots": summaries,
                "orchestrator_messages": list(sess.pending_messages),
                "output_contract": ACTION_INTENT_JSON_V1,
            }

            # Emit operator_note.received (exact FR-38 shape) for free-text from GUI prompt (FR-37/38, Step 12)
            for m in list(sess.pending_messages or []):
                if isinstance(m, dict) and m.get("kind") == "operator_note":
                    sink.emit(WorkerEvent(
                        ts=datetime.now(timezone.utc).isoformat(),
                        run_id=run_id,
                        event_type=WorkerEventType.OPERATOR_NOTE_RECEIVED.value,
                        payload={"run_id": run_id, "text": str(m.get("text", ""))[:200], "source": "gui_prompt"},
                    ))

            # ------------------------------------------------------------------
            # Reason (FR-14/16/20/21/25 routing + safety already inside ReasonerBridge)
            # ------------------------------------------------------------------
            try:
                intent = reasoner.decide(ri, timeout_ms=reasoning_to)
            except ReasonerError:
                # Reasoner already emitted parse/timeout/auth events and killed its subprocess
                break
            except Exception as e:
                sink.emit(WorkerEvent(
                    ts=datetime.now(timezone.utc).isoformat(),
                    run_id=run_id,
                    event_type=WorkerEventType.REASONER_PARSE_ERROR.value,
                    payload={"error": str(e)[:160]},
                ))
                break

            # ------------------------------------------------------------------
            # Safety / confirmation gate (FR-09/18/19)
            # ------------------------------------------------------------------
            try:
                gate = safety.require_confirmation(intent)
            except Exception:
                gate = type("G", (), {"gate": GateType.ALLOW.value, "reason": "gate-error"})()

            if getattr(gate, "gate", None) == GateType.REQUIRE_CONFIRMATION.value or (
                isinstance(gate, dict) and gate.get("gate") == GateType.REQUIRE_CONFIRMATION.value
            ):
                # Dry-run preview + enter suspended wait (FR-09/19)
                sink.emit(WorkerEvent(
                    ts=datetime.now(timezone.utc).isoformat(),
                    run_id=run_id,
                    event_type=WorkerEventType.ACTION_DRY_RUN.value,
                    payload={
                        "intent_id": getattr(intent, "intent_id", "i"),
                        "risk_level": getattr(intent, "risk_level", "high"),
                        "rationale": getattr(intent, "rationale", "")[:200],
                    },
                ))
                iid = getattr(intent, "intent_id", "pending-intent")
                outcome = self._controller.await_confirmation(run_id, iid, confirm_to)

                if outcome.outcome != ConfirmationOutcomeType.ACCEPTED.value or not getattr(outcome, "token", None):
                    sink.emit(WorkerEvent(
                        ts=datetime.now(timezone.utc).isoformat(),
                        run_id=run_id,
                        event_type=WorkerEventType.ACTION_REJECTED.value,
                        payload={"intent_id": iid, "reason": "confirmation_failed_or_timeout"},
                    ))
                    continue

                # Valid token received — attach for the validate() call
                try:
                    intent.confirmation_token = outcome.token  # type: ignore[attr-defined]
                except Exception:
                    pass

            # ------------------------------------------------------------------
            # Full SafetyKernel validation + target resolution (FR-03/04/07/12/23/24)
            # ------------------------------------------------------------------
            try:
                val = safety.validate(
                    intent,
                    sess.worker_session,
                    current_step=sess.current_step,
                    current_actions=sess.current_actions,
                    snapshots=raw_snaps,
                )
            except Exception as ve:
                sink.emit(WorkerEvent(
                    ts=datetime.now(timezone.utc).isoformat(),
                    run_id=run_id,
                    event_type=WorkerEventType.SAFETY_VIOLATION.value,
                    payload={"code": "validate_exception", "message": str(ve)[:160]},
                ))
                continue

            if not getattr(val, "valid", False):
                for v in (getattr(val, "violations", None) or []):
                    sink.emit(WorkerEvent(
                        ts=datetime.now(timezone.utc).isoformat(),
                        run_id=run_id,
                        event_type=WorkerEventType.SAFETY_VIOLATION.value,
                        payload=v if isinstance(v, dict) else {"message": str(v)},
                    ))
                sink.emit(WorkerEvent(
                    ts=datetime.now(timezone.utc).isoformat(),
                    run_id=run_id,
                    event_type=WorkerEventType.ACTION_REJECTED.value,
                    payload={"intent_id": getattr(intent, "intent_id", None)},
                ))
                continue

            action_spec = getattr(val, "normalized_action", None)
            if action_spec is None:
                continue

            # ------------------------------------------------------------------
            # Execute (guaranteed isolated scope + resolved coordinate target for spatial)
            # ------------------------------------------------------------------
            try:
                result: ActionResult = executor.execute(action_spec)
            except Exception as ex:
                result = ActionResult(
                    status=ActionStatus.FAILED.value,
                    executed_at=datetime.now(timezone.utc).isoformat(),
                    error_code="executor_exception",
                )
                sink.emit(WorkerEvent(
                    ts=datetime.now(timezone.utc).isoformat(),
                    run_id=run_id,
                    event_type=WorkerEventType.ACTION_REJECTED.value,
                    payload={"error": str(ex)[:120]},
                ))

            # Audit the outcome (FR-08/13)
            sink.emit(WorkerEvent(
                ts=datetime.now(timezone.utc).isoformat(),
                run_id=run_id,
                event_type=WorkerEventType.ACTION_EXECUTED.value if result.status == ActionStatus.OK.value else WorkerEventType.ACTION_REJECTED.value,
                payload={
                    "action_id": getattr(action_spec, "action_id", None),
                    "type": getattr(action_spec, "type", None),
                    "risk_level": getattr(action_spec, "risk_level", None),
                    "status": result.status,
                    "resolved_target": getattr(result, "resolved_target", None),
                    "spawned": getattr(result, "spawned_process_ids", None),
                },
            ))

            sess.current_actions += 1

            # Simple objective-completion heuristic for unit tests / minimal runs
            rat = (getattr(action_spec, "rationale", "") or "").lower()
            if "done" in rat or "complete" in rat or "objective satisfied" in rat:
                break

            # Small cooperative yield so stop_event is responsive
            if stop_event.wait(timeout=0.005):
                break

        # Loop exit — final termination event is emitted by close_session via controller
        # (we already emit SESSION_TERMINATED on the explicit close path)

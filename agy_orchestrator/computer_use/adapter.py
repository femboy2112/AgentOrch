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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

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
from .audit import AuditEventSink
from .capability import is_available as cap_is_available
from .session import SessionController, DEFAULT_BUDGETS

# Component imports (all previous steps)
from .perception import PerceptionPipeline
from .reasoner import ReasonerBridge, ReasonerError
from .safety import SafetyKernel
from .action_executor import ActionExecutor
from .process_supervisor import ProcessSupervisor


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
        **kw: Any,
    ) -> None:
        self._controller = session_controller or SessionController(audit_sink_factory=audit_sink_factory)
        self._reasoner = reasoner
        self._perception = perception
        self._safety = safety
        self._executor = executor
        self._supervisor = supervisor
        self._audit_factory = audit_sink_factory or (lambda rid: AuditEventSink(run_id=rid))
        self._active_runs: Dict[str, Dict[str, Any]] = {}  # run_id -> {"thread": , "stop": Event, ...}
        self._lock = threading.RLock()

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

        # Create the authoritative session (this also spawns Xvfb when action_exec possible)
        sess = self._controller.create_session(req)

        # Per-run audit sink (FR-13) — everything from here on goes through it
        sink = self._audit_factory(run_id) or AuditEventSink(run_id=run_id)
        # Make sure the sink directory exists (harness contract)
        try:
            sink.events_path.parent.mkdir(parents=True, exist_ok=True)
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
        safety = self._safety or SafetyKernel(supervisor=sup)
        executor = self._executor or ActionExecutor(
            isolated_display=sess.isolated_display,
            supervisor=sup,
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
        except Exception as exc:
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

            try:
                raw_snaps = perception.snapshot_set(
                    scopes,
                    run_id=run_id,
                    displays=displays_map,
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
                    summaries[sc] = perception.make_summary(snap)
                except Exception:
                    summaries[sc] = {"snapshot_id": getattr(snap, "snapshot_id", "s"), "scope": sc}

            # ------------------------------------------------------------------
            # Build ReasoningInput (constraints enforce isolated actuation only)
            # ------------------------------------------------------------------
            ri = {
                "run_id": run_id,
                "session_mode": mode,
                "task_priority": task_priority,
                "objective": objective,
                "constraints": {
                    "must_use_display_scope": "isolated",
                    "max_actions_remaining": max(0, max_actions - sess.current_actions),
                    "max_steps_remaining": max(0, max_steps - sess.current_step),
                    "disallowed_ops": [],
                },
                "snapshots": summaries,
                "orchestrator_messages": list(sess.pending_messages),
                "output_contract": ACTION_INTENT_JSON_V1,
            }

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

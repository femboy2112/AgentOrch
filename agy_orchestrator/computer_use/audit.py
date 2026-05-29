"""AuditEventSink for computer-use worker (Step 6).

Safe, append-only, crash-resistant JSONL writer for WorkerEvent records.

Location: runs/<run_id>/events.jsonl (created on demand).

Contract:
- emit(WorkerEvent) serializes via the model's deterministic to_dict + sort_keys JSON.
- One line per event, UTF-8, trailing \n.
- Atomic append: full line is first staged + fsync'd to a temp sibling in the same dir,
  *then* appended to the target under "ab" + fsync. Any error during the target-append
  phase leaves the target file containing *only* previously complete lines (no partial
  or truncated JSON from the failing event).
- Registered callbacks (for harness EVENT_BUS fanout) are invoked only after the line
  has been durably appended. Callback exceptions are swallowed (matching dashboard bus).
- This is the *sole* audit trail (FR-13). All future perception.snapshot,
  reasoner.*, action.*, confirmation.*, safety.*, resource.*, and session.*
  transitions MUST emit through an instance of this sink.

Supports every value in WorkerEventType exactly as defined in models.py (including the real-gui permission.* and baseline.* events per FR-38).
FR-38: baseline.captured, permission.prompt_shown/granted/denied,
foreign_interaction_blocked, and operator_note.received are emitted at
decision points (safety gate, prompter, session baseline, adapter note routing)
with exact uniform payload shapes:
  baseline.captured: {"run_id": r, "baseline_pid_count": n, "baseline_window_count": m, "display": ":0"}
  permission.prompt_shown: {"run_id": r, "pid": p, "action_type": t, "policy": pol, "ask_mode": a}
  permission.granted: {"run_id": r, "pid": p, "grant_scope": s, "operator_text"?: txt, "implicit"?: bool}
  permission.denied: {"run_id": r, "pid"?: p, "reason": why}
  foreign_interaction_blocked: {"run_id": r, "pid"?: p, "policy"?: pol, "reason": why}
  operator_note.received: {"run_id": r, "pid"?: p, "text": txt, "source"?: "gui_prompt"}

No X11, no GUI, no real-:0 side effects. Pure stdlib + models.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .models import WorkerEvent, WorkerEventType

logger = logging.getLogger(__name__)


class AuditEventSink:
    """Crash-safe append-only sink for the computer-use audit trail (FR-13)."""

    def __init__(
        self,
        run_id: str,
        *,
        events_path: Optional[Path] = None,
        runs_root: Optional[Path] = None,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        self.run_id = run_id

        if events_path is not None:
            self.events_path: Path = Path(events_path).resolve()
        else:
            root = Path(runs_root).resolve() if runs_root is not None else Path("runs")
            self.events_path = (root / run_id / "events.jsonl").resolve()

        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.events_path.exists():
            # Match harness touch() contract (creates the file early for watchers)
            self.events_path.touch(mode=0o644)

        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []

    def add_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback (e.g. harness EVENT_BUS sink).

        Callbacks receive the *plain dict* form of the WorkerEvent (post to_dict).
        They are invoked after the line is safely on disk.
        """
        if not callable(cb):
            raise TypeError("callback must be callable")
        self._callbacks.append(cb)

    # Back-compat alias (Step-6 test + harness EVENT_BUS use register_callback)
    register_callback = add_callback

    def emit(self, event: WorkerEvent) -> None:
        """Append one event line (canonical shape) then fan out to callbacks.

        The disk append is the source of truth. Callbacks are best-effort.
        On I/O error the file is guaranteed to contain only complete prior lines
        (see _safe_append_line); the exception is propagated so callers can
        decide fail-closed policy per the reliability NFR.
        """
        if not isinstance(event, WorkerEvent):
            raise TypeError("emit expects a WorkerEvent instance")

        # Re-validate even for pathological construction (supports test that bypasses dataclass __post_init__)
        if event.event_type not in {e.value for e in WorkerEventType}:
            raise ValueError(f"invalid WorkerEventType: {event.event_type}")

        if event.run_id != self.run_id:
            logger.warning(
                "AuditEventSink: emitting event for run %s into sink for %s",
                event.run_id, self.run_id
            )

        try:
            d: Dict[str, Any] = event.to_dict()
            # Deterministic, compact, stable key order — identical contract used
            # for reasoner envelopes and all prior model tests.
            line = json.dumps(d, sort_keys=True, ensure_ascii=False) + "\n"
            self._safe_append_line(line)

            # Fanout only after durable write. Swallow per-bus convention.
            for cb in list(self._callbacks):
                try:
                    cb(d)
                except Exception as cb_exc:  # pragma: no cover (defensive)
                    logger.debug("AuditEventSink callback error (ignored): %s", cb_exc)

        except Exception as exc:
            logger.warning(
                "AuditEventSink: failed to append event %s for run %s: %s",
                getattr(event, "event_type", "?"), self.run_id, exc
            )
            raise

    def _safe_append_line(self, line: str) -> None:
        """Stage full line to temp (fsync), then append complete bytes to target.

        Guarantees: if any exception occurs during target mutation (including after
        open() succeeds but mid-write), the target is truncated back to its exact
        pre-append size before the exception propagates. Therefore the on-disk file
        always contains *only* complete, previously committed lines — never a
        truncated/partial JSON record from the failing event.

        Temp staging file is always cleaned in finally (even on error).
        """
        target = self.events_path
        target.parent.mkdir(parents=True, exist_ok=True)

        # Stage the *entire* line to a unique sibling temp + fsync (durable before touching target)
        fd, tmp_str = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=f".cu-audit-{self.run_id}-",
            suffix=".jsonl.tmp",
        )
        tmp = Path(tmp_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                tf.write(line)
                tf.flush()
                os.fsync(tf.fileno())

            # Complete staged payload (tmp closed by with; safe to read)
            data = tmp.read_bytes()

            # Snapshot size for rollback on any error during append phase
            pre_size = target.stat().st_size if target.exists() else 0

            try:
                with open(target, "ab") as af:
                    af.write(data)
                    af.flush()
                    os.fsync(af.fileno())
            except Exception:
                # Roll back any partial bytes written before the failure (ENOSPC, etc.)
                try:
                    with open(target, "ab") as af:
                        os.ftruncate(af.fileno(), pre_size)
                        os.fsync(af.fileno())
                except Exception:
                    # Best-effort; reader (read_all) already tolerates a trailing bad line
                    pass
                raise
        finally:
            # Always remove staging temp (contains either the full line or nothing useful)
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Test / introspection helpers (not part of public worker contract)
    # ------------------------------------------------------------------

    def read_all(self) -> List[Dict[str, Any]]:
        """Return all events currently persisted (parsed). Used by tests.

        Tolerates a trailing partial line (simulated crash mid-write) so that
        tests can assert "all prior *complete* events survived".
        """
        if not self.events_path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for ln in self.events_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                # Trailing partial line from a simulated mid-event failure — ignore for read_all
                pass
        return out

    def __repr__(self) -> str:  # pragma: no cover
        return f"AuditEventSink(run_id={self.run_id!r}, path={self.events_path})"

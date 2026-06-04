from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import AsyncIterator, Callable, Dict, List, Optional


def _env_int(name: str, default: int, *, minimum: int) -> int:
    """Read a positive int env knob. 0/unset/garbage -> default; clamp to minimum.

    Follows the repo's AGY_* convention: a healthy default with an env override.
    """
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        val = int(raw)
    except Exception:
        return default
    if val <= 0:
        return default
    return max(val, minimum)


# Per-run in-memory replay buffer cap (drop-oldest ring). A long/chatty master
# run can emit tens of thousands of token events; without a cap self.queues
# grows O(total events) and OOMs the broker. Default keeps enough history for
# late SSE replay on any realistic single run while staying bounded. Older rows
# are DROPPED from the head — subscribe() keys off _event_id so a resuming
# subscriber still advances correctly past dropped ids.
_REPLAY_MAX_DEFAULT = 10_000

# Number of finished+closed runs whose state is retained after close(). A
# long-lived broker (broker.py runs dispatch_async per job) services many runs;
# without eviction every run's full history + closed-set entry leaks forever.
# We keep a small recent window so late SSE replay of the just-finished run
# still works, then evict the oldest.
_RETAIN_DEFAULT = 8

_WORKER_EVENT_KINDS = {
    "lifecycle",
    "reasoning",
    "message",
    "tool_call",
    "tool_result",
    "usage",
    "stderr",
    "watchdog",
    "heartbeat",
}


def _to_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_branch(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except Exception:
        return None


def _normalize_worker_event(
    payload: dict,
    *,
    run_id: str,
    worker: str,
    model: Optional[str],
    effort: Optional[str],
    branch: Optional[int],
) -> dict:
    kind = str(payload.get("kind") or "stderr")
    kind = kind if kind in _WORKER_EVENT_KINDS else "stderr"
    data = payload.get("data", {})
    if not isinstance(data, dict):
        data = {}

    resolved_branch = _to_branch(payload.get("branch"))
    if resolved_branch is None:
        resolved_branch = _to_branch(branch)

    return {
        "ts": _to_float(payload.get("ts"), time.time()),
        "run_id": run_id,
        "worker": str(payload.get("worker") or worker),
        "model": str(payload.get("model") or model or "n/a"),
        "effort": str(payload.get("effort") or effort or "n/a"),
        "branch": resolved_branch,
        "kind": kind,
        "text": str(payload.get("text", "")),
        "data": data,
    }


class EventBus:
    """Per-run-id queues with sink fanout and replay support."""

    def __init__(self) -> None:
        self.queues: Dict[str, List[dict]] = {}
        self.sinks: Dict[str, List[Callable[[dict], None]]] = {}
        self.closed: set[str] = set()
        self._wake: Dict[str, asyncio.Event] = {}
        self._next_ids: Dict[str, int] = {}
        # FIFO of closed run_ids retained for late replay; oldest evicted first.
        self._closed_order: List[str] = []
        # Per-run replay ring cap (drop-oldest) + retained-closed-run window.
        self._replay_max = _env_int(
            "AGY_EVENTBUS_REPLAY_MAX", _REPLAY_MAX_DEFAULT, minimum=1
        )
        self._retain = _env_int("AGY_EVENTBUS_RETAIN", _RETAIN_DEFAULT, minimum=1)

    def reset(self) -> None:
        """Clear all per-run state.

        The dashboard server owns lifecycle and may reset the shared bus at
        startup to avoid leaking run state across app instances/tests.
        """
        self.queues.clear()
        self.sinks.clear()
        self.closed.clear()
        self._wake.clear()
        self._next_ids.clear()
        self._closed_order.clear()

    def _ensure(self, run_id: str) -> None:
        if run_id not in self.queues:
            self.queues[run_id] = []
            self.sinks[run_id] = []
            self._wake[run_id] = asyncio.Event()
            self._next_ids[run_id] = 0

    def publisher_for(
        self,
        run_id: str,
        *,
        worker: str,
        model: Optional[str],
        effort: Optional[str],
        branch: Optional[int] = None,
    ) -> Callable[[dict], None]:
        self._ensure(run_id)

        def _publish(event: dict) -> None:
            if run_id in self.closed:
                return

            payload = dict(event or {})
            clean_event = _normalize_worker_event(
                payload,
                run_id=run_id,
                worker=worker,
                model=model,
                effort=effort,
                branch=branch,
            )

            event_id = int(payload.get("_event_id", self._next_ids[run_id]))

            # Internal _event_id is needed for queues but stripped before yielding/sinking
            internal_event = dict(clean_event)
            internal_event["_event_id"] = event_id

            self._next_ids[run_id] = max(self._next_ids[run_id], event_id + 1)
            q = self.queues[run_id]
            q.append(internal_event)
            # Drop-oldest ring buffer: bound in-memory replay history so a
            # chatty long run can't OOM. subscribe() resumes by _event_id, so
            # dropping the head is safe for a subscriber that has already
            # advanced past it; a late subscriber simply starts at the oldest
            # retained id (the dropped prefix is unrecoverable from memory, as
            # documented by AGY_EVENTBUS_REPLAY_MAX).
            if len(q) > self._replay_max:
                del q[: len(q) - self._replay_max]

            for sink in list(self.sinks.get(run_id, [])):
                try:
                    sink(clean_event)
                except Exception:
                    pass
            try:
                self._wake[run_id].set()
            except Exception:
                pass

        return _publish

    async def subscribe(
        self,
        run_id: str,
        *,
        last_event_id: Optional[str] = None,
        after_id: Optional[int] = None,
    ) -> AsyncIterator[dict]:
        self._ensure(run_id)
        floor = after_id if after_id is not None else -1
        if last_event_id is not None:
            try:
                floor = int(last_event_id)
            except Exception:
                floor = -1

        # Track progress by last-delivered _event_id (NOT by list index): the
        # replay ring may drop the head between wakeups, which would shift list
        # indices. Keying off the monotonic _event_id keeps a resuming
        # subscriber correct even when older rows have been dropped.
        last_delivered = floor

        while True:
            events = self.queues[run_id]
            for event in events:
                event_id = event.get("_event_id")
                if not isinstance(event_id, int):
                    continue
                if event_id <= last_delivered:
                    continue
                last_delivered = event_id
                clean = {k: v for k, v in event.items() if k != "_event_id"}
                yield clean

            if run_id in self.closed:
                return
            wake = self._wake[run_id]
            await wake.wait()
            wake.clear()

    def add_sink(self, run_id: str, sink: Callable[[dict], None]) -> None:
        self._ensure(run_id)
        self.sinks[run_id].append(sink)

    def _evict_run(self, run_id: str) -> None:
        """Fully release every per-run structure for a closed run."""
        self.queues.pop(run_id, None)
        self.sinks.pop(run_id, None)
        self._wake.pop(run_id, None)
        self._next_ids.pop(run_id, None)
        self.closed.discard(run_id)

    def close(self, run_id: str) -> None:
        self._ensure(run_id)
        if run_id in self.closed:
            return
        self.closed.add(run_id)
        try:
            self._wake[run_id].set()
        except Exception:
            pass
        # Bound retained finished-run state. Keep the most recent _retain closed
        # runs (so late SSE replay of a just-finished run still works) and fully
        # evict everything older — otherwise a long-lived broker accumulates
        # every run's queue/wake/next-id/closed entry until OOM.
        self._closed_order.append(run_id)
        while len(self._closed_order) > self._retain:
            stale = self._closed_order.pop(0)
            self._evict_run(stale)

    @staticmethod
    def replay_jsonl(path: Path, *, last_event_id: Optional[str] = None) -> List[dict]:
        if not path.exists():
            return []
        floor = -1
        if last_event_id is not None:
            try:
                floor = int(last_event_id)
            except Exception:
                floor = -1
        out: List[dict] = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for idx, raw in enumerate(f):
                line = raw.strip()
                if not line:
                    continue
                try:
                    raw_event = json.loads(line)
                except Exception:
                    continue
                if isinstance(raw_event, dict):
                    event_id = raw_event.get("_event_id")
                    if not isinstance(event_id, int):
                        # Preserve tolerant append-only replay by falling back to
                        # stable line index when older rows lack _event_id.
                        event_id = idx
                    if event_id <= floor:
                        continue
                    clean = _normalize_worker_event(
                        raw_event,
                        run_id=str(raw_event.get("run_id") or "unknown"),
                        worker=str(raw_event.get("worker") or "unknown"),
                        model=raw_event.get("model") if isinstance(raw_event.get("model"), str) else None,
                        effort=raw_event.get("effort") if isinstance(raw_event.get("effort"), str) else None,
                        branch=_to_branch(raw_event.get("branch")),
                    )
                    out.append(clean)
        return out

    @staticmethod
    def replay_events(events_path: Path, *, after_id: int = -1) -> List[tuple[int, dict]]:
        """Return (original_event_id, clean_WorkerEvent) tuples for SSE replay.
        Tolerates legacy jsonl lines that contain _event_id (extracts for id + SSE,
        strips for the yielded WorkerEvent payload to match §3 exactly).
        """
        if not events_path.exists():
            return []
        out: List[tuple[int, dict]] = []
        with events_path.open("r", encoding="utf-8", errors="replace") as f:
            for idx, raw in enumerate(f):
                line = raw.strip()
                if not line:
                    continue
                try:
                    raw_event = json.loads(line)
                except Exception:
                    continue
                if not isinstance(raw_event, dict):
                    continue
                event_id = raw_event.get("_event_id")
                if not isinstance(event_id, int):
                    event_id = idx
                if event_id <= after_id:
                    continue
                clean = _normalize_worker_event(
                    raw_event,
                    run_id=str(raw_event.get("run_id") or "unknown"),
                    worker=str(raw_event.get("worker") or "unknown"),
                    model=raw_event.get("model") if isinstance(raw_event.get("model"), str) else None,
                    effort=raw_event.get("effort") if isinstance(raw_event.get("effort"), str) else None,
                    branch=_to_branch(raw_event.get("branch")),
                )
                out.append((event_id, clean))
        return out

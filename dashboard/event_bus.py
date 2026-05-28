from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator, Callable, Dict, List, Optional


_WORKER_EVENT_KINDS = {
    "lifecycle",
    "reasoning",
    "message",
    "tool_call",
    "tool_result",
    "usage",
    "stderr",
    "watchdog",
}


class EventBus:
    """Per-run-id queues with sink fanout and replay support."""

    def __init__(self) -> None:
        self.queues: Dict[str, List[dict]] = {}
        self.sinks: Dict[str, List[Callable[[dict], None]]] = {}
        self.closed: set[str] = set()
        self._wake: Dict[str, asyncio.Event] = {}
        self._next_ids: Dict[str, int] = {}

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
            payload["ts"] = float(payload.get("ts") or time.time())
            payload["run_id"] = run_id
            payload["worker"] = payload.get("worker") or worker
            payload["model"] = payload.get("model") or model or "n/a"
            payload["effort"] = payload.get("effort") or effort or "n/a"
            payload["branch"] = payload.get("branch", branch)
            kind = str(payload.get("kind") or "stderr")
            payload["kind"] = kind if kind in _WORKER_EVENT_KINDS else "stderr"
            payload["text"] = payload.get("text", "")
            if payload.get("data") is None or "data" not in payload or not isinstance(payload.get("data"), dict):
                payload["data"] = {}

            event_id = int(payload.get("_event_id", self._next_ids[run_id]))
            payload["_event_id"] = event_id
            self._next_ids[run_id] = max(self._next_ids[run_id], event_id + 1)
            self.queues[run_id].append(payload)

            clean = {k: v for k, v in payload.items() if k != "_event_id"}
            for sink in list(self.sinks.get(run_id, [])):
                try:
                    sink(clean)
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

        events = self.queues[run_id]
        idx = 0
        if floor >= 0:
            while idx < len(events):
                event_id = events[idx].get("_event_id")
                if isinstance(event_id, int) and event_id <= floor:
                    idx += 1
                    continue
                break

        while True:
            events = self.queues[run_id]
            while idx < len(events):
                event = events[idx]
                idx += 1
                event_id = event.get("_event_id")
                if isinstance(event_id, int) and event_id <= floor:
                    continue
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

    def close(self, run_id: str) -> None:
        self._ensure(run_id)
        if run_id in self.closed:
            return
        self.closed.add(run_id)
        try:
            self._wake[run_id].set()
        except Exception:
            pass

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
                    clean = {k: v for k, v in raw_event.items() if k != "_event_id"}
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
                clean = {k: v for k, v in raw_event.items() if k != "_event_id"}
                out.append((event_id, clean))
        return out

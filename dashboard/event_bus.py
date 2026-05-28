from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator, Callable, Dict, List, Optional


_CLOSE = object()


class EventBus:
    """Per-run-id queues with sink fanout and replay support."""

    def __init__(self) -> None:
        self.queues: Dict[str, asyncio.Queue] = {}
        self.sinks: Dict[str, List[Callable[[dict], None]]] = {}
        self.closed: set[str] = set()
        self._history: Dict[str, List[dict]] = {}
        self._next_ids: Dict[str, int] = {}

    def _ensure(self, run_id: str) -> None:
        if run_id not in self.queues:
            self.queues[run_id] = asyncio.Queue()
            self.sinks[run_id] = []
            self._history[run_id] = []
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
            payload["kind"] = payload.get("kind") or "stderr"
            payload["text"] = payload.get("text", "")
            if payload.get("data") is None or "data" not in payload:
                payload["data"] = {}

            event_id = int(payload.get("_event_id", self._next_ids[run_id]))
            payload["_event_id"] = event_id
            self._next_ids[run_id] = event_id + 1
            self._history[run_id].append(payload)

            for sink in list(self.sinks.get(run_id, [])):
                try:
                    sink(payload)
                except Exception:
                    pass
            try:
                self.queues[run_id].put_nowait(payload)
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
        floor = -1
        if after_id is not None:
            floor = after_id
        if last_event_id is not None:
            try:
                floor = int(last_event_id)
            except Exception:
                floor = -1

        for event in list(self._history.get(run_id, [])):
            event_id = event.get("_event_id")
            if isinstance(event_id, int) and event_id <= floor:
                continue
            yield event

        if run_id in self.closed:
            return

        q = self.queues[run_id]
        while True:
            event = await q.get()
            if event is _CLOSE:
                return
            event_id = event.get("_event_id")
            if isinstance(event_id, int) and event_id <= floor:
                continue
            yield event

    def add_sink(self, run_id: str, sink: Callable[[dict], None]) -> None:
        self._ensure(run_id)
        self.sinks[run_id].append(sink)

    def close(self, run_id: str) -> None:
        self._ensure(run_id)
        if run_id in self.closed:
            return
        self.closed.add(run_id)
        try:
            self.queues[run_id].put_nowait(_CLOSE)
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
                if idx <= floor:
                    continue
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if isinstance(event, dict):
                    out.append(event)
        return out

    @staticmethod
    def replay_events(events_path: Path, *, after_id: int = -1) -> List[tuple[int, dict]]:
        out: List[tuple[int, dict]] = []
        for idx, event in enumerate(EventBus.replay_jsonl(events_path, last_event_id=str(after_id))):
            out.append((after_id + 1 + idx, event))
        return out


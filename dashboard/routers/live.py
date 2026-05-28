from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from dashboard.event_bus import EventBus

router = APIRouter()


def _sse_pack(*, event: str, data: object, event_id: Optional[int] = None) -> bytes:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append("data: " + json.dumps(data, ensure_ascii=False))
    lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _read_meta(runs_dir: Path, run_id: str) -> dict:
    meta_path = runs_dir / run_id / "meta.json"
    if not meta_path.exists():
        return {"run_id": run_id}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {"run_id": run_id}


@router.get("/live")
def list_live(request: Request) -> dict:
    state = request.app.state.dashboard
    running = sorted(state.running.values(), key=lambda r: r.get("started_at", 0.0), reverse=True)
    return {"running": running}


@router.post("/live/{run_id}/kill")
def kill_live(run_id: str, request: Request) -> dict:
    state = request.app.state.dashboard
    task = state.tasks.get(run_id)
    if task is None:
        return {"killed": False}
    task.cancel()
    return {"killed": True}


async def _iter_worker_events(request: Request, run_id: str) -> AsyncIterator[bytes]:
    state = request.app.state.dashboard
    events_path = state.runs_dir / run_id / "events.jsonl"

    after_id = -1
    last_event_id = request.headers.get("last-event-id")
    if last_event_id is not None:
        try:
            after_id = int(last_event_id)
        except Exception:
            after_id = -1

    # Replay from persisted JSONL first (if present), then continue with live bus.
    for event_id, event in EventBus.replay_events(events_path, after_id=after_id):
        if await request.is_disconnected():
            return
        after_id = event_id
        yield _sse_pack(event="worker_event", data=event, event_id=event_id)

    # Live events from bus are clean (§3 shape, no _event_id). Synthesize SSE ids
    # by incrementing from the last replay id so reconnect protocol works.
    live_id = (after_id + 1) if after_id >= 0 else 0
    async for event in state.bus.subscribe(run_id, after_id=after_id):
        if await request.is_disconnected():
            return
        eid = live_id
        live_id += 1
        after_id = eid
        yield _sse_pack(event="worker_event", data=event, event_id=eid)

    yield _sse_pack(event="done", data=_read_meta(state.runs_dir, run_id))


@router.get("/sse/{run_id}")
async def run_sse(run_id: str, request: Request) -> Response:
    return StreamingResponse(_iter_worker_events(request, run_id), media_type="text/event-stream")


async def _iter_recent(request: Request) -> AsyncIterator[bytes]:
    state = request.app.state.dashboard
    while True:
        if await request.is_disconnected():
            return
        try:
            item = await asyncio.wait_for(state.recent_queue.get(), timeout=2.0)
            yield _sse_pack(event="done", data=item)
        except asyncio.TimeoutError:
            # Keep connection healthy for long idle windows.
            yield b": keepalive\n\n"


@router.get("/sse/recent")
async def recent_sse(request: Request) -> Response:
    return StreamingResponse(_iter_recent(request), media_type="text/event-stream")

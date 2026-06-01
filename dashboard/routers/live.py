from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from dashboard.event_bus import EventBus

router = APIRouter()

_CLI_RUN_STALE_SECONDS = 300.0


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


def _read_last_heartbeat(events_path: Path) -> Optional[dict]:
    """Return the data payload of the last run-level 'heartbeat' event, if any.

    The heartbeat (#40) is an *explicit* liveness signal (step, free-mem,
    elapsed, seconds-since-progress) so a watcher reads one structured row
    instead of inferring liveness from file mtimes. Returns None for runs that
    predate the heartbeat (the mtime heuristic still bounds the scan).
    """
    try:
        last: Optional[dict] = None
        with events_path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or '"heartbeat"' not in line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if isinstance(ev, dict) and ev.get("kind") == "heartbeat":
                    data = ev.get("data")
                    last = data if isinstance(data, dict) else {}
        return last
    except OSError:
        return None


def _scan_filesystem_runs(runs_dir: Path, already_tracked: set[str]) -> list[dict]:
    """Return running dispatches discovered from runs/ on disk."""
    if not runs_dir.exists():
        return []

    now = time.time()
    out: list[dict] = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if run_id in already_tracked:
            continue
        if (run_dir / "meta.json").exists():
            continue

        events_path = run_dir / "events.jsonl"
        if not events_path.exists():
            continue
        try:
            mtime = events_path.stat().st_mtime
        except OSError:
            continue
        if now - mtime > _CLI_RUN_STALE_SECONDS:
            continue

        prompt_path = run_dir / "prompt.txt"
        try:
            started_at = prompt_path.stat().st_mtime if prompt_path.exists() else mtime
        except OSError:
            started_at = mtime

        instruction_preview = ""
        try:
            if prompt_path.exists():
                head = prompt_path.read_text(encoding="utf-8", errors="replace")[:240]
                if "## Instruction" in head:
                    head = head.split("## Instruction", 1)[1].lstrip()[:160]
                else:
                    head = head[:160]
                instruction_preview = head.replace("\n", " ").strip()
        except OSError:
            pass

        row = {
            "run_id": run_id,
            "started_at": started_at,
            "mode": "cli",
            "instruction_preview": instruction_preview,
            "source": "cli",
        }
        # Explicit liveness from the heartbeat (#40), when present: surfaces the
        # current step and seconds-since-progress so a stuck-but-still-emitting
        # run is visibly stalled rather than just "mtime fresh".
        hb = _read_last_heartbeat(events_path)
        if hb is not None:
            row["heartbeat"] = hb
            if hb.get("step") is not None:
                row["step"] = hb.get("step")
            if hb.get("since_progress_s") is not None:
                row["since_progress_s"] = hb.get("since_progress_s")
        out.append(row)
    return out


@router.get("/live")
def list_live(request: Request) -> dict:
    state = request.app.state.dashboard
    tracked = sorted(state.running.values(), key=lambda r: r.get("started_at", 0.0), reverse=True)
    fs_runs = _scan_filesystem_runs(state.runs_dir, set(state.running.keys()))
    merged = sorted(tracked + fs_runs, key=lambda r: r.get("started_at", 0.0), reverse=True)
    return {"running": merged}


@router.post("/live/{run_id}/kill")
def kill_live(run_id: str, request: Request) -> dict:
    state = request.app.state.dashboard
    task = state.tasks.get(run_id)
    if task is None:
        return {"killed": False}
    task.cancel()
    return {"killed": True}


async def _iter_worker_events(request: Request, run_id: str, resume_after: Optional[int] = None) -> AsyncIterator[bytes]:
    state = request.app.state.dashboard
    events_path = state.runs_dir / run_id / "events.jsonl"

    after_id = resume_after if resume_after is not None else -1
    if resume_after is None:
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
async def run_sse(run_id: str, request: Request, after_id: Optional[int] = None) -> Response:
    return StreamingResponse(_iter_worker_events(request, run_id, resume_after=after_id), media_type="text/event-stream")


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

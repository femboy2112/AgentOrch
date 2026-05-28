from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from harness import dispatch as dispatch_mod

router = APIRouter()


class DispatchRequest(BaseModel):
    instruction: str
    mode: Literal["direct", "adversarial", "feedback", "cascade", "master"] = "adversarial"
    generator_chain: Optional[list[str]] = None
    critic_chain: Optional[list[str]] = None
    test_cmd: Optional[str] = None
    web_search: bool = False
    fallback: bool = True
    cycles: int = Field(default=2, ge=1)
    max_iterations: int = Field(default=5, ge=1)
    branches: int = Field(default=3, ge=1)


async def _run_dispatch(request: Request, run_id: str, payload: DispatchRequest) -> None:
    state = request.app.state.dashboard
    done_payload: dict = {"run_id": run_id, "success": False, "error": "dispatch aborted"}
    try:
        result = await asyncio.to_thread(
            dispatch_mod.dispatch,
            payload.instruction,
            run_id=run_id,
            mode=payload.mode,
            generator_chain=payload.generator_chain,
            critic_chain=payload.critic_chain,
            test_cmd=payload.test_cmd,
            web_search=payload.web_search,
            fallback=payload.fallback,
            cycles=payload.cycles,
            max_iterations=payload.max_iterations,
            branches=payload.branches,
            dashboard_stream_json=True,
        )
        done_payload = {
            "run_id": run_id,
            "success": bool(result.success),
            "mode": result.mode,
            "duration_s": result.duration_s,
        }
        meta_path = Path(result.run_dir) / "meta.json"
        if meta_path.exists():
            try:
                done_payload["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
    except Exception as exc:
        state.bus.close(run_id)
        done_payload = {
            "run_id": run_id,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        state.running.pop(run_id, None)
        state.tasks.pop(run_id, None)
        state.recent_done.append(done_payload)
        try:
            state.recent_queue.put_nowait(done_payload)
        except Exception:
            pass


@router.post("/dispatch")
async def create_dispatch(payload: DispatchRequest, request: Request) -> dict:
    if not payload.instruction.strip():
        raise HTTPException(status_code=400, detail="instruction must be non-empty")

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    state = request.app.state.dashboard
    if run_id in state.tasks:
        raise HTTPException(status_code=409, detail="duplicate run id")

    running = {
        "run_id": run_id,
        "started_at": time.time(),
        "mode": payload.mode,
        "generator": payload.generator_chain or list(dispatch_mod.roles.GENERATOR_CHAIN),
        "critic": payload.critic_chain or list(dispatch_mod.roles.CRITIC_CHAIN),
    }
    state.running[run_id] = running
    state.tasks[run_id] = asyncio.create_task(_run_dispatch(request, run_id, payload))
    return {"run_id": run_id}

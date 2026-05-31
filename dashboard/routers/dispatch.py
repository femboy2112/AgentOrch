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
from harness import roles

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
    out_dir: Optional[str] = None


def _parse_chain_csv(raw: Optional[str], default: list[str]) -> list[str]:
    if raw is None:
        return list(default)
    tokens = [chunk.strip().lower() for chunk in raw.split(",")]
    out = [token for token in tokens if token]
    return out or list(default)


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
            out_dir=payload.out_dir,
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


@router.get("/dispatch/budget")
def get_dispatch_budget(
    mode: str = "adversarial",
    generator: Optional[str] = None,
    critic: Optional[str] = None,
) -> dict:
    generator_chain = _parse_chain_csv(generator, list(roles.GENERATOR_CHAIN))
    critic_chain = _parse_chain_csv(critic, list(roles.CRITIC_CHAIN))
    workers: list[str] = list(generator_chain)
    if mode in {"adversarial", "master"}:
        workers.extend(critic_chain)

    seen: set[str] = set()
    rows: list[dict] = []
    cal = roles._get_calibration()

    for worker in workers:
        if worker in seen:
            continue
        seen.add(worker)
        if worker not in roles.AGENT_CLASSES:
            raise HTTPException(status_code=400, detail=f"unknown worker in chain: {worker}")

        _, cfg = roles._cfg_for_token(worker)
        model = str(cfg.get("model") or "n/a")
        effort_val = cfg.get("effort")
        effort = str(effort_val if effort_val is not None else "n/a")
        effort_lookup = None if effort == "n/a" else effort

        max_output_bytes, stall_seconds = cal.budget_for(worker, model, effort_lookup)
        rows.append(
            {
                "worker": worker,
                "model": model,
                "effort": effort,
                "max_output_bytes": max_output_bytes,
                "stall_seconds": stall_seconds,
                "has_calibration": cal.has_data_for(worker, model, effort_lookup),
            }
        )

    return {"rows": rows}

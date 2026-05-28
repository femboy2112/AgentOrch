from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque

from fastapi import FastAPI

from dashboard.event_bus import EventBus
from harness import dispatch as dispatch_mod


@dataclass
class DashboardState:
    bus: EventBus
    runs_dir: Path
    tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    running: dict[str, dict] = field(default_factory=dict)
    recent_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    recent_done: Deque[dict] = field(default_factory=lambda: deque(maxlen=32))


def create_app() -> FastAPI:
    app = FastAPI(title="AgentOrch Dashboard", version="0.1.0")
    app.state.dashboard = DashboardState(
        bus=dispatch_mod.EVENT_BUS,
        runs_dir=dispatch_mod.RUNS_DIR,
    )

    from dashboard.routers import dispatch, health, live, runs

    app.include_router(health.router)
    app.include_router(dispatch.router, prefix="/api")
    app.include_router(runs.router, prefix="/api")
    app.include_router(live.router, prefix="/api")
    return app


app = create_app()

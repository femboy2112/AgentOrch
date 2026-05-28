from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Deque

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

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


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    state: DashboardState = app.state.dashboard
    state.bus.reset()
    try:
        yield
    finally:
        tasks = list(state.tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for run_id in list(state.bus.queues):
            state.bus.close(run_id)
        state.tasks.clear()
        state.running.clear()


def create_app() -> FastAPI:
    app = FastAPI(title="AgentOrch Dashboard", version="0.1.0", lifespan=_lifespan)
    app.state.dashboard = DashboardState(
        bus=dispatch_mod.EVENT_BUS,
        runs_dir=dispatch_mod.RUNS_DIR,
    )
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="dashboard-static")

    def _index_response() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/", include_in_schema=False)
    def dashboard_index() -> RedirectResponse:
        return RedirectResponse(url="/dispatch", status_code=307)

    @app.get("/dispatch", include_in_schema=False)
    @app.get("/live", include_in_schema=False)
    @app.get("/runs", include_in_schema=False)
    @app.get("/runs/{run_id:path}", include_in_schema=False)
    def dashboard_page() -> FileResponse:
        return _index_response()

    from dashboard.routers import dispatch, health, live, runs

    app.include_router(health.router)
    app.include_router(dispatch.router, prefix="/api")
    app.include_router(runs.router, prefix="/api")
    app.include_router(live.router, prefix="/api")
    return app


app = create_app()

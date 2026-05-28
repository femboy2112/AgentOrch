from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient

from dashboard.server import create_app


def test_dashboard_shell_serves_index_and_static_assets() -> None:
    app = create_app()
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200
    assert "AgentOrch Dashboard" in root.text
    assert "/static/js/app.js" in root.text

    dispatch = client.get("/dispatch")
    assert dispatch.status_code == 200
    assert "AgentOrch Dashboard" in dispatch.text

    runs = client.get("/runs")
    assert runs.status_code == 200
    assert "AgentOrch Dashboard" in runs.text

    run_detail = client.get("/runs/20260528-120000-000")
    assert run_detail.status_code == 200
    assert "AgentOrch Dashboard" in run_detail.text

    css = client.get("/static/css/tokens.css")
    assert css.status_code == 200
    assert "--bg-0" in css.text

    runs_js = client.get("/static/js/pages/runs.js")
    assert runs_js.status_code == 200
    assert "renderRuns" in runs_js.text

    detail_js = client.get("/static/js/pages/run_detail.js")
    assert detail_js.status_code == 200
    assert "renderRunDetail" in detail_js.text

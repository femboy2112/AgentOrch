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

    css = client.get("/static/css/tokens.css")
    assert css.status_code == 200
    assert "--bg-0" in css.text

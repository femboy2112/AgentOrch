from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient

from dashboard.server import create_app
from harness import dispatch as dispatch_mod
from harness.dispatch import DispatchResult


def _parse_sse(body: str) -> list[dict]:
    out: list[dict] = []
    block: list[str] = []
    for line in body.splitlines():
        if line.strip() == "":
            if block:
                out.append(_parse_sse_block(block))
                block = []
            continue
        block.append(line)
    if block:
        out.append(_parse_sse_block(block))
    return out


def _parse_sse_block(lines: list[str]) -> dict:
    parsed = {"event": "message", "data": None, "id": None}
    for line in lines:
        if line.startswith("event: "):
            parsed["event"] = line.split(": ", 1)[1]
        elif line.startswith("id: "):
            parsed["id"] = line.split(": ", 1)[1]
        elif line.startswith("data: "):
            parsed["data"] = json.loads(line.split(": ", 1)[1])
    return parsed


def test_dashboard_api_dispatch_and_sse(tmp_path: Path, monkeypatch):
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", runs_dir)

    def fake_dispatch(
        instruction: str,
        *,
        run_id: str | None = None,
        mode: str = "direct",
        context=None,
        generator_chain=None,
        critic_chain=None,
        fallback: bool = True,
        cycles: int = 2,
        max_iterations: int = 5,
        branches: int = 3,
        test_cmd=None,
        web_search: bool = False,
        dashboard_stream_json: bool = False,
    ) -> DispatchResult:
        assert dashboard_stream_json is True
        assert run_id is not None
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        events_path = run_dir / "events.jsonl"
        events_path.touch()

        def _sink(event: dict) -> None:
            with events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")

        dispatch_mod.EVENT_BUS.add_sink(run_id, _sink)
        pub = dispatch_mod.EVENT_BUS.publisher_for(
            run_id,
            worker="codex",
            model="gpt-5",
            effort="high",
        )

        pub({"kind": "reasoning", "text": "thinking", "data": {}})
        time.sleep(0.01)
        pub({"kind": "message", "text": "done", "data": {}})
        dispatch_mod.EVENT_BUS.close(run_id)
        dispatch_mod.EVENT_BUS.sinks.pop(run_id, None)

        result = DispatchResult(
            run_id=run_id,
            run_dir=str(run_dir),
            mode=mode,
            generator="codex:gpt-5",
            critic=None,
            success=True,
            duration_s=0.1,
            changed_files=[],
            added=[],
            modified=[],
            deleted=[],
            error=None,
            quality={"confidence": "unverified"},
        )
        (run_dir / "prompt.txt").write_text(instruction, encoding="utf-8")
        (run_dir / "stdout.log").write_text("ok\n", encoding="utf-8")
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
        (run_dir / "changed-files.diff").write_text("(no file changes detected)\n", encoding="utf-8")
        (run_dir / "meta.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        return result

    monkeypatch.setattr(dispatch_mod, "dispatch", fake_dispatch)

    app = create_app()
    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.text == "ok"

    resp = client.post(
        "/api/dispatch",
        json={
            "instruction": "stub run",
            "mode": "direct",
            "generator_chain": ["codex"],
            "critic_chain": ["agy"],
        },
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    deadline = time.time() + 2.0
    while time.time() < deadline and run_id in app.state.dashboard.running:
        time.sleep(0.01)

    sse = client.get(f"/api/sse/{run_id}")
    assert sse.status_code == 200
    events = _parse_sse(sse.text)
    worker_events = [e for e in events if e["event"] == "worker_event"]
    assert [e["data"]["kind"] for e in worker_events] == ["reasoning", "message"]
    required_fields = {"ts", "run_id", "worker", "model", "effort", "branch", "kind", "text", "data"}
    for row in worker_events:
        payload = row["data"]
        assert required_fields.issubset(payload)
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["run_id"] == run_id

    runs = client.get("/api/runs")
    assert runs.status_code == 200
    rows = runs.json()["runs"]
    assert any(row["run_id"] == run_id for row in rows)

    detail = client.get(f"/api/runs/{run_id}")
    assert detail.status_code == 200
    assert detail.json()["meta"]["run_id"] == run_id

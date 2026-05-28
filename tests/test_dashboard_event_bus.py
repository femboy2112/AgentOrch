from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dashboard.event_bus import EventBus
from harness import dispatch as dispatch_mod
from harness import roles


async def _collect(bus: EventBus, run_id: str, last_event_id: str | None = None) -> list[dict]:
    out: list[dict] = []
    async for event in bus.subscribe(run_id, last_event_id=last_event_id):
        out.append(event)
    return out


def test_event_bus_publish_subscribe_close_replay():
    bus = EventBus()
    pub = bus.publisher_for("r1", worker="codex", model="gpt-5", effort="high")
    pub({"kind": "reasoning", "text": "step 1", "data": {}})
    pub({"kind": "message", "text": "done", "data": {}})
    bus.close("r1")

    got = asyncio.run(_collect(bus, "r1"))
    assert len(got) == 2
    assert [e["kind"] for e in got] == ["reasoning", "message"]
    for event in got:
        assert "_event_id" not in event
        assert {"ts", "run_id", "worker", "model", "effort", "branch", "kind", "text", "data"}.issubset(event)

    replay = asyncio.run(_collect(bus, "r1", last_event_id="0"))
    assert len(replay) == 1
    assert "_event_id" not in replay[0]


def test_event_bus_jsonl_replay_tolerates_partial_tail(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps({"kind": "reasoning"}) + "\n"
        + json.dumps({"kind": "message"}) + "\n"
        + '{"_event_id": ',
        encoding="utf-8",
    )
    replay = EventBus.replay_jsonl(path, last_event_id="0")
    assert len(replay) == 1
    assert "_event_id" not in replay[0]
    assert replay[0]["kind"] == "message"


def test_event_bus_unknown_kind_falls_back_to_stderr():
    bus = EventBus()
    pub = bus.publisher_for("r-unknown", worker="codex", model="gpt-5", effort="high")
    pub({"kind": "not_a_kind", "text": "x", "data": {"k": 1}})
    bus.close("r-unknown")
    got = asyncio.run(_collect(bus, "r-unknown"))
    assert len(got) == 1
    assert got[0]["kind"] == "stderr"
    assert got[0]["text"] == "x"
    assert got[0]["data"] == {"k": 1}


def test_event_bus_jsonl_replay_filters_by_event_id_not_line_index(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps({"kind": "reasoning", "_event_id": 10}) + "\n"
        + '{"bad-json":'
        + "\n"
        + json.dumps({"kind": "message", "_event_id": 11}) + "\n",
        encoding="utf-8",
    )
    replay = EventBus.replay_jsonl(path, last_event_id="10")
    assert len(replay) == 1
    assert "_event_id" not in replay[0]
    assert replay[0]["kind"] == "message"


def test_event_bus_late_subscriber_gets_replay_plus_live_without_duplicates():
    bus = EventBus()
    pub = bus.publisher_for("r2", worker="codex", model="gpt-5", effort="high")
    pub({"kind": "reasoning", "text": "first", "data": {}})

    async def _run() -> list[dict]:
        got: list[dict] = []

        async def _consume() -> None:
            async for event in bus.subscribe("r2"):
                got.append(event)
                if len(got) == 2:
                    bus.close("r2")

        task = asyncio.create_task(_consume())
        await asyncio.sleep(0)
        pub({"kind": "message", "text": "second", "data": {}})
        await task
        return got

    got = asyncio.run(_run())
    assert [e["text"] for e in got] == ["first", "second"]
    for e in got:
        assert "_event_id" not in e


def test_event_bus_multiple_subscribers_each_receive_full_stream():
    bus = EventBus()
    pub = bus.publisher_for("r3", worker="codex", model="gpt-5", effort="high")

    async def _run() -> tuple[list[dict], list[dict]]:
        t1 = asyncio.create_task(_collect(bus, "r3"))
        t2 = asyncio.create_task(_collect(bus, "r3"))
        await asyncio.sleep(0)
        pub({"kind": "reasoning", "text": "a", "data": {}})
        pub({"kind": "message", "text": "b", "data": {}})
        bus.close("r3")
        return await asyncio.gather(t1, t2)

    got1, got2 = asyncio.run(_run())
    assert [e["kind"] for e in got1] == ["reasoning", "message"]
    assert [e["kind"] for e in got2] == ["reasoning", "message"]


class _FakeEventAgent:
    def __init__(self, *a, **kw):
        self.prompt = kw.get("prompt", "")
        self.event_callback = None

    async def run_async(self, piped_input=None):
        if self.event_callback is not None:
            self.event_callback({"kind": "reasoning", "text": "working", "data": {}})
            self.event_callback({"kind": "message", "text": "done", "data": {}})
        return "stub complete"


def _point_harness_at(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dispatch_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", tmp_path / "runs")


def test_dispatch_writes_events_jsonl(tmp_path: Path, monkeypatch):
    _point_harness_at(tmp_path, monkeypatch)

    def _build_role_agent(chain, **kwargs):
        agent = _FakeEventAgent(prompt=kwargs.get("prompt", ""))
        hook = kwargs.get("post_construct_hook")
        if hook is not None:
            hook(agent, chain[0], {"model": "standard", "effort": "high"})
        return agent

    monkeypatch.setattr(roles, "build_role_agent", _build_role_agent)

    result = dispatch_mod.dispatch("write a file", mode="direct")
    events_path = Path(result.run_dir) / "events.jsonl"
    assert events_path.exists()
    lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(e.get("kind") == "reasoning" for e in lines)
    assert any(e.get("kind") == "message" for e in lines)
    assert any(e.get("data", {}).get("event") == "dispatch_started" for e in lines if e.get("kind") == "lifecycle")
    assert any(e.get("data", {}).get("event") == "dispatch_finished" for e in lines if e.get("kind") == "lifecycle")

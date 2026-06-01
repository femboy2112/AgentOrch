"""Issue #40 — run-level watchdog + heartbeat + notification hook.

The per-agent stall watchdog (harness/roles.py) only catches a *silent* worker.
These tests exercise the *run*-level monitor that complements it:

  * forward-progress tracking that ignores raw token chatter (so a
    stuck-but-chatty run is still seen as not progressing),
  * the stall watchdog that aborts such a run (RunStalled),
  * the periodic heartbeat event,
  * the best-effort, non-fatal notifier (webhook + command forms).

All hermetic: a fake clock drives the progress/abort math, the async cases use
tiny real thresholds, and the notifier is tested via a local command (no
network) plus a monkeypatched webhook.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from harness.run_monitor import (
    Notifier,
    RunMonitor,
    RunStalled,
    read_free_mem_mb,
)


def _run(coro):
    return asyncio.run(coro)


# --- forward-progress tracking (chatter must NOT count) --- #

def test_chatter_does_not_count_as_progress():
    clk = [0.0]
    m = RunMonitor(run_id="r", heartbeat_interval=0, run_stall_abort=10, clock=lambda: clk[0])
    clk[0] = 7.0
    # stderr/stdout/heartbeat are raw chatter -> progress clock stays at t0.
    m.observe({"kind": "stderr", "text": "still thinking..."})
    m.observe({"kind": "stdout", "text": "tokens"})
    m.observe({"kind": "heartbeat", "data": {}})
    assert m.since_progress() == 7.0


def test_milestone_resets_progress_clock():
    clk = [0.0]
    m = RunMonitor(run_id="r", heartbeat_interval=0, run_stall_abort=10, clock=lambda: clk[0])
    clk[0] = 5.0
    m.observe({"kind": "lifecycle", "data": {"event": "agent_started"}})
    assert m.since_progress() == 0.0
    clk[0] = 9.0
    m.observe({"kind": "usage", "data": {}})  # a call completed -> progress
    assert m.since_progress() == 0.0


def test_should_abort_only_after_window():
    clk = [0.0]
    m = RunMonitor(run_id="r", heartbeat_interval=0, run_stall_abort=10, clock=lambda: clk[0])
    clk[0] = 10.0
    assert m.should_abort() is False  # exactly at the window, not past it
    clk[0] = 10.01
    assert m.should_abort() is True


def test_no_watchdog_never_aborts():
    clk = [0.0]
    m = RunMonitor(run_id="r", heartbeat_interval=0, run_stall_abort=None, clock=lambda: clk[0])
    clk[0] = 10_000.0
    assert m.should_abort() is False


def test_step_tracking_from_orchestration():
    m = RunMonitor(run_id="r", heartbeat_interval=0)
    m.observe({"kind": "lifecycle", "data": {"orchestration": {
        "phase": "step", "action": "started", "step_index": 3, "step_total": 8}}})
    assert m._step == "3/8"
    hb = m.heartbeat_event()
    assert hb["kind"] == "heartbeat"
    assert hb["data"]["step"] == "3/8"
    assert "elapsed_s" in hb["data"] and "since_progress_s" in hb["data"]


# --- async supervision --- #

def test_watchdog_aborts_stuck_run():
    emitted = []
    m = RunMonitor(run_id="r", emit=emitted.append, heartbeat_interval=0.05, run_stall_abort=0.2)

    async def stuck():
        await asyncio.sleep(5)  # never makes progress
        return ("out", None)

    with pytest.raises(RunStalled) as ei:
        _run(m.run(stuck()))
    assert ei.value.reason == "stalled"
    assert m.abort_reason == "stalled"
    assert m.heartbeats_emitted >= 1                       # heartbeats fired meanwhile
    assert any(e.get("kind") == "heartbeat" for e in emitted)


def test_normal_completion_returns_value_and_no_abort():
    m = RunMonitor(run_id="r", heartbeat_interval=0.05, run_stall_abort=5.0)

    async def quick():
        await asyncio.sleep(0.05)
        return ("done", 42)

    assert _run(m.run(quick())) == ("done", 42)
    assert m.abort_reason is None


def test_no_supervision_awaits_directly():
    # heartbeat off + no watchdog -> run() awaits the coroutine with zero extra
    # machinery (behaviour identical to the pre-#40 path).
    m = RunMonitor(run_id="r", heartbeat_interval=0, run_stall_abort=None)

    async def coro():
        return "direct"

    assert _run(m.run(coro())) == "direct"
    assert m._supervision_needed is False


# --- notifier --- #

def test_notifier_disabled_is_noop():
    n = Notifier(run_id="r")
    assert n.enabled is False
    n.fire("start", {"x": 1})  # must not raise


def test_notifier_build_body_and_env():
    n = Notifier(command="true", run_id="abc")
    body = n.build_body("stall", {"step": "2/5"})
    assert body == {"event": "stall", "run_id": "abc", "step": "2/5"}
    env = n.command_env("stall", body)
    assert env["AGY_NOTIFY_EVENT"] == "stall"
    assert env["AGY_NOTIFY_RUN_ID"] == "abc"
    assert json.loads(env["AGY_NOTIFY_PAYLOAD"]) == body


def test_notifier_command_receives_payload(tmp_path: Path):
    sink = tmp_path / "notify.out"
    # The command reads the JSON payload from stdin and appends it to a file.
    n = Notifier(command=f"cat >> {sink}", run_id="run9")
    n.deliver("finish", n.build_body("finish", {"success": True}))
    got = json.loads(sink.read_text().strip())
    assert got == {"event": "finish", "run_id": "run9", "success": True}


def test_notifier_webhook_best_effort(monkeypatch):
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())

        class _R:
            def close(self):
                pass

        return _R()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    n = Notifier(webhook="https://example.test/hook", run_id="run10")
    n.deliver("oom", n.build_body("oom", {"free_mem_mb": 12}))
    assert captured["url"] == "https://example.test/hook"
    assert captured["body"] == {"event": "oom", "run_id": "run10", "free_mem_mb": 12}


def test_notifier_command_failure_is_swallowed():
    # A broken command must never raise out of deliver() (best-effort).
    n = Notifier(command="this-binary-does-not-exist-xyz --nope", run_id="r")
    n.deliver("start", n.build_body("start", {}))  # no exception


def test_read_free_mem_mb_returns_int_or_none():
    v = read_free_mem_mb()
    assert v is None or (isinstance(v, int) and v >= 0)


# --- dispatch-level integration (watchdog + heartbeat reach meta/events) --- #

def test_dispatch_run_stall_abort_classifies_stalled(tmp_path, monkeypatch):
    from harness import dispatch as dispatch_mod

    async def _stuck_workflow(*args, **kwargs):
        await asyncio.sleep(5)  # makes no run-level progress
        return ("out", None)

    monkeypatch.setattr(dispatch_mod, "_run_workflow", _stuck_workflow)

    result = dispatch_mod.dispatch(
        "do something that wedges",
        mode="direct",
        out_dir=str(tmp_path),
        run_stall_abort=0.15,
        heartbeat_interval=0.05,
    )
    assert result.success is False
    assert result.run_outcome == "stalled"
    assert "watchdog" in (result.error or "")

    meta = json.loads((Path(result.run_dir) / "meta.json").read_text())
    assert meta["run_outcome"] == "stalled"
    assert meta["success"] is False


def test_dispatch_emits_heartbeat_events(tmp_path, monkeypatch):
    from harness import dispatch as dispatch_mod

    async def _slow_ok_workflow(*args, **kwargs):
        await asyncio.sleep(0.18)  # long enough for a couple heartbeats
        return ("ok", None)

    monkeypatch.setattr(dispatch_mod, "_run_workflow", _slow_ok_workflow)

    result = dispatch_mod.dispatch(
        "a normal run",
        mode="direct",
        out_dir=str(tmp_path),
        run_stall_abort=None,         # no watchdog: should finish cleanly
        heartbeat_interval=0.05,
    )
    assert result.success is True
    assert result.run_outcome is None

    events = (Path(result.run_dir) / "events.jsonl").read_text().splitlines()
    heartbeats = [json.loads(l) for l in events if '"heartbeat"' in l]
    assert heartbeats, "expected at least one heartbeat event in events.jsonl"
    hb = heartbeats[-1]["data"]
    assert "elapsed_s" in hb and "since_progress_s" in hb

"""Tests for the workflow harness: snapshotting, diffing, and the dispatch loop.

The dispatch test stubs the worker agent so no real agy/codex/claude CLI is
invoked — it only verifies the harness machinery (run dir, snapshot, diff,
artifacts, graceful failure).
"""
from __future__ import annotations

import json
from pathlib import Path

from harness import dispatch as dispatch_mod
from harness import roles
from harness.snapshot import diff_snapshots, take_snapshot


def test_snapshot_diff_add_modify_delete(tmp_path):
    (tmp_path / "keep.txt").write_text("same\n")
    (tmp_path / "edit.txt").write_text("line1\nline2\n")
    (tmp_path / "gone.txt").write_text("bye\n")
    before = take_snapshot(tmp_path)

    (tmp_path / "edit.txt").write_text("line1\nCHANGED\n")
    (tmp_path / "gone.txt").unlink()
    (tmp_path / "new.txt").write_text("hello\n")
    after = take_snapshot(tmp_path)

    diff = diff_snapshots(before, after)
    assert diff.added == ["new.txt"]
    assert diff.modified == ["edit.txt"]
    assert diff.deleted == ["gone.txt"]
    assert "keep.txt" not in diff.changed
    assert "CHANGED" in diff.unified
    assert not diff.is_empty


def test_snapshot_ignores_runs_and_pycache(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "x.log").write_text("noise\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "m.pyc").write_bytes(b"\x00\x01")
    (tmp_path / "real.py").write_text("code\n")
    snap = take_snapshot(tmp_path)
    assert "real.py" in snap.files
    assert not any(p.startswith("runs") or "__pycache__" in p for p in snap.files)


class _FakeWriterAgent:
    """Stub worker: writes a file into the project root and returns a summary."""

    def __init__(self, *a, **kw):
        self.prompt = kw.get("prompt", "")

    async def run_async(self, piped_input=None):
        (dispatch_mod.PROJECT_ROOT / "GENERATED.txt").write_text("worker output\n")
        return "Created GENERATED.txt"


class _BoomAgent:
    def __init__(self, *a, **kw):
        pass

    async def run_async(self, piped_input=None):
        raise RuntimeError("provider walled")


class _FakeCUAdapter:
    """Captures the cu request payload forwarded by harness.dispatch."""

    last_req = None

    def __init__(self, *a, **kw):
        pass

    def start(self, req):
        _FakeCUAdapter.last_req = dict(req)
        return type("RunHandle", (), {"status": "completed", "run_id": req["run_id"], "events_path": None})()


def _point_harness_at(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", tmp_path / "runs")


def test_dispatch_direct_captures_run(tmp_path, monkeypatch):
    _point_harness_at(tmp_path, monkeypatch)
    monkeypatch.setattr(roles, "build_role_agent", lambda *a, **k: _FakeWriterAgent())

    result = dispatch_mod.dispatch("write a file", mode="direct")

    assert result.success is True
    assert "GENERATED.txt" in result.added
    run_dir = Path(result.run_dir)
    assert (run_dir / "prompt.txt").read_text().strip().endswith("write a file")
    assert "Created GENERATED.txt" in (run_dir / "stdout.log").read_text()
    assert "GENERATED.txt" in (run_dir / "changed-files.diff").read_text()
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["success"] is True and meta["mode"] == "direct"


def test_dispatch_failure_is_graceful(tmp_path, monkeypatch):
    _point_harness_at(tmp_path, monkeypatch)
    monkeypatch.setattr(roles, "build_role_agent", lambda *a, **k: _BoomAgent())

    result = dispatch_mod.dispatch("do something", mode="direct")

    assert result.success is False
    assert "provider walled" in (result.error or "")
    # Even on failure, artifacts exist and a (no changes) diff is recorded.
    run_dir = Path(result.run_dir)
    assert (run_dir / "stderr.log").exists()
    assert "no file changes" in (run_dir / "changed-files.diff").read_text()


def test_dispatch_forwards_computer_use_real_gui_and_browser_fields(tmp_path, monkeypatch):
    _point_harness_at(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "agy_orchestrator.computer_use.adapter.ComputerUseWorkerAdapter",
        _FakeCUAdapter,
    )

    res = dispatch_mod.dispatch(
        "cu objective",
        mode="direct",
        generator_chain=["computer-use"],
        computer_use_mode="REAL",
        real_gui_policy="children",
        ask_mode="off",
        browser_engine="bing",
        browser_display=":0",
    )

    assert res.success is True
    req = _FakeCUAdapter.last_req
    assert req is not None
    assert req["mode"] == "REAL"
    assert req["real_gui_policy"] == "children"
    assert req["ask_mode"] == "off"
    assert req["browser_engine"] == "bing"
    assert req["browser_display"] == ":0"

from __future__ import annotations

import json
from pathlib import Path

from agy_orchestrator.core import calibration as calibration_mod
from harness import dispatch as dispatch_mod


class _StubWorkflow:
    def __init__(
        self,
        *,
        stage_used=None,
        n_candidates=None,
        n_passed=None,
        winner_index=None,
    ):
        self.verified = True
        self.approved = False
        self.stalled = False
        self.iterations_used = 1
        self.stage_used = stage_used
        self.n_candidates = n_candidates
        self.n_passed = n_passed
        self.winner_index = winner_index


def _read_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_pat_dispatch_logs_stage_used(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    work_dir = tmp_path / "work"
    live = tmp_path / "live.jsonl"
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(calibration_mod, "DEFAULT_LIVE_LEDGER_PATH", live)

    async def _stub_run_workflow(*args, **kwargs):
        wd = Path(kwargs["working_directory"])
        (wd / "created.txt").write_text("x", encoding="utf-8")
        return "ok", _StubWorkflow(stage_used=0)

    monkeypatch.setattr(dispatch_mod, "_run_workflow", _stub_run_workflow)

    result = dispatch_mod.dispatch(
        "noop",
        mode="pat",
        generator_chain=["codex"],
        out_dir=work_dir,
    )
    assert result.success is True
    rows = _read_rows(live)
    assert len(rows) == 1
    row = rows[0]
    assert row["mode"] == "pat"
    assert row["stage_used"] == 0


def test_vote_dispatch_logs_winner_index_and_npassed(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    work_dir = tmp_path / "work"
    live = tmp_path / "live.jsonl"
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(calibration_mod, "DEFAULT_LIVE_LEDGER_PATH", live)

    async def _stub_run_workflow(*args, **kwargs):
        wd = Path(kwargs["working_directory"])
        (wd / "created.txt").write_text("x", encoding="utf-8")
        return "ok", _StubWorkflow(
            n_candidates=3,
            n_passed=1,
            winner_index=0,
        )

    monkeypatch.setattr(dispatch_mod, "_run_workflow", _stub_run_workflow)

    result = dispatch_mod.dispatch(
        "noop",
        mode="vote",
        generator_chain=["codex"],
        out_dir=work_dir,
    )
    assert result.success is True
    rows = _read_rows(live)
    assert len(rows) == 1
    row = rows[0]
    assert row["mode"] == "vote"
    assert row["n_candidates"] == 3
    assert row["n_passed"] == 1
    assert row["winner_index"] == 0

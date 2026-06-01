"""Baseline verifier detection tests for harness.dispatch."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import VerifierResult
from harness import dispatch as dispatch_mod
from harness import roles


class _StubAgent(AgentInstance):
    """No-op worker used to keep dispatch tests model-free."""

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        self._emit_event({"kind": "lifecycle", "data": {"event": "agent_started", "detail": {}}})
        self._emit_event({
            "kind": "lifecycle",
            "data": {"event": "agent_finished", "detail": {"success": True, "returncode": 0}},
        })
        return "stub-output"

    @classmethod
    async def get_available_models(cls) -> List[str]:
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        return 100.0

    def build_command(self, piped_input: Optional[str] = None) -> List[str]:
        return ["true"]


class _ScriptedVerifier:
    """Scripted verifier sequence: first call is baseline, next calls are workflow."""

    scripted: List[object] = []
    seen_cwds: List[str] = []

    def __init__(self, test_commands=None, timeout=None):
        self._calls = 0

    async def verify(self, working_directory: str) -> VerifierResult:
        type(self).seen_cwds.append(working_directory)
        if not type(self).scripted:
            raise AssertionError("scripted verifier has no scripted results")
        idx = self._calls
        self._calls += 1
        item = type(self).scripted[idx] if idx < len(type(self).scripted) else type(self).scripted[-1]
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _patch_roles(monkeypatch):
    def _build_role_agent(chain, *, prompt, fallback, cycles, codex_config=None,
                          post_construct_hook=None):
        agent = _StubAgent(prompt=prompt, model="stub", effort="n/a")
        if post_construct_hook is not None:
            post_construct_hook(agent, chain[0], {"model": "stub", "effort": "n/a"})
        return agent

    monkeypatch.setattr(roles, "build_role_agent", _build_role_agent)
    yield


def _vr(ok: bool, *, returncode: int = 0, error_hash: Optional[str] = None,
        duration_ms: int = 0) -> VerifierResult:
    return VerifierResult(
        ok=ok,
        message="ok" if ok else "failed",
        returncode=returncode,
        error_hash=error_hash,
        duration_ms=duration_ms,
    )


def _run_feedback(tmp_path: Path, monkeypatch, scripted: List[object],
                  baseline_gate: bool = True):
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(dispatch_mod, "QualityVerifier", _ScriptedVerifier)
    _ScriptedVerifier.scripted = scripted
    _ScriptedVerifier.seen_cwds = []
    out_dir = tmp_path / "work"
    # The verifier-delta telemetry derives from the pre-run baseline, which is now
    # opt-in for non-vote modes (single-run speed: it's skipped by default so a full
    # --test-cmd suite doesn't sit on the critical path). These tests exercise that
    # telemetry path, so they request it explicitly via baseline_gate=True.
    return dispatch_mod.dispatch(
        "noop",
        mode="feedback",
        generator_chain=["codex"],
        test_cmd="true",
        max_iterations=1,
        out_dir=out_dir,
        baseline_gate=baseline_gate,
    )


def test_baseline_passing_final_passing_is_preserved(tmp_path, monkeypatch):
    result = _run_feedback(tmp_path, monkeypatch, [_vr(True, duration_ms=1), _vr(True, duration_ms=2)])
    assert result.success is True
    assert result.quality["verifier_delta"] == "preserved"
    assert result.quality["baseline_ok"] is True


def test_baseline_passing_final_failing_is_regressed(tmp_path, monkeypatch):
    result = _run_feedback(
        tmp_path,
        monkeypatch,
        [_vr(True, duration_ms=1), _vr(False, returncode=3, error_hash="deadbeef", duration_ms=2)],
    )
    assert result.success is True
    assert result.quality["verifier_delta"] == "regressed"
    assert result.quality["baseline_ok"] is True


def test_baseline_failing_final_passing_is_fixed(tmp_path, monkeypatch):
    result = _run_feedback(
        tmp_path,
        monkeypatch,
        [_vr(False, returncode=2, error_hash="abc123ef", duration_ms=1), _vr(True, duration_ms=2)],
    )
    assert result.success is True
    assert result.quality["verifier_delta"] == "fixed"
    assert result.quality["baseline_ok"] is False


def test_baseline_failing_final_failing_is_unchanged(tmp_path, monkeypatch):
    result = _run_feedback(
        tmp_path,
        monkeypatch,
        [_vr(False, returncode=2, error_hash="abc123ef", duration_ms=1), _vr(False, returncode=5, duration_ms=2)],
    )
    assert result.success is True
    assert result.quality["verifier_delta"] == "unchanged"
    assert result.quality["baseline_ok"] is False


def test_baseline_skipped_by_default_no_delta(tmp_path, monkeypatch):
    # Single-run speed (rank-1): without --baseline-gate the pre-run baseline does
    # NOT run for non-vote modes, so only the in-loop workflow verifier executes
    # (one scripted call) and there is no verifier_delta telemetry. The real gate
    # still runs, so success is unaffected.
    result = _run_feedback(
        tmp_path, monkeypatch, [_vr(True, duration_ms=2)], baseline_gate=False,
    )
    assert result.success is True
    assert result.quality.get("verifier_delta") is None
    assert result.quality.get("baseline_ok") is None
    # Exactly one verify happened (the in-loop gate), not two (baseline + gate).
    assert len(_ScriptedVerifier.seen_cwds) == 1


def test_no_verifier_means_no_delta(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", tmp_path / "runs")
    result = dispatch_mod.dispatch("noop", mode="direct", generator_chain=["codex"])
    assert result.success is True
    assert "verifier_delta" not in result.quality


def test_baseline_crash_is_swallowed(tmp_path, monkeypatch):
    result = _run_feedback(
        tmp_path,
        monkeypatch,
        [RuntimeError("baseline blew up"), _vr(False, returncode=7, duration_ms=2)],
    )
    assert result.success is True
    assert result.quality.get("verifier_delta") is None
    assert result.quality.get("baseline_ok") is None


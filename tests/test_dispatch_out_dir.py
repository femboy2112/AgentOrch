"""Tests for the `--out-dir` plumbing.

The feature: a dispatch must let the operator pin the worker's cwd to a
caller-chosen directory so a worker invocation from another repo doesn't
land files inside AgentOrch itself. Three behaviours under test:

  1. Default (no out_dir) — worker cwd stays None (inherit parent), snapshot
     scope stays PROJECT_ROOT — the historical behaviour, no regression.
  2. With out_dir — agent.cwd is set to the resolved out_dir, and snapshot
     before/after is scoped to that directory so the diff covers files the
     worker actually wrote.
  3. The runs/<id>/ artifacts always stay under AgentOrch's RUNS_DIR
     regardless of out_dir — they're orchestrator-internal logs.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest

from agy_orchestrator.core.agent import AgentInstance
from harness import dispatch as dispatch_mod
from harness import roles


class _StubAgent(AgentInstance):
    """Records what cwd the orchestration assigned, then no-ops the run.

    We don't need a real subprocess for these tests — we only care that the
    harness threads cwd into the agent. The recording happens via a class-level
    list because `_post_construct_hook` constructs the agent for us.
    """

    constructed: List["_StubAgent"] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).constructed.append(self)

    @classmethod
    async def get_available_models(cls) -> List[str]:
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        return 100.0

    def build_command(self, piped_input: Optional[str] = None) -> List[str]:
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        # Skip the real subprocess; just fire the lifecycle events the harness
        # expects so the dispatch loop completes normally.
        self._emit_event({"kind": "lifecycle", "data": {"event": "agent_started", "detail": {}}})
        self._emit_event({
            "kind": "lifecycle",
            "data": {"event": "agent_finished", "detail": {"success": True, "returncode": 0}},
        })
        return "stub-output"


@pytest.fixture(autouse=True)
def _patch_roles(monkeypatch):
    """Force the stub agent everywhere build_role_agent would pick a real one."""
    _StubAgent.constructed = []

    def _build_role_agent(chain, *, prompt, fallback, cycles, codex_config=None,
                          post_construct_hook=None):
        agent = _StubAgent(prompt=prompt, model="stub", effort="n/a")
        if post_construct_hook is not None:
            post_construct_hook(agent, chain[0], {"model": "stub", "effort": "n/a"})
        return agent

    monkeypatch.setattr(roles, "build_role_agent", _build_role_agent)
    yield


def test_default_dispatch_leaves_agent_cwd_unset(tmp_path, monkeypatch):
    """Smoke: no out_dir ⇒ agent.cwd stays None (inherit parent cwd)."""
    # Keep runs out of the repo by redirecting RUNS_DIR.
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", tmp_path / "runs")
    result = dispatch_mod.dispatch("noop", mode="direct", generator_chain=["codex"])
    assert result.success is True
    assert len(_StubAgent.constructed) == 1
    assert _StubAgent.constructed[0].cwd is None


def test_out_dir_pins_agent_cwd(tmp_path, monkeypatch):
    """With out_dir set, the agent's cwd is the resolved out_dir path."""
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", tmp_path / "runs")
    target = tmp_path / "external-repo"
    result = dispatch_mod.dispatch(
        "noop", mode="direct", generator_chain=["codex"], out_dir=target,
    )
    assert result.success is True
    assert _StubAgent.constructed[0].cwd == str(target.resolve())
    # out_dir is created on demand so workers can't fail on missing parents.
    assert target.exists() and target.is_dir()


def test_run_artifacts_stay_under_runs_dir_even_with_out_dir(tmp_path, monkeypatch):
    """runs/<id>/ artifacts must NOT relocate when out_dir is set — they are
    orchestrator-internal, not user data."""
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", runs_root)
    target = tmp_path / "elsewhere"
    result = dispatch_mod.dispatch(
        "noop", mode="direct", generator_chain=["codex"], out_dir=target,
    )
    assert Path(result.run_dir).resolve().is_relative_to(runs_root.resolve())
    # And the prompt/meta were written to that run dir, not into out_dir.
    assert (Path(result.run_dir) / "prompt.txt").exists()
    assert (Path(result.run_dir) / "meta.json").exists()


def test_snapshot_diff_scope_follows_out_dir(tmp_path, monkeypatch):
    """When out_dir is set, the before/after snapshot is taken at out_dir, so
    a file the worker writes there shows up in the dispatch's changed-files."""
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", tmp_path / "runs")
    target = tmp_path / "scoped"
    target.mkdir()

    # The stub agent doesn't really write files; simulate the worker's write
    # between before and after by patching take_snapshot to lazily fingerprint
    # the live filesystem (which is what the real snapshot does anyway).
    # Easier: just write a file inside out_dir BEFORE dispatch, run, then
    # confirm the snapshot's root was the right path by writing AFTER too and
    # round-tripping through dispatch.
    from harness import snapshot as snap_mod
    seen_roots: List[Path] = []
    orig = snap_mod.take_snapshot

    def _spy(root):
        seen_roots.append(Path(root).resolve())
        return orig(root)

    monkeypatch.setattr(dispatch_mod, "take_snapshot", _spy)

    dispatch_mod.dispatch(
        "noop", mode="direct", generator_chain=["codex"], out_dir=target,
    )
    # Snapshot taken twice (before + after) and BOTH at out_dir, not PROJECT_ROOT.
    assert len(seen_roots) == 2
    assert all(r == target.resolve() for r in seen_roots)


def test_out_dir_string_and_path_both_accepted(tmp_path, monkeypatch):
    """Accept both str and Path so the dashboard/JSON form can pass either."""
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", tmp_path / "runs")
    target = tmp_path / "as-string"

    dispatch_mod.dispatch(
        "noop", mode="direct", generator_chain=["codex"], out_dir=str(target),
    )
    assert _StubAgent.constructed[-1].cwd == str(target.resolve())


def test_out_dir_expands_user(tmp_path, monkeypatch):
    """`~`-prefixed paths are expanded; documented behaviour for the CLI."""
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("HOME", str(tmp_path))
    dispatch_mod.dispatch(
        "noop", mode="direct", generator_chain=["codex"], out_dir="~/sub",
    )
    assert _StubAgent.constructed[-1].cwd == str((tmp_path / "sub").resolve())

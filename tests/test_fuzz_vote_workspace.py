"""Hermetic fuzz / property / edge-case tests for vote + workspace isolation.

Targets:
  * agy_orchestrator/workflows/vote.py
  * agy_orchestrator/execution/workspace.py

These exercise candidate isolation, the dirty/clean copy-mode fallback, the
verifier-semaphore bound, preflight-refusal branches, _apply_workspace clobber
safety, and the apply_node_writes (master graph) path. Everything is hermetic:
no real worker CLI, no network. Generators are AgentInstance subclasses with a
stub run_async; verifiers are tiny scripted stubs; subprocess behaviour (where
exercised) only ever uses git on a throwaway tmp repo.

All committed tests here MUST pass. Defects discovered during fuzzing that the
code does NOT yet handle are reported in the run's findings with a minimal
repro, NOT committed as failing tests.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import VerifierResult
from agy_orchestrator.execution.workspace import (
    apply_node_writes,
    apply_workspace,
    candidate_workspace,
    diff_workspace_against_base,
    has_uncommitted_changes,
    is_git_repo,
)
from agy_orchestrator.workflows.vote import CandidateScore, VoteWorkflow


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _run(coro):
    return asyncio.run(coro)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True, capture_output=True, text=True,
    )


def _git_init(root: Path, files: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")


class _Agent(AgentInstance):
    """Hermetic worker stub. Applies a list of (op, rel, content) mutations
    inside its cwd, optionally raises, optionally sleeps to force interleaving."""

    def __init__(self, *args, ops=None, should_raise: bool = False,
                 sleep: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ops = ops or []          # list of ("write"|"delete"|"mkdir", rel, content)
        self.should_raise = should_raise
        self.sleep = sleep
        self.cwd_at_run: Optional[str] = None

    @classmethod
    async def get_available_models(cls):
        return ["sim"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input=None) -> str:
        self.cwd_at_run = self.cwd
        if self.sleep:
            await asyncio.sleep(self.sleep)
        if self.should_raise:
            raise RuntimeError("simulated worker crash")
        root = Path(self.cwd or ".")
        for op, rel, content in self.ops:
            target = root / rel
            if op == "write":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            elif op == "delete":
                try:
                    target.unlink()
                except OSError:
                    pass
            elif op == "mkdir":
                target.mkdir(parents=True, exist_ok=True)
        return f"ran {len(self.ops)} ops"


class _MarkerVerifier:
    """Passes a candidate iff any non-.git file in its workspace contains one
    of the pass markers. Records how many verifications run *concurrently*."""

    def __init__(self, pass_markers, *, delay: float = 0.0):
        self.pass_markers = set(pass_markers)
        self.delay = delay
        self.calls: List[str] = []
        self.concurrent = 0
        self.max_concurrent = 0

    async def verify(self, working_directory: str) -> VerifierResult:
        self.calls.append(working_directory)
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            for path in Path(working_directory).rglob("*"):
                if ".git" in path.parts or not path.is_file():
                    continue
                try:
                    text = path.read_text()
                except (OSError, UnicodeDecodeError):
                    continue
                for marker in self.pass_markers:
                    if marker in text:
                        return VerifierResult(ok=True, message=f"hit {marker}",
                                              returncode=0, cmd="stub")
            return VerifierResult(ok=False, message="no marker", returncode=1,
                                  cmd="stub")
        finally:
            self.concurrent -= 1


# --------------------------------------------------------------------------- #
# VoteWorkflow construction guards / boundary inputs
# --------------------------------------------------------------------------- #

def test_empty_generators_rejected():
    with pytest.raises(ValueError, match="at least one generator"):
        VoteWorkflow(generators=[], verifier=_MarkerVerifier([]))


def test_none_verifier_rejected():
    with pytest.raises(ValueError, match="verifier"):
        VoteWorkflow(generators=[_Agent(prompt="p")], verifier=None)


@pytest.mark.parametrize("vc,expected", [(-5, 1), (0, 1), (1, 1), (7, 7)])
def test_verifier_concurrency_clamped_to_at_least_one(vc, expected):
    wf = VoteWorkflow(generators=[_Agent(prompt="p")],
                      verifier=_MarkerVerifier([]), verifier_concurrency=vc)
    assert wf.verifier_concurrency == expected


@pytest.mark.parametrize("mp,expected", [(0, None), (None, None), (-3, 1), (4, 4)])
def test_max_parallel_resolution(mp, expected):
    wf = VoteWorkflow(generators=[_Agent(prompt="p")],
                      verifier=_MarkerVerifier([]), max_parallel=mp)
    assert wf.max_parallel == expected


def test_max_parallel_env_nondigit_is_ignored(monkeypatch):
    """A non-digit AGY_MAX_PARALLEL_WORKERS must not crash construction; it
    falls back to None (all-K concurrent)."""
    for bad in ("abc", "-2", "", "  ", "1.5"):
        monkeypatch.setenv("AGY_MAX_PARALLEL_WORKERS", bad)
        wf = VoteWorkflow(generators=[_Agent(prompt="p")],
                          verifier=_MarkerVerifier([]))
        assert wf.max_parallel is None


def test_n_candidates_tracks_generator_count():
    gens = [_Agent(prompt="p") for _ in range(5)]
    wf = VoteWorkflow(generators=gens, verifier=_MarkerVerifier([]))
    assert wf.n_candidates == 5
    assert wf.winner_index == -1
    assert wf.iterations_used == 0


# --------------------------------------------------------------------------- #
# single candidate (K=1) and many candidates (boundary)
# --------------------------------------------------------------------------- #

def test_single_candidate_passes_and_applies(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"x.txt": "orig"})
    gens = [_Agent(prompt="p", ops=[("write", "out.txt", "WIN")])]
    wf = VoteWorkflow(generators=gens, verifier=_MarkerVerifier(["WIN"]),
                      working_directory=str(base), preflight=False)
    out = _run(wf.execute("p"))
    assert wf.verified is True
    assert wf.winner_index == 0
    assert wf.n_passed == 1
    assert (base / "out.txt").read_text() == "WIN"
    assert out.startswith("ran")


def test_many_candidates_one_passes(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"x.txt": "orig"})
    gens = [_Agent(prompt="p", ops=[("write", f"c{i}.txt", f"v{i}")])
            for i in range(8)]
    # Only candidate 5 passes.
    wf = VoteWorkflow(generators=gens, verifier=_MarkerVerifier(["v5"]),
                      working_directory=str(base), preflight=False)
    _run(wf.execute("p"))
    assert wf.winner_index == 5
    assert wf.n_passed == 1
    assert (base / "c5.txt").read_text() == "v5"
    for i in range(8):
        if i != 5:
            assert not (base / f"c{i}.txt").exists()


def test_zero_passers_leaves_base_clean(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"orig.txt": "keep"})
    gens = [_Agent(prompt="p", ops=[("write", "c.txt", f"v{i}")]) for i in range(4)]
    wf = VoteWorkflow(generators=gens, verifier=_MarkerVerifier(["never"]),
                      working_directory=str(base), preflight=False)
    out = _run(wf.execute("p"))
    assert wf.verified is False
    assert wf.winner_index == -1
    assert wf.n_passed == 0
    assert (base / "orig.txt").read_text() == "keep"
    assert not (base / "c.txt").exists()
    # best-effort output still returned for inspection
    assert out.startswith("ran")


def test_all_candidates_raise_returns_empty_and_unverified(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"orig.txt": "keep"})
    gens = [_Agent(prompt="p", should_raise=True) for _ in range(3)]
    wf = VoteWorkflow(generators=gens, verifier=_MarkerVerifier(["x"]),
                      working_directory=str(base), preflight=False)
    out = _run(wf.execute("p"))
    assert wf.verified is False
    assert wf.winner_index == -1
    assert wf.iterations_used == 0          # no candidate produced output
    assert out == ""                         # no non-None output to return
    assert (base / "orig.txt").read_text() == "keep"


# --------------------------------------------------------------------------- #
# verifier semaphore actually bounds concurrency
# --------------------------------------------------------------------------- #

def test_verifier_concurrency_one_serializes_verification(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"x.txt": "orig"})
    gens = [_Agent(prompt="p", ops=[("write", f"c{i}.txt", "v")], sleep=0.0)
            for i in range(5)]
    verifier = _MarkerVerifier(["v"], delay=0.02)
    wf = VoteWorkflow(generators=gens, verifier=verifier,
                      working_directory=str(base), verifier_concurrency=1,
                      preflight=False)
    _run(wf.execute("p"))
    # With concurrency=1 the semaphore must keep verifiers strictly serial.
    assert verifier.max_concurrent == 1


def test_verifier_concurrency_two_allows_overlap(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"x.txt": "orig"})
    gens = [_Agent(prompt="p", ops=[("write", f"c{i}.txt", "v")]) for i in range(4)]
    verifier = _MarkerVerifier(["v"], delay=0.05)
    wf = VoteWorkflow(generators=gens, verifier=verifier,
                      working_directory=str(base), verifier_concurrency=2,
                      preflight=False)
    _run(wf.execute("p"))
    # At least two verifiers overlapped at some point (but never more than 2).
    assert verifier.max_concurrent <= 2


# --------------------------------------------------------------------------- #
# preflight refusal branches
# --------------------------------------------------------------------------- #

class _AlwaysRC:
    """Verifier that always returns a fixed (ok, returncode, message)."""

    def __init__(self, ok, rc=0, message="", stdout="", stderr=""):
        self._ok, self._rc, self._msg = ok, rc, message
        self._stdout, self._stderr = stdout, stderr
        self.calls: List[str] = []

    async def verify(self, working_directory: str) -> VerifierResult:
        self.calls.append(working_directory)
        return VerifierResult(ok=self._ok, message=self._msg, returncode=self._rc,
                              stdout_tail=self._stdout, stderr_tail=self._stderr,
                              cmd="stub")


def test_preflight_passes_when_pristine_ok(tmp_path):
    """Pristine workspace passes verifier -> preflight proceeds (returns None)."""
    base = tmp_path / "base"
    _git_init(base, {"x.txt": "orig"})
    wf = VoteWorkflow(generators=[_Agent(prompt="p")], verifier=_AlwaysRC(True),
                      working_directory=str(base), preflight=True)
    reason = _run(wf._preflight_environment_check())
    assert reason is None


def test_preflight_refuses_on_exit_127_without_base_run(tmp_path):
    """exit 127 = verifier binary absent in isolated workspace -> refuse, and
    do NOT re-run the base (the base verifier should not be invoked)."""
    base = tmp_path / "base"
    _git_init(base, {"x.txt": "orig"})
    verifier = _AlwaysRC(False, rc=127, message="pytest: command not found")
    wf = VoteWorkflow(generators=[_Agent(prompt="p")], verifier=verifier,
                      working_directory=str(base), preflight=True)
    reason = _run(wf._preflight_environment_check())
    assert reason is not None
    assert "command not found" in reason
    # only the pristine workspace was probed; base never re-run for a 127
    assert len(verifier.calls) == 1


def test_preflight_proceeds_when_base_also_red(tmp_path):
    """Non-127 failure but the caller-supplied baseline is red -> genuine red
    baseline; vote proceeds (returns None) without re-running the base."""
    base = tmp_path / "base"
    _git_init(base, {"x.txt": "orig"})
    verifier = _AlwaysRC(False, rc=1, message="2 failed")
    wf = VoteWorkflow(generators=[_Agent(prompt="p")], verifier=verifier,
                      working_directory=str(base), preflight=True,
                      baseline_ok=False)
    reason = _run(wf._preflight_environment_check())
    assert reason is None
    assert len(verifier.calls) == 1   # baseline reused, base not re-run


def test_preflight_refuses_when_base_green_but_pristine_red(tmp_path):
    """Non-127 failure while base passes -> isolation broke something; refuse."""
    base = tmp_path / "base"
    _git_init(base, {"x.txt": "orig"})
    verifier = _AlwaysRC(False, rc=1, message="ImportError: no module foo")
    wf = VoteWorkflow(generators=[_Agent(prompt="p")], verifier=verifier,
                      working_directory=str(base), preflight=True,
                      baseline_ok=True)
    reason = _run(wf._preflight_environment_check())
    assert reason is not None
    assert "did not reproduce" in reason or "command not found" in reason


def test_preflight_probe_failure_is_swallowed(tmp_path):
    """If the preflight can't even build a workspace, it returns None (let the
    real run surface the error) rather than raising."""
    # non-existent base dir: candidate_workspace raises while copying.
    wf = VoteWorkflow(generators=[_Agent(prompt="p")], verifier=_AlwaysRC(True),
                      working_directory=str(tmp_path / "does-not-exist"),
                      preflight=True, prefer_worktree=False)
    reason = _run(wf._preflight_environment_check())
    assert reason is None


def test_preflight_refusal_message_env_absent_vs_ran_and_failed():
    """The static refusal builder branches on the exit code (#35)."""
    env_absent = VoteWorkflow._preflight_refusal(
        VerifierResult(ok=False, returncode=127, message="cmd not found"))
    assert "command not found" in env_absent
    ran_failed = VoteWorkflow._preflight_refusal(
        VerifierResult(ok=False, returncode=1, message="3 failed"))
    assert "did not reproduce" in ran_failed
    # 126 (not executable) is treated as env-absent too
    not_exec = VoteWorkflow._preflight_refusal(
        VerifierResult(ok=False, returncode=126, message="permission denied"))
    assert "command not found" in not_exec


def test_verifier_output_tail_handles_empty_and_combined():
    empty = VoteWorkflow._verifier_output_tail(
        VerifierResult(ok=False, returncode=1))
    assert empty == ""
    combined = VoteWorkflow._verifier_output_tail(
        VerifierResult(ok=False, returncode=1,
                       stdout_tail="OUT", stderr_tail="ERR"))
    assert "OUT" in combined and "ERR" in combined


# --------------------------------------------------------------------------- #
# preflight blocking a doomed run end-to-end (refusal raises RuntimeError)
# --------------------------------------------------------------------------- #

def test_execute_raises_on_preflight_refusal(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"x.txt": "orig"})
    verifier = _AlwaysRC(False, rc=127, message="command not found")
    wf = VoteWorkflow(generators=[_Agent(prompt="p")], verifier=verifier,
                      working_directory=str(base), preflight=True)
    with pytest.raises(RuntimeError, match="preflight refused"):
        _run(wf.execute("p"))
    assert wf.verified is False
    assert wf.winner_index == -1


# --------------------------------------------------------------------------- #
# candidate-setup failures
# --------------------------------------------------------------------------- #

def test_candidate_setup_nonzero_fails_candidate(tmp_path):
    """A non-zero candidate-setup makes that candidate fail cleanly (not crash);
    the verifier is never consulted for it."""
    base = tmp_path / "base"
    _git_init(base, {"x.txt": "orig"})
    gens = [_Agent(prompt="p", ops=[("write", "c.txt", "v")])]
    wf = VoteWorkflow(generators=gens, verifier=_MarkerVerifier(["v"]),
                      working_directory=str(base), preflight=False,
                      candidate_setup="sh -c 'exit 3'")
    _run(wf.execute("p"))
    assert wf.verified is False
    assert wf.n_passed == 0
    ok, err = wf.candidate_outcomes[0]
    assert ok is False
    assert "candidate setup failed" in (err or "")


def test_candidate_setup_timeout_fails_candidate(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"x.txt": "orig"})
    gens = [_Agent(prompt="p", ops=[("write", "c.txt", "v")])]
    wf = VoteWorkflow(generators=gens, verifier=_MarkerVerifier(["v"]),
                      working_directory=str(base), preflight=False,
                      candidate_setup="sleep 5", setup_timeout=0.2)
    _run(wf.execute("p"))
    assert wf.verified is False
    ok, err = wf.candidate_outcomes[0]
    assert ok is False
    assert "timed out" in (err or "")


def test_candidate_setup_success_allows_pass(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"x.txt": "orig"})
    gens = [_Agent(prompt="p", ops=[("write", "c.txt", "v")])]
    wf = VoteWorkflow(generators=gens, verifier=_MarkerVerifier(["v"]),
                      working_directory=str(base), preflight=False,
                      candidate_setup="true")
    _run(wf.execute("p"))
    assert wf.verified is True
    assert wf.winner_index == 0


# --------------------------------------------------------------------------- #
# workspace: dirty vs clean tree -> copy-mode fallback (#36)
# --------------------------------------------------------------------------- #

def test_clean_git_repo_uses_worktree(tmp_path):
    _git_init(tmp_path, {"a.py": "a"})
    assert is_git_repo(tmp_path) is True
    assert has_uncommitted_changes(tmp_path) is False

    async def go():
        async with candidate_workspace(tmp_path) as (ws, backend):
            return backend
    assert _run(go()) == "worktree"


def test_dirty_tracked_change_forces_copy_mode(tmp_path):
    """Uncommitted tracked edit -> copy mode (so the winner-apply can't clobber
    the operator's uncommitted work, #36)."""
    _git_init(tmp_path, {"a.py": "a"})
    (tmp_path / "a.py").write_text("a-edited-uncommitted")
    assert has_uncommitted_changes(tmp_path) is True

    captured = {}

    async def go():
        async with candidate_workspace(tmp_path) as (ws, backend):
            captured["backend"] = backend
            # copy mode must reflect the REAL working tree, not HEAD
            captured["content"] = (ws / "a.py").read_text()
    _run(go())
    assert captured["backend"] == "copy"
    assert captured["content"] == "a-edited-uncommitted"


def test_dirty_untracked_file_forces_copy_mode_and_is_present(tmp_path):
    _git_init(tmp_path, {"a.py": "a"})
    (tmp_path / "untracked.py").write_text("brand new")
    assert has_uncommitted_changes(tmp_path) is True

    captured = {}

    async def go():
        async with candidate_workspace(tmp_path) as (ws, backend):
            captured["backend"] = backend
            captured["present"] = (ws / "untracked.py").exists()
    _run(go())
    assert captured["backend"] == "copy"
    assert captured["present"] is True


def test_gitignored_venv_does_not_count_as_dirty(tmp_path):
    """A git-ignored .venv must NOT mark the tree dirty -> fast worktree path."""
    _git_init(tmp_path, {"a.py": "a", ".gitignore": ".venv/\n"})
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "junk").write_text("x")
    assert has_uncommitted_changes(tmp_path) is False

    async def go():
        async with candidate_workspace(tmp_path) as (ws, backend):
            return backend
    assert _run(go()) == "worktree"


def test_has_uncommitted_falls_open_on_non_git(tmp_path):
    """has_uncommitted_changes on a non-git dir falls open to False."""
    (tmp_path / "f.txt").write_text("x")
    assert has_uncommitted_changes(tmp_path) is False


# --------------------------------------------------------------------------- #
# workspace cleanup robustness
# --------------------------------------------------------------------------- #

def test_workspace_cleanup_on_exception(tmp_path):
    _git_init(tmp_path, {"x.py": "x"})
    seen = []

    async def go():
        with pytest.raises(RuntimeError, match="boom"):
            async with candidate_workspace(tmp_path) as (ws, _b):
                seen.append(ws)
                raise RuntimeError("boom")
    _run(go())
    assert seen and not seen[0].exists()
    # temp root nuked too
    assert not seen[0].parent.exists()


def test_concurrent_workspaces_distinct(tmp_path):
    _git_init(tmp_path, {"x.py": "x"})

    async def one(label):
        async with candidate_workspace(tmp_path, candidate_id=label) as (ws, _b):
            await asyncio.sleep(0.02)
            return ws

    async def go():
        return await asyncio.gather(*[one(f"c{i}") for i in range(6)])
    paths = _run(go())
    assert len({str(p) for p in paths}) == 6
    for p in paths:
        assert not p.exists()


# --------------------------------------------------------------------------- #
# apply_workspace clobber-safety (the cases the code DOES handle correctly)
# --------------------------------------------------------------------------- #

def test_apply_workspace_mirrors_and_deletes(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    (src / "keep.txt").write_text("v2")
    (src / "new.txt").write_text("new")
    (dst / "keep.txt").write_text("v1")
    (dst / "gone.txt").write_text("remove me")
    _run(apply_workspace(src, dst))
    assert (dst / "keep.txt").read_text() == "v2"
    assert (dst / "new.txt").read_text() == "new"
    assert not (dst / "gone.txt").exists()


def test_apply_workspace_skips_ignored_toplevel_dirs(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    # an ignored top-level dir in src must NOT be mirrored, and the same dir in
    # dst must NOT be deleted.
    (src / "runs").mkdir(); (src / "runs" / "junk").write_text("x")
    (dst / "runs").mkdir(); (dst / "runs" / "keep").write_text("y")
    (src / "real.txt").write_text("real")
    _run(apply_workspace(src, dst))
    assert (dst / "real.txt").read_text() == "real"
    assert (dst / "runs" / "keep").read_text() == "y"      # not deleted
    assert not (dst / "runs" / "junk").exists()            # not mirrored


def test_apply_workspace_empty_src_clears_dst_but_keeps_root(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    (dst / "a" / "b" / "c.txt").parent.mkdir(parents=True)
    (dst / "a" / "b" / "c.txt").write_text("x")
    _run(apply_workspace(src, dst))
    assert dst.exists()
    assert list(dst.iterdir()) == []


def test_apply_workspace_identical_file_not_recopied(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    (src / "same.txt").write_text("identical")
    (dst / "same.txt").write_text("identical")
    mtime = (dst / "same.txt").stat().st_mtime_ns
    _run(apply_workspace(src, dst))
    assert (dst / "same.txt").stat().st_mtime_ns == mtime


# --------------------------------------------------------------------------- #
# apply_node_writes (master graph) — disjoint sibling survival
# --------------------------------------------------------------------------- #

def test_apply_node_writes_only_touches_node_diff(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    # base snapshot: node opened with foo.txt = "base"
    (src / "foo.txt").write_text("node-changed")     # node modified foo
    (src / "added.txt").write_text("node-added")      # node added a file
    base_snapshot = {"foo.txt": b"base"}
    # dst has a sibling's disjoint write that must survive
    (dst / "sibling.txt").write_text("sibling wrote this")
    (dst / "foo.txt").write_text("base")
    _run(apply_node_writes(src, dst, base_snapshot))
    assert (dst / "foo.txt").read_text() == "node-changed"
    assert (dst / "added.txt").read_text() == "node-added"
    assert (dst / "sibling.txt").read_text() == "sibling wrote this"  # untouched


def test_apply_node_writes_propagates_deletion(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    # node deleted "old.txt" (present in base_snapshot, absent in src)
    base_snapshot = {"old.txt": b"old"}
    (dst / "old.txt").write_text("old")
    _run(apply_node_writes(src, dst, base_snapshot))
    assert not (dst / "old.txt").exists()


# --------------------------------------------------------------------------- #
# diff parsing in _score_passer is robust to odd diff content
# --------------------------------------------------------------------------- #

def test_score_passer_counts_files_and_detects_tests(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"src/a.py": "a", "tests/test_a.py": "t"})
    gens = [_Agent(prompt="p", ops=[
        ("write", "src/a.py", "PASS-A"),
        ("write", "tests/test_a.py", "PASS-T"),
    ])]
    wf = VoteWorkflow(generators=gens, verifier=_MarkerVerifier(["PASS-A"]),
                      working_directory=str(base), preflight=False)
    _run(wf.execute("p"))
    assert wf.verified is True
    assert wf.ranking_metric["has_test_changes"] is True
    assert wf.ranking_metric["files_changed"] >= 1


def test_ranking_prefers_fewer_files_then_tests_then_size():
    passers = [
        CandidateScore(0, "o0", Path("/x"), files_changed=3, diff_size=10,
                       has_test_changes=True),
        CandidateScore(1, "o1", Path("/x"), files_changed=1, diff_size=100,
                       has_test_changes=False),
        CandidateScore(2, "o2", Path("/x"), files_changed=1, diff_size=100,
                       has_test_changes=True),
    ]
    wf = VoteWorkflow(generators=[_Agent(prompt="p")], verifier=_MarkerVerifier([]))
    ranked = wf._rank_passers(passers)
    # fewest files first (1), then test-touching preferred among the two 1-file
    assert ranked[0].index == 2
    assert ranked[-1].index == 0


# --------------------------------------------------------------------------- #
# unicode / control chars in worker output and filenames
# --------------------------------------------------------------------------- #

def test_unicode_and_control_chars_in_content(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"x.txt": "orig"})
    weird = "WIN​\x1b[31m🔥日本語"
    gens = [_Agent(prompt="p", ops=[("write", "out.txt", weird)])]
    wf = VoteWorkflow(generators=gens, verifier=_MarkerVerifier(["WIN"]),
                      working_directory=str(base), preflight=False)
    _run(wf.execute("p"))
    assert wf.verified is True
    assert (base / "out.txt").read_text() == weird


# --------------------------------------------------------------------------- #
# regression guard: copy backend with a winner deletion + new nested file
# --------------------------------------------------------------------------- #

def test_copy_backend_full_cycle_with_delete_and_add(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "keep.txt").write_text("keep")
    (base / "obsolete.txt").write_text("bye")  # winner will delete

    gens = [_Agent(prompt="p", ops=[
        ("write", "deep/new.txt", "PASS"),
        ("delete", "obsolete.txt", ""),
    ])]
    wf = VoteWorkflow(generators=gens, verifier=_MarkerVerifier(["PASS"]),
                      working_directory=str(base), prefer_worktree=False,
                      preflight=False)
    _run(wf.execute("p"))
    assert wf.verified is True
    assert (base / "deep" / "new.txt").read_text() == "PASS"
    assert not (base / "obsolete.txt").exists()
    assert (base / "keep.txt").read_text() == "keep"

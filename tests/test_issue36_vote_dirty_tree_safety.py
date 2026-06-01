"""Issue #36 — vote must not destroy uncommitted/untracked work on a dirty tree.

The reported symptom ("generation happens in the shared --out-dir") was a
misdiagnosis: generation IS isolated — each candidate runs with cwd set to its
own ``/tmp`` worktree/copy (`vote.py::_run_one`). The real, high-severity bug is
narrower and worse: a ``git worktree`` candidate is checked out at HEAD only, so
on a DIRTY out-dir it silently omits the operator's uncommitted edits + untracked
files; the (HEAD-based) winner is then mirrored back over the operator's tree and
``_apply_workspace``'s file-removal step DELETES those uncommitted/untracked
changes.

Fix: ``candidate_workspace`` falls back to copy-mode when the git base is dirty
(copy snapshots the real working tree, ignored dirs excluded), so candidates
reflect what the operator actually has and the winner can be applied without
clobbering their work. Clean repos keep the fast worktree path.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Optional

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import VerifierResult
from agy_orchestrator.execution.workspace import (
    candidate_workspace,
    has_uncommitted_changes,
)
from agy_orchestrator.workflows.vote import VoteWorkflow


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)


def _clean_repo(tmp_path: Path) -> Path:
    base = tmp_path / "repo"
    base.mkdir()
    _git(base, "init", "-q", "-b", "main")
    _git(base, "config", "user.email", "t@t")
    _git(base, "config", "user.name", "t")
    (base / "orig.py").write_text("base\n")
    (base / ".gitignore").write_text(".venv/\nruns/\n")
    _git(base, "add", "-A")
    _git(base, "commit", "-q", "-m", "init")
    return base


def _run(coro):
    return asyncio.run(coro)


class _Gen(AgentInstance):
    def __init__(self, *a, fname="cand.py", content="x", **k):
        super().__init__(*a, **k)
        self.fname = fname
        self.content = content

    @classmethod
    async def get_available_models(cls):
        return ["m"]

    @classmethod
    async def get_model_usage(cls, m):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input=None):
        Path(self.cwd, self.fname).write_text(self.content)
        return f"wrote {self.fname} in {self.cwd}"


class _PassVerifier:
    async def verify(self, working_directory):
        return VerifierResult(ok=True, message="ok", returncode=0)


# --- the dirty-tree detector --- #

def test_has_uncommitted_changes_detects_dirty_and_clean(tmp_path):
    base = _clean_repo(tmp_path)
    assert has_uncommitted_changes(base) is False
    # An ignored file does NOT count as dirty (porcelain respects .gitignore).
    (base / ".venv").mkdir()
    (base / ".venv" / "x").write_text("v")
    assert has_uncommitted_changes(base) is False
    # A tracked edit -> dirty.
    (base / "orig.py").write_text("base\nEDIT\n")
    assert has_uncommitted_changes(base) is True
    _git(base, "checkout", "--", "orig.py")
    assert has_uncommitted_changes(base) is False
    # An untracked, non-ignored file -> dirty.
    (base / "new.py").write_text("n")
    assert has_uncommitted_changes(base) is True


# --- backend routing --- #

def test_clean_repo_uses_worktree(tmp_path):
    base = _clean_repo(tmp_path)

    async def go():
        async with candidate_workspace(base, candidate_id="c", prefer_worktree=True) as (_ws, backend):
            return backend
    assert _run(go()) == "worktree"


def test_dirty_repo_falls_back_to_copy_and_keeps_working_state(tmp_path):
    base = _clean_repo(tmp_path)
    (base / "orig.py").write_text("base\nUNCOMMITTED\n")
    (base / "untracked.py").write_text("UNTRACKED\n")
    (base / ".venv").mkdir()
    (base / ".venv" / "x").write_text("v")  # ignored -> must be skipped

    async def go():
        async with candidate_workspace(base, candidate_id="c", prefer_worktree=True) as (ws, backend):
            return (
                backend,
                "UNCOMMITTED" in (ws / "orig.py").read_text(),
                (ws / "untracked.py").exists(),
                (ws / ".venv").exists(),
            )

    backend, has_uncommitted, has_untracked, has_venv = _run(go())
    assert backend == "copy"
    assert has_uncommitted is True      # candidate sees the real working state
    assert has_untracked is True
    assert has_venv is False            # ignored heavy dir still skipped


# --- the headline guarantee: a vote run on a dirty tree must not lose work --- #

def test_vote_on_dirty_tree_preserves_uncommitted_and_untracked(tmp_path):
    base = _clean_repo(tmp_path)
    (base / "orig.py").write_text("base\nUNCOMMITTED\n")   # uncommitted tracked edit
    (base / "untracked.py").write_text("UNTRACKED\n")       # untracked file

    gens = [
        _Gen(prompt="p", fname="cand0.py", content="0"),
        _Gen(prompt="p", fname="cand1.py", content="1"),
    ]
    wf = VoteWorkflow(generators=gens, verifier=_PassVerifier(),
                      working_directory=str(base), preflight=False)
    _run(wf.execute("p"))

    assert wf.n_passed == 2                 # both candidates ran in isolation
    assert wf.winner_index in (0, 1)
    # The operator's uncommitted edit + untracked file SURVIVE the run.
    assert (base / "orig.py").read_text() == "base\nUNCOMMITTED\n"
    assert (base / "untracked.py").exists()
    # The winner's new file was applied on top.
    winner_file = "cand0.py" if wf.winner_index == 0 else "cand1.py"
    assert (base / winner_file).exists()

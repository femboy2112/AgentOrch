"""VoteWorkflow — K-parallel candidates in isolated workspaces.

Each candidate runs in its own workspace, verifier grades each in isolation,
all passers are ranked deterministically and the top candidate is applied.
This is the real-parallel mode the disk-state limitation previously blocked.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import VerifierResult
from agy_orchestrator.workflows.vote import VoteWorkflow

# --- test helpers --- #


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)


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


def _run(coro):
    return asyncio.run(coro)


class _SimAgent(AgentInstance):
    """Simulates a worker that writes a candidate-specific file into its cwd
    when run_async is called. Records its cwd so tests can assert isolation.
    """

    def __init__(self, *args, write_filename: str = "result.txt",
                 write_content: str = "default-content",
                 extra_writes: Optional[List[Tuple[str, str]]] = None,
                 delete_paths: Optional[List[str]] = None,
                 should_raise: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.write_filename = write_filename
        self.write_content = write_content
        self.extra_writes = extra_writes or []
        self.delete_paths = delete_paths or []
        self.should_raise = should_raise
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
        if self.should_raise:
            raise RuntimeError("simulated worker crash")
        # Write into the cwd the harness/vote workflow set on us.
        target = Path(self.cwd or ".") / self.write_filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.write_content)
        for rel, content in self.extra_writes:
            p = Path(self.cwd or ".") / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        for rel in self.delete_paths:
            try:
                (Path(self.cwd or ".") / rel).unlink()
            except OSError:
                pass
        return f"wrote {self.write_filename} with {self.write_content}"


class _ScriptedVerifier:
    """Returns scripted (pass, msg) tuples — verdicts is a dict keyed by
    a marker string the verifier looks for inside working_directory.
    Lets us test "candidate 1 passes, candidate 0 fails" deterministically.
    """

    def __init__(self, marker_to_outcome: dict):
        # marker_to_outcome: {"win-content": True, "lose-content": False}
        self._verdicts = marker_to_outcome
        self.calls: List[str] = []

    async def verify(self, working_directory: str) -> VerifierResult:
        self.calls.append(working_directory)
        # Look for any marker file in working_directory whose content
        # matches one of the verdicts.
        for path in Path(working_directory).rglob("*"):
            if not path.is_file():
                continue
            # A real verifier (make/pytest) never greps .git; skip it so the
            # worktree-vs-base asymmetry (.git is a dir in base, a file in a
            # worktree) can't produce a phantom match.
            if ".git" in path.parts:
                continue
            try:
                content = path.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            for marker, verdict in self._verdicts.items():
                if marker in content:
                    return VerifierResult(
                        ok=verdict,
                        message=f"matched {marker}",
                        returncode=0 if verdict else 1,
                        cmd="stub",
                        duration_ms=0,
                    )
        return VerifierResult(
            ok=False,
            message="no marker found",
            returncode=1,
            cmd="stub",
            duration_ms=0,
        )


# --- core happy-path --- #

def test_winner_chosen_by_ranking_when_multiple_pass(tmp_path):
    """K=3 candidates all pass; ranker should pick the smallest-change
    candidate by files-changed first."""
    base = tmp_path / "base"
    _git_init(
        base,
        {
            "existing.txt": "original",
            "cand0/a.txt": "seed",
            "cand0/b.txt": "seed",
            "cand0/c.txt": "seed",
            "cand1/only.txt": "seed",
            "cand2/a.txt": "seed",
            "cand2/b.txt": "seed",
            "cand2/c.txt": "seed",
            "cand2/d.txt": "seed",
            "cand2/e.txt": "seed",
        },
    )

    gens = [
        _SimAgent(
            prompt="p",
            write_filename="cand0/a.txt",
            write_content="pass-0",
            extra_writes=[("cand0/b.txt", "pass-0"), ("cand0/c.txt", "pass-0")],
        ),
        _SimAgent(prompt="p", write_filename="cand1/only.txt", write_content="pass-1"),
        _SimAgent(
            prompt="p",
            write_filename="cand2/a.txt",
            write_content="pass-2",
            extra_writes=[
                ("cand2/b.txt", "pass-2"),
                ("cand2/c.txt", "pass-2"),
                ("cand2/d.txt", "pass-2"),
                ("cand2/e.txt", "pass-2"),
            ],
        ),
    ]

    verifier = _ScriptedVerifier({"pass-0": True, "pass-1": True, "pass-2": True})
    wf = VoteWorkflow(generators=gens, verifier=verifier, working_directory=str(base))
    out = _run(wf.execute("do the thing"))

    # Winner identified.
    assert wf.winner_index == 1
    assert wf.verified is True
    assert wf.n_passed == 3
    assert wf.n_candidates == 3
    assert wf.iterations_used == 3
    # Base reflects the winner's content.
    assert (base / "cand1/only.txt").read_text() == "pass-1"
    # Losers' edits were not mirrored into base.
    assert (base / "cand0/a.txt").read_text() == "seed"
    assert (base / "cand2/a.txt").read_text() == "seed"
    # Existing files in base are preserved.
    assert (base / "existing.txt").read_text() == "original"
    # Each agent ran in a distinct cwd (none == base).
    cwds = [g.cwd_at_run for g in gens]
    assert len(set(cwds)) == 3
    assert str(base) not in cwds
    assert out.startswith("wrote")


def test_test_changing_candidate_preferred_over_terser_non_test(tmp_path):
    """With equal files-changed count, test-touching candidate is preferred
    over non-test candidate even when both pass."""
    base = tmp_path / "base"
    _git_init(
        base,
        {
            "existing.txt": "original",
            "src/foo.py": "seed",
            "notes/bar.txt": "seed",
            "tests/test_foo.py": "seed",
        },
    )

    gens = [
        _SimAgent(
            prompt="p",
            write_filename="src/foo.py",
            write_content="pass",
            extra_writes=[("notes/bar.txt", "pass")],
        ),
        _SimAgent(
            prompt="p",
            write_filename="src/foo.py",
            write_content="pass",
            extra_writes=[("tests/test_foo.py", "pass")],
        ),
    ]
    verifier = _ScriptedVerifier({"pass": True})
    wf = VoteWorkflow(generators=gens, verifier=verifier, working_directory=str(base))
    _run(wf.execute("p"))

    assert wf.winner_index == 1
    assert (base / "tests/test_foo.py").read_text() == "pass"


def test_ranking_metric_attribute_exposed(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"existing.txt": "original"})

    gens = [
        _SimAgent(prompt="p", write_filename="a.txt", write_content="pass"),
        _SimAgent(prompt="p", write_filename="b.txt", write_content="pass"),
    ]
    verifier = _ScriptedVerifier({"pass": True})
    wf = VoteWorkflow(generators=gens, verifier=verifier, working_directory=str(base))
    _run(wf.execute("p"))

    assert isinstance(wf.ranking_metric, dict)
    assert {"files_changed", "diff_size", "has_test_changes"} <= set(wf.ranking_metric)


def test_isolation_during_run_each_candidate_sees_own_cwd(tmp_path):
    """While the workflow runs, each candidate must see its OWN write,
    not a peer's. This is the property the per-candidate workspace
    was built to guarantee."""
    base = tmp_path / "base"
    _git_init(base, {"x.py": "x"})

    gens = [
        _SimAgent(prompt="p", write_filename=f"only_in_{i}.txt", write_content=f"v{i}")
        for i in range(4)
    ]
    verifier = _ScriptedVerifier({"v0": True})  # candidate 0 passes
    wf = VoteWorkflow(generators=gens, verifier=verifier, working_directory=str(base))
    _run(wf.execute("p"))

    # All 4 ran in distinct workspaces, none was the base.
    assert len(set(g.cwd_at_run for g in gens)) == 4
    for g in gens:
        assert g.cwd_at_run != str(base)
    # Winner = candidate 0; ONLY its file should land in base.
    assert (base / "only_in_0.txt").read_text() == "v0"
    for i in (1, 2, 3):
        assert not (base / f"only_in_{i}.txt").exists()


# --- 0/K passes --- #

def test_no_candidate_passes_leaves_base_clean(tmp_path):
    """When 0/K pass, work_dir must NOT be touched. The dispatch can still
    return SOME text (so the operator sees output) but the disk side-effect
    is reverted by virtue of nothing being applied."""
    base = tmp_path / "base"
    _git_init(base, {"orig.txt": "original"})

    gens = [
        _SimAgent(prompt="p", write_filename="cand.txt", write_content=f"v{i}")
        for i in range(3)
    ]
    # No verdict matches any content -> all fail.
    verifier = _ScriptedVerifier({"never-matches": True})
    wf = VoteWorkflow(generators=gens, verifier=verifier, working_directory=str(base))
    out = _run(wf.execute("p"))

    assert wf.verified is False
    assert wf.n_passed == 0
    assert wf.winner_index == -1
    # Critical: base is untouched.
    assert (base / "orig.txt").read_text() == "original"
    assert not (base / "cand.txt").exists()
    # Some output IS returned for operator inspection.
    assert out.startswith("wrote")


# --- error resilience --- #

def test_candidate_exception_does_not_block_others(tmp_path):
    """Candidate 1 raises mid-run; candidates 0 and 2 should still run
    and grade normally."""
    base = tmp_path / "base"
    _git_init(base, {"x.txt": "x"})

    gens = [
        _SimAgent(prompt="p", write_content="lose-v0"),
        _SimAgent(prompt="p", write_content="never-runs", should_raise=True),
        _SimAgent(prompt="p", write_content="win-v2"),
    ]
    verifier = _ScriptedVerifier({"win-v2": True})
    wf = VoteWorkflow(generators=gens, verifier=verifier, working_directory=str(base))
    out = _run(wf.execute("p"))

    assert wf.winner_index == 2
    assert wf.verified is True
    assert wf.iterations_used == 2   # only 2 candidates produced output
    assert out.startswith("wrote")


# --- guardrails --- #

def test_verifier_required():
    gens = [_SimAgent(prompt="p")]
    with pytest.raises(ValueError, match="verifier"):
        VoteWorkflow(generators=gens, verifier=None)


def test_empty_generators_rejected():
    verifier = _ScriptedVerifier({})
    with pytest.raises(ValueError, match="at least one generator"):
        VoteWorkflow(generators=[], verifier=verifier)


def test_prefer_worktree_false_uses_copy_backend(tmp_path):
    """Operator escape hatch threads through cleanly even when the base is
    a git repo. We can't easily inspect which backend was used post-hoc
    without log-snooping, but we CAN verify the workflow still completes
    correctly."""
    base = tmp_path / "base"
    _git_init(base, {"a.txt": "a"})

    gens = [_SimAgent(prompt="p", write_content="win-content")]
    verifier = _ScriptedVerifier({"win-content": True})
    wf = VoteWorkflow(
        generators=gens, verifier=verifier,
        working_directory=str(base), prefer_worktree=False,
    )
    _run(wf.execute("p"))
    assert wf.verified is True


def test_apply_skips_identical_files(tmp_path):
    """If the winner's workspace has files identical to the base (it
    barely wrote anything new), the apply step should be a no-op for
    those files. We assert this indirectly: a file that the worker DID
    NOT touch must keep its original mtime."""
    base = tmp_path / "base"
    _git_init(base, {"keepme.txt": "untouched", "target.txt": "before"})
    keepme = base / "keepme.txt"
    pristine_mtime = keepme.stat().st_mtime_ns

    gens = [_SimAgent(prompt="p", write_filename="target.txt", write_content="win-after")]
    verifier = _ScriptedVerifier({"win-after": True})
    wf = VoteWorkflow(generators=gens, verifier=verifier, working_directory=str(base))
    _run(wf.execute("p"))

    # Touched file changed.
    assert (base / "target.txt").read_text() == "win-after"
    # Untouched file's mtime stayed the same (no spurious copy2 over identical content).
    assert keepme.stat().st_mtime_ns == pristine_mtime


def test_winner_deletion_propagates_to_base(tmp_path):
    """Winner deletes obsolete.py; apply step should mirror that deletion."""
    base = tmp_path / "base"
    _git_init(base, {"keep.txt": "keep", "obsolete.py": "obsolete"})

    gens = [
        _SimAgent(
            prompt="p",
            write_filename="cand.txt",
            write_content="win-marker",
            delete_paths=["obsolete.py"],
        )
    ]
    verifier = _ScriptedVerifier({"win-marker": True})
    wf = VoteWorkflow(generators=gens, verifier=verifier, working_directory=str(base))
    _run(wf.execute("p"))

    assert not (base / "obsolete.py").exists()
    assert (base / "keep.txt").read_text() == "keep"
    assert (base / "cand.txt").read_text() == "win-marker"

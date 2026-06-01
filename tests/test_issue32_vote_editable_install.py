"""Issue #32 — `vote` mode on editable-install / git-ignored-venv repos.

Two independent defects, two fixes (both in VoteWorkflow):

  * Defect 1 (correctness, fatal): a candidate workspace is a `git worktree`
    (tracked files only) or a copy that skips `.venv`, so a verifier that
    resolves its tools out of a git-ignored venv (`make check` ->
    `.venv/bin/pytest`, the common editable-install layout) fails for EVERY
    candidate environmentally -> 0/K pass -> nothing applied, after a long run.
    Fix: a fail-fast preflight that refuses early when a pristine isolated
    workspace fails the verifier that the real base tree passes — while still
    proceeding on a genuinely red baseline (the fix-failing-tests use case).

  * Defect 2 (host safety): K verifier `make check` runs fired concurrently
    (`asyncio.gather`) could OOM/freeze a small-RAM box. Fix: an
    `asyncio.Semaphore` caps concurrent verifier runs (default 1 = serial,
    like master mode), independent of K — generation stays parallel.
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


# --- helpers (mirroring tests/test_vote_workflow.py) --- #

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
    """Writes a candidate-specific file into its cwd when run."""

    def __init__(self, *args, write_filename: str = "result.txt",
                 write_content: str = "pass", **kwargs):
        super().__init__(*args, **kwargs)
        self.write_filename = write_filename
        self.write_content = write_content
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
        target = Path(self.cwd or ".") / self.write_filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.write_content)
        return f"wrote {self.write_filename}"


class _VenvAwareVerifier:
    """Faithful stand-in for `make check` -> `.venv/bin/pytest`: ok iff a
    `.venv/bin/pytest` exists in the working_directory. The real base tree has
    it (untracked / git-ignored, present on disk); a fresh worktree checkout
    does not — exactly the editable-install trap of issue #32."""

    def __init__(self):
        self.calls: List[str] = []

    async def verify(self, working_directory: str) -> VerifierResult:
        self.calls.append(working_directory)
        present = (Path(working_directory) / ".venv" / "bin" / "pytest").exists()
        return VerifierResult(
            ok=present,
            message="ok" if present else "make: .venv/bin/pytest: No such file or directory",
            returncode=0 if present else 127,
            cmd="make check",
            duration_ms=0,
        )


class _MarkerVerifier:
    """ok iff any file under working_directory contains `marker`."""

    def __init__(self, marker: str):
        self.marker = marker
        self.calls: List[str] = []

    async def verify(self, working_directory: str) -> VerifierResult:
        self.calls.append(working_directory)
        for p in Path(working_directory).rglob("*"):
            if p.is_file():
                try:
                    if self.marker in p.read_text():
                        return VerifierResult(ok=True, message="matched",
                                              returncode=0, cmd="stub", duration_ms=0)
                except (OSError, UnicodeDecodeError):
                    continue
        return VerifierResult(ok=False, message="no marker", returncode=1,
                              cmd="stub", duration_ms=0)


class _ConcurrencyTrackingVerifier:
    """Records the peak number of overlapping verify() calls."""

    def __init__(self, hold: float = 0.03):
        self.hold = hold
        self.current = 0
        self.max_seen = 0
        self._lock = asyncio.Lock()

    async def verify(self, working_directory: str) -> VerifierResult:
        async with self._lock:
            self.current += 1
            self.max_seen = max(self.max_seen, self.current)
        await asyncio.sleep(self.hold)
        async with self._lock:
            self.current -= 1
        return VerifierResult(ok=True, message="ok", returncode=0,
                              cmd="stub", duration_ms=0)


# --- Defect 1: fail-fast preflight --- #

def _editable_install_base(tmp_path) -> Path:
    """A git repo with a git-ignored .venv present on disk (untracked)."""
    base = tmp_path / "base"
    _git_init(base, {"src.py": "x", ".gitignore": ".venv/\n"})
    pytest_bin = base / ".venv" / "bin" / "pytest"
    pytest_bin.parent.mkdir(parents=True, exist_ok=True)
    pytest_bin.write_text("#!/bin/sh\n")  # untracked, git-ignored -> absent in worktree
    return base


def test_preflight_refuses_when_isolation_strips_the_venv(tmp_path):
    base = _editable_install_base(tmp_path)
    gens = [_SimAgent(prompt="p", write_content="pass") for _ in range(3)]
    verifier = _VenvAwareVerifier()
    wf = VoteWorkflow(generators=gens, verifier=verifier, working_directory=str(base))

    with pytest.raises(RuntimeError) as exc:
        _run(wf.execute("do the thing"))

    msg = str(exc.value).lower()
    assert "preflight" in msg
    assert ".venv" in msg
    assert "master" in msg          # steers to the unaffected mode
    # Refused BEFORE fanning out: no candidate ever ran.
    assert all(g.cwd_at_run is None for g in gens)
    assert wf.verified is False
    assert wf.n_passed == 0
    assert wf.winner_index == -1


def test_preflight_proceeds_on_a_genuinely_red_baseline(tmp_path):
    """If the verifier is red on BOTH the pristine workspace and the base, the
    baseline is legitimately red (vote is being used to fix it) — proceed, do
    not refuse. The winning candidate writes the marker that turns it green."""
    base = tmp_path / "base"
    _git_init(base, {"x.py": "x"})  # no marker anywhere -> base is red
    gens = [
        _SimAgent(prompt="p", write_filename="fix.py", write_content="GREEN"),
        _SimAgent(prompt="p", write_filename="other.py", write_content="still-red"),
    ]
    verifier = _MarkerVerifier("GREEN")
    wf = VoteWorkflow(generators=gens, verifier=verifier, working_directory=str(base))

    out = _run(wf.execute("p"))  # must NOT raise

    assert wf.verified is True
    assert wf.winner_index == 0
    assert (base / "fix.py").read_text() == "GREEN"
    assert out.startswith("wrote")


def test_preflight_proceeds_when_pristine_workspace_already_passes(tmp_path):
    """A self-contained repo whose verifier is green on HEAD: preflight sees a
    passing pristine workspace and proceeds without even probing the base."""
    base = tmp_path / "base"
    _git_init(base, {"already.py": "GREEN committed"})  # marker is tracked
    verifier = _MarkerVerifier("GREEN")
    gens = [_SimAgent(prompt="p", write_filename="more.py", write_content="GREEN")]
    wf = VoteWorkflow(generators=gens, verifier=verifier, working_directory=str(base))

    out = _run(wf.execute("p"))  # must NOT raise

    assert wf.verified is True
    # pristine workspace already green -> base was never probed.
    assert str(base) not in verifier.calls
    assert out.startswith("wrote")


def test_preflight_can_be_disabled(tmp_path):
    """preflight=False skips the probe entirely (caller opts out)."""
    base = _editable_install_base(tmp_path)
    gens = [_SimAgent(prompt="p", write_content="pass")]
    verifier = _VenvAwareVerifier()
    wf = VoteWorkflow(generators=gens, verifier=verifier,
                      working_directory=str(base), preflight=False)
    # No RuntimeError; it runs, every candidate fails the venv check -> 0/K.
    _run(wf.execute("p"))
    assert wf.verified is False
    assert wf.n_passed == 0


# --- Defect 2: bounded verifier concurrency --- #

def test_default_verifier_concurrency_is_serial(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"x.py": "x"})
    wf = VoteWorkflow(generators=[_SimAgent(prompt="p")],
                      verifier=_MarkerVerifier("z"), working_directory=str(base))
    assert wf.verifier_concurrency == 1


def test_verifier_concurrency_cap_one_serializes_verifiers(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"x.py": "x"})
    gens = [_SimAgent(prompt="p", write_filename=f"c{i}.txt", write_content="pass")
            for i in range(4)]
    verifier = _ConcurrencyTrackingVerifier()
    wf = VoteWorkflow(generators=gens, verifier=verifier, working_directory=str(base),
                      verifier_concurrency=1, preflight=False)
    _run(wf.execute("p"))
    # 4 candidates, but verifiers never overlapped.
    assert verifier.max_seen == 1


def test_verifier_concurrency_cap_bounds_overlap(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"x.py": "x"})
    gens = [_SimAgent(prompt="p", write_filename=f"c{i}.txt", write_content="pass")
            for i in range(5)]
    verifier = _ConcurrencyTrackingVerifier()
    wf = VoteWorkflow(generators=gens, verifier=verifier, working_directory=str(base),
                      verifier_concurrency=2, preflight=False)
    _run(wf.execute("p"))
    # 5 candidates want to verify; the cap holds peak overlap at 2.
    assert verifier.max_seen == 2

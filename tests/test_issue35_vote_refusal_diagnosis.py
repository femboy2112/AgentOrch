"""Issue #35 — vote preflight refusal must diagnose the ACTUAL failure, not
hard-code the pre-#34 ".venv absent / exit-127" narrative.

After #34's `--candidate-setup`, "pristine verify fails" is most often NOT the
venv-absent case — the bootstrap usually builds the venv, so the dominant
remaining failure is "candidate env built but doesn't reproduce the working
tree" (undeclared/transitive dep, untracked-file/CWD/state dependence). Two
defects this fixes:

  1. The message branches on the pristine verifier's exit code:
       * 127/126 (command not found) -> env-absent guidance (binary missing).
       * any other non-zero (ran but failed) -> "env built, verifier RAN, suite
         FAILED — most likely an undeclared dependency" guidance.
  2. The refusal surfaces the pristine verifier's captured stdout/stderr tail
     (the actual failing test names), not just `exit code N`.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Optional

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import VerifierResult
from agy_orchestrator.workflows.vote import VoteWorkflow


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
    def __init__(self, *args, write_filename="r.txt", write_content="pass", **kwargs):
        super().__init__(*args, **kwargs)
        self.write_filename = write_filename
        self.write_content = write_content

    @classmethod
    async def get_available_models(cls):
        return ["sim"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input=None) -> str:
        p = Path(self.cwd or ".") / self.write_filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.write_content)
        return f"wrote {self.write_filename}"


class _BaseGreenWorkspaceFails:
    """Base tree passes; any isolated workspace fails with a configurable
    exit code + captured output — exercising each refusal branch (#35)."""

    def __init__(self, base: Path, returncode: int, message: str,
                 stdout_tail: str = "", stderr_tail: str = ""):
        self.base = str(base)
        self.returncode = returncode
        self.message = message
        self.stdout_tail = stdout_tail
        self.stderr_tail = stderr_tail

    async def verify(self, working_directory: str) -> VerifierResult:
        if working_directory == self.base:
            return VerifierResult(ok=True, message="All tests passed", returncode=0)
        return VerifierResult(
            ok=False, message=self.message, returncode=self.returncode,
            stdout_tail=self.stdout_tail, stderr_tail=self.stderr_tail,
        )


def _base(tmp_path) -> Path:
    base = tmp_path / "base"
    _git_init(base, {"src.py": "x"})
    return base


# --- defect 1: exit code branches the narrative --- #

def test_exit_127_keeps_env_absent_narrative(tmp_path):
    base = _base(tmp_path)
    verifier = _BaseGreenWorkspaceFails(
        base, returncode=127,
        message="Command failed with exit code 127: .venv/bin/pytest -q",
        stderr_tail=".venv/bin/pytest: No such file or directory",
    )
    wf = VoteWorkflow(generators=[_SimAgent(prompt="p")], verifier=verifier,
                      working_directory=str(base))
    with pytest.raises(RuntimeError) as ei:
        _run(wf.execute("p"))
    m = str(ei.value)
    assert "command not found" in m
    assert "candidate-setup" in m           # the env-absent fix steer
    assert ".venv/bin/pytest" in m
    # Must NOT misapply the env-absent label as if tests ran.
    assert "verifier RAN" not in m


def test_exit_1_uses_ran_but_failed_narrative_not_venv_absent(tmp_path):
    """The headline #35 bug: exit 1 (verifier ran, tests failed) must NOT claim
    the binary is absent — it must point at undeclared deps / unreproduced env."""
    base = _base(tmp_path)
    verifier = _BaseGreenWorkspaceFails(
        base, returncode=1,
        message="Command failed with exit code 1: .venv/bin/pytest -q",
        stdout_tail="FAILED tests/test_async.py::test_await - no plugin 'asyncio'",
    )
    wf = VoteWorkflow(generators=[_SimAgent(prompt="p")], verifier=verifier,
                      working_directory=str(base))
    with pytest.raises(RuntimeError) as ei:
        _run(wf.execute("p"))
    m = str(ei.value)
    assert "exit 1" in m
    assert "RAN" in m and "FAILED" in m         # it ran, suite failed
    assert "undeclared" in m or "transitive" in m
    # The stale boilerplate must be gone for this branch.
    assert "is absent and EVERY candidate" not in m
    assert "command not found" not in m


# --- defect 2: the captured output is surfaced, not discarded --- #

def test_refusal_surfaces_verifier_output_tail(tmp_path):
    base = _base(tmp_path)
    verifier = _BaseGreenWorkspaceFails(
        base, returncode=1,
        message="Command failed with exit code 1: .venv/bin/pytest -q",
        stdout_tail="FAILED tests/test_async.py::test_await - missing pytest-asyncio",
        stderr_tail="ERROR: plugin not found",
    )
    wf = VoteWorkflow(generators=[_SimAgent(prompt="p")], verifier=verifier,
                      working_directory=str(base))
    with pytest.raises(RuntimeError) as ei:
        _run(wf.execute("p"))
    m = str(ei.value)
    # The actual failing test name is visible without a manual repro.
    assert "test_await" in m
    assert "pytest-asyncio" in m
    assert "[stdout]" in m and "[stderr]" in m


def test_output_tail_empty_when_nothing_captured(tmp_path):
    base = _base(tmp_path)
    verifier = _BaseGreenWorkspaceFails(
        base, returncode=1,
        message="Command failed with exit code 1: make check",
    )  # no stdout/stderr tails
    wf = VoteWorkflow(generators=[_SimAgent(prompt="p")], verifier=verifier,
                      working_directory=str(base))
    with pytest.raises(RuntimeError) as ei:
        _run(wf.execute("p"))
    m = str(ei.value)
    # No empty "output (tail)" section when there's nothing to show.
    assert "pristine verifier output (tail)" not in m
    # Still includes the verifier's message.
    assert "exit code 1" in m


def test_verifier_output_tail_helper_is_bounded_and_labeled():
    r = VerifierResult(ok=False, message="x", returncode=1,
                       stdout_tail="A" * 5000, stderr_tail="B" * 5000)
    tail = VoteWorkflow._verifier_output_tail(r, limit=1200)
    assert len(tail) <= 1200
    # Empty result -> empty string (no spurious labels).
    assert VoteWorkflow._verifier_output_tail(
        VerifierResult(ok=False, message="x", returncode=1)) == ""

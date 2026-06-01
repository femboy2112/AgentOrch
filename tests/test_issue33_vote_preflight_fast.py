"""Issue #33 — the #32 vote preflight must refuse fast, not re-run the base.

The preflight's job is "pristine isolated workspace fails the verifier that the
base tree passes -> refuse." Getting the "base passes" half by re-running the
full base verifier doubles time-to-refuse on slow-gate repos (~5 min wasted),
because the harness already ran that exact gate on the same unchanged tree
seconds earlier. Two fixes:

  * exit 127 ("command not found") is environmental by definition -> refuse
    immediately, no base run at all.
  * for any other non-zero exit, reuse the caller-supplied baseline result
    instead of recomputing it; only probe the base when no baseline is known.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import List, Optional

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
        p = Path(self.cwd or ".") / self.write_filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.write_content)
        return f"wrote {self.write_filename}"


class _RecordingVerifier:
    """ok iff `marker` appears in some file (skipping .git); `fail_rc` on miss.
    Records every working_directory it was asked to verify."""

    def __init__(self, marker: str, fail_rc: int = 1):
        self.marker = marker
        self.fail_rc = fail_rc
        self.calls: List[str] = []

    async def verify(self, working_directory: str) -> VerifierResult:
        self.calls.append(working_directory)
        for p in Path(working_directory).rglob("*"):
            if p.is_file() and ".git" not in p.parts:
                try:
                    if self.marker in p.read_text():
                        return VerifierResult(ok=True, message="ok", returncode=0,
                                              cmd="stub", duration_ms=0)
                except (OSError, UnicodeDecodeError):
                    continue
        return VerifierResult(ok=False, message=f"missing {self.marker}",
                              returncode=self.fail_rc, cmd="stub", duration_ms=0)


class _VenvVerifier:
    """ok iff `.venv/bin/pytest` exists; exit 127 (command not found) otherwise —
    the editable-install signature."""

    def __init__(self):
        self.calls: List[str] = []

    async def verify(self, working_directory: str) -> VerifierResult:
        self.calls.append(working_directory)
        present = (Path(working_directory) / ".venv" / "bin" / "pytest").exists()
        return VerifierResult(
            ok=present,
            message="ok" if present else "Command failed with exit code 127",
            returncode=0 if present else 127, cmd="make check", duration_ms=0,
        )


def _venv_base(tmp_path) -> Path:
    base = tmp_path / "base"
    _git_init(base, {"src.py": "x", ".gitignore": ".venv/\n"})
    binp = base / ".venv" / "bin" / "pytest"
    binp.parent.mkdir(parents=True, exist_ok=True)
    binp.write_text("#!/bin/sh\n")  # git-ignored -> absent in a worktree
    return base


# --- exit 127 short-circuits: zero base runs --- #

def test_exit_127_refuses_without_running_the_base(tmp_path):
    base = _venv_base(tmp_path)
    verifier = _VenvVerifier()
    wf = VoteWorkflow(generators=[_SimAgent(prompt="p")], verifier=verifier,
                      working_directory=str(base))
    with pytest.raises(RuntimeError, match="preflight"):
        _run(wf.execute("p"))
    # Only the pristine workspace was verified; the base tree was NOT re-run.
    assert len(verifier.calls) == 1
    assert str(base) not in verifier.calls


# --- non-127: reuse the caller's baseline, don't recompute --- #

def test_non127_failure_reuses_baseline_ok_true_to_refuse(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"x.py": "x"})  # no GREEN marker anywhere
    verifier = _RecordingVerifier("GREEN", fail_rc=1)
    wf = VoteWorkflow(generators=[_SimAgent(prompt="p")], verifier=verifier,
                      working_directory=str(base), baseline_ok=True)
    with pytest.raises(RuntimeError, match="preflight"):
        _run(wf.execute("p"))
    # pristine failed rc=1; baseline_ok=True was reused -> base NOT re-verified.
    assert str(base) not in verifier.calls
    assert len(verifier.calls) == 1


def test_non127_failure_reuses_baseline_ok_false_to_proceed(tmp_path):
    base = tmp_path / "base"
    _git_init(base, {"x.py": "x"})
    verifier = _RecordingVerifier("GREEN", fail_rc=1)
    gens = [_SimAgent(prompt="p", write_filename="fix.py", write_content="GREEN")]
    wf = VoteWorkflow(generators=gens, verifier=verifier,
                      working_directory=str(base), baseline_ok=False)
    out = _run(wf.execute("p"))  # red baseline -> proceed, candidate turns green
    assert wf.verified is True
    assert out.startswith("wrote")
    # Base was never verified by the preflight (reused baseline_ok=False).
    assert str(base) not in verifier.calls


def test_non127_failure_without_baseline_probes_base_once(tmp_path):
    """Fallback preserved: no reusable baseline + non-127 fail -> probe base
    exactly once to disambiguate (here it's red, so proceed)."""
    base = tmp_path / "base"
    _git_init(base, {"x.py": "x"})
    verifier = _RecordingVerifier("GREEN", fail_rc=1)
    gens = [_SimAgent(prompt="p", write_filename="fix.py", write_content="GREEN")]
    wf = VoteWorkflow(generators=gens, verifier=verifier, working_directory=str(base))
    out = _run(wf.execute("p"))
    assert wf.verified is True
    assert out.startswith("wrote")
    # The base WAS probed once (no baseline supplied).
    assert verifier.calls.count(str(base)) == 1

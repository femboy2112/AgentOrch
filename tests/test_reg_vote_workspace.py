"""Regression tests for the vote candidate-isolation / winner-apply subsystem.

Subsystem: workflows/vote.py + execution/workspace.py — candidate isolation /
winner apply.

Hermetic: no network, no real worker CLI. A stub AgentInstance subclass stands
in for any worker; an in-memory verifier grades by scanning for a marker. Each
test pins a confirmed-and-fixed defect so a regression re-introducing it is
caught.

Covered findings:
  * vote-workspace-1 — a winning candidate that does a file->package refactor
    (replaces the regular file ``mod.py`` with the directory
    ``mod.py/__init__.py``) used to crash ``apply_workspace`` with
    FileExistsError on ``dst_path.parent.mkdir`` (an intermediate dst component
    was a regular file), leaving work_dir half-applied and the exception
    escaping ``VoteWorkflow.execute()`` after ``verified=True``. The inverse
    (dir->file) silently lost the winner's file. Now both land correctly and
    ``execute()`` does not raise.
  * vote-workspace-2 — a malformed ``AGY_SETUP_TIMEOUT`` (e.g. ``"abc"``) used
    to raise an uncaught ValueError out of ``VoteWorkflow.__init__``, bricking
    every vote dispatch at construction. Now it falls back to the 1200s default;
    valid values (including ``"0"`` / empty = no timeout) and an explicit
    ``setup_timeout`` arg are unchanged.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import VerifierResult
from agy_orchestrator.workflows.vote import VoteWorkflow


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


class _RefactorAgent(AgentInstance):
    """Worker that replaces the file ``mod.py`` with the package
    ``mod.py/__init__.py`` carrying a pass-marker."""

    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input=None):
        p = Path(self.cwd) / "mod.py"
        p.unlink()
        p.mkdir()
        (p / "__init__.py").write_text("PASS-MARKER")
        return "done"


class _NoopAgent(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input=None):
        return "done"


class _MarkerVerifier:
    """Passes iff some file under the workspace contains PASS-MARKER."""

    async def verify(self, working_directory):
        for f in Path(working_directory).rglob("*"):
            if ".git" in f.parts or not f.is_file():
                continue
            try:
                if "PASS-MARKER" in f.read_text():
                    return VerifierResult(ok=True, returncode=0)
            except Exception:
                pass
        return VerifierResult(ok=False, returncode=1)


class _OkVerifier:
    async def verify(self, working_directory):
        return VerifierResult(ok=True, returncode=0)


def _make_base_repo() -> Path:
    base = Path(tempfile.mkdtemp()) / "base"
    base.mkdir(parents=True)
    _git(base, "init", "-q", "-b", "main")
    _git(base, "config", "user.email", "t@t")
    _git(base, "config", "user.name", "t")
    (base / "mod.py").write_text("print(1)")
    _git(base, "add", "-A")
    _git(base, "commit", "-q", "-m", "i")
    return base


# ---------------------------------------------------------------------------
# vote-workspace-1: file<->dir refactor through the full execute() apply path.
# ---------------------------------------------------------------------------

def test_vote_execute_applies_file_to_package_winner_no_crash():
    base = _make_base_repo()
    wf = VoteWorkflow(
        generators=[_RefactorAgent(prompt="p")],
        verifier=_MarkerVerifier(),
        working_directory=str(base),
        preflight=False,
    )
    # Before the fix this raised FileExistsError out of execute().
    out = asyncio.run(wf.execute("p"))

    assert out == "done"
    assert wf.verified is True
    target = base / "mod.py" / "__init__.py"
    assert target.is_file(), "winner's package file must land at the intended path"
    assert target.read_text() == "PASS-MARKER"


# ---------------------------------------------------------------------------
# vote-workspace-2: malformed AGY_SETUP_TIMEOUT must not crash __init__.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["abc", "10s", "  ", "1,2"])
def test_malformed_setup_timeout_env_falls_back_to_default(monkeypatch, bad):
    monkeypatch.setenv("AGY_SETUP_TIMEOUT", bad)
    wf = VoteWorkflow(
        generators=[_NoopAgent(prompt="p")],
        verifier=_OkVerifier(),
        working_directory=".",
    )
    assert wf.setup_timeout == 1200.0


@pytest.mark.parametrize("value,expected", [("55", 55.0), ("0", 0.0), ("", 0.0)])
def test_valid_setup_timeout_env_unchanged(monkeypatch, value, expected):
    monkeypatch.setenv("AGY_SETUP_TIMEOUT", value)
    wf = VoteWorkflow(
        generators=[_NoopAgent(prompt="p")],
        verifier=_OkVerifier(),
        working_directory=".",
    )
    assert wf.setup_timeout == expected


def test_explicit_setup_timeout_arg_overrides_bad_env(monkeypatch):
    monkeypatch.setenv("AGY_SETUP_TIMEOUT", "abc")
    wf = VoteWorkflow(
        generators=[_NoopAgent(prompt="p")],
        verifier=_OkVerifier(),
        working_directory=".",
        setup_timeout=9.0,
    )
    assert wf.setup_timeout == 9.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

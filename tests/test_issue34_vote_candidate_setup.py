"""Issue #34 — make `vote` actually WORK on editable-install repos.

#32 made vote *safe* on the git-ignored-venv + `pip install -e .` layout by
refusing with a preflight. #34 adds the deferred *capability*: an opt-in
per-candidate environment bootstrap (`candidate_setup`) run inside each
isolated workspace before its verifier. With it, each candidate gets its own
venv whose editable install resolves `import <pkg>` to THAT candidate's source
— so vote isolation becomes sound and the preflight proceeds instead of
refusing.

Here the real `python -m venv && pip install -e .` is simulated by a setup
command that materialises the `.venv/bin/pytest` the verifier looks for.
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


class _VenvVerifier:
    """ok iff `.venv/bin/pytest` exists (exit 127 otherwise) — the
    editable-install signature a per-candidate setup must satisfy."""

    def __init__(self):
        self.calls: List[str] = []

    async def verify(self, working_directory: str) -> VerifierResult:
        self.calls.append(working_directory)
        present = (Path(working_directory) / ".venv" / "bin" / "pytest").exists()
        return VerifierResult(
            ok=present,
            message="ok" if present else "exit code 127",
            returncode=0 if present else 127, cmd="make check", duration_ms=0,
        )


def _venv_base(tmp_path) -> Path:
    """Editable-install repo: a git-ignored .venv exists in the base but never
    in a worktree (so a fresh candidate has no .venv until setup builds one)."""
    base = tmp_path / "base"
    _git_init(base, {"src.py": "x", ".gitignore": ".venv/\n"})
    binp = base / ".venv" / "bin" / "pytest"
    binp.parent.mkdir(parents=True, exist_ok=True)
    binp.write_text("#!/bin/sh\n")
    return base


# Simulates `python -m venv .venv && .venv/bin/pip install -e .`:
_SETUP_BOOTSTRAP = "mkdir -p .venv/bin && printf '#!/bin/sh\\n' > .venv/bin/pytest"


# --- the headline #34 win: setup makes vote PROCEED on an editable repo --- #

def test_candidate_setup_makes_vote_proceed_on_editable_repo(tmp_path):
    base = _venv_base(tmp_path)
    gens = [
        _SimAgent(prompt="p", write_filename="a.txt", write_content="pass"),
        _SimAgent(prompt="p", write_filename="b.txt", write_content="pass"),
    ]
    verifier = _VenvVerifier()
    wf = VoteWorkflow(generators=gens, verifier=verifier, working_directory=str(base),
                      candidate_setup=_SETUP_BOOTSTRAP)
    out = _run(wf.execute("p"))  # must NOT raise: setup builds each candidate's venv

    assert wf.verified is True
    assert wf.n_passed == 2          # both candidates bootstrapped + passed
    assert wf.winner_index in (0, 1)
    assert out.startswith("wrote")
    # The winner's edit landed in base; the per-candidate .venv did NOT (ignored).
    winner_file = "a.txt" if wf.winner_index == 0 else "b.txt"
    assert (base / winner_file).read_text() == "pass"
    # The candidate .venv is ignored on apply -> base's own venv is untouched.
    assert (base / ".venv" / "bin" / "pytest").read_text() == "#!/bin/sh\n"


def test_without_setup_same_repo_still_refuses(tmp_path):
    """Control: no setup -> the #32 guardrail still fires (capability is opt-in)."""
    base = _venv_base(tmp_path)
    wf = VoteWorkflow(generators=[_SimAgent(prompt="p")], verifier=_VenvVerifier(),
                      working_directory=str(base))  # candidate_setup=None
    with pytest.raises(RuntimeError, match="preflight"):
        _run(wf.execute("p"))


# --- setup failures are handled cleanly --- #

def test_preflight_refuses_when_setup_fails(tmp_path):
    base = _venv_base(tmp_path)
    wf = VoteWorkflow(generators=[_SimAgent(prompt="p")], verifier=_VenvVerifier(),
                      working_directory=str(base), candidate_setup="exit 7")
    with pytest.raises(RuntimeError, match="setup"):
        _run(wf.execute("p"))


def test_candidate_setup_failure_fails_that_candidate_not_crash(tmp_path):
    """With preflight disabled, a candidate whose setup fails is marked failed
    (not a crash); 0/K -> nothing applied."""
    base = tmp_path / "base"
    _git_init(base, {"x.py": "x"})
    gens = [_SimAgent(prompt="p", write_content="pass")]
    wf = VoteWorkflow(generators=gens, verifier=_VenvVerifier(),
                      working_directory=str(base), candidate_setup="exit 7",
                      preflight=False)
    _run(wf.execute("p"))
    assert wf.verified is False
    assert wf.n_passed == 0
    ok, msg = wf.candidate_outcomes[0]
    assert ok is False
    assert "setup" in (msg or "")


def test_setup_runs_in_the_workspace_and_base_venv_is_untouched(tmp_path):
    """The bootstrap must run in the isolated workspace, never the base tree —
    otherwise it would pollute the operator's real venv. A passing candidate
    proves setup ran in its workspace (the verifier needs the workspace .venv);
    base having no .venv proves it wasn't bootstrapped there (and .venv is
    ignored on apply, so the candidate's venv isn't mirrored back either)."""
    base = tmp_path / "base"
    _git_init(base, {"x.py": "x"})  # no .venv anywhere to start
    gens = [_SimAgent(prompt="p", write_filename="a.txt", write_content="pass")]
    verifier = _VenvVerifier()
    wf = VoteWorkflow(generators=gens, verifier=verifier, working_directory=str(base),
                      candidate_setup=_SETUP_BOOTSTRAP, preflight=False)
    _run(wf.execute("p"))
    assert wf.verified is True  # candidate bootstrapped + passed in its workspace
    # Base was never bootstrapped, and the candidate .venv wasn't mirrored back.
    assert not (base / ".venv").exists()

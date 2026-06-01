"""Issue #37 — master/pat checkpoint resume must verify the out-dir base.

The #31 salvage checkpoint is keyed only by ``sha256(prompt)``. A re-dispatch of
the same instruction therefore resumes "from the last completed step" even when
the out-dir was ``git reset --hard`` back to a clean baseline between runs —
silently building later steps on a tree missing the earlier steps' (uncommitted)
edits.

Fix: the checkpoint records a base fingerprint (HEAD + a hash of
``git status --porcelain``) at save time, and ``_load_checkpoint`` re-fingerprints
the out-dir on resume. Default policy ("auto") resumes only when the fingerprint
still matches; on divergence it DISCARDS the stale checkpoint and starts fresh.
``--resume`` (force) overrides; ``--fresh`` (never) ignores the checkpoint outright.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agy_orchestrator.workflows.master import MasterWorkflow


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)


def _repo(tmp_path: Path) -> Path:
    base = tmp_path / "repo"
    base.mkdir()
    _git(base, "init", "-q", "-b", "main")
    _git(base, "config", "user.email", "t@t")
    _git(base, "config", "user.name", "t")
    (base / "orig.py").write_text("base\n")
    _git(base, "add", "-A")
    _git(base, "commit", "-q", "-m", "init")
    return base


class _DummyAgent:
    def __init__(self, *a, **k):
        pass


def _wf(work_dir: Path, ckpt: Path, *, resume_policy: str = "auto") -> MasterWorkflow:
    return MasterWorkflow(
        model="m",
        effort="high",
        agent_class=_DummyAgent,
        working_directory=str(work_dir),
        checkpoint_path=str(ckpt),
        resume_policy=resume_policy,
    )


PROMPT = "build the thing"
TASKS = ["step a", "step b", "step c"]


# --- fingerprint primitive --- #

def test_base_fingerprint_none_for_non_git(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    wf = _wf(plain, tmp_path / "c.json")
    assert wf._base_fingerprint() is None


def test_base_fingerprint_changes_with_tree(tmp_path):
    base = _repo(tmp_path)
    wf = _wf(base, tmp_path / "c.json")
    fp_clean = wf._base_fingerprint()
    assert fp_clean and ":" in fp_clean
    # An uncommitted edit (what a completed master step leaves) flips it.
    (base / "orig.py").write_text("base\nstep-edit\n")
    fp_dirty = wf._base_fingerprint()
    assert fp_dirty != fp_clean
    # Reverting (git reset --hard equivalent) restores the original fingerprint.
    _git(base, "checkout", "--", "orig.py")
    assert wf._base_fingerprint() == fp_clean


# --- resume gating --- #

def test_resume_when_base_matches(tmp_path):
    base = _repo(tmp_path)
    ckpt = tmp_path / "c.json"
    # A completed step left an uncommitted edit; checkpoint captures that state.
    (base / "step1.py").write_text("done by step 1\n")
    _wf(base, ckpt)._save_checkpoint(PROMPT, TASKS, 1, "ctx", "sess")
    # Same tree on resume -> resumes at the saved step.
    resumed = _wf(base, ckpt)._load_checkpoint(PROMPT)
    assert resumed is not None
    tasks, completed, ctx, sess = resumed
    assert (tasks, completed, ctx, sess) == (TASKS, 1, "ctx", "sess")


def test_diverged_base_starts_fresh_by_default(tmp_path):
    base = _repo(tmp_path)
    ckpt = tmp_path / "c.json"
    (base / "step1.py").write_text("done by step 1\n")
    _wf(base, ckpt)._save_checkpoint(PROMPT, TASKS, 1, "ctx", "sess")
    # Operator resets the tree (the completed step's edit is gone).
    (base / "step1.py").unlink()
    # Default "auto": refuse to silently resume -> start fresh.
    assert _wf(base, ckpt)._load_checkpoint(PROMPT) is None


def test_force_resume_on_divergence(tmp_path):
    base = _repo(tmp_path)
    ckpt = tmp_path / "c.json"
    (base / "step1.py").write_text("done by step 1\n")
    _wf(base, ckpt)._save_checkpoint(PROMPT, TASKS, 1, "ctx", "sess")
    (base / "step1.py").unlink()
    # --resume forces resume despite the diverged tree.
    resumed = _wf(base, ckpt, resume_policy="force")._load_checkpoint(PROMPT)
    assert resumed is not None
    assert resumed[1] == 1


def test_fresh_ignores_checkpoint_even_when_matching(tmp_path):
    base = _repo(tmp_path)
    ckpt = tmp_path / "c.json"
    (base / "step1.py").write_text("done by step 1\n")
    _wf(base, ckpt)._save_checkpoint(PROMPT, TASKS, 1, "ctx", "sess")
    # --fresh: never resume, even though the tree matches.
    assert _wf(base, ckpt, resume_policy="never")._load_checkpoint(PROMPT) is None


def test_legacy_checkpoint_without_fingerprint_resumes(tmp_path):
    # A pre-#37 checkpoint has no base_fingerprint key -> can't verify -> resume on
    # trust (preserves #31 salvage behavior), just loudly.
    import json
    base = _repo(tmp_path)
    ckpt = tmp_path / "c.json"
    key = _wf(base, ckpt)._checkpoint_key(PROMPT)
    ckpt.write_text(json.dumps({
        "key": key, "tasks": TASKS, "completed": 2,
        "project_context": "ctx", "session_id": "sess",
    }))
    resumed = _wf(base, ckpt)._load_checkpoint(PROMPT)
    assert resumed is not None and resumed[1] == 2


def test_invalid_resume_policy_rejected(tmp_path):
    with pytest.raises(ValueError):
        _wf(tmp_path, tmp_path / "c.json", resume_policy="bogus")

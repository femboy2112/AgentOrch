"""Phase 1 — ``--git-pr`` preflight + worktree setup wired into dispatch.

These are hermetic: the workflow is replaced with an async stub that just writes
a file into its ``working_directory``, so no real worker/network runs. We assert
the dispatch (a) refuses unsafe trees, (b) runs the worker in an ISOLATED
worktree, (c) leaves the operator's checkout untouched, and (d) lands the work on
the temp branch with a persisted session.
"""
from __future__ import annotations

import subprocess
import types
from pathlib import Path

import pytest

from harness import dispatch as dispatch_mod
from harness import gitpr


def _git(repo, *args):
    cp = subprocess.run(["git", "-C", str(repo), *args],
                        capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr
    return cp.stdout


def _init_repo(tmp_path) -> Path:
    repo = tmp_path / "target"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "dev@example.com")
    _git(repo, "config", "user.name", "Dev")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _dispatch(repo, monkeypatch, *, writer=None, verified=False, **kw):
    """Run a git-pr dispatch with the workflow stubbed out."""
    captured = {}

    async def fake_run_workflow(mode, prompt, **kwargs):
        wd = Path(kwargs["working_directory"])
        captured["work_dir"] = wd
        if writer is not None:
            writer(wd)
        return "stub-output", types.SimpleNamespace(verified=verified)

    monkeypatch.setattr(dispatch_mod, "_run_workflow", fake_run_workflow)
    result = dispatch_mod.dispatch(
        "add the feature", git_pr=True, out_dir=str(repo), mode="direct",
        generator_chain=["codex"], fallback=False, telegram_enabled=False,
        heartbeat_interval=0, **kw,
    )
    return result, captured


# --------------------------------------------------------------------------- #
# Preflight refusals (raise BEFORE any worker runs)
# --------------------------------------------------------------------------- #
def test_refuses_non_git_dir(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ValueError, match="git repos|git repository"):
        dispatch_mod.dispatch("x", git_pr=True, out_dir=str(plain),
                              mode="direct", generator_chain=["codex"],
                              fallback=False, telegram_enabled=False,
                              heartbeat_interval=0)


def test_refuses_dirty_tree(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "uncommitted.txt").write_text("wip\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean working tree"):
        dispatch_mod.dispatch("x", git_pr=True, out_dir=str(repo),
                              mode="direct", generator_chain=["codex"],
                              fallback=False, telegram_enabled=False,
                              heartbeat_interval=0)


def test_refuses_detached_head(tmp_path):
    repo = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "--detach")
    with pytest.raises(ValueError, match="detached"):
        dispatch_mod.dispatch("x", git_pr=True, out_dir=str(repo),
                              mode="direct", generator_chain=["codex"],
                              fallback=False, telegram_enabled=False,
                              heartbeat_interval=0)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_runs_in_isolated_worktree_not_the_checkout(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    result, captured = _dispatch(
        repo, monkeypatch,
        writer=lambda wd: (wd / "feature.py").write_text("print(1)\n", encoding="utf-8"),
    )
    wd = captured["work_dir"]
    # The worker ran in a throwaway worktree, NOT the operator's repo.
    assert wd != repo
    assert "agentorch-gitpr" in str(wd)
    # The operator's own checkout never moved and stays clean.
    assert gitpr.current_branch(repo) == "main"
    assert gitpr.is_dirty(repo) is False
    # The worktree dir is torn down after the run (branch persists, see below).
    assert not wd.exists()


def test_work_lands_on_temp_branch(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    result, _ = _dispatch(
        repo, monkeypatch,
        writer=lambda wd: (wd / "feature.py").write_text("x\n", encoding="utf-8"),
    )
    branch = gitpr.branch_name_for_run(result.run_id)
    # The temp branch exists in the operator's repo with the worker's commit.
    log = _git(repo, "log", "--oneline", branch)
    assert result.run_id in log  # commit message embeds the run id
    files = _git(repo, "show", "--name-only", "--format=", branch)
    assert "feature.py" in files
    # main is unchanged — no feature.py on the base branch.
    assert "feature.py" not in _git(repo, "ls-tree", "-r", "--name-only", "main")


def test_session_persisted(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    result, _ = _dispatch(
        repo, monkeypatch, verified=True,
        writer=lambda wd: (wd / "f.py").write_text("y\n", encoding="utf-8"),
    )
    sess = gitpr.load_session(Path(result.run_dir))
    assert sess is not None
    assert sess.status == "branch_ready"
    assert sess.base_branch == "main"
    assert sess.temp_branch == gitpr.branch_name_for_run(result.run_id)
    assert sess.verified is True
    assert len(sess.commits) == 1
    assert sess.commits[0]["outcome"] == "verified"


def test_no_changes_makes_no_commit(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    result, _ = _dispatch(repo, monkeypatch, writer=None)  # worker writes nothing
    sess = gitpr.load_session(Path(result.run_dir))
    assert sess.status == "branch_ready"
    assert sess.commits == []  # nothing to commit -> no empty commit
    # Branch still exists (created at base), just with no new commits.
    assert gitpr.head_sha(repo) == _git(
        repo, "rev-parse", gitpr.branch_name_for_run(result.run_id)).strip()


# --------------------------------------------------------------------------- #
# Back-compat: without --git-pr nothing git-pr happens
# --------------------------------------------------------------------------- #
def test_without_flag_no_session_no_branch(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)

    async def fake_run_workflow(mode, prompt, **kwargs):
        # default path: worker writes straight into the operator's out_dir
        (Path(kwargs["working_directory"]) / "inplace.py").write_text("z\n",
                                                                       encoding="utf-8")
        return "out", types.SimpleNamespace(verified=False)

    monkeypatch.setattr(dispatch_mod, "_run_workflow", fake_run_workflow)
    result = dispatch_mod.dispatch(
        "x", out_dir=str(repo), mode="direct", generator_chain=["codex"],
        fallback=False, telegram_enabled=False, heartbeat_interval=0,
    )
    # No pr_session.json, no agentorch/* branch, write landed in the checkout.
    assert gitpr.load_session(Path(result.run_dir)) is None
    branches = _git(repo, "branch", "--list", "agentorch/*")
    assert branches.strip() == ""
    assert (repo / "inplace.py").exists()

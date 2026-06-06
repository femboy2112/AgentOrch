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


# --------------------------------------------------------------------------- #
# Phase 2 — per-accepted-step commits (linear master) + graph gate
# --------------------------------------------------------------------------- #
def _run_with_step_events(repo, monkeypatch, run_id, steps, *, plan_graph=None):
    """Stub a master-like run: write each step's file, then emit its
    step-completed orchestration event through the bus (which the git-pr per-step
    sink listens to). ``steps`` = list of (filename, outcome)."""
    async def fake(mode, prompt, **kwargs):
        wd = Path(kwargs["working_directory"])
        pub = dispatch_mod.EVENT_BUS.publisher_for(
            run_id, worker="codex", model="m", effort="hi")
        total = len(steps)
        for i, (fname, outcome) in enumerate(steps, start=1):
            (wd / fname).write_text(f"# {fname}\n", encoding="utf-8")
            pub({"kind": "lifecycle", "data": {
                "event": "orchestration_transition",
                "orchestration": {
                    "workflow": "master", "phase": "step", "action": "completed",
                    "step_index": i, "step_total": total,
                    "step_title": f"do {fname}", "outcome": outcome,
                }}})
        return "out", types.SimpleNamespace(verified=True)

    monkeypatch.setattr(dispatch_mod, "_run_workflow", fake)
    return dispatch_mod.dispatch(
        "multi step build", git_pr=True, out_dir=str(repo), run_id=run_id,
        mode="master", generator_chain=["codex"], fallback=False,
        telegram_enabled=False, heartbeat_interval=0, plan_graph=plan_graph,
    )


def test_linear_per_step_commits(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    result = _run_with_step_events(
        repo, monkeypatch, "p2-linear",
        [("a.py", "verified"), ("b.py", "approved")])
    branch = gitpr.branch_name_for_run("p2-linear")
    subjects = _git(repo, "log", "--format=%s", branch).splitlines()
    assert "step 1/2: do a.py [verified]" in subjects
    assert "step 2/2: do b.py [approved]" in subjects
    sess = gitpr.load_session(Path(result.run_dir))
    assert [c["step"] for c in sess.commits] == [1, 2]
    assert [c["outcome"] for c in sess.commits] == ["verified", "approved"]
    # each step's commit holds exactly that step's file
    f1 = _git(repo, "show", "--name-only", "--format=", sess.commits[0]["sha"])
    assert "a.py" in f1 and "b.py" not in f1
    # nothing left over -> the finalize sweep added no extra commit
    assert len(sess.commits) == 2


def test_unaccepted_step_swept_not_committed_per_step(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    result = _run_with_step_events(
        repo, monkeypatch, "p2-stall",
        [("a.py", "verified"), ("b.py", "stalled")])
    sess = gitpr.load_session(Path(result.run_dir))
    step_commits = [c for c in sess.commits if "step" in c]
    assert len(step_commits) == 1 and step_commits[0]["step"] == 1
    # b.py (the non-accepted step's output) still reaches the branch via the
    # finalize sweep — work is never lost, it's just not a per-step commit.
    branch = gitpr.branch_name_for_run("p2-stall")
    assert "b.py" in _git(repo, "ls-tree", "-r", "--name-only", branch)
    assert len(sess.commits) == 2  # 1 step commit + 1 sweep commit


def test_graph_run_uses_single_finalize_commit(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    from agy_orchestrator.execution.graph_plan import GraphPlan, PlanNode
    gp = GraphPlan(nodes=[PlanNode(id="n1", task="t1"),
                          PlanNode(id="n2", task="t2")])
    result = _run_with_step_events(
        repo, monkeypatch, "p2-graph",
        [("a.py", "verified"), ("b.py", "verified")], plan_graph=gp)
    sess = gitpr.load_session(Path(result.run_dir))
    # Graph gate: no per-step commits; the merged result is one finalize commit.
    assert all("step" not in c for c in sess.commits)
    assert len(sess.commits) == 1
    branch = gitpr.branch_name_for_run("p2-graph")
    files = _git(repo, "ls-tree", "-r", "--name-only", branch)
    assert "a.py" in files and "b.py" in files

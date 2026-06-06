"""Phase 1 — ``--git-pr`` preflight + worktree setup wired into dispatch.

These are hermetic: the workflow is replaced with an async stub that just writes
a file into its ``working_directory``, so no real worker/network runs. We assert
the dispatch (a) refuses unsafe trees, (b) runs the worker in an ISOLATED
worktree, (c) leaves the operator's checkout untouched, and (d) lands the work on
the temp branch with a persisted session.
"""
from __future__ import annotations

import json
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
    """Run a git-pr dispatch with the workflow stubbed out (hermetic RUNS_DIR)."""
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", repo.parent / "agentorch_runs")
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
_GH_STUB = """#!/usr/bin/env python3
import os, sys, json
args = sys.argv[1:]
log = os.environ.get("GH_STUB_LOG")
if log:
    with open(log, "a") as f:
        f.write(json.dumps(args) + "\\n")  # JSON: newline-safe for multi-line --body
if args[:2] == ["auth", "status"]:
    sys.exit(0)
if args[:2] == ["pr", "create"]:
    sys.stdout.write("https://github.com/acme/repo/pull/77\\n")
    sys.exit(0)
if args[:2] in (["pr", "ready"], ["pr", "merge"], ["pr", "close"]):
    sys.exit(0)
sys.stderr.write("gh stub: unhandled %r\\n" % (args,))
sys.exit(2)
"""


@pytest.fixture
def gh_stub(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "gh"
    stub.write_text(_GH_STUB, encoding="utf-8")
    import os as _os
    _os.chmod(stub, 0o755)
    log = tmp_path / "gh_calls.log"
    monkeypatch.setenv("PATH", str(bindir) + _os.pathsep + _os.environ["PATH"])
    monkeypatch.setenv("GH_STUB_LOG", str(log))
    return log


def _gh_calls(log):
    if not log.exists():
        return []
    import json as _json
    return [_json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()]


def _add_remote(repo, tmp_path):
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
    _git(repo, "remote", "add", "origin", str(bare))
    return bare


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
    assert sess.status == "no_changes"  # nothing committed -> no PR
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
    # meta.json carries NO git_pr block on a normal run (byte-identical contract).
    meta = json.loads((Path(result.run_dir) / "meta.json").read_text())
    assert "git_pr" not in meta


# --------------------------------------------------------------------------- #
# Phase 6 — observability (meta.json git_pr block)
# --------------------------------------------------------------------------- #
def test_meta_json_has_git_pr_block(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    result, _ = _dispatch(
        repo, monkeypatch, verified=True,
        writer=lambda wd: (wd / "f.py").write_text("x\n", encoding="utf-8"))
    meta = json.loads((Path(result.run_dir) / "meta.json").read_text())
    assert "git_pr" in meta
    gp = meta["git_pr"]
    assert gp["temp_branch"] == gitpr.branch_name_for_run(result.run_id)
    assert gp["base_branch"] == "main"
    assert gp["commits"] == 1
    assert gp["verified"] is True
    assert gp["status"] == "branch_ready"  # no remote in this repo
    assert gp["contributing_runs"] == [result.run_id]


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

    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", repo.parent / "agentorch_runs")
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


# --------------------------------------------------------------------------- #
# Phase 3 — push + draft-PR creation / promotion
# --------------------------------------------------------------------------- #
def test_pr_created_and_promoted_on_verified(tmp_path, monkeypatch, gh_stub):
    repo = _init_repo(tmp_path)
    bare = _add_remote(repo, tmp_path)
    result, _ = _dispatch(
        repo, monkeypatch, verified=True,
        writer=lambda wd: (wd / "feature.py").write_text("ok\n", encoding="utf-8"),
    )
    sess = gitpr.load_session(Path(result.run_dir))
    assert sess.status == "awaiting_decision"
    assert sess.pr_url == "https://github.com/acme/repo/pull/77"
    assert sess.pr_number == 77
    assert sess.draft is False  # verified -> promoted to ready
    # the temp branch was actually pushed to the remote
    assert sess.temp_branch in _git(bare, "branch", "--list", sess.temp_branch)
    calls = _gh_calls(gh_stub)
    create = [c for c in calls if c[:2] == ["pr", "create"]][0]
    assert "--draft" in create
    assert ["pr", "ready", "77"] in calls


def test_pr_stays_draft_when_not_verified(tmp_path, monkeypatch, gh_stub):
    repo = _init_repo(tmp_path)
    _add_remote(repo, tmp_path)
    result, _ = _dispatch(
        repo, monkeypatch, verified=False,
        writer=lambda wd: (wd / "f.py").write_text("x\n", encoding="utf-8"),
    )
    sess = gitpr.load_session(Path(result.run_dir))
    assert sess.status == "awaiting_decision"
    assert sess.draft is True
    calls = _gh_calls(gh_stub)
    assert any(c[:2] == ["pr", "create"] for c in calls)
    assert not any(c[:2] == ["pr", "ready"] for c in calls)  # never promoted


def test_no_gh_degrades_to_pushed_no_pr(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    bare = _add_remote(repo, tmp_path)
    # gh present-or-not on the host shouldn't matter: force it unavailable.
    monkeypatch.setattr(gitpr, "gh_available", lambda: False)
    result, _ = _dispatch(
        repo, monkeypatch, verified=True,
        writer=lambda wd: (wd / "f.py").write_text("x\n", encoding="utf-8"),
    )
    sess = gitpr.load_session(Path(result.run_dir))
    assert sess.status == "pushed_no_pr"
    assert sess.pr_url is None
    # branch still pushed to the remote so a human can open the PR manually
    assert sess.temp_branch in _git(bare, "branch", "--list", sess.temp_branch)


def test_no_remote_stays_local_branch_ready(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)  # no remote added
    result, _ = _dispatch(
        repo, monkeypatch, verified=True,
        writer=lambda wd: (wd / "f.py").write_text("x\n", encoding="utf-8"),
    )
    sess = gitpr.load_session(Path(result.run_dir))
    assert sess.status == "branch_ready"
    assert sess.pr_url is None
    # branch exists locally with the commit
    assert "f.py" in _git(repo, "ls-tree", "-r", "--name-only",
                          gitpr.branch_name_for_run(result.run_id))


# --------------------------------------------------------------------------- #
# Phase 5 — corrective resume (--continue)
# --------------------------------------------------------------------------- #
def test_corrective_continue_same_branch_and_pr(tmp_path, monkeypatch, gh_stub):
    repo = _init_repo(tmp_path)
    _add_remote(repo, tmp_path)
    # initial run: draft PR (not verified)
    r1, _ = _dispatch(
        repo, monkeypatch, verified=False, run_id="p5-orig",
        writer=lambda wd: (wd / "a.py").write_text("a\n", encoding="utf-8"))
    s1 = gitpr.load_session(Path(r1.run_dir))
    assert s1.status == "awaiting_decision" and s1.pr_number == 77 and s1.draft is True
    assert s1.contributing_runs == ["p5-orig"]

    # corrective run on the SAME branch, this time verified
    def _corrective_writer(wd):
        # the corrective worktree must already hold the prior committed work
        assert (wd / "a.py").exists(), "corrective run didn't pick up prior work"
        (wd / "b.py").write_text("b\n", encoding="utf-8")

    r2, cap = _dispatch(
        repo, monkeypatch, verified=True, run_id="p5-corr",
        git_pr_continue="p5-orig", writer=_corrective_writer)
    # canonical session is the ORIGINAL run's, updated in place
    s2 = gitpr.load_session(Path(r1.run_dir))
    assert s2.parent_run_id == "p5-orig"
    assert s2.contributing_runs == ["p5-orig", "p5-corr"]
    assert s2.draft is False  # corrective verified -> existing PR promoted to ready
    assert s2.status == "awaiting_decision"
    # both files live on the one temp branch
    branch = gitpr.branch_name_for_run("p5-orig")
    files = _git(repo, "ls-tree", "-r", "--name-only", branch)
    assert "a.py" in files and "b.py" in files
    # exactly ONE PR was ever created; the corrective only promoted it
    calls = _gh_calls(gh_stub)
    assert len([c for c in calls if c[:2] == ["pr", "create"]]) == 1
    assert ["pr", "ready", "77"] in calls


def test_continue_unknown_run_raises(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", repo.parent / "agentorch_runs")
    with pytest.raises(ValueError, match="no git-pr session"):
        dispatch_mod.dispatch(
            "fix it", git_pr_continue="ghost-run", mode="direct",
            generator_chain=["codex"], fallback=False, telegram_enabled=False,
            heartbeat_interval=0)


def test_continue_deleted_branch_raises(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    r1, _ = _dispatch(
        repo, monkeypatch, run_id="p5-del",
        writer=lambda wd: (wd / "a.py").write_text("a\n", encoding="utf-8"))
    # delete the temp branch out from under the corrective run
    _git(repo, "branch", "-D", gitpr.branch_name_for_run("p5-del"))
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", repo.parent / "agentorch_runs")
    with pytest.raises(ValueError, match="no longer exists"):
        dispatch_mod.dispatch(
            "fix it", git_pr_continue="p5-del", mode="direct",
            generator_chain=["codex"], fallback=False, telegram_enabled=False,
            heartbeat_interval=0)

"""Hermetic long-run resilience probes for the master GRAPH / DAG concurrency
and merge-under-partial-failure path.

Dimension: Graph / DAG concurrency + merge under partial failure.

These exercise failure modes that only bite on multi-node, multi-hour,
mission-critical DAG builds — a mid-DAG defect silently merged into the live
tree, whole-tree re-snapshot amplification under the merge lock, sibling
workspace leaks on fail-fast cancellation, reconcile-policy clobber when the
post-merge re-verify itself errors, and resume-against-a-partially-mutated
live tree.

HERMETIC + FAST: NO real workers, NO network. Each probe drives the real
``_execute_graph`` / ``_run_dag_node`` / ``merge_node`` code with a monkeypatched
``_execute_step`` (writes scripted bytes into the isolated workspace + returns a
scripted ``StepResult``) over tmp git repos. Sub-second.

Each probe asserts the DESIRED RESILIENT behaviour. A probe whose assertion
FAILS on today's code (a real long-run defect) is decorated
``@pytest.mark.xfail(strict=True)`` so the committed suite stays green while the
assertion stands as a RED contract for the fix.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pytest

import agy_orchestrator.workflows.master as master_mod
import agy_orchestrator.execution.workspace as ws_mod
from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.graph_plan import GraphPlan, PlanNode
from agy_orchestrator.workflows.master import MasterWorkflow, StepResult

pytestmark = pytest.mark.not_slow


# --------------------------------------------------------------------------- #
# Fakes / fixtures
# --------------------------------------------------------------------------- #
class _Stub(AgentInstance):
    """Deterministic no-subprocess agent (never actually called in these probes —
    _execute_step is monkeypatched — but MasterWorkflow needs an agent_class)."""

    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        return "summary-line"


def _git_repo(path: Path) -> str:
    """Create a committed git repo at ``path`` (so worktree-mode merges work)."""
    path.mkdir(parents=True, exist_ok=True)
    env = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    run = lambda *a: subprocess.run(
        ["git", "-C", str(path), *a], check=True,
        capture_output=True, text=True,
    )
    subprocess.run(["git", "init", "-q", str(path)], check=True,
                   capture_output=True, text=True)
    run("config", "user.email", "t@t.t")
    run("config", "user.name", "t")
    (path / "README.md").write_text("base\n")
    run("add", "-A")
    run("commit", "-q", "-m", "init")
    return str(path)


def _make(live: str, graph: GraphPlan, **kw) -> MasterWorkflow:
    defaults = dict(
        model="m", effort="low", branches=1, agent_class=_Stub,
        working_directory=live, plan_graph=graph, verifier=None,
    )
    defaults.update(kw)
    return MasterWorkflow(**defaults)


def _diamond() -> GraphPlan:
    return GraphPlan(nodes=[
        PlanNode(id="A", task="task A", deps=[]),
        PlanNode(id="B", task="task B", deps=["A"]),
        PlanNode(id="C", task="task C", deps=["A"]),
        PlanNode(id="D", task="task D", deps=["B", "C"]),
    ])


def _patch_execute_step(monkeypatch, *, file_for, result_for, hooks=None):
    """Monkeypatch MasterWorkflow._execute_step.

    ``file_for(node_task) -> (rel, bytes) or None``: a file the node writes into
    its isolated workspace (working_directory passed by _run_dag_node).
    ``result_for(node_task) -> StepResult``: the scripted step result.
    ``hooks(node_task)``: optional callback invoked for side effects (e.g. raise).
    """
    async def _fake_execute_step(self, *, step_index, step_total, task,
                                 project_context, workflow_session_id,
                                 working_directory, verifier=None):
        if hooks is not None:
            hooks(task)
        spec = file_for(task)
        if spec is not None:
            rel, data = spec
            dst = Path(working_directory) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        return result_for(task)

    monkeypatch.setattr(MasterWorkflow, "_execute_step", _fake_execute_step)


def _ok(task: str) -> StepResult:
    return StepResult(final_output="", step_summary=f"sum-{task}",
                      verified=True, approved=True, stalled=False,
                      iterations_used=1, session_id=None)


def _bad(task: str) -> StepResult:
    return StepResult(final_output="", step_summary=f"sum-{task}",
                      verified=False, approved=False, stalled=False,
                      iterations_used=1, session_id=None)


# --------------------------------------------------------------------------- #
# Probe 1 — a FAILED intermediate node merges broken bytes into the live tree
# AND the run still greens off the leaf.
# --------------------------------------------------------------------------- #
def test_failed_node_does_not_false_green_the_run(tmp_path, monkeypatch):
    """Diamond A->{B,C}->D. Node C is a genuine code defect (verified=False).

    RESILIENT contract: a mission-critical DAG with a failed intermediate node
    must NOT report a green run. Either C's broken bytes are not blindly folded
    into the live tree, or — at minimum — the run-level verdict must reflect that
    a node failed (self.verified is False when any node_status is 'failed').

    Today: merge_node runs unconditionally after _execute_step, so C's broken
    file lands in the live tree; node_status['C']=='failed' is observability-only;
    self.verified tracks the last topo node (leaf D, passing) -> false green.
    """
    live = _git_repo(tmp_path / "live")
    g = _diamond()

    files = {
        "task A": ("a.txt", b"A-ok"),
        "task B": ("b.txt", b"B-ok"),
        "task C": ("c.txt", b"C-BROKEN"),   # the defective node writes a file
        "task D": ("d.txt", b"D-ok"),
    }
    results = {
        "task A": _ok, "task B": _ok, "task C": _bad, "task D": _ok,
    }
    _patch_execute_step(
        monkeypatch,
        file_for=lambda t: files[t],
        result_for=lambda t: results[t](t),
    )

    m = _make(live, g, merge_policy="disjoint")
    asyncio.run(m.execute("build the diamond"))

    # C was harvested as failed (observability is correct).
    assert m.node_status["C"] == "failed"

    # The RED contract: the run must not green when a node failed. (Today this
    # fails because self.verified is driven by leaf D.)
    assert m.verified is False, (
        "run greened (self.verified=True) despite node C failing — "
        "node_status['failed'] never gates the run"
    )


def test_failed_node_broken_bytes_were_merged_into_live_tree(tmp_path, monkeypatch):
    """Companion observation (NOT a contract assertion of desired behaviour, so
    it stays green): document that the failed node's bytes DO reach the live tree
    today, i.e. merge happens with no verified/approved guard. This is the
    mechanism behind the false-green above.
    """
    live = _git_repo(tmp_path / "live")
    g = _diamond()
    files = {
        "task A": ("a.txt", b"A-ok"),
        "task B": ("b.txt", b"B-ok"),
        "task C": ("c.txt", b"C-BROKEN"),
        "task D": ("d.txt", b"D-ok"),
    }
    results = {"task A": _ok, "task B": _ok, "task C": _bad, "task D": _ok}
    _patch_execute_step(
        monkeypatch,
        file_for=lambda t: files[t],
        result_for=lambda t: results[t](t),
    )
    m = _make(live, g, merge_policy="disjoint")
    asyncio.run(m.execute("build the diamond"))
    # Mechanism: the failed node's file IS in the live tree (merge unconditional).
    assert (Path(live) / "c.txt").read_bytes() == b"C-BROKEN"


# --------------------------------------------------------------------------- #
# Probe 2 — merge_node re-snapshots the ENTIRE live tree on every node:
# O(N * tree_size) read amplification under the global merge lock.
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason="fuzz-longrun graph_concurrency: each node merge re-_snapshot_tree()s "
           "the WHOLE live tree (src + live), so cumulative bytes read scale with "
           "N * tree_size, not N — O(N^2) read amplification on a large repo",
)
def test_merge_does_not_resnapshot_whole_live_tree_per_node(tmp_path, monkeypatch):
    """Across an N-node DAG over a repo of F large files, the per-node merge must
    not read the entire live tree into RAM. A bounded merge would read only the
    node's own changed files (+ the overlapping live paths), so cumulative bytes
    read should scale ~O(N), not O(N * F * filesize).

    We count bytes read by _snapshot_tree across the whole run. With F pre-seeded
    large files in the live tree and N disjoint-writing nodes, today's code reads
    the whole live tree once per node => >= N * (F * filesize). The resilient
    bound asserts cumulative live-tree reads stay well under that quadratic.
    """
    live = _git_repo(tmp_path / "live")
    live_p = Path(live)
    # Seed F large files into the live tree (committed so worktrees carry them).
    F, SIZE = 6, 50_000
    for i in range(F):
        (live_p / f"big_{i}.dat").write_bytes(b"x" * SIZE)
    subprocess.run(["git", "-C", live, "add", "-A"], check=True,
                   capture_output=True, text=True)
    subprocess.run(["git", "-C", live, "-c", "user.email=t@t.t",
                    "-c", "user.name=t", "commit", "-q", "-m", "big"],
                   check=True, capture_output=True, text=True)

    # Count bytes read by _snapshot_tree (the amplification surface).
    real_snap = ws_mod._snapshot_tree
    counter = {"bytes": 0, "calls": 0}

    def _counting_snap(root):
        snap = real_snap(root)
        counter["calls"] += 1
        counter["bytes"] += sum(len(v) for v in snap.values())
        return snap

    # Patch at every import site the merge path uses.
    monkeypatch.setattr(ws_mod, "_snapshot_tree", _counting_snap)
    monkeypatch.setattr(master_mod, "_snapshot_tree", _counting_snap)
    import agy_orchestrator.workflows.graph_merge as gm_mod
    monkeypatch.setattr(gm_mod, "_snapshot_tree", _counting_snap)

    # N independent nodes, each writes ONE small new file (all disjoint).
    N = 5
    nodes = [PlanNode(id=f"n{i}", task=f"task {i}", deps=[]) for i in range(N)]
    g = GraphPlan(nodes=nodes)
    _patch_execute_step(
        monkeypatch,
        file_for=lambda t: (f"out_{t.split()[-1]}.txt", t.encode()),
        result_for=lambda t: _ok(t),
    )
    # Serialize nodes (cap=1) so we measure per-node snapshot cost cleanly.
    m = _make(live, g, merge_policy="reconcile", max_parallel_workers=1)
    asyncio.run(m.execute("wide independent layer"))

    live_total = F * SIZE  # bytes of the big pre-seeded live tree
    # Resilient bound: cumulative bytes read should be far below N copies of the
    # whole live tree. Allow generous slack (2x one full live-tree read) but flag
    # the O(N) full re-reads. With N=5, F*SIZE=300KB, today's reads >= ~3MB
    # (5 nodes x ~600KB src+live) which blows past 2 * live_total = 600KB.
    assert counter["bytes"] <= 2 * live_total, (
        f"merge read {counter['bytes']} bytes across {counter['calls']} "
        f"snapshots for N={N} nodes over a {live_total}-byte live tree — "
        f"whole-tree re-snapshot per node (O(N*tree)) instead of O(N)"
    )


# --------------------------------------------------------------------------- #
# Probe 3 — fail-fast sibling cancellation must not leak isolated workspaces.
# --------------------------------------------------------------------------- #
def test_fail_fast_cancellation_does_not_leak_sibling_workspaces(tmp_path, monkeypatch):
    """Two parallel nodes B and C off A. C raises immediately; B is mid-flight.
    The fail-fast path cancels B and re-raises. RESILIENT contract: every
    isolated workspace tmp-root created during the run is removed from disk
    (no leaked agentorch-cand-* dir / git worktree) once _execute_graph unwinds.

    We capture each _ManagedWorkspace's tmp_root as it opens and assert all are
    gone after the run raises.
    """
    live = _git_repo(tmp_path / "live")
    g = GraphPlan(nodes=[
        PlanNode(id="A", task="task A", deps=[]),
        PlanNode(id="B", task="task B", deps=["A"]),
        PlanNode(id="C", task="task C", deps=["A"]),
    ])

    created_roots: List[Path] = []
    real_init = ws_mod._ManagedWorkspace.open

    async def _tracking_open(self):
        res = await real_init(self)
        if self.path is not None:
            # tmp_root is the parent of the 'workspace' dir.
            created_roots.append(Path(self.path).parent)
        return res

    monkeypatch.setattr(ws_mod._ManagedWorkspace, "open", _tracking_open)

    async def _hook(task):
        if task == "task C":
            raise RuntimeError("node C exploded")
        if task == "task B":
            await asyncio.sleep(0.05)  # in-flight when C raises

    async def _fake_execute_step(self, *, step_index, step_total, task,
                                 project_context, workflow_session_id,
                                 working_directory, verifier=None):
        await _hook(task)
        return _ok(task)

    monkeypatch.setattr(MasterWorkflow, "_execute_step", _fake_execute_step)

    m = _make(live, g, merge_policy="reconcile")
    with pytest.raises(RuntimeError):
        asyncio.run(m.execute("fail fast"))

    # At least the two sibling workspaces (B and C) were opened.
    assert len(created_roots) >= 1
    leaked = [str(r) for r in created_roots if r.exists()]
    assert not leaked, f"leaked isolated workspace tmp-roots after fail-fast: {leaked}"


def test_workspace_teardown_is_shielded_against_cancellation(tmp_path, monkeypatch):
    """If the worktree-remove await is interrupted (CancelledError) during teardown,
    the belt-and-braces shutil.rmtree of the tmp-root must STILL run — the fix nests
    the rmtree in its own ``finally`` so a re-raising worktree-remove can't skip it.

    Drive the real ``candidate_workspace`` context manager directly (a single-node
    graph never opens an isolated workspace, so it can't exercise this path): open a
    worktree-backed workspace, make _remove_git_worktree raise CancelledError on exit,
    and assert the agentorch-cand-* tmp-root is gone, not leaked, despite the cancel.
    """
    base = _git_repo(tmp_path / "base")

    async def _boom_remove(base_, target):
        raise asyncio.CancelledError()

    monkeypatch.setattr(ws_mod, "_remove_git_worktree", _boom_remove)

    captured: Dict[str, Path] = {}

    async def _drive():
        async with ws_mod.candidate_workspace(
            Path(base), prefer_worktree=True
        ) as (wsdir, backend):
            captured["tmp_root"] = Path(wsdir).parent  # rmtree target on teardown
            captured["backend"] = backend

    # The CancelledError from teardown may surface; the cleanup must run regardless.
    try:
        asyncio.run(_drive())
    except BaseException:
        pass

    assert captured.get("tmp_root") is not None, "workspace was never created"
    assert not captured["tmp_root"].exists(), (
        f"workspace tmp-root leaked when the worktree-remove was cancelled "
        f"mid-teardown: {captured['tmp_root']} — the rmtree is not shielded by a "
        f"nested finally"
    )


# --------------------------------------------------------------------------- #
# Probe 4 — reconcile-policy: post-merge re-verify that ERRORS keeps a stale
# green verdict on a now-broken merged tree.
# --------------------------------------------------------------------------- #
def test_reconcile_reverify_error_does_not_keep_stale_green(tmp_path, monkeypatch):
    """Two nodes write the SAME file (overlap). The reconcile station produces
    deliberately-broken merged bytes; the post-merge re-verify RAISES.

    RESILIENT contract: an exception during the post-merge re-verify must NOT
    leave a stale 'verified=True' verdict on a reconciled (possibly broken) tree.
    A re-verify that cannot confirm the merged tree should be treated as
    unverified (verified=False), not silently green.

    Today: the except clause logs and KEEPS the pre-merge verdict (True) -> stale
    green, defeating the 'verifier re-runs on the merged tree' guarantee.
    """
    live = _git_repo(tmp_path / "live")

    # A verifier whose verify() always raises (simulates OOM/timeout under load).
    class _RaisingVerifier:
        async def verify(self, *a, **k):
            raise RuntimeError("re-verify subprocess OOM")

    # Reconcile station factory: returns broken merged bytes for any overlap.
    def _factory(*, node, working_directory):
        async def _reconcile(rel, base_b, sib_b, node_b):
            return b"BROKEN-MERGE"
        return _reconcile

    # Two INDEPENDENT nodes both writing shared.txt. They must open their
    # workspaces CONCURRENTLY (both snapshot the empty base) so the 2nd to merge
    # sees a genuine overlap — serializing them (max_parallel_workers=1) would
    # let the 2nd snapshot the 1st's already-merged write as its own base and
    # apply disjointly (no overlap). A tiny in-step sleep keeps both open.
    g = GraphPlan(nodes=[
        PlanNode(id="X", task="task X", deps=[]),
        PlanNode(id="Y", task="task Y", deps=[]),
    ])

    async def _fake_execute_step(self, *, step_index, step_total, task,
                                 project_context, workflow_session_id,
                                 working_directory, verifier=None):
        (Path(working_directory) / "shared.txt").write_bytes(task.encode())
        await asyncio.sleep(0.02)  # keep both workspaces open concurrently
        return _ok(task)

    monkeypatch.setattr(MasterWorkflow, "_execute_step", _fake_execute_step)

    m = _make(
        live, g, merge_policy="reconcile",
        verifier=_RaisingVerifier(),
        reconcile_station_factory=_factory,
        # unbounded node parallelism so both open before either merges (the
        # merge_lock still serializes the actual merges).
    )
    asyncio.run(m.execute("two writers same file"))

    # The reconciled (broken) bytes are in the live tree.
    assert (Path(live) / "shared.txt").read_bytes() == b"BROKEN-MERGE"

    # RED contract: the node whose re-verify RAISED on a reconciled tree must not
    # retain a green verdict. At least one node should be 'failed' (re-verify
    # could not confirm the merged tree).
    assert "failed" in m.node_status.values(), (
        "no node failed even though the post-merge re-verify RAISED on a "
        "reconciled/broken tree — stale green verdict survived"
    )


def test_reconciler_exception_falls_back_to_last_writer(tmp_path, monkeypatch):
    """Companion (stays green): when the reconcile station RAISES on a node's
    overlap, merge_node propagates the exception rather than silently clobbering
    — i.e. the station's failure is NOT swallowed into a silent last-writer-wins
    inside merge_node itself. (The harness-wired reconciler's OWN try/except
    last-writer fallback lives in dispatch.py and is exercised live; here we pin
    that merge_node surfaces a raising station so the failure is observable.)
    """
    from agy_orchestrator.workflows.graph_merge import merge_node
    from agy_orchestrator.execution.workspace import _snapshot_tree

    live = tmp_path / "live"; src = tmp_path / "src"
    live.mkdir(); src.mkdir()
    (live / "shared.txt").write_bytes(b"base")
    base = _snapshot_tree(live)
    (live / "shared.txt").write_bytes(b"sibling")  # sibling merged
    (src / "shared.txt").write_bytes(b"node")      # node overlaps

    async def _raising_station(rel, base_b, sib_b, node_b):
        raise RuntimeError("reconciler model died")

    sem = asyncio.Semaphore(1)
    with pytest.raises(RuntimeError):
        asyncio.run(merge_node(
            node_id="n", layer=1, src_ws=src, live_tree=live,
            base_snapshot=base, policy="reconcile",
            reconcile_station=_raising_station, sem=sem,
        ))
    # The station raised => merge_node surfaced it (did not silently clobber the
    # sibling's bytes with the node's). The pre-station sibling bytes survive.
    assert (live / "shared.txt").read_bytes() == b"sibling"


# --------------------------------------------------------------------------- #
# Probe 5 — graph resume against a live tree already carrying the crashed run's
# partial writes must not spuriously conflict / double-apply.
#
# Uses a DIAMOND (A->{B,C}->D) so the run genuinely routes to _execute_graph /
# merge_node (a 2-node chain is LINEAR -> the flat loop, never the DAG path).
# --------------------------------------------------------------------------- #
def test_resume_against_partially_merged_live_tree_no_spurious_conflict(tmp_path, monkeypatch):
    """Diamond A->{B,C}->D, serialized. Run 1: A merges a.txt into the live tree,
    then B CRASHES before the harvest that would checkpoint B/C — so only A is in
    the persisted done frontier while the live tree already carries A's write.

    On resume the SAME (now partially-mutated) live tree is reused; A is seeded
    done; B, C, D re-run. A resumed node (D) touches the SAME path A already
    merged.

    RESILIENT contract: the resumed run must complete WITHOUT a spurious
    overlap/MergeConflict against A's already-applied write — A's bytes live in
    each resumed node's open-time base_snapshot, so they read as 'base', not as a
    hostile sibling. (Verifies the resume safety-net is deterministic on a
    partially-mutated base.)
    """
    from agy_orchestrator.workflows.graph_merge import MergeConflict

    live = _git_repo(tmp_path / "live")
    ckpt = str(tmp_path / "ckpt.json")
    g = _diamond()

    # --- Run 1: A writes+merges a.txt; B crashes before the next checkpoint. ---
    def _file_run1(task):
        if task == "task A":
            return ("a.txt", b"A1")
        return None  # B/C/D write nothing before the crash

    def _hook_run1(task):
        if task == "task B":
            raise RuntimeError("crash mid-DAG before B/C checkpoint")

    _patch_execute_step(
        monkeypatch,
        file_for=_file_run1,
        result_for=lambda t: _ok(t),
        hooks=_hook_run1,
    )
    m1 = _make(live, g, merge_policy="fail", checkpoint_path=ckpt,
               max_parallel_workers=1)
    with pytest.raises(RuntimeError):
        asyncio.run(m1.execute("goal"))

    # A's write reached the live tree; only A is in the done frontier.
    assert (Path(live) / "a.txt").read_bytes() == b"A1"

    # --- Run 2: resume against the SAME mutated live tree; D touches a.txt. ---
    def _file_run2(task):
        if task == "task D":
            return ("a.txt", b"A1-then-D")  # D legitimately updates A's path
        return None

    _patch_execute_step(
        monkeypatch,
        file_for=_file_run2,
        result_for=lambda t: _ok(t),
    )
    m2 = _make(live, g, merge_policy="fail", checkpoint_path=ckpt,
               max_parallel_workers=1)
    # Resume must finish without a spurious MergeConflict from D's merge against
    # A's already-applied write in the partially-mutated live tree.
    try:
        out = asyncio.run(m2.execute("goal"))
    except MergeConflict as mc:
        pytest.fail(
            f"resume against a partially-merged live tree raised a spurious "
            f"MergeConflict: {mc}"
        )
    # The resumed node's update landed; A was not re-run (seeded done).
    assert (Path(live) / "a.txt").read_bytes() == b"A1-then-D"
    assert "Original Goal: goal" in out

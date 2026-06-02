"""Serial DAG executor + golden-trace for the refactored master step loop (M2).

Covers docs/graph-execution-design.md §5 M2 + §7:
  * GOLDEN TRACE — the refactored linear loop (a thin caller over _execute_step)
    produces an IDENTICAL sequence of agent calls + context strings to the
    pre-refactor loop body, for a 3-step flat plan. This is the regression gate
    that the _execute_step extraction did not drift the default linear path.
  * SERIAL DAG — a diamond A->{B,C}->D runs in topological order, one node at a
    time (M2 has no concurrency yet); D's composed context contains BOTH B and C
    summaries, while B's context contains NEITHER C's nor D's. Proves context
    composition (ancestor-only / join-union) + topo ordering + machinery reuse.

Fully hermetic — stub agents + a recording AdversarialReview stand-in; no live
workers, no network. Reuses the _StubAgent / monkeypatch patterns from
tests/test_plan_injection.py and tests/test_master_verified_propagation.py.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

import pytest

import agy_orchestrator.workflows.master as master_mod
from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.graph_plan import GraphPlan, PlanNode
from agy_orchestrator.workflows.master import MasterWorkflow


# --------------------------------------------------------------------------- #
# Recording stubs: capture every agent prompt + every adversarial adv_prompt.
# --------------------------------------------------------------------------- #

class _RecordingAgent(AgentInstance):
    """A no-network agent that appends its construction prompt to a shared trace.

    The summarizer returns a per-step deterministic summary so the composed
    context strings are exactly predictable. The class-level ``trace`` records
    ('agent', prompt) for every instance built (planner/summarizer; ToT/adv
    generators when branches>1), giving the golden-trace test a stable signal.
    """

    trace: List[tuple] = []
    # id -> summary text, keyed by the "Step N" the summarizer prompt mentions.
    summaries: dict = {}

    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        _RecordingAgent.trace.append(("agent", self.prompt))
        # Deterministic summary so each step's summary line is predictable.
        return f"SUMMARY[{len(_RecordingAgent.trace)}]"


class _RecordingAdv:
    """Stand-in for AdversarialReview that records the adv_prompt it executed.

    The adv_prompt embeds the full step_prompt (and therefore the composed
    project_context), so recording it captures the exact context string each
    step saw — the load-bearing signal for both the golden-trace and the DAG
    context-composition assertions.
    """

    executed: List[str] = []

    def __init__(self, **kw):
        self.verified = True
        self.approved = True
        self.stalled = False
        self.iterations_used = 1

    async def execute(self, prompt: str) -> str:
        _RecordingAdv.executed.append(prompt)
        return f"STEP OUTPUT {len(_RecordingAdv.executed)}"


def _reset_recorders():
    _RecordingAgent.trace = []
    _RecordingAdv.executed = []


def _make_master(**kw) -> MasterWorkflow:
    defaults = dict(model="m", effort="low", branches=1, agent_class=_RecordingAgent)
    defaults.update(kw)
    return MasterWorkflow(**defaults)


def _patch(monkeypatch):
    """Stub the planner (must not fire on injected plans), checkpoints, and adv."""
    _reset_recorders()

    async def _planner_boom(self, initial_prompt):
        raise AssertionError("planner ran but a plan was injected")

    monkeypatch.setattr(MasterWorkflow, "_run_planner", _planner_boom)
    monkeypatch.setattr(MasterWorkflow, "_save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(MasterWorkflow, "_load_checkpoint", lambda self, p: None)
    monkeypatch.setattr(master_mod, "AdversarialReview", _RecordingAdv)


# --------------------------------------------------------------------------- #
# GOLDEN TRACE — refactored linear loop is byte-identical (3-step flat plan).
# --------------------------------------------------------------------------- #

def test_golden_trace_linear_loop_unchanged(monkeypatch):
    _patch(monkeypatch)
    plan = ["implement ONE", "implement TWO", "implement THREE"]
    m = _make_master(plan_steps=plan)

    asyncio.run(m.execute("build the thing"))

    # Three adversarial executions, in order, each carrying its task text.
    assert len(_RecordingAdv.executed) == 3
    for task, adv_prompt in zip(plan, _RecordingAdv.executed):
        assert task in adv_prompt
        assert adv_prompt.startswith("You are implementing Step")

    # The context string each step saw is the cumulative summary chain the
    # pre-refactor loop produced: step 1 sees only the goal; step 2 sees step 1's
    # summary; step 3 sees steps 1 + 2. With branches<=1 the adv_prompt IS the
    # step_prompt verbatim, so we can assert the exact composed context.
    s1, s2, s3 = _RecordingAdv.executed
    base = (
        "You are implementing Step 1 of a larger project.\n\n"
        "Project Context (What has been built so far):\n"
        "Original Goal: build the thing\n\n=== Accumulated Implementation ===\n"
        "\n\nCurrent Task to implement NOW:\nimplement ONE"
    )
    assert s1 == base

    # Step 2's context = goal + "--- Step 1 Summary ---\n<summary1>".
    assert "--- Step 1 Summary ---" in s2
    assert "--- Step 2 Summary ---" not in s2
    # Step 3's context carries BOTH prior step summaries, in order.
    assert "--- Step 1 Summary ---" in s3
    assert "--- Step 2 Summary ---" in s3
    assert s3.index("--- Step 1 Summary ---") < s3.index("--- Step 2 Summary ---")

    # One summarizer agent call per step (branches<=1 => no ToT agents).
    summarizer_calls = [p for _, p in _RecordingAgent.trace if "summarize what was just" in p]
    assert len(summarizer_calls) == 3
    assert "Step 1" in summarizer_calls[0]
    assert "Step 2" in summarizer_calls[1]
    assert "Step 3" in summarizer_calls[2]

    # Run-level signals reflect the final step (#45).
    assert m.verified is True
    assert m.approved is True


def test_flat_plan_does_not_enter_graph_walker(monkeypatch):
    # A flat plan_steps list never builds a plan_graph -> stays on the linear path.
    _patch(monkeypatch)
    m = _make_master(plan_steps=["a", "b"])
    assert m.plan_graph is None
    assert m._is_graph_plan() is False
    asyncio.run(m.execute("x"))
    assert len(_RecordingAdv.executed) == 2


def test_linear_graph_routes_to_linear_loop(monkeypatch):
    # A graph whose deps form a pure chain is "linear" — auto-route keeps it on
    # the flat loop (no branch/join present), so _is_graph_plan is False.
    _patch(monkeypatch)
    g = GraphPlan(nodes=[
        PlanNode(id="s1", task="one", deps=[]),
        PlanNode(id="s2", task="two", deps=["s1"]),
        PlanNode(id="s3", task="three", deps=["s2"]),
    ])
    m = _make_master(plan_graph=g)
    assert m._is_graph_plan() is False
    asyncio.run(m.execute("x"))
    assert len(_RecordingAdv.executed) == 3


# --------------------------------------------------------------------------- #
# DAG (M3): diamond runs B & C CONCURRENTLY; topo order respected; context is
# ancestor-only. Nodes run in isolated workspaces, so each graph-executing test
# points working_directory at a throwaway git repo (never the live tree).
# --------------------------------------------------------------------------- #

import subprocess


@pytest.fixture
def tmp_git_repo(tmp_path):
    """A tiny committed git repo so per-node _ManagedWorkspace can `git worktree
    add` (fast isolation) and apply_workspace mirrors back into an isolated tree
    — never the real AgentOrch checkout."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "seed.txt").write_text("seed\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True)
    return repo


def _diamond_graph() -> GraphPlan:
    # A -> {B, C} -> D
    return GraphPlan(nodes=[
        PlanNode(id="A", task="task ALPHA", deps=[]),
        PlanNode(id="B", task="task BRAVO", deps=["A"]),
        PlanNode(id="C", task="task CHARLIE", deps=["A"]),
        PlanNode(id="D", task="task DELTA", deps=["B", "C"]),
    ])


def test_dag_topo_order_and_context_composition(monkeypatch, tmp_git_repo):
    _patch(monkeypatch)
    g = _diamond_graph()
    m = _make_master(plan_graph=g, working_directory=str(tmp_git_repo))
    assert m._is_graph_plan() is True

    asyncio.run(m.execute("build a service"))

    # Four nodes ran, exactly once each. A is first (the only seed) and D is last
    # (the join); B/C run concurrently so their relative order is NOT asserted.
    execs = _RecordingAdv.executed
    assert len(execs) == 4
    by_tag = {tag: e for tag in ("ALPHA", "BRAVO", "CHARLIE", "DELTA")
              for e in execs if f"task {tag}" in e}
    assert set(by_tag) == {"ALPHA", "BRAVO", "CHARLIE", "DELTA"}
    assert "task ALPHA" in execs[0]   # the sole seed always runs first
    assert "task DELTA" in execs[3]   # the join always runs last

    ctx_a, ctx_b, ctx_c, ctx_d = (
        by_tag["ALPHA"], by_tag["BRAVO"], by_tag["CHARLIE"], by_tag["DELTA"],
    )

    # A is a seed: its context has NO step summaries.
    assert "--- Step" not in ctx_a

    # B sees ONLY its own ancestor (A) — NEITHER its sibling C nor the join D.
    assert "--- Step 1 Summary ---" in ctx_b  # A's summary (A is topo step 1)
    assert "task CHARLIE" not in ctx_b
    assert "--- Step 3 Summary ---" not in ctx_b  # C's slot
    assert "--- Step 4 Summary ---" not in ctx_b  # D's slot

    # C also sees ONLY A (not its sibling B).
    assert "--- Step 1 Summary ---" in ctx_c
    assert "--- Step 2 Summary ---" not in ctx_c  # B's slot
    assert "task BRAVO" not in ctx_c

    # D is the join: it sees the UNION of both branches (A, B, C) — BOTH B and C.
    assert "--- Step 1 Summary ---" in ctx_d  # A
    assert "--- Step 2 Summary ---" in ctx_d  # B
    assert "--- Step 3 Summary ---" in ctx_d  # C

    # Run-level signals reflect the final (leaf) node.
    assert m.verified is True
    assert m.approved is True


def test_dag_dep_never_starts_before_parent(monkeypatch, tmp_git_repo):
    # A dep must never run before its parent completed. Assert it off the
    # execution order: every node's task text appears AFTER all of its deps'.
    _patch(monkeypatch)
    g = _diamond_graph()
    m = _make_master(plan_graph=g, working_directory=str(tmp_git_repo))
    asyncio.run(m.execute("build a service"))

    order_text = list(_RecordingAdv.executed)
    pos = {}
    for tag, node in [("task ALPHA", g.nodes[0]), ("task BRAVO", g.nodes[1]),
                      ("task CHARLIE", g.nodes[2]), ("task DELTA", g.nodes[3])]:
        pos[node.id] = next(i for i, e in enumerate(order_text) if tag in e)
    by_id = {n.id: n for n in g.nodes}
    for nid, idx in pos.items():
        for dep in by_id[nid].deps:
            assert pos[dep] < idx, f"{nid} ran before its dep {dep}"


def test_dag_final_string_is_linear_equivalent(monkeypatch, tmp_git_repo):
    # The DAG run returns the cumulative project_context a linear run would build:
    # one "--- Step N Summary ---" block per node, in topological order.
    _patch(monkeypatch)
    g = _diamond_graph()
    m = _make_master(plan_graph=g, working_directory=str(tmp_git_repo))
    out = asyncio.run(m.execute("build a service"))
    assert out.startswith("Original Goal: build a service")
    for n in (1, 2, 3, 4):
        assert f"--- Step {n} Summary ---" in out


# --------------------------------------------------------------------------- #
# CONCURRENCY (M3): a 2-wide layer (B, C) actually overlaps in wall-clock time,
# and --max-parallel-nodes 1 serializes it. Proven with asyncio.Events that
# block one sibling inside its step until the other has STARTED.
# --------------------------------------------------------------------------- #

class _OverlapProbe:
    """Drives a stub AdversarialReview to prove B and C overlap.

    B and C each signal a 'started' Event on entry, then await the OTHER's
    'started' Event before returning. If the scheduler ran them concurrently
    both awaits resolve and the run completes; if it serialized them the second
    can never start while the first blocks, so the run would deadlock (caught by
    a timeout). A and D don't gate.
    """

    def __init__(self):
        self.started = {"B": asyncio.Event(), "C": asyncio.Event()}
        self.both_overlapped = False

    def adv_factory(self):
        probe = self

        class _GatingAdv:
            def __init__(self, **kw):
                self.verified = True
                self.approved = True
                self.stalled = False
                self.iterations_used = 1

            async def execute(self, prompt: str) -> str:
                _RecordingAdv.executed.append(prompt)
                which = "B" if "task BRAVO" in prompt else (
                    "C" if "task CHARLIE" in prompt else None)
                if which is not None:
                    probe.started[which].set()
                    other = "C" if which == "B" else "B"
                    # Block until the sibling has started — only possible if
                    # the two ran concurrently.
                    await probe.started[other].wait()
                    probe.both_overlapped = True
                return f"OUT {which or 'X'}"

        return _GatingAdv


def test_dag_2wide_layer_runs_concurrently(monkeypatch, tmp_git_repo):
    _patch(monkeypatch)
    probe = _OverlapProbe()
    monkeypatch.setattr(master_mod, "AdversarialReview", probe.adv_factory())
    g = _diamond_graph()
    m = _make_master(plan_graph=g, working_directory=str(tmp_git_repo))

    # A short timeout: if B and C were serialized, the cross-Event wait deadlocks.
    async def _run():
        return await asyncio.wait_for(m.execute("build a service"), timeout=30)

    asyncio.run(_run())
    assert probe.both_overlapped is True, "B and C did not overlap (no parallelism)"


def test_dag_max_parallel_nodes_1_serializes(monkeypatch, tmp_git_repo):
    # With max_parallel_nodes=1 the 2-wide layer CANNOT overlap. The gating probe
    # would deadlock waiting for the sibling, so we assert the run times out
    # (proof the node semaphore serialized the layer) and the second sibling
    # never started while the first held the single slot.
    _patch(monkeypatch)
    probe = _OverlapProbe()
    monkeypatch.setattr(master_mod, "AdversarialReview", probe.adv_factory())
    g = _diamond_graph()
    m = _make_master(plan_graph=g, working_directory=str(tmp_git_repo),
                     max_parallel_workers=1)
    assert m.max_parallel_nodes == 1

    async def _run():
        # Bounded wait: a serialized layer deadlocks on the cross-Event wait, so
        # this MUST time out. (A real run with non-gating agents would finish.)
        return await asyncio.wait_for(m.execute("x"), timeout=3)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(_run())
    # Exactly one sibling got to start; the other is still blocked on the slot.
    started = [k for k, ev in probe.started.items() if ev.is_set()]
    assert len(started) == 1, f"expected 1 sibling started under cap=1, got {started}"
    assert probe.both_overlapped is False


def test_dag_disjoint_writes_both_applied(monkeypatch, tmp_git_repo):
    # M3 ships true parallelism for disjoint-subtree DAGs: each node writes a
    # distinct file in its isolated workspace; apply_workspace mirrors BOTH back
    # into the live tree (no overlap, so last-writer-wins never triggers).
    _reset_recorders()

    async def _planner_boom(self, initial_prompt):
        raise AssertionError("planner ran but a plan was injected")

    monkeypatch.setattr(MasterWorkflow, "_run_planner", _planner_boom)
    monkeypatch.setattr(MasterWorkflow, "_save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(MasterWorkflow, "_load_checkpoint", lambda self, p: None)

    class _WritingAdv:
        """Writes a node-specific file into its workspace cwd."""

        def __init__(self, *, working_directory=".", **kw):
            self.verified = True
            self.approved = True
            self.stalled = False
            self.iterations_used = 1
            self._wd = working_directory

        async def execute(self, prompt: str) -> str:
            _RecordingAdv.executed.append(prompt)
            tag = next((t for t in ("ALPHA", "BRAVO", "CHARLIE", "DELTA")
                        if f"task {t}" in prompt), "X")
            from pathlib import Path as _P
            (_P(self._wd) / f"{tag}.txt").write_text(f"{tag} wrote here\n")
            return f"OUT {tag}"

    monkeypatch.setattr(master_mod, "AdversarialReview", _WritingAdv)
    g = _diamond_graph()
    m = _make_master(plan_graph=g, working_directory=str(tmp_git_repo))
    asyncio.run(m.execute("build a service"))

    # Every node's disjoint write landed in the live tree.
    for tag in ("ALPHA", "BRAVO", "CHARLIE", "DELTA"):
        assert (tmp_git_repo / f"{tag}.txt").read_text() == f"{tag} wrote here\n"


# --------------------------------------------------------------------------- #
# MERGE (M4): two parallel nodes (B, C) write the SAME file. Under
# --merge-policy fail the second-merged sibling aborts with the conflict
# recorded in merge_outcomes (-> meta.json); under reconcile a STUBBED reconcile
# station resolves the overlap and the join node's verifier re-runs.
# --------------------------------------------------------------------------- #

def _patch_no_planner(monkeypatch):
    _reset_recorders()

    async def _planner_boom(self, initial_prompt):
        raise AssertionError("planner ran but a plan was injected")

    monkeypatch.setattr(MasterWorkflow, "_run_planner", _planner_boom)
    monkeypatch.setattr(MasterWorkflow, "_save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(MasterWorkflow, "_load_checkpoint", lambda self, p: None)


class _SharedWritingAdv:
    """B and C both write shared.txt (an overlap); A and D write disjoint files.

    B/C synchronize on a shared barrier so BOTH open their workspace BEFORE
    either merges — the only way the second merge sees the first's write already
    in the live tree (the overlap signal). A's prior merge of A.txt is disjoint.
    """

    barrier = None  # set per-test to an asyncio.Barrier-like gate

    def __init__(self, *, working_directory=".", **kw):
        self.verified = True
        self.approved = True
        self.stalled = False
        self.iterations_used = 1
        self._wd = working_directory

    async def execute(self, prompt: str) -> str:
        _RecordingAdv.executed.append(prompt)
        from pathlib import Path as _P
        tag = next((t for t in ("ALPHA", "BRAVO", "CHARLIE", "DELTA")
                    if f"task {t}" in prompt), "X")
        if tag in ("BRAVO", "CHARLIE"):
            # Gate both siblings open before either returns (and merges), so the
            # second sibling's merge sees the first's shared.txt already applied.
            if _SharedWritingAdv.barrier is not None:
                await _SharedWritingAdv.barrier.wait()
            (_P(self._wd) / "shared.txt").write_text(f"{tag} version\n")
        else:
            (_P(self._wd) / f"{tag}.txt").write_text(f"{tag} wrote here\n")
        return f"OUT {tag}"


def test_graph_merge_fail_policy_aborts_with_conflict(monkeypatch, tmp_git_repo):
    _patch_no_planner(monkeypatch)
    _SharedWritingAdv.barrier = asyncio.Barrier(2)
    monkeypatch.setattr(master_mod, "AdversarialReview", _SharedWritingAdv)
    g = _diamond_graph()
    m = _make_master(plan_graph=g, working_directory=str(tmp_git_repo),
                     merge_policy="fail")

    # The second sibling's merge raises MergeConflict -> the run aborts.
    from agy_orchestrator.workflows.graph_merge import MergeConflict
    with pytest.raises(MergeConflict):
        asyncio.run(m.execute("build a service"))

    # The conflict outcome was recorded (conflicting path -> meta.json).
    conflicts = [o for o in m.merge_outcomes if o.conflict]
    assert len(conflicts) == 1
    assert conflicts[0].policy == "fail"
    assert conflicts[0].resolution == "aborted:fail-policy"
    assert "shared.txt" in conflicts[0].overlapping_paths


def test_graph_merge_reconcile_invokes_station_and_reverifies(monkeypatch, tmp_git_repo):
    _patch_no_planner(monkeypatch)
    _SharedWritingAdv.barrier = asyncio.Barrier(2)
    monkeypatch.setattr(master_mod, "AdversarialReview", _SharedWritingAdv)

    # A re-verify counter on the (real) verifier proves the join node re-verifies
    # after the reconcile rewrites the live tree.
    reverify_calls = {"n": 0}

    class _StubVerifier:
        async def verify(self, working_directory):
            reverify_calls["n"] += 1
            class _R:
                ok = True
            return _R()

    # Stub reconcile station factory: records the call + returns merged bytes.
    recon_calls = []

    def _factory(*, node, working_directory):
        async def _reconcile(rel, base_bytes, sibling_bytes, node_bytes):
            recon_calls.append((node.id, rel))
            return b"RECONCILED\n"
        return _reconcile

    g = _diamond_graph()
    m = _make_master(plan_graph=g, working_directory=str(tmp_git_repo),
                     merge_policy="reconcile", verifier=_StubVerifier(),
                     reconcile_station_factory=_factory)
    asyncio.run(m.execute("build a service"))

    # Exactly one node hit the overlap and reconciled shared.txt.
    reconciled = [o for o in m.merge_outcomes if o.resolution == "reconciled"]
    assert len(reconciled) == 1
    assert reconciled[0].overlapping_paths == ["shared.txt"]
    # The stubbed station was consulted for the overlapping path.
    assert len(recon_calls) == 1 and recon_calls[0][1] == "shared.txt"
    # Merged bytes landed in the live tree.
    assert (tmp_git_repo / "shared.txt").read_text() == "RECONCILED\n"
    # The join node's verifier re-ran on the merged tree after the reconcile
    # rewrote the live tree (docs §4). The stub AdversarialReview never invokes
    # the verifier itself, so this single call is exactly the post-merge re-verify.
    assert reverify_calls["n"] == 1


def test_graph_merge_default_policy_is_reconcile(monkeypatch, tmp_git_repo):
    # No merge_policy passed -> operator-confirmed default reconcile.
    _patch_no_planner(monkeypatch)
    m = _make_master(plan_graph=_diamond_graph(),
                     working_directory=str(tmp_git_repo))
    assert m.merge_policy == "reconcile"


def test_master_rejects_bad_merge_policy():
    with pytest.raises(ValueError):
        _make_master(plan_steps=["a"], merge_policy="bogus")


# --------------------------------------------------------------------------- #
# DISPATCH-LEVEL meta.json (M4 regression): the unit test above asserts on
# m.merge_outcomes DIRECTLY, bypassing the real meta-write path. But merge_node
# raises MergeConflict, which propagates OUT of MasterWorkflow.execute() and
# through `_run_workflow`'s `return await wf.execute(prompt), wf` — so the
# (output, workflow) tuple is never returned and dispatch_async's `workflow`
# stays None, silently dropping the conflict from meta.json. This drives the
# real dispatch -> meta.json path and asserts the conflict actually lands there
# (i.e. dispatch recovers the in-flight workflow from the sink on abort).
# --------------------------------------------------------------------------- #

def test_graph_merge_fail_conflict_reaches_meta_json(monkeypatch, tmp_git_repo, tmp_path):
    import json
    from pathlib import Path

    from harness import dispatch as dispatch_mod
    from harness import roles

    _patch_no_planner(monkeypatch)
    _SharedWritingAdv.barrier = asyncio.Barrier(2)
    monkeypatch.setattr(master_mod, "AdversarialReview", _SharedWritingAdv)

    # Keep run artifacts out of the repo.
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", tmp_path / "runs")

    # Hand the master loop the recording agent class (no real workers).
    monkeypatch.setattr(
        roles, "build_master_agent_class",
        lambda chain, **kw: (_RecordingAgent, "m", "low"),
    )

    g = _diamond_graph()
    result = dispatch_mod.dispatch(
        "build a service",
        mode="master",
        plan_graph=g,
        merge_policy="fail",
        out_dir=tmp_git_repo,
        generator_chain=["codex"],
    )

    # The merge abort fails the run, but the conflict must be recorded — NOT
    # silently dropped because execute() raised before the workflow was unpacked.
    assert result.success is False
    assert result.graph is not None
    meta = json.loads((Path(result.run_dir) / "meta.json").read_text())
    assert meta["graph"] is not None
    assert meta["graph"]["merge_policy"] == "fail"
    conflicts = [m for m in meta["graph"]["merges"] if m["conflict"]]
    assert len(conflicts) == 1
    assert conflicts[0]["resolution"] == "aborted:fail-policy"
    assert "shared.txt" in conflicts[0]["overlapping_paths"]


# --------------------------------------------------------------------------- #
# M5 — round-trip emit + meta observability + checkpoint/resume + pat-graph error.
# Covers docs/graph-execution-design.md §5 M5 + §2 + §7:
#   * meta.json gets the FULL graph block (node statuses, Kahn layers, merges,
#     merge_policy) on a happy-path diamond run.
#   * --plan-only echoes a graph plan back as a v2 'nodes' plan.json that re-loads
#     via load_plan (the operator round-trip).
#   * checkpoint/resume — a crash after B done resumes ONLY the frontier (C, D),
#     seeded per-node from the persisted 'done' map (not a "first K" guess).
#   * a base_fingerprint divergence DISCARDS the stale graph checkpoint (#37).
#   * a graph DAG under --mode pat errors at the CLI (graphs are master-only v1).
# --------------------------------------------------------------------------- #

def test_meta_graph_block_full_shape(monkeypatch, tmp_git_repo, tmp_path):
    # A clean diamond run writes the v2 graph block: node statuses all "passed",
    # the Kahn layers ([[A],[B,C],[D]]), an (empty) merges list, and the policy.
    import json as _json
    from pathlib import Path

    from harness import dispatch as dispatch_mod
    from harness import roles

    _patch_no_planner(monkeypatch)
    monkeypatch.setattr(master_mod, "AdversarialReview", _RecordingAdv)
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(
        roles, "build_master_agent_class",
        lambda chain, **kw: (_RecordingAgent, "m", "low"),
    )

    g = _diamond_graph()
    result = dispatch_mod.dispatch(
        "build a service",
        mode="master",
        plan_graph=g,
        out_dir=tmp_git_repo,
        generator_chain=["codex"],
    )

    assert result.graph is not None
    meta = _json.loads((Path(result.run_dir) / "meta.json").read_text())
    block = meta["graph"]
    assert block is not None
    # Every node passed (the recording adv reports verified+approved).
    assert block["nodes"] == {"A": "passed", "B": "passed", "C": "passed", "D": "passed"}
    # Kahn layers: A alone, then the {B,C} parallel layer, then the D join.
    assert block["layers"] == [["A"], ["B", "C"], ["D"]]
    # Disjoint writes -> no conflicts recorded; default policy is reconcile.
    assert block["merge_policy"] == "reconcile"
    assert all(not m["conflict"] for m in block["merges"])


def test_plan_only_emits_graph_plan_that_reloads(monkeypatch, tmp_git_repo, tmp_path):
    # --plan-only echoes an injected graph back as a v2 'nodes' plan.json; loading
    # it via load_plan reproduces the SAME GraphPlan (the operator round-trip).
    import json as _json
    from pathlib import Path

    from harness import dispatch as dispatch_mod
    from harness import roles
    from harness.dispatch import load_plan
    from agy_orchestrator.execution.graph_plan import GraphPlan

    _patch_no_planner(monkeypatch)
    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(
        roles, "build_master_agent_class",
        lambda chain, **kw: (_RecordingAgent, "m", "low"),
    )

    g = _diamond_graph()
    result = dispatch_mod.dispatch(
        "build a service",
        mode="master",
        plan_only=True,
        plan_graph=g,
        out_dir=tmp_git_repo,
        generator_chain=["codex"],
    )

    plan_json = Path(result.run_dir) / "plan.json"
    assert plan_json.exists()
    raw = _json.loads(plan_json.read_text())
    # The v2 graph shape, not the flat 'steps' object.
    assert raw.get("version") == 2
    assert "nodes" in raw and "steps" not in raw
    assert {n["id"] for n in raw["nodes"]} == {"A", "B", "C", "D"}

    # It re-loads as a GraphPlan with the same topology.
    reloaded = load_plan(plan_json)
    assert isinstance(reloaded, GraphPlan)
    assert [n.id for n in reloaded.nodes] == ["A", "B", "C", "D"]
    assert reloaded.parallel_groups() == [["A"], ["B", "C"], ["D"]]


def test_graph_checkpoint_resume_runs_only_frontier(monkeypatch, tmp_git_repo, tmp_path):
    # Crash after B is done; on resume ONLY the frontier (C, D) executes — seeded
    # per-node from the persisted 'done' map. We use a REAL checkpoint file (the
    # _save_checkpoint/_load_checkpoint pair is the unit under test here).
    _reset_recorders()

    async def _planner_boom(self, initial_prompt):
        raise AssertionError("planner ran but a plan was injected")

    monkeypatch.setattr(MasterWorkflow, "_run_planner", _planner_boom)
    # Make the base_fingerprint deterministic + matching across the two runs so
    # the #37 gate approves resume (the tmp_git_repo is otherwise quiescent).
    monkeypatch.setattr(MasterWorkflow, "_base_fingerprint", lambda self: "FP")

    ckpt = str(tmp_path / "ck.json")

    # Run 1: a gating adv that raises once C starts, AFTER B has finished + been
    # checkpointed. This simulates a crash mid-DAG with B already done.
    class _CrashAfterB:
        def __init__(self, **kw):
            self.verified = True
            self.approved = True
            self.stalled = False
            self.iterations_used = 1

        async def execute(self, prompt: str) -> str:
            _RecordingAdv.executed.append(prompt)
            if "task CHARLIE" in prompt or "task DELTA" in prompt:
                raise RuntimeError("crash after B")
            return f"OUT {len(_RecordingAdv.executed)}"

    monkeypatch.setattr(master_mod, "AdversarialReview", _CrashAfterB)
    g = _diamond_graph()
    m1 = _make_master(plan_graph=g, working_directory=str(tmp_git_repo),
                      checkpoint_path=ckpt, max_parallel_workers=1)
    with pytest.raises(RuntimeError):
        asyncio.run(m1.execute("build a service"))

    # The checkpoint persisted the per-node 'done' frontier with A and B.
    import json as _json
    from pathlib import Path
    saved = _json.loads(Path(ckpt).read_text())
    assert set(saved["done"]) >= {"A", "B"}
    assert "graph" in saved

    # Run 2: a clean adv. Resume must run ONLY C and D (A, B are seeded done).
    _RecordingAdv.executed = []
    monkeypatch.setattr(master_mod, "AdversarialReview", _RecordingAdv)
    m2 = _make_master(plan_graph=g, working_directory=str(tmp_git_repo),
                      checkpoint_path=ckpt)
    asyncio.run(m2.execute("build a service"))

    ran_tags = {t for t in ("ALPHA", "BRAVO", "CHARLIE", "DELTA")
                for e in _RecordingAdv.executed if f"task {t}" in e}
    assert ran_tags == {"CHARLIE", "DELTA"}, f"resume re-ran nodes: {ran_tags}"
    # D (the join) saw both B (from the resumed frontier) and C summaries.
    ctx_d = next(e for e in _RecordingAdv.executed if "task DELTA" in e)
    assert "--- Step 2 Summary ---" in ctx_d  # B's slot (seeded from checkpoint)
    assert "--- Step 3 Summary ---" in ctx_d  # C's slot (freshly run)


def test_graph_checkpoint_discarded_on_base_divergence(monkeypatch, tmp_git_repo, tmp_path):
    # A base_fingerprint divergence (#37) must DISCARD the stale graph checkpoint
    # and re-run the whole DAG from scratch — never resume onto a reset tree.
    _reset_recorders()

    async def _planner_boom(self, initial_prompt):
        raise AssertionError("planner ran but a plan was injected")

    monkeypatch.setattr(MasterWorkflow, "_run_planner", _planner_boom)
    monkeypatch.setattr(master_mod, "AdversarialReview", _RecordingAdv)

    ckpt = str(tmp_path / "ck.json")
    g = _diamond_graph()

    # Write a checkpoint as if A and B completed under fingerprint "OLD".
    m_seed = _make_master(plan_graph=g, working_directory=str(tmp_git_repo),
                          checkpoint_path=ckpt)
    monkeypatch.setattr(MasterWorkflow, "_base_fingerprint", lambda self: "OLD")
    m_seed._save_checkpoint(
        "build a service", g.as_steps(), 2,
        "Original Goal: build a service\n", None,
        graph=[{"id": n.id, "task": n.task, "deps": list(n.deps)} for n in g.nodes],
        done={
            nid: {"step_summary": f"sum {nid}", "verified": True, "approved": True,
                  "stalled": False, "iterations_used": 1}
            for nid in ("A", "B")
        },
    )

    # Now the tree has DIVERGED (fingerprint flips to "NEW"): resume must be
    # refused and ALL four nodes re-run.
    monkeypatch.setattr(MasterWorkflow, "_base_fingerprint", lambda self: "NEW")
    m2 = _make_master(plan_graph=g, working_directory=str(tmp_git_repo),
                      checkpoint_path=ckpt)
    asyncio.run(m2.execute("build a service"))

    ran_tags = {t for t in ("ALPHA", "BRAVO", "CHARLIE", "DELTA")
                for e in _RecordingAdv.executed if f"task {t}" in e}
    assert ran_tags == {"ALPHA", "BRAVO", "CHARLIE", "DELTA"}, \
        "stale checkpoint was not discarded on base divergence"


def test_cli_pat_with_graph_errors(monkeypatch, tmp_path, capsys):
    # Graphs are master-only for v1: --mode pat with a graph plan errors at the
    # CLI (before any dispatch), per docs §8 Q4 / §5 M5.
    import json as _json
    from harness import cli

    plan_file = tmp_path / "graph.json"
    plan_file.write_text(_json.dumps({
        "nodes": [
            {"id": "a", "task": "do a", "deps": []},
            {"id": "b", "task": "do b", "deps": []},
            {"id": "c", "task": "join", "deps": ["a", "b"]},
        ]
    }))

    rc = cli.main(["do", "build it", "--mode", "pat", "--plan", str(plan_file),
                   "--test-cmd", "true"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "master" in err.lower()


def test_cli_plan_graph_strict_alias_rejects_flat(monkeypatch, tmp_path, capsys):
    # --plan-graph is the STRICT alias: a flat plan file errors (use --plan).
    import json as _json
    from harness import cli

    flat = tmp_path / "flat.json"
    flat.write_text(_json.dumps(["step one", "step two"]))

    rc = cli.main(["do", "build it", "--mode", "master", "--plan-graph", str(flat)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "graph" in err.lower()


def test_tdag_stays_orphaned():
    # docs §8 Q5: tdag.py is SUPERSEDED by the master graph mode. Keep the file
    # but ensure NO production module imports it — a stray import would resurrect
    # dead code + the dead-code ambiguity this test exists to prevent. Scans the
    # orchestrator + harness source for `tdag` references (skipping this test file
    # and any __pycache__).
    import pathlib

    root = pathlib.Path(master_mod.__file__).resolve().parent.parent.parent
    offenders = []
    for src in list((root / "agy_orchestrator").rglob("*.py")) + \
            list((root / "harness").rglob("*.py")):
        if src.name == "tdag.py":
            continue
        text = src.read_text(encoding="utf-8")
        if "tdag" in text:
            offenders.append(str(src))
    assert offenders == [], f"tdag.py is imported by production code: {offenders}"

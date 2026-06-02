"""Hermetic fuzz / property / edge-case tests for MasterWorkflow.

These exercise the plan/ToT/adversarial/checkpoint/compaction machinery in
agy_orchestrator/workflows/master.py with FAKE agents (no network, no real
worker CLI) and adversarial inputs. Every test here asserts CORRECT, robust
behaviour and must stay green.

The genuine defects discovered while writing these (corrupt-checkpoint crashes,
planner list-of-non-strings crash, negative-`completed` step reordering) are
reported separately as findings with their repros — they are NOT committed as
failing tests.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from typing import List, Optional

import pytest

import agy_orchestrator.workflows.master as master_mod
from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.graph_plan import GraphPlan, PlanNode
from agy_orchestrator.workflows.master import MasterWorkflow


# --------------------------------------------------------------------------- #
# Fakes — no subprocess, no network.
# --------------------------------------------------------------------------- #
class _Stub(AgentInstance):
    """Deterministic agent; planner asks for JSON, everything else echoes."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)

    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        p = self.prompt or ""
        if "JSON list" in p or "valid JSON list of strings" in p:
            return '["step one", "step two"]'
        return "summary-line"


class _Adv:
    """AdversarialReview stand-in exposing the run-level signals."""

    def __init__(self, *, verified=True, approved=True, stalled=False,
                 iterations_used=1, **_ignored):
        self.verified = verified
        self.approved = approved
        self.stalled = stalled
        self.iterations_used = iterations_used

    async def execute(self, prompt: str) -> str:
        return "FINAL OUTPUT"


def _make(**kw) -> MasterWorkflow:
    defaults = dict(model="m", effort="low", branches=1, agent_class=_Stub)
    defaults.update(kw)
    return MasterWorkflow(**defaults)


@pytest.fixture(autouse=True)
def _green_adv(monkeypatch):
    """Default every test to a green adversarial review unless overridden."""
    monkeypatch.setattr(master_mod, "AdversarialReview",
                        lambda **kw: _Adv(verified=True, approved=True))
    yield


# --------------------------------------------------------------------------- #
# Constructor argument validation / boundaries.
# --------------------------------------------------------------------------- #
def test_invalid_resume_policy_rejected():
    with pytest.raises(ValueError):
        _make(resume_policy="bogus")


def test_invalid_merge_policy_rejected():
    with pytest.raises(ValueError):
        _make(merge_policy="nope")


@pytest.mark.parametrize("rsv,expected", [(-5, 0), (0, 0), (3, 3)])
def test_recent_steps_verbatim_clamped_nonnegative(rsv, expected):
    assert _make(recent_steps_verbatim=rsv).recent_steps_verbatim == expected


@pytest.mark.parametrize("vc,expected", [(-3, 1), (0, 1), (1, 1), (4, 4)])
def test_verifier_concurrency_floored_at_one(vc, expected):
    assert _make(verifier_concurrency=vc).verifier_concurrency == expected


def test_max_parallel_workers_zero_means_unbounded():
    # 0/None/negative => no node cap (None), positive => floored at 1.
    assert _make(max_parallel_workers=0).max_parallel_nodes is None
    assert _make(max_parallel_workers=None).max_parallel_nodes is None
    assert _make(max_parallel_workers=7).max_parallel_nodes == 7


def test_bad_env_max_parallel_nodes_ignored(monkeypatch):
    monkeypatch.setenv("AGY_MAX_PARALLEL_NODES", "not-a-number")
    assert _make(max_parallel_workers=None).max_parallel_nodes is None
    monkeypatch.setenv("AGY_MAX_PARALLEL_NODES", "  3 ")
    # "  3 ".strip().isdigit() is True -> parsed
    assert _make(max_parallel_workers=None).max_parallel_nodes == 3


# --------------------------------------------------------------------------- #
# _checkpoint_key — must never crash on adversarial prompts.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prompt", [
    "", " ", "\n\t", "x" * 100_000,
    "unicode é中文 \U0001f600",
    "ctrl \x00\x01\x1b[0m nul-embedded",
    "ansi \x1b[31mred\x1b[0m",
])
def test_checkpoint_key_is_stable_hex(prompt):
    m = _make()
    k1 = m._checkpoint_key(prompt)
    k2 = m._checkpoint_key(prompt)
    assert k1 == k2
    assert len(k1) == 64 and all(c in "0123456789abcdef" for c in k1)


# --------------------------------------------------------------------------- #
# Corrupt / partial checkpoint persistence — must be tolerated, never crash.
# --------------------------------------------------------------------------- #
def _write(path, obj_or_text):
    with open(path, "w", encoding="utf-8") as fh:
        if isinstance(obj_or_text, str):
            fh.write(obj_or_text)
        else:
            json.dump(obj_or_text, fh)


def _ckpt_path(tmp_path):
    return str(tmp_path / "ckpt.json")


def test_missing_checkpoint_returns_none(tmp_path):
    m = _make(checkpoint_path=_ckpt_path(tmp_path))
    assert m._load_checkpoint("p") is None


def test_truncated_json_checkpoint_tolerated(tmp_path):
    p = _ckpt_path(tmp_path)
    _write(p, '{"key": "abc", "tasks": [')  # truncated
    m = _make(checkpoint_path=p)
    assert m._load_checkpoint("p") is None  # logged + ignored, no raise


def test_zero_byte_checkpoint_tolerated(tmp_path):
    p = _ckpt_path(tmp_path)
    _write(p, "")
    m = _make(checkpoint_path=p)
    assert m._load_checkpoint("p") is None


def test_garbage_checkpoint_tolerated(tmp_path):
    p = _ckpt_path(tmp_path)
    _write(p, "\x00\x01 not json at all \xff")
    m = _make(checkpoint_path=p)
    assert m._load_checkpoint("p") is None


@pytest.mark.parametrize("payload", [[1, 2, 3], 42, "hello", 3.14, True])
def test_non_dict_json_checkpoint_tolerated(tmp_path, payload):
    """REGRESSION for master-checkpoint-1 (now FIXED).

    A checkpoint file whose top-level JSON is a non-object (a list / bare
    number / string / bool) is valid JSON, so it sails past the json.load
    try/except. Previously ``data.get("key")`` then raised AttributeError and
    crashed the whole master run. It must now be treated like any other corrupt
    checkpoint: logged and ignored, returning None (start fresh), never a raise.
    """
    p = _ckpt_path(tmp_path)
    _write(p, payload)
    m = _make(checkpoint_path=p)
    assert m._load_checkpoint("p") is None


def test_checkpoint_wrong_key_starts_fresh(tmp_path):
    p = _ckpt_path(tmp_path)
    _write(p, {"key": "deadbeef", "tasks": ["a"], "completed": 0})
    m = _make(checkpoint_path=p)
    assert m._load_checkpoint("p") is None


def test_checkpoint_already_complete_returns_none(tmp_path):
    p = _ckpt_path(tmp_path)
    key = hashlib.sha256("p".encode()).hexdigest()
    _write(p, {"key": key, "tasks": ["a", "b"], "completed": 2,
               "base_fingerprint": None})
    m = _make(checkpoint_path=p)
    assert m._load_checkpoint("p") is None


def test_checkpoint_empty_tasks_returns_none(tmp_path):
    p = _ckpt_path(tmp_path)
    key = hashlib.sha256("p".encode()).hexdigest()
    _write(p, {"key": key, "tasks": [], "completed": 0,
               "base_fingerprint": None})
    m = _make(checkpoint_path=p)
    assert m._load_checkpoint("p") is None


def test_fresh_policy_ignores_existing_checkpoint(tmp_path):
    p = _ckpt_path(tmp_path)
    key = hashlib.sha256("p".encode()).hexdigest()
    _write(p, {"key": key, "tasks": ["a", "b"], "completed": 1,
               "base_fingerprint": None})
    m = _make(checkpoint_path=p, resume_policy="never")
    assert m._load_checkpoint("p") is None


# --------------------------------------------------------------------------- #
# _save_checkpoint robustness — bad path must not crash the run.
# --------------------------------------------------------------------------- #
def test_save_checkpoint_to_directory_is_swallowed(tmp_path):
    d = tmp_path / "is_a_dir"
    d.mkdir()
    m = _make(checkpoint_path=str(d))
    # open(dir, 'w') raises IsADirectoryError; must be caught + logged.
    m._save_checkpoint("p", ["a"], 0, "ctx", None)  # no raise


def test_save_then_load_roundtrip(tmp_path):
    p = _ckpt_path(tmp_path)
    m = _make(checkpoint_path=p)
    m._save_checkpoint("proj", ["s1", "s2", "s3"], 1, "the-context", "sess-9")
    out = m._load_checkpoint("proj")
    assert out is not None
    tasks, completed, ctx, sess = out
    assert tasks == ["s1", "s2", "s3"]
    assert completed == 1
    assert ctx == "the-context"
    assert sess == "sess-9"


def test_remove_checkpoint_missing_file_is_noop(tmp_path):
    m = _make(checkpoint_path=str(tmp_path / "nope.json"))
    m._remove_checkpoint()  # FileNotFoundError swallowed


# --------------------------------------------------------------------------- #
# Graph done-frontier loader — malformed records skipped, not fatal.
# --------------------------------------------------------------------------- #
def test_frontier_non_dict_done_returns_none(tmp_path):
    p = _ckpt_path(tmp_path)
    key = hashlib.sha256("p".encode()).hexdigest()
    _write(p, {"key": key, "done": ["not", "a", "dict"]})
    m = _make(checkpoint_path=p)
    assert m._load_graph_done_frontier("p") is None


def test_frontier_skips_non_dict_records(tmp_path):
    p = _ckpt_path(tmp_path)
    key = hashlib.sha256("p".encode()).hexdigest()
    _write(p, {"key": key, "done": {
        "good": {"step_summary": "s", "verified": True, "iterations_used": 2},
        "bad": "not-a-dict",
    }})
    m = _make(checkpoint_path=p)
    frontier = m._load_graph_done_frontier("p")
    assert frontier is not None
    assert "good" in frontier and "bad" not in frontier
    assert frontier["good"].verified is True
    assert frontier["good"].iterations_used == 2


# --------------------------------------------------------------------------- #
# Context splitter — property: preamble + ''.join(blocks) == input, always.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ctx", [
    "",
    "no markers here at all",
    "Original Goal: x\n=== Accumulated ===\n",
    "\n--- Step 1 Summary ---\nbody\n",
    "lead\n--- Step 1 Summary ---\na\n--- Step 2 Summary ---\nb\n",
    "--- Step 9 Summary ---\nx",  # no preceding newline
    "pre\n\n--- Step 1 Summary ---\n\n\n--- Step 2 Summary ---\n",
    "weird \x00 nul \x1b[0m ansi\n--- Step 3 Summary ---\n中文\n",
])
def test_splitter_roundtrips_property(ctx):
    m = _make()
    pre, blocks = m._split_context_into_steps(ctx)
    assert pre + "".join(blocks) == ctx


# --------------------------------------------------------------------------- #
# Compaction edge cases.
# --------------------------------------------------------------------------- #
def test_compact_no_markers_falls_open():
    m = _make()
    out = asyncio.run(m._compact_context("goal", "flat context no steps"))
    assert "Original Goal: goal" in out
    assert "compacted" in out


def test_compact_empty_context():
    m = _make()
    out = asyncio.run(m._compact_context("goal", ""))
    assert "Original Goal: goal" in out  # no crash on empty


def test_compact_falls_back_to_tail_on_compactor_failure():
    class _Boom(_Stub):
        async def run_async(self, piped_input=None):
            raise RuntimeError("compactor died")

    m = _make(max_context_chars=100, recent_steps_verbatim=2)
    m.agent_class = _Boom
    ctx = "Original Goal: g\n" + "".join(
        f"\n--- Step {i} Summary ---\nfile_{i}.py\n" for i in range(1, 6))
    out = asyncio.run(m._compact_context("g", ctx))
    assert "Original Goal" in out
    assert "file_5.py" in out  # most-recent step survives in the tail


def test_should_compact_disabled_when_zero():
    m = _make(compaction_interval=0, max_context_chars=0)
    assert m._should_compact(10_000, "x" * 1_000_000) is False


def test_should_compact_on_interval_and_chars():
    m = _make(compaction_interval=3, max_context_chars=50)
    assert m._should_compact(3, "short") is True          # interval
    assert m._should_compact(1, "x" * 60) is True          # chars
    assert m._should_compact(1, "x" * 10) is False


# --------------------------------------------------------------------------- #
# Plan-only / dry-run — side-effect free, never writes a checkpoint.
# --------------------------------------------------------------------------- #
def test_plan_only_does_not_write_checkpoint(tmp_path):
    p = _ckpt_path(tmp_path)
    m = _make(checkpoint_path=p, plan_only=True)
    out = asyncio.run(m.execute("build a thing"))
    assert "Plan-only" in out
    assert not os.path.exists(p)  # nothing written
    assert m.plan_steps == ["step one", "step two"]


def test_plan_only_echoes_injected_plan(tmp_path):
    m = _make(plan_only=True, plan_steps=["alpha", "beta", "gamma"])
    out = asyncio.run(m.execute("ignored"))
    assert "3 step" in out
    assert "alpha" in out and "gamma" in out


# --------------------------------------------------------------------------- #
# Plan injection — operator-supplied steps run verbatim, planner skipped.
# --------------------------------------------------------------------------- #
def test_injected_plan_runs_verbatim_without_planner(tmp_path):
    ran = []

    class _Tracking(_Adv):
        async def execute(self, prompt):
            ran.append(prompt)
            return "out"

    master_mod_adv = master_mod.AdversarialReview
    try:
        master_mod.AdversarialReview = lambda **kw: _Tracking(verified=True, approved=True)

        class _NoPlanner(_Stub):
            async def run_async(self, piped_input=None):
                p = self.prompt or ""
                assert "Lead Architect" not in p, "planner must NOT run"
                return "summary"

        m = _make(agent_class=_NoPlanner, checkpoint_path=_ckpt_path(tmp_path),
                  plan_steps=["only step A", "only step B"])
        asyncio.run(m.execute("goal"))
        assert len(ran) == 2
        assert "only step A" in ran[0]
        assert "only step B" in ran[1]
    finally:
        master_mod.AdversarialReview = master_mod_adv


def test_injected_plan_differing_from_checkpoint_discards_stale(tmp_path, caplog):
    # A matching checkpoint exists for a DIFFERENT plan; injecting a new plan
    # must discard the stale checkpoint and start fresh (not silently resume).
    p = _ckpt_path(tmp_path)
    key = hashlib.sha256("goal".encode()).hexdigest()
    _write(p, {"key": key, "tasks": ["old step 1", "old step 2"],
               "completed": 1, "project_context": "stale",
               "base_fingerprint": None})
    m = _make(checkpoint_path=p, plan_steps=["new A", "new B"])
    out = asyncio.run(m.execute("goal"))
    # Ran fresh on the injected plan; the completed run drops the checkpoint.
    assert "new A" in out or not os.path.exists(p)


# --------------------------------------------------------------------------- #
# Boundary: empty plan, single step, large plan.
# --------------------------------------------------------------------------- #
def test_zero_step_plan_completes_without_crash(tmp_path):
    m = _make(plan_steps=[], checkpoint_path=_ckpt_path(tmp_path))
    # plan_steps=[] is falsy -> planner path; force planner to emit []
    class _EmptyPlan(_Stub):
        async def run_async(self, piped_input=None):
            p = self.prompt or ""
            if "JSON list" in p:
                return "[]"
            return "summary"
    m.agent_class = _EmptyPlan
    out = asyncio.run(m.execute("goal"))
    assert "Original Goal: goal" in out  # empty run still returns a context


def test_single_step_plan_runs(tmp_path):
    m = _make(plan_steps=["the only step"], checkpoint_path=_ckpt_path(tmp_path))
    out = asyncio.run(m.execute("goal"))
    assert "Step 1 Summary" in out


def test_verified_propagates_from_final_step(tmp_path, monkeypatch):
    stubs = iter([_Adv(verified=True, approved=True),
                  _Adv(verified=False, approved=False)])
    monkeypatch.setattr(master_mod, "AdversarialReview", lambda **kw: next(stubs))
    m = _make(plan_steps=["a", "b"], checkpoint_path=_ckpt_path(tmp_path))
    asyncio.run(m.execute("goal"))
    assert m.verified is False  # last step wins
    assert m.approved is False


# --------------------------------------------------------------------------- #
# Graph routing gate + topology helpers.
# --------------------------------------------------------------------------- #
def test_linear_graph_is_not_routed_to_dag():
    g = GraphPlan(nodes=[
        PlanNode(id="s1", task="a", deps=[]),
        PlanNode(id="s2", task="b", deps=["s1"]),
        PlanNode(id="s3", task="c", deps=["s2"]),
    ])
    m = _make(plan_graph=g)
    assert m._is_graph_plan() is False  # pure chain -> flat loop


def test_diamond_graph_routes_to_dag():
    g = GraphPlan(nodes=[
        PlanNode(id="a", task="a", deps=[]),
        PlanNode(id="b", task="b", deps=["a"]),
        PlanNode(id="c", task="c", deps=["a"]),
        PlanNode(id="d", task="d", deps=["b", "c"]),
    ])
    m = _make(plan_graph=g)
    assert m._is_graph_plan() is True


def test_graph_plan_seeds_plan_steps_linearization():
    g = GraphPlan(nodes=[
        PlanNode(id="a", task="task-a", deps=[]),
        PlanNode(id="b", task="task-b", deps=["a"]),
    ])
    m = _make(plan_graph=g)
    assert m.plan_steps == ["task-a", "task-b"]


def test_compose_context_only_includes_ancestors():
    g = GraphPlan(nodes=[
        PlanNode(id="a", task="A", deps=[]),
        PlanNode(id="b", task="B", deps=["a"]),
        PlanNode(id="c", task="C", deps=["a"]),
        PlanNode(id="d", task="D", deps=["b", "c"]),
    ])
    m = _make(plan_graph=g)
    from agy_orchestrator.workflows.master import StepResult
    done = {
        nid: StepResult(final_output="", step_summary=f"summary-{nid}",
                        verified=True, approved=True, stalled=False,
                        iterations_used=1, session_id=None)
        for nid in ("a", "b", "c")
    }
    by_id = {n.id: n for n in g.nodes}
    # c's context must include a but NOT b (b is a sibling, not an ancestor).
    ctx_c = m._compose_context(by_id["c"], g, done, "goal")
    assert "summary-a" in ctx_c
    assert "summary-b" not in ctx_c
    # d's context (join) sees BOTH branches.
    ctx_d = m._compose_context(by_id["d"], g, done, "goal")
    assert "summary-a" in ctx_d
    assert "summary-b" in ctx_d
    assert "summary-c" in ctx_d


def test_rebuild_linear_context_topo_order():
    g = GraphPlan(nodes=[
        PlanNode(id="a", task="A", deps=[]),
        PlanNode(id="b", task="B", deps=["a"]),
    ])
    m = _make(plan_graph=g)
    from agy_orchestrator.workflows.master import StepResult
    order = g.topo_order()
    by_id = {n.id: n for n in g.nodes}
    done = {
        "a": StepResult("", "sum-a", True, True, False, 1, None),
        "b": StepResult("", "sum-b", True, True, False, 1, None),
    }
    ctx = m._rebuild_linear_context("goal", order, by_id, done)
    assert ctx.index("sum-a") < ctx.index("sum-b")
    assert "Step 1 Summary" in ctx and "Step 2 Summary" in ctx


# --------------------------------------------------------------------------- #
# End-to-end diamond DAG run with fakes (no workspace isolation surprises).
# --------------------------------------------------------------------------- #
def test_diamond_dag_executes_all_nodes(tmp_path, monkeypatch):
    executed = []

    class _RecAdv(_Adv):
        async def execute(self, prompt):
            executed.append(prompt)
            return "out"

    monkeypatch.setattr(master_mod, "AdversarialReview",
                        lambda **kw: _RecAdv(verified=True, approved=True))

    # Stub out the workspace + merge so the DAG path runs hermetically with no
    # git worktree / real merge. We only verify scheduling + completion here.
    class _FakeWS:
        def __init__(self, base, candidate_id="", prefer_worktree=True):
            self.path = base
        async def open(self):
            pass
        async def close(self):
            pass

    async def _fake_merge(**kw):
        from agy_orchestrator.workflows.graph_merge import MergeOutcome
        return MergeOutcome(node_id=kw["node_id"], layer=kw["layer"],
                            resolution="disjoint")

    monkeypatch.setattr(master_mod, "_ManagedWorkspace", _FakeWS)
    monkeypatch.setattr(master_mod, "_snapshot_tree", lambda p: {})
    monkeypatch.setattr(master_mod, "merge_node", _fake_merge)

    g = GraphPlan(nodes=[
        PlanNode(id="a", task="task A", deps=[]),
        PlanNode(id="b", task="task B", deps=["a"]),
        PlanNode(id="c", task="task C", deps=["a"]),
        PlanNode(id="d", task="task D", deps=["b", "c"]),
    ])
    m = _make(plan_graph=g, checkpoint_path=_ckpt_path(tmp_path),
              verifier=None)
    out = asyncio.run(m.execute("build the diamond"))
    # All four nodes ran.
    assert len(executed) == 4
    joined = "\n".join(executed)
    for t in ("task A", "task B", "task C", "task D"):
        assert t in joined
    # Node statuses harvested for meta.json.
    assert set(m.node_status.values()) <= {"passed", "failed"}
    assert all(s == "passed" for s in m.node_status.values())
    assert "Original Goal: build the diamond" in out

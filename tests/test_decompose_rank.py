"""Tests for the best-of-K ranking and as-needed decomposition workflows.

All model-free: generators are tiny canned-reply stubs subclassing the
AgentInstance contract, and the decomposer is driven by plain callables (no
subprocess, no real models) — mirroring tests/test_massaging.py.
"""
from __future__ import annotations

import asyncio

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.workflows.generate_and_rank import GenerateAndRankWorkflow
from agy_orchestrator.workflows.decompose import AdaptiveDecomposer


# --------------------------------------------------------------------------- #
# Fake agents (same shape as test_massaging.py)
# --------------------------------------------------------------------------- #
class _CannedAgent(AgentInstance):
    """Returns a fixed string; counts how many times it was run."""

    def __init__(self, reply: str = "", prompt: str = ""):
        super().__init__(prompt=prompt)
        self._reply = reply
        self.calls = 0

    @classmethod
    async def get_available_models(cls):
        return []

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        return 100.0

    def build_command(self, piped_input=None):
        return []

    async def run_async(self, piped_input=None) -> str:
        self.calls += 1
        return self._reply


class _ScriptedCritic(_CannedAgent):
    """Yields a sequence of replies in order, then repeats the final one."""

    def __init__(self, replies):
        super().__init__()
        self._replies = list(replies)
        self._i = 0

    async def run_async(self, piped_input=None) -> str:
        self.calls += 1
        r = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        return r


# --------------------------------------------------------------------------- #
# (1) GenerateAndRank — verifier gate (K==1) + judge ranking (K>1)
# --------------------------------------------------------------------------- #
def test_rank_single_candidate_verifier_gate_passes():
    """K==1: the verifier acts as a pass/fail gate on the resident artifact."""

    class _Verifier:
        async def verify(self, working_directory="."):
            return (True, "")

    gen = _CannedAgent("the only candidate")
    wf = GenerateAndRankWorkflow([gen], verifier=_Verifier())
    out = asyncio.run(wf.execute("do it"))
    assert out == "the only candidate"
    assert wf.verified is True
    assert wf.n_candidates == 1
    assert wf.n_passed == 1


def test_rank_single_candidate_verifier_gate_fails():
    """K==1: a failing verifier sets verified=False but still returns the candidate."""

    class _Verifier:
        async def verify(self, working_directory="."):
            return (False, "AssertionError: x != y")

    gen = _CannedAgent("broken candidate")
    wf = GenerateAndRankWorkflow([gen], verifier=_Verifier())
    out = asyncio.run(wf.execute("do it"))
    assert out == "broken candidate"
    assert wf.verified is False
    assert wf.n_passed == 0


def test_rank_falls_back_to_judge_when_many_candidates():
    """K>1: ranking is driven by the LLM judge scoring the candidate TEXT."""
    a = _CannedAgent("weak answer")
    b = _CannedAgent("strong answer")
    c = _CannedAgent("middling answer")

    # Candidates are scored in PARALLEL with independent (cloned) rankers, so the
    # judge scores by the content handed to it, not by call order.
    class _ContentJudge(_CannedAgent):
        async def run_async(self, piped_input=None):
            self.calls += 1
            if "strong" in self.prompt:
                return "score: 9"
            if "middling" in self.prompt:
                return "score: 5"
            return "score: 3"

    wf = GenerateAndRankWorkflow([a, b, c], ranker=_ContentJudge())
    out = asyncio.run(wf.execute("do it"))
    assert out == "strong answer"
    assert wf.n_candidates == 3


def test_rank_many_judge_ties_break_by_first_occurrence():
    """K>1 with no verifier: equal judge scores keep candidate order (stable argmax)."""
    a = _CannedAgent("first")
    b = _CannedAgent("second")
    judge = _ScriptedCritic(["score: 7", "score: 7"])
    wf = GenerateAndRankWorkflow([a, b], ranker=judge)
    out = asyncio.run(wf.execute("do it"))
    assert out == "first"


def test_rank_many_no_ranker_returns_first():
    """K>1 with neither verifier-reorder nor ranker: return the first candidate."""
    a = _CannedAgent("alpha")
    b = _CannedAgent("beta")
    wf = GenerateAndRankWorkflow([a, b])
    out = asyncio.run(wf.execute("do it"))
    assert out == "alpha"
    assert wf.n_candidates == 2


def test_rank_requires_at_least_one_generator():
    import pytest
    with pytest.raises(ValueError):
        GenerateAndRankWorkflow([])


# --------------------------------------------------------------------------- #
# (2) AdaptiveDecomposer — solve direct, recurse on failure, respect max_depth
# --------------------------------------------------------------------------- #
def test_decompose_solves_directly_without_splitting():
    """First attempt succeeds -> no decomposition at all."""
    decompose_calls = {"n": 0}

    def solve_one(task):
        return (f"solved:{task}", True)

    def decompose(task):
        decompose_calls["n"] += 1
        return ["never used"]

    d = AdaptiveDecomposer()
    out, ok = d.run("big task", solve_one, decompose)
    assert ok is True
    assert out == "solved:big task"
    assert decompose_calls["n"] == 0       # decompose never invoked
    assert d.n_decompositions == 0
    assert d.depth_reached == 0
    assert d.root.ok is True
    assert d.root.children == []


def test_decompose_recurses_on_failure_and_aggregates():
    """Root fails -> split into subtasks that each succeed -> aggregated output."""

    def solve_one(task):
        # The root task fails; its two subtasks succeed.
        if task == "root":
            return ("partial", False)
        return (f"done:{task}", True)

    def decompose(task):
        return ["sub-a", "sub-b"]

    d = AdaptiveDecomposer()
    out, ok = d.run("root", solve_one, decompose)
    assert ok is True
    assert out == "done:sub-a\ndone:sub-b"
    assert d.n_decompositions == 1
    assert d.depth_reached == 1
    assert len(d.root.children) == 2
    assert all(c.ok for c in d.root.children)


def test_decompose_overall_fails_if_any_subtask_fails():
    """A task succeeds only if ALL its subtasks do (AND semantics)."""

    def solve_one(task):
        if task == "root":
            return ("", False)
        return (f"done:{task}", task != "sub-b")  # sub-b fails even decomposed

    def decompose(task):
        if task == "root":
            return ["sub-a", "sub-b"]
        return []  # subtasks can't decompose further

    d = AdaptiveDecomposer(max_depth=2)
    out, ok = d.run("root", solve_one, decompose)
    assert ok is False
    assert d.root.ok is False


def test_decompose_respects_max_depth():
    """A task that always fails stops recursing at max_depth."""
    depths_seen = []

    def solve_one(task):
        return ("nope", False)  # everything fails -> always tries to decompose

    def decompose(task):
        # Each level splits into one deeper subtask, so depth would grow unbounded
        # if max_depth were not enforced.
        return [f"{task}.child"]

    d = AdaptiveDecomposer(max_depth=2)
    out, ok = d.run("root", solve_one, decompose)
    assert ok is False
    # depth 0 (root) -> decompose -> depth 1 -> decompose -> depth 2 (max, no more split).
    assert d.depth_reached == 2
    assert d.n_decompositions == 2  # decomposed at depth 0 and depth 1 only


def test_decompose_respects_max_subtasks():
    """decompose() output is capped at max_subtasks."""

    def solve_one(task):
        return (f"ok:{task}", task != "root")  # only root fails

    def decompose(task):
        return [f"s{i}" for i in range(10)]  # 10 proposed, cap should trim

    d = AdaptiveDecomposer(max_depth=1, max_subtasks=3)
    out, ok = d.run("root", solve_one, decompose)
    assert ok is True
    assert len(d.root.children) == 3       # capped from 10 to 3

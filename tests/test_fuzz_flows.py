"""Hermetic fuzz / property / edge-case tests for the test-time workflows:
tree_of_thought, generate_and_rank, decompose, test_feedback, cascade, pat.

ALL model-free and network-free: agents are tiny canned/scripted stubs
subclassing the AgentInstance contract (no subprocess, no real provider), the
verifier is a fake returning a VerifierResult, and the decomposer is driven by
plain callables. Each test runs in milliseconds and needs no credentials.

These assert CORRECT, robust behaviour under adversarial inputs (ties, all-equal
scores, judge crashes, empty / boundary configs, malformed replies). Genuine
defects found while fuzzing are reported separately (not committed as failing
tests) so this file stays green.
"""
from __future__ import annotations

import asyncio

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import VerifierResult
from agy_orchestrator.workflows.cascade import CascadeWorkflow
from agy_orchestrator.workflows.decompose import AdaptiveDecomposer
from agy_orchestrator.workflows.generate_and_rank import GenerateAndRankWorkflow
from agy_orchestrator.workflows.pat import PatWorkflow
# Aliased so pytest does not try to collect the workflow class as a test case.
from agy_orchestrator.workflows.test_feedback import (
    TestFeedbackWorkflow as FeedbackWorkflow,
)
from agy_orchestrator.workflows.tree_of_thought import (
    TreeOfThought,
    _normalize,
    _parse_score,
    build_judge_prompt,
)


# --------------------------------------------------------------------------- #
# Stubs (same shape as tests/test_decompose_rank.py)
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


class _ScriptedAgent(_CannedAgent):
    """Yields a sequence of replies in order, repeating the final one."""

    def __init__(self, replies):
        super().__init__()
        self._replies = list(replies)
        self._i = 0

    async def run_async(self, piped_input=None) -> str:
        self.calls += 1
        r = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        return r


class _RaisingAgent(_CannedAgent):
    """run_async always raises — models a usage wall / crashed worker."""

    async def run_async(self, piped_input=None) -> str:
        self.calls += 1
        raise RuntimeError("simulated worker crash")


class _FakeVerifier:
    """Returns a scripted sequence of VerifierResults (last one repeats)."""

    def __init__(self, results):
        self._results = list(results)
        self._i = 0
        self.calls = 0

    async def verify(self, working_directory: str = ".") -> VerifierResult:
        self.calls += 1
        r = self._results[min(self._i, len(self._results) - 1)]
        self._i += 1
        return r


def _ok(msg="ok"):
    return VerifierResult(ok=True, message=msg)


def _fail(msg="boom"):
    return VerifierResult(ok=False, message=msg)


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# _parse_score: adversarial / control-char / boundary inputs never crash and
# always clamp to [0, 10].
# --------------------------------------------------------------------------- #
def test_parse_score_clamps_and_never_crashes():
    cases = {
        "": 0,
        "no digits at all": 0,
        "rating: 999": 10,        # clamped high
        "score 0": 0,             # zero allowed
        "10/10": 10,
        "8/10": 8,                # not 810
        "\x00\x008\x00": 8,       # embedded NULs
        "<think>9</think>3": 3,   # think block stripped, visible verdict wins
        "the value is 008": 8,    # leading zeros
        "score -5": 5,            # the regex grabs the 5; still in range
    }
    for reply, expected in cases.items():
        score = _parse_score(reply)
        assert 0 <= score <= 10
        assert score == expected, (reply, score, expected)


def test_parse_score_huge_string_is_bounded():
    big = ("blah " * 100000) + "score 7"
    assert _parse_score(big) == 7


def test_normalize_collapses_whitespace_and_think():
    assert _normalize("<think>x</think>  A   B  ") == "a b"
    assert _normalize("") == ""
    assert _normalize("   ") == ""


def test_build_judge_prompt_with_and_without_requirement():
    p1 = build_judge_prompt("CODE", noun="solution")
    assert "CODE" in p1 and "Requirement" not in p1
    p2 = build_judge_prompt("CODE", noun="candidate", requirement="REQ")
    assert "CODE" in p2 and "REQ" in p2


# --------------------------------------------------------------------------- #
# TreeOfThought — vote selector
# --------------------------------------------------------------------------- #
def test_tot_vote_majority_wins():
    branches = [_CannedAgent("A"), _CannedAgent("B"), _CannedAgent("A")]
    tot = TreeOfThought(branches, _CannedAgent("5"), selector="vote")
    assert run(tot.execute()) == "A"


def test_tot_vote_all_distinct_tie_picks_earliest():
    branches = [_CannedAgent("X"), _CannedAgent("Y"), _CannedAgent("Z")]
    tot = TreeOfThought(branches, _CannedAgent("5"), selector="vote")
    assert run(tot.execute()) == "X"


def test_tot_vote_two_way_tie_picks_earliest_occurrence():
    # a,b,a,b -> a and b each 2; earliest first-occurrence (a) wins.
    branches = [_CannedAgent("a"), _CannedAgent("b"), _CannedAgent("a"), _CannedAgent("b")]
    tot = TreeOfThought(branches, _CannedAgent("5"), selector="vote")
    assert run(tot.execute()) == "a"


def test_tot_vote_normalizes_whitespace_for_agreement():
    # "  hi   there " and "hi there" normalize equal -> 2 votes, beats the singleton.
    branches = [_CannedAgent("  hi   there "), _CannedAgent("other"), _CannedAgent("hi there")]
    tot = TreeOfThought(branches, _CannedAgent("5"), selector="vote")
    out = run(tot.execute())
    assert out in ("  hi   there ", "hi there")  # one of the agreeing pair


def test_tot_single_branch_skips_evaluator():
    # One surviving branch => returned without an evaluator pass.
    ev = _CannedAgent("5")
    tot = TreeOfThought([_CannedAgent("only")], ev, selector="judge")
    assert run(tot.execute()) == "only"
    assert ev.calls == 0


def test_tot_one_branch_survives_swarm_crash():
    # Two branches, one crashes; the survivor is returned with no judge pass.
    ev = _CannedAgent("5")
    tot = TreeOfThought([_RaisingAgent(), _CannedAgent("survivor")], ev, selector="vote")
    assert run(tot.execute()) == "survivor"


# --------------------------------------------------------------------------- #
# TreeOfThought — judge selector
# --------------------------------------------------------------------------- #
def test_tot_judge_picks_highest_score():
    # Use a sentinel token that does NOT appear in the judge rubric so the score
    # tracks the candidate text deterministically.
    branches = [_CannedAgent("ZZLOW"), _CannedAgent("ZZWIN")]
    class _ScoreByContent(_CannedAgent):
        async def run_async(self, piped_input=None) -> str:
            return "9" if "ZZWIN" in (self.prompt or "") else "2"
    tot = TreeOfThought(branches, _ScoreByContent(), selector="judge")
    assert run(tot.execute()) == "ZZWIN"


def test_tot_judge_all_equal_scores_picks_first():
    branches = [_CannedAgent("a"), _CannedAgent("b"), _CannedAgent("c")]
    tot = TreeOfThought(branches, _CannedAgent("7"), selector="judge")
    assert run(tot.execute()) == "a"


def test_tot_judge_crash_scores_zero_and_still_returns():
    # Evaluator always raises -> every branch scores 0 -> argmax = first branch.
    branches = [_CannedAgent("a"), _CannedAgent("b")]
    tot = TreeOfThought(branches, _RaisingAgent(), selector="judge")
    assert run(tot.execute()) == "a"


def test_tot_event_callback_exception_is_swallowed():
    def boom(_ev):
        raise RuntimeError("callback exploded")
    branches = [_CannedAgent("a"), _CannedAgent("a")]
    tot = TreeOfThought(branches, _CannedAgent("5"), selector="vote", event_callback=boom)
    # A throwing callback must not abort selection.
    assert run(tot.execute()) == "a"


# --------------------------------------------------------------------------- #
# GenerateAndRank
# --------------------------------------------------------------------------- #
def test_gar_requires_at_least_one_generator():
    try:
        GenerateAndRankWorkflow([])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_gar_single_candidate_verifier_gate_pass():
    wf = GenerateAndRankWorkflow([_CannedAgent("c1")], verifier=_FakeVerifier([_ok()]))
    assert run(wf.execute("p")) == "c1"
    assert wf.verified is True
    assert wf.n_candidates == 1 and wf.n_passed == 1


def test_gar_single_candidate_verifier_gate_fail():
    wf = GenerateAndRankWorkflow([_CannedAgent("c1")], verifier=_FakeVerifier([_fail()]))
    assert run(wf.execute("p")) == "c1"  # best-of-one still returned
    assert wf.verified is False and wf.n_passed == 0


def test_gar_single_candidate_no_verifier():
    wf = GenerateAndRankWorkflow([_CannedAgent("c1")])
    assert run(wf.execute("p")) == "c1"
    assert wf.verified is False


def test_gar_many_no_ranker_returns_first():
    wf = GenerateAndRankWorkflow([_CannedAgent("c1"), _CannedAgent("c2")])
    assert run(wf.execute("p")) == "c1"


def test_gar_many_judge_picks_best():
    class _ScoreByContent(_CannedAgent):
        async def run_async(self, piped_input=None) -> str:
            return "9" if "WIN" in (self.prompt or "") else "1"
    wf = GenerateAndRankWorkflow(
        [_CannedAgent("lose"), _CannedAgent("WIN")],
        ranker=_ScoreByContent(),
    )
    assert run(wf.execute("p")) == "WIN"


def test_gar_many_judge_all_crash_returns_first():
    wf = GenerateAndRankWorkflow(
        [_CannedAgent("c1"), _CannedAgent("c2")],
        ranker=_RaisingAgent(),
    )
    assert run(wf.execute("p")) == "c1"


def test_gar_many_judge_then_verifier_sets_signal_only():
    # Judge picks the winner; verifier (run once) sets verified without reordering.
    class _ScoreByContent(_CannedAgent):
        async def run_async(self, piped_input=None) -> str:
            return "9" if "WIN" in (self.prompt or "") else "1"
    wf = GenerateAndRankWorkflow(
        [_CannedAgent("lose"), _CannedAgent("WIN")],
        ranker=_ScoreByContent(),
        verifier=_FakeVerifier([_ok()]),
    )
    assert run(wf.execute("p")) == "WIN"
    assert wf.verified is True


# --------------------------------------------------------------------------- #
# AdaptiveDecomposer
# --------------------------------------------------------------------------- #
def test_decompose_solves_directly_no_split():
    dec = AdaptiveDecomposer()
    out, ok = dec.run("t", lambda t: ("done", True), lambda t: ["never"])
    assert ok is True and out == "done"
    assert dec.n_decompositions == 0 and dec.depth_reached == 0


def test_decompose_on_failure_splits_and_aggregates():
    dec = AdaptiveDecomposer(max_depth=2, max_subtasks=6)
    # Root fails; its two children solve directly.
    def solve(t):
        return (f"out:{t}", t != "root")
    def split(t):
        return ["a", "b"]
    out, ok = dec.run("root", solve, split)
    assert ok is True
    assert out == "out:a\nout:b"
    assert dec.n_decompositions == 1 and dec.depth_reached == 1


def test_decompose_and_is_strict():
    # All-AND: one failing child fails the parent.
    dec = AdaptiveDecomposer(max_depth=1, max_subtasks=6)
    def solve(t):
        return (f"out:{t}", t == "a")  # only "a" passes
    out, ok = dec.run("root", solve, lambda t: ["a", "b"])
    assert ok is False


def test_decompose_max_depth_zero_gives_up_without_split():
    dec = AdaptiveDecomposer(max_depth=0)
    out, ok = dec.run("root", lambda t: ("fail", False), lambda t: ["a", "b"])
    assert ok is False and out == "fail"
    assert dec.n_decompositions == 0


def test_decompose_empty_subtasks_gives_up():
    dec = AdaptiveDecomposer(max_depth=3)
    out, ok = dec.run("root", lambda t: ("fail", False), lambda t: [])
    assert ok is False and out == "fail"
    assert dec.n_decompositions == 0


def test_decompose_respects_max_subtasks_cap():
    seen = []
    def solve(t):
        seen.append(t)
        return ("o", t != "root")
    dec = AdaptiveDecomposer(max_depth=1, max_subtasks=2)
    dec.run("root", solve, lambda t: ["s1", "s2", "s3", "s4"])
    # root + first 2 subtasks only.
    assert seen == ["root", "s1", "s2"]


def test_decompose_cyclic_subtasks_terminate_by_depth():
    # decompose that always returns the same task is bounded by max_depth.
    dec = AdaptiveDecomposer(max_depth=4, max_subtasks=1)
    out, ok = dec.run("x", lambda t: ("o", False), lambda t: [t])
    assert ok is False
    assert dec.depth_reached == 4  # bounded, no infinite recursion


def test_decompose_as_dict_is_json_friendly():
    dec = AdaptiveDecomposer(max_depth=1, max_subtasks=2)
    dec.run("root", lambda t: ("o", t != "root"), lambda t: ["a", "b"])
    d = dec.root.as_dict()
    assert d["task"] == "root" and d["depth"] == 0
    assert [c["task"] for c in d["children"]] == ["a", "b"]
    import json
    json.dumps(d)  # must be serializable


# --------------------------------------------------------------------------- #
# TestFeedbackWorkflow
# --------------------------------------------------------------------------- #
def test_feedback_requires_verifier():
    try:
        FeedbackWorkflow(_CannedAgent(), None)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_feedback_passes_first_try():
    gen = _CannedAgent("good")
    wf = FeedbackWorkflow(gen, _FakeVerifier([_ok()]), max_iterations=4)
    assert run(wf.execute("p")) == "good"
    assert wf.verified is True and wf.iterations_used == 1
    assert gen.calls == 1


def test_feedback_repairs_then_passes():
    gen = _ScriptedAgent(["bad", "fixed"])
    wf = FeedbackWorkflow(gen, _FakeVerifier([_fail(), _ok()]), max_iterations=4)
    assert run(wf.execute("p")) == "fixed"
    assert wf.verified is True and wf.iterations_used == 2


def test_feedback_exhausts_iterations():
    gen = _CannedAgent("stillbad")
    wf = FeedbackWorkflow(gen, _FakeVerifier([_fail()]), max_iterations=3)
    assert run(wf.execute("p")) == "stillbad"
    assert wf.verified is False and wf.iterations_used == 3
    assert gen.calls == 3


def test_feedback_zero_iterations_no_work():
    gen = _CannedAgent("x")
    wf = FeedbackWorkflow(gen, _FakeVerifier([_fail()]), max_iterations=0)
    assert run(wf.execute("p")) == ""  # nothing produced
    assert wf.verified is False and wf.iterations_used == 0 and gen.calls == 0


def test_feedback_negative_iterations_no_work():
    gen = _CannedAgent("x")
    wf = FeedbackWorkflow(gen, _FakeVerifier([_fail()]), max_iterations=-7)
    assert run(wf.execute("p")) == ""
    assert wf.iterations_used == 0 and gen.calls == 0


def test_feedback_none_message_does_not_crash():
    gen = _ScriptedAgent(["a", "b"])
    wf = FeedbackWorkflow(gen, _FakeVerifier([_fail("__none__"), _ok()]), max_iterations=4)
    # Simulate a None message on the failing result by patching the result objects.
    wf.verifier._results[0] = VerifierResult(ok=False, message=None)
    assert run(wf.execute("p")) == "b"
    assert wf.verified is True


# --------------------------------------------------------------------------- #
# CascadeWorkflow
# --------------------------------------------------------------------------- #
def test_cascade_requires_stage_and_verifier():
    try:
        CascadeWorkflow([], _FakeVerifier([_ok()]))
        assert False
    except ValueError:
        pass
    try:
        CascadeWorkflow([_CannedAgent()], None)
        assert False
    except ValueError:
        pass


def test_cascade_first_stage_passes():
    s1, s2 = _CannedAgent("s1"), _CannedAgent("s2")
    c = CascadeWorkflow([s1, s2], _FakeVerifier([_ok()]), max_iterations_per_stage=2)
    assert run(c.execute("p")) == "s1"
    assert c.verified is True and c.stage_used == 0 and c.stalled is False
    assert s2.calls == 0  # second stage never reached


def test_cascade_escalates_to_second_stage():
    s1, s2 = _CannedAgent("s1"), _CannedAgent("s2")
    # Stage1: 1 iter fails. Stage2: 1 iter passes.
    c = CascadeWorkflow([s1, s2], _FakeVerifier([_fail(), _ok()]), max_iterations_per_stage=1)
    assert run(c.execute("p")) == "s2"
    assert c.verified is True and c.stage_used == 1
    assert c.iterations_used == 2  # one round per stage


def test_cascade_all_stages_fail_stalls():
    c = CascadeWorkflow([_CannedAgent("s1"), _CannedAgent("s2")],
                        _FakeVerifier([_fail()]), max_iterations_per_stage=1)
    out = run(c.execute("p"))
    assert out == "s2"  # last stage output surfaced
    assert c.verified is False and c.stage_used == -1 and c.stalled is True
    assert c.iterations_used == 2


# --------------------------------------------------------------------------- #
# PatWorkflow
# --------------------------------------------------------------------------- #
class _FakeMaster:
    def __init__(self, out="master-out", verified=False, approved=False, iterations_used=3):
        self._out = out
        self.verified = verified
        self.approved = approved
        self.iterations_used = iterations_used
        self.executed_prompt = None

    async def execute(self, prompt: str) -> str:
        self.executed_prompt = prompt
        return self._out


def test_pat_requires_verifier_generator_master():
    m = _FakeMaster()
    try:
        PatWorkflow(_CannedAgent(), m, None)
        assert False
    except ValueError:
        pass
    try:
        PatWorkflow(None, m, _FakeVerifier([_ok()]))
        assert False
    except ValueError:
        pass
    try:
        PatWorkflow(_CannedAgent(), None, _FakeVerifier([_ok()]))
        assert False
    except ValueError:
        pass


def test_pat_stage1_passes_skips_master():
    m = _FakeMaster()
    pat = PatWorkflow(_CannedAgent("direct"), m, _FakeVerifier([_ok()]))
    assert run(pat.execute("p")) == "direct"
    assert pat.verified is True and pat.stage_used == 0
    assert pat.approved is True and pat.stalled is False
    assert m.executed_prompt is None  # master never ran


def test_pat_escalates_and_reverifies_when_master_unverified():
    m = _FakeMaster(out="master-out", verified=False)
    v = _FakeVerifier([_fail(), _ok()])  # stage1 fails, final re-verify passes
    pat = PatWorkflow(_CannedAgent("direct"), m, v)
    assert run(pat.execute("p")) == "master-out"
    assert pat.stage_used == 1 and pat.verified is True
    assert pat.approved is True and pat.stalled is False
    assert m.executed_prompt is not None and "direct" in m.executed_prompt


def test_pat_escalates_master_self_verified_no_reverify():
    m = _FakeMaster(out="m", verified=True)
    v = _FakeVerifier([_fail()])  # only the stage1 verify happens
    pat = PatWorkflow(_CannedAgent("direct"), m, v)
    assert run(pat.execute("p")) == "m"
    assert pat.verified is True and v.calls == 1  # no second verify call


def test_pat_escalation_stalls_when_all_fail():
    m = _FakeMaster(out="m", verified=False, approved=False)
    v = _FakeVerifier([_fail()])  # stage1 fails AND final re-verify fails
    pat = PatWorkflow(_CannedAgent("direct"), m, v)
    assert run(pat.execute("p")) == "m"
    assert pat.verified is False and pat.stalled is True
    assert pat.iterations_used == 1 + m.iterations_used


def test_pat_none_message_in_escalation_prompt_does_not_crash():
    m = _FakeMaster(out="m", verified=True)
    v = _FakeVerifier([VerifierResult(ok=False, message=None)])
    pat = PatWorkflow(_CannedAgent("direct"), m, v)
    assert run(pat.execute("p")) == "m"
    assert "None" in m.executed_prompt  # message rendered, no crash


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))

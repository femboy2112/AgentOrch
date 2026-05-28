"""Tests for the quality-massaging primitives:
  * robust ToT score parsing + majority-vote selector
  * robust adversarial convergence + diff-only revise prompt
  * test-feedback and cascade escalation loops

All model-free: branch/evaluator/critic agents are tiny canned-reply stubs that
subclass the AgentInstance contract (no subprocess, no real models).
"""
from __future__ import annotations

import asyncio

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import VerifierResult
from agy_orchestrator.workflows.adversarial import AdversarialReview
from agy_orchestrator.workflows.tree_of_thought import TreeOfThought


# --------------------------------------------------------------------------- #
# Fake agents
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
    """Generator-ish stub that yields a sequence of replies in order, then the
    final one repeats. Lets a test drive a multi-iteration adversarial loop."""

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
# (1) ToT — robust scoring + vote selector
# --------------------------------------------------------------------------- #
def test_score_parsing_handles_fraction_and_chatter():
    branches = [_CannedAgent("solution A")]
    evaluator = _CannedAgent("I rate this 8/10")
    best = asyncio.run(TreeOfThought(branches, evaluator).execute())
    assert best == "solution A"  # picked (single branch)
    # Re-run the parser via a 2-branch tie-break to confirm 8 (not 810) was used.
    from agy_orchestrator.workflows.tree_of_thought import _parse_score
    assert _parse_score("I rate this 8/10") == 8
    assert _parse_score("I'd rate this 7.") == 7
    assert _parse_score("garbage with no number") == 0
    assert _parse_score("99") == 10  # clamped to [0, 10]
    # Chatty judge that enumerates issues before its verdict: explicit pattern wins.
    assert _parse_score("Issue 1: a bug. Issue 2: style. Overall score: 9") == 9
    # No explicit pattern -> the verdict (last integer) wins, not the first.
    assert _parse_score("Problem 1 here, problem 2 there. Final: 6") == 6


def test_parallel_swarm_tolerates_a_failed_branch():
    """ToT/Master must survive one branch crashing — survivors still selected."""
    import asyncio as _a

    from agy_orchestrator.execution.pipeline import ParallelSwarm

    class _Boom(_CannedAgent):
        async def run_async(self, piped_input=None):
            raise RuntimeError("usage wall")

    good = _CannedAgent("survivor")
    out = _a.run(ParallelSwarm([_Boom(), good]).execute())
    assert out == ["survivor"]


def test_judge_picks_higher_score():
    branches = [_CannedAgent("alpha"), _CannedAgent("omega")]

    # Branches are scored in PARALLEL with independent (cloned) evaluators, so the
    # judge must score by the content it is handed, not by call order. Markers are
    # chosen to not collide with words in the rubric prompt.
    class _ContentJudge(_CannedAgent):
        async def run_async(self, piped_input=None):
            self.calls += 1
            return "score: 9" if "omega" in self.prompt else "score: 3"

    best = asyncio.run(TreeOfThought(branches, _ContentJudge()).execute())
    assert best == "omega"


def test_vote_selector_majority_and_no_evaluator_call():
    branches = [_CannedAgent("A"), _CannedAgent("A"), _CannedAgent("B")]
    evaluator = _CannedAgent("should never be called")
    best = asyncio.run(
        TreeOfThought(branches, evaluator, selector="vote").execute()
    )
    assert best == "A"
    assert evaluator.calls == 0  # vote spends ZERO evaluator passes


def test_vote_strips_think_block_in_normalization():
    # Two reasoning-model answers with different <think> but identical answers
    # must be counted as the same vote and beat the lone third answer.
    a1 = _CannedAgent("<think>long reasoning one</think>The answer is 42.")
    a2 = _CannedAgent("<think>totally different reasoning</think>the answer is 42.")
    b = _CannedAgent("The answer is 7.")
    best = asyncio.run(
        TreeOfThought([a1, a2, b], _CannedAgent(), selector="vote").execute()
    )
    assert "42" in best


def test_vote_ties_break_by_first_occurrence():
    branches = [_CannedAgent("X"), _CannedAgent("Y")]
    best = asyncio.run(
        TreeOfThought(branches, _CannedAgent(), selector="vote").execute()
    )
    assert best == "X"  # 1-1 tie -> earliest


# --------------------------------------------------------------------------- #
# (2) Adversarial — robust convergence + diff-only
# --------------------------------------------------------------------------- #
def test_exact_approved_converges():
    gen = _CannedAgent("the output")
    critic = _CannedAgent("APPROVED")
    out = asyncio.run(AdversarialReview(gen, critic, max_iterations=3).execute("do it"))
    assert out == "the output"
    assert gen.calls == 1  # converged on first round


def test_chatty_non_approval_does_not_converge():
    gen = _CannedAgent("the output")
    critic = _CannedAgent("This is not APPROVED, fix X")
    out = asyncio.run(AdversarialReview(gen, critic, max_iterations=2).execute("do it"))
    # Never approved -> loops to the cap and returns the last output.
    assert out == "the output"
    assert gen.calls == 2


def test_i_would_not_write_approved_does_not_converge():
    gen = _CannedAgent("out")
    critic = _CannedAgent("I would not write APPROVED until you fix the bug")
    out = asyncio.run(AdversarialReview(gen, critic, max_iterations=2).execute("do it"))
    assert out == "out"
    assert gen.calls == 2  # ran to cap, no false-positive convergence


def test_last_line_approved_converges():
    gen = _CannedAgent("out")
    critic = _CannedAgent("Looks correct and complete.\nAPPROVED.")
    out = asyncio.run(AdversarialReview(gen, critic, max_iterations=3).execute("do it"))
    assert out == "out"
    assert gen.calls == 1


def test_approved_inside_think_block_does_not_converge():
    gen = _CannedAgent("out")
    critic = _CannedAgent("<think>I could say APPROVED</think>No, fix the loop bound.")
    out = asyncio.run(AdversarialReview(gen, critic, max_iterations=2).execute("do it"))
    assert out == "out"
    assert gen.calls == 2


def test_diff_only_changes_revise_prompt():
    # Force one non-approving round so a revise prompt is built, then capture it.
    gen = _CannedAgent("draft")
    critic = _CannedAgent("needs work")

    plain = AdversarialReview(gen, critic, max_iterations=2, diff_only=False)
    asyncio.run(plain.execute("do it"))
    plain_prompt = gen.prompt

    gen2 = _CannedAgent("draft")
    critic2 = _CannedAgent("needs work")
    diff = AdversarialReview(gen2, critic2, max_iterations=2, diff_only=True)
    asyncio.run(diff.execute("do it"))
    diff_prompt = gen2.prompt

    assert "byte-identical" in diff_prompt
    assert "byte-identical" not in plain_prompt


# --------------------------------------------------------------------------- #
# (3) test-feedback + cascade escalation
# --------------------------------------------------------------------------- #
def test_test_feedback_loops_until_verifier_passes():
    """TestFeedbackWorkflow: generator revises until the verifier (strong oracle) passes."""
    import asyncio as _a

    from agy_orchestrator.workflows.test_feedback import TestFeedbackWorkflow

    gen = _ScriptedCritic(["bad code v1", "good code v2"])

    class _Verifier:  # passes only once it sees the second output attempt
        def __init__(self): self.calls = 0
        async def verify(self, working_directory="."):
            self.calls += 1
            ok = self.calls >= 2
            return VerifierResult(
                ok=ok,
                message="" if ok else "AssertionError: x != y",
                returncode=0 if ok else 1,
                cmd="stub",
                duration_ms=0,
            )

    out = _a.run(TestFeedbackWorkflow(gen, _Verifier(), max_iterations=4).execute("build X"))
    assert out == "good code v2"


def test_test_feedback_requires_verifier():
    import pytest

    from agy_orchestrator.workflows.test_feedback import TestFeedbackWorkflow
    with pytest.raises(ValueError):
        TestFeedbackWorkflow(_CannedAgent("x"), None)


def test_adversarial_stalls_on_repeated_critique():
    """A constant critique with a high cap should bail early (stuck), not burn all 5."""
    gen = _CannedAgent("draft")
    critic = _CannedAgent("Fix the same thing again")  # identical every round
    wf = AdversarialReview(gen, critic, max_iterations=5)
    out = asyncio.run(wf.execute("do it"))
    assert out == "draft"
    assert gen.calls == 2          # round 1 sets baseline, round 2 detects repeat -> stop
    assert wf.stalled is True
    assert wf.approved is False


def test_cascade_escalates_to_stronger_stage():
    """Cheap stage fails the verifier; cascade escalates to the strong stage that passes."""
    import asyncio as _a

    from agy_orchestrator.workflows.cascade import CascadeWorkflow

    cheap = _CannedAgent("cheap attempt")   # never passes
    strong = _CannedAgent("strong attempt")  # passes

    class _Verifier:
        async def verify(self, working_directory="."):
            # cheap stage's output fails; strong stage's output passes.
            # We can't see the output here, so gate on call count: stage1 (cheap,
            # 1 iter) fails, stage2 (strong) passes on its first try.
            self.n = getattr(self, "n", 0) + 1
            ok = self.n >= 2
            return VerifierResult(
                ok=ok,
                message="" if ok else "FAILED: needs work",
                returncode=0 if ok else 1,
                cmd="stub",
                duration_ms=0,
            )

    wf = CascadeWorkflow([cheap, strong], _Verifier(), max_iterations_per_stage=1)
    out = _a.run(wf.execute("do it"))
    assert wf.verified is True
    assert wf.stage_used == 1   # the strong stage
    assert out == "strong attempt"


def test_cascade_requires_verifier():
    import pytest

    from agy_orchestrator.workflows.cascade import CascadeWorkflow
    with pytest.raises(ValueError):
        CascadeWorkflow([_CannedAgent("x")], None)

from __future__ import annotations

import asyncio

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.workflows.spec import (
    DEFAULT_SPEC_SECTIONS,
    SpecWorkflow,
    build_architect_prompt,
    build_critic_prompt,
)


class _SeqAgent(AgentInstance):
    """Returns a queued reply per call (last reply repeats once exhausted),
    recording every prompt it saw."""

    def __init__(self, replies, prompt: str = ""):
        super().__init__(prompt=prompt)
        self._replies = list(replies)
        self.prompts_seen = []
        self.calls = 0

    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input=None) -> str:
        self.prompts_seen.append(self.prompt)
        self.calls += 1
        idx = min(self.calls - 1, len(self._replies) - 1)
        return self._replies[idx]


def test_architect_prompt_has_template_no_decomp_and_inputs():
    prompt = build_architect_prompt(
        "build a URL shortener", ["must use Postgres", "p99 < 50ms"],
        list(DEFAULT_SPEC_SECTIONS),
    )
    # Goal + constraints are present.
    assert "build a URL shortener" in prompt
    assert "must use Postgres" in prompt
    assert "p99 < 50ms" in prompt
    # The no-decomposition rule is explicit.
    assert "SYSTEM DESIGN ONLY" in prompt
    assert "task breakdown" in prompt
    # Provenance + honesty rails.
    assert "[assumption]" in prompt
    assert "Open Questions" in prompt
    # Sections are numbered into the prompt.
    assert "1. Overview" in prompt


def test_approval_first_round_returns_draft_and_no_revise():
    architect = _SeqAgent(["DRAFT-1"])
    critic = _SeqAgent(["APPROVED"])
    wf = SpecWorkflow(architect, critic, max_iterations=3)

    out = asyncio.run(wf.execute("a goal", ["c1"]))

    assert out == "DRAFT-1"
    assert wf.approved is True
    assert wf.stalled is False
    assert wf.iterations_used == 1
    # Architect authored once (draft), never revised.
    assert architect.calls == 1
    # Critic prompt carried the rubric + the draft + the constraint.
    assert "BUILDABILITY COMPLETENESS" in critic.prompts_seen[0]
    assert "DRAFT-1" in critic.prompts_seen[0]
    assert "c1" in critic.prompts_seen[0]


def test_revise_then_approve_uses_revised_draft():
    architect = _SeqAgent(["DRAFT-1", "DRAFT-2"])
    critic = _SeqAgent(["needs: add a data model section", "APPROVED"])
    wf = SpecWorkflow(architect, critic, max_iterations=3)

    out = asyncio.run(wf.execute("a goal", None))

    assert out == "DRAFT-2"
    assert wf.approved is True
    assert wf.iterations_used == 2
    # The revise prompt fed back the prior draft + the critique.
    revise_prompt = architect.prompts_seen[1]
    assert "DRAFT-1" in revise_prompt
    assert "add a data model section" in revise_prompt
    assert "SYSTEM DESIGN ONLY" in revise_prompt


def test_stall_on_unchanged_critique_bails_early():
    architect = _SeqAgent(["DRAFT-1", "DRAFT-2", "DRAFT-3"])
    # Same critique twice -> the gap set has converged; bail.
    critic = _SeqAgent(["fix the interfaces", "fix the interfaces", "APPROVED"])
    wf = SpecWorkflow(architect, critic, max_iterations=5)

    out = asyncio.run(wf.execute("a goal"))

    assert wf.stalled is True
    assert wf.approved is False
    # Round 1 critique -> revise -> round 2 identical critique -> stall.
    assert wf.iterations_used == 2
    assert out == "DRAFT-2"


def test_max_iterations_one_returns_first_draft_unapproved():
    architect = _SeqAgent(["ONLY-DRAFT"])
    critic = _SeqAgent(["please add error handling"])
    wf = SpecWorkflow(architect, critic, max_iterations=1)

    out = asyncio.run(wf.execute("a goal"))

    assert out == "ONLY-DRAFT"
    assert wf.approved is False
    assert wf.stalled is False
    assert wf.iterations_used == 1
    # No revise round attempted (only one iteration allotted).
    assert architect.calls == 1


def test_critic_prompt_flags_decomposition_check():
    prompt = build_critic_prompt("a goal", ["c1"], "the draft")
    assert "NO task breakdown" in prompt
    assert "APPROVED" in prompt
    assert "the draft" in prompt

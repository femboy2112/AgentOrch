from __future__ import annotations

import asyncio

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.workflows.adversarial import (
    CATASTROPHIC_FOCUS_PREAMBLE,
    AdversarialReview,
)


class _StubGenerator(AgentInstance):
    def __init__(self, reply: str = "generated output", prompt: str = ""):
        super().__init__(prompt=prompt)
        self._reply = reply
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
        self.calls += 1
        return self._reply


class _CapturingCritic(AgentInstance):
    def __init__(self, reply: str = "APPROVED", prompt: str = ""):
        super().__init__(prompt=prompt)
        self._reply = reply
        self.prompts_seen = []

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
        return self._reply


def test_critic_prompt_prepends_catastrophic_focus_preamble():
    gen = _StubGenerator(reply="result")
    critic = _CapturingCritic(reply="APPROVED")
    wf = AdversarialReview(
        gen,
        critic,
        max_iterations=3,
        critic_preamble=CATASTROPHIC_FOCUS_PREAMBLE,
    )

    out = asyncio.run(wf.execute("implement X"))

    assert out == "result"
    assert len(critic.prompts_seen) == 1
    prompt = critic.prompts_seen[0]
    assert prompt.startswith(CATASTROPHIC_FOCUS_PREAMBLE)
    assert "CRITICAL REVIEW INSTRUCTIONS" in prompt


def test_default_critic_prompt_has_no_catastrophic_preamble():
    gen = _StubGenerator(reply="result")
    critic = _CapturingCritic(reply="APPROVED")
    wf = AdversarialReview(gen, critic, max_iterations=3)

    out = asyncio.run(wf.execute("implement X"))

    assert out == "result"
    assert len(critic.prompts_seen) == 1
    prompt = critic.prompts_seen[0]
    assert CATASTROPHIC_FOCUS_PREAMBLE not in prompt
    assert prompt.startswith(
        "Please review the following output against the original requirement.\n"
    )
    assert "CRITICAL REVIEW INSTRUCTIONS" in prompt

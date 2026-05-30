from __future__ import annotations

import asyncio
from typing import List, Optional

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent
from agy_orchestrator.workflows.adversarial import AdversarialReview
from agy_orchestrator.workflows.master import MasterWorkflow
from agy_orchestrator.workflows.tree_of_thought import TreeOfThought


def _run(coro):
    return asyncio.run(coro)


def _orchestration_rows(events: List[dict]) -> List[dict]:
    out: List[dict] = []
    for evt in events:
        data = evt.get("data", {})
        if evt.get("kind") != "lifecycle" or data.get("event") != "orchestration_transition":
            continue
        orch = data.get("orchestration", {})
        if isinstance(orch, dict):
            out.append(orch)
    return out


class _StaticAgent(AgentInstance):
    def __init__(self, *args, reply: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.reply = reply

    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input: Optional[str] = None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        return self.reply


class _MasterStubAgent(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input: Optional[str] = None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        if "Lead Architect" in self.prompt:
            return '["Step one", "Step two"]'
        if "Please review the following output against the original requirement." in self.prompt:
            return "APPROVED"
        if "In 3-5 bullet points, summarize what was just implemented" in self.prompt:
            return "- summary"
        return "generated output"


class _JudgeByPrompt(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input: Optional[str] = None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        return "score: 9" if "beta" in self.prompt else "score: 2"


class _FallbackFail(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input: Optional[str] = None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        self.stderr = "simulated failure"
        raise RuntimeError("simulated failure")


class _FallbackOk(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input: Optional[str] = None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        self.returncode = 0
        self.stdout = "ok"
        return "ok"


def test_master_emits_plan_and_step_transitions():
    events: List[dict] = []
    wf = MasterWorkflow(
        model="m",
        effort="low",
        branches=1,
        max_iterations=2,
        agent_class=_MasterStubAgent,
        event_callback=events.append,
    )

    _run(wf.execute("build something"))
    rows = _orchestration_rows(events)

    plan_rows = [r for r in rows if r.get("phase") == "plan" and r.get("action") == "completed"]
    assert len(plan_rows) == 1
    assert plan_rows[0]["step_total"] == 2

    step_rows = [r for r in rows if r.get("phase") == "step"]
    assert [(r["action"], r["step_index"], r["step_total"]) for r in step_rows] == [
        ("started", 1, 2),
        ("completed", 1, 2),
        ("started", 2, 2),
        ("completed", 2, 2),
    ]


def test_tree_of_thought_emits_selected_branch():
    events: List[dict] = []
    branches = [_StaticAgent(prompt="x", reply="alpha"), _StaticAgent(prompt="x", reply="beta")]
    tot = TreeOfThought(
        branch_instances=branches,
        evaluator_instance=_JudgeByPrompt(prompt=""),
        selector="judge",
        event_callback=events.append,
    )

    out = _run(tot.execute())
    assert out == "beta"
    rows = _orchestration_rows(events)
    selected = [r for r in rows if r.get("phase") == "tot" and r.get("action") == "branch_selected"]
    assert len(selected) == 1
    assert selected[0]["selected_branch"] == 2


def test_adversarial_emits_iteration_start_and_completion():
    events: List[dict] = []
    wf = AdversarialReview(
        generator_instance=_StaticAgent(prompt="", reply="draft"),
        critic_instance=_StaticAgent(prompt="", reply="APPROVED"),
        max_iterations=3,
        event_callback=events.append,
    )

    out = _run(wf.execute("task"))
    assert out == "draft"
    rows = [r for r in _orchestration_rows(events) if r.get("phase") == "adversarial"]
    assert [(r["action"], r["iteration"], r["iteration_total"]) for r in rows] == [
        ("iteration_started", 1, 3),
        ("iteration_completed", 1, 3),
    ]


def test_fallback_emits_reroute_transition_on_rollover():
    events: List[dict] = []
    Agent = make_fallback_agent([_FallbackFail, _FallbackOk], cycles=1)
    agent = Agent(prompt="task")
    agent.event_callback = events.append

    out = _run(agent.run_async())
    assert out == "ok"
    rows = [r for r in _orchestration_rows(events) if r.get("phase") == "fallback"]
    assert len(rows) >= 1
    assert rows[0]["action"] == "reroute"
    assert rows[0]["from_worker"] == "_FallbackFail"
    assert rows[0]["to_worker"] == "_FallbackOk"
    assert rows[0]["reason"] == "error"


def test_callbacks_optional_no_crash():
    wf = MasterWorkflow(
        model="m",
        effort="low",
        branches=1,
        max_iterations=1,
        agent_class=_MasterStubAgent,
        event_callback=None,
    )
    out = _run(wf.execute("build something"))
    assert "Step 1 Summary" in out

    vote = TreeOfThought(
        branch_instances=[_StaticAgent(prompt="x", reply="one"), _StaticAgent(prompt="x", reply="one")],
        evaluator_instance=_StaticAgent(prompt="", reply="10"),
        selector="vote",
        event_callback=None,
    )
    assert _run(vote.execute()) == "one"

    adv = AdversarialReview(
        generator_instance=_StaticAgent(prompt="", reply="draft"),
        critic_instance=_StaticAgent(prompt="", reply="APPROVED"),
        max_iterations=1,
        event_callback=None,
    )
    assert _run(adv.execute("task")) == "draft"

    Agent = make_fallback_agent([_FallbackFail, _FallbackOk], cycles=1)
    assert _run(Agent(prompt="task").run_async()) == "ok"

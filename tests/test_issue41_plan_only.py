"""Issue #41 — --plan-only / --dry-run: emit the master/pat plan, write nothing.

master mode decomposes the instruction and immediately starts executing it,
writing to the live out-dir step by step. --plan-only runs ONLY the planner,
emits the decomposed plan (return value + orchestration event + plan_steps), and
exits before any worker touches the out-dir — no execution loop, no checkpoint.
"""
from __future__ import annotations

import asyncio

from agy_orchestrator.workflows.master import MasterWorkflow


def _planner_class(counter: list, plan_json: str):
    """Build a fake agent_class whose every instantiation bumps `counter` and
    whose run_async returns a fixed planner payload. Lets a test assert that
    plan-only instantiates the planner ONCE and no execution-phase workers."""

    class _Planner:
        def __init__(self, prompt=None, model=None, effort=None, **kwargs):
            counter.append(1)
            self.prompt = prompt
            self.session_id = "sess-1"

        async def run_async(self, piped_input=None):
            return plan_json

    return _Planner


PLAN = '["Step 1: do A", "Step 2: do B", "Step 3: do C"]'
STEPS = ["Step 1: do A", "Step 2: do B", "Step 3: do C"]


def test_plan_only_emits_plan_and_does_not_execute(tmp_path):
    counter: list = []
    events: list = []
    ckpt = tmp_path / "c.json"
    wf = MasterWorkflow(
        model="m", effort="high",
        agent_class=_planner_class(counter, PLAN),
        working_directory=str(tmp_path),
        checkpoint_path=str(ckpt),
        plan_only=True,
        event_callback=lambda e: events.append(e),
    )
    out = asyncio.run(wf.execute("build X"))

    assert wf.plan_steps == STEPS
    # ONLY the planner ran — no ToT / adversarial execution workers were spawned.
    assert len(counter) == 1
    # Human-readable plan returned, marked as a dry-run.
    assert "plan-only" in out.lower()
    assert "Step 1: do A" in out and "Step 3: do C" in out
    # Side-effect-free on disk: no checkpoint written.
    assert not ckpt.exists()
    # The plan was emitted as an orchestration event for the dashboard.
    plan_evts = [
        e for e in events
        if e.get("data", {}).get("orchestration", {}).get("phase") == "plan"
    ]
    assert plan_evts and plan_evts[0]["data"]["orchestration"]["step_total"] == 3


def test_plan_only_ignores_existing_checkpoint(tmp_path):
    # plan-only always re-plans (it shows the plan for THIS instruction), and a
    # checkpoint from a prior killed run must not divert it into resume.
    import json
    counter: list = []
    ckpt = tmp_path / "c.json"
    ckpt.write_text(json.dumps({
        "key": "whatever", "tasks": ["old step"], "completed": 0,
        "project_context": "", "session_id": None,
    }))
    wf = MasterWorkflow(
        model="m", effort="high",
        agent_class=_planner_class(counter, PLAN),
        working_directory=str(tmp_path),
        checkpoint_path=str(ckpt),
        plan_only=True,
    )
    asyncio.run(wf.execute("build X"))
    assert wf.plan_steps == STEPS          # fresh plan, not the checkpoint's "old step"
    assert len(counter) == 1


def test_non_plan_only_default_is_unchanged(tmp_path):
    # Sanity: plan_only defaults off and plan_steps starts None.
    wf = MasterWorkflow(model="m", effort="high", working_directory=str(tmp_path))
    assert wf.plan_only is False
    assert wf.plan_steps is None

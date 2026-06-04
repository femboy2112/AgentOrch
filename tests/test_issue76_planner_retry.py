"""Issue #76: MasterWorkflow._run_planner makes ONE planner call; on a JSON-array
parse failure it SILENTLY collapses to tasks=[initial_prompt] (one mega-step) with
only a WARNING — so a multi-deliverable --spec that IS plannable runs as a single
step, losing per-step verify + adversarial refinement.

Contract:
  (1) RETRY: a first unparseable reply triggers ONE strict JSON-only re-prompt; a
      successful re-prompt yields the real multi-step plan (no collapse).
  (2) SURFACE: if STILL unparseable after the re-prompt, do NOT silently collapse —
      set self.plan_degraded=True + self.plan_parse_error, emit a distinct
      "plan_parse_failed" orchestration event, and proceed as a single step.
  A valid first reply does NOT re-prompt and is NOT degraded; a genuinely valid
  single-element plan (["one step"]) is NOT degraded.

Hermetic: a scripted fake agent (no network / real CLI), AGY-style stub.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.workflows.master import MasterWorkflow


class _SeqPlanner(AgentInstance):
    """Returns successive scripted replies; records call count. Every instance the
    workflow constructs shares one class-level reply queue so the re-prompt (which
    may build a fresh agent) still advances the script."""

    _replies: List[str] = []
    calls: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(prompt=kwargs.get("prompt", ""))

    @classmethod
    def script(cls, replies):
        cls._replies = list(replies)
        cls.calls = 0

    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        i = min(type(self).calls, len(type(self)._replies) - 1)
        type(self).calls += 1
        return type(self)._replies[i]


def _make(tmp_path, events=None):
    cb = None
    if events is not None:
        def cb(ev):
            events.append(ev)
    return MasterWorkflow(
        model="m", effort="low", branches=1,
        agent_class=_SeqPlanner,
        checkpoint_path=str(tmp_path / "ckpt.json"),
        event_callback=cb,
    )


def _orch_actions(events):
    out = []
    for ev in events:
        data = ev.get("data", {}) if isinstance(ev, dict) else {}
        orch = data.get("orchestration") if isinstance(data, dict) else None
        if isinstance(orch, dict) and "action" in orch:
            out.append(orch["action"])
    return out


@pytest.mark.not_slow
def test_reprompt_recovers_multistep_plan(tmp_path):
    # First reply is prose (no JSON array); the JSON-only re-prompt yields the real
    # multi-step plan. No collapse, no degraded flag.
    _SeqPlanner.script([
        "Sure, here is my plan: first do X, then Y, then Z.",
        '["Step 1: scaffold", "Step 2: core", "Step 3: UI"]',
    ])
    wf = _make(tmp_path)
    tasks, _sid = asyncio.run(wf._run_planner("build a big thing"))

    assert _SeqPlanner.calls == 2            # re-prompted exactly once
    assert tasks == ["Step 1: scaffold", "Step 2: core", "Step 3: UI"]
    assert wf.plan_degraded is False
    assert wf.plan_parse_error is None


@pytest.mark.not_slow
def test_still_unparseable_surfaces_degraded(tmp_path):
    # Both replies are prose: do NOT crash, but DO surface — degraded flag set,
    # distinct event emitted, single-step fallback used.
    events: list = []
    _SeqPlanner.script(["no json here", "still no json, sorry"])
    wf = _make(tmp_path, events)
    tasks, _sid = asyncio.run(wf._run_planner("build a big thing"))

    assert _SeqPlanner.calls == 2
    assert tasks == ["build a big thing"]            # single-step fallback (no crash)
    assert wf.plan_degraded is True
    assert wf.plan_parse_error is not None
    assert "plan_parse_failed" in _orch_actions(events)


@pytest.mark.not_slow
def test_good_first_reply_no_reprompt(tmp_path):
    events: list = []
    _SeqPlanner.script(['["A: setup", "B: build"]'])
    wf = _make(tmp_path, events)
    tasks, _sid = asyncio.run(wf._run_planner("p"))

    assert _SeqPlanner.calls == 1                     # no wasteful re-prompt
    assert tasks == ["A: setup", "B: build"]
    assert wf.plan_degraded is False
    assert wf.plan_parse_error is None
    assert "plan_parse_failed" not in _orch_actions(events)


@pytest.mark.not_slow
def test_valid_single_element_plan_not_degraded(tmp_path):
    # A planner that legitimately returns a one-step plan is NOT a degradation.
    _SeqPlanner.script(['["Only step: do the whole thing"]'])
    wf = _make(tmp_path)
    tasks, _sid = asyncio.run(wf._run_planner("p"))

    assert _SeqPlanner.calls == 1
    assert tasks == ["Only step: do the whole thing"]
    assert wf.plan_degraded is False
    assert wf.plan_parse_error is None

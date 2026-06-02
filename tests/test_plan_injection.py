"""Plan injection: feed an edited plan back in for verbatim execution.

Closes the round-trip that --plan-only opened: --plan-only emits plan.json, the
operator reviews/edits it, then --plan <file> executes those steps verbatim and
the master planner is SKIPPED. Covers the loader/validator (harness.dispatch.
load_plan_steps) and MasterWorkflow's plan_steps input path. Fully hermetic — no
workers, no network.
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

import pytest

import agy_orchestrator.workflows.master as master_mod
from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.workflows.master import MasterWorkflow
from harness.dispatch import load_plan_steps


# --------------------------------------------------------------------------- #
# load_plan_steps: accepts both shapes, rejects malformed input.
# --------------------------------------------------------------------------- #

def test_load_plan_json_object_shape(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"instruction": "build X", "n_steps": 2,
                             "steps": ["step one", "step two"]}))
    assert load_plan_steps(str(p)) == ["step one", "step two"]


def test_load_plan_bare_list_shape(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(["a", "b", "c"]))
    assert load_plan_steps(str(p)) == ["a", "b", "c"]


def test_load_plan_missing_file():
    with pytest.raises(ValueError, match="no such plan file"):
        load_plan_steps("/nonexistent/plan.json")


def test_load_plan_bad_json(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text("{not valid json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_plan_steps(str(p))


def test_load_plan_wrong_shape(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"instruction": "x"}))  # no "steps"
    with pytest.raises(ValueError, match="no steps"):
        load_plan_steps(str(p))


def test_load_plan_empty_list(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps([]))
    with pytest.raises(ValueError, match="no steps"):
        load_plan_steps(str(p))


def test_load_plan_non_string_step(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(["ok", 42]))
    with pytest.raises(ValueError, match="step 2 must be a non-empty string"):
        load_plan_steps(str(p))


def test_load_plan_blank_step(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(["ok", "   "]))
    with pytest.raises(ValueError, match="step 2 must be a non-empty string"):
        load_plan_steps(str(p))


# --------------------------------------------------------------------------- #
# MasterWorkflow plan_steps input: executes verbatim, skips the planner.
# --------------------------------------------------------------------------- #

class _StubAgent(AgentInstance):
    """No-network agent for the summarizer phase."""

    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        return "step summary"


class _RecordingAdv:
    """Stand-in for AdversarialReview that records the prompt it executed."""

    executed_prompts: list = []

    def __init__(self, **kw):
        self.verified = True
        self.approved = True
        self.stalled = False
        self.iterations_used = 1

    async def execute(self, prompt: str) -> str:
        _RecordingAdv.executed_prompts.append(prompt)
        return "STEP OUTPUT"


def _make_master(**kw) -> MasterWorkflow:
    defaults = dict(model="m", effort="low", branches=1, agent_class=_StubAgent)
    defaults.update(kw)
    return MasterWorkflow(**defaults)


def _patch(monkeypatch, *, planner_must_not_run=True, checkpoint=None):
    _RecordingAdv.executed_prompts = []

    async def _planner_boom(self, initial_prompt):
        raise AssertionError("planner ran but an injected plan was supplied")

    if planner_must_not_run:
        monkeypatch.setattr(MasterWorkflow, "_run_planner", _planner_boom)
    monkeypatch.setattr(MasterWorkflow, "_save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(MasterWorkflow, "_load_checkpoint",
                        lambda self, p: checkpoint)
    monkeypatch.setattr(master_mod, "AdversarialReview", _RecordingAdv)


def test_injected_plan_executes_verbatim_and_skips_planner(monkeypatch):
    _patch(monkeypatch)
    m = _make_master(plan_steps=["implement ALPHA", "implement BETA"])

    asyncio.run(m.execute("build a thing"))

    # Two steps ran, in order, carrying the supplied task text — and the planner
    # (patched to raise) was never invoked.
    assert len(_RecordingAdv.executed_prompts) == 2
    assert "implement ALPHA" in _RecordingAdv.executed_prompts[0]
    assert "implement BETA" in _RecordingAdv.executed_prompts[1]


def test_plan_only_with_injection_echoes_without_planning(monkeypatch):
    # --plan-only + an injected plan emits the supplied plan, no planner call.
    async def _planner_boom(self, initial_prompt):
        raise AssertionError("planner ran in plan-only with an injected plan")

    monkeypatch.setattr(MasterWorkflow, "_run_planner", _planner_boom)
    m = _make_master(plan_only=True, plan_steps=["only step"])

    out = asyncio.run(m.execute("build a thing"))
    assert "only step" in out
    assert m.plan_steps == ["only step"]


def test_matching_checkpoint_is_honored(monkeypatch):
    # A checkpoint whose tasks EQUAL the injected plan = a resumed injected run;
    # honor it (resume in place) rather than restarting from step 0.
    ckpt = (["implement ALPHA", "implement BETA"], 1,
            "Original Goal: x\n", None)
    _patch(monkeypatch, checkpoint=ckpt)
    m = _make_master(plan_steps=["implement ALPHA", "implement BETA"])

    asyncio.run(m.execute("build a thing"))
    # Resumed at index 1 => only the second step runs.
    assert len(_RecordingAdv.executed_prompts) == 1
    assert "implement BETA" in _RecordingAdv.executed_prompts[0]


def test_edited_plan_discards_stale_checkpoint(monkeypatch):
    # The operator edited the plan since the last run: the checkpoint's tasks
    # differ from the injected plan, so the stale checkpoint must be dropped and
    # the NEW plan run fresh from step 0.
    ckpt = (["OLD step one", "OLD step two"], 1, "Original Goal: x\n", None)
    _patch(monkeypatch, checkpoint=ckpt)
    m = _make_master(plan_steps=["NEW only step"])

    asyncio.run(m.execute("build a thing"))
    assert len(_RecordingAdv.executed_prompts) == 1
    assert "NEW only step" in _RecordingAdv.executed_prompts[0]

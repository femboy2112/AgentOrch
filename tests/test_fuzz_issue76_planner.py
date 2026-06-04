"""Fuzz #76 — planner-decomposition collapse must be RECOVERED-or-SURFACED.

The Master planner asks a worker for a JSON list of step prompts. Issue #76:
a single unparseable reply does NOT mean the project is un-plannable — the
planner often emits a perfectly good plan wrapped in prose/markdown. Silently
collapsing to one mega-step there loses every step's verify + adversarial
refinement on a multi-deliverable build. The landed fix:

  * bracket-extract recovers a prose-wrapped JSON array WITHOUT a re-prompt;
  * a first hard-unparseable reply triggers EXACTLY ONE JSON-only re-prompt;
  * only a *double* failure (re-prompt also unparseable, or the re-prompt call
    itself raising) is a real degradation — and THAT must be SURFACED:
    ``plan_degraded`` + ``plan_parse_error`` set, an ERROR logged, and a
    distinct ``plan_parse_failed`` orchestration event emitted (never silent);
  * a genuinely valid plan — including a legit SINGLE-element array — is never
    mis-flagged as degraded.

These probes drive ``MasterWorkflow._run_planner`` directly with a scripted,
no-network agent and assert the retry fires once, degradation surfaces ONLY on a
real double-failure, and valid plans are never mis-flagged.

Do NOT modify source. RED contracts (a DESIRED behavior the code does not yet
satisfy) are marked ``xfail(strict=True)``.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.workflows.master import MasterWorkflow

pytestmark = pytest.mark.not_slow


class _ScriptedAgent(AgentInstance):
    """No-network agent returning a queued sequence of canned outputs.

    Each instantiation pops the next scripted reply (planner call 1, then the
    re-prompt). A reply may be a string (returned) or an Exception subclass /
    instance (raised from ``run_async`` to model a re-prompt worker call that
    itself blows up). Construction count is tracked on the class so a probe can
    assert the re-prompt fired exactly once (two total constructions) or never
    (one construction).
    """

    script: List[object] = []
    constructions: int = 0
    prompts_seen: List[str] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).constructions += 1
        type(self).prompts_seen.append(kwargs.get("prompt", args[0] if args else ""))
        # mimic a worker that establishes a session id
        if not getattr(self, "session_id", None):
            self.session_id = "sess-1"

    @classmethod
    def fresh(cls, script: List[object]) -> type:
        sub = type("_ScriptedAgentInst", (cls,), {})
        sub.script = list(script)
        sub.constructions = 0
        sub.prompts_seen = []
        return sub

    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        idx = type(self).constructions - 1
        reply = type(self).script[idx]
        if isinstance(reply, BaseException) or (
            isinstance(reply, type) and issubclass(reply, BaseException)
        ):
            raise reply if isinstance(reply, BaseException) else reply("re-prompt boom")
        return reply


def _make_master(agent_cls):
    events: List[dict] = []

    def _cb(ev: dict) -> None:
        events.append(ev)

    wf = MasterWorkflow(
        model="m",
        effort="low",
        branches=1,
        agent_class=agent_cls,
        event_callback=_cb,
    )
    return wf, events


def _run_planner(agent_cls):
    wf, events = _make_master(agent_cls)
    tasks, _sid = asyncio.run(wf._run_planner("Build a thing"))
    return wf, events, tasks


def _orch_actions(events: List[dict]) -> List[str]:
    out = []
    for ev in events:
        data = ev.get("data", {}) if isinstance(ev, dict) else {}
        orch = data.get("orchestration", {}) if isinstance(data, dict) else {}
        if "action" in orch:
            out.append(orch["action"])
    return out


# ---------------------------------------------------------------------------
# 1. Prose-wrapped JSON — bracket-extract recovers, NO re-prompt.
# ---------------------------------------------------------------------------
def test_prose_wrapped_json_recovers_without_reprompt():
    cls = _ScriptedAgent.fresh([
        'Sure! Here is the plan:\n```json\n["Step 1: scaffold", "Step 2: tests"]\n```\nGood luck.',
    ])
    wf, events, tasks = _run_planner(cls)
    assert tasks == ["Step 1: scaffold", "Step 2: tests"]
    # exactly ONE planner construction — no re-prompt
    assert cls.constructions == 1, "prose-wrapped JSON must NOT trigger a re-prompt"
    assert "reprompt_json" not in _orch_actions(events)
    assert wf.plan_degraded is False
    assert wf.plan_parse_error is None


# ---------------------------------------------------------------------------
# 2. Double-failure: malformed on BOTH first call and the re-prompt.
#    -> retry fires ONCE, degradation SURFACED (flag + error attr + event).
# ---------------------------------------------------------------------------
def test_double_malformed_surfaces_degradation():
    cls = _ScriptedAgent.fresh([
        "I cannot output JSON, sorry — here is prose only.",
        "Still just prose, no array here either.",
    ])
    wf, events, tasks = _run_planner(cls)
    # retry fired exactly once (two constructions)
    assert cls.constructions == 2, "exactly one re-prompt must fire"
    actions = _orch_actions(events)
    assert "reprompt_json" in actions, "the JSON-only re-prompt event must be emitted"
    # degradation surfaced, NOT silent
    assert wf.plan_degraded is True
    assert wf.plan_parse_error is not None
    assert "plan_parse_failed" in actions, "a distinct degradation event must surface"
    # collapse-to-single-step fallback (does not crash)
    assert tasks == ["Build a thing"]


# ---------------------------------------------------------------------------
# 3. JSON object {} (not a list) on the first call -> re-prompt recovers.
#    A bare {} is NOT a valid plan; the code must re-ask, not accept it.
# ---------------------------------------------------------------------------
def test_object_not_list_triggers_reprompt_then_recovers():
    cls = _ScriptedAgent.fresh([
        "{}",
        '["Step 1: only real step"]',
    ])
    wf, events, tasks = _run_planner(cls)
    assert cls.constructions == 2, "a bare {} object must trigger the re-prompt"
    assert "reprompt_json" in _orch_actions(events)
    assert tasks == ["Step 1: only real step"]
    assert wf.plan_degraded is False


# ---------------------------------------------------------------------------
# 3b. JSON object {} on BOTH calls -> degraded + surfaced (a {} is never a plan).
#     DESIRED: an object is recognised as not-a-list and the double-{} run is
#     marked degraded. The current bracket-extract finds no '[' in "{}", so it
#     raises and routes through the SAME double-failure path as malformed prose
#     — which DOES surface degradation. So this is RESILIENT.
# ---------------------------------------------------------------------------
def test_object_on_both_calls_is_degraded():
    cls = _ScriptedAgent.fresh(["{}", '{"steps": "nope"}'])
    wf, events, tasks = _run_planner(cls)
    # NOTE: the second reply {"steps":"nope"} contains NO '[' -> ValueError ->
    # double-failure path -> degraded. (A reply like {"plan":[...]} would instead
    # bracket-extract the inner array; not exercised here.)
    assert wf.plan_degraded is True
    assert "plan_parse_failed" in _orch_actions(events)
    assert tasks == ["Build a thing"]


# ---------------------------------------------------------------------------
# 4. Empty array [] -> parses cleanly to zero steps.
#    DESIRED: an EMPTY plan is a degenerate decomposition (the project would do
#    NOTHING) and must be SURFACED as degraded, exactly like a parse failure —
#    it must never silently run zero steps. The landed fix only sets
#    plan_degraded inside the JSON-parse except branch; "[]" parses fine, so the
#    flag stays False and an empty step list is accepted SILENTLY. RED contract.
# ---------------------------------------------------------------------------
def test_empty_array_is_surfaced_as_degraded():
    cls = _ScriptedAgent.fresh(["[]", "[]"])
    wf, events, tasks = _run_planner(cls)
    # An empty decomposition is degenerate and must be flagged + surfaced.
    assert wf.plan_degraded is True, "an empty plan must be surfaced as degraded"
    assert "plan_parse_failed" in _orch_actions(events) or wf.plan_parse_error is not None


# ---------------------------------------------------------------------------
# 5. Legit SINGLE-element array -> a real one-step plan, NEVER mis-flagged.
# ---------------------------------------------------------------------------
def test_single_element_array_not_degraded():
    cls = _ScriptedAgent.fresh(['["Step 1: do the entire thing in one go"]'])
    wf, events, tasks = _run_planner(cls)
    assert tasks == ["Step 1: do the entire thing in one go"]
    assert cls.constructions == 1, "a valid plan must not re-prompt"
    assert wf.plan_degraded is False, "a legit single-step plan must NOT be flagged degraded"
    assert wf.plan_parse_error is None
    assert "plan_parse_failed" not in _orch_actions(events)


# ---------------------------------------------------------------------------
# 6. Non-JSON containing stray '[' ']' chars -> extraction grabs garbage that
#    fails json.loads -> re-prompt; recoverable.
# ---------------------------------------------------------------------------
def test_stray_brackets_non_json_triggers_reprompt():
    cls = _ScriptedAgent.fresh([
        "Refer to item [3] in the spec and section [B] below; no JSON here.",
        '["Step 1: recovered after re-prompt"]',
    ])
    wf, events, tasks = _run_planner(cls)
    assert cls.constructions == 2, "stray-bracket garbage must trigger the re-prompt"
    assert "reprompt_json" in _orch_actions(events)
    assert tasks == ["Step 1: recovered after re-prompt"]
    assert wf.plan_degraded is False


# ---------------------------------------------------------------------------
# 6b. Stray brackets on BOTH calls -> degraded + surfaced (double failure).
# ---------------------------------------------------------------------------
def test_stray_brackets_on_both_calls_degraded():
    cls = _ScriptedAgent.fresh([
        "see [a] and [b]",
        "still [garbage] not [json]",
    ])
    wf, events, tasks = _run_planner(cls)
    assert wf.plan_degraded is True
    assert wf.plan_parse_error is not None
    assert "plan_parse_failed" in _orch_actions(events)
    assert tasks == ["Build a thing"]


# ---------------------------------------------------------------------------
# 7. The re-prompt CALL ITSELF raises (worker error) -> must be caught and
#    surfaced as degraded, NOT propagate / crash the run.
# ---------------------------------------------------------------------------
def test_reprompt_call_raising_is_caught_and_surfaced():
    cls = _ScriptedAgent.fresh([
        "prose only, no array",
        RuntimeError("re-prompt worker exploded"),
    ])
    wf, events, tasks = _run_planner(cls)
    assert cls.constructions == 2
    assert wf.plan_degraded is True, "a raising re-prompt must surface degradation, not crash"
    assert wf.plan_parse_error is not None
    assert "plan_parse_failed" in _orch_actions(events)
    assert tasks == ["Build a thing"]

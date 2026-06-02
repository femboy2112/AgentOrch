"""Hermetic tests for the Reconciliation / Integration-Skeptic station (issue #43).

The agent is fully mocked (a canned reply, no LLM/network), so these exercise the
station's PARSING + CLASSIFICATION + DISPOSITION plumbing end-to-end, not the model.
Mirrors the FakeAgent pattern in tests/test_adversarial_preamble.py.
"""
from __future__ import annotations

import asyncio
import json
import textwrap

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import VerifierResult
from agy_orchestrator.workflows.reconcile import (
    DISPOSITIONS,
    MechanismFinding,
    ReconciliationResult,
    ReconciliationReview,
    Witness,
)


class _StubAgent(AgentInstance):
    """An AgentInstance whose run_async returns a canned reply and records prompts."""

    def __init__(self, reply: str, prompt: str = ""):
        super().__init__(prompt=prompt)
        self._reply = reply
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
        self.calls += 1
        self.prompts_seen.append(self.prompt)
        return self._reply


def _run(reply: str, **kwargs) -> ReconciliationResult:
    agent = _StubAgent(reply)
    station = ReconciliationReview(agent, goal="build a live-learning agent", **kwargs)
    result = asyncio.run(station.execute())
    # The station also stashes the result + reconciled flag for the ledger.
    assert station.result is result
    assert station.reconciled == result.reconciled
    return result


# --------------------------------------------------------------------------- #
# (a) a dead-wiring report -> reconciled is False, finding carries sub_kind + line
# --------------------------------------------------------------------------- #
def test_stub_constant_finding_not_reconciled():
    reply = json.dumps(
        {
            "reconciled": True,  # the model LIES; the station must override it
            "findings": [
                {
                    "name": "surprise",
                    "classification": "exists_not_load_bearing",
                    "sub_kind": "stub_constant",
                    "location": "agent/world.py:88",
                    "witness": {"value": 0, "description": "ablation moved nothing"},
                    "evidence": "surprise() hardcodes return 0.0; real impl in _real_surprise()",
                }
            ],
        }
    )
    result = _run(reply)
    assert result.reconciled is False
    dead = result.findings()
    assert len(dead) == 1
    assert dead[0].sub_kind == "stub_constant"
    assert dead[0].location == "agent/world.py:88"
    assert dead[0].is_dead is True
    assert result.verdict == "not_reconciled"


def test_uncalled_finding_not_reconciled():
    reply = json.dumps(
        {
            "findings": [
                {
                    "name": "neural_substrate",
                    "classification": "exists_not_load_bearing",
                    "sub_kind": "uncalled",
                    "location": "agent/substrate.py:12",
                    "witness": {"value": 0.0, "description": "tick() never calls it"},
                    "evidence": "no call site for substrate.step on the live tick path",
                }
            ]
        }
    )
    result = _run(reply)
    assert result.reconciled is False
    assert result.findings()[0].sub_kind == "uncalled"


def test_bypassed_proxy_finding_not_reconciled():
    reply = json.dumps(
        {
            "findings": [
                {
                    "name": "learned_similarity",
                    "classification": "exists_not_load_bearing",
                    "sub_kind": "bypassed_proxy",
                    "location": "agent/sim.py:41",
                    "witness": {"value": 0, "description": "returns co-occurrence count, not the trained output"},
                    "evidence": "trained output discarded; co-occurrence table returned instead",
                }
            ]
        }
    )
    result = _run(reply)
    assert result.reconciled is False
    f = result.findings()[0]
    assert f.sub_kind == "bypassed_proxy"
    assert f.classification == "exists_not_load_bearing"


# --------------------------------------------------------------------------- #
# (b) all load_bearing -> reconciled True, no dead findings
# --------------------------------------------------------------------------- #
def test_all_load_bearing_reconciled():
    reply = json.dumps(
        {
            "findings": [
                {
                    "name": "planner",
                    "classification": "load_bearing",
                    "sub_kind": None,
                    "location": "agent/plan.py:5",
                    "witness": {"value": 0.42, "description": "ablating planner drops reward 0.42"},
                    "evidence": "called from run_loop() line 5",
                },
                {
                    "name": "memory",
                    "classification": "load_bearing",
                    "location": "agent/mem.py:9",
                    "witness": {"value": 3.1, "description": "removing memory tanks recall"},
                    "evidence": "read on every tick",
                },
            ]
        }
    )
    result = _run(reply)
    assert result.reconciled is True
    assert result.findings() == []
    assert len(result.load_bearing_findings()) == 2
    assert result.verdict == "reconciled"


# --------------------------------------------------------------------------- #
# (c) witness == 0 vs witness > 0 discrimination
# --------------------------------------------------------------------------- #
def test_witness_zero_is_dead_witness_positive_is_live():
    dead = Witness(value=0.0, description="nothing moved")
    live = Witness(value=0.7, description="signal moved")
    assert dead.is_dead is True
    assert live.is_dead is False

    # A mechanism classified load_bearing with a positive witness is NOT flagged;
    # a stub with a zero witness IS. The discriminator separates dead-wiring from
    # legitimately-incomplete (witness > 0, metric low is acceptable).
    reply = json.dumps(
        {
            "findings": [
                {
                    "name": "incomplete_but_wired",
                    "classification": "load_bearing",
                    "location": "agent/x.py:1",
                    "witness": {"value": 0.05, "description": "wired; metric low but moves"},
                    "evidence": "on path",
                },
                {
                    "name": "dead_stub",
                    "classification": "exists_not_load_bearing",
                    "sub_kind": "stub_constant",
                    "location": "agent/y.py:2",
                    "witness": {"value": 0, "description": "ablation moves nothing"},
                    "evidence": "hardcoded",
                },
            ]
        }
    )
    result = _run(reply)
    assert result.reconciled is False
    assert {f.name for f in result.findings()} == {"dead_stub"}
    live_f = result.load_bearing_findings()[0]
    assert live_f.witness.value == 0.05
    assert live_f.witness.is_dead is False


# --------------------------------------------------------------------------- #
# (d) default disposition is "warn" and does NOT raise/fail
# --------------------------------------------------------------------------- #
def test_default_disposition_warn_does_not_fail():
    reply = json.dumps(
        {
            "findings": [
                {
                    "name": "dead",
                    "classification": "exists_not_load_bearing",
                    "sub_kind": "uncalled",
                    "location": "a.py:1",
                    "witness": {"value": 0},
                    "evidence": "uncalled",
                }
            ]
        }
    )
    result = _run(reply)  # no exception raised even though there's a dead finding
    assert result.disposition == "warn"
    assert result.reconciled is False
    # warn never fails the run, even with a dead finding present.
    assert result.should_fail_run is False


def test_fail_disposition_flags_run():
    reply = json.dumps(
        {
            "findings": [
                {
                    "name": "dead",
                    "classification": "exists_not_load_bearing",
                    "sub_kind": "uncalled",
                    "location": "a.py:1",
                    "witness": {"value": 0},
                    "evidence": "uncalled",
                }
            ]
        }
    )
    result = _run(reply, disposition="fail")
    assert result.should_fail_run is True
    # ...but a SUBSTANTIVE reconciled run under 'fail' still does not fail. A trace
    # that actually examined a mechanism and found it load-bearing is clean (#51:
    # an EMPTY trace no longer counts as clean — it is hollow, see substance tests).
    clean = _run(
        json.dumps(
            {
                "findings": [
                    {
                        "name": "live",
                        "classification": "load_bearing",
                        "location": "a.py:1",
                        "witness": {"value": 2.0},
                        "evidence": "on path",
                    }
                ]
            }
        ),
        disposition="fail",
    )
    assert clean.should_fail_run is False


def test_invalid_disposition_rejected():
    agent = _StubAgent("{}")
    try:
        ReconciliationReview(agent, goal="g", disposition="explode")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid disposition")
    # the canonical set is what we advertise
    assert set(DISPOSITIONS) == {"warn", "fail", "open-task"}


# --------------------------------------------------------------------------- #
# (e) robust parsing: ```json fences and thinking blocks
# --------------------------------------------------------------------------- #
def test_parse_json_fenced_with_thinking_block():
    reply = textwrap.dedent(
        """\
        <think>
        Let me trace surprise(). It looks plausible but I should check the call site.
        I'll write APPROVED-style prose here that must NOT leak into the parse.
        </think>
        Here is my reconciliation report:

        ```json
        {
          "reconciled": false,
          "findings": [
            {
              "name": "surprise",
              "classification": "exists_not_load_bearing",
              "sub_kind": "stub_constant",
              "location": "world.py:88",
              "witness": {"value": 0, "description": "dead"},
              "evidence": "hardcoded 0.0"
            }
          ]
        }
        ```

        Hope that helps!
        """
    )
    result = _run(reply)
    assert result.parse_error is None
    assert result.reconciled is False
    assert result.findings()[0].name == "surprise"
    assert result.findings()[0].sub_kind == "stub_constant"


def test_parse_unfenced_json_with_surrounding_prose():
    reply = (
        "I traced everything. Verdict below.\n"
        '{"findings": [{"name": "x", "classification": "load_bearing", '
        '"location": "x.py:1", "witness": {"value": 2.0}, "evidence": "on path"}]}\n'
        "That is all."
    )
    result = _run(reply)
    assert result.parse_error is None
    assert result.reconciled is True


def test_unparseable_reply_is_not_reconciled():
    result = _run("the model rambled and produced no JSON at all")
    assert result.parse_error is not None
    # An unreadable verdict must NOT pass as all-clear.
    assert result.reconciled is False


def test_dead_finding_missing_sub_kind_defaults_to_uncalled():
    reply = json.dumps(
        {
            "findings": [
                {
                    "name": "m",
                    "classification": "exists_not_load_bearing",
                    "location": "m.py:3",
                    "witness": {"value": 0},
                    "evidence": "no caller",
                }
            ]
        }
    )
    result = _run(reply)
    assert result.findings()[0].sub_kind == "uncalled"


def test_file_line_split_fields_are_joined():
    reply = json.dumps(
        {
            "findings": [
                {
                    "name": "m",
                    "classification": "exists_not_load_bearing",
                    "sub_kind": "uncalled",
                    "file": "pkg/mod.py",
                    "line": 17,
                    "witness": {"value": 0},
                    "evidence": "uncalled",
                }
            ]
        }
    )
    result = _run(reply)
    assert result.findings()[0].location == "pkg/mod.py:17"


# --------------------------------------------------------------------------- #
# (f) the verdict is a DISTINCT object, not a VerifierResult, not auto-folded
# --------------------------------------------------------------------------- #
def test_verdict_is_distinct_from_verifier_result():
    # A SUBSTANTIVE reconciled trace (>=1 mechanism, all load-bearing); an EMPTY
    # trace is hollow and never reconciled (#51), so use a real finding here.
    result = _run(
        json.dumps(
            {
                "findings": [
                    {
                        "name": "live",
                        "classification": "load_bearing",
                        "location": "a.py:1",
                        "witness": {"value": 2.0},
                        "evidence": "on path",
                    }
                ]
            }
        )
    )
    assert isinstance(result, ReconciliationResult)
    assert not isinstance(result, VerifierResult)
    # ReconciliationResult must NOT be auto-truthy-folded the way VerifierResult is
    # (VerifierResult.__bool__ returns .ok). A caller has to read .reconciled.
    assert not hasattr(result, "ok")
    # It uses default object truthiness (always truthy), so it can NEVER be used
    # as a drop-in boolean gate that tracks reconciled — a `if result:` check
    # would be a bug, forcing callers to read .reconciled explicitly.
    assert "__bool__" not in ReconciliationResult.__dict__
    not_reconciled = _run(
        json.dumps(
            {
                "findings": [
                    {
                        "name": "d",
                        "classification": "exists_not_load_bearing",
                        "sub_kind": "uncalled",
                        "location": "a.py:1",
                        "witness": {"value": 0},
                        "evidence": "x",
                    }
                ]
            }
        )
    )
    assert not_reconciled.reconciled is False
    assert bool(not_reconciled) is True  # truthy despite NOT being reconciled
    # And it must not masquerade as a green verifier even when reconciled.
    assert result.verdict == "reconciled"
    assert result.reconciled is True


def test_absent_does_not_flip_reconciled_but_is_listed():
    reply = json.dumps(
        {
            "findings": [
                {
                    "name": "missing_thing",
                    "classification": "absent",
                    "location": None,
                    "witness": {"value": 0, "description": "spec wants it, not present"},
                    "evidence": "no such module",
                }
            ]
        }
    )
    result = _run(reply)
    # absent is a planning gap, not dead wiring -> reconciled stays True,
    # but it is still surfaced via absent_findings().
    assert result.reconciled is True
    assert result.findings() == []
    assert len(result.absent_findings()) == 1
    assert result.absent_findings()[0].name == "missing_thing"


def test_to_dict_is_json_serializable_for_artifact():
    reply = json.dumps(
        {
            "findings": [
                {
                    "name": "surprise",
                    "classification": "exists_not_load_bearing",
                    "sub_kind": "stub_constant",
                    "location": "w.py:1",
                    "witness": {"value": 0, "description": "dead"},
                    "evidence": "hardcoded",
                }
            ]
        }
    )
    result = _run(reply)
    blob = json.dumps(result.to_dict())  # must not raise
    parsed = json.loads(blob)
    assert parsed["reconciled"] is False
    assert parsed["verdict"] == "not_reconciled"
    assert parsed["findings"][0]["sub_kind"] == "stub_constant"


# --------------------------------------------------------------------------- #
# Acceptance #6: a known dead-wired component vs a genuinely load-bearing one.
# The station (agent mocked to return the realistic verdict per the fixture)
# flags the dead one and not the live one — end-to-end through YOUR plumbing.
# --------------------------------------------------------------------------- #
def test_acceptance6_flags_dead_not_live_on_mini_repo(tmp_path):
    # A tiny mini-repo: world.py has a dead-wired surprise() (hardcoded 0.0 with
    # the real impl in a sibling method) plus an uncalled helper, AND a genuinely
    # load-bearing reward() that the live loop actually calls.
    world = tmp_path / "world.py"
    world.write_text(
        textwrap.dedent(
            """\
            class World:
                def surprise(self):
                    return 0.0  # DEAD: shadows the real impl below

                def _real_surprise(self):
                    return self._novelty * 1.5  # never called on the live path

                def reward(self):
                    return self._score  # LIVE: read every tick

                def tick(self):
                    # live path uses reward(), never surprise()/_real_surprise()
                    return self.reward()

            def unused_helper():  # uncalled
                return 42
            """
        )
    )

    # The mocked agent returns the realistic verdict a real trace would produce
    # against THIS fixture: surprise=stub_constant (dead), unused_helper=uncalled
    # (dead), reward=load_bearing (live, witness>0).
    reply = json.dumps(
        {
            "reconciled": False,
            "findings": [
                {
                    "name": "surprise",
                    "classification": "exists_not_load_bearing",
                    "sub_kind": "stub_constant",
                    "location": "world.py:2",
                    "witness": {"value": 0, "description": "ablation moves nothing; hardcoded 0.0"},
                    "evidence": "surprise() returns 0.0; _real_surprise() (line 5) is uncalled",
                },
                {
                    "name": "unused_helper",
                    "classification": "exists_not_load_bearing",
                    "sub_kind": "uncalled",
                    "location": "world.py:18",
                    "witness": {"value": 0, "description": "no call site"},
                    "evidence": "no caller in tick() or anywhere on the live path",
                },
                {
                    "name": "reward",
                    "classification": "load_bearing",
                    "location": "world.py:11",
                    "witness": {"value": 1.0, "description": "ablating reward() changes tick() output"},
                    "evidence": "tick() calls reward() at line 14",
                },
            ],
        }
    )
    agent = _StubAgent(reply)
    station = ReconciliationReview(
        agent,
        goal="agent must compute surprise and reward on every live tick",
        working_directory=str(tmp_path),
        disposition="warn",
    )
    result = asyncio.run(station.execute())

    # The dead ones are flagged; the live one is not.
    dead_names = {f.name for f in result.findings()}
    assert dead_names == {"surprise", "unused_helper"}
    assert "reward" not in dead_names
    assert result.reconciled is False
    assert {f.name for f in result.load_bearing_findings()} == {"reward"}
    # warn disposition: surfaced loudly but the run is not failed.
    assert result.should_fail_run is False
    # The working_directory was threaded into the trace prompt.
    assert str(tmp_path) in agent.prompts_seen[0]


def test_event_callback_is_optional_and_invoked():
    events = []
    reply = json.dumps({"findings": []})
    agent = _StubAgent(reply)
    station = ReconciliationReview(
        agent, goal="g", event_callback=lambda e: events.append(e)
    )
    asyncio.run(station.execute())
    kinds = [e.get("data", {}).get("orchestration", {}).get("action") for e in events]
    assert "trace_started" in kinds
    assert "trace_completed" in kinds

    # And a None callback is a no-op (no crash).
    station2 = ReconciliationReview(_StubAgent(reply), goal="g", event_callback=None)
    asyncio.run(station2.execute())


def test_event_callback_failure_is_swallowed():
    def boom(_e):
        raise RuntimeError("observability must never break execution")

    reply = json.dumps(
        {
            "findings": [
                {
                    "name": "live",
                    "classification": "load_bearing",
                    "location": "a.py:1",
                    "witness": {"value": 2.0},
                    "evidence": "on path",
                }
            ]
        }
    )
    station = ReconciliationReview(_StubAgent(reply), goal="g", event_callback=boom)
    # Must not propagate the callback's exception.
    result = asyncio.run(station.execute())
    assert result.reconciled is True


def test_mechanism_finding_dataclass_defaults():
    f = MechanismFinding(name="m", classification="load_bearing")
    assert f.sub_kind is None
    assert f.location is None
    assert isinstance(f.witness, Witness)
    assert f.witness.value == 0.0
    assert f.is_dead is False

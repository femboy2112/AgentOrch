"""Regression test for issue #70 — master's in-loop critic must be DISTINCT.

In master mode the Phase-B adversarial reviewer was built from the GENERATOR's
agent class (codex-led), so the configured --critic / CRITIC_CHAIN (agy-led) was
parsed + logged but never ran — every master review was codex-critiquing-codex
(same-family self-verification). The fix threads a distinct
``critic_agent_class`` (+ critic_model / critic_effort) into MasterWorkflow and
builds the in-loop ``AdversarialReview.critic_instance`` from it; the harness
wires that class from the critic chain for master + pat.

These tests are HERMETIC: no real worker CLIs, no network, no cost. Stub agents
subclass AgentInstance with canned no-op behavior, and AdversarialReview is
monkeypatched with a capturing stub that records the generator/critic instances
it is constructed with. The planner is forced to a single step and the
checkpoint save is a no-op (mirrors _patch_single_step in
test_master_verified_propagation.py).
"""
from __future__ import annotations

import asyncio
from typing import Optional

import agy_orchestrator.workflows.master as master_mod
from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.workflows.master import MasterWorkflow
from harness import roles


class _StubAgent(AgentInstance):
    """Deterministic, no-network agent for every internal master call."""

    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        # Planner output: a JSON array forcing exactly one step. (The planner is
        # monkeypatched below anyway, but this keeps the stub self-consistent.)
        return '["the only step"]'


class GenStub(_StubAgent):
    """Stands in for the codex-led generator chain."""


class CriticStub(_StubAgent):
    """Stands in for the agy-led critic chain — a DISTINCT family."""


class _CapturingAdv:
    """Capturing stand-in for AdversarialReview.

    Records the generator_instance and critic_instance it was constructed with so
    a test can assert the in-loop reviewer is the DISTINCT critic, not a reuse of
    the generator. Exposes the run-level signals AdversarialReview would.
    """

    captured: dict = {}

    def __init__(self, *, generator_instance, critic_instance, **_ignored):
        _CapturingAdv.captured = {
            "generator_instance": generator_instance,
            "critic_instance": critic_instance,
        }
        self.verified = True
        self.approved = True
        self.stalled = False
        self.iterations_used = 1

    async def execute(self, prompt: str) -> str:
        return "FINAL STEP OUTPUT"


def _make_master(**kw) -> MasterWorkflow:
    defaults = dict(model="gen-m", effort="low", branches=1, agent_class=GenStub)
    defaults.update(kw)
    return MasterWorkflow(**defaults)


def _patch_single_step(monkeypatch):
    """Force a one-step run and capture the AdversarialReview construction."""
    async def _one_task(self, initial_prompt):
        return (["the only step"], None)

    _CapturingAdv.captured = {}
    monkeypatch.setattr(MasterWorkflow, "_run_planner", _one_task)
    monkeypatch.setattr(MasterWorkflow, "_save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(master_mod, "AdversarialReview", _CapturingAdv)


def test_master_uses_distinct_critic_class(monkeypatch):
    """The in-loop reviewer is built from the DISTINCT critic class (#70)."""
    _patch_single_step(monkeypatch)
    m = _make_master(agent_class=GenStub, critic_agent_class=CriticStub)

    asyncio.run(m.execute("goal"))

    captured = _CapturingAdv.captured
    assert isinstance(captured["critic_instance"], CriticStub), (
        "in-loop critic must be the distinct CriticStub, not a generator reuse"
    )
    assert isinstance(captured["generator_instance"], GenStub)
    # Distinct family — not the same class — is the whole point.
    assert not isinstance(captured["critic_instance"], type(None))
    assert type(captured["critic_instance"]) is not type(captured["generator_instance"])


def test_master_critic_falls_back_to_generator_when_not_wired(monkeypatch):
    """No distinct critic class wired => back-compat: critic is the generator."""
    _patch_single_step(monkeypatch)
    m = _make_master(agent_class=GenStub)  # no critic_agent_class

    asyncio.run(m.execute("goal"))

    captured = _CapturingAdv.captured
    assert isinstance(captured["critic_instance"], GenStub)
    assert isinstance(captured["generator_instance"], GenStub)


def test_master_critic_uses_critic_model_and_effort(monkeypatch):
    """critic_model / critic_effort are applied to the critic instance (#70)."""
    _patch_single_step(monkeypatch)
    m = _make_master(
        agent_class=GenStub,
        critic_agent_class=CriticStub,
        critic_model="critic-m",
        critic_effort="high",
    )

    asyncio.run(m.execute("goal"))

    critic = _CapturingAdv.captured["critic_instance"]
    # AgentInstance stores `model` and sets every other kwarg (e.g. effort) as an
    # attribute, so the critic carries its OWN model, not the generator's.
    assert critic.model == "critic-m"
    assert getattr(critic, "effort", None) == "high"
    # And the generator kept the generator model.
    assert _CapturingAdv.captured["generator_instance"].model == "gen-m"


def test_dispatch_master_wires_distinct_critic_class():
    """A codex-led generator chain and an agy-led critic chain build DISTINCT
    agent classes — which is exactly what harness.dispatch._run_workflow's master
    branch passes as agent_class vs critic_agent_class (#70).

    We assert via the real wiring primitive (roles.build_master_agent_class) the
    dispatch master/pat branches call: a codex-led chain and an agy-led chain
    produce different agent classes (so the in-loop critic cannot be the
    generator). This keeps the test hermetic — no MasterWorkflow run, no worker
    CLIs — while still exercising the genuine wiring path. The cross-family guard
    (check_chains_cross_family) confirms these two leads are flagged as distinct
    families, the property #70 depends on.
    """
    gen_cls, gen_model, gen_effort = roles.build_master_agent_class(["codex"])
    crit_cls, crit_model, crit_effort = roles.build_master_agent_class(["agy"])

    assert gen_cls is not crit_cls, (
        "codex-led generator and agy-led critic must be different agent classes"
    )
    # The guard recognizes codex vs agy as a cross-family (non-self-verification)
    # pairing — i.e. it does NOT warn. A same-family pairing would warn.
    assert roles.check_chains_cross_family(["codex"], ["agy"]) is None
    assert roles.check_chains_cross_family(["codex"], ["codex"]) is not None

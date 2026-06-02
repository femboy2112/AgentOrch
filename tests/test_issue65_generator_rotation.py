"""Issue #65: when a step fails programmatic VERIFICATION (the generator produced
output but it didn't pass), the adversarial loop must — after K consecutive
verify-failures — advance the generator's fallback chain to the next configured
provider, instead of pinning one model across all max_iterations while the other
generators sit idle.
"""
from __future__ import annotations

import asyncio

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent
from agy_orchestrator.execution.verifier import VerifierResult
from agy_orchestrator.workflows.adversarial import AdversarialReview


# --------------------------------------------------------------------------- #
# FallbackAgent honours rotate_offset
# --------------------------------------------------------------------------- #
def _tagged_agent(tag: str):
    class _A(AgentInstance):
        TAG = tag

        @classmethod
        async def get_available_models(cls):
            return ["m"]

        @classmethod
        async def get_model_usage(cls, model):
            return 100.0

        def build_command(self, piped_input=None):
            return ["true"]

        async def run_async(self, piped_input=None):
            self.stdout = self.TAG
            self.stderr = ""
            self.returncode = 0
            return self.TAG

    _A.__name__ = f"Agent_{tag}"
    return _A


@pytest.mark.not_slow
def test_fallback_rotate_offset_changes_leader():
    chain = [_tagged_agent("A"), _tagged_agent("B"), _tagged_agent("C")]
    Fb = make_fallback_agent(chain, cycles=1)

    fb0 = Fb(prompt="x")
    assert asyncio.run(fb0.run_async()) == "A"          # default: provider 1 leads
    assert fb0.last_provider == "Agent_A"

    fb1 = Fb(prompt="x")
    fb1.rotate_offset = 1
    assert asyncio.run(fb1.run_async()) == "B"          # rotated: provider 2 leads
    assert fb1.last_provider == "Agent_B"

    fb2 = Fb(prompt="x")
    fb2.rotate_offset = 2
    assert asyncio.run(fb2.run_async()) == "C"


# --------------------------------------------------------------------------- #
# Adversarial loop rotates the generator after K consecutive verify-fails
# --------------------------------------------------------------------------- #
class _FakeGenChain:
    """Stands in for a 3-provider FallbackAgent generator."""

    def __init__(self, chain_len=3):
        self._chain = [object() for _ in range(chain_len)]  # only len is read
        self.rotate_offset = 0
        self.prompt = ""
        self.model = "stub"
        self.effort = "high"
        self.calls = 0
        self.offsets_seen = []

    async def run_async(self):
        self.calls += 1
        self.offsets_seen.append(self.rotate_offset)
        return f"OUT {self.calls}"


class _FakeCritic:
    def __init__(self):
        self.prompt = ""

    async def run_async(self):  # pragma: no cover - verifier short-circuits
        return "needs work"


class _AlwaysFailVerifier:
    """Plain (non-infra) verify failure every call, so the loop regenerates."""

    def __init__(self):
        self.calls = 0

    async def verify(self, working_directory: str = "."):
        self.calls += 1
        return VerifierResult(ok=False, message="2 failed", returncode=1,
                              stderr_tail="AssertionError: boom")


def _review(gen, *, events=None, **kw):
    cb = (lambda e: events.append(e)) if events is not None else None
    return AdversarialReview(
        generator_instance=gen, critic_instance=_FakeCritic(),
        verifier=_AlwaysFailVerifier(), max_iterations=5, event_callback=cb, **kw,
    )


@pytest.mark.not_slow
def test_adversarial_rotates_generator_after_k_verify_fails(monkeypatch):
    monkeypatch.setenv("AGY_GENERATOR_ROTATE_AFTER", "2")
    gen = _FakeGenChain(chain_len=3)
    events = []
    rv = _review(gen, events=events)
    asyncio.run(rv.execute("build it"))

    # 5 iterations, rotate after every 2 verify-fails, capped at chain_len-1 (=2):
    # offsets seen across the 5 generator calls should advance 0,0 -> 1,1 -> 2.
    assert gen.offsets_seen == [0, 0, 1, 1, 2]
    assert rv.generator_rotations == 2
    rot_events = [e for e in events
                  if (e.get("data", {}).get("orchestration", {}) or {}).get("action")
                  == "generator_rotation"]
    assert len(rot_events) == 2


@pytest.mark.not_slow
def test_single_provider_generator_never_rotates(monkeypatch):
    monkeypatch.setenv("AGY_GENERATOR_ROTATE_AFTER", "2")
    gen = _FakeGenChain(chain_len=1)   # one provider -> nothing to rotate to
    rv = _review(gen)
    asyncio.run(rv.execute("build it"))
    assert rv.generator_rotations == 0
    assert set(gen.offsets_seen) == {0}

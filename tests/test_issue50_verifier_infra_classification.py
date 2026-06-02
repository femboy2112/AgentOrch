"""Issue #50 — the adversarial loop conflated a verifier TIMEOUT / RESOURCE-kill
with a genuine code test FAILURE.

The verifier already classifies these distinctly (``VerifierResult.timeout`` /
``.resource_exceeded``), but the consumer in ``adversarial.py`` discarded the
distinction: any non-``ok`` result bounced "back to the generator", so a too-slow
baseline suite (or an OOM) burned the entire ``max_iterations`` budget
regenerating code that was never broken, logging the same opaque "verification
failed" each round.

These tests pin the fix: on timeout/resource_exceeded the loop must NOT
regenerate (generator called exactly once), must flag the infra cause distinctly,
and must NOT mark the step verified/approved — while a GENUINE failure still loops
to the budget (no regression).
"""

import asyncio
from typing import Optional

from agy_orchestrator.execution.verifier import VerifierResult
from agy_orchestrator.workflows.adversarial import AdversarialReview


class _FakeGen:
    """Stand-in generator: counts run_async calls, never spawns a subprocess."""

    def __init__(self):
        self.calls = 0
        self.prompt = ""
        self.model = "stub-model"
        self.effort = "high"

    async def run_async(self):
        self.calls += 1
        return f"GENERATED OUTPUT {self.calls}"


class _FakeCritic:
    def __init__(self):
        self.prompt = ""

    async def run_async(self):  # pragma: no cover - never reached in these tests
        return "needs work"


class _FakeVerifier:
    def __init__(self, result: VerifierResult):
        self.result = result
        self.calls = 0

    async def verify(self, working_directory: str = "."):
        self.calls += 1
        return self.result


def _review(verifier, **kw):
    events = kw.pop("events", None)
    cb = (lambda e: events.append(e)) if events is not None else None
    return AdversarialReview(
        generator_instance=_FakeGen(),
        critic_instance=_FakeCritic(),
        verifier=verifier,
        max_iterations=5,
        event_callback=cb,
        **kw,
    )


def test_timeout_does_not_regenerate(monkeypatch):
    v = _FakeVerifier(VerifierResult(ok=False, message="timed out after 600s", timeout=True))
    rv = _review(v)
    out = asyncio.run(rv.execute("build the thing"))
    # Generated ONCE, verified ONCE — no regeneration against unbroken code.
    assert rv.generator.calls == 1
    assert v.calls == 1
    assert rv.iterations_used == 1
    assert rv.verifier_infra_failed is True
    assert rv.infra_reason == "verifier_timeout"
    assert rv.verified is False and rv.approved is False
    assert out == "GENERATED OUTPUT 1"


def test_resource_exceeded_does_not_regenerate():
    v = _FakeVerifier(
        VerifierResult(ok=False, message="OOM-killed in scope", resource_exceeded=True)
    )
    rv = _review(v)
    asyncio.run(rv.execute("build the thing"))
    assert rv.generator.calls == 1
    assert v.calls == 1
    assert rv.verifier_infra_failed is True
    assert rv.infra_reason == "verifier_resource_exceeded"
    assert rv.verified is False and rv.approved is False


def test_genuine_failure_still_loops_to_budget():
    # ok=False but NOT timeout/resource — a real test defect. Must still bounce to
    # the generator every iteration (no regression of the repair loop).
    v = _FakeVerifier(VerifierResult(ok=False, message="exit code 1", timeout=False))
    rv = _review(v)
    asyncio.run(rv.execute("build the thing"))
    assert rv.generator.calls == 5  # regenerated each of max_iterations
    assert v.calls == 5
    assert rv.verifier_infra_failed is False
    assert rv.infra_reason is None


def test_passing_verifier_unaffected():
    v = _FakeVerifier(VerifierResult(ok=True, message="all passed"))
    rv = _review(v)
    asyncio.run(rv.execute("build the thing"))
    assert rv.generator.calls == 1
    assert rv.verified is True and rv.approved is True
    assert rv.verifier_infra_failed is False


def test_timeout_emits_distinct_outcome_event():
    v = _FakeVerifier(VerifierResult(ok=False, message="timed out", timeout=True))
    events = []
    rv = _review(v, events=events)
    asyncio.run(rv.execute("build the thing"))
    outcomes = [
        e["data"]["orchestration"].get("outcome")
        for e in events
        if e.get("data", {}).get("event") == "orchestration_transition"
    ]
    assert "verifier_timeout" in outcomes


def test_iteration_outcome_reports_infra_reason():
    v = _FakeVerifier(VerifierResult(ok=False, message="OOM", resource_exceeded=True))
    rv = _review(v)
    asyncio.run(rv.execute("x"))
    assert rv._iteration_outcome() == "verifier_resource_exceeded"

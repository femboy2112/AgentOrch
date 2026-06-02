"""Regression tests for agy_orchestrator/workflows/adversarial.py.

Subsystem: workflows/adversarial.py — AdversarialReview loop / verify-gate /
generator-rotation / infra-classification / stall-detection / event emission.

Hermetic: no network, no real worker CLI, no credentials. Generators / critics /
verifiers are pure in-process awaitable stubs.

Covered findings:
  * adversarial-1 — a critic whose ``run_async`` returns a non-string (notably
    ``None``, which a real claude worker produces for a
    ``{"type":"result","result":null}`` empty/cancelled turn) used to crash the
    whole adversarial run with ``TypeError: expected string or bytes-like
    object, got 'NoneType'`` because ``execute()`` fed the raw critic reply
    straight into ``_is_approved()`` and the ``_THINK_RE.sub(...)`` stall check,
    both of which call ``re.sub`` on the feedback. The fix coerces a None /
    non-string critic reply to ``""`` (treated as "not approved"), so the loop
    continues instead of aborting the dispatch. This strikes only on
    critic-gated runs (no verifier, or a verifier that is never green) since the
    verifier-pass/fail paths short-circuit before the critic.
"""
from __future__ import annotations

import asyncio

import pytest

from agy_orchestrator.workflows.adversarial import AdversarialReview, _is_approved


def run(coro):
    return asyncio.run(coro)


class StubGen:
    """Minimal generator stub. Records calls; returns scripted output."""

    def __init__(self, output="generated output"):
        self.prompt = ""
        self.model = "stub-model"
        self.effort = "high"
        self.calls = 0
        self._output = output

    async def run_async(self):
        self.calls += 1
        return self._output


class NoneCritic:
    """Mimics a claude worker emitting {"type":"result","result": null}."""

    def __init__(self):
        self.prompt = ""
        self.calls = 0

    async def run_async(self):
        self.calls += 1
        return None


class ScriptedCritic:
    """Returns scripted (possibly non-string) feedbacks, last value sticks."""

    def __init__(self, feedbacks):
        self.prompt = ""
        self.calls = 0
        self._feedbacks = list(feedbacks)

    async def run_async(self):
        self.calls += 1
        if len(self._feedbacks) > 1:
            return self._feedbacks.pop(0)
        return self._feedbacks[0]


# --------------------------------------------------------------------------- #
# adversarial-1: None / non-string critic reply must not crash the loop
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_none_critic_reply_does_not_crash_run():
    """A critic that returns None must not raise; the loop treats it as
    'not approved' and runs to max_iterations, returning the last output."""
    gen = StubGen("generated output")
    critic = NoneCritic()
    rv = AdversarialReview(gen, critic, verifier=None, max_iterations=2)

    result = run(rv.execute("build a thing"))  # must NOT raise TypeError

    assert result == "generated output"
    assert rv.approved is False
    # The critic was consulted (not short-circuited away) and the loop iterated.
    assert critic.calls >= 1
    assert gen.calls >= 1


@pytest.mark.not_slow
def test_none_critic_reply_is_not_approval():
    """None / empty critic feedback must never count as approval."""
    assert _is_approved(None) is False
    assert _is_approved("") is False


@pytest.mark.not_slow
@pytest.mark.parametrize("bad", [None, 123, b"APPROVED", ["APPROVED"], object()])
def test_is_approved_tolerates_non_string(bad):
    """_is_approved must coerce any non-string without raising; only a genuine
    string 'APPROVED' approves, so all these non-strings are not approved."""
    assert _is_approved(bad) is False


@pytest.mark.not_slow
def test_none_then_real_feedback_continues_then_stalls_gracefully():
    """A None reply on the first round followed by real (non-approving) feedback
    must keep iterating without crashing on either round."""
    gen = StubGen("out")
    critic = ScriptedCritic([None, "still needs work"])
    rv = AdversarialReview(gen, critic, verifier=None, max_iterations=3)

    result = run(rv.execute("task"))  # must NOT raise

    assert result == "out"
    assert rv.approved is False
    assert critic.calls >= 2


@pytest.mark.not_slow
def test_none_reply_then_approval_still_converges():
    """If the critic returns None then later approves, the run converges
    normally (the None round is just a non-approving pass)."""
    gen = StubGen("out")
    critic = ScriptedCritic([None, "APPROVED"])
    rv = AdversarialReview(gen, critic, verifier=None, max_iterations=4)

    result = run(rv.execute("task"))

    assert result == "out"
    assert rv.approved is True

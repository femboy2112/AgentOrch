"""Hermetic fuzz / property / edge-case tests for workflows/adversarial.py.

Every test here is HERMETIC: no real worker CLI, no network, no credentials.
Generators / critics / verifiers are pure in-process fakes whose ``run_async`` /
``verify`` are awaitable stubs. These assert the CURRENT, robust behaviour of the
adversarial loop so a regression that breaks it shows up red. Bugs that genuinely
crash the loop are reported in the findings (not committed as failing tests).
"""
from __future__ import annotations

import asyncio

import pytest

from agy_orchestrator.execution.verifier import VerifierResult
from agy_orchestrator.workflows.adversarial import AdversarialReview, _is_approved


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeGen:
    """Generator stub. ``outputs`` is a list popped per call (last value sticks)."""

    def __init__(self, outputs=None):
        self.prompt = ""
        self.model = "stub-model"
        self.effort = "high"
        self.calls = 0
        self.prompts_seen = []
        self._outputs = list(outputs) if outputs is not None else ["OUTPUT"]

    async def run_async(self):
        self.calls += 1
        self.prompts_seen.append(self.prompt)
        if self._outputs:
            if len(self._outputs) > 1:
                return self._outputs.pop(0)
            return self._outputs[0]
        return "OUTPUT"


class FakeCritic:
    """Critic stub returning a fixed or scripted feedback string."""

    def __init__(self, feedbacks=None):
        self.prompt = ""
        self.calls = 0
        self._feedbacks = list(feedbacks) if feedbacks is not None else ["needs work"]

    async def run_async(self):
        self.calls += 1
        if self._feedbacks:
            if len(self._feedbacks) > 1:
                return self._feedbacks.pop(0)
            return self._feedbacks[0]
        return "needs work"


class ScriptedVerifier:
    """Returns scripted VerifierResults, last value sticks."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0
        self.dirs_seen = []

    async def verify(self, working_directory="."):
        self.calls += 1
        self.dirs_seen.append(working_directory)
        if len(self._results) > 1:
            return self._results.pop(0)
        return self._results[0]


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# _is_approved property tests
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
@pytest.mark.parametrize(
    "text,expected",
    [
        ("APPROVED", True),
        ("approved", True),
        ("  APPROVED  ", True),
        ("APPROVED.", True),
        ("APPROVED!!!", True),
        ("line one\nAPPROVED", True),
        ("<think>not yet</think>\nAPPROVED", True),
        ("NOT APPROVED", False),
        ("this is NOT approved yet", False),
        ("I would never write APPROVED here", False),
        ("<think>APPROVED</think>still has issues", False),
        ("", False),
        ("   ", False),
        ("\n\n\t", False),
        ("APPROVED but actually no", False),  # APPROVED not the last token
    ],
)
def test_is_approved_property(text, expected):
    assert _is_approved(text) is expected


@pytest.mark.not_slow
def test_is_approved_handles_control_chars_and_unicode():
    # ANSI escapes / unicode / NUL embedded around an otherwise-approving reply
    assert _is_approved("\x1b[32mAPPROVED\x1b[0m") is False  # ansi is not punctuation
    # NUL is NOT stripped as trailing punctuation, so it defeats the exact match
    # (the safe direction: an ambiguous reply does not count as approval).
    assert _is_approved("APPROVED\x00") is False
    # a clean reply with unicode in earlier lines still approves on the last line
    assert _is_approved("résumé looks good\nAPPROVED") is True


@pytest.mark.not_slow
def test_is_approved_huge_input_no_crash():
    big = ("blah blah " * 100000) + "\nAPPROVED"
    assert _is_approved(big) is True
    big2 = "x" * 2_000_000
    assert _is_approved(big2) is False


# --------------------------------------------------------------------------- #
# Iteration boundary: 0, negative, 1, convergence
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
@pytest.mark.parametrize("mi", [0, -1, -5])
def test_zero_or_negative_iterations_no_call_no_crash(mi):
    gen, critic = FakeGen(), FakeCritic()
    rv = AdversarialReview(gen, critic, max_iterations=mi)
    out = run(rv.execute("do it"))
    assert out == ""
    assert gen.calls == 0
    assert critic.calls == 0
    assert rv.iterations_used == 0
    assert rv.approved is False
    assert rv.verified is False


@pytest.mark.not_slow
def test_single_iteration_approval():
    gen = FakeGen(["the code"])
    critic = FakeCritic(["APPROVED"])
    rv = AdversarialReview(gen, critic, max_iterations=1)
    out = run(rv.execute("do it"))
    assert out == "the code"
    assert rv.approved is True
    assert rv.iterations_used == 1
    assert gen.calls == 1 and critic.calls == 1


@pytest.mark.not_slow
def test_single_iteration_no_approval_returns_last_output():
    gen = FakeGen(["the code"])
    critic = FakeCritic(["change X"])
    rv = AdversarialReview(gen, critic, max_iterations=1)
    out = run(rv.execute("do it"))
    assert out == "the code"
    assert rv.approved is False
    assert rv.iterations_used == 1


@pytest.mark.not_slow
def test_large_max_iterations_terminates_on_approval():
    gen = FakeGen(["a", "b", "c"])
    critic = FakeCritic(["nope", "still nope", "APPROVED"])
    rv = AdversarialReview(gen, critic, max_iterations=10_000_000)
    out = run(rv.execute("do it"))
    assert out == "c"
    assert rv.approved is True
    assert rv.iterations_used == 3  # did not run anywhere near the cap


# --------------------------------------------------------------------------- #
# Stall detection (repeated critique)
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_repeated_critique_stalls_early():
    gen = FakeGen(["x"])
    # identical critique twice -> should bail on the 2nd
    critic = FakeCritic(["please fix the bug", "please fix the bug", "please fix the bug"])
    rv = AdversarialReview(gen, critic, max_iterations=9)
    out = run(rv.execute("do it"))
    assert rv.stalled is True
    assert rv.approved is False
    assert rv.iterations_used == 2  # detected on the second pass
    assert out == "x"


@pytest.mark.not_slow
def test_stall_ignores_whitespace_and_think_blocks():
    gen = FakeGen(["x"])
    critic = FakeCritic([
        "Fix   the\tbug",
        "<think>different reasoning</think>fix the bug",  # normalizes to same
    ])
    rv = AdversarialReview(gen, critic, max_iterations=9)
    run(rv.execute("do it"))
    assert rv.stalled is True
    assert rv.iterations_used == 2


@pytest.mark.not_slow
def test_changing_critique_does_not_stall():
    gen = FakeGen(["x"])
    critic = FakeCritic(["fix A", "fix B", "fix C"])
    rv = AdversarialReview(gen, critic, max_iterations=3)
    run(rv.execute("do it"))
    assert rv.stalled is False
    assert rv.iterations_used == 3


# --------------------------------------------------------------------------- #
# Verifier gate: pass / fail / infra
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_verifier_pass_short_circuits_critic():
    gen = FakeGen(["code"])
    critic = FakeCritic(["should never run"])
    v = ScriptedVerifier([VerifierResult(ok=True, message="all green")])
    rv = AdversarialReview(gen, critic, verifier=v, max_iterations=5)
    out = run(rv.execute("do it"))
    assert out == "code"
    assert rv.verified is True
    assert rv.approved is True
    assert critic.calls == 0  # ground truth: critic never consulted


@pytest.mark.not_slow
def test_verifier_timeout_bails_without_regenerating():
    gen = FakeGen(["code"])
    critic = FakeCritic()
    v = ScriptedVerifier([VerifierResult(ok=False, message="slow", timeout=True)])
    rv = AdversarialReview(gen, critic, verifier=v, max_iterations=5)
    run(rv.execute("do it"))
    assert rv.verifier_infra_failed is True
    assert rv.infra_reason == "verifier_timeout"
    assert rv.verified is False
    assert gen.calls == 1  # did NOT regenerate against a timeout
    assert v.calls == 1


@pytest.mark.not_slow
def test_verifier_oom_bails_without_regenerating():
    gen = FakeGen(["code"])
    critic = FakeCritic()
    v = ScriptedVerifier([VerifierResult(ok=False, message="killed", resource_exceeded=True)])
    rv = AdversarialReview(gen, critic, verifier=v, max_iterations=5)
    run(rv.execute("do it"))
    assert rv.verifier_infra_failed is True
    assert rv.infra_reason == "verifier_resource_exceeded"
    assert gen.calls == 1


@pytest.mark.not_slow
def test_infra_suspected_allows_exactly_one_regenerate_then_bails():
    gen = FakeGen(["v1", "v2", "v3", "v4"])
    critic = FakeCritic()
    # infra-suspected on every verify -> 1 regenerate, then bail
    v = ScriptedVerifier([
        VerifierResult(ok=False, message="make: not found", infra_suspected=True),
    ])
    rv = AdversarialReview(gen, critic, verifier=v, max_iterations=5)
    run(rv.execute("do it"))
    assert rv.verifier_infra_failed is True
    assert rv.infra_reason == "verifier_infra_suspected"
    assert gen.calls == 2  # initial + exactly one regenerate
    assert v.calls == 2


@pytest.mark.not_slow
def test_plain_verify_fail_regenerates_to_max_iterations():
    gen = FakeGen(["bad"])
    critic = FakeCritic()
    v = ScriptedVerifier([VerifierResult(ok=False, message="2 failed", returncode=1)])
    rv = AdversarialReview(gen, critic, verifier=v, max_iterations=3)
    out = run(rv.execute("do it"))
    assert out == "bad"
    assert rv.verified is False
    assert rv.verifier_infra_failed is False
    assert gen.calls == 3  # regenerated every iteration
    assert v.calls == 3


@pytest.mark.not_slow
def test_verifier_recovers_after_initial_fail():
    gen = FakeGen(["bad", "good"])
    critic = FakeCritic()
    v = ScriptedVerifier([
        VerifierResult(ok=False, message="fail", returncode=1),
        VerifierResult(ok=True, message="pass"),
    ])
    rv = AdversarialReview(gen, critic, verifier=v, max_iterations=5)
    out = run(rv.execute("do it"))
    assert rv.verified is True
    assert out == "good"
    assert gen.calls == 2


@pytest.mark.not_slow
def test_verifier_threads_working_directory():
    gen = FakeGen(["code"])
    critic = FakeCritic()
    v = ScriptedVerifier([VerifierResult(ok=True)])
    rv = AdversarialReview(gen, critic, verifier=v, max_iterations=2,
                           working_directory="/tmp/somewhere")
    run(rv.execute("do it"))
    assert v.dirs_seen == ["/tmp/somewhere"]


# --------------------------------------------------------------------------- #
# Generator rotation at chain boundaries
# --------------------------------------------------------------------------- #
class FakeFallbackGen(FakeGen):
    def __init__(self, chain_len, outputs=None):
        super().__init__(outputs)
        self._chain = [type("C%d" % i, (), {})() for i in range(chain_len)]
        self.rotate_offset = 0
        self.offsets_seen = []

    async def run_async(self):
        self.offsets_seen.append(self.rotate_offset)
        return await super().run_async()


@pytest.mark.not_slow
def test_single_provider_never_rotates(monkeypatch):
    monkeypatch.setenv("AGY_GENERATOR_ROTATE_AFTER", "2")
    gen = FakeFallbackGen(chain_len=1, outputs=["bad"])
    v = ScriptedVerifier([VerifierResult(ok=False, message="x", returncode=1)])
    rv = AdversarialReview(gen, FakeCritic(), verifier=v, max_iterations=6)
    run(rv.execute("go"))
    assert rv.generator_rotations == 0
    assert set(gen.offsets_seen) == {0}


@pytest.mark.not_slow
def test_rotation_capped_at_chain_len_minus_one(monkeypatch):
    monkeypatch.setenv("AGY_GENERATOR_ROTATE_AFTER", "1")
    gen = FakeFallbackGen(chain_len=2, outputs=["bad"])
    v = ScriptedVerifier([VerifierResult(ok=False, message="x", returncode=1)])
    rv = AdversarialReview(gen, FakeCritic(), verifier=v, max_iterations=6)
    run(rv.execute("go"))
    # chain_len-1 == 1 rotation maximum, never more
    assert rv.generator_rotations == 1
    assert max(gen.offsets_seen) == 1


@pytest.mark.not_slow
def test_rotate_after_env_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("AGY_GENERATOR_ROTATE_AFTER", "not-a-number")
    rv = AdversarialReview(FakeGen(), FakeCritic(), max_iterations=1)
    assert rv._rotate_after == 2


@pytest.mark.not_slow
def test_rotate_after_zero_clamped_to_one(monkeypatch):
    monkeypatch.setenv("AGY_GENERATOR_ROTATE_AFTER", "0")
    rv = AdversarialReview(FakeGen(), FakeCritic(), max_iterations=1)
    assert rv._rotate_after == 1


# --------------------------------------------------------------------------- #
# Event emission robustness
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_event_callback_exception_is_swallowed():
    def boom(_evt):
        raise RuntimeError("callback exploded")

    gen = FakeGen(["code"])
    critic = FakeCritic(["APPROVED"])
    rv = AdversarialReview(gen, critic, max_iterations=2, event_callback=boom)
    # a raising callback must NOT propagate out of execute()
    out = run(rv.execute("do it"))
    assert out == "code"
    assert rv.approved is True


@pytest.mark.not_slow
def test_events_emitted_each_iteration_no_double_on_terminal():
    events = []
    gen = FakeGen(["a", "b"])
    critic = FakeCritic(["fix it", "fix that"])
    rv = AdversarialReview(gen, critic, max_iterations=2,
                           event_callback=lambda e: events.append(e))
    run(rv.execute("go"))
    completed = [
        e for e in events
        if (e.get("data", {}).get("orchestration", {}) or {}).get("action")
        == "iteration_completed"
    ]
    started = [
        e for e in events
        if (e.get("data", {}).get("orchestration", {}) or {}).get("action")
        == "iteration_started"
    ]
    # exactly one started + one completed per iteration, no duplicates
    assert len(started) == 2
    assert len(completed) == 2


@pytest.mark.not_slow
def test_no_terminal_event_when_zero_iterations():
    events = []
    rv = AdversarialReview(FakeGen(), FakeCritic(), max_iterations=0,
                           event_callback=lambda e: events.append(e))
    run(rv.execute("go"))
    # nothing ran -> no orchestration_transition lifecycle events at all
    transitions = [
        e for e in events
        if e.get("data", {}).get("event") == "orchestration_transition"
    ]
    assert transitions == []


# --------------------------------------------------------------------------- #
# Prompt growth carries the failing output forward (feedback loop integrity)
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_verify_fail_feeds_last_output_into_next_prompt():
    gen = FakeGen(["first-output", "second-output"])
    critic = FakeCritic()
    v = ScriptedVerifier([
        VerifierResult(ok=False, message="ERR-XYZ", returncode=1),
        VerifierResult(ok=True),
    ])
    rv = AdversarialReview(gen, critic, verifier=v, max_iterations=3)
    run(rv.execute("BUILD THE THING"))
    # second prompt must contain the original requirement, the failing output,
    # and the error message
    second_prompt = gen.prompts_seen[1]
    assert "BUILD THE THING" in second_prompt
    assert "first-output" in second_prompt
    assert "ERR-XYZ" in second_prompt


@pytest.mark.not_slow
def test_critic_feedback_carried_forward_in_revise_prompt():
    gen = FakeGen(["v1", "v2"])
    critic = FakeCritic(["RENAME the function foo", "APPROVED"])
    rv = AdversarialReview(gen, critic, max_iterations=3)
    run(rv.execute("REQ"))
    second = gen.prompts_seen[1]
    assert "REQ" in second
    assert "v1" in second
    assert "RENAME the function foo" in second


@pytest.mark.not_slow
def test_diff_only_changes_revise_instruction():
    gen = FakeGen(["v1", "v2"])
    critic = FakeCritic(["tweak it", "APPROVED"])
    rv = AdversarialReview(gen, critic, max_iterations=3, diff_only=True)
    run(rv.execute("REQ"))
    assert "byte-identical" in gen.prompts_seen[1]


# --------------------------------------------------------------------------- #
# Critic preamble / requirement plumbing
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_critic_preamble_prepended():
    seen = {}

    class CapCritic(FakeCritic):
        async def run_async(self):
            seen["prompt"] = self.prompt
            return "APPROVED"

    gen = FakeGen(["code"])
    rv = AdversarialReview(gen, CapCritic(), max_iterations=2,
                           critic_preamble="SAFETY-FIRST: ")
    run(rv.execute("REQ"))
    assert seen["prompt"].startswith("SAFETY-FIRST: ")


@pytest.mark.not_slow
def test_critic_requirement_overrides_initial_prompt_for_critic():
    seen = {}

    class CapCritic(FakeCritic):
        async def run_async(self):
            seen["prompt"] = self.prompt
            return "APPROVED"

    gen = FakeGen(["code"])
    rv = AdversarialReview(gen, CapCritic(), max_iterations=2,
                           critic_requirement="STRIPPED-REQ")
    run(rv.execute("FULL-GENERATOR-PROMPT"))
    assert "STRIPPED-REQ" in seen["prompt"]
    assert "FULL-GENERATOR-PROMPT" not in seen["prompt"]
    # but the GENERATOR still got the full prompt
    assert gen.prompts_seen[0] == "FULL-GENERATOR-PROMPT"


# --------------------------------------------------------------------------- #
# Adversarial string inputs to the loop (no crash, no infinite loop)
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
@pytest.mark.parametrize("weird", ["", "   ", "\x00\x01\x02", "\x1b[31mx\x1b[0m", "é" * 1000])
def test_weird_initial_prompts_do_not_crash(weird):
    gen = FakeGen(["out"])
    critic = FakeCritic(["APPROVED"])
    rv = AdversarialReview(gen, critic, max_iterations=2)
    out = run(rv.execute(weird))
    assert out == "out"


@pytest.mark.not_slow
def test_empty_critic_feedback_continues_not_approved():
    gen = FakeGen(["x"])
    critic = FakeCritic([""])  # empty feedback every time
    rv = AdversarialReview(gen, critic, max_iterations=3)
    run(rv.execute("REQ"))
    # empty != approved; first round sets prev_norm="" then second round stalls
    assert rv.approved is False
    assert rv.stalled is True
    assert rv.iterations_used == 2

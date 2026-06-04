"""Long-run resilience probes for the cascade / pat / pipeline edge paths.

These are the DEFERRED-coverage gap from the prior fuzz-harden batch (8e21495):
cascade / pat / pipeline never got the infra-class-verifier and isolation
treatment that the adversarial loop got in #50. Each probe simulates a
condition that only bites on a multi-step / mission-critical run and asserts
the DESIRED resilient behaviour.

HERMETIC + FAST: every agent is a scripted/canned AgentInstance stub (no real
worker, no subprocess, no network), the verifier is a fake returning a
VerifierResult, hangs are simulated with an asyncio.Event that is never set and
bounded by asyncio.wait_for, and partial-write damage is plain tmp_path file ops.
Each test runs in milliseconds.

Verdicts (see the StructuredOutput summary):
  * resilient  -> assertion passes today; kept as a regression guard (no xfail).
  * bug        -> assertion FAILS against current source (a real long-run defect);
                  decorated @pytest.mark.xfail(strict=True) so the committed
                  suite stays green and flips to a hard failure once fixed.
"""
from __future__ import annotations

import asyncio

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.pipeline import ParallelSwarm
from agy_orchestrator.execution.verifier import VerifierResult
from agy_orchestrator.workflows.cascade import CascadeWorkflow
from agy_orchestrator.workflows.pat import PatWorkflow

pytestmark = pytest.mark.not_slow


# --------------------------------------------------------------------------- #
# Stubs (mirroring tests/test_fuzz_flows.py)
# --------------------------------------------------------------------------- #
class _CannedAgent(AgentInstance):
    """Returns a fixed string; counts how many times it was run."""

    def __init__(self, reply: str = "", prompt: str = ""):
        super().__init__(prompt=prompt)
        self._reply = reply
        self.calls = 0

    @classmethod
    async def get_available_models(cls):
        return []

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        return 100.0

    def build_command(self, piped_input=None):
        return []

    async def run_async(self, piped_input=None) -> str:
        self.calls += 1
        return self._reply


class _HangingAgent(_CannedAgent):
    """run_async awaits an Event that is never set — models a wedged worker that
    never returns (and is NOT an exception, so gather() cannot reap it)."""

    def __init__(self):
        super().__init__()
        self._never = asyncio.Event()

    async def run_async(self, piped_input=None) -> str:
        self.calls += 1
        await self._never.wait()  # blocks forever
        return "unreachable"


class _FakeVerifier:
    """Returns a scripted sequence of VerifierResults (last one repeats)."""

    def __init__(self, results):
        self._results = list(results)
        self._i = 0
        self.calls = 0
        self.observed_dirs = []

    async def verify(self, working_directory: str = ".") -> VerifierResult:
        self.calls += 1
        self.observed_dirs.append(working_directory)
        r = self._results[min(self._i, len(self._results) - 1)]
        self._i += 1
        return r


class _FakeMaster:
    """MasterWorkflow stand-in — records whether it was charged (executed)."""

    def __init__(self, out="master-out", verified=False, approved=False, iterations_used=3):
        self._out = out
        self.verified = verified
        self.approved = approved
        self.iterations_used = iterations_used
        self.invoked = False
        self.executed_prompt = None

    async def execute(self, prompt: str) -> str:
        self.invoked = True
        self.executed_prompt = prompt
        return self._out


def _ok(msg="ok"):
    return VerifierResult(ok=True, message=msg)


def _fail(msg="boom"):
    return VerifierResult(ok=False, message=msg)


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Probe 1 — PaT escalates on an infra-class stage-1 verifier failure
#
# A heavy mission-critical verifier that TIMES OUT / OOMs returns ok=False but
# with timeout=True / resource_exceeded=True / infra_suspected=True. #50 taught
# the ADVERSARIAL loop not to treat that as a code defect, but pat.py:92 still
# branches only on result.ok. So PaT charges the whole MasterWorkflow tax for a
# HOST flap, not a code defect.
#
# DESIRED: an infra-class stage-1 failure must NOT silently pay the master tax —
# either it short-circuits (don't escalate on a non-defect) or it surfaces a
# distinct infra signal instead of charging master blindly. Strongest contract:
# master is not invoked when the only failure was infra-class.
# --------------------------------------------------------------------------- #
def test_pat_infra_stage1_failure_does_not_charge_master():
    m = _FakeMaster()
    infra = VerifierResult(ok=False, message="timed out", timeout=True)
    pat = PatWorkflow(_CannedAgent("direct"), m, _FakeVerifier([infra, _ok()]))
    run(pat.execute("p"))
    # An infra blip (the host verifier flapped) is NOT a stage-1 defect; the full
    # MasterWorkflow must not be charged for it.
    assert m.invoked is False, (
        "infra-class stage-1 verifier failure escalated to master (paid the full "
        "plan tax for a host flap, not a code defect)"
    )


# --------------------------------------------------------------------------- #
# Probe 2 — Cascade does not escalate to the strong stage on an infra-class
# stage-1 verifier failure.
#
# cascade.py:58 drives TestFeedbackWorkflow which branches only on result.ok
# (test_feedback.py:63). A stage-1 resource_exceeded=True verdict therefore looks
# identical to a genuine code failure, so the cheap stage burns its whole repair
# budget and cascade escalates to the expensive model — all on a host OOM.
#
# DESIRED: an infra-class verdict must not consume the strong (stage 2) model.
# --------------------------------------------------------------------------- #
def test_cascade_infra_failure_does_not_consume_strong_stage():
    cheap, strong = _CannedAgent("cheap"), _CannedAgent("strong")
    infra = VerifierResult(ok=False, message="OOM-killed", resource_exceeded=True)
    c = CascadeWorkflow([cheap, strong], _FakeVerifier([infra]),
                        max_iterations_per_stage=1)
    run(c.execute("p"))
    # The strong/expensive model must not be charged because the HOST verifier
    # OOM'd on the cheap stage — that is not a defect the strong model can fix.
    assert strong.calls == 0, (
        "cascade escalated to the strong stage on an infra-class (OOM) verifier "
        "verdict — burned the expensive model on a host flap"
    )


# --------------------------------------------------------------------------- #
# Probe 3 — Cascade stages share one un-isolated working directory: a failed
# cheap stage's partial writes poison the strong stage and its gate.
#
# cascade.py:56 hands self.working_directory straight to every stage's
# TestFeedbackWorkflow; there is no /tmp workspace copy or git-reset between
# stages (unlike graph mode). So the cheap stage's broken partial edit is the
# baseline the strong stage starts from.
#
# DESIRED: the strong stage should observe a tree NOT corrupted by the failed
# cheap stage (per-stage isolation / reset). We document the observed tree state
# at the strong stage's entry.
# --------------------------------------------------------------------------- #
def test_cascade_stages_are_isolated_from_partial_writes(tmp_path):
    sentinel = tmp_path / "module.py"
    sentinel.write_text("ORIGINAL\n")

    class _CorruptingAgent(_CannedAgent):
        """Cheap stage: leaves a broken partial edit, then fails the verifier."""

        async def run_async(self, piped_input=None) -> str:
            self.calls += 1
            sentinel.write_text("HALF-WRITTEN-GARBAGE")  # partial/broken edit
            return "cheap"

    observed = {}

    class _ObservingAgent(_CannedAgent):
        """Strong stage: records the tree state it inherits at entry."""

        async def run_async(self, piped_input=None) -> str:
            self.calls += 1
            observed["content"] = sentinel.read_text()
            return "strong"

    c = CascadeWorkflow(
        [_CorruptingAgent(), _ObservingAgent()],
        _FakeVerifier([_fail(), _ok()]),  # cheap fails, strong passes
        max_iterations_per_stage=1,
        working_directory=str(tmp_path),
    )
    run(c.execute("p"))
    # The strong stage must not inherit the cheap stage's debris; escalation
    # should be a clean retry, not a retry on top of a poisoned tree.
    assert observed.get("content") == "ORIGINAL\n", (
        "strong stage started from the cheap stage's corrupted partial write "
        "(no isolation/reset between cascade stages)"
    )


# --------------------------------------------------------------------------- #
# Probe 4 — ParallelSwarm imposes no per-branch deadline: one hung branch blocks
# asyncio.gather forever.
#
# pipeline.py:39 gathers with return_exceptions=True — that reaps CRASHES but a
# HANG is not an exception, so gather blocks until the wedged task completes
# (never). ToT / GenerateAndRank build the swarm from K branches, so one wedged
# worker stalls the whole multi-step build with no swarm-level timeout.
#
# DESIRED: the swarm should bound each branch so one hang cannot stall it forever
# (i.e. it should complete with the survivors). We probe by giving the swarm a
# generous wall (0.5s) and asserting it RETURNS rather than hanging. A control
# with two normal branches confirms the harness itself is not the limiter.
# --------------------------------------------------------------------------- #
def test_swarm_completes_when_one_branch_hangs(monkeypatch):
    # Control: two normal branches complete fast.
    swarm_ok = ParallelSwarm([_CannedAgent("a"), _CannedAgent("b")])
    assert run(asyncio.wait_for(swarm_ok.execute("in"), timeout=0.5)) == ["a", "b"]

    # The fix: AGY_SWARM_BRANCH_TIMEOUT bounds each branch with asyncio.wait_for, so
    # one wedged worker fails its OWN slot (reaped by return_exceptions=True) instead
    # of blocking the whole gather forever. With the knob armed, the survivor returns.
    monkeypatch.setenv("AGY_SWARM_BRANCH_TIMEOUT", "0.1")
    swarm_hang = ParallelSwarm([_HangingAgent(), _CannedAgent("survivor")])

    async def _drive():
        return await swarm_hang.execute("in")

    try:
        result = run(asyncio.wait_for(_drive(), timeout=0.5))
    except asyncio.TimeoutError:
        pytest.fail(
            "ParallelSwarm branch deadline did not fire: one wedged worker blocked "
            "asyncio.gather (the multi-step build would stall indefinitely)"
        )
    # The bounded hang fails its slot; the survivor still comes through.
    assert "survivor" in result


# --------------------------------------------------------------------------- #
# Probe 5 — Cascade has no escalation BUDGET / same-error early-exit: a
# persistently-red verifier (host broken / flaky test none of the models can fix)
# makes every stage exhaust its full repair budget producing identical failures.
#
# cascade.py:51 walks every stage, each running max_iterations_per_stage full
# generator+verifier rounds (test_feedback.py:57). With the SAME error every
# round there is no convergence and nothing curtails the stages*iters waste;
# iterations_used (cascade.py:59) only reports it after the fact.
#
# DESIRED: when every round returns the identical failure, cascade should
# short-circuit (an escalation budget / same-error early-exit) rather than pay
# the full stages*iters. We assert fewer than the worst-case 3*4=12 calls.
# --------------------------------------------------------------------------- #
def test_cascade_short_circuits_on_repeated_identical_failure():
    stages = [_CannedAgent("s1"), _CannedAgent("s2"), _CannedAgent("s3")]
    # The verifier ALWAYS returns the same error (a host/flaky-test condition no
    # model can repair). error_hash makes the identity explicit.
    same = VerifierResult(ok=False, message="identical boom", error_hash="deadbeef")
    c = CascadeWorkflow(stages, _FakeVerifier([same]), max_iterations_per_stage=4)
    run(c.execute("p"))
    total_calls = sum(s.calls for s in stages)
    # A resilient cascade detects the non-converging identical failure and bails
    # well short of the full 3*4 = 12 generator calls.
    assert total_calls < 12, (
        f"cascade ran the full worst-case {total_calls} generator calls on a "
        "repeated-identical verifier failure (no escalation budget / same-error "
        "early-exit)"
    )


# --------------------------------------------------------------------------- #
# Probe 6 (bonus from the analyst low-sev mode) — max_iterations_per_stage<=0
# makes cascade a silent no-op that still reports a "clean" terminal result.
#
# With max_iterations=0, each stage's feedback loop runs zero rounds
# (test_feedback.py:57), so NO generator ever runs and verified stays False.
# Cascade walks every stage (all no-ops) and returns '' with stalled=True /
# stage_used=-1 — indistinguishable at the result level from "all models tried
# and failed". The only tell is iterations_used==0.
#
# DESIRED: a config that ran ZERO work must be distinguishable from genuine
# all-stage failure — either it raises / refuses, or it exposes a signal the
# operator/ledger can branch on (here: iterations_used==0 with stalled True must
# not look like a normal converged-fail). We assert the misconfig is surfaced
# rather than silently swallowed: zero work must not present as a normal stall.
# --------------------------------------------------------------------------- #
def test_cascade_zero_iterations_is_distinguishable_from_real_failure():
    s1, s2 = _CannedAgent("s1"), _CannedAgent("s2")
    c = CascadeWorkflow([s1, s2], _FakeVerifier([_fail()]),
                        max_iterations_per_stage=0)
    out = run(c.execute("p"))
    # Confirm the no-op shape first (these hold on current source):
    assert s1.calls == 0 and s2.calls == 0
    assert c.iterations_used == 0
    # DESIRED resilient contract: a run where ZERO work happened must not present
    # as an ordinary converged-but-failed stall. It should either refuse loudly
    # (raise) or carry a distinct "no work attempted" signal — NOT silently look
    # like a normal stalled result with empty output.
    assert not (c.stalled is True and out == "" and c.stage_used == -1), (
        "cascade with max_iterations<=0 ran zero generators yet reported a normal "
        "stalled/empty result — a misconfiguration masquerading as all-stage failure"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))

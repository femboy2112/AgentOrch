"""PatWorkflow — Plan-after-Trial.

Stage 1 = direct generator, Stage 2 = MasterWorkflow on verifier failure.
The whole point is to skip the master tax on tasks the direct path solves.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional, Tuple

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import VerifierResult
from agy_orchestrator.workflows.pat import PatWorkflow


class _StubGenerator(AgentInstance):
    """No-op agent: returns a canned string per call, records prompts seen."""

    def __init__(self, *args, canned_outputs: Optional[List[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._canned = list(canned_outputs or ["stub-out"])
        self._idx = 0
        self.prompts_seen: List[str] = []

    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        self.prompts_seen.append(self.prompt)
        out = self._canned[min(self._idx, len(self._canned) - 1)]
        self._idx += 1
        return out


class _ScriptedVerifier:
    """Returns scripted outcomes per call. The whole point is
    to control whether Stage 1 'passes' and Stage 2 ever runs."""

    def __init__(self, results: List[Tuple[bool, str]]):
        self._results = list(results)
        self._idx = 0
        self.calls = 0

    async def verify(self, working_directory: str) -> VerifierResult:
        self.calls += 1
        ok, message = self._results[min(self._idx, len(self._results) - 1)]
        self._idx += 1
        return VerifierResult(
            ok=ok,
            message=message,
            returncode=0 if ok else 1,
            cmd="stub",
            duration_ms=0,
        )


class _StubMaster:
    """Stand-in for MasterWorkflow that records whether it was ever invoked."""

    def __init__(self, output: str = "master-out", verified: bool = True,
                 iterations_used: int = 5):
        self.output = output
        self.verified = verified
        self.approved = verified
        self.iterations_used = iterations_used
        self.invoked = False
        self.prompt_seen: Optional[str] = None

    async def execute(self, prompt: str) -> str:
        self.invoked = True
        self.prompt_seen = prompt
        return self.output


def _run(coro):
    return asyncio.run(coro)


def test_stage1_passes_skips_master():
    """The whole point of PaT: if the direct attempt verifies, master never
    runs. This is the 40% cost savings case from the paper."""
    gen = _StubGenerator(prompt="task", canned_outputs=["direct-out"])
    master = _StubMaster()
    verifier = _ScriptedVerifier([(True, "ok")])

    wf = PatWorkflow(direct_generator=gen, master_workflow=master, verifier=verifier)
    out = _run(wf.execute("task"))

    assert out == "direct-out"
    assert master.invoked is False, "master should NOT run when Stage 1 verifies"
    assert wf.verified is True
    assert wf.approved is True
    assert wf.stage_used == 0
    assert wf.iterations_used == 1
    assert verifier.calls == 1


def test_stage1_fails_escalates_to_master():
    """When the direct attempt fails the verifier, PaT must invoke master
    and forward the failure context so the planner doesn't start blind."""
    gen = _StubGenerator(prompt="task", canned_outputs=["direct-attempt"])
    master = _StubMaster(output="master-fixed", verified=True, iterations_used=4)
    # Stage 1 verify fails; master internally signals verified=True so PaT
    # doesn't need to re-verify after.
    verifier = _ScriptedVerifier([(False, "tests failed: missing X")])

    wf = PatWorkflow(direct_generator=gen, master_workflow=master, verifier=verifier)
    out = _run(wf.execute("original task"))

    assert out == "master-fixed"
    assert master.invoked is True
    assert wf.verified is True   # carried over from master.verified
    assert wf.stage_used == 1
    # Direct (1) + master's reported iterations (4)
    assert wf.iterations_used == 5
    # The escalation prompt to master must surface BOTH the failed
    # attempt AND the verifier error so the planner can route around it.
    assert "direct-attempt" in master.prompt_seen
    assert "tests failed: missing X" in master.prompt_seen
    assert "original task" in master.prompt_seen


def test_stage1_fail_master_silent_reverifies():
    """Some master paths complete without setting `verified` (e.g. all
    iterations exhausted). PaT must re-verify the final master output to
    produce a clean ledger signal — otherwise we'd silently report
    'verified=False' even when the work actually passes."""
    gen = _StubGenerator(prompt="t", canned_outputs=["bad-direct"])
    master = _StubMaster(output="master-result", verified=False, iterations_used=8)
    # First call: Stage 1 fails. Second call: PaT re-verifies after master,
    # this time succeeds (e.g. master fixed it but didn't set verified).
    verifier = _ScriptedVerifier([(False, "err"), (True, "ok-after-master")])

    wf = PatWorkflow(direct_generator=gen, master_workflow=master, verifier=verifier)
    out = _run(wf.execute("t"))

    assert out == "master-result"
    assert master.invoked is True
    assert wf.verified is True   # picked up from the re-verify
    assert verifier.calls == 2   # one for stage 1, one for post-master rescue


def test_stage1_fail_master_silent_still_failing_marks_stalled():
    """Both stages fail and master never set verified — PaT must surface
    that as stalled, not silently approve."""
    gen = _StubGenerator(prompt="t", canned_outputs=["bad-direct"])
    master = _StubMaster(output="bad-master", verified=False)
    verifier = _ScriptedVerifier([(False, "err1"), (False, "err2")])

    wf = PatWorkflow(direct_generator=gen, master_workflow=master, verifier=verifier)
    out = _run(wf.execute("t"))

    assert out == "bad-master"
    assert wf.verified is False
    assert wf.approved is False
    assert wf.stalled is True
    assert wf.stage_used == 1


def test_verifier_required():
    """No verifier = no way to decide Stage 1 success. Must raise loudly
    at construction so misconfigured dispatches fail fast."""
    gen = _StubGenerator(prompt="t")
    master = _StubMaster()
    with pytest.raises(ValueError, match="QualityVerifier"):
        PatWorkflow(direct_generator=gen, master_workflow=master, verifier=None)


def test_working_directory_propagates_to_verifier():
    """Cross-repo dispatch (--out-dir) must reach Stage 1 verifier too —
    same bug class as the one we fixed in adversarial.py. Regression
    guard."""
    gen = _StubGenerator(prompt="t", canned_outputs=["x"])
    master = _StubMaster()
    seen_cwds: List[str] = []

    class _CwdRecordingVerifier:
        async def verify(self, working_directory: str):
            seen_cwds.append(working_directory)
            return VerifierResult(
                ok=True,
                message="ok",
                returncode=0,
                cmd="stub",
                duration_ms=0,
            )

    wf = PatWorkflow(
        direct_generator=gen, master_workflow=master,
        verifier=_CwdRecordingVerifier(),
        working_directory="/tmp/elsewhere",
    )
    _run(wf.execute("t"))
    assert seen_cwds == ["/tmp/elsewhere"]

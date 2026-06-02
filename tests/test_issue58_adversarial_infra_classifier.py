"""Issue #58 — the adversarial loop must classify NON-timeout infra-class verifier
failures instead of regenerating code against a problem the code can't fix.

#50 already branches the loop on a verifier TIMEOUT / RESOURCE-kill (precise
flags). But any OTHER infra failure (suite couldn't import because a sibling left
the venv half-installed, "make: command not found", disk-full write, permission
flip, conftest import error) returns ``ok=False`` with a NORMAL returncode, so it
fell through to "send back to generator" and burned the whole ``max_iterations``
budget regenerating unbroken code.

The fix adds a conservative ``looks_like_infra`` classifier (core/agent.py) and a
``VerifierResult.infra_suspected`` flag (verifier.py). Because the markers are
HEURISTIC, the adversarial loop treats a hit as a SOFT signal: surface the real
cause + distinct log/event, allow exactly ONE regenerate (a real code bug that
merely *looks* infra-ish still gets a fix attempt), then bail.

These tests pin: the classifier (markers + non-firing), the verifier flag (set on
a marker'd failure, never on pass/timeout/OOM/generic), and the loop behaviour
(one regenerate then bail with a distinct reason + event; generic failure and
passing/timeout paths unchanged — no #50 regression).
"""

import asyncio

from agy_orchestrator.core.agent import looks_like_infra
from agy_orchestrator.execution.verifier import VerifierResult
from agy_orchestrator.workflows.adversarial import AdversarialReview


# ---- classifier (core/agent.py) -------------------------------------------

def test_looks_like_infra_fires_on_each_marker():
    for tail in (
        "make: command not found",
        "ModuleNotFoundError: No module named 'pytest'",
        "OSError: [Errno 28] No space left on device",
        "PermissionError: [Errno 13] Permission denied: '/var/run/x'",
        "ImportError while loading conftest '/repo/tests/conftest.py'.",
    ):
        assert looks_like_infra(tail) is True, tail


def test_looks_like_infra_case_insensitive():
    assert looks_like_infra("COMMAND NOT FOUND: make") is True
    assert looks_like_infra("No Module Named 'foo'") is True


def test_looks_like_infra_does_not_fire_on_normal_failure():
    # A plain assertion failure / nonzero exit is NOT infra.
    assert looks_like_infra("AssertionError: 2 != 3") is False
    assert looks_like_infra("1 failed, 4 passed in 0.12s") is False
    assert looks_like_infra("") is False
    assert looks_like_infra(None) is False


# ---- VerifierResult flag (verifier.py) ------------------------------------

def test_verifier_result_infra_suspected_default_false():
    # Backward-compat: the field defaults False so an unaware consumer is unchanged.
    assert VerifierResult(ok=True).infra_suspected is False
    assert VerifierResult(ok=False, message="boom").infra_suspected is False


def test_verify_sets_infra_suspected_on_marker(tmp_path):
    # Hermetic: a real subprocess that emits an infra marker to stderr and exits
    # nonzero. verify() must flag infra_suspected (and NOT resource_exceeded).
    from agy_orchestrator.execution.verifier import QualityVerifier

    v = QualityVerifier(["sh -c 'echo \"make: command not found\" 1>&2; exit 2'"])
    res = asyncio.run(v.verify(working_directory=str(tmp_path)))
    assert res.ok is False
    assert res.infra_suspected is True
    assert res.resource_exceeded is False
    assert res.timeout is False


def test_verify_does_not_flag_infra_on_plain_failure(tmp_path):
    from agy_orchestrator.execution.verifier import QualityVerifier

    v = QualityVerifier(["sh -c 'echo \"AssertionError: boom\" 1>&2; exit 1'"])
    res = asyncio.run(v.verify(working_directory=str(tmp_path)))
    assert res.ok is False
    assert res.infra_suspected is False


# ---- adversarial loop (adversarial.py) ------------------------------------

class _FakeGen:
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

    async def run_async(self):  # pragma: no cover - verifier short-circuits
        return "needs work"


class _FakeVerifier:
    """Returns ``result`` every call (counts how many times it was asked)."""

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


def _infra_result():
    return VerifierResult(
        ok=False,
        message="Command failed with exit code 2: make check",
        returncode=2,
        stderr_tail="make: command not found",
        infra_suspected=True,
    )


def test_infra_suspected_allows_one_regenerate_then_bails():
    # SOFT signal: regenerate exactly once (gen called twice — initial + one retry),
    # then bail with the distinct infra reason. Does NOT thrash to max_iterations.
    v = _FakeVerifier(_infra_result())
    rv = _review(v)
    out = asyncio.run(rv.execute("build the thing"))
    assert rv.generator.calls == 2  # initial generate + exactly ONE regenerate
    assert v.calls == 2
    assert rv.verifier_infra_failed is True
    assert rv.infra_reason == "verifier_infra_suspected"
    assert rv.verified is False and rv.approved is False
    assert out == "GENERATED OUTPUT 2"


def test_infra_suspected_emits_distinct_outcome_event():
    v = _FakeVerifier(_infra_result())
    events = []
    rv = _review(v, events=events)
    asyncio.run(rv.execute("build the thing"))
    outcomes = [
        e["data"]["orchestration"].get("outcome")
        for e in events
        if e.get("data", {}).get("event") == "orchestration_transition"
    ]
    assert "verifier_infra_suspected" in outcomes


def test_infra_suspected_iteration_outcome_reports_reason():
    v = _FakeVerifier(_infra_result())
    rv = _review(v)
    asyncio.run(rv.execute("x"))
    assert rv._iteration_outcome() == "verifier_infra_suspected"


def test_genuine_failure_without_infra_marker_still_loops():
    # No infra_suspected flag => unchanged repair loop to the iteration budget.
    v = _FakeVerifier(
        VerifierResult(ok=False, message="exit code 1", returncode=1,
                       stderr_tail="AssertionError: boom")
    )
    rv = _review(v)
    asyncio.run(rv.execute("build the thing"))
    assert rv.generator.calls == 5
    assert v.calls == 5
    assert rv.verifier_infra_failed is False
    assert rv.infra_reason is None


def test_timeout_branch_unaffected_by_infra_change():
    # #50 precise bail must still win and NOT be reclassified as infra_suspected.
    v = _FakeVerifier(VerifierResult(ok=False, message="timed out", timeout=True))
    rv = _review(v)
    asyncio.run(rv.execute("build the thing"))
    assert rv.generator.calls == 1
    assert rv.infra_reason == "verifier_timeout"


def test_passing_verifier_unaffected_by_infra_change():
    v = _FakeVerifier(VerifierResult(ok=True, message="all passed"))
    rv = _review(v)
    asyncio.run(rv.execute("build the thing"))
    assert rv.generator.calls == 1
    assert rv.verified is True and rv.approved is True
    assert rv.verifier_infra_failed is False

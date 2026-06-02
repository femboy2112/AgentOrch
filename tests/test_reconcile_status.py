"""Reconcile gate disposition logic (#44).

The gate now ALWAYS records why reconcile did or didn't run. The decision is a
pure helper, _decide_reconcile_status, returning "run" or "skipped:<reason>".
The hard verifier gate is kept ONLY for the "fail" disposition; for warn/open-task
the read-only station runs regardless of verifier_green (we most want the
dead-wiring trace exactly when the build is shaky).
"""
from dataclasses import asdict

from harness.dispatch import DispatchResult, _decide_reconcile_status


def test_not_enabled():
    assert _decide_reconcile_status(
        reconcile_enabled=False, plan_only=False, has_output=True,
        disposition="warn", verifier_green=True,
    ) == "skipped:not_enabled"


def test_plan_only():
    assert _decide_reconcile_status(
        reconcile_enabled=True, plan_only=True, has_output=True,
        disposition="warn", verifier_green=True,
    ) == "skipped:plan_only"


def test_no_output():
    assert _decide_reconcile_status(
        reconcile_enabled=True, plan_only=False, has_output=False,
        disposition="warn", verifier_green=True,
    ) == "skipped:no_output"


def test_fail_disposition_blocks_on_non_green():
    assert _decide_reconcile_status(
        reconcile_enabled=True, plan_only=False, has_output=True,
        disposition="fail", verifier_green=False,
    ) == "skipped:verifier_not_green"


def test_fail_disposition_runs_when_green():
    assert _decide_reconcile_status(
        reconcile_enabled=True, plan_only=False, has_output=True,
        disposition="fail", verifier_green=True,
    ) == "run"


def test_warn_disposition_runs_even_when_not_green():
    # This is the gate-reconsideration #44 asks for: warn-disposition runs the
    # read-only station even on a non-green verifier (build is shaky -> we want
    # the trace), so it is NOT skipped.
    status = _decide_reconcile_status(
        reconcile_enabled=True, plan_only=False, has_output=True,
        disposition="warn", verifier_green=False,
    )
    assert status == "run"
    assert not status.startswith("skipped")


def test_open_task_disposition_runs_when_not_green():
    assert _decide_reconcile_status(
        reconcile_enabled=True, plan_only=False, has_output=True,
        disposition="open-task", verifier_green=False,
    ) == "run"


def _minimal_result(**kw) -> DispatchResult:
    base = dict(run_id="r", run_dir="/tmp/r", mode="master",
                generator="codex", critic="agy", success=True, duration_s=1.0)
    base.update(kw)
    return DispatchResult(**base)


def test_reconcile_status_serializes_into_meta_json():
    # meta.json is json.dumps(asdict(result)) — the #44 headline is that
    # reconcile_status reaches meta.json so a null `reconciliation` is never
    # ambiguous. Assert it survives asdict() (the exact meta.json path).
    meta = asdict(_minimal_result(reconcile_status="ran"))
    assert "reconcile_status" in meta
    assert meta["reconcile_status"] == "ran"


def test_reconcile_status_present_even_when_not_reconcile_run():
    # A non-reconcile run must STILL carry an explicit status (not a bare null),
    # so absence/None is never confused with "skipped". The dispatch finally-block
    # backstop sets "skipped:not_enabled"; assert that value round-trips to meta.
    meta = asdict(_minimal_result(reconcile_status="skipped:not_enabled",
                                  reconciliation=None))
    assert meta["reconciliation"] is None
    assert meta["reconcile_status"] == "skipped:not_enabled"

"""Issue #43 — dispatch-level wiring of the reconciliation station.

The station's own logic (parsing, classification, witness, distinct verdict) is
covered by tests/test_issue43_reconcile.py. THESE tests cover how dispatch.py
invokes it: opt-in gating (--reconcile / mission-critical / AGY_RECONCILE), the
durable runs/<id>/reconcile.json artifact, the meta.json field, and that the
warn-only default never fails a good build while an explicit "fail" disposition
does.

Hermetic: ``_run_workflow`` is monkeypatched to a trivial coroutine and
``_run_reconciliation`` to a canned verdict, so no worker CLI / LLM is spawned.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness import dispatch as dispatch_mod
from agy_orchestrator.workflows.reconcile import (
    MechanismFinding,
    ReconciliationResult,
    Witness,
)


def _ok_workflow_factory(output="built the thing"):
    async def _wf(*args, **kwargs):
        return (output, None)
    return _wf


def _canned(disposition="warn", *, dead=False):
    findings = []
    if dead:
        findings.append(MechanismFinding(
            name="surprise", classification="exists_not_load_bearing",
            sub_kind="stub_constant", location="world.py:42",
            witness=Witness(value=0.0, description="ablation moved nothing"),
            evidence="surprise() returns 0.0; real impl in _real_surprise (uncalled)",
        ))
    return ReconciliationResult(
        reconciled=not dead, findings_list=findings, disposition=disposition,
    )


def _patch_recon(monkeypatch, result):
    async def _fake_recon(*args, **kwargs):
        return result
    monkeypatch.setattr(dispatch_mod, "_run_reconciliation", _fake_recon)


def test_reconcile_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow_factory())
    monkeypatch.delenv("AGY_RECONCILE", raising=False)
    # If it DID run, this would blow up (no real agent); assert it does not.
    async def _boom(*a, **k):
        raise AssertionError("reconciliation should not run when disabled")
    monkeypatch.setattr(dispatch_mod, "_run_reconciliation", _boom)

    result = dispatch_mod.dispatch("do x", mode="direct", out_dir=str(tmp_path),
                                   heartbeat_interval=0)
    assert result.success is True
    assert result.reconciliation is None
    assert not (Path(result.run_dir) / "reconcile.json").exists()


def test_reconcile_clean_build(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow_factory())
    _patch_recon(monkeypatch, _canned(dead=False))

    result = dispatch_mod.dispatch("do x", mode="direct", out_dir=str(tmp_path),
                                   reconcile=True, heartbeat_interval=0)
    assert result.success is True
    assert result.reconciliation is not None
    assert result.reconciliation["reconciled"] is True
    assert result.reconciliation["verdict"] == "reconciled"
    art = json.loads((Path(result.run_dir) / "reconcile.json").read_text())
    assert art["reconciled"] is True
    meta = json.loads((Path(result.run_dir) / "meta.json").read_text())
    assert meta["reconciliation"]["verdict"] == "reconciled"


def test_reconcile_warn_does_not_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow_factory())
    _patch_recon(monkeypatch, _canned(disposition="warn", dead=True))

    result = dispatch_mod.dispatch("do x", mode="direct", out_dir=str(tmp_path),
                                   reconcile=True, heartbeat_interval=0)
    # warn default: dead wiring surfaced loudly but the run still SUCCEEDS.
    assert result.success is True
    assert result.reconciliation["reconciled"] is False
    findings = result.reconciliation["findings"]
    assert findings[0]["sub_kind"] == "stub_constant"
    assert findings[0]["location"] == "world.py:42"


def test_reconcile_fail_disposition_flips_success(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow_factory())
    _patch_recon(monkeypatch, _canned(disposition="fail", dead=True))

    result = dispatch_mod.dispatch("do x", mode="direct", out_dir=str(tmp_path),
                                   reconcile=True, reconcile_disposition="fail",
                                   heartbeat_interval=0)
    assert result.success is False
    assert "reconciliation failed" in (result.error or "")
    assert result.reconciliation["reconciled"] is False


def test_mission_critical_enables_reconcile_warn(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow_factory())
    # default-ON for mission-critical, but disposition stays warn (operator's
    # chosen default) -> a dead finding does NOT fail the run.
    _patch_recon(monkeypatch, _canned(disposition="warn", dead=True))

    result = dispatch_mod.dispatch("do x", mode="direct", out_dir=str(tmp_path),
                                   mission_critical=True, heartbeat_interval=0)
    assert result.success is True
    assert result.reconciliation is not None
    assert result.reconciliation["reconciled"] is False


def test_reconcile_skipped_when_no_output(tmp_path, monkeypatch):
    # A build that produced nothing didn't converge -> no reconciliation.
    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow_factory(output=""))
    async def _boom(*a, **k):
        raise AssertionError("should not reconcile a non-converged build")
    monkeypatch.setattr(dispatch_mod, "_run_reconciliation", _boom)

    result = dispatch_mod.dispatch("do x", mode="direct", out_dir=str(tmp_path),
                                   reconcile=True, heartbeat_interval=0)
    assert result.reconciliation is None


def test_reconcile_error_is_best_effort(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow_factory())
    async def _raise(*a, **k):
        raise RuntimeError("station blew up")
    monkeypatch.setattr(dispatch_mod, "_run_reconciliation", _raise)

    result = dispatch_mod.dispatch("do x", mode="direct", out_dir=str(tmp_path),
                                   reconcile=True, heartbeat_interval=0)
    # A station failure must never fail an otherwise-good build.
    assert result.success is True
    assert result.reconciliation is None


def test_reconcile_env_var_enables(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow_factory())
    monkeypatch.setenv("AGY_RECONCILE", "1")
    _patch_recon(monkeypatch, _canned(dead=False))

    result = dispatch_mod.dispatch("do x", mode="direct", out_dir=str(tmp_path),
                                   heartbeat_interval=0)
    assert result.reconciliation is not None

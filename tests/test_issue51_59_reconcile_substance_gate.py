"""Reconcile substance gate + critic starvation (#51, #59).

#51: reconcile_status="ran" must reflect a NON-HOLLOW trace. A reconcile whose
critic chain returned no JSON (parse_error), traced zero mechanisms (empty), or
whose critic was starved (watchdog/fallback exhaustion) is HOLLOW — it records a
distinct ran:* variant and must NOT contribute reconciled=true.

#59: a recon agent that DIDN'T raise but was watchdog-killed / fallback-exhausted
returns a degenerate reply; it must set reconcile_status=ran:critic_starved and
not pass as a clean "ran".

These are one coherent change (ran:critic_starved is one of #51's variants). The
agent is fully mocked (no LLM / network); dispatch-level tests mirror the
monkeypatch pattern in tests/test_issue43_reconcile_wiring.py.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agy_orchestrator.core.agent import WATCHDOG_MARKER, AgentInstance
from agy_orchestrator.workflows.reconcile import (
    MechanismFinding,
    ReconciliationResult,
    ReconciliationReview,
    Witness,
)

import harness.dispatch as dispatch_mod
from harness.dispatch import _detect_recon_starvation


# --------------------------------------------------------------------------- #
# stub agent (mirrors test_issue43_reconcile.py's _StubAgent), with optional
# post-run watchdog/stderr telemetry so #59 starvation can be simulated.
# --------------------------------------------------------------------------- #
class _StubAgent(AgentInstance):
    def __init__(self, reply: str, *, watchdog_reason=None, stderr=""):
        super().__init__(prompt="")
        self._reply = reply
        self._post_watchdog = watchdog_reason
        self._post_stderr = stderr

    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input=None) -> str:
        # Simulate a run that returned WITHOUT raising but was starved: the agent's
        # post-run telemetry carries the watchdog reason / exhaustion marker.
        if self._post_watchdog is not None:
            self._watchdog_reason = self._post_watchdog
        if self._post_stderr:
            self.stderr = self._post_stderr
        return self._reply


def _load_bearing_reply() -> str:
    return json.dumps(
        {
            "findings": [
                {
                    "name": "learner",
                    "classification": "load_bearing",
                    "location": "world.py:10",
                    "witness": {"value": 2.0},
                    "evidence": "on the real loop",
                }
            ]
        }
    )


def _dead_reply() -> str:
    return json.dumps(
        {
            "findings": [
                {
                    "name": "surprise",
                    "classification": "exists_not_load_bearing",
                    "sub_kind": "stub_constant",
                    "location": "world.py:42",
                    "witness": {"value": 0},
                    "evidence": "hardcoded 0.0",
                }
            ]
        }
    )


# --------------------------------------------------------------------------- #
# (#51) substance_status reflects the RESULT, not merely a non-exception return.
# --------------------------------------------------------------------------- #
def test_substance_status_n_findings():
    r = ReconciliationResult(
        reconciled=True,
        findings_list=[
            MechanismFinding(name="a", classification="load_bearing"),
            MechanismFinding(name="b", classification="load_bearing"),
        ],
    )
    assert r.substance_status() == "ran:2_findings"
    assert r.is_substantive is True


def test_substance_status_empty_is_hollow_and_not_reconciled():
    # A clean parse with ZERO mechanisms traced is HOLLOW: the trace never looked
    # at anything, so it can't read as "all clear".
    station = ReconciliationReview(_StubAgent(json.dumps({"findings": []})), goal="g")
    result = asyncio.run(station.execute())
    assert result.parse_error is None
    assert result.findings_list == []
    assert result.substance_status() == "ran:empty"
    assert result.is_substantive is False
    assert result.reconciled is False  # #51: empty does not contribute reconciled=true


def test_substance_status_parse_error_is_hollow():
    station = ReconciliationReview(_StubAgent("the model rambled, no JSON"), goal="g")
    result = asyncio.run(station.execute())
    assert result.parse_error is not None
    assert result.substance_status() == "ran:parse_error"
    assert result.is_substantive is False
    assert result.reconciled is False


def test_substance_status_findings_is_substantive():
    station = ReconciliationReview(_StubAgent(_load_bearing_reply()), goal="g")
    result = asyncio.run(station.execute())
    assert result.substance_status() == "ran:1_findings"
    assert result.is_substantive is True
    assert result.reconciled is True


# --------------------------------------------------------------------------- #
# (#59) mark_starved forces reconciled=False regardless of how the parse landed.
# --------------------------------------------------------------------------- #
def test_mark_starved_forces_not_reconciled_even_on_clean_parse():
    # A starved critic that HAPPENED to emit a well-formed all-load-bearing JSON
    # must still be hollow — the trace was killed, the verdict is untrustworthy.
    r = ReconciliationResult(
        reconciled=True,
        findings_list=[MechanismFinding(name="a", classification="load_bearing")],
    )
    assert r.reconciled is True
    r.mark_starved("stalled")
    assert r.is_starved is True
    assert r.starved_reason == "stalled"
    assert r.reconciled is False
    assert r.substance_status() == "ran:critic_starved"
    assert r.is_substantive is False


def test_mark_starved_empty_reason_defaults():
    r = ReconciliationResult(reconciled=True)
    r.mark_starved("")
    assert r.starved_reason == "critic_starved"
    assert r.substance_status() == "ran:critic_starved"


def test_to_dict_carries_substance_fields():
    r = ReconciliationResult(reconciled=True, findings_list=[])
    d = r.to_dict()
    assert d["substantive"] is False
    assert d["substance_status"] == "ran:empty"
    assert d["starved_reason"] is None


# --------------------------------------------------------------------------- #
# (#59) _detect_recon_starvation reads the agent's post-run telemetry.
# --------------------------------------------------------------------------- #
def test_detect_starvation_via_watchdog_reason():
    a = _StubAgent("{}")
    a._watchdog_reason = "stalled"
    assert _detect_recon_starvation(a) == "stalled"


def test_detect_starvation_via_stderr_watchdog_marker():
    # A FallbackAgent copies the sub's stderr (carrying the marker) up on a
    # success-shaped return; the slug must be recovered.
    a = _StubAgent("{}")
    a.stderr = f"...noise... {WATCHDOG_MARKER}verbose] more noise"
    assert _detect_recon_starvation(a) == "verbose"


def test_detect_starvation_via_fallback_exhausted_marker():
    a = _StubAgent("{}")
    a.stderr = "All 6 fallback attempts exhausted (3 providers x 2 cycles)"
    assert _detect_recon_starvation(a) == "fallback_exhausted"


def test_detect_starvation_none_on_healthy_agent():
    a = _StubAgent("{}")
    assert _detect_recon_starvation(a) is None


# --------------------------------------------------------------------------- #
# dispatch-level: reconcile_status is substance-derived end-to-end.
# Mirrors tests/test_issue43_reconcile_wiring.py monkeypatch shape.
# --------------------------------------------------------------------------- #
def _ok_workflow_factory(output="built the thing"):
    async def _wf(*args, **kwargs):
        return (output, None)
    return _wf


def _patch_recon(monkeypatch, result):
    async def _fake_recon(*args, **kwargs):
        return result
    monkeypatch.setattr(dispatch_mod, "_run_reconciliation", _fake_recon)


def test_dispatch_reconcile_status_findings(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow_factory())
    _patch_recon(
        monkeypatch,
        ReconciliationResult(
            reconciled=True,
            findings_list=[MechanismFinding(name="a", classification="load_bearing")],
        ),
    )
    result = dispatch_mod.dispatch("do x", mode="direct", out_dir=str(tmp_path),
                                   reconcile=True, heartbeat_interval=0)
    meta = json.loads((Path(result.run_dir) / "meta.json").read_text())
    assert meta["reconcile_status"] == "ran:1_findings"
    assert meta["reconciliation"]["substantive"] is True


def test_dispatch_reconcile_status_empty_is_hollow(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow_factory())
    _patch_recon(monkeypatch, ReconciliationResult(reconciled=False, findings_list=[]))
    result = dispatch_mod.dispatch("do x", mode="direct", out_dir=str(tmp_path),
                                   reconcile=True, heartbeat_interval=0)
    meta = json.loads((Path(result.run_dir) / "meta.json").read_text())
    # Hollow: ran:empty, NOT a bare green "ran".
    assert meta["reconcile_status"] == "ran:empty"
    assert meta["reconciliation"]["substantive"] is False
    assert meta["reconciliation"]["reconciled"] is False
    # warn default: a hollow trace never fails an otherwise-good build.
    assert result.success is True


def test_dispatch_reconcile_status_critic_starved(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow_factory())
    starved = ReconciliationResult(
        reconciled=True,
        findings_list=[MechanismFinding(name="a", classification="load_bearing")],
    ).mark_starved("fallback_exhausted")
    _patch_recon(monkeypatch, starved)
    result = dispatch_mod.dispatch("do x", mode="direct", out_dir=str(tmp_path),
                                   reconcile=True, heartbeat_interval=0)
    meta = json.loads((Path(result.run_dir) / "meta.json").read_text())
    assert meta["reconcile_status"] == "ran:critic_starved"
    assert meta["reconciliation"]["reconciled"] is False
    assert meta["reconciliation"]["starved_reason"] == "fallback_exhausted"


def test_dispatch_starved_under_fail_disposition_flips_success(tmp_path, monkeypatch):
    # A starved trace is not reconciled, so under a fail-disposition gate the run
    # fails — but with a hollow-aware error, not a "0 findings" message.
    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow_factory())
    starved = ReconciliationResult(
        reconciled=True, disposition="fail"
    ).mark_starved("stalled")
    _patch_recon(monkeypatch, starved)
    result = dispatch_mod.dispatch(
        "do x", mode="direct", out_dir=str(tmp_path),
        reconcile=True, reconcile_disposition="fail", heartbeat_interval=0,
    )
    assert result.success is False
    assert "did not verify" in (result.error or "")
    assert "ran:critic_starved" in (result.error or "")


def test_run_reconciliation_marks_starved(tmp_path, monkeypatch):
    # End-to-end through _run_reconciliation: a recon agent that returns a clean
    # reply but whose post-run telemetry shows a watchdog trip is marked starved.
    starved_agent = _StubAgent(_load_bearing_reply(), watchdog_reason="stalled")
    monkeypatch.setattr(dispatch_mod, "_build_role_agent_compat",
                        lambda *a, **k: starved_agent)

    result = asyncio.run(dispatch_mod._run_reconciliation(
        run_id="r", goal="g", critic_chain=["stub"], fallback=False,
        codex_config=None, critic_overrides=None, watchdog_scale=1.0,
        post_construct_hook=None, working_directory=str(tmp_path), disposition="warn",
    ))
    assert result.is_starved is True
    assert result.starved_reason == "stalled"
    assert result.reconciled is False
    assert result.substance_status() == "ran:critic_starved"


def test_run_reconciliation_healthy_not_starved(tmp_path, monkeypatch):
    healthy = _StubAgent(_dead_reply())  # a substantive (dead-wiring) trace
    monkeypatch.setattr(dispatch_mod, "_build_role_agent_compat",
                        lambda *a, **k: healthy)

    result = asyncio.run(dispatch_mod._run_reconciliation(
        run_id="r", goal="g", critic_chain=["stub"], fallback=False,
        codex_config=None, critic_overrides=None, watchdog_scale=1.0,
        post_construct_hook=None, working_directory=str(tmp_path), disposition="warn",
    ))
    assert result.is_starved is False
    assert result.substance_status() == "ran:1_findings"
    assert result.is_substantive is True

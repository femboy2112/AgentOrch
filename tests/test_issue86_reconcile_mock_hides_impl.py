"""Issue #86: mock-only coverage must not silently certify an unmeasured load-bearing claim.

A ``master --mission-critical --reconcile`` run shipped ``verified + reconciled`` for a
goal-named cert whose only test MOCK-REPLACED the symbol (so a real ``KeyError`` on first
call was invisible) and whose witness was ``measured: false`` (inferred by reading source,
never executed). Two additive gates close the calibration escape:

* Part A — ``to_dict()`` surfaces ``measured_coverage`` + ``witnesses_unmeasured`` and the
  dispatch honesty-note qualifier, so "reconciled" stops *overstating*. Never flips the
  verdict.
* Part B — a deterministic AST pass over the TEST suite flags a load_bearing+measured:false
  finding whose symbol is only mock-patched (never exercised for real). Default: FLAG only
  (``mock_hides_implementation=True`` + honest note), which can NEVER over-suppress a legit
  run (a unit-test mock can coexist with a real integration exercise that static analysis
  cannot see — the #81 over-suppression lesson). Opt-in ``AGY_RECONCILE_MOCK_STRICT=1``
  under the ``fail`` disposition flips it to dead wiring so the corpse hard-fails.

Hermetic: the agent is a canned-reply stub; Part B writes tiny temp source/test trees and
points ``working_directory`` at them — no worker CLI or network.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.workflows.reconcile import (
    MechanismFinding,
    ReconciliationResult,
    ReconciliationReview,
    Witness,
)
from harness.dispatch import _apply_reconcile_honesty_note


class _StubAgent(AgentInstance):
    """AgentInstance whose run_async returns canned reconcile JSON."""

    def __init__(self, reply: str, prompt: str = ""):
        super().__init__(prompt=prompt)
        self._reply = reply

    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input=None) -> str:
        return self._reply


def _station(tmp_path) -> ReconciliationReview:
    return ReconciliationReview(
        _StubAgent("{}"),
        goal="g",
        working_directory=str(tmp_path),
        disposition="fail",
    )


def _run(reply: str, tmp_path) -> ReconciliationResult:
    station = ReconciliationReview(
        _StubAgent(reply),
        goal="cert aggregator is load-bearing",
        working_directory=str(tmp_path),
        disposition="fail",
    )
    result = asyncio.run(station.execute())
    assert station.result is result
    return result


def _load_bearing(name="_d1e_world_unit", *, measured=False, location="pkg/cert.py:1"):
    return MechanismFinding(
        name=name,
        classification="load_bearing",
        location=location,
        witness=Witness(value=1.0, description="model inferred moved", measured=measured),
        evidence="reported as on path",
    )


def _reply(name="_d1e_world_unit", location="pkg/cert.py:1"):
    return json.dumps(
        {
            "findings": [
                {
                    "name": name,
                    "classification": "load_bearing",
                    "location": location,
                    "witness": {"value": 1.0, "description": "model inferred moved"},
                    "evidence": "reported as on path",
                }
            ]
        }
    )


def _write_tree(tmp_path, test_source: str, prod_source: str | None = None):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "cert.py").write_text(
        prod_source
        or "def _d1e_world_unit():\n"
           "    raise KeyError('first real call crashes')\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_cert.py").write_text(test_source, encoding="utf-8")


# --------------------------------------------------------------------------- Part A

@pytest.mark.not_slow
def test_to_dict_reports_measured_coverage_and_unmeasured_flag():
    result = ReconciliationResult(
        reconciled=True,
        findings_list=[
            _load_bearing("_d1e_world_unit"),
            MechanismFinding(name="missing", classification="absent"),
        ],
    )
    d = result.to_dict()
    assert d["measured_coverage"] == {"load_bearing": 1, "measured": 0, "unmeasured": 1}
    assert d["witnesses_unmeasured"] is True
    assert d["mock_hidden_load_bearing"] is False


@pytest.mark.not_slow
def test_witnesses_unmeasured_false_when_measured_or_no_load_bearing():
    measured = ReconciliationResult(reconciled=True, findings_list=[_load_bearing(measured=True)])
    assert measured.to_dict()["witnesses_unmeasured"] is False
    assert measured.to_dict()["measured_coverage"] == {
        "load_bearing": 1, "measured": 1, "unmeasured": 0,
    }

    none = ReconciliationResult(
        reconciled=False,
        findings_list=[
            MechanismFinding(name="dead", classification="exists_not_load_bearing", sub_kind="uncalled")
        ],
    )
    assert none.to_dict()["witnesses_unmeasured"] is False
    assert none.to_dict()["measured_coverage"] == {
        "load_bearing": 0, "measured": 0, "unmeasured": 0,
    }


@pytest.mark.not_slow
def test_unmeasured_honesty_qualifier_reaches_summary_and_ledger_note():
    reconciliation = ReconciliationResult(reconciled=True, findings_list=[_load_bearing()]).to_dict()
    quality = {"confidence": "verified", "note": "tests passed"}
    _apply_reconcile_honesty_note(reconciliation, quality)
    assert reconciliation["verdict"].startswith("reconciled")
    assert "load-bearing witnesses inferred, not ablated" in reconciliation["verdict"]
    assert "load-bearing witnesses inferred, not ablated" in quality["note"]


@pytest.mark.not_slow
def test_mock_hidden_qualifier_reaches_summary_and_ledger_note():
    f = _load_bearing()
    f.mock_hides_implementation = True
    reconciliation = ReconciliationResult(reconciled=True, findings_list=[f]).to_dict()
    assert reconciliation["mock_hidden_load_bearing"] is True
    quality = {"confidence": "verified", "note": "tests passed"}
    _apply_reconcile_honesty_note(reconciliation, quality)
    assert "only mock-covered in tests" in reconciliation["verdict"]
    assert "only mock-covered in tests" in quality["note"]


@pytest.mark.not_slow
def test_honesty_note_noop_when_clean():
    reconciliation = ReconciliationResult(reconciled=True, findings_list=[_load_bearing(measured=True)]).to_dict()
    quality = {"confidence": "verified", "note": "tests passed"}
    _apply_reconcile_honesty_note(reconciliation, quality)
    assert reconciliation["verdict"] == "reconciled"
    assert quality["note"] == "tests passed"


# --------------------------------------------------------------------------- Part B

@pytest.mark.not_slow
def test_mock_only_flagged_by_default_no_flip(tmp_path):
    """Default (no strict): the corpse is FLAGGED but the verdict is NOT flipped."""
    _write_tree(
        tmp_path,
        "from unittest.mock import patch\n"
        "from pkg import cert\n\n"
        "@patch('pkg.cert._d1e_world_unit')\n"
        "def test_mocked(mock_world):\n"
        "    assert mock_world is not None\n\n"
        "def test_patch_object():\n"
        "    with patch.object(cert, '_d1e_world_unit'):\n"
        "        assert True\n",
    )
    result = _run(_reply(), tmp_path)

    finding = result.findings_list[0]
    assert finding.mock_hides_implementation is True
    assert finding.classification == "load_bearing"  # NOT flipped by default
    assert result.reconciled is True
    assert result.should_fail_run is False
    assert "only mock-patched in tests" in finding.evidence
    d = result.to_dict()
    assert d["mock_hidden_load_bearing"] is True
    assert d["findings"][0]["mock_hides_implementation"] is True


@pytest.mark.not_slow
def test_mock_only_flipped_in_strict_mode_under_fail(tmp_path, monkeypatch):
    """Opt-in strict + fail disposition: the corpse flips to dead wiring and hard-fails."""
    monkeypatch.setenv("AGY_RECONCILE_MOCK_STRICT", "1")
    _write_tree(
        tmp_path,
        "from unittest.mock import patch\n\n"
        "@patch('pkg.cert._d1e_world_unit')\n"
        "def test_mocked(mock_world):\n"
        "    assert mock_world is not None\n",
    )
    result = _run(_reply(), tmp_path)

    finding = result.findings_list[0]
    assert finding.mock_hides_implementation is True
    assert finding.classification == "exists_not_load_bearing"
    assert finding.sub_kind == "mocked_none"
    assert finding.witness.value == 0.0
    assert finding.is_dead is True
    assert result.reconciled is False
    assert result.should_fail_run is True


@pytest.mark.not_slow
def test_strict_mode_does_not_flip_under_warn_disposition(tmp_path, monkeypatch):
    """Strict only bites under 'fail'; a warn-disposition run is flag-only."""
    monkeypatch.setenv("AGY_RECONCILE_MOCK_STRICT", "1")
    _write_tree(
        tmp_path,
        "from unittest.mock import patch\n\n"
        "@patch('pkg.cert._d1e_world_unit')\n"
        "def test_mocked(mock_world):\n"
        "    assert mock_world is not None\n",
    )
    station = ReconciliationReview(
        _StubAgent(_reply()),
        goal="g",
        working_directory=str(tmp_path),
        disposition="warn",
    )
    result = asyncio.run(station.execute())
    finding = result.findings_list[0]
    assert finding.mock_hides_implementation is True
    assert finding.classification == "load_bearing"
    assert result.reconciled is True


@pytest.mark.not_slow
def test_symbol_with_direct_real_test_call_is_not_flagged(tmp_path, monkeypatch):
    """A symbol a test invokes for REAL (not just mocks) is genuinely exercised -> never flagged."""
    monkeypatch.setenv("AGY_RECONCILE_MOCK_STRICT", "1")
    _write_tree(
        tmp_path,
        "from unittest.mock import patch\n"
        "from pkg import cert\n\n"
        "@patch('pkg.cert._d1e_world_unit')\n"
        "def test_mocked(mock_world):\n"
        "    assert mock_world is not None\n\n"
        "def test_real_call():\n"
        "    cert._d1e_world_unit()\n",
    )
    result = _run(_reply(), tmp_path)
    finding = result.findings_list[0]
    assert result.reconciled is True
    assert finding.classification == "load_bearing"
    assert finding.mock_hides_implementation is None


@pytest.mark.not_slow
def test_never_mock_patched_symbol_is_not_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_RECONCILE_MOCK_STRICT", "1")
    _write_tree(tmp_path, "def test_placeholder():\n    assert True\n")
    result = _run(_reply(), tmp_path)
    finding = result.findings_list[0]
    assert result.reconciled is True
    assert finding.classification == "load_bearing"
    assert finding.mock_hides_implementation is None


@pytest.mark.not_slow
def test_measured_load_bearing_is_not_flagged_even_when_mocked(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_RECONCILE_MOCK_STRICT", "1")
    _write_tree(
        tmp_path,
        "from unittest.mock import patch\n\n"
        "@patch('pkg.cert._d1e_world_unit')\n"
        "def test_mocked(mock_world):\n"
        "    assert mock_world is not None\n",
    )
    result = ReconciliationResult(
        reconciled=True, disposition="fail", findings_list=[_load_bearing(measured=True)],
    )
    out = _station(tmp_path)._flag_mock_hidden_findings(result)
    finding = out.findings_list[0]
    assert out.reconciled is True
    assert finding.classification == "load_bearing"
    assert finding.mock_hides_implementation is None


@pytest.mark.not_slow
def test_mock_check_env_opt_out_is_noop(tmp_path, monkeypatch):
    _write_tree(
        tmp_path,
        "from unittest.mock import patch\n\n"
        "@patch('pkg.cert._d1e_world_unit')\n"
        "def test_mocked(mock_world):\n"
        "    assert mock_world is not None\n",
    )
    result = ReconciliationResult(reconciled=True, disposition="fail", findings_list=[_load_bearing()])
    before = result.to_dict()
    monkeypatch.setenv("AGY_RECONCILE_MOCK_CHECK", "0")
    monkeypatch.setenv("AGY_RECONCILE_MOCK_STRICT", "1")
    out = _station(tmp_path)._flag_mock_hidden_findings(result)
    assert out.to_dict() == before


@pytest.mark.not_slow
def test_unparseable_test_file_is_noop_no_raise(tmp_path, monkeypatch):
    """An unparseable TEST file means the real-call census is untrustworthy -> no-op (#81)."""
    monkeypatch.setenv("AGY_RECONCILE_MOCK_STRICT", "1")
    _write_tree(
        tmp_path,
        "from unittest.mock import patch\n\n"
        "@patch('pkg.cert._d1e_world_unit')\n"
        "def test_mocked(mock_world):\n"
        "    assert mock_world is not None\n",
    )
    (tmp_path / "tests" / "test_broken.py").write_text("def broken(:\n", encoding="utf-8")
    result = ReconciliationResult(reconciled=True, disposition="fail", findings_list=[_load_bearing()])
    before = result.to_dict()
    out = _station(tmp_path)._flag_mock_hidden_findings(result)
    assert out.to_dict() == before


@pytest.mark.not_slow
def test_empty_tree_is_noop_no_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_RECONCILE_MOCK_STRICT", "1")
    result = ReconciliationResult(reconciled=True, disposition="fail", findings_list=[_load_bearing()])
    out = _station(tmp_path)._flag_mock_hidden_findings(result)
    assert out.reconciled is True
    assert out.findings_list[0].mock_hides_implementation is None


@pytest.mark.not_slow
def test_unparseable_production_file_is_irrelevant(tmp_path):
    """A broken NON-test file must not disable the gate (only the test census matters)."""
    _write_tree(
        tmp_path,
        "from unittest.mock import patch\n\n"
        "@patch('pkg.cert._d1e_world_unit')\n"
        "def test_mocked(mock_world):\n"
        "    assert mock_world is not None\n",
    )
    (tmp_path / "pkg" / "broken_prod.py").write_text("def broken(:\n", encoding="utf-8")
    result = _run(_reply(), tmp_path)
    # default flag-only: still flagged despite the broken production file
    assert result.findings_list[0].mock_hides_implementation is True
    assert result.reconciled is True


@pytest.mark.not_slow
def test_symbol_resolved_from_location_when_finding_name_is_headline(tmp_path):
    """The flagged symbol is resolved from location (def at the line), not the prose name."""
    _write_tree(
        tmp_path,
        "from unittest.mock import patch\n\n"
        "@patch('pkg.cert._d1e_world_unit')\n"
        "def test_mocked(mock_world):\n"
        "    assert mock_world is not None\n",
    )
    result = _run(_reply(name="collect_d1e_cert goal headline", location="pkg/cert.py:1"), tmp_path)
    finding = result.findings_list[0]
    assert finding.mock_hides_implementation is True


# ------------------------------------------- Part B: fuzz-driven idiom coverage (#86)

@pytest.mark.not_slow
@pytest.mark.parametrize(
    "test_source",
    [
        # monkeypatch.setattr, string-target form
        "def test_m(monkeypatch):\n"
        "    monkeypatch.setattr('pkg.cert._d1e_world_unit', object())\n",
        # monkeypatch.setattr, (module, 'name', value) form
        "from pkg import cert\n"
        "def test_m(monkeypatch):\n"
        "    monkeypatch.setattr(cert, '_d1e_world_unit', object())\n",
        # pytest-mock mocker.patch
        "def test_m(mocker):\n"
        "    mocker.patch('pkg.cert._d1e_world_unit')\n",
        # builtin setattr with a Mock value
        "from unittest.mock import Mock\n"
        "from pkg import cert\n"
        "def test_m():\n"
        "    setattr(cert, '_d1e_world_unit', Mock())\n",
        # patch.multiple keyword form
        "from unittest.mock import patch, DEFAULT\n"
        "def test_m():\n"
        "    with patch.multiple('pkg.cert', _d1e_world_unit=DEFAULT):\n"
        "        pass\n",
        # mock.patch via aliased module import
        "import unittest.mock as mock\n"
        "@mock.patch('pkg.cert._d1e_world_unit')\n"
        "def test_m(m):\n"
        "    assert m\n",
    ],
)
def test_common_mock_idioms_are_detected(tmp_path, test_source):
    _write_tree(tmp_path, test_source)
    result = _run(_reply(), tmp_path)
    assert result.findings_list[0].mock_hides_implementation is True


@pytest.mark.not_slow
def test_class_method_location_resolves_to_method_not_class(tmp_path):
    """Innermost-def resolution: a method inside a class must resolve to the METHOD."""
    _write_tree(
        tmp_path,
        "from unittest.mock import patch\n"
        "from pkg import cert\n\n"
        "def test_m():\n"
        "    with patch.object(cert.World, '_d1e_world_unit'):\n"
        "        pass\n",
        prod_source=(
            "class World:\n"
            "    def _d1e_world_unit(self):\n"
            "        raise KeyError('first real call crashes')\n"
        ),
    )
    result = _run(_reply(location="pkg/cert.py:2"), tmp_path)
    assert result.findings_list[0].mock_hides_implementation is True


@pytest.mark.not_slow
def test_setattr_with_real_value_is_not_a_mock(tmp_path, monkeypatch):
    """builtin setattr(obj, 'sym', real_value) is NOT a mock -> symbol not flagged."""
    monkeypatch.setenv("AGY_RECONCILE_MOCK_STRICT", "1")
    _write_tree(
        tmp_path,
        "from pkg import cert\n"
        "def test_m():\n"
        "    setattr(cert, '_d1e_world_unit', lambda: 1)\n"
        "    cert._d1e_world_unit()\n",
    )
    result = _run(_reply(), tmp_path)
    finding = result.findings_list[0]
    assert finding.mock_hides_implementation is None
    assert result.reconciled is True


@pytest.mark.not_slow
def test_env_opt_out_tolerates_whitespace(tmp_path, monkeypatch):
    _write_tree(
        tmp_path,
        "from unittest.mock import patch\n\n"
        "@patch('pkg.cert._d1e_world_unit')\n"
        "def test_mocked(m):\n"
        "    assert m\n",
    )
    result = ReconciliationResult(reconciled=True, disposition="fail", findings_list=[_load_bearing()])
    before = result.to_dict()
    monkeypatch.setenv("AGY_RECONCILE_MOCK_CHECK", "  0  ")
    out = _station(tmp_path)._flag_mock_hidden_findings(result)
    assert out.to_dict() == before


@pytest.mark.not_slow
def test_honesty_note_survives_missing_verdict_key():
    """Defensive: _apply_reconcile_honesty_note must not KeyError on a partial dict."""
    _apply_reconcile_honesty_note({"witnesses_unmeasured": True}, {"note": "x"})
    _apply_reconcile_honesty_note(None, None)
    _apply_reconcile_honesty_note({"verdict": "reconciled", "witnesses_unmeasured": True}, None)

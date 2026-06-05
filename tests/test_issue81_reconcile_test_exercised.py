"""Issue #81: the reconcile classifier traces reachability from the cert/production
entrypoint ONLY, so a TEST MODULE flagged ``uncalled`` (pytest invokes it, not the
runtime) is false-positive ``exists_not_load_bearing`` → reconciled=False → hard-fail
under disposition=fail. The fix excuses such test-module findings (kept in the trace,
NOT counted as runtime dead wiring) and re-derives the verdict.

Scope is deliberately narrow and SAFE: only a finding whose own location is a pytest-
collected module (test_*.py / *_test.py / conftest.py) is excused — a strictly
decidable fact. The issue's OTHER case (a test-gated control/lesion arm in PRODUCTION
code) is NOT excused deterministically: distinguishing it from a genuine dead stub that
merely has a unit test is a semantic judgement a grep cannot make without over-
suppressing real dead wiring (the silent hollow-win this station exists to prevent).
That case is steered at the source by the mandate preamble instead. The over-
suppression tests below are direct regression guards from an adversarial fuzz pass.
"""
from __future__ import annotations

import pytest

from agy_orchestrator.workflows.reconcile import (
    MechanismFinding,
    ReconciliationResult,
    ReconciliationReview,
    RECONCILE_MANDATE_PREAMBLE,
    Witness,
    _is_test_path,
    _location_path,
)


def _station(tmp_path) -> ReconciliationReview:
    # The agent is never invoked by the excusal pass; any object suffices.
    return ReconciliationReview(agent=object(), goal="g", working_directory=str(tmp_path))


def _dead(name, location, sub_kind="uncalled", evidence="", measured=False):
    w = Witness(value=0.0, measured=measured)
    return MechanismFinding(
        name=name, classification="exists_not_load_bearing", sub_kind=sub_kind,
        location=location, witness=w, evidence=evidence,
    )


def _excuse(tmp_path, *findings):
    result = ReconciliationResult(
        reconciled=False, disposition="fail", findings_list=list(findings))
    return _station(tmp_path)._excuse_test_exercised_findings(result)


# --- path helpers ---------------------------------------------------------------

@pytest.mark.not_slow
@pytest.mark.parametrize("loc, expected", [
    ("tests/test_foo.py:12", True),
    ("tests/test_d0c_continuous_life_smoke.py:1", True),
    ("pkg/sub/something_test.py:3", True),
    ("tests/conftest.py", True),
    ("a/tests/helpers/test_x.py:9", True),
    ("tests/test_x.py::TestClass::test_method", True),  # pytest nodeid form
    ("test_x.py", True),                       # top-level pytest module
    ("TESTS/TEST_FOO.PY:1", True),             # case-insensitive
    # NOT pytest-collected -> must NOT read as a test (could be production code):
    ("test/scoring_engine.py:10", False),      # singular test/ dir, non-test filename
    ("tests/fixtures.py:1", False),            # under tests/ but not a collection glob
    ("tests/data/helper.py:2", False),
    ("scripts/cert_d1w_density.py:2634", False),
    ("agy_orchestrator/workflows/reconcile.py:10", False),
    ("src/contestant.py:5", False),            # 'test' substring must NOT match
    ("src/latest_model.py:5", False),
    (None, False),
])
def test_is_test_path(loc, expected):
    assert _is_test_path(loc) is expected


@pytest.mark.not_slow
def test_location_path_strips_line_and_col_and_handles_nonstring():
    assert _location_path("scripts/cert.py:2634") == "scripts/cert.py"
    assert _location_path("a\\b\\test_x.py:1:5") == "a/b/test_x.py"
    assert _location_path(None) is None


# --- the core fix: test modules are excused -------------------------------------

@pytest.mark.not_slow
def test_test_module_finding_is_excused_and_reconciles(tmp_path):
    # The issue's case: an S5 smoke TEST MODULE flagged uncalled-from-runtime.
    out = _excuse(tmp_path, _dead(
        "S5 default streaming smoke", "tests/test_d0c_continuous_life_smoke.py:1",
        evidence="No runtime call from the cert entrypoint traced; test-time only.",
    ))
    assert out.reconciled is True, "a test module is not runtime dead wiring (#81)"
    assert out.findings() == [], "excused finding drops out of the actionable set"
    assert len(out.excused_findings()) == 1
    assert out.excused_findings()[0].excused == "test_module"
    assert out.should_fail_run is False


@pytest.mark.not_slow
def test_mixed_test_module_and_real_dead_still_fails(tmp_path):
    # One excusable (test module) + one genuine runtime-dead -> run still fails on the
    # real one; the excused one is surfaced but not actionable.
    out = _excuse(
        tmp_path,
        _dead("S5 smoke", "tests/test_smoke.py:1"),
        _dead("scorer", "agy_orchestrator/scorer.py:10", sub_kind="stub_constant",
              evidence="returns hardcoded 0.0; never on the live path"),
    )
    assert out.reconciled is False
    assert [f.name for f in out.findings()] == ["scorer"]
    assert [f.name for f in out.excused_findings()] == ["S5 smoke"]


# --- OVER-SUPPRESSION regression guards (from the adversarial fuzz pass) ---------

@pytest.mark.not_slow
def test_descriptive_lesion_prose_does_not_excuse_real_dead_stub(tmp_path):
    # FUZZ HIGH: a genuine dead stub_constant whose EVIDENCE prose merely *describes*
    # it as a "lesion"/"control path" — and which (like any real symbol) has a unit
    # test referencing its name — must NOT be excused.
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_scorer.py").write_text(
        "from pkg.scorer import compute_anomaly_score\n"
        "def test_it():\n    assert compute_anomaly_score(0) == 0\n"
    )
    out = _excuse(tmp_path, _dead(
        "compute_anomaly_score", "pkg/scorer.py:10", sub_kind="stub_constant",
        evidence="This is a lesion in the pipeline: compute_anomaly_score returns a "
                 "hardcoded 0.0 and is never wired into the live control path.",
    ))
    assert out.reconciled is False, "descriptive prose must not excuse real dead wiring"
    assert out.excused_findings() == []
    assert len(out.findings()) == 1


@pytest.mark.not_slow
def test_production_file_under_singular_test_dir_not_excused(tmp_path):
    # FUZZ HIGH: a MEASURED-dead production module under a singular test/ directory
    # (NOT a pytest-collected file) must stay dead — it is production code.
    (tmp_path / "test").mkdir()
    out = _excuse(tmp_path, _dead(
        "scorer", "test/scoring_engine.py:10", sub_kind="stub_constant",
        evidence="returns hardcoded 0.0; never called", measured=True,
    ))
    assert out.reconciled is False, "a production file under test/ is not a test module"
    assert out.excused_findings() == []


@pytest.mark.not_slow
def test_control_arm_in_production_code_not_excused_deterministically(tmp_path):
    # The issue's case #1 (a control/lesion arm in PRODUCTION code) is intentionally
    # NOT excused by the deterministic pass — it's steered by the prompt instead. Here
    # we lock in that the pass does NOT silently bless it (no grep-based over-suppress).
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_perc.py").write_text(
        "def test_lesion_control():\n    assert collect(batch_over_prefix_lesion=True)\n"
    )
    out = _excuse(tmp_path, _dead(
        "S1 legacy batch-over-prefix control (lesion arm)",
        "scripts/cert_d1w_density.py:2634", sub_kind="bypassed_proxy",
        evidence="Not used on normal live path; requires explicit "
                 "batch_over_prefix_lesion=True.",
    ))
    assert out.reconciled is False
    assert out.excused_findings() == []
    assert len(out.findings()) == 1


# --- non-dead findings untouched + serialization --------------------------------

@pytest.mark.not_slow
def test_load_bearing_and_absent_findings_untouched(tmp_path):
    result = ReconciliationResult(
        reconciled=True, disposition="fail",
        findings_list=[
            MechanismFinding(name="S2", classification="load_bearing",
                             location="pkg/s2.py:1", witness=Witness(value=3.0)),
            MechanismFinding(name="S9", classification="absent", location=None),
        ],
    )
    out = _station(tmp_path)._excuse_test_exercised_findings(result)
    assert out.excused_findings() == []
    assert len(out.load_bearing_findings()) == 1
    assert len(out.absent_findings()) == 1


@pytest.mark.not_slow
def test_excused_serialized_in_to_dict(tmp_path):
    import json
    out = _excuse(tmp_path, _dead("S5", "tests/test_smoke.py:1"))
    d = out.to_dict()
    blob = json.dumps(d)  # must not raise
    assert json.loads(blob)["findings"][0]["excused"] == "test_module"
    assert d["reconciled"] is True


@pytest.mark.not_slow
def test_empty_findings_stay_hollow(tmp_path):
    # No findings -> hollow-empty rule still holds (an empty trace can't read green).
    out = _excuse(tmp_path)
    assert out.reconciled is False
    assert out.findings() == [] and out.excused_findings() == []


@pytest.mark.not_slow
def test_starved_verdict_not_resurrected_by_excusal(tmp_path):
    # Defense-in-depth: excusing the only (test-module) finding must NOT flip a
    # mark_starved()'d hollow verdict to reconciled=True.
    result = ReconciliationResult(
        reconciled=False, disposition="fail",
        findings_list=[_dead("S5 smoke", "tests/test_smoke.py:1")],
    ).mark_starved("stalled")
    out = _station(tmp_path)._excuse_test_exercised_findings(result)
    assert out.reconciled is False, "a starved/hollow verdict can't be resurrected green"
    assert out.is_starved is True


# --- prompt steers the agent for the production-control-arm case -----------------

@pytest.mark.not_slow
def test_prompt_mandate_mentions_test_established_load_bearing():
    p = RECONCILE_MANDATE_PREAMBLE.lower()
    assert "test module" in p
    assert "lesion" in p
    assert "load-bearing for its claim" in p

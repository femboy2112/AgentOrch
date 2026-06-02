"""Issues #54 / #55 / #60 — verifier resource bounds + observability.

#54  The orchestrator's OWN verifier subprocess inherited no BLAS/xdist thread
     pins (symmetric gap from #48), so a --test-cmd running its own pytest plus
     default numeric thread pools nested-oversubscribed the host. The verifier
     now injects the SAME {OPENBLAS,OMP,MKL,NUMEXPR}_NUM_THREADS=1 set into the
     child env, guarded by the same AGY_WORKER_RESOURCE_BOUND=0 opt-out.

#55  Surface the FINAL converged verifier's wall-clock + timeout margin, warn
     when the margin is < 20% of the budget, clamp any -n K to the host core
     count, and flag suspected oversubscription (final >> same-suite baseline).

#60  A non-timeout verifier failure used to log ONLY "Command failed with exit
     code N"; pytest writes "FAILED test::name" to STDOUT so the failing-test
     identity was persisted nowhere. The verifier now logs a bounded stdout
     tail, persists the full tails to a per-iteration artifact, and emits the
     parsed failing-test nodeids into the event stream.

Hermetic: no network, no real worker CLI, no systemd scope; commands are plain
shell builtins (`true`, `printf`, `exit`) run under tmp paths.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from agy_orchestrator.execution.verifier import (
    QualityVerifier,
    VerifierResult,
    _parse_failed_tests,
)
from harness.dispatch import _verifier_telemetry


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- #54 thread pins

def test_verifier_env_pins_thread_vars_by_default(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)
    v = QualityVerifier(test_commands=["true"])
    env = v._verifier_env()
    assert env is not None
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"


def test_verifier_env_respects_existing_value(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    v = QualityVerifier(test_commands=["true"])
    env = v._verifier_env()
    assert env["OMP_NUM_THREADS"] == "4"          # parent/operator value kept


def test_verifier_env_disabled_inherits_parent(monkeypatch):
    monkeypatch.setenv("AGY_WORKER_RESOURCE_BOUND", "0")
    v = QualityVerifier(test_commands=["true"])
    assert v._verifier_env() is None               # -> inherit parent env unchanged


def test_thread_pin_actually_in_child_env(tmp_path, monkeypatch):
    # End-to-end: the pinned var reaches the subprocess. printf its value to stdout.
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    v = QualityVerifier(test_commands=['printf "omp=%s" "$OMP_NUM_THREADS"; exit 1'])
    res = _run(v.verify(str(tmp_path)))
    assert res.ok is False
    assert "omp=1" in res.stdout_tail


# ----------------------------------------------------------------- #55b -n clamp

def test_clamp_xdist_clamps_above_cpu(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    v = QualityVerifier(test_commands=["x"])
    assert v._clamp_xdist("pytest -q -n 8") == "pytest -q -n 2"
    assert v._clamp_xdist("pytest -n8 -q") == "pytest -n2 -q"
    assert v._clamp_xdist("pytest --numprocesses=8") == "pytest --numprocesses=2"


def test_clamp_xdist_leaves_within_budget_and_auto(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    v = QualityVerifier(test_commands=["x"])
    assert v._clamp_xdist("pytest -n 2") == "pytest -n 2"      # under budget, kept
    assert v._clamp_xdist("pytest -n auto") == "pytest -n auto"  # auto untouched


def test_clamp_xdist_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("AGY_WORKER_RESOURCE_BOUND", "0")
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    v = QualityVerifier(test_commands=["x"])
    assert v._clamp_xdist("pytest -n 16") == "pytest -n 16"    # operator control


# ------------------------------------------------------------- #55 telemetry

def test_verifier_telemetry_records_duration_and_margin():
    final = VerifierResult(ok=True, duration_ms=1000)
    tel = _verifier_telemetry(final, timeout_s=10, baseline_result=None, run_id="r")
    assert tel["verifier_duration_ms"] == 1000
    assert tel["verifier_timeout_margin"] == 9000          # 10s - 1s
    assert tel["verifier_margin_low"] is False
    assert tel["verifier_oversubscribed"] is None          # no baseline to compare


def test_verifier_telemetry_low_margin_flag():
    final = VerifierResult(ok=True, duration_ms=9500)
    tel = _verifier_telemetry(final, timeout_s=10, baseline_result=None, run_id="r")
    assert tel["verifier_margin_low"] is True              # < 20% budget left


def test_verifier_telemetry_oversubscription_flag():
    baseline = VerifierResult(ok=True, duration_ms=1000)
    final = VerifierResult(ok=True, duration_ms=3000)      # 3x baseline
    tel = _verifier_telemetry(final, timeout_s=600,
                              baseline_result=baseline, run_id="r")
    assert tel["verifier_oversubscribed"] is True


def test_verifier_telemetry_not_oversubscribed_when_similar():
    baseline = VerifierResult(ok=True, duration_ms=1000)
    final = VerifierResult(ok=True, duration_ms=1100)
    tel = _verifier_telemetry(final, timeout_s=600,
                              baseline_result=baseline, run_id="r")
    assert tel["verifier_oversubscribed"] is False


def test_verifier_telemetry_empty_without_result():
    assert _verifier_telemetry(None, timeout_s=10,
                               baseline_result=None, run_id="r") == {}


def test_last_result_recorded(tmp_path):
    v = QualityVerifier(test_commands=["true"])
    assert v.last_result is None
    res = _run(v.verify(str(tmp_path)))
    assert v.last_result is res
    assert res.ok is True


# ----------------------------------------------------------- #60 output capture

def test_parse_failed_tests_extracts_nodeids():
    out = (
        "some noise\n"
        "FAILED tests/test_a.py::test_one - AssertionError\n"
        "FAILED tests/test_b.py::test_two\n"
        "ERROR tests/test_c.py::test_three\n"
        "=== short test summary info ===\n"
    )
    got = _parse_failed_tests(out)
    assert got == [
        "tests/test_a.py::test_one",
        "tests/test_b.py::test_two",
        "tests/test_c.py::test_three",
    ]


def test_parse_failed_tests_dedup_and_bounded():
    out = "\n".join(f"FAILED t.py::test_{i}" for i in range(100))
    out += "\nFAILED t.py::test_0"          # duplicate of the first
    got = _parse_failed_tests(out, limit=25)
    assert len(got) == 25
    assert got[0] == "t.py::test_0"


def test_failure_logs_stdout_tail(tmp_path, caplog):
    # A failing command whose FAILED summary lives on STDOUT must reach WARNING.
    cmd = ('printf "FAILED tests/test_x.py::test_boom\\n'
           '=== short test summary ===\\n"; exit 1')
    v = QualityVerifier(test_commands=[cmd])
    with caplog.at_level("WARNING"):
        res = _run(v.verify(str(tmp_path)))
    assert res.ok is False
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "captured stdout" in joined
    assert "tests/test_x.py::test_boom" in joined


def test_failure_persists_artifact(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cmd = 'printf "FAILED tests/test_x.py::test_boom\\n"; exit 1'
    v = QualityVerifier(test_commands=[cmd])
    v.run_dir = str(run_dir)
    v.step_index = 2
    _run(v.verify(str(tmp_path)))
    artifact = run_dir / "verify_step2_iter1.log"
    assert artifact.exists()
    body = artifact.read_text()
    assert "tests/test_x.py::test_boom" in body
    assert "=== STDOUT ===" in body
    # A second failure within the same step lands in a distinct iter artifact.
    _run(v.verify(str(tmp_path)))
    assert (run_dir / "verify_step2_iter2.log").exists()


def test_failure_emits_failed_tests_event(tmp_path):
    events = []
    cmd = 'printf "FAILED tests/test_x.py::test_boom\\n"; exit 1'
    v = QualityVerifier(test_commands=[cmd])
    v.event_callback = lambda e: events.append(e)
    v.step_index = 1
    _run(v.verify(str(tmp_path)))
    assert len(events) == 1
    detail = events[0]["data"]["detail"]
    assert detail["failed_tests"] == ["tests/test_x.py::test_boom"]
    assert detail["step"] == 1


def test_passing_run_emits_no_failure_event(tmp_path):
    events = []
    v = QualityVerifier(test_commands=["true"])
    v.event_callback = lambda e: events.append(e)
    v.run_dir = str(tmp_path)
    res = _run(v.verify(str(tmp_path)))
    assert res.ok is True
    assert events == []
    # No artifact written on success.
    assert not list(tmp_path.glob("verify_step*_iter*.log"))


def test_timeout_path_does_not_emit_failure_event(tmp_path):
    # A timeout is infra, not a code defect; #60 failure-reporting is for
    # non-timeout failures only (the timeout branch returns before _report_failure).
    events = []
    v = QualityVerifier(test_commands=["sleep 5"], timeout=0.2)
    v.event_callback = lambda e: events.append(e)
    res = _run(v.verify(str(tmp_path)))
    assert res.timeout is True
    assert events == []

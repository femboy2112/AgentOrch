"""Hermetic fuzz / property / edge-case tests for execution/verifier.py.

NEVER spawns a real worker CLI and NEVER hits the network. Subprocess behaviour
is exercised only with `sh`/`echo`/`sleep`/`exit`. Pure helpers are driven
directly. Every test must run in well under a second.

Covers: empty/None test_commands, the -n xdist clamp math (incl. the token-
boundary corruption regression), head/tail log persistence math (incl. the
max_chars==1 cap-defeat regression), failing-test parse on adversarial pytest
output, full-log-cap env parsing, infra/timeout/OOM classification, process-group
reap helpers, mem-cap argv when systemd is absent, and an unwritable run_dir.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from agy_orchestrator.execution import verifier as verifier_mod
from agy_orchestrator.execution.verifier import (
    QualityVerifier,
    VerifierResult,
    _full_log_cap,
    _kill_process_group,
    _parse_failed_tests,
    _safe_getpgid,
)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# empty / None test_commands
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_empty_commands_is_ok_noop():
    v = QualityVerifier(test_commands=[])
    r = _run(v.verify("/tmp"))
    assert r.ok is True
    assert r.returncode == 0
    assert bool(r) is True
    # _finalize must record it.
    assert v.last_result is r


@pytest.mark.not_slow
def test_none_commands_is_ok_noop():
    # `not None` is True, so the empty-path is taken without iterating None.
    v = QualityVerifier(test_commands=None)
    r = _run(v.verify("/tmp"))
    assert r.ok is True


# --------------------------------------------------------------------------- #
# -n xdist clamp math
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_clamp_xdist_clamps_above_cpu(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    v = QualityVerifier(test_commands=["x"])
    assert v._clamp_xdist("pytest -n 8") == "pytest -n 4"
    assert v._clamp_xdist("pytest -n8") == "pytest -n4"
    assert v._clamp_xdist("pytest --numprocesses=8") == "pytest --numprocesses=4"
    assert v._clamp_xdist("pytest --numprocesses 99") == "pytest --numprocesses 4"


@pytest.mark.not_slow
def test_clamp_xdist_leaves_at_or_below_cpu_untouched(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    v = QualityVerifier(test_commands=["x"])
    assert v._clamp_xdist("pytest -n 8") == "pytest -n 8"
    assert v._clamp_xdist("pytest -n 0") == "pytest -n 0"
    assert v._clamp_xdist("pytest -n 1") == "pytest -n 1"


@pytest.mark.not_slow
def test_clamp_xdist_leaves_auto_and_logical(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    v = QualityVerifier(test_commands=["x"])
    assert v._clamp_xdist("pytest -n auto") == "pytest -n auto"
    assert v._clamp_xdist("pytest -n logical") == "pytest -n logical"


@pytest.mark.not_slow
def test_clamp_xdist_does_not_corrupt_embedded_token(monkeypatch):
    """Regression: a test-file path or -k filter containing `-n<digits>` as a
    substring must NOT be mistaken for an xdist flag and rewritten."""
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    v = QualityVerifier(test_commands=["x"])
    assert v._clamp_xdist("pytest tests/foo-n99.py") == "pytest tests/foo-n99.py"
    assert v._clamp_xdist("pytest -k test-n99-x") == "pytest -k test-n99-x"
    assert v._clamp_xdist("pytest tests/test_issue99-n200.py") == (
        "pytest tests/test_issue99-n200.py"
    )


@pytest.mark.not_slow
def test_clamp_xdist_flag_at_string_start_and_multiple(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    v = QualityVerifier(test_commands=["x"])
    assert v._clamp_xdist("-n 99") == "-n 4"
    assert v._clamp_xdist("pytest -n 99 --numprocesses=99 more") == (
        "pytest -n 4 --numprocesses=4 more"
    )


@pytest.mark.not_slow
def test_clamp_xdist_huge_number_does_not_overflow(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    v = QualityVerifier(test_commands=["x"])
    out = v._clamp_xdist("pytest -n 99999999999999999999999999999")
    assert out == "pytest -n 2"


@pytest.mark.not_slow
def test_clamp_xdist_disabled_via_opt_out(monkeypatch):
    monkeypatch.setenv("AGY_WORKER_RESOURCE_BOUND", "0")
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    v = QualityVerifier(test_commands=["x"])
    assert v._clamp_xdist("pytest -n 99") == "pytest -n 99"


# --------------------------------------------------------------------------- #
# head/tail log persistence math
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_head_tail_passthrough_under_cap():
    assert QualityVerifier._head_tail("short", 1000) == "short"


@pytest.mark.not_slow
def test_head_tail_unbounded_for_nonpositive_cap():
    text = "x" * 5000
    assert QualityVerifier._head_tail(text, 0) == text
    assert QualityVerifier._head_tail(text, -10) == text


@pytest.mark.not_slow
@pytest.mark.parametrize("cap", [1, 2, 3, 5, 10, 100, 999])
def test_head_tail_never_exceeds_input_length(cap):
    """Property: a positive cap must never expand the text. The max_chars==1 case
    used to return text[-0:] (the WHOLE string) plus the elision marker, blowing
    past the input length and defeating the cap. The output must stay <= input."""
    text = "A" + ("x" * 4000) + "Z"
    out = QualityVerifier._head_tail(text, cap)
    assert len(out) <= len(text), f"cap={cap} produced longer-than-input output"


@pytest.mark.not_slow
def test_head_tail_cap_one_is_bounded_head_cut():
    text = "x" * 100000
    out = QualityVerifier._head_tail(text, 1)
    # Must not contain the whole input. A single-char head cut is acceptable.
    assert len(out) <= 1
    assert out == "x"


@pytest.mark.not_slow
def test_head_tail_keeps_both_ends_when_room():
    text = "HEADMARK" + ("x" * 5000) + "TAILMARK"
    out = QualityVerifier._head_tail(text, 400)
    assert "HEADMARK" in out and "TAILMARK" in out
    assert "elided" in out
    assert len(out) < len(text)


# --------------------------------------------------------------------------- #
# failing-test parse on adversarial pytest output
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_parse_failed_tests_empty_inputs():
    assert _parse_failed_tests("") == []
    assert _parse_failed_tests(None) == []


@pytest.mark.not_slow
def test_parse_failed_tests_dedup_and_order():
    out = (
        "FAILED tests/a.py::test_one - AssertionError\n"
        "ERROR tests/b.py::test_two\n"
        "FAILED tests/a.py::test_one - AssertionError\n"
    )
    assert _parse_failed_tests(out) == ["tests/a.py::test_one", "tests/b.py::test_two"]


@pytest.mark.not_slow
def test_parse_failed_tests_bounded():
    lines = "\n".join(f"FAILED t.py::test_{i}" for i in range(500))
    out = _parse_failed_tests(lines, limit=25)
    assert len(out) == 25


@pytest.mark.not_slow
def test_parse_failed_tests_only_anchored_at_line_start():
    # "FAILED" not at line start must NOT be parsed as a nodeid.
    out = "some prose mentioning FAILED tests/x.py::t inline\n"
    assert _parse_failed_tests(out) == []


@pytest.mark.not_slow
def test_parse_failed_tests_control_chars_and_unicode():
    out = (
        "FAILED tests/é.py::test_☃ - boom\n"
        "FAILED\ttests/tab.py::t\n"  # tab after keyword is whitespace -> matches
    )
    parsed = _parse_failed_tests(out)
    assert "tests/é.py::test_☃" in parsed
    assert "tests/tab.py::t" in parsed


# --------------------------------------------------------------------------- #
# _full_log_cap env parsing
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_full_log_cap_default(monkeypatch):
    monkeypatch.delenv("AGY_VERIFY_FULL_LOG_MAX", raising=False)
    assert _full_log_cap() == 1024 * 1024


@pytest.mark.not_slow
@pytest.mark.parametrize("val", ["garbage", "1.5", "  ", "nan"])
def test_full_log_cap_malformed_falls_back(monkeypatch, val):
    monkeypatch.setenv("AGY_VERIFY_FULL_LOG_MAX", val)
    assert _full_log_cap() == 1024 * 1024


@pytest.mark.not_slow
def test_full_log_cap_zero_and_negative(monkeypatch):
    monkeypatch.setenv("AGY_VERIFY_FULL_LOG_MAX", "0")
    assert _full_log_cap() == 0
    monkeypatch.setenv("AGY_VERIFY_FULL_LOG_MAX", "-5")
    assert _full_log_cap() == -5  # negative -> _head_tail treats as unbounded


# --------------------------------------------------------------------------- #
# process-group reap helpers
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_safe_getpgid_on_bogus_pid_returns_none():
    class _P:
        pid = -1
    assert _safe_getpgid(_P()) is None


@pytest.mark.not_slow
def test_kill_process_group_none_pgid_is_noop():
    # None pgid and no fallback: must not raise.
    _kill_process_group(None)


@pytest.mark.not_slow
def test_kill_process_group_already_gone_swallowed():
    # A pgid that surely does not exist -> ProcessLookupError -> swallowed.
    _kill_process_group(2_000_000_000)


@pytest.mark.not_slow
def test_kill_process_group_fallback_used_when_no_pgid():
    killed = {"n": 0}

    class _Proc:
        def kill(self):
            killed["n"] += 1

    _kill_process_group(None, fallback_process=_Proc())
    assert killed["n"] == 1


@pytest.mark.not_slow
def test_kill_process_group_fallback_kill_raises_swallowed():
    class _Proc:
        def kill(self):
            raise ProcessLookupError("gone")

    _kill_process_group(None, fallback_process=_Proc())  # must not raise


# --------------------------------------------------------------------------- #
# mem-cap argv
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_exec_argv_none_when_cap_inactive(monkeypatch):
    # No mem_max -> not active -> plain shell path (None argv).
    monkeypatch.delenv("AGY_VERIFIER_MEM_MAX", raising=False)
    v = QualityVerifier(test_commands=["x"])
    assert v._cap_active is False
    assert v._exec_argv("echo hi") is None


@pytest.mark.not_slow
def test_mem_cap_degrades_uncapped_when_systemd_absent(monkeypatch):
    # Force the systemd probe to report unavailable: cap must NOT activate, and
    # construction must not raise.
    monkeypatch.setattr(
        QualityVerifier, "_systemd_scope_available", staticmethod(lambda: False)
    )
    v = QualityVerifier(test_commands=["x"], mem_max="3G")
    assert v._cap_active is False
    assert v._exec_argv("echo hi") is None


@pytest.mark.not_slow
def test_mem_cap_argv_shape_when_active(monkeypatch):
    monkeypatch.setattr(
        QualityVerifier, "_systemd_scope_available", staticmethod(lambda: True)
    )
    v = QualityVerifier(test_commands=["x"], mem_max="3G")
    assert v._cap_active is True
    argv = v._exec_argv("echo hi")
    assert argv is not None
    assert argv[0] == "systemd-run"
    assert "MemoryMax=3G" in argv
    assert argv[-3:] == ["/bin/sh", "-c", "echo hi"]


# --------------------------------------------------------------------------- #
# end-to-end subprocess paths (hermetic: sh/echo/exit only)
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_success_path_rc0(monkeypatch):
    monkeypatch.setenv("AGY_WORKER_RESOURCE_BOUND", "0")
    v = QualityVerifier(test_commands=["true"], timeout=10)
    r = _run(v.verify("/tmp"))
    assert r.ok is True
    assert r.returncode == 0
    assert r.message == "All tests passed"


@pytest.mark.not_slow
def test_failure_path_sets_returncode_and_hash(monkeypatch):
    monkeypatch.setenv("AGY_WORKER_RESOURCE_BOUND", "0")
    v = QualityVerifier(
        test_commands=['sh -c "echo boom 1>&2; exit 3"'], timeout=10
    )
    r = _run(v.verify("/tmp"))
    assert r.ok is False
    assert r.returncode == 3
    assert r.timeout is False
    assert r.resource_exceeded is False
    assert "boom" in r.stderr_tail
    assert r.error_hash and len(r.error_hash) == 16


@pytest.mark.not_slow
def test_timeout_path(monkeypatch):
    monkeypatch.setenv("AGY_WORKER_RESOURCE_BOUND", "0")
    v = QualityVerifier(test_commands=["sleep 5"], timeout=0.3)
    r = _run(v.verify("/tmp"))
    assert r.ok is False
    assert r.timeout is True
    # returncode is set (process killed) or -1 sentinel, never crashes.
    assert isinstance(r.returncode, int)


@pytest.mark.not_slow
def test_infra_suspected_on_infra_stderr(monkeypatch):
    monkeypatch.setenv("AGY_WORKER_RESOURCE_BOUND", "0")
    # An infra-class marker in stderr on a genuine (non-timeout) failure.
    v = QualityVerifier(
        test_commands=['sh -c "echo No module named foo 1>&2; exit 1"'],
        timeout=10,
    )
    r = _run(v.verify("/tmp"))
    assert r.ok is False
    assert r.infra_suspected is True
    assert r.resource_exceeded is False


@pytest.mark.not_slow
def test_huge_stdout_is_tail_bounded(monkeypatch):
    monkeypatch.setenv("AGY_WORKER_RESOURCE_BOUND", "0")
    # ~200 KB of no-newline output; the result tail must be bounded to 2 KB.
    v = QualityVerifier(
        test_commands=['sh -c "head -c 200000 /dev/zero | tr \\"\\\\0\\" x; exit 1"'],
        timeout=10,
    )
    r = _run(v.verify("/tmp"))
    assert r.ok is False
    assert len(r.stdout_tail) <= 2000


@pytest.mark.not_slow
def test_run_dir_missing_is_swallowed(monkeypatch):
    monkeypatch.setenv("AGY_WORKER_RESOURCE_BOUND", "0")
    v = QualityVerifier(
        test_commands=['sh -c "echo FAILED t::x; exit 1"'], timeout=10
    )
    v.run_dir = "/tmp/agentorch_no_such_dir_xyz_123/sub"
    r = _run(v.verify("/tmp"))  # must NOT raise despite unwritable run_dir
    assert r.ok is False


@pytest.mark.not_slow
def test_run_dir_is_a_file_is_swallowed(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_WORKER_RESOURCE_BOUND", "0")
    f = tmp_path / "not_a_dir"
    f.write_text("x")
    v = QualityVerifier(
        test_commands=['sh -c "echo FAILED t::x; exit 1"'], timeout=10
    )
    v.run_dir = str(f)  # base/'...' -> NotADirectoryError, must be swallowed
    r = _run(v.verify("/tmp"))
    assert r.ok is False


@pytest.mark.not_slow
def test_event_callback_raising_is_swallowed(monkeypatch):
    monkeypatch.setenv("AGY_WORKER_RESOURCE_BOUND", "0")

    def boom(_):
        raise RuntimeError("cb")

    v = QualityVerifier(
        test_commands=['sh -c "echo FAILED a::b; exit 1"'], timeout=10
    )
    v.event_callback = boom
    r = _run(v.verify("/tmp"))  # callback raising must not corrupt the run
    assert r.ok is False


@pytest.mark.not_slow
def test_event_callback_receives_failed_tests(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_WORKER_RESOURCE_BOUND", "0")
    events = []
    v = QualityVerifier(
        test_commands=['sh -c "echo FAILED pkg/m.py::test_z; exit 1"'], timeout=10
    )
    v.run_dir = str(tmp_path)
    v.event_callback = events.append
    r = _run(v.verify("/tmp"))
    assert r.ok is False
    assert any(
        e.get("data", {}).get("event") == "verifier_failed" for e in events
    )
    detail = events[-1]["data"]["detail"]
    assert "pkg/m.py::test_z" in detail["failed_tests"]


# --------------------------------------------------------------------------- #
# VerifierResult unpacking / bool contract
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_verifier_result_unpacks_and_bools():
    r = VerifierResult(ok=True, message="m")
    ok, msg = r
    assert ok is True and msg == "m"
    assert bool(r) is True
    assert bool(VerifierResult(ok=False)) is False

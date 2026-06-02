"""Hermetic fuzz / property / edge-case tests for agy_orchestrator/core/agent.py.

NO real worker CLI is ever spawned and NO network is touched: every test either
drives a pure helper function directly or subclasses AgentInstance with a
build_command that returns a trivial 'sh'/'printf'/'bash'/'python3 -c' command.
Each test runs in well under a few seconds and needs no credentials.

Axes exercised: malformed/adversarial stderr+stdout (invalid UTF-8, NUL, ANSI,
huge strings, None/empty), classifier precedence (usage vs context vs transport
vs wedged vs infra), timeout/absolute-cap math (zero/negative/NaN/Inf), watchdog
trips (verbose / stall / transport-stall) and their NON-tripping, the streaming
drain on oversized/binary output, resource-bound env injection with adversarial
values, throwing observability callbacks, cancellation reap, and the
tolerant-decode hardening (a stray non-UTF-8 byte must not crash a success or
mask a fast-fail classification).
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from agy_orchestrator.core import agent as A
from agy_orchestrator.core.agent import (
    AgentInstance,
    ABSOLUTE_TIMEOUT_FACTOR,
    WATCHDOG_VERBOSE,
    WATCHDOG_STALLED,
    WATCHDOG_TRANSPORT_STALL,
    apply_worker_resource_bounds,
    is_context_overflow,
    is_transport_error,
    is_usage_wall,
    is_wedged_session,
    looks_like_infra,
)


# --------------------------------------------------------------------------- #
# Minimal hermetic agent harness
# --------------------------------------------------------------------------- #
class _Base(AgentInstance):
    """A concrete AgentInstance whose command is whatever ``_cmd`` is set to."""

    _cmd: list = ["true"]

    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return list(self._cmd)


def _make(cmd, **attrs):
    a = _Base(prompt="x")
    a._cmd = cmd
    a.max_retries = attrs.pop("max_retries", 1)
    a.timeout = attrs.pop("timeout", 10)
    for k, v in attrs.items():
        setattr(a, k, v)
    return a


# --------------------------------------------------------------------------- #
# 1. Classifier pure functions — None/empty/garbage must never raise, and the
#    documented precedence (overflow/usage/transport are mutually exclusive)
#    must hold even on adversarial overlapping input.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fn", [
    is_usage_wall, is_context_overflow, is_transport_error,
    is_wedged_session, looks_like_infra,
])
@pytest.mark.parametrize("val", [
    None, "", "   ", "\n\t", "\x00\x00", "ok normal line",
    "x" * 200000, "\x1b[31mANSI\x1b[0m", "unicode 中文 \U0001f600",
])
def test_classifiers_never_raise_on_adversarial_input(fn, val):
    out = fn(val)
    assert out is True or out is False


def test_usage_and_context_are_mutually_exclusive():
    # "context_length_exceeded" contains "exceeded"-ish text but must classify as
    # overflow only, NEVER as a quota wall (issue #47 regression).
    s = "Error: context_length_exceeded for this prompt"
    assert is_context_overflow(s) is True
    assert is_usage_wall(s) is False


def test_context_overflow_wins_even_when_usage_marker_also_present():
    # Adversarial: a line that literally contains BOTH a quota marker and a
    # context-overflow marker. Overflow must win (is_usage_wall short-circuits to
    # False when an overflow marker is present).
    s = "rate limit hit but also context window exceeded"
    assert is_context_overflow(s) is True
    assert is_usage_wall(s) is False
    assert is_transport_error(s) is False  # neither quota nor overflow -> not transport


def test_transport_yields_to_usage_and_overflow():
    # A transport marker co-located with a usage marker is NOT a transport error.
    assert is_transport_error("websocket dropped; usage limit reached") is False
    assert is_transport_error("connection reset 503") is True


def test_bare_exceeded_is_not_a_usage_wall():
    # The historical bug: a bare "exceeded" swallowed codex overflows as quota.
    assert is_usage_wall("the value exceeded the threshold") is False


# --------------------------------------------------------------------------- #
# 2. _to_int_token — NaN/Inf/bool/negative/garbage must degrade to None, never
#    raise and never return a bogus int.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("val,expected", [
    (None, None),
    (True, None),          # bool is not a token count
    (False, None),
    (-1, None),            # negative -> unavailable
    (0, 0),
    (5, 5),
    (5.9, 5),
    ("7", 7),
    ("-3", None),
    ("0xff", None),
    ("nan", None),
    (float("nan"), None),
    (float("inf"), None),
    (float("-inf"), None),
    (object(), None),
])
def test_to_int_token_degrades_to_none(val, expected):
    assert AgentInstance._to_int_token(val) == expected


# --------------------------------------------------------------------------- #
# 3. _absolute_cap math — tightest-of-set wins; zero/unset -> factor*timeout or
#    uncapped; negative values are ignored, not honored as a cap.
# --------------------------------------------------------------------------- #
def test_absolute_cap_defaults_to_factor_times_timeout():
    a = _make(["true"], timeout=100, absolute_timeout=0, worker_cmd_timeout=0)
    assert a._absolute_cap() == pytest.approx(100 * ABSOLUTE_TIMEOUT_FACTOR)


def test_absolute_cap_tightest_wins():
    a = _make(["true"], timeout=100, absolute_timeout=50, worker_cmd_timeout=20)
    assert a._absolute_cap() == 20  # min of the explicit caps


def test_absolute_cap_uncapped_when_everything_zero():
    a = _make(["true"], timeout=0, absolute_timeout=0, worker_cmd_timeout=0)
    assert a._absolute_cap() == 0.0


def test_absolute_cap_ignores_nonpositive_explicit_caps():
    # A 0 or negative explicit cap must not be treated as "0 second ceiling";
    # it falls back to the timeout-derived default.
    a = _make(["true"], timeout=10, absolute_timeout=-5, worker_cmd_timeout=0)
    assert a._absolute_cap() == pytest.approx(10 * ABSOLUTE_TIMEOUT_FACTOR)


# --------------------------------------------------------------------------- #
# 4. apply_worker_resource_bounds — adversarial env values must not raise,
#    markers with shell metacharacters must be quoted, operator values respected.
# --------------------------------------------------------------------------- #
def test_resource_bounds_disabled_returns_same_object(monkeypatch):
    monkeypatch.setenv("AGY_WORKER_RESOURCE_BOUND", "0")
    src = {"FOO": "bar"}
    assert apply_worker_resource_bounds(src) is src


def test_resource_bounds_garbage_xdist_falls_back(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.setenv("AGY_WORKER_PYTEST_XDIST", "not-a-number")
    monkeypatch.delenv("AGY_WORKER_PYTEST_MARKERS", raising=False)
    env = apply_worker_resource_bounds({})
    assert "-n 2" in env["PYTEST_ADDOPTS"]  # default K=2


def test_resource_bounds_nonpositive_xdist_serial(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.setenv("AGY_WORKER_PYTEST_XDIST", "0")
    monkeypatch.delenv("AGY_WORKER_PYTEST_MARKERS", raising=False)
    env = apply_worker_resource_bounds({})
    assert "-p no:xdist" in env["PYTEST_ADDOPTS"]


def test_resource_bounds_markers_are_shell_quoted(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.delenv("AGY_WORKER_PYTEST_XDIST", raising=False)
    monkeypatch.setenv("AGY_WORKER_PYTEST_MARKERS", "not slow; rm -rf /")
    env = apply_worker_resource_bounds({})
    # The injection must be quoted so a metachar-laden marker can't break out.
    assert "'not slow; rm -rf /'" in env["PYTEST_ADDOPTS"]


def test_resource_bounds_respects_existing_thread_pins(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    env = apply_worker_resource_bounds({"OMP_NUM_THREADS": "8"})
    assert env["OMP_NUM_THREADS"] == "8"          # operator value preserved
    assert env["OPENBLAS_NUM_THREADS"] == "1"     # absent one pinned


# --------------------------------------------------------------------------- #
# 5. TOLERANT DECODE (the fix): a worker that emits invalid UTF-8 / NUL bytes
#    must NOT crash the call. On success it returns; on a usage wall it still
#    fails over FAST (the classification runs after a clean decode).
# --------------------------------------------------------------------------- #
def test_invalid_utf8_on_successful_stdout_does_not_crash():
    a = _make(["printf", r"\xff\xfe\x80ok"], max_retries=1, timeout=10)
    out = asyncio.run(a.run_async())  # must NOT raise
    assert a.returncode == 0
    assert "ok" in out  # bad bytes replaced, real text preserved


def test_invalid_utf8_on_successful_stdout_stream_mode():
    a = _make(["printf", r"\xff\xfe\x80ok"], max_retries=1, timeout=10)
    a.event_callback = lambda e: None  # force stream mode
    out = asyncio.run(a.run_async())
    assert a.returncode == 0
    assert "ok" in out


def test_embedded_nul_in_stdout_survives():
    a = _make(["printf", r"a\x00b"], max_retries=1, timeout=10)
    out = asyncio.run(a.run_async())
    assert a.returncode == 0
    assert "a" in out and "b" in out


def test_usage_wall_with_bad_byte_fails_fast_not_retried():
    # The load-bearing case: a quota stderr carrying one invalid byte must STILL
    # be recognised as a usage wall and fail over immediately, not crash decode
    # and burn the full retry budget.
    a = _make(
        ["bash", "-c", r'printf "usage limit \xff reached" >&2; exit 1'],
        max_retries=3, timeout=10,
    )
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="usage wall"):
        asyncio.run(a.run_async())
    elapsed = time.monotonic() - started
    # No backoff loop ran (3 retries with 2**n backoff would be multiple seconds).
    assert elapsed < 2.0
    assert "usage" in a.stderr.lower()


def test_context_overflow_with_bad_byte_classifies():
    a = _make(
        ["bash", "-c", r'printf "context_length_exceeded \xff" >&2; exit 1'],
        max_retries=3, timeout=10,
    )
    with pytest.raises(RuntimeError, match="context overflow"):
        asyncio.run(a.run_async())


# --------------------------------------------------------------------------- #
# 6. Streaming drain — oversized single line and binary noise.
# --------------------------------------------------------------------------- #
def test_stream_drain_oversized_binary_line():
    # 100 KiB of mixed bytes including invalid UTF-8, no newline, on stdout.
    code = "import sys; sys.stdout.buffer.write(b'\\xff\\x00ABC'*20000)"
    a = _make(["python3", "-c", code], max_retries=1, timeout=20)
    a.event_callback = lambda e: None
    out = asyncio.run(a.run_async())  # must NOT raise (issue #30 + tolerant decode)
    assert a.returncode == 0
    assert "ABC" in out


# --------------------------------------------------------------------------- #
# 7. Watchdog — verbose trip, classic stall trip, and NON-trip when disabled.
# --------------------------------------------------------------------------- #
def test_watchdog_verbose_trips_on_runaway_output():
    code = "import sys; sys.stdout.write('x'*500000)"
    a = _make(["python3", "-c", code], max_retries=1, timeout=20)
    a.max_output_bytes = 1000
    a.stall_seconds = 0
    with pytest.raises(RuntimeError, match=r"\[watchdog:verbose\]"):
        asyncio.run(a.run_async())
    assert a._watchdog_reason == WATCHDOG_VERBOSE
    assert a.last_out_bytes >= 500000


def test_watchdog_classic_stall_trips_when_no_output_ever():
    # A worker that produces NOTHING and just sleeps trips the classic stall.
    a = _make(["sleep", "10"], max_retries=1, timeout=30)
    a.stall_seconds = 0.5
    a.max_output_bytes = 0
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=r"\[watchdog:stalled\]"):
        asyncio.run(a.run_async())
    elapsed = time.monotonic() - started
    assert a._watchdog_reason == WATCHDOG_STALLED
    assert elapsed < 6.0  # killed shortly after the 0.5s stall (2s poll cadence)


def test_watchdog_transport_stall_trips_on_retry_noise_only():
    cmd = ["bash", "-c",
           'for i in $(seq 1 30); do echo "connection reset, retrying" >&2; sleep 0.2; done']
    a = _make(cmd, max_retries=1, timeout=30)
    a.stall_seconds = 1.0
    a.max_output_bytes = 0
    a.event_callback = lambda e: None
    with pytest.raises(RuntimeError, match=r"\[watchdog:transport_stall\]"):
        asyncio.run(a.run_async())
    assert a._watchdog_reason == WATCHDOG_TRANSPORT_STALL


def test_watchdog_does_not_trip_on_legitimate_bursty_progress():
    # Emits real (non-noise) stderr work every 0.2s for ~1.4s with a 1s stall
    # budget — must NOT trip because real progress keeps advancing the clock.
    cmd = ["bash", "-c", "for i in $(seq 1 7); do echo applying_patch >&2; sleep 0.2; done"]
    a = _make(cmd, max_retries=1, timeout=30)
    a.stall_seconds = 1.0
    a.max_output_bytes = 0
    a.event_callback = lambda e: None
    asyncio.run(a.run_async())  # must NOT raise
    assert a.returncode == 0
    assert a._watchdog_reason is None


# --------------------------------------------------------------------------- #
# 8. Subprocess pathologies — instant exit nonzero, no stderr; missing binary;
#    empty command — all must fail bounded (raise RuntimeError), never hang/leak.
# --------------------------------------------------------------------------- #
def test_nonzero_exit_no_stderr_raises_bounded():
    a = _make(["false"], max_retries=1, timeout=10)
    with pytest.raises(RuntimeError, match="failed after"):
        asyncio.run(a.run_async())
    assert a.returncode != 0


def test_missing_binary_fails_bounded():
    a = _make(["this_binary_does_not_exist_zzz999"], max_retries=2, timeout=10)
    started = time.monotonic()
    with pytest.raises(RuntimeError):
        asyncio.run(a.run_async())
    assert time.monotonic() - started < 10


def test_empty_command_fails_bounded():
    a = _make([], max_retries=1, timeout=5)
    with pytest.raises(RuntimeError):
        asyncio.run(a.run_async())


# --------------------------------------------------------------------------- #
# 9. Timeout / liveness — a flat hang is killed fast; the tree is reaped.
# --------------------------------------------------------------------------- #
def test_flat_hang_times_out_fast():
    a = _make(["sleep", "30"], max_retries=1, timeout=0.5)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        asyncio.run(a.run_async())
    assert time.monotonic() - started < 8
    assert "timed out" in a.stderr


def test_zero_timeout_means_no_ceiling_for_quick_command():
    a = _make(["true"], max_retries=1, timeout=0)
    asyncio.run(a.run_async())
    assert a.returncode == 0


# --------------------------------------------------------------------------- #
# 10. Observability isolation — a throwing event_callback must never break the
#     run or change the result.
# --------------------------------------------------------------------------- #
def test_throwing_event_callback_does_not_break_run():
    def boom(_e):
        raise ValueError("observability blew up")

    a = _make(["printf", "hello"], max_retries=1, timeout=10)
    a.event_callback = boom
    out = asyncio.run(a.run_async())
    assert a.returncode == 0
    assert out == "hello"


# --------------------------------------------------------------------------- #
# 11. Cancellation mid-flight (a losing ToT/swarm branch) reaps the whole tree.
# --------------------------------------------------------------------------- #
def test_cancellation_reaps_grandchild(tmp_path):
    pidfile = tmp_path / "g.pid"
    cmd = ["bash", "-c", f"sleep 30 & echo $! > {pidfile}; wait"]

    async def driver():
        a = _make(cmd, max_retries=1, timeout=30)
        a.event_callback = lambda e: None
        task = asyncio.ensure_future(a.run_async())
        # Wait until the grandchild has recorded its pid.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not pidfile.exists():
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(driver())

    assert pidfile.exists()
    gpid = int(pidfile.read_text().strip())

    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and alive(gpid):
        time.sleep(0.05)
    assert not alive(gpid), "grandchild survived cancellation — tree not reaped"


# --------------------------------------------------------------------------- #
# 12. _post_exit_drain_grace — garbage env never raises (returns a float).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("val", ["5", "0", "abc", "", "2.5", "-1", "nan", "inf"])
def test_drain_grace_never_raises(monkeypatch, val):
    monkeypatch.setenv("AGY_WORKER_DRAIN_GRACE", val)
    out = A._post_exit_drain_grace()
    assert isinstance(out, float)


# --------------------------------------------------------------------------- #
# 13. _worker_name fallback for an unrecognised subclass name.
# --------------------------------------------------------------------------- #
def test_worker_name_defaults_to_agy_for_unknown_class():
    a = _make(["true"])
    assert a._worker_name() == "agy"  # _Base has no known worker substring

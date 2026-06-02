"""Regression tests for confirmed defects in agy_orchestrator/core/agent.py.

HERMETIC: no real worker CLI is spawned and no network is touched. Every test
subclasses AgentInstance with a build_command that returns a trivial
'sh'/'printf' command (only sh/printf/exit are used), or drives a pure helper
directly. Each test runs in well under a second.

Covered defects:
  * agent-core-1 — strict UTF-8 decode of worker stdout/stderr crashed the call
    on any non-UTF-8 byte. Two harms, one regression test each:
      (a) a returncode==0 SUCCESS carrying a stray byte became a generic
          exception, retried to exhaustion, then a hard RuntimeError;
      (b) a quota-wall stderr carrying a stray byte crashed BEFORE the
          is_usage_wall fast-fail classification, defeating instant fallover.
"""
from __future__ import annotations

import asyncio
import time

from agy_orchestrator.core.agent import AgentInstance


class _Base(AgentInstance):
    """Concrete AgentInstance whose command is whatever ``_cmd`` is set to."""

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
    a.max_retries = attrs.pop("max_retries", 2)
    a.timeout = attrs.pop("timeout", 10)
    for k, v in attrs.items():
        setattr(a, k, v)
    return a


# --------------------------------------------------------------------------- #
# agent-core-1 (a): a SUCCESS (returncode 0) that emits a stray non-UTF-8 byte
# on stdout must NOT crash. It must decode tolerantly and return cleanly.
# --------------------------------------------------------------------------- #
def test_success_with_invalid_utf8_stdout_does_not_crash():
    # printf emits "result-ok" + a raw 0xFF byte (\377 octal) + newline, exit 0.
    a = _make(["sh", "-c", r'printf "result-ok\377\n"; exit 0'])
    out = asyncio.run(a.run_async())
    assert a.returncode == 0
    # The replacement char (U+FFFD) stands in for the bad byte; payload survives.
    assert "result-ok" in out
    # No UnicodeDecodeError leaked through as stderr / exception.


# --------------------------------------------------------------------------- #
# agent-core-1 (b): a quota-wall stderr carrying a stray non-UTF-8 byte must
# still reach the is_usage_wall fast-fail branch — raising "usage wall: ..." on
# the FIRST attempt, instantly, rather than burning the whole retry budget.
# --------------------------------------------------------------------------- #
def test_usage_wall_with_invalid_utf8_stderr_fast_fails():
    # stderr carries a recognizable quota-wall phrase + a raw 0xFF byte, exit 1.
    a = _make(
        ["sh", "-c", r'printf "usage limit \377 reached" >&2; exit 1'],
        max_retries=3,
    )
    t0 = time.monotonic()
    err = None
    try:
        asyncio.run(a.run_async())
    except RuntimeError as e:  # noqa: PERF203
        err = e
    elapsed = time.monotonic() - t0

    assert err is not None, "a quota wall must raise"
    # Classified as a usage wall (fast-fail), NOT a generic decode crash.
    assert "usage wall" in str(err).lower()
    # Fast-fail means no retry/backoff loop: comfortably sub-second.
    assert elapsed < 1.0, f"expected instant fast-fail, took {elapsed:.2f}s"


# --------------------------------------------------------------------------- #
# Tolerant decode must not corrupt the success payload on the non-streaming
# path: a plain (all-ASCII) success returns its stdout unchanged.
# --------------------------------------------------------------------------- #
def test_plain_ascii_success_unaffected():
    a = _make(["sh", "-c", r'printf "hello world\n"; exit 0'])
    out = asyncio.run(a.run_async())
    assert a.returncode == 0
    assert "hello world" in out

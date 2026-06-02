"""Issue #62: a wedged agent session (codex's stdin-closed exec_command hang) must
FAST-FAIL to fallback instead of burning every retry/stall cycle on the same
provider, and must be classified distinctly (not a usage wall, not a code failure).
"""
from __future__ import annotations

import asyncio
import time

import pytest

from agy_orchestrator.core.agent import AgentInstance, is_wedged_session
from agy_orchestrator.core.agents.fallback_agent import _reason_category


# --------------------------------------------------------------------------- #
# Marker detection
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
@pytest.mark.parametrize("text", [
    "ERROR codex_core::tools::router: error=write_stdin failed: stdin is closed "
    "for this session; rerun exec_command with tty=true to keep stdin open",
    "stdin is closed for this session",
    "rerun exec_command with tty=true",
    "STDIN IS CLOSED FOR THIS SESSION",  # case-insensitive
])
def test_is_wedged_session_detects_marker(text):
    assert is_wedged_session(text) is True


@pytest.mark.not_slow
@pytest.mark.parametrize("text", [
    "",
    "normal codex exec progress line",
    "connection reset, retrying",            # transport noise, not a wedge
    "ModuleNotFoundError: No module named x",  # infra, not a wedge
])
def test_is_wedged_session_ignores_benign(text):
    assert is_wedged_session(text) is False


# --------------------------------------------------------------------------- #
# Classification precedence
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_reason_category_wedged():
    assert _reason_category(looked_like_usage=False, watchdog_reason=None,
                            wedged=True) == "wedged"


@pytest.mark.not_slow
def test_wedged_beats_stalled_and_usage():
    # A wedged session ends in a -9 stall-kill (watchdog_reason='stalled') and its
    # accumulated stderr can carry other noise; wedged must still win so telemetry
    # never reads it as exhaustion or a plain stall.
    assert _reason_category(looked_like_usage=True, watchdog_reason="stalled",
                            wedged=True) == "wedged"


@pytest.mark.not_slow
def test_context_overflow_still_beats_wedged():
    # Overflow is checked first by design (a smaller next step can still use the
    # same provider); wedged must not override it.
    assert _reason_category(looked_like_usage=False, watchdog_reason=None,
                            context_overflow=True, wedged=True) == "context_overflow"


# --------------------------------------------------------------------------- #
# Fast-fail: a wedged stderr must NOT consume all the retries
# --------------------------------------------------------------------------- #
class _WedgedAgent(AgentInstance):
    """A worker whose call emits the codex stdin-closed marker then exits non-zero,
    standing in for a wedged codex session."""

    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return [
            "sh", "-c",
            "echo 'write_stdin failed: stdin is closed for this session; "
            "rerun exec_command with tty=true' >&2; exit 1",
        ]


@pytest.mark.not_slow
def test_wedged_session_fast_fails_without_burning_retries():
    agent = _WedgedAgent(prompt="x")
    agent.max_retries = 3   # would normally retry 3x with backoff

    started = time.monotonic()
    # The exception message is labelled "wedged session" so the fallback layer /
    # telemetry classify it distinctly (matched here); agent.stderr keeps the raw
    # worker output that tripped the detector.
    with pytest.raises(RuntimeError, match="wedged session"):
        asyncio.run(agent.run_async())
    elapsed = time.monotonic() - started

    # If it had retried all 3 attempts it would have slept through two backoffs
    # (min 2^1*0.5 + 2^2*0.5 = ~3s of guaranteed sleep). Fast-fail means ~0 sleep.
    assert elapsed < 2.0
    assert is_wedged_session(agent.stderr)

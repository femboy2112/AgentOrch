"""Issue #47 — codex ``context_length_exceeded`` must NOT read as a usage/quota wall.

A context-window overflow (codex's own remote compaction failed on an oversized
prompt) is a different condition from a quota wall: the same provider can serve the
next, smaller step. The bare ``"exceeded"`` marker used to swallow it. These tests
pin the split classifier + the distinct fallback ``reason_category``.
"""

import asyncio
from typing import List, Optional

from agy_orchestrator.core.agent import (
    AgentInstance,
    is_context_overflow,
    is_usage_wall,
)
from agy_orchestrator.core.agents.fallback_agent import (
    _reason_category,
    make_fallback_agent,
)

# The exact codex stderr from the issue's repro (run 20260601-224922-837).
_CODEX_OVERFLOW = (
    "codex_core::compact_remote: remote compaction failed\n"
    "  last_api_response_total_tokens = 112553\n"
    "  model_context_window_tokens   = Some(121600)\n"
    '  compact_error = {"message": "Your input exceeds the context window of this '
    'model. Please adjust your input and try again.", "code": "context_length_exceeded"}'
)


def _run(coro):
    return asyncio.run(coro)


def test_context_overflow_is_not_a_usage_wall():
    assert is_context_overflow(_CODEX_OVERFLOW) is True
    # The crux of #47: the overflow must NOT be classified as a quota wall.
    assert is_usage_wall(_CODEX_OVERFLOW) is False


def test_context_length_exceeded_substring_alone():
    assert is_context_overflow('"code": "context_length_exceeded"') is True
    assert is_usage_wall('"code": "context_length_exceeded"') is False


def test_bare_exceeded_no_longer_trips_usage_wall():
    # Dropping bare "exceeded" from USAGE_MARKERS: an unqualified "exceeded" must
    # not, on its own, be read as a quota wall anymore.
    assert is_usage_wall("some value exceeded the threshold") is False


def test_real_quota_walls_still_classified():
    for s in (
        "429 rate limit exceeded",
        "quota exceeded for this org",
        "usage limit reached",
        "insufficient_quota",
        "too many requests",
        "plan limit hit",
    ):
        assert is_usage_wall(s) is True, s
        assert is_context_overflow(s) is False, s


def test_reason_category_overflow_takes_precedence():
    # Even if both flags were somehow set, overflow wins (it is checked first).
    assert _reason_category(looked_like_usage=True, watchdog_reason=None,
                            context_overflow=True) == "context_overflow"
    assert _reason_category(looked_like_usage=True, watchdog_reason=None) == "usage"
    assert _reason_category(looked_like_usage=False, watchdog_reason=None) == "error"


class _FallbackOverflowFail(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input: Optional[str] = None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        self.stderr = _CODEX_OVERFLOW
        raise RuntimeError("context overflow: remote compaction failed")


class _FallbackOk(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input: Optional[str] = None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        self.returncode = 0
        self.stdout = "ok"
        return "ok"


def _orchestration_rows(events: List[dict]) -> List[dict]:
    out: List[dict] = []
    for evt in events:
        data = evt.get("data", {})
        if evt.get("kind") != "lifecycle" or data.get("event") != "orchestration_transition":
            continue
        orch = data.get("orchestration", {})
        if isinstance(orch, dict):
            out.append(orch)
    return out


def test_fallback_emits_context_overflow_reason_category():
    events: List[dict] = []
    Agent = make_fallback_agent([_FallbackOverflowFail, _FallbackOk], cycles=1)
    agent = Agent(prompt="task")
    agent.event_callback = events.append

    out = _run(agent.run_async())
    assert out == "ok"
    rows = [r for r in _orchestration_rows(events) if r.get("phase") == "fallback"]
    assert rows, "expected a fallback orchestration event"
    # The overflow must be its own category, never "usage".
    assert rows[0]["reason_category"] == "context_overflow"

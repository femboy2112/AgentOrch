"""Single-run speed wins 2-5 (see docs/single-run-speed-audit.md).

Each win is quality-safe; these tests pin the behaviour that makes it so.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import List, Optional

import pytest

from agy_orchestrator.core.agent import AgentInstance


# --------------------------------------------------------------------------- #
# Win 2 — cache_read_ratio surfaced per run
# --------------------------------------------------------------------------- #
def test_cache_read_ratio_basic_and_none_denominator():
    from harness.dispatch import _cache_read_ratio

    assert _cache_read_ratio(60, 40) == 0.6          # 60 / (60+40)
    assert _cache_read_ratio(0, 100) == 0.0
    # Unknown denominator (codex reports total only -> input None) must be None,
    # NOT a misleading 0% hit rate.
    assert _cache_read_ratio(None, 100) is None
    assert _cache_read_ratio(50, None) is None
    assert _cache_read_ratio(0, 0) is None


def test_summary_includes_cache_read_ratio(tmp_path):
    from harness.dispatch import _summarize_token_usage

    events = tmp_path / "events.jsonl"
    rows = [
        {"kind": "usage", "worker": "codex", "model": "m", "data": {
            "usage_kind": "call", "token_source": "cli",
            "input_tokens": 40, "output_tokens": 10,
            "cache_read_tokens": 60, "total_tokens": 110}},
    ]
    events.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    summ = _summarize_token_usage(events)
    assert summ["per_worker"]["codex"]["cache_read_ratio"] == 0.6
    assert summ["grand_total"]["cache_read_ratio"] == 0.6


# --------------------------------------------------------------------------- #
# Win 3 — watchdog poll decoupled from completion detection (no streaming tail)
# --------------------------------------------------------------------------- #
class _SleepThenPrintAgent(AgentInstance):
    """Real subprocess: sleep, print, exit — to exercise the genuine watchdog path."""

    @classmethod
    async def get_available_models(cls) -> List[str]:
        return ["m"]

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        return 100.0

    def build_command(self, piped_input: Optional[str] = None) -> List[str]:
        return ["sh", "-c", "sleep 0.3; printf done"]


def test_watchdog_armed_call_has_no_2s_tail():
    """An armed watchdog must not hold a finished call for a full 2s poll slice."""
    async def _run() -> float:
        a = _SleepThenPrintAgent(prompt="x", model="m")
        a.stall_seconds = 300.0        # arm it, exactly as calibration does by default
        a.max_output_bytes = 200_000
        t0 = time.monotonic()
        out = await a.run_async()
        assert "done" in out
        return time.monotonic() - t0

    wall = asyncio.run(_run())
    # Child sleeps 0.3s; before the fix the watchdog's bare sleep(2.0) added up to
    # ~2s of tail. Assert the call returns promptly (generous bound for CI jitter).
    assert wall < 1.0, f"watchdog tail not removed: {wall:.2f}s"


def test_watchdog_still_trips_stall():
    """Decoupling completion detection must NOT disable the stall kill."""
    class _HangAgent(_SleepThenPrintAgent):
        def build_command(self, piped_input: Optional[str] = None) -> List[str]:
            return ["sh", "-c", "sleep 30"]  # emits nothing, should be stall-killed

    async def _run():
        a = _HangAgent(prompt="x", model="m")
        a.stall_seconds = 2.0          # trip quickly
        a.max_output_bytes = 0
        a.timeout = 30
        a.max_retries = 1              # one attempt -> deterministic timing
        t0 = time.monotonic()
        reason = None
        try:
            await a.run_async()
        except RuntimeError:
            reason = a._watchdog_reason  # a watchdog kill raises after the attempt
        return time.monotonic() - t0, reason

    wall, reason = asyncio.run(_run())
    assert wall < 8.0                   # killed near the 2s stall, not at 30s
    assert reason == "stalled"


# --------------------------------------------------------------------------- #
# Win 4 — preamble-stripped critic requirement
# --------------------------------------------------------------------------- #
def test_build_prompt_can_omit_preamble():
    from harness.dispatch import _build_prompt, WORKER_PREAMBLE

    full = _build_prompt("do the thing", "ctx", "SPEC BODY")
    critic = _build_prompt("do the thing", "ctx", "SPEC BODY", include_preamble=False)
    assert WORKER_PREAMBLE in full
    assert WORKER_PREAMBLE not in critic
    # The actual requirement content is preserved in both.
    assert "do the thing" in critic
    assert "SPEC BODY" in critic
    assert "ctx" in critic


class _CaptureAgent(AgentInstance):
    """Records the prompt it was last asked to run; returns a canned reply."""

    last_prompt = ""
    reply = "APPROVED"

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        type(self).last_prompt = self.prompt
        return type(self).reply

    @classmethod
    async def get_available_models(cls):
        return ["m"]

    @classmethod
    async def get_model_usage(cls, model: str):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]


def test_adversarial_critic_uses_requirement_not_preamble():
    from agy_orchestrator.workflows.adversarial import AdversarialReview

    class _Gen(_CaptureAgent):
        reply = "the generated output"

    class _Critic(_CaptureAgent):
        reply = "APPROVED"

    gen, critic = _Gen(prompt=""), _Critic(prompt="")
    adv = AdversarialReview(
        gen, critic, verifier=None, max_iterations=1,
        critic_requirement="GOAL ONLY no preamble",
    )
    asyncio.run(adv.execute("FULL PROMPT WITH PREAMBLE"))
    # The critic judged against the stripped requirement, not the generator's prompt.
    assert "GOAL ONLY no preamble" in _Critic.last_prompt
    assert "FULL PROMPT WITH PREAMBLE" not in _Critic.last_prompt
    # The generator still received the full prompt.
    assert _Gen.last_prompt == "FULL PROMPT WITH PREAMBLE"


def test_adversarial_critic_requirement_defaults_to_initial_prompt():
    from agy_orchestrator.workflows.adversarial import AdversarialReview

    class _Gen(_CaptureAgent):
        reply = "out"

    class _Critic(_CaptureAgent):
        reply = "APPROVED"

    adv = AdversarialReview(_Gen(prompt=""), _Critic(prompt=""), verifier=None, max_iterations=1)
    asyncio.run(adv.execute("THE WHOLE REQUIREMENT"))
    assert "THE WHOLE REQUIREMENT" in _Critic.last_prompt


# --------------------------------------------------------------------------- #
# Win 5 — ToT judge grounded in the requirement
# --------------------------------------------------------------------------- #
def test_build_judge_prompt_with_requirement():
    from agy_orchestrator.workflows.tree_of_thought import build_judge_prompt

    blind = build_judge_prompt("candidate code", noun="solution")
    grounded = build_judge_prompt("candidate code", noun="solution",
                                  requirement="must expose word_count(s)")
    assert "must expose word_count(s)" not in blind
    assert "must expose word_count(s)" in grounded
    assert "candidate code" in grounded


def test_tot_judge_receives_requirement():
    from agy_orchestrator.workflows.tree_of_thought import TreeOfThought

    seen = []

    class _Branch(_CaptureAgent):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
        async def run_async(self, piped_input=None):
            return f"branch-output-{id(self) % 100}"

    class _Judge(_CaptureAgent):
        async def run_async(self, piped_input=None):
            seen.append(self.prompt)
            return "8"

    branches = [_Branch(prompt=""), _Branch(prompt="")]
    tot = TreeOfThought(branches, _Judge(prompt=""), selector="judge",
                        requirement="THE STEP REQUIREMENT")
    asyncio.run(tot.execute())
    assert seen, "judge was never invoked"
    assert all("THE STEP REQUIREMENT" in p for p in seen)

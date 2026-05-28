"""Offline tests for MasterWorkflow session compaction (no real LLM/CLI calls)."""
from __future__ import annotations

import asyncio
import unittest

from agy_orchestrator.workflows import master as master_mod
from agy_orchestrator.workflows.master import MasterWorkflow


class FakeAgent:
    """Deterministic stand-in for an agent CLI: no subprocess, canned outputs."""

    def __init__(self, prompt: str = "", model=None, effort=None, session_id=None, **kw):
        self.prompt = prompt
        self.session_id = session_id

    async def run_async(self, piped_input=None) -> str:
        if "Lead Architect" in self.prompt:  # planner
            return '["step one", "step two", "step three", "step four"]'
        if "Condense the following project progress" in self.prompt:  # compactor
            return "DIGEST-BODY"
        # Any resumed/fresh call establishes a session id (mimics claude --output-format json).
        if self.session_id is None:
            self.session_id = "sess"
        return "bullet summary output"


class _StubToT:
    def __init__(self, branches, evaluator):
        pass

    async def execute(self) -> str:
        return "tot-best"


class _StubAdv:
    def __init__(self, generator_instance=None, critic_instance=None, verifier=None,
                 max_iterations=1, working_directory="."):
        pass

    async def execute(self, prompt) -> str:
        return "adv-final"


class CompactionLogicTests(unittest.TestCase):
    def test_should_compact_on_interval_and_size(self):
        wf = MasterWorkflow(model="m", effort="low", compaction_interval=3, max_context_chars=1000)
        self.assertFalse(wf._should_compact(2, "short"))
        self.assertTrue(wf._should_compact(3, "short"))          # interval hit
        self.assertTrue(wf._should_compact(1, "x" * 1000))       # size hit
        # disabled when both thresholds are 0
        wf0 = MasterWorkflow(model="m", effort="low", compaction_interval=0, max_context_chars=0)
        self.assertFalse(wf0._should_compact(99, "x" * 99999))

    def test_compact_context_returns_bounded_digest(self):
        wf = MasterWorkflow(model="m", effort="low", agent_class=FakeAgent)
        out = asyncio.run(wf._compact_context("the goal", "X" * 50000))
        self.assertIn("Original Goal: the goal", out)
        self.assertIn("(compacted)", out)
        self.assertIn("DIGEST-BODY", out)

    def test_compact_context_falls_back_on_failure(self):
        class BoomAgent(FakeAgent):
            async def run_async(self, piped_input=None):
                raise RuntimeError("provider down")

        wf = MasterWorkflow(model="m", effort="low", agent_class=BoomAgent, max_context_chars=100)
        ctx = "Original Goal: g\n" + ("tail" * 200)
        out = asyncio.run(wf._compact_context("g", ctx))
        self.assertIn("Original Goal: g", out)
        self.assertLessEqual(len(out), 100 + 200)  # truncated to recent tail + header


class CompactionLoopTests(unittest.TestCase):
    def setUp(self):
        self._tot, self._adv = master_mod.TreeOfThought, master_mod.AdversarialReview
        master_mod.TreeOfThought = _StubToT
        master_mod.AdversarialReview = _StubAdv

    def tearDown(self):
        master_mod.TreeOfThought = self._tot
        master_mod.AdversarialReview = self._adv

    def test_compaction_runs_over_a_long_chained_run(self):
        # 4 steps, compact every 2 -> compaction fires; final context is the compacted digest.
        wf = MasterWorkflow(
            model="m", effort="low", branches=1, max_iterations=1,
            agent_class=FakeAgent, compaction_interval=2, max_context_chars=0,
        )
        result = asyncio.run(wf.execute("build a thing"))
        self.assertIn("(compacted)", result)        # compaction occurred
        self.assertIn("DIGEST-BODY", result)
        self.assertLess(len(result), 6000)           # context stayed bounded

    def test_no_compaction_when_disabled(self):
        wf = MasterWorkflow(
            model="m", effort="low", branches=1, max_iterations=1,
            agent_class=FakeAgent, compaction_interval=0, max_context_chars=0,
        )
        result = asyncio.run(wf.execute("build a thing"))
        self.assertNotIn("(compacted)", result)      # accumulates all 4 step summaries
        self.assertEqual(result.count("Summary"), 4)


if __name__ == "__main__":
    unittest.main()

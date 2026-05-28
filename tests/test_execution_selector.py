"""Tests for execution-based candidate selection."""
from __future__ import annotations

import asyncio

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.execution_selector import ExecutionSelector
from agy_orchestrator.workflows.generate_and_rank import GenerateAndRankWorkflow


class _CannedAgent(AgentInstance):
    """Returns a fixed string; counts calls."""

    def __init__(self, reply: str = "", prompt: str = ""):
        super().__init__(prompt=prompt)
        self._reply = reply
        self.calls = 0

    @classmethod
    async def get_available_models(cls):
        return []

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        return 100.0

    def build_command(self, piped_input=None):
        return []

    async def run_async(self, piped_input=None) -> str:
        self.calls += 1
        return self._reply


def _tiny_test_source() -> str:
    return """
    from solution import add

    def test_add_pos():
        assert add(2, 3) == 5

    def test_add_zero():
        assert add(-1, 1) == 0
    """


def test_execution_selector_picks_highest_pass_rate_candidate():
    selector = ExecutionSelector(_tiny_test_source())

    correct = """```python\ndef add(a, b):\n    return a + b\n```"""
    wrong = """```python\ndef add(a, b):\n    return a - b\n```"""
    broken = """```python\ndef add(a, b)\n    return a + b\n```"""

    best_index, scores = asyncio.run(selector.select([correct, wrong, broken]))

    assert best_index == 0
    correct_passed, correct_total = scores[0]
    wrong_passed, wrong_total = scores[1]
    broken_passed, broken_total = scores[2]

    assert correct_total > 0
    assert (correct_passed, correct_total) == (correct_total, correct_total)
    assert wrong_passed == 0
    assert wrong_total > 0
    assert (broken_passed, broken_total) == (0, 0)


def test_generate_and_rank_uses_execution_selector_for_k_gt_1():
    wrong = _CannedAgent("""```python\ndef add(a, b):\n    return a - b\n```""")
    correct = _CannedAgent("""```python\ndef add(a, b):\n    return a + b\n```""")

    wf = GenerateAndRankWorkflow(
        [wrong, correct],
        execution_selector=ExecutionSelector(_tiny_test_source()),
    )
    out = asyncio.run(wf.execute("write add"))

    assert "return a + b" in out
    assert wf.verified is True
    assert wf.n_candidates == 2
    assert wf.n_passed == 1

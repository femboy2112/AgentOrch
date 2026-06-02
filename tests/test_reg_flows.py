"""Regression tests for confirmed defects in the flows subsystem
(tree_of_thought / generate_and_rank / decompose / test_feedback / cascade / pat).

Hermetic: no real workers, no network, no subprocess. Pure functions and stub
AgentInstances only.
"""
import asyncio
import sys

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.workflows.decompose import AdaptiveDecomposer
from agy_orchestrator.workflows.tree_of_thought import TreeOfThought


class _StubAgent(AgentInstance):
    """Minimal AgentInstance returning a canned reply (no subprocess)."""

    def __init__(self, reply: str = "x"):
        super().__init__(prompt="")
        self._reply = reply

    @classmethod
    async def get_available_models(cls):
        return []

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        return 100.0

    def build_command(self, piped_input=None):
        return []

    async def run_async(self, piped_input=None) -> str:
        return self._reply


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# flows-decompose-1: AdaptiveDecomposer must give up gracefully when an
# operator-supplied max_depth exceeds the interpreter recursion limit and the
# decompose keeps yielding a subtask — instead of overflowing the Python stack
# with a RecursionError and leaving self.root a half-built tree.
# --------------------------------------------------------------------------- #
def test_decompose_huge_max_depth_gives_up_without_recursionerror():
    # max_depth far above sys.getrecursionlimit() (~1000), self-referential
    # decompose that always yields one child and always fails.
    dec = AdaptiveDecomposer(max_depth=5000, max_subtasks=1)
    out, ok = dec.run("x", lambda t: ("out", False), lambda t: [t])
    # Documented give-up contract holds: (failed_output, False), no crash.
    assert (out, ok) == ("out", False)
    # Bounded below the recursion limit, well under the requested 5000.
    assert dec.depth_reached < sys.getrecursionlimit()
    assert dec.depth_reached >= 1
    # The ledger tree is consistent (fully serializable, not half-built garbage).
    import json
    json.dumps(dec.root.as_dict())


def test_decompose_huge_max_depth_with_low_recursion_limit():
    # Make the clamp observable even on small trees: a tight recursion limit
    # forces give-up well before max_depth without a RecursionError.
    original = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(200)
        dec = AdaptiveDecomposer(max_depth=10_000, max_subtasks=1)
        out, ok = dec.run("x", lambda t: ("out", False), lambda t: [t])
        assert (out, ok) == ("out", False)
        assert dec.depth_reached < 200
    finally:
        sys.setrecursionlimit(original)


def test_decompose_normal_max_depth_still_bounded_by_max_depth():
    # Control: a small max_depth (below the recursion ceiling) is still the bound.
    dec = AdaptiveDecomposer(max_depth=4, max_subtasks=1)
    out, ok = dec.run("x", lambda t: ("o", False), lambda t: [t])
    assert ok is False
    assert dec.depth_reached == 4


# --------------------------------------------------------------------------- #
# flows-tot-1: TreeOfThought must reject an empty branch_instances list at
# construction (mirroring GenerateAndRank/Cascade) rather than crashing later
# inside execute() with "max() iterable argument is empty".
# --------------------------------------------------------------------------- #
def test_tot_empty_branches_raises_at_construction():
    with pytest.raises(ValueError):
        TreeOfThought([], _StubAgent(), selector="vote")
    with pytest.raises(ValueError):
        TreeOfThought([], _StubAgent(), selector="judge")


def test_tot_nonempty_branches_still_constructs():
    # Sanity: the validation does not reject a valid single-branch config.
    tot = TreeOfThought([_StubAgent("only")], _StubAgent(), selector="vote")
    assert run(tot.execute()) == "only"

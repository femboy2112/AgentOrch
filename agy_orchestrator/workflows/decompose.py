"""As-needed recursive decomposition — ADaPT-style inverse granularity.

The research fan-out's answer to "what granularity should we plan at?": do NOT
decompose up front. Try the task whole; only when the attempt
FAILS do we break it into subtasks and recurse, so the decomposition depth adapts
to the model's competence — a strong model solves most tasks directly (no extra
passes), a weak one triggers deeper splitting exactly where it stumbles (ADaPT:
"As-Needed Decomposition and Planning", arXiv:2311.05772).

This module is deliberately MODEL-AGNOSTIC: it takes two callables —
``solve_one(subtask) -> (output, ok)`` and ``decompose(task) -> List[subtask]`` —
so the recursion logic is pure and unit-testable with plain stubs (no agents, no
verifier, no subprocess). The orchestrator wires real workers into the callables.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DecompNode:
    """A node in the decomposition tree, recorded for the run ledger.

    ``task`` is the subtask attempted, ``ok`` whether its direct attempt passed,
    ``depth`` its level (0 = root), and ``children`` the nodes it decomposed into
    (empty if it was solved directly or hit ``max_depth``).
    """

    def __init__(self, task: Any, depth: int):
        self.task = task
        self.depth = depth
        self.ok: bool = False
        self.children: List["DecompNode"] = []

    def as_dict(self) -> dict:
        """JSON-friendly view of the subtree (handy for the ledger)."""
        return {
            "task": self.task,
            "depth": self.depth,
            "ok": self.ok,
            "children": [c.as_dict() for c in self.children],
        }


class AdaptiveDecomposer:
    """Solve a task by attempting it whole, decomposing ON FAILURE, and recursing.

    ``solve_one(task) -> (output, ok)`` attempts a single task directly.
    ``decompose(task) -> List[subtask]`` splits a task into smaller subtasks.

    Both are caller-supplied callables so this stays model-free and testable.

    On a failed direct attempt the task is decomposed (up to ``max_subtasks``
    subtasks) and each subtask is solved recursively (down to ``max_depth``). The
    aggregated outputs of the solved subtasks are joined; the overall ``ok`` is the
    AND of the subtasks' results (a task succeeds only if all its parts do).

    Ledger signals (populated by ``run``):
      * ``self.root``          — the root ``DecompNode`` (full tree via ``.as_dict()``).
      * ``self.depth_reached`` — the deepest level any decomposition reached.
      * ``self.n_decompositions`` — how many times ``decompose`` was invoked.
    """

    def __init__(self, max_depth: int = 2, max_subtasks: int = 6):
        self.max_depth = max_depth
        self.max_subtasks = max_subtasks
        # Signals for the run ledger.
        self.root: Optional[DecompNode] = None
        self.depth_reached = 0
        self.n_decompositions = 0

    def run(
        self,
        task: Any,
        solve_one: Callable[[Any], Tuple[str, bool]],
        decompose: Callable[[Any], List[Any]],
    ) -> Tuple[str, bool]:
        """Solve ``task``, decomposing as needed. Returns ``(output, ok)``."""
        self.root = DecompNode(task, depth=0)
        self.depth_reached = 0
        self.n_decompositions = 0
        return self._solve(task, solve_one, decompose, self.root)

    def _solve(
        self,
        task: Any,
        solve_one: Callable[[Any], Tuple[str, bool]],
        decompose: Callable[[Any], List[Any]],
        node: DecompNode,
    ) -> Tuple[str, bool]:
        self.depth_reached = max(self.depth_reached, node.depth)

        # 1) Try the task whole first — the inverse-granularity bet is that most
        #    attempts succeed directly and cost no decomposition.
        output, ok = solve_one(task)
        node.ok = ok
        if ok:
            logger.info("Decompose: solved directly at depth %d.", node.depth)
            return output, True

        # 2) Failed. If we're out of depth budget, return the failed attempt as-is
        #    (no further splitting — a weak model can't recurse forever).
        if node.depth >= self.max_depth:
            logger.info("Decompose: failed at depth %d and max_depth reached — giving up.",
                        node.depth)
            return output, False

        # 3) Decompose ON FAILURE and solve each subtask recursively.
        subtasks = list(decompose(task))[: self.max_subtasks]
        if not subtasks:
            logger.info("Decompose: no subtasks produced at depth %d — giving up.", node.depth)
            return output, False
        self.n_decompositions += 1
        logger.info("Decompose: depth %d failed — split into %d subtask(s).",
                    node.depth, len(subtasks))

        outputs: List[str] = []
        all_ok = True
        for sub in subtasks:
            child = DecompNode(sub, depth=node.depth + 1)
            node.children.append(child)
            sub_out, sub_ok = self._solve(sub, solve_one, decompose, child)
            outputs.append(sub_out)
            all_ok = all_ok and sub_ok

        node.ok = all_ok
        return "\n".join(outputs), all_ok

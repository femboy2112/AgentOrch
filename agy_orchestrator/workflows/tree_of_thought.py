import logging
import asyncio
import re
from collections import Counter
from typing import Dict, List, Tuple
from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.pipeline import ParallelSwarm

logger = logging.getLogger(__name__)

# A leading reasoning-model <think> block; stripped before comparing/parsing
# so the visible answer (not the chain-of-thought) drives selection.
_THINK_RE = re.compile(r"^\s*<think>.*?</think>", re.IGNORECASE | re.DOTALL)
# Explicit score patterns first (most reliable): "8/10", "score: 8", "rating 8".
_SCORE_RE = re.compile(r"(?:score|rating|rate)\D{0,4}(\d{1,2})|(\d{1,2})\s*/\s*10", re.IGNORECASE)
# Fallback: any 1-2 digit run (so "8/10" -> 8, not 810).
_INT_RE = re.compile(r"\d{1,2}")


def _strip_think(text: str) -> str:
    """Drop a leading ``<think>...</think>`` block (reasoning models emit one)."""
    return _THINK_RE.sub("", text, count=1)


def _parse_score(reply: str) -> int:
    """Robustly extract a 1-10 quality score from an evaluator's reply.

    Prefers an EXPLICIT score pattern ("8/10", "score: 8", "rating 8"); a chatty
    judge that writes "issue 1: ..." before its verdict would mis-score under a
    naive first-integer grab. Falls back to the LAST integer in the reply (the
    verdict usually comes last, mirroring the last-line discipline in
    ``_is_approved``). Returns 0 when no integer is present so garbage can't score.
    """
    body = _strip_think(reply)
    m = _SCORE_RE.search(body)
    if m:
        return max(0, min(10, int(m.group(1) or m.group(2))))
    ints = _INT_RE.findall(body)
    if not ints:
        return 0
    return max(0, min(10, int(ints[-1])))


def _normalize(text: str) -> str:
    """Normalize a branch output for vote comparison: strip a leading reasoning
    block, collapse whitespace, lowercase."""
    body = _strip_think(text)
    return re.sub(r"\s+", " ", body).strip().lower()


class TreeOfThought:
    """
    Implements a single-layer Tree-of-Thought (ToT) generation strategy.
    Generates multiple independent solution paths and selects the best one.

    ``selector`` controls selection:
      * ``"judge"`` (default) — an Evaluator instance scores each branch 1-10
        and the argmax wins (one extra model pass per branch).
      * ``"vote"`` — pick by majority vote over normalized branch outputs; the
        evaluator is never invoked (best-of-N with ZERO extra model passes).
    """
    def __init__(
        self,
        branch_instances: List[AgentInstance],
        evaluator_instance: AgentInstance,
        selector: str = "judge",
    ):
        self.branch_instances = branch_instances
        self.evaluator = evaluator_instance
        self.selector = selector

    async def execute(self) -> str:
        logger.info(f"Generating {len(self.branch_instances)} ToT branches concurrently...")

        # Execute branches in parallel
        swarm = ParallelSwarm(self.branch_instances)
        outputs = await swarm.execute()

        if self.selector == "vote":
            return self._select_by_vote(outputs)
        return await self._select_by_judge(outputs)

    def _select_by_vote(self, outputs: List[str]) -> str:
        """Majority vote over normalized outputs; ties broken by first occurrence.
        No evaluator pass is spent."""
        logger.info("Selecting branch by majority vote (no evaluator pass)...")
        counts: Counter = Counter()
        first_index: Dict[str, int] = {}
        for idx, out in enumerate(outputs):
            key = _normalize(out)
            counts[key] += 1
            first_index.setdefault(key, idx)

        # Highest count wins; on a tie prefer the earliest first occurrence.
        best_key = max(counts, key=lambda k: (counts[k], -first_index[k]))
        tally = {f"#{first_index[k] + 1}": counts[k] for k in counts}
        logger.info("Vote tally (branch->votes): %s", tally)
        winner_idx = first_index[best_key]
        logger.info(f"Selected branch {winner_idx + 1} with {counts[best_key]} vote(s)")
        return outputs[winner_idx]

    async def _select_by_judge(self, outputs: List[str]) -> str:
        logger.info("Evaluating generated branches...")
        scored_outputs: List[Tuple[int, str]] = []

        for idx, out in enumerate(outputs):
            self.evaluator.prompt = (
                f"Evaluate the following solution on a scale of 1-10 based on correctness, "
                f"efficiency, and adherence to best practices for the task at hand. "
                f"Hold it to a high, domain-appropriate quality bar judged against the task's own "
                f"goals (correct and robust code, precise and well-structured writing, etc.) — "
                f"not a fixed template.\n"
                f"Penalize heavily (score < 5) for incorrect results, bugs, or identifiers, signatures, "
                f"or interfaces that must match across files or components but don't.\n"
                f"Reply ONLY with the integer score.\n\nSolution:\n{out}"
            )
            score_str = await self.evaluator.run_async()
            score = _parse_score(score_str)

            logger.info(f"Branch {idx+1} evaluated with score: {score}")
            scored_outputs.append((score, out))

        # Select the branch with the highest score (stable: ties keep branch order).
        scored_outputs.sort(key=lambda x: x[0], reverse=True)
        best_score, best_output = scored_outputs[0]

        logger.info(f"Selected best branch with score {best_score}")
        return best_output

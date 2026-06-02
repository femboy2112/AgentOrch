import asyncio
import copy
import logging
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.pipeline import ParallelSwarm

# Shared evaluator rubric used to score one candidate solution 1-10. Kept here so
# tree_of_thought and generate_and_rank score on identical criteria.
JUDGE_RUBRIC = (
    "Evaluate the following {noun} on a scale of 1-10 based on correctness, "
    "efficiency, and adherence to best practices for the task at hand. "
    "Hold it to a high, domain-appropriate quality bar judged against the task's own "
    "goals (correct and robust code, precise and well-structured writing, etc.) — "
    "not a fixed template.\n"
    "Penalize heavily (score < 5) for incorrect results, bugs, or identifiers, signatures, "
    "or interfaces that must match across files or components but don't.\n"
    "Reply ONLY with the integer score.\n\n{noun_cap}:\n{solution}"
)

# Variant used when the caller can supply the REQUIREMENT the candidate must
# satisfy. Without it the judge can only assess a branch's internal consistency,
# blind to the goal — so a branch that is self-consistent but solves the wrong
# problem scores as well as a correct one. Grounding the score in the requirement
# is a strict improvement to selection quality (single-run speed audit, win 5).
JUDGE_RUBRIC_WITH_REQ = (
    "Evaluate the following {noun} on a scale of 1-10 by how fully and correctly it "
    "satisfies the REQUIREMENT below, then by efficiency and best practices.\n"
    "Penalize heavily (score < 5) for not meeting the requirement, incorrect results, "
    "bugs, or identifiers, signatures, or interfaces that must match across files or "
    "components but don't.\n"
    "Reply ONLY with the integer score.\n\n"
    "Requirement:\n{requirement}\n\n{noun_cap}:\n{solution}"
)


def build_judge_prompt(solution: str, noun: str = "solution",
                       requirement: Optional[str] = None) -> str:
    if requirement:
        return JUDGE_RUBRIC_WITH_REQ.format(
            noun=noun, noun_cap=noun.capitalize(), solution=solution,
            requirement=requirement,
        )
    return JUDGE_RUBRIC.format(noun=noun, noun_cap=noun.capitalize(), solution=solution)

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
        event_callback: Optional[Callable[[dict], None]] = None,
        requirement: Optional[str] = None,
    ):
        if not branch_instances:
            raise ValueError("TreeOfThought needs at least one branch instance")
        self.branch_instances = branch_instances
        self.evaluator = evaluator_instance
        self.selector = selector
        self.event_callback = event_callback
        # The goal each branch must satisfy. When provided, the judge scores
        # branches AGAINST it instead of blind (win 5). None preserves the prior
        # requirement-free rubric (e.g. generate_and_rank).
        self.requirement = requirement

    def _emit_orchestration(self, **fields) -> None:
        cb = self.event_callback
        if cb is None:
            return
        orchestration = {"workflow": "master"}
        for key, value in fields.items():
            if value is not None:
                orchestration[key] = value
        try:
            cb(
                {
                    "kind": "lifecycle",
                    "data": {
                        "event": "orchestration_transition",
                        "orchestration": orchestration,
                    },
                }
            )
        except Exception:
            pass

    async def execute(self) -> str:
        logger.info(f"Generating {len(self.branch_instances)} ToT branches concurrently...")

        # Execute branches in parallel
        swarm = ParallelSwarm(self.branch_instances)
        outputs = await swarm.execute()

        # Degenerate case (configured with 1 branch, or only one branch survived):
        # there is nothing to choose between, so skip the evaluator pass entirely.
        if len(outputs) == 1:
            logger.info("Only one branch output — returning it without an evaluator pass.")
            return outputs[0]

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
        branch_scores = [counts[_normalize(out)] for out in outputs]
        logger.info(f"Selected branch {winner_idx + 1} with {counts[best_key]} vote(s)")
        self._emit_orchestration(
            phase="tot",
            action="branch_selected",
            selected_branch=winner_idx + 1,
            branch_total=len(outputs),
            selector="vote",
            score=counts[best_key],
            scores=branch_scores,
        )
        return outputs[winner_idx]

    async def _select_by_judge(self, outputs: List[str]) -> str:
        logger.info("Evaluating %d branches concurrently...", len(outputs))

        # Score every branch in parallel: clone the evaluator per branch (each gets
        # its own prompt/session) and run them concurrently, instead of awaiting one
        # mutated evaluator in a loop (which serialized N model calls on the hot path).
        async def _score(out: str) -> int:
            judge = copy.copy(self.evaluator)
            judge.session_id = None  # independent: don't chain branch scorings
            judge.prompt = build_judge_prompt(out, noun="solution",
                                              requirement=self.requirement)
            return _parse_score(await judge.run_async())

        results = await asyncio.gather(*[_score(o) for o in outputs],
                                       return_exceptions=True)

        scored_outputs: List[Tuple[int, int, str]] = []  # (score, -idx, text) stable argmax
        for idx, (out, res) in enumerate(zip(outputs, results)):
            if isinstance(res, Exception):
                logger.warning("Branch %d evaluation failed (%s); scoring 0.", idx + 1, res)
                score = 0
            else:
                score = res
            logger.info("Branch %d evaluated with score: %d", idx + 1, score)
            scored_outputs.append((score, -idx, out))

        best_score, neg_idx, best_output = max(scored_outputs, key=lambda t: (t[0], t[1]))
        branch_scores = [score for score, _, _ in scored_outputs]
        logger.info("Selected best branch (#%d) with score %d", -neg_idx + 1, best_score)
        self._emit_orchestration(
            phase="tot",
            action="branch_selected",
            selected_branch=-neg_idx + 1,
            branch_total=len(outputs),
            selector="judge",
            score=best_score,
            scores=branch_scores,
        )
        return best_output

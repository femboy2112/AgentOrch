"""GenerateAndRankWorkflow — best-of-K generation with verifier-asymmetry ranking.

The research fan-out's cheapest quality lever for a weak generator: generate K
independent candidates, then RANK them with a strong signal. Verification is
asymmetric — checking a candidate is far easier (and more reliable) than writing
one — so a weak generator paired with a good verifier approximates a strong
generator ("Small LMs Need Strong Verifiers", arXiv:2404.17140). When no hard
verifier is available we fall back to an LLM judge that scores 1-10 (reusing the
robust score parsing from tree_of_thought).

DISK-STATE LIMITATION (read carefully — it drives the semantics):
Our agents write their artifacts to disk, and only ONE candidate's artifact can
be on disk at a time — running K generators sequentially leaves only the LAST
candidate's files in place, so a programmatic QualityVerifier (which inspects the
working directory) can only meaningfully judge the candidate currently resident.
We do NOT have per-candidate isolation (separate worktrees), so we cannot verify
each of the K candidates independently. Therefore:

  * K == 1 + verifier  -> the verifier acts as a pass/fail GATE on that single
    candidate (its artifact is the one on disk). ``self.verified`` reflects it.
  * K  > 1             -> ranking is driven by the LLM judge/ranker, which scores
    the candidate TEXT (no disk dependency, so all K are comparable). If a verifier
    is also supplied it is run once as a final gate on the chosen candidate (which,
    being the last thing the winner would be re-applied as, is out of scope here),
    so for K>1 the verifier only sets the ``self.verified`` signal on whatever is
    currently on disk — it does NOT reorder candidates. This keeps the semantics
    correct rather than pretending to verify text that was never applied.
  * K  > 1 + execution selector -> for code tasks we can lift this limitation by
    executing EACH candidate in its own fresh sandbox. That per-candidate
    isolation makes execution-based ranking sound again, so pass-count ranking can
    replace text-only LLM judging.

In short: the verifier is exploited as a true gate exactly when it can be (K==1),
and the judge ranks the text otherwise. This is honest about what disk state lets
us measure while still capturing the verifier asymmetry whenever it is sound.
"""
from __future__ import annotations

import asyncio
import copy
import logging
from typing import List, Optional, Tuple

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.execution_selector import ExecutionSelector
from agy_orchestrator.execution.pipeline import ParallelSwarm
from agy_orchestrator.execution.verifier import QualityVerifier
from agy_orchestrator.workflows.tree_of_thought import _parse_score, build_judge_prompt

logger = logging.getLogger(__name__)


class GenerateAndRankWorkflow:
    """Generate K candidates concurrently, then return the best-ranked one.

    Ranking precedence:
      1. If a verifier can be applied (K == 1), a passing candidate ranks above a
         failing one — verification is the strongest, most reliable signal.
      2. Otherwise (or to break ties among candidates), an optional LLM ranker
         scores each candidate 1-10 (argmax wins, stable on ties).
      3. With no verifier and no ranker, the first candidate is returned.

    Ledger signals:
      * ``self.verified``     — the chosen candidate passed the verifier gate.
      * ``self.n_candidates`` — how many candidates actually came back (a crashed
        generator is dropped by ParallelSwarm, so this can be < K).
      * ``self.n_passed``     — candidates known to pass the verifier (0 or 1 here,
        given the disk-state limitation above).
    """

    def __init__(
        self,
        generator_instances: List[AgentInstance],
        verifier: Optional[QualityVerifier] = None,
        ranker: Optional[AgentInstance] = None,
        execution_selector: Optional[ExecutionSelector] = None,
        working_directory: str = ".",
    ):
        if not generator_instances:
            raise ValueError("GenerateAndRankWorkflow needs at least one generator instance")
        self.generators = generator_instances
        self.verifier = verifier
        self.ranker = ranker
        self.execution_selector = execution_selector
        self.working_directory = working_directory
        # Signals for the run ledger (task #9).
        self.verified = False
        self.n_candidates = 0
        self.n_passed = 0

    async def execute(self, prompt: str) -> str:
        logger.info("GenerateAndRank: generating %d candidate(s) concurrently...",
                    len(self.generators))
        for gen in self.generators:
            gen.prompt = prompt
        swarm = ParallelSwarm(self.generators)
        candidates = await swarm.execute()
        self.n_candidates = len(candidates)
        logger.info("GenerateAndRank: %d candidate(s) returned.", self.n_candidates)

        if self.n_candidates == 1:
            # Single candidate: its artifact is the one on disk, so the verifier is
            # a sound pass/fail gate. We still return it (best of one) regardless,
            # but record whether it verified.
            return await self._rank_single(candidates[0])
        return await self._rank_many(candidates)

    async def _rank_single(self, candidate: str) -> str:
        """K == 1: use the verifier as a true gate on the resident artifact."""
        if self.verifier is not None:
            result = await self.verifier.verify(working_directory=self.working_directory)
            self.verified = result.ok
            self.n_passed = 1 if result.ok else 0
            if result.ok:
                logger.info("GenerateAndRank: sole candidate passed the verifier gate.")
            else:
                logger.info("GenerateAndRank: sole candidate failed the verifier: %s",
                            result.message or "(no detail)")
        return candidate

    async def _rank_many(self, candidates: List[str]) -> str:
        """K > 1: rank candidate TEXT with the judge (disk verification is unsound
        across un-isolated candidates). A verifier, if present, is run once only to
        set the ``verified`` signal on the artifact currently on disk."""
        if self.execution_selector is not None:
            logger.info("GenerateAndRank: ranking %d candidates by execution...",
                        len(candidates))
            best_index, scores = await self.execution_selector.select(candidates)
            for idx, (passed, total) in enumerate(scores):
                logger.info("GenerateAndRank: candidate %d execution score = %d/%d.",
                            idx + 1, passed, total)
            best_passed, best_total = scores[best_index]
            self.verified = best_total > 0 and best_passed == best_total
            self.n_passed = sum(
                1 for passed, total in scores if total > 0 and passed == total
            )
            logger.info("GenerateAndRank: selected candidate %d by execution score %d/%d.",
                        best_index + 1, best_passed, best_total)
            return candidates[best_index]

        if self.ranker is not None:
            best = await self._rank_by_judge(candidates)
        else:
            logger.info("GenerateAndRank: no ranker supplied — returning the first candidate.")
            best = candidates[0]

        # The verifier cannot reorder un-isolated candidates, but we still report
        # whether whatever is currently on disk passes, for the ledger.
        if self.verifier is not None:
            result = await self.verifier.verify(working_directory=self.working_directory)
            self.verified = result.ok
            self.n_passed = 1 if result.ok else 0
        return best

    async def _rank_by_judge(self, candidates: List[str]) -> str:
        """Score each candidate 1-10 with the LLM ranker concurrently; argmax wins."""
        logger.info("GenerateAndRank: ranking %d candidates with the LLM judge...",
                    len(candidates))

        # Clone the ranker per candidate so all scorings run in parallel instead of
        # serializing N model calls through one mutated instance.
        async def _score(cand: str) -> int:
            judge = copy.copy(self.ranker)
            judge.session_id = None
            judge.prompt = build_judge_prompt(cand, noun="candidate")
            return _parse_score(await judge.run_async())

        results = await asyncio.gather(*[_score(c) for c in candidates],
                                       return_exceptions=True)

        scored: List[Tuple[int, int, str]] = []  # (score, -index, text) for stable argmax
        for idx, (cand, res) in enumerate(zip(candidates, results)):
            score = 0 if isinstance(res, Exception) else res
            if isinstance(res, Exception):
                logger.warning("GenerateAndRank: candidate %d scoring failed (%s); scoring 0.",
                               idx + 1, res)
            logger.info("GenerateAndRank: candidate %d scored %d.", idx + 1, score)
            scored.append((score, -idx, cand))

        # Highest score wins; ties keep candidate order (the -index secondary key).
        best_score, neg_idx, best = max(scored, key=lambda t: (t[0], t[1]))
        logger.info("GenerateAndRank: selected candidate %d with score %d.",
                    -neg_idx + 1, best_score)
        return best

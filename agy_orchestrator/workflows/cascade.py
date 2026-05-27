"""CascadeWorkflow — cheap-first model cascade with as-needed escalation.

The research fan-out's recommended routing for our one-model-resident, RAM-tight
box: try the CHEAP/fast model first, and escalate to a stronger (slower) model
ONLY when the strong verifier (tests/lint/build) still fails. Most steps are
within the small model's competence, so we pay the big model's cost only on the
hard ones — far cheaper than always running the big model, and higher quality
than always running the small one. (Model cascades / FrugalGPT; ADaPT's
escalate-on-failure, arXiv:2311.05772.)

Each stage runs the proven TestFeedbackWorkflow (generate -> run tests -> feed
the error back -> repair). The verifier is the gate between stages: a stage
"succeeds" only if it makes the verifier pass. Returns as soon as a stage passes.
"""
from __future__ import annotations

import logging
from typing import List

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import QualityVerifier
from agy_orchestrator.workflows.test_feedback import TestFeedbackWorkflow

logger = logging.getLogger(__name__)


class CascadeWorkflow:
    def __init__(
        self,
        stages: List[AgentInstance],
        verifier: QualityVerifier,
        max_iterations_per_stage: int = 2,
        working_directory: str = ".",
    ):
        if not stages:
            raise ValueError("CascadeWorkflow needs at least one generator stage")
        if verifier is None:
            raise ValueError("CascadeWorkflow requires a QualityVerifier (the escalation gate)")
        self.stages = stages
        self.verifier = verifier
        self.max_iterations_per_stage = max_iterations_per_stage
        self.working_directory = working_directory
        # Signals for the ledger.
        self.verified = False
        self.stage_used = -1          # index of the stage that passed (-1 = none)
        self.iterations_used = 0      # total repair rounds across stages
        self.stalled = False

    async def execute(self, initial_prompt: str) -> str:
        last_output = ""
        for i, gen in enumerate(self.stages):
            logger.info("Cascade stage %d/%d (%s)", i + 1, len(self.stages), type(gen).__name__)
            wf = TestFeedbackWorkflow(
                gen, self.verifier,
                max_iterations=self.max_iterations_per_stage,
                working_directory=self.working_directory,
            )
            last_output = await wf.execute(initial_prompt)
            self.iterations_used += wf.iterations_used
            if wf.verified:
                logger.info("Cascade passed at stage %d after %d total repair rounds.",
                            i + 1, self.iterations_used)
                self.verified = True
                self.stage_used = i
                return last_output
            logger.info("Stage %d did not pass the verifier; escalating.", i + 1)
        logger.warning("Cascade exhausted all %d stages without passing.", len(self.stages))
        self.stalled = True
        return last_output

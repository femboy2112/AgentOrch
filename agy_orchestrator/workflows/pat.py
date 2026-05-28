"""PatWorkflow — Plan-after-Trial: try direct first, plan only on failure.

Most coding tasks the operator dispatches don't need the full master-mode
plan+ToT+adversarial pipeline. A direct one-shot to a competent worker
usually solves them; the plan tax is then wasted compute.

PaT inverts the default: **try the cheap, fast, plan-less path first**, gate
it on the programmatic verifier, and escalate to master mode ONLY when the
verifier fails. On easy tasks we skip the plan entirely; on hard tasks we
pay the master cost AFTER ruling out the simpler path.

This is workflow-tier escalation, not model-tier escalation: cascade
(workflows/cascade.py) walks down the model chain; PaT walks up the
workflow complexity ladder for the same chain. The two are composable —
nothing stops a future caller from running PaT inside cascade or vice
versa — but they live as separate modes so the operator picks the axis
that matches the task.

Paper anchor: "PaT: Planning-after-Trial for Efficient Test-Time Code
Generation" (arxiv 2605.07248, May 2026) — reports 40% cost reduction at
equal quality vs always-plan-first on coding benchmarks; further 69%
savings with heterogeneous pairing (cheap generator + strong planner on
escalation). Our default chain (--generator codex,agy,grok) already
provides that heterogeneity.

The verifier is mandatory: PaT cannot decide whether the direct attempt
succeeded without it. Callers that want a critic-only or plan-always
workflow should use adversarial or master directly.
"""
from __future__ import annotations

import logging
from typing import Optional

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import QualityVerifier
from agy_orchestrator.workflows.master import MasterWorkflow

logger = logging.getLogger(__name__)


class PatWorkflow:
    """Plan-after-Trial: direct attempt → verifier → master on failure.

    Stage 1 (always runs):  one direct generator call, no critic loop.
    Verifier gate.
    Stage 2 (only on Stage 1 verifier failure): the full MasterWorkflow.

    Returns the FIRST stage output that passes the verifier, or the
    last-stage output if neither stage passes (still useful — the operator
    sees what was tried).
    """

    def __init__(
        self,
        direct_generator: AgentInstance,
        master_workflow: MasterWorkflow,
        verifier: QualityVerifier,
        working_directory: str = ".",
    ):
        if verifier is None:
            raise ValueError(
                "PatWorkflow requires a QualityVerifier — without it, PaT cannot "
                "decide whether the direct attempt succeeded. Use 'master' mode "
                "if you want plan-always behaviour without a programmatic gate."
            )
        if direct_generator is None:
            raise ValueError("PatWorkflow needs a direct_generator for Stage 1.")
        if master_workflow is None:
            raise ValueError("PatWorkflow needs a master_workflow for the escalation stage.")
        self.direct_generator = direct_generator
        self.master_workflow = master_workflow
        self.verifier = verifier
        self.working_directory = working_directory
        # Ledger signals.
        self.verified = False
        self.stage_used = -1          # 0 = direct passed, 1 = master ran (passed or not), -1 = none
        self.iterations_used = 0      # at least 1 for the direct attempt
        self.stalled = False
        self.approved = False         # carries through from the inner workflow when used

    async def execute(self, initial_prompt: str) -> str:
        # Stage 1: direct generator, no critic.
        logger.info("PaT Stage 1: direct attempt (no plan, no critic)")
        self.direct_generator.prompt = initial_prompt
        direct_output = await self.direct_generator.run_async()
        self.iterations_used = 1

        success, error_msg = await self.verifier.verify(
            working_directory=self.working_directory,
        )
        if success:
            logger.info("PaT Stage 1 passed verifier — skipping master mode.")
            self.verified = True
            self.approved = True
            self.stage_used = 0
            return direct_output

        # Stage 2: escalate to master. The direct output and its failure
        # are surfaced in the prompt so master's planner can route around
        # the observed failure instead of starting blind.
        logger.info("PaT Stage 1 failed verifier; escalating to master mode.")
        escalation_prompt = (
            f"{initial_prompt}\n\n"
            f"## Prior direct attempt (failed verifier)\n"
            f"{direct_output}\n\n"
            f"## Verifier error\n"
            f"{error_msg}\n\n"
            f"Plan and execute the work needed to make the verifier pass. "
            f"Treat the prior attempt as one data point; you may reuse or "
            f"discard any of it."
        )
        master_output = await self.master_workflow.execute(escalation_prompt)
        self.stage_used = 1
        # Master's internal signals — the inner adversarial review tracks
        # whether its final iteration verified. Surface that here.
        inner_verified = getattr(self.master_workflow, "verified", False)
        inner_approved = getattr(self.master_workflow, "approved", False)
        # Some MasterWorkflow paths don't set verified directly; fall back
        # to re-running the verifier one more time on the master output to
        # get a clean signal for the run ledger.
        if not inner_verified:
            final_ok, _ = await self.verifier.verify(working_directory=self.working_directory)
            self.verified = final_ok
        else:
            self.verified = True
        self.approved = self.verified or inner_approved
        self.stalled = not self.approved
        # Master tracks its own iteration count via the steps it ran; add it
        # to the Stage 1 count so the ledger reflects total work done.
        self.iterations_used += getattr(self.master_workflow, "iterations_used", 0)
        return master_output


def _build_pat_workflow(
    *,
    direct_generator: AgentInstance,
    master_workflow: MasterWorkflow,
    verifier: Optional[QualityVerifier],
    working_directory: str = ".",
) -> PatWorkflow:
    """Module-level convenience builder used by harness/dispatch.py."""
    return PatWorkflow(
        direct_generator=direct_generator,
        master_workflow=master_workflow,
        verifier=verifier,
        working_directory=working_directory,
    )

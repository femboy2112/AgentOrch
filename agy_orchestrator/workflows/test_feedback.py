"""TestFeedbackWorkflow — generate, run the REAL tests, feed failures back, repair.

The verifier is a STRONG, non-LLM oracle, so even a model that cannot reliably
self-critique ("Small LMs Need Strong Verifiers", arXiv:2404.17140) still
converges when handed the actual error.

It is deliberately NOT the adversarial loop: there is no LLM critic. The generator
writes, the programmatic ``QualityVerifier`` (tests/lint/build) judges, and on
failure the verbatim error + the failing output are fed back with a PRESERVING
instruction (fix only what failed, keep what works) — feeding a weak 4B many
critiques at once makes it regress. Returns as soon as the verifier passes.
"""
from __future__ import annotations

import logging

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import QualityVerifier

logger = logging.getLogger(__name__)


class TestFeedbackWorkflow:
    def __init__(
        self,
        generator_instance: AgentInstance,
        verifier: QualityVerifier,
        max_iterations: int = 4,
        working_directory: str = ".",
        preserve: bool = True,
    ):
        if verifier is None:
            raise ValueError("TestFeedbackWorkflow requires a QualityVerifier (the strong oracle)")
        self.generator = generator_instance
        self.verifier = verifier
        self.max_iterations = max_iterations
        self.working_directory = working_directory
        # Preserve = tell the model to fix only what failed and keep working parts
        # byte-identical (avoids the weak-model rewrite-and-regress failure mode).
        self.preserve = preserve
        # Quality signals for the run ledger (task #9).
        self.iterations_used = 0
        self.verified = False

    def _revise_prompt(self, initial_prompt: str, last_output: str, error: str) -> str:
        keep = ("Fix ONLY what the error indicates and output the complete corrected "
                "version. Re-read the requirement for edge cases; keep everything that "
                "already works exactly as it is — do not rewrite from scratch."
                if self.preserve else
                "Fix the issues and output the corrected version.")
        return (f"{initial_prompt}\n\nYour last output:\n{last_output}\n\n"
                f"It FAILED verification with this error:\n{error}\n\n{keep}")

    async def execute(self, initial_prompt: str) -> str:
        current_prompt = initial_prompt
        last_output = ""
        for iteration in range(self.max_iterations):
            logger.info("TestFeedback iteration %d/%d", iteration + 1, self.max_iterations)
            self.iterations_used = iteration + 1
            self.generator.prompt = current_prompt
            last_output = await self.generator.run_async()
            result = await self.verifier.verify(working_directory=self.working_directory)
            if result.ok:
                logger.info("Verifier passed on iteration %d — accepting.", iteration + 1)
                self.verified = True
                return last_output
            logger.info("Verifier failed; feeding the error back for repair.")
            current_prompt = self._revise_prompt(
                initial_prompt,
                last_output,
                result.message or "(no detail)",
            )
        logger.warning("TestFeedback exhausted %d iterations without passing.", self.max_iterations)
        return last_output

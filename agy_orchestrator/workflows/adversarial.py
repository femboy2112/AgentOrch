import logging
import re
from typing import Callable, Optional

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import QualityVerifier

logger = logging.getLogger(__name__)

# Any reasoning-model <think> block; stripped before checking convergence so
# a critic that merely *mentions* APPROVED inside its thinking can't converge.
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

CATASTROPHIC_FOCUS_PREAMBLE = (
    "Review this adversarially and exhaustively: treat token cost as no object, and "
    "your overriding goal is to surface any flaw that could be CATASTROPHIC - "
    "anything that could exhaust resources, spawn unbounded processes, or crash or "
    "hang the host machine or this process itself. Trace the control flow and "
    "construct the adversarial input that triggers the worst case. Leave nothing "
    "unremarked; default to flagging.\n\n"
)


def _is_approved(feedback: str) -> bool:
    """True only when the critic genuinely approves.

    A bare substring match false-positives on a chatty local critic ("this is
    NOT approved yet", "I would not write APPROVED"). Instead, after dropping any
    ``<think>...</think>`` block and surrounding whitespace, approve only if the
    whole reply is exactly ``APPROVED`` OR its last non-empty line is exactly
    ``APPROVED`` (trailing punctuation allowed).
    """
    stripped = _THINK_RE.sub("", feedback).strip()
    if stripped.upper() == "APPROVED":
        return True
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if not lines:
        return False
    last = re.sub(r"[\s.!?:;,]+$", "", lines[-1].strip())
    return last.upper() == "APPROVED"

class AdversarialReview:
    """
    Executes a continuous loop where a Generator produces output,
    and a Critic reviews it against specifications. The loop
    continues until the Critic explicitly approves the output.
    """
    def __init__(
        self,
        generator_instance: AgentInstance,
        critic_instance: AgentInstance,
        verifier: Optional[QualityVerifier] = None,
        max_iterations: int = 5,
        diff_only: bool = False,
        working_directory: str = ".",
        critic_preamble: str = "",
        event_callback: Optional[Callable[[dict], None]] = None,
    ):
        self.generator = generator_instance
        self.critic = critic_instance
        self.verifier = verifier
        self.max_iterations = max_iterations
        # When True, the revise prompt tells the generator to touch ONLY what the
        # critique named — stops a weak model rewriting good parts into bad.
        self.diff_only = diff_only
        # Where the verifier should run the test_cmd. Must match where the
        # workers wrote their files; the harness threads `out_dir` to here so a
        # cross-repo dispatch (caller's `out_dir != PROJECT_ROOT`) doesn't lie
        # with `make check` in the wrong tree. Default "." preserves the prior
        # behaviour exactly for in-repo runs.
        self.working_directory = working_directory
        # When non-empty, this text is prepended to the critic prompt to push the review toward catastrophic-failure focus (opt-in, e.g. mission-critical dispatches).
        self.critic_preamble = critic_preamble
        # Quality signals, populated by execute() for the run ledger (task #9).
        self.iterations_used = 0
        self.approved = False
        self.verified = False  # programmatic verifier passed
        self.stalled = False   # bailed early on a repeated critique
        self.event_callback = event_callback

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

    def _iteration_outcome(self) -> str:
        if self.verified:
            return "verified"
        if self.approved:
            return "approved"
        if self.stalled:
            return "stalled"
        return "continue"

    async def execute(self, initial_prompt: str) -> str:
        current_prompt = initial_prompt
        last_output = ""
        prev_feedback_norm: Optional[str] = None

        for iteration in range(self.max_iterations):
            logger.info(f"Adversarial Review Iteration {iteration+1}/{self.max_iterations}")
            self.iterations_used = iteration + 1
            self._emit_orchestration(
                phase="adversarial",
                action="iteration_started",
                iteration=iteration + 1,
                iteration_total=self.max_iterations,
                model=getattr(self.generator, "model", None),
                effort=getattr(self.generator, "effort", None),
            )

            self.generator.prompt = current_prompt
            last_output = await self.generator.run_async()

            # Programmatic Verification Gate. A passing verifier (tests/lint) is
            # GROUND TRUTH — return immediately rather than falling through to the
            # LLM critic, which can only talk a weak model into REGRESSING output
            # that already passes ("Small LMs Need Strong Verifiers", 2404.17140).
            if self.verifier:
                result = await self.verifier.verify(working_directory=self.working_directory)
                if result.ok:
                    logger.info("Programmatic verification passed — accepting (no critic pass needed).")
                    self.verified = True
                    self.approved = True
                    self._emit_orchestration(
                        phase="adversarial",
                        action="iteration_completed",
                        iteration=iteration + 1,
                        iteration_total=self.max_iterations,
                        outcome=self._iteration_outcome(),
                        verified=self.verified,
                        approved=self.approved,
                    )
                    return last_output
                logger.info("Programmatic verification failed. Sending back to generator.")
                # Feed the failing CODE back with the error, not just the error —
                # regenerating from scratch loses the working parts (matches the
                # test-feedback loop in workflows/test_feedback.py).
                current_prompt = (
                    f"{initial_prompt}\n\nYour last output:\n{last_output}\n\n"
                    f"It FAILED verification with this error:\n{result.message}\n\n"
                    f"Fix ONLY what the error indicates and output the complete corrected "
                    f"version. Re-read the requirement for edge cases; keep everything that "
                    f"already works unchanged."
                )
                if iteration + 1 < self.max_iterations:
                    self._emit_orchestration(
                        phase="adversarial",
                        action="iteration_completed",
                        iteration=iteration + 1,
                        iteration_total=self.max_iterations,
                        outcome=self._iteration_outcome(),
                        verified=self.verified,
                        approved=self.approved,
                    )
                continue

            # LLM Critic Gate
            critic_prompt = (
                f"Please review the following output against the original requirement.\n"
                f"Original Requirement:\n{initial_prompt}\n\n"
                f"Generated Output:\n{last_output}\n\n"
                f"CRITICAL REVIEW INSTRUCTIONS:\n"
                f"1. CORRECTNESS: Verify the output fully and accurately satisfies the requirement. "
                f"Look for bugs, logical errors, and any identifiers, signatures, or interfaces that "
                f"must match but don't (e.g. names shared across files, function signatures, API contracts).\n"
                f"2. EXCELLENCE: Hold the output to a high, domain-appropriate quality bar for whatever "
                f"the task actually is — code should be clear, robust, and idiomatic; writing should be "
                f"precise and well-structured; any other deliverable should meet professional standards "
                f"for its kind. Judge against the task's own goals, not a fixed template.\n"
                f"3. CONVERGENCE: If the output fully meets the requirement with no defects, you MUST reply "
                f"exactly with 'APPROVED'. Do not get stuck in an endless loop of minor subjective nitpicks. "
                f"Reward work that is correct and complete.\n"
                f"If it meets all requirements with no defects, reply exactly with 'APPROVED'. "
                f"Otherwise, provide specific, actionable changes needed."
            )
            if self.critic_preamble:
                critic_prompt = self.critic_preamble + critic_prompt
            self.critic.prompt = critic_prompt
            critic_feedback = await self.critic.run_async()

            if _is_approved(critic_feedback):
                logger.info("Critic approved the output.")
                self.approved = True
                self._emit_orchestration(
                    phase="adversarial",
                    action="iteration_completed",
                    iteration=iteration + 1,
                    iteration_total=self.max_iterations,
                    outcome=self._iteration_outcome(),
                    verified=self.verified,
                    approved=self.approved,
                )
                return last_output

            # Adaptive cap: if the critic returns essentially the SAME critique as
            # last round, the loop is stuck — a weak generator isn't acting on the
            # feedback. Bail rather than burn identical passes to max_iterations.
            fb_norm = _THINK_RE.sub("", critic_feedback).strip().lower()
            fb_norm = re.sub(r"\s+", " ", fb_norm)
            if prev_feedback_norm is not None and fb_norm == prev_feedback_norm:
                logger.info("Critique unchanged from last round — stalling out early.")
                self.stalled = True
                self._emit_orchestration(
                    phase="adversarial",
                    action="iteration_completed",
                    iteration=iteration + 1,
                    iteration_total=self.max_iterations,
                    outcome=self._iteration_outcome(),
                    verified=self.verified,
                    approved=self.approved,
                )
                return last_output
            prev_feedback_norm = fb_norm

            logger.info("Critic requested changes. Iterating...")
            if self.diff_only:
                revise_instruction = (
                    "Change ONLY what the critique explicitly named and leave "
                    "everything else byte-identical. Do not rewrite, reformat, or "
                    "touch any part the critique did not mention. Output the full "
                    "updated version."
                )
            else:
                revise_instruction = (
                    "Please carefully update the output based on this feedback."
                )
            current_prompt = (
                f"{initial_prompt}\n\n"
                f"Your last output:\n{last_output}\n\n"
                f"Critic Feedback:\n{critic_feedback}\n\n"
                f"{revise_instruction}"
            )
            if iteration + 1 < self.max_iterations:
                self._emit_orchestration(
                    phase="adversarial",
                    action="iteration_completed",
                    iteration=iteration + 1,
                    iteration_total=self.max_iterations,
                    outcome=self._iteration_outcome(),
                    verified=self.verified,
                    approved=self.approved,
                )

        logger.warning(f"Max iterations ({self.max_iterations}) reached without Critic approval.")
        if self.max_iterations > 0:
            self._emit_orchestration(
                phase="adversarial",
                action="iteration_completed",
                iteration=self.max_iterations,
                iteration_total=self.max_iterations,
                outcome=self._iteration_outcome(),
                verified=self.verified,
                approved=self.approved,
            )
        return last_output

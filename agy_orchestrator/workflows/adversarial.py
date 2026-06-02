import logging
import os
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
    if not isinstance(feedback, str):
        feedback = "" if feedback is None else str(feedback)
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
        critic_requirement: Optional[str] = None,
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
        # The requirement the critic judges against. None => use initial_prompt (the
        # generator's full prompt). The harness passes a preamble-stripped view here
        # so the critic isn't re-reading worker process-discipline boilerplate every
        # iteration (win 4). The GENERATOR still gets the full initial_prompt.
        self.critic_requirement = critic_requirement
        # Quality signals, populated by execute() for the run ledger (task #9).
        self.iterations_used = 0
        self.approved = False
        self.verified = False  # programmatic verifier passed
        self.stalled = False   # bailed early on a repeated critique
        # #50: the verifier TIMED OUT or was RESOURCE-killed (OOM) — an infra
        # condition, NOT a code defect. Regenerating can't fix a too-slow suite
        # or an oversubscribed host, so the loop bails instead of burning the
        # iteration budget against unbroken code. Surfaced for the run ledger.
        self.verifier_infra_failed = False
        self.infra_reason: Optional[str] = None
        # #58: a NON-timeout verifier failure whose stderr matched an infra-class
        # marker (missing build tool, half-installed venv, full disk, permission
        # flip, conftest import error). These markers are heuristic, so — unlike
        # the #50 timeout/OOM bail, which is precise — this is a SOFT signal: the
        # loop allows exactly ONE regenerate (a real code bug that merely *looks*
        # infra-ish still gets a fix attempt) and then bails rather than burning
        # the whole iteration budget against an unfixable host problem. Counts the
        # infra-suspected regenerate attempts already spent.
        self._infra_suspected_retries = 0
        # #65: rotate the generator's fallback chain after K consecutive VERIFY
        # failures so a model stuck in a local optimum (re-emitting the same
        # failing change with full feedback) hands the step to the next configured
        # generator instead of monopolizing all max_iterations. Only meaningful for
        # a multi-provider generator (a FallbackAgent with >1 provider); a no-op
        # otherwise. The critic chain is left untouched.
        self._consecutive_verify_fails = 0
        self.generator_rotations = 0
        try:
            self._rotate_after = max(1, int(os.environ.get("AGY_GENERATOR_ROTATE_AFTER", "2")))
        except Exception:
            self._rotate_after = 2
        self._gen_chain_len = len(getattr(generator_instance, "_chain", []) or [])
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
        if self.verifier_infra_failed:
            return self.infra_reason or "verifier_infra_failed"
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
                # #50: a verifier TIMEOUT or RESOURCE-kill is an INFRA condition,
                # not a code defect. The verifier already distinguishes these
                # (VerifierResult.timeout / .resource_exceeded) precisely so a
                # consumer won't misread them — but the old blanket "send back to
                # generator" discarded that, regenerating against unbroken code
                # until the iteration budget was exhausted and logging the same
                # opaque "verification failed" every round. Branch out instead:
                # don't regenerate (it cannot help a too-slow suite or an OOM),
                # surface the real cause distinctly, and bail this step now.
                if result.timeout or result.resource_exceeded:
                    self.verifier_infra_failed = True
                    if result.timeout:
                        self.infra_reason = "verifier_timeout"
                        logger.error(
                            "Verifier TIMED OUT (infra, not a code defect): %s. "
                            "NOT regenerating — regeneration cannot fix a timeout. "
                            "Raise AGY_TEST_TIMEOUT, or reduce host load / -n so the "
                            "baseline suite fits the budget.",
                            result.message,
                        )
                    else:
                        self.infra_reason = "verifier_resource_exceeded"
                        logger.error(
                            "Verifier was RESOURCE-killed (OOM, infra not a code "
                            "defect): %s. NOT regenerating — regeneration cannot fix "
                            "an OOM. Raise the verifier memory cap or reduce -n.",
                            result.message,
                        )
                    self._emit_orchestration(
                        phase="adversarial",
                        action="iteration_completed",
                        iteration=iteration + 1,
                        iteration_total=self.max_iterations,
                        outcome=self.infra_reason,
                        verified=self.verified,
                        approved=self.approved,
                    )
                    return last_output
                # #58: any OTHER infra-class failure (suite couldn't import because
                # a sibling left the venv half-installed, "make: command not found",
                # disk-full write) returns ok=False with a NORMAL returncode — #50's
                # precise timeout/OOM branch above doesn't catch it, so it used to
                # fall through to "regenerate" and burn the whole iteration budget
                # regenerating code against a host problem the code can't fix. The
                # verifier flags these heuristically (VerifierResult.infra_suspected).
                # Treat it like #50 — surface the real cause, distinct log/event —
                # but as a SOFT signal: the markers are heuristic, so allow exactly
                # ONE regenerate (a real bug that merely *looks* infra-ish still gets
                # a fix attempt), then bail rather than thrash to max_iterations.
                if result.infra_suspected:
                    if self._infra_suspected_retries >= 1:
                        self.verifier_infra_failed = True
                        self.infra_reason = "verifier_infra_suspected"
                        logger.error(
                            "Verifier failed with an INFRA-class error again after a "
                            "regenerate (host env, not a code defect): %s. NOT "
                            "regenerating further — fix the environment (missing build "
                            "tool, half-installed venv, disk space, permissions, "
                            "conftest import) and re-run.",
                            result.message,
                        )
                        self._emit_orchestration(
                            phase="adversarial",
                            action="iteration_completed",
                            iteration=iteration + 1,
                            iteration_total=self.max_iterations,
                            outcome=self.infra_reason,
                            verified=self.verified,
                            approved=self.approved,
                        )
                        return last_output
                    self._infra_suspected_retries += 1
                    logger.warning(
                        "Verifier failed with what looks like an INFRA-class error "
                        "(host env, not a code defect): %s. Markers are heuristic, so "
                        "allowing ONE regenerate in case it's a genuine code bug; will "
                        "bail if it recurs.",
                        result.message,
                    )
                    # Fall through to the normal regenerate path (feed the failing
                    # code + error back); the retry counter gates the next round.
                logger.info("Programmatic verification failed. Sending back to generator.")
                # #65: after K consecutive verify-failures, advance the generator's
                # fallback chain so the NEXT configured provider attempts the step —
                # a model stuck re-emitting the same failing change (with full
                # feedback) shouldn't monopolize every iteration while the other
                # generators sit idle. No-op for a single-provider generator; the
                # critic chain is untouched. Capped at chain_len-1 rotations so we
                # cycle through distinct providers rather than looping back.
                self._consecutive_verify_fails += 1
                if (self._gen_chain_len > 1
                        and self.generator_rotations < self._gen_chain_len - 1
                        and self._consecutive_verify_fails >= self._rotate_after
                        and iteration + 1 < self.max_iterations):
                    self.generator_rotations += 1
                    self._consecutive_verify_fails = 0
                    try:
                        self.generator.rotate_offset = (
                            int(getattr(self.generator, "rotate_offset", 0) or 0) + 1
                        )
                    except Exception:
                        pass
                    nxt = None
                    try:
                        chain = getattr(self.generator, "_chain", []) or []
                        if chain:
                            nxt = chain[self.generator_rotations % len(chain)].__name__
                    except Exception:
                        nxt = None
                    logger.warning(
                        "Generator failed verification %d× consecutively; rotating to "
                        "the next configured generator%s for the remaining iterations "
                        "(#65).",
                        self._rotate_after, f" ({nxt})" if nxt else "",
                    )
                    self._emit_orchestration(
                        phase="adversarial",
                        action="generator_rotation",
                        iteration=iteration + 1,
                        iteration_total=self.max_iterations,
                        to_worker=nxt,
                        generator_rotations=self.generator_rotations,
                    )
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
            critic_requirement = self.critic_requirement or initial_prompt
            critic_prompt = (
                f"Please review the following output against the original requirement.\n"
                f"Original Requirement:\n{critic_requirement}\n\n"
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
            # A critic worker can legitimately return None / a non-string: e.g. a
            # claude worker emitting {"type":"result","result":null} for an empty
            # or cancelled turn surfaces as None. Coerce here so the downstream
            # re.sub-based approval and stall checks never raise TypeError; an
            # empty reply is correctly treated as "not approved" and the loop
            # continues instead of aborting the whole dispatch.
            if not isinstance(critic_feedback, str):
                critic_feedback = "" if critic_feedback is None else str(critic_feedback)

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

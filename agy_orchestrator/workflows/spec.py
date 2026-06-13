"""FloodSpec — turn a short goal + a few constraints into a COMPLETE, buildable
system-design document (system design only; no task decomposition).

This fills the gap upstream of ``master`` mode. ``master``'s planner
(workflows/master.py) takes a raw goal and jumps straight to *decomposition*
("break this into implementation steps") — which means for a one-line goal the
planner is silently inventing the whole system design *inside* the step list,
unreviewed. FloodSpec makes that design EXPLICIT and reviewable first:

    short goal + constraints  ->  [FloodSpec]  ->  design doc  ->  [master / etc]
                                   (the WHAT,        (review gate)    (the HOW:
                                    reviewable)                        decompose + build)

The output deliberately contains NO build sequence / step list / task breakdown —
decomposition stays downstream where it belongs.

Algorithm (chosen from a literature + tooling sweep, not guessed):

* SINGLE coherent author, NOT a multi-agent writing swarm. Conversational
  multi-agent *authoring* of architecture docs measurably loses to a single
  author on consistency/clarity/feasibility (arXiv:2510.22787). The win from
  multiplicity is in *critique*, not generation.
* RUBRIC-BOUND critic, not free-form self-review. Self-Refine (arXiv:2303.17651)
  improves documents when feedback is anchored to an external rubric; free-form
  "is this good?" self-correction can *degrade* quality (arXiv:2310.01798). So
  the critic scores fixed completeness dimensions and returns located, actionable
  gaps.
* BOUNDED refine loop with a fix-point stop. Gains flatten in 1-2 rounds; we cap
  at 3 and bail early when the critique stops changing (the gap set has
  converged) — reusing AdversarialReview's stall detector.
* HONESTY RAILS against hallucinated requirements (arXiv:2505.13360: models
  silently invent ~41% of unstated requirements, fragilely). Provenance tags
  ([goal-derived]/[assumption]/[industry-default]), a mandatory Assumptions
  section, and an Open Questions section — "mark, don't assume" (GitHub Spec Kit).

The 13-section template is the consensus of AWS Kiro's design.md, GitHub Spec
Kit's plan/spec templates, and standard RFC/design-doc practice.
"""
import logging
import re
from typing import Callable, List, Optional

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.workflows.adversarial import _THINK_RE, _is_approved

logger = logging.getLogger(__name__)

# System-design sections only. Decomposition / build-ordering is explicitly NOT
# here — that is the downstream planner's job.
DEFAULT_SPEC_SECTIONS: List[str] = [
    "Overview — what is being built, the problem it solves, how this design delivers value",
    "Goals & Non-Goals — what the design must achieve; explicit non-goals to bound it",
    "Requirements — functional requirements (testable; EARS form where natural) and "
    "non-functional requirements (performance, scale, security, reliability)",
    "Architecture — high-level structure, major components, data flow",
    "Components & Interfaces — per component: responsibility, public interface/contract "
    "(inputs, outputs, types), interaction patterns",
    "Data Models — entities, typed schemas, relationships, persistence, lifecycle",
    "Error Handling & Failure Modes — what can fail and how the system responds/degrades",
    "Testing Strategy — how correctness is validated (unit/integration/e2e), tied to "
    "acceptance criteria (this is a STRATEGY, not a task list)",
    "Constraints & Guardrails — hard limits, behavioral must-never rules, platform/tech constraints",
    "Alternatives Considered — options evaluated and why rejected",
    "Out of Scope — explicitly excluded, to prevent scope drift",
    "Assumptions — every gap-filling assumption, as a falsifiable statement the reviewer can veto",
    "Open Questions — decision-relevant gaps that cannot be safely defaulted (surfaced, not invented)",
]

# Prepended to the author so it knows its job and its hard rules.
_NO_DECOMPOSITION_RULE = (
    "This document is SYSTEM DESIGN ONLY. Do NOT include a task breakdown, an "
    "implementation/build order, a numbered step list, milestones, sprints, or time "
    "estimates — decomposition happens in a later stage and must NOT appear here."
)


def _format_sections(sections: List[str]) -> str:
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(sections))


def _format_constraints(constraints: Optional[List[str]]) -> str:
    if not constraints:
        return "(none specified — derive sensible defaults and record them as Assumptions)"
    return "\n".join(f"- {c.strip()}" for c in constraints if c and c.strip())


def build_architect_prompt(
    goal: str, constraints: Optional[List[str]], sections: List[str]
) -> str:
    return (
        "You are a Lead System Architect. Produce a COMPLETE, buildable system-design "
        "document for the goal below, honoring every constraint.\n\n"
        f"{_NO_DECOMPOSITION_RULE}\n\n"
        "Write the document in Markdown with exactly these top-level sections, in order:\n"
        f"{_format_sections(sections)}\n\n"
        "Rules:\n"
        "- Be specific enough that two competent engineers reading this would build "
        "essentially the same system. Define component responsibilities, public "
        "interfaces/contracts (inputs, outputs, types), data models, and failure "
        "behavior concretely.\n"
        "- Honor EVERY constraint. Each constraint must be reflected in the design and "
        "referenced wherever it affects a component.\n"
        "- Write functional requirements as testable statements; where natural use EARS "
        "form: \"WHEN <trigger> THEN the system SHALL <response>\" / \"IF <precondition> "
        "THEN the system SHALL <response>\".\n"
        "- Tag each non-obvious design decision with its provenance: [goal-derived] "
        "(entailed by the goal/constraints), [assumption] (a gap you are filling to make "
        "the design buildable), or [industry-default] (standard practice).\n"
        "- Put every [assumption] in the Assumptions section as a falsifiable statement "
        "the reviewer can veto. Do NOT silently invent requirements: if a decision-relevant "
        "gap cannot be safely defaulted, list it in Open Questions instead of fabricating "
        "an answer.\n"
        "- Do not over-specify. Completeness means \"buildable without further clarification "
        "on decisions that matter,\" not maximal verbosity — default cosmetic/easily-changed "
        "choices and tag them rather than expanding them into prose.\n\n"
        f"## Goal\n{goal.strip()}\n\n"
        f"## Constraints\n{_format_constraints(constraints)}\n"
    )


def build_critic_prompt(
    goal: str, constraints: Optional[List[str]], draft: str
) -> str:
    return (
        "You are a meticulous design reviewer. Review the design document below for "
        "BUILDABILITY COMPLETENESS against the goal and constraints. Find genuine gaps "
        "that would block or misdirect an implementer — do not nitpick style.\n\n"
        f"## Goal\n{goal.strip()}\n\n"
        f"## Constraints\n{_format_constraints(constraints)}\n\n"
        f"## Design Document\n{draft}\n\n"
        "Evaluate these dimensions. For each gap, name the SECTION + the exact "
        "missing/violated item + the concrete fix. Skip dimensions that are fully "
        "satisfied.\n"
        "1. CONSTRAINT COVERAGE — every constraint reflected in the design AND referenced "
        "in each component it affects.\n"
        "2. INTERFACE COMPLETENESS — every component has defined inputs/outputs, "
        "types/schemas, and error/edge behavior.\n"
        "3. DATA MODEL — entities, fields, relationships, persistence, lifecycle specified.\n"
        "4. DEPENDENCIES & INTEGRATIONS — every external system has a named contract "
        "(protocol, auth, failure mode).\n"
        "5. NON-FUNCTIONAL — performance, scale, security/authz, concurrency, observability "
        "addressed or explicitly out of scope.\n"
        "6. FAILURE & EDGE BEHAVIOR — error paths, retries, partial failure, "
        "empty/boundary states.\n"
        "7. INTERNAL CONSISTENCY — no contradictions, dangling references, or interface "
        "mismatches between sections.\n"
        "8. UNAMBIGUITY / BUILDABILITY — each decision specific enough that two engineers "
        "would build the same thing.\n"
        "Also: every [goal-derived] tag must trace to the goal/constraints — flag any that "
        "are really assumptions and should be demoted to the Assumptions section. Confirm "
        "the document contains NO task breakdown or build ordering; flag it if it does.\n\n"
        "CONVERGENCE: If the document is buildable-complete — all constraints honored, no "
        "blocking gaps, assumptions and open questions explicit — reply EXACTLY with "
        "'APPROVED' and nothing else. Do not withhold approval for subjective or cosmetic "
        "preferences, and do not demand maximal verbosity. Otherwise output ONLY the "
        "prioritized list of actionable gaps."
    )


def build_revise_prompt(draft: str, critique: str, sections: List[str]) -> str:
    return (
        "You are revising your own system-design document.\n\n"
        f"Your previous document:\n{draft}\n\n"
        f"A reviewer found these gaps:\n{critique}\n\n"
        "Revise the document to close these gaps as a single coherent rewrite that "
        "preserves everything already correct. Keep the same section structure, the "
        "provenance tags ([goal-derived]/[assumption]/[industry-default]), and the "
        "Assumptions / Open Questions sections. "
        f"{_NO_DECOMPOSITION_RULE} "
        "Output the COMPLETE updated document only."
    )


class SpecWorkflow:
    """Single-author draft + rubric-bound critic loop -> a complete design doc.

    The architect drafts; the critic reviews against a fixed completeness rubric
    and either replies exactly ``APPROVED`` or returns located, actionable gaps;
    the architect revises. Bounded by ``max_iterations`` and an early bail when
    the critique stops changing (the gap set has converged). There is no
    programmatic verifier — for a design document the critic IS the oracle.
    """

    def __init__(
        self,
        architect_instance: AgentInstance,
        critic_instance: AgentInstance,
        max_iterations: int = 3,
        sections: Optional[List[str]] = None,
        draft_callback: Optional[Callable[[str, int], None]] = None,
    ):
        self.architect = architect_instance
        self.critic = critic_instance
        # 3 is the empirical sweet spot: Self-Refine gains flatten by round 2-3
        # and unbounded doc-critique loops risk oscillation (arXiv:2310.01798).
        self.max_iterations = max(1, int(max_iterations))
        self.sections = sections or list(DEFAULT_SPEC_SECTIONS)
        # Quality signals (mirrors AdversarialReview, for the run ledger).
        self.iterations_used = 0
        self.approved = False
        self.stalled = False
        # Issue #90: invoked with (draft, iterations_used) after EVERY architect
        # turn so the caller can flush the current complete draft to disk
        # incrementally — a SIGTERM / timeout mid-run then leaves the most-recent
        # draft persisted instead of zero bytes. Best-effort; never affects the loop.
        self._draft_callback = draft_callback

    def _emit_draft(self, draft: str) -> None:
        cb = self._draft_callback
        if cb is None:
            return
        try:
            cb(draft, self.iterations_used)
        except Exception:  # a persistence error must never break authoring
            logger.debug("FloodSpec draft_callback raised", exc_info=True)

    async def execute(self, goal: str, constraints: Optional[List[str]] = None) -> str:
        draft = ""
        prev_feedback_norm: Optional[str] = None

        # Round 1: author the first complete draft from the goal + constraints.
        self.architect.prompt = build_architect_prompt(goal, constraints, self.sections)
        draft = await self.architect.run_async()
        self._emit_draft(draft)

        for iteration in range(self.max_iterations):
            self.iterations_used = iteration + 1
            logger.info(
                "FloodSpec review iteration %d/%d", iteration + 1, self.max_iterations
            )

            self.critic.prompt = build_critic_prompt(goal, constraints, draft)
            critique = await self.critic.run_async()

            if _is_approved(critique):
                logger.info("Critic approved the spec — complete.")
                self.approved = True
                return draft

            # Fix-point stop: if the critique is essentially identical to last
            # round, the gap set has converged and the author isn't closing it —
            # bail rather than burn identical passes.
            fb_norm = _THINK_RE.sub("", critique).strip().lower()
            fb_norm = re.sub(r"\s+", " ", fb_norm)
            if prev_feedback_norm is not None and fb_norm == prev_feedback_norm:
                logger.info("Critique unchanged from last round — stalling out early.")
                self.stalled = True
                return draft
            prev_feedback_norm = fb_norm

            # Last allotted iteration already used: don't revise into a draft we
            # won't get to re-review. Return the current best.
            if iteration + 1 >= self.max_iterations:
                break

            logger.info("Critic requested changes. Revising spec...")
            self.architect.prompt = build_revise_prompt(draft, critique, self.sections)
            draft = await self.architect.run_async()
            self._emit_draft(draft)

        logger.warning(
            "FloodSpec reached max iterations (%d) without explicit approval.",
            self.max_iterations,
        )
        return draft

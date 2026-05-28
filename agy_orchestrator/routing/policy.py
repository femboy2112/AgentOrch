'''Rule-based router for --mode auto.

The router inspects task features and picks one of the 7 concrete modes
(direct, adversarial, feedback, cascade, master, pat, vote). Each rule
carries a one-line reason so the operator sees WHY a mode was picked.

This is a deliberately rule-based starting point. The live ledger (see
agy_orchestrator/core/calibration.py) collects the outcome signals
needed for a future learned policy; for now the rules below encode the
empirical priors from the project's experiments (docs/experiments.md)
and the mode picker in AGENTS.md.
'''
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TaskFeatures:
    instruction: str
    has_test_cmd: bool
    has_context: bool
    prompt_chars: int
    explicit_branches: Optional[int]
    generator_chain_size: int


@dataclass
class RoutingDecision:
    mode: str
    reason: str
    branches: Optional[int] = None
    max_iterations: Optional[int] = None


# Keywords (lowercase) that suggest a multi-step planning task.
_PLAN_KEYWORDS = (
    'plan', 'phases', 'multi-step', 'multistep', 'spec',
    'architecture', 'design and implement', 'whole feature',
)

# Keywords (lowercase) that suggest task ambiguity warranting K candidates.
_AMBIGUITY_KEYWORDS = (
    'unsure', 'try', 'figure out', 'explore', 'investigate',
    'open ended', 'open-ended', 'multiple approaches',
)

# Threshold for 'tiny precise task' (R3) — under this length and with no
# extra --context, feedback's faster repair loop beats pat's escalation.
_TINY_PROMPT_CHARS = 400

# Threshold above which a prompt is treated as multi-step (R1/R5 input).
_PLAN_PROMPT_CHARS = 1200


def _looks_like_plan(instruction: str, prompt_chars: int) -> bool:
    low = instruction.lower()
    if prompt_chars > _PLAN_PROMPT_CHARS:
        return True
    return any(kw in low for kw in _PLAN_KEYWORDS)


def _looks_ambiguous(instruction: str, explicit_branches: Optional[int]) -> bool:
    if explicit_branches is not None and explicit_branches > 1:
        return True
    low = instruction.lower()
    return any(kw in low for kw in _AMBIGUITY_KEYWORDS)


class RoutingPolicy:
    def choose(self, task: TaskFeatures) -> RoutingDecision:
        is_plan = _looks_like_plan(task.instruction, task.prompt_chars)
        ambiguous = _looks_ambiguous(task.instruction, task.explicit_branches)

        # R1: no test_cmd + plan-shaped → master
        if not task.has_test_cmd and is_plan:
            return RoutingDecision(
                mode='master',
                reason='No verifier; multi-step language detected — master mode plans the work',
            )

        # R2: no test_cmd → adversarial (default for unverified)
        if not task.has_test_cmd:
            return RoutingDecision(
                mode='adversarial',
                reason='No verifier; adversarial gen+critic loop is the strongest non-test gate',
            )

        # R3: test_cmd + tiny precise → feedback
        if (task.prompt_chars < _TINY_PROMPT_CHARS and not task.has_context):
            return RoutingDecision(
                mode='feedback',
                reason='Tiny precise task with verifier — feedback loop is fast and direct',
            )

        # R4: test_cmd + ambiguity → vote K
        if ambiguous:
            k = max(3, task.explicit_branches or 3)
            return RoutingDecision(
                mode='vote',
                branches=k,
                reason='Test-graded ambiguity — vote K candidates in isolated workspaces',
            )

        # R5: test_cmd + plan-shaped → pat (master's planning available on escalation)
        if is_plan:
            return RoutingDecision(
                mode='pat',
                reason='Multi-step task with verifier — PaT skips plan tax on easy paths',
            )

        # R6: test_cmd, general → pat
        return RoutingDecision(
            mode='pat',
            reason='Default with verifier: PaT tries direct first, escalates only on failure',
        )


def from_dispatch_args(
    *,
    instruction: str,
    context: Optional[str],
    test_cmd: Optional[str],
    branches: int,
    generator_chain: List[str],
) -> TaskFeatures:
    '''Build TaskFeatures from the kwargs harness.dispatch.dispatch_async accepts.'''
    return TaskFeatures(
        instruction=instruction,
        has_test_cmd=bool(test_cmd) and bool(test_cmd.strip()),
        has_context=bool(context) and bool(context.strip()),
        prompt_chars=len(instruction),
        # CLI default is branches=3; treat anything strictly above that as
        # operator-explicit (the operator passed --branches N to the CLI).
        explicit_branches=branches if branches > 3 else None,
        generator_chain_size=len(generator_chain),
    )


# Ordered for introspection: docs, tests, and a future 'why-pick' diagnostic
# can read this without re-implementing the rule order.
AUTO_DECISIONS: Tuple[Tuple[str, str], ...] = (
    ('master', 'R1: no verifier + multi-step language'),
    ('adversarial', 'R2: no verifier, general'),
    ('feedback', 'R3: test_cmd + tiny precise'),
    ('vote', 'R4: test_cmd + ambiguity / explicit branches'),
    ('pat', 'R5: test_cmd + multi-step language'),
    ('pat', 'R6: test_cmd, general default'),
)

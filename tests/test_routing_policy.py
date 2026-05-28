from agy_orchestrator.routing.policy import (
    AUTO_DECISIONS,
    RoutingPolicy,
    TaskFeatures,
    from_dispatch_args,
)


def test_no_test_no_plan_language_picks_adversarial() -> None:
    task = TaskFeatures(
        instruction='rename a variable in one file',
        has_test_cmd=False,
        has_context=False,
        prompt_chars=32,
        explicit_branches=None,
        generator_chain_size=2,
    )
    policy = RoutingPolicy()
    decision = policy.choose(task)

    assert decision.mode == 'adversarial'
    assert decision.reason and len(decision.reason) > 10


def test_no_test_with_plan_language_picks_master() -> None:
    task = TaskFeatures(
        instruction='Create a multi-step plan and implement phase one',
        has_test_cmd=False,
        has_context=False,
        prompt_chars=50,
        explicit_branches=None,
        generator_chain_size=3,
    )
    policy = RoutingPolicy()
    decision = policy.choose(task)

    assert decision.mode == 'master'
    assert decision.reason and len(decision.reason) > 10


def test_test_tiny_precise_picks_feedback() -> None:
    task = TaskFeatures(
        instruction='fix typo in README',
        has_test_cmd=True,
        has_context=False,
        prompt_chars=18,
        explicit_branches=None,
        generator_chain_size=2,
    )
    policy = RoutingPolicy()
    decision = policy.choose(task)

    assert decision.mode == 'feedback'
    assert decision.reason and len(decision.reason) > 10


def test_test_explicit_branches_picks_vote() -> None:
    task = TaskFeatures(
        instruction='fix tests for parser',
        has_test_cmd=True,
        has_context=True,
        prompt_chars=20,
        explicit_branches=5,
        generator_chain_size=3,
    )
    policy = RoutingPolicy()
    decision = policy.choose(task)

    assert decision.mode == 'vote'
    assert decision.reason and len(decision.reason) > 10
    assert decision.branches is not None and decision.branches >= 3


def test_test_ambiguity_keyword_picks_vote() -> None:
    task = TaskFeatures(
        instruction='Investigate multiple approaches for this failing test',
        has_test_cmd=True,
        has_context=True,
        prompt_chars=53,
        explicit_branches=None,
        generator_chain_size=3,
    )
    policy = RoutingPolicy()
    decision = policy.choose(task)

    assert decision.mode == 'vote'
    assert decision.reason and len(decision.reason) > 10
    assert decision.branches is not None and decision.branches >= 3


def test_test_plan_language_picks_pat() -> None:
    task = TaskFeatures(
        instruction='Design and implement a whole feature with tests',
        has_test_cmd=True,
        has_context=True,
        prompt_chars=47,
        explicit_branches=None,
        generator_chain_size=3,
    )
    policy = RoutingPolicy()
    decision = policy.choose(task)

    assert decision.mode == 'pat'
    assert decision.reason and len(decision.reason) > 10


def test_test_general_picks_pat() -> None:
    task = TaskFeatures(
        instruction='update docs and cleanup lint warnings',
        has_test_cmd=True,
        has_context=True,
        prompt_chars=37,
        explicit_branches=None,
        generator_chain_size=2,
    )
    policy = RoutingPolicy()
    decision = policy.choose(task)

    assert decision.mode == 'pat'
    assert decision.reason and len(decision.reason) > 10


def test_from_dispatch_args_round_trips_task_features() -> None:
    task = from_dispatch_args(
        instruction='fix the bug',
        context=None,
        test_cmd='pytest -q',
        branches=3,
        generator_chain=['codex', 'agy'],
    )

    assert task.has_test_cmd is True
    assert task.has_context is False
    assert task.prompt_chars == 11
    assert task.explicit_branches is None
    assert task.generator_chain_size == 2


def test_all_decisions_have_nonempty_reason() -> None:
    policy = RoutingPolicy()
    tasks = [
        TaskFeatures('simple edit', False, False, 11, None, 2),
        TaskFeatures('multi-step architecture update', False, False, 30, None, 2),
        TaskFeatures('tiny fix', True, False, 8, None, 2),
        TaskFeatures('explore alternatives', True, True, 20, None, 3),
        TaskFeatures('whole feature rollout', True, True, 20, None, 3),
        TaskFeatures('general tested task', True, True, 19, None, 2),
    ]

    for task in tasks:
        decision = policy.choose(task)
        assert decision.reason and len(decision.reason) > 10


def test_auto_decisions_constant_lists_all_modes() -> None:
    assert len(AUTO_DECISIONS) == 6
    modes = [mode for mode, _ in AUTO_DECISIONS]
    reasons = [reason for _, reason in AUTO_DECISIONS]

    assert modes == ['master', 'adversarial', 'feedback', 'vote', 'pat', 'pat']
    assert all(reason for reason in reasons)

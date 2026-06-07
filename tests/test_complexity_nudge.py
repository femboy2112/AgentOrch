from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agy_orchestrator.workflows.adversarial import (
    CATASTROPHIC_FOCUS_PREAMBLE,
    COMPLEXITY_FOCUS_PREAMBLE,
)
from harness import cli
from harness import dispatch as dispatch_mod
from harness.dispatch import COMPLEXITY_MANDATE


class _StubWorkflow:
    verified = False
    approved = True
    stalled = False
    iterations_used = 1


def _cli_result(**kwargs):
    return SimpleNamespace(
        success=True,
        run_id="r-cli",
        run_dir="runs/r-cli",
        mode=kwargs.get("mode", "adversarial"),
        generator="codex",
        critic="agy",
        duration_s=0.0,
        quality=None,
        error=None,
        changed_files=[],
        added=[],
        modified=[],
        deleted=[],
    )


def test_generator_prompt_optimize_complexity_appends_mandate(tmp_path, monkeypatch):
    captured: dict[str, str] = {}
    runs_dir = tmp_path / "runs"

    async def _stub_run_workflow(mode, prompt, **kwargs):
        captured["prompt"] = prompt
        return "ok", _StubWorkflow()

    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(dispatch_mod, "_run_workflow", _stub_run_workflow)

    dispatch_mod.dispatch(
        "implement fast lookup",
        context="use the local helper",
        spec="approved spec",
        mode="direct",
        run_id="complexity-on",
        out_dir=tmp_path / "work-on",
        optimize_complexity=True,
    )

    assert captured["prompt"].endswith(COMPLEXITY_MANDATE)
    assert (runs_dir / "complexity-on" / "prompt.txt").read_text(
        encoding="utf-8"
    ).endswith(COMPLEXITY_MANDATE)


def test_generator_prompt_optimize_complexity_off_is_byte_identical(tmp_path, monkeypatch):
    captured: dict[str, str] = {}
    runs_dir = tmp_path / "runs"
    instruction = "implement fast lookup"
    context = "use the local helper"
    spec = "approved spec"

    async def _stub_run_workflow(mode, prompt, **kwargs):
        captured["prompt"] = prompt
        return "ok", _StubWorkflow()

    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(dispatch_mod, "_run_workflow", _stub_run_workflow)

    dispatch_mod.dispatch(
        instruction,
        context=context,
        spec=spec,
        mode="direct",
        run_id="complexity-off",
        out_dir=tmp_path / "work-off",
    )

    expected = dispatch_mod._build_prompt(instruction, context, spec)
    assert COMPLEXITY_MANDATE not in captured["prompt"]
    assert captured["prompt"] == expected
    assert (
        runs_dir / "complexity-off" / "prompt.txt"
    ).read_text(encoding="utf-8") == expected


def test_critic_preamble_composes_complexity_and_mission_critical(monkeypatch):
    captured: list[str] = []

    class _FakeReview(_StubWorkflow):
        def __init__(self, *args, critic_preamble="", **kwargs):
            captured.append(critic_preamble)

        async def execute(self, prompt):
            return "ok"

    monkeypatch.setattr(dispatch_mod, "_build_role_agent_compat", lambda *a, **k: object())
    monkeypatch.setattr(dispatch_mod, "AdversarialReview", _FakeReview)

    def _run(*, optimize_complexity=False, mission_critical=False):
        asyncio.run(
            dispatch_mod._run_workflow(
                "adversarial",
                "prompt",
                run_id="r-pre",
                generator_chain=["codex"],
                critic_chain=["agy"],
                fallback=True,
                cycles=1,
                max_iterations=1,
                branches=1,
                verifier=None,
                codex_config=None,
                optimize_complexity=optimize_complexity,
                mission_critical=mission_critical,
            )
        )
        return captured.pop()

    assert _run() == ""
    assert _run(optimize_complexity=True).startswith(COMPLEXITY_FOCUS_PREAMBLE)
    assert CATASTROPHIC_FOCUS_PREAMBLE not in _run(optimize_complexity=True)
    assert _run(mission_critical=True) == CATASTROPHIC_FOCUS_PREAMBLE
    both = _run(optimize_complexity=True, mission_critical=True)
    assert COMPLEXITY_FOCUS_PREAMBLE in both
    assert CATASTROPHIC_FOCUS_PREAMBLE in both
    assert both.index(COMPLEXITY_FOCUS_PREAMBLE) < both.index(
        CATASTROPHIC_FOCUS_PREAMBLE
    )


def test_cli_optimize_complexity_defaults_false_and_forwards_true(monkeypatch):
    captured: list[dict] = []

    def _fake_dispatch(instruction, **kwargs):
        captured.append(kwargs)
        return _cli_result(**kwargs)

    monkeypatch.setattr(cli, "dispatch", _fake_dispatch)
    monkeypatch.setattr(cli, "_print_result", lambda result: None)
    monkeypatch.setattr(cli.broker_client, "is_running", lambda *a, **k: False)

    assert cli.main(["do", "plain dispatch"]) == 0
    assert captured[-1]["optimize_complexity"] is False

    assert cli.main(["do", "fast dispatch", "--optimize-complexity"]) == 0
    assert captured[-1]["optimize_complexity"] is True


def test_cli_optimize_complexity_parser_default(monkeypatch):
    captured: list[bool] = []

    def _spy(args):
        captured.append(args.optimize_complexity)
        return 0

    monkeypatch.setattr(cli, "_cmd_do", _spy)

    assert cli.main(["do", "plain dispatch"]) == 0
    assert captured[-1] is False

    assert cli.main(["do", "fast dispatch", "--optimize-complexity"]) == 0
    assert captured[-1] is True


def test_complexity_constants_are_non_empty_and_named():
    assert COMPLEXITY_MANDATE
    assert COMPLEXITY_FOCUS_PREAMBLE
    assert "complexity" in COMPLEXITY_MANDATE.lower()
    assert "complexity" in COMPLEXITY_FOCUS_PREAMBLE.lower()

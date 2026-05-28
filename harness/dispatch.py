"""Dispatch core: turn one instruction into a logged, diffed worker run.

A dispatch:
  1. builds a code-writing prompt around the instruction,
  2. snapshots the repo,
  3. runs the chosen workflow (direct / adversarial / master) with the
     fallback-wrapped role agents,
  4. snapshots again and diffs,
  5. writes everything to runs/<timestamp>/ and returns a structured result.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.ledger import build_ledger
from agy_orchestrator.execution.verifier import QualityVerifier
from agy_orchestrator.workflows.adversarial import AdversarialReview
from agy_orchestrator.workflows.cascade import CascadeWorkflow
from agy_orchestrator.workflows.master import MasterWorkflow
from agy_orchestrator.workflows.test_feedback import TestFeedbackWorkflow
from dashboard.event_bus import EventBus
from harness import roles
from harness.snapshot import diff_snapshots, take_snapshot

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"

logger = logging.getLogger("harness")
EVENT_BUS = EventBus()

WORKER_PREAMBLE = (
    "You are a coding worker operating inside an existing project repository at the "
    "current working directory. Implement the instruction below by creating and "
    "modifying files DIRECTLY on disk in this directory. Make the changes yourself — "
    "do not ask questions and do not merely describe what to do. Keep the change "
    "minimal and tightly scoped to the instruction, and match the existing code "
    "style. Do NOT run `sudo`. When finished, end your reply with a short list of the "
    "files you created or modified and a one-line reason for each."
)


@dataclass
class DispatchResult:
    run_id: str
    run_dir: str
    mode: str
    generator: str
    critic: Optional[str]
    success: bool
    duration_s: float
    changed_files: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    error: Optional[str] = None
    # Quality-cost ledger (task #9): confidence label + signals for this run.
    quality: Optional[dict] = None


def _build_prompt(instruction: str, context: Optional[str]) -> str:
    parts = [WORKER_PREAMBLE, "\n## Instruction\n" + instruction.strip()]
    if context:
        parts.append("\n## Additional context\n" + context.strip())
    return "\n".join(parts)


def _worker_hint(agent: AgentInstance, fallback_chain: Optional[List[str]]) -> str:
    if fallback_chain:
        return fallback_chain[0]
    if hasattr(agent, "_worker_name"):
        return agent._worker_name()
    return "agy"


async def _run_workflow(
    mode: str,
    prompt: str,
    *,
    generator_chain: List[str],
    critic_chain: List[str],
    fallback: bool,
    cycles: int,
    max_iterations: int,
    branches: int,
    verifier: Optional[QualityVerifier],
    codex_config: Optional[List[str]],
    post_construct_hook: Optional[roles.RolePostConstructHook] = None,
) -> tuple:
    """Run the workflow; return (output, workflow_or_None) so the caller can read
    the workflow's quality signals for the run ledger."""
    if mode == "direct":
        gen = roles.build_role_agent(
            generator_chain, prompt=prompt, fallback=fallback, cycles=cycles,
            codex_config=codex_config,
            post_construct_hook=post_construct_hook,
        )
        return await gen.run_async(), None

    if mode == "adversarial":
        gen = roles.build_role_agent(
            generator_chain, prompt=prompt, fallback=fallback, cycles=cycles,
            codex_config=codex_config,
            post_construct_hook=post_construct_hook,
        )
        critic = roles.build_role_agent(
            critic_chain, prompt="", fallback=fallback, cycles=cycles,
            codex_config=codex_config,
            post_construct_hook=post_construct_hook,
        )
        wf = AdversarialReview(gen, critic, verifier, max_iterations=max_iterations)
        return await wf.execute(prompt), wf

    if mode == "feedback":
        # Generator + programmatic verifier loop, no LLM critic. Requires a
        # --test-cmd (the strong oracle) to gate quality on real test results.
        if verifier is None:
            raise ValueError("feedback mode requires --test-cmd (the verifier oracle)")
        gen = roles.build_role_agent(
            generator_chain, prompt=prompt, fallback=fallback, cycles=cycles,
            codex_config=codex_config,
            post_construct_hook=post_construct_hook,
        )
        wf = TestFeedbackWorkflow(gen, verifier, max_iterations=max_iterations)
        return await wf.execute(prompt), wf

    if mode == "cascade":
        # Cheap-first escalation: each generator-chain token is a STAGE (cheap ->
        # strong); run the test-feedback loop per stage, escalate only on verifier
        # failure. Requires --test-cmd (the gate between stages).
        if verifier is None:
            raise ValueError("cascade mode requires --test-cmd (the escalation gate)")
        stages = [
            roles.build_role_agent([token], prompt=prompt, fallback=False,
                                   cycles=cycles, codex_config=codex_config,
                                   post_construct_hook=post_construct_hook)
            for token in generator_chain
        ]
        wf = CascadeWorkflow(stages, verifier, max_iterations_per_stage=max_iterations)
        return await wf.execute(prompt), wf

    if mode == "master":
        agent_class, model, effort = roles.build_master_agent_class(
            generator_chain, fallback=fallback, cycles=cycles,
            codex_config=codex_config,
            post_construct_hook=post_construct_hook,
        )
        wf = MasterWorkflow(
            model=model,
            effort=effort,
            branches=branches,
            max_iterations=max_iterations,
            verifier=verifier,
            agent_class=agent_class,
        )
        return await wf.execute(prompt), wf

    raise ValueError(f"unknown mode: {mode}")


async def dispatch_async(
    instruction: str,
    *,
    run_id: Optional[str] = None,
    mode: str = "adversarial",
    context: Optional[str] = None,
    generator_chain: Optional[List[str]] = None,
    critic_chain: Optional[List[str]] = None,
    fallback: bool = True,
    cycles: int = 2,
    max_iterations: int = 5,
    branches: int = 3,
    test_cmd: Optional[str] = None,
    web_search: bool = False,
    dashboard_stream_json: bool = False,
) -> DispatchResult:
    """Execute one instruction and capture the run."""
    generator_chain = generator_chain or list(roles.GENERATOR_CHAIN)
    critic_chain = critic_chain or list(roles.CRITIC_CHAIN)
    codex_config = ["tools.web_search=true"] if web_search else None

    run_id = run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    events_path.touch()

    def _sink(event: dict) -> None:
        with events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    prompt = _build_prompt(instruction, context)
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    gen_desc = roles.describe_chain(generator_chain, fallback)
    crit_desc = roles.describe_chain(critic_chain, fallback) if mode == "adversarial" else None

    def _post_construct_hook(agent: AgentInstance, worker: str, cfg: Dict[str, object]) -> None:
        model = str(cfg.get("model") or getattr(agent, "model", None) or "n/a")
        effort_val = cfg.get("effort")
        effort = str(effort_val if effort_val not in (None, "n/a") else getattr(agent, "effort", None) or "n/a")
        agent.event_callback = EVENT_BUS.publisher_for(
            run_id,
            worker=worker,
            model=model,
            effort=effort,
            branch=getattr(agent, "branch", None),
        )
        # Dashboard-only stream JSON; normal CLI path remains --output-format json.
        if worker == "claude" and hasattr(agent, "dashboard_stream_json"):
            setattr(agent, "dashboard_stream_json", bool(dashboard_stream_json))

    EVENT_BUS.add_sink(run_id, _sink)
    dispatch_pub = EVENT_BUS.publisher_for(
        run_id,
        worker=generator_chain[0],
        model="n/a",
        effort="n/a",
    )
    dispatch_pub({
        "kind": "lifecycle",
        "data": {
            "event": "dispatch_started",
            "detail": {
                "mode": mode,
                "generator_chain": generator_chain,
                "critic_chain": critic_chain,
            },
        },
    })

    # Route all tracking logs into this run's stderr.log while still showing them.
    file_handler = logging.FileHandler(run_dir / "stderr.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    logger.info("Dispatch %s | mode=%s | generator=%s%s",
                run_id, mode, gen_desc, f" | critic={crit_desc}" if crit_desc else "")

    verifier = QualityVerifier(test_commands=[test_cmd]) if test_cmd else None

    before = take_snapshot(PROJECT_ROOT)
    started = time.monotonic()
    success = True
    error: Optional[str] = None
    output = ""
    workflow = None
    try:
        output, workflow = await _run_workflow(
            mode, prompt,
            generator_chain=generator_chain, critic_chain=critic_chain,
            fallback=fallback, cycles=cycles, max_iterations=max_iterations,
            branches=branches, verifier=verifier, codex_config=codex_config,
            post_construct_hook=_post_construct_hook,
        )
    except Exception as exc:  # graceful: record, never crash the operator's shell
        success = False
        error = f"{type(exc).__name__}: {exc}"
        logger.error("Dispatch %s failed: %s", run_id, error)
    finally:
        duration = time.monotonic() - started
        dispatch_pub({
            "kind": "lifecycle",
            "data": {
                "event": "dispatch_finished",
                "detail": {"success": success, "duration_s": round(duration, 1)},
            },
        })
        EVENT_BUS.close(run_id)
        EVENT_BUS.sinks.pop(run_id, None)
        root_logger.removeHandler(file_handler)
        file_handler.close()

    # Quality-cost ledger (task #9): how much to trust this run.
    quality = build_ledger(
        workflow, mode=mode, had_verifier=verifier is not None,
        produced_output=bool(output and output.strip()),
    )
    logger.info("Dispatch %s | confidence=%s (%s)", run_id, quality["confidence"], quality["note"])

    after = take_snapshot(PROJECT_ROOT)
    diff = diff_snapshots(before, after)

    (run_dir / "stdout.log").write_text(output, encoding="utf-8")
    (run_dir / "changed-files.diff").write_text(
        diff.unified or "(no file changes detected)\n", encoding="utf-8"
    )

    result = DispatchResult(
        run_id=run_id,
        run_dir=str(run_dir),
        mode=mode,
        generator=gen_desc,
        critic=crit_desc,
        success=success,
        duration_s=round(duration, 1),
        changed_files=diff.changed,
        added=diff.added,
        modified=diff.modified,
        deleted=diff.deleted,
        error=error,
        quality=quality,
    )
    (run_dir / "meta.json").write_text(
        json.dumps(asdict(result), indent=2), encoding="utf-8"
    )
    return result


def dispatch(
    instruction: str,
    *,
    run_id: Optional[str] = None,
    mode: str = "adversarial",
    context: Optional[str] = None,
    generator_chain: Optional[List[str]] = None,
    critic_chain: Optional[List[str]] = None,
    fallback: bool = True,
    cycles: int = 2,
    max_iterations: int = 5,
    branches: int = 3,
    test_cmd: Optional[str] = None,
    web_search: bool = False,
    dashboard_stream_json: bool = False,
) -> DispatchResult:
    """Execute one instruction and capture the run. Synchronous entrypoint."""
    return asyncio.run(
        dispatch_async(
            instruction,
            run_id=run_id,
            mode=mode,
            context=context,
            generator_chain=generator_chain,
            critic_chain=critic_chain,
            fallback=fallback,
            cycles=cycles,
            max_iterations=max_iterations,
            branches=branches,
            test_cmd=test_cmd,
            web_search=web_search,
            dashboard_stream_json=dashboard_stream_json,
        )
    )

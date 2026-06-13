"""FloodSpec operator layer: ``python -m harness spec "GOAL" -c "constraint" ...``

Turns a short goal + a few constraints into a complete, buildable system-design
document and captures the run under ``runs/<id>/`` like every other dispatch.
The document is the deliverable (``runs/<id>/spec.md``); the operator reviews it
and then hands it to a build with ``do --spec runs/<id>/spec.md``.

A spec run is not a code dispatch — it writes a markdown design doc, not files
into the repo — so there is no snapshot/diff and no programmatic verifier (the
critic is the oracle; see workflows/spec.py). But it IS surfaced in the
dashboard exactly like a dispatch: it streams worker events into
``runs/<id>/events.jsonl`` (so the Live tab shows it while running) and writes a
dashboard-shaped ``meta.json`` + ``prompt.txt``/``stdout.log`` (so the Runs tab
renders it when finished).
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
from agy_orchestrator.workflows.spec import SpecWorkflow, build_architect_prompt
from dashboard.event_bus import EventBus
from harness import roles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"

logger = logging.getLogger("harness")
EVENT_BUS = EventBus()


@dataclass
class SpecResult:
    run_id: str
    run_dir: str
    spec_path: str
    # Dashboard-shaped fields (mirror DispatchResult so the Runs/Live tabs render
    # a spec run with no special-casing). ``generator`` carries the architect
    # chain; the CLI labels it "architect".
    mode: str
    generator: str
    critic: str
    success: bool
    duration_s: float
    approved: bool = False
    stalled: bool = False
    iterations_used: int = 0
    chars: int = 0
    error: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    changed_files: List[str] = field(default_factory=list)
    quality: Optional[dict] = None
    architect_author: Optional[str] = None
    architect_demoted: bool = False


async def generate_spec_async(
    goal: str,
    *,
    constraints: Optional[List[str]] = None,
    run_id: Optional[str] = None,
    architect_chain: Optional[List[str]] = None,
    critic_chain: Optional[List[str]] = None,
    fallback: bool = True,
    cycles: int = 2,
    max_iterations: int = 3,
    output_path: Optional[str] = None,
    architect_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    critic_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    watchdog_scale: float = 1.0,
    watchdog_max_bytes: Optional[int] = None,
    telegram: Optional[bool] = None,
    telegram_verbosity: Optional[str] = None,
) -> SpecResult:
    """Author a design doc for ``goal`` and capture the run.

    ``architect_chain``/``critic_chain`` default to the generator/critic role
    chains. Cross-provider by default (codex architect, agy critic) — the
    diverse-critic finding (multiplicity helps in critique, not authoring) and
    the account-sharing rule both point the same way.
    """
    constraints = [c for c in (constraints or []) if c and c.strip()]
    architect_chain = architect_chain or list(roles.GENERATOR_CHAIN)
    critic_chain = critic_chain or list(roles.CRITIC_CHAIN)

    run_id = run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    arch_desc = roles.describe_chain(architect_chain, fallback)
    crit_desc = roles.describe_chain(critic_chain, fallback)

    # Same cross-family guard the adversarial path uses — a same-family critic
    # is self-verification and under-flags (arxiv 2512.02304).
    family_warning = roles.check_chains_cross_family(architect_chain, critic_chain)
    if family_warning:
        logger.warning(family_warning)

    # --- dashboard wiring: stream events into runs/<id>/events.jsonl ---------
    events_path = run_dir / "events.jsonl"
    events_path.touch()

    def _sink(event: dict) -> None:
        with events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    EVENT_BUS.add_sink(run_id, _sink)
    telegram_notifier = None
    try:
        from harness.dispatch import _build_telegram_notifier

        telegram_notifier = _build_telegram_notifier(
            run_id=run_id,
            mode="spec",
            enabled=telegram,
            verbosity=telegram_verbosity,
            instruction=goal,
        )
        if telegram_notifier is not None:
            EVENT_BUS.add_sink(run_id, telegram_notifier)
    except Exception as exc:
        logger.debug("spec telegram notifier setup failed: %s", exc)
        telegram_notifier = None

    def _post_construct_hook(agent: AgentInstance, worker: str, cfg: Dict[str, object]) -> None:
        model = str(cfg.get("model") or getattr(agent, "model", None) or "n/a")
        effort_val = cfg.get("effort")
        effort = str(effort_val if effort_val not in (None, "n/a") else getattr(agent, "effort", None) or "n/a")
        agent.event_callback = EVENT_BUS.publisher_for(
            run_id, worker=worker, model=model, effort=effort,
        )

    architect: AgentInstance = roles.build_role_agent(
        architect_chain, prompt="", fallback=fallback, cycles=cycles,
        overrides=architect_overrides,
        watchdog_scale=watchdog_scale,
        watchdog_max_bytes=watchdog_max_bytes,
        post_construct_hook=_post_construct_hook,
    )
    critic: AgentInstance = roles.build_role_agent(
        critic_chain, prompt="", fallback=fallback, cycles=cycles,
        overrides=critic_overrides,
        watchdog_scale=watchdog_scale,
        watchdog_max_bytes=watchdog_max_bytes,
        post_construct_hook=_post_construct_hook,
    )

    wf = SpecWorkflow(architect, critic, max_iterations=max_iterations)

    # Persist inputs. prompt.txt carries a ``## Instruction`` block so the
    # dashboard's instruction preview (Live + Runs) renders the goal, plus the
    # full architect prompt for reproducibility.
    goal_blob = goal.strip()
    if constraints:
        goal_blob += "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)
    architect_prompt = build_architect_prompt(goal, constraints, wf.sections)
    (run_dir / "prompt.txt").write_text(
        "## Instruction\n" + goal_blob + "\n\n## Architect prompt\n" + architect_prompt,
        encoding="utf-8",
    )

    # Route tracking logs into this run's stderr.log (parity with dispatch).
    file_handler = logging.FileHandler(run_dir / "stderr.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    dispatch_pub = EVENT_BUS.publisher_for(
        run_id, worker=architect_chain[0], model="n/a", effort="n/a",
    )
    dispatch_pub({
        "kind": "lifecycle",
        "data": {
            "event": "dispatch_started",
            "detail": {
                "mode": "spec",
                "generator_chain": architect_chain,
                "critic_chain": critic_chain,
            },
        },
    })

    logger.info(
        "FloodSpec %s | architect=%s | critic=%s | max_iter=%d",
        run_id, arch_desc, crit_desc, wf.max_iterations,
    )

    started = time.monotonic()
    success = True
    error: Optional[str] = None
    doc = ""
    architect_author = architect_chain[0]
    architect_demoted = False

    def _author_token(agent: AgentInstance, chain: List[str]) -> str:
        lp = getattr(agent, "last_provider", None)
        if not lp:
            return chain[0]
        for tok in chain:
            try:
                if roles._class_for(tok).__name__ == lp:
                    return tok
            except Exception:
                continue
        return lp

    try:
        doc = await wf.execute(goal, constraints)
        architect_author = _author_token(architect, architect_chain)
        architect_demoted = bool(success and architect_author != architect_chain[0])
        if architect_demoted:
            logger.warning(
                "FloodSpec %s: lead architect '%s' did NOT author the doc — it was written by '%s'. "
                "The lead was likely SIGKILLed by the verbose byte-budget watchdog or hit a usage wall; "
                "raise --watchdog-scale / --watchdog-max-bytes to keep your primary architect. "
                "See runs/%s/events.jsonl.",
                run_id, architect_chain[0], architect_author, run_id,
            )
    except Exception as exc:  # graceful: capture, never crash the operator shell
        success = False
        error = f"{type(exc).__name__}: {exc}"
        logger.error("FloodSpec %s failed: %s", run_id, error)
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

    spec_path = run_dir / "spec.md"
    spec_path.write_text(doc or "(no document produced)\n", encoding="utf-8")
    # stdout.log mirrors the doc so the dashboard Stdout tab shows the deliverable
    # (spec.md is not one of the fixed artifact tabs).
    (run_dir / "stdout.log").write_text(doc or "", encoding="utf-8")
    # No file changes in a spec run; keep the Diff tab honest.
    (run_dir / "changed-files.diff").write_text(
        "(spec run — no file changes)\n", encoding="utf-8"
    )
    # Optional convenience copy into a caller-chosen location (e.g. the target
    # repo's DESIGN.md). The runs/ artifact is always the canonical one.
    if output_path and doc:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(doc, encoding="utf-8")

    approved = bool(getattr(wf, "approved", False))
    stalled = bool(getattr(wf, "stalled", False))
    confidence = "approved" if approved else "unverified"

    result = SpecResult(
        run_id=run_id,
        run_dir=str(run_dir),
        spec_path=str(spec_path),
        mode="spec",
        generator=arch_desc,
        critic=crit_desc,
        success=success,
        duration_s=round(duration, 1),
        approved=approved,
        stalled=stalled,
        iterations_used=int(getattr(wf, "iterations_used", 0)),
        chars=len(doc or ""),
        error=error,
        constraints=constraints,
        quality={
            "confidence": confidence,
            "note": (
                "critic approved" if approved
                else ("critique stalled (gap set converged)" if stalled
                      else "max iterations reached without approval")
            ),
        },
        architect_author=architect_author,
        architect_demoted=architect_demoted,
    )
    (run_dir / "meta.json").write_text(
        json.dumps(asdict(result), indent=2), encoding="utf-8"
    )
    if telegram_notifier is not None:
        try:
            telegram_notifier.finished(asdict(result))
        except Exception as exc:
            logger.debug("spec telegram finish summary failed: %s", exc)
    return result


def generate_spec(goal: str, **kwargs) -> SpecResult:
    """Synchronous entrypoint."""
    return asyncio.run(generate_spec_async(goal, **kwargs))

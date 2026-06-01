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
import hashlib
import inspect
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.core.calibration import append_live_row
from agy_orchestrator.execution.ledger import build_ledger
from agy_orchestrator.execution.verifier import QualityVerifier, VerifierResult
from agy_orchestrator.workflows.adversarial import (
    CATASTROPHIC_FOCUS_PREAMBLE,
    AdversarialReview,
)
from agy_orchestrator.workflows.cascade import CascadeWorkflow
from agy_orchestrator.workflows.master import MasterWorkflow
from agy_orchestrator.workflows.pat import PatWorkflow
from agy_orchestrator.workflows.test_feedback import TestFeedbackWorkflow
from agy_orchestrator.workflows.vote import VoteWorkflow
from dashboard.event_bus import EventBus
from harness import roles
from harness.run_monitor import Notifier, RunMonitor, RunStalled
from harness.snapshot import diff_snapshots, take_snapshot

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
# Stable, instruction-keyed master/pat checkpoints (issue #31, salvage-on-death).
# A long master run that dies abruptly mid-step (e.g. systemd scope exit-144 from
# a worker's stray pkill) leaves completed-step edits on disk; this directory lets
# a re-dispatch of the SAME instruction resume from the last completed step instead
# of restarting. Under runs/ (gitignored), separate from per-run runs/<id>/ logs so
# the path is stable across run_ids.
CHECKPOINT_DIR = RUNS_DIR / ".checkpoints"

logger = logging.getLogger("harness")
EVENT_BUS = EventBus()


def _master_checkpoint_path(prompt: str) -> str:
    """Deterministic checkpoint file for a master/pat run, keyed by the full prompt.

    Same instruction (same built prompt) -> same path, so an abruptly-killed run
    is resumable simply by re-dispatching it (issue #31, fix option 4).
    MasterWorkflow refuses to resume a checkpoint whose key doesn't match the
    prompt, and refuses once completed==len(tasks) — so a *completed* or a
    *different* instruction starts clean. The checkpoint is removed on successful
    completion (see MasterWorkflow.execute), so this directory holds only the
    in-flight / salvageable runs.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return str(CHECKPOINT_DIR / f"{key}.json")

def _glob_to_regex(glob: str) -> "re.Pattern[str]":
    """Compile a path glob to an anchored regex (issue #38).

    Git-style semantics on POSIX-relative paths: ``*`` matches within a single
    path segment, ``**`` spans directories, ``**/`` matches zero or more leading
    directories (so ``**/*.lock`` matches ``a/b/c.lock`` AND a root ``c.lock``),
    ``?`` matches one non-slash char. Everything else is literal.
    """
    g = glob.strip()
    if g.startswith("./"):
        g = g[2:]
    out: List[str] = []
    i, n = 0, len(g)
    while i < n:
        if g.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif g.startswith("**", i):
            out.append(".*")
            i += 2
        elif g[i] == "*":
            out.append("[^/]*")
            i += 1
        elif g[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(g[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def evaluate_path_policy(
    paths: List[str],
    protect_globs: Optional[List[str]] = None,
    allow_globs: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Return path-policy violations for a change set (issue #38).

    A path violates the policy if it matches any ``protect`` (denylist) glob, or
    if any ``allow`` (allowlist) globs are given and it matches NONE of them.
    Returns ``[{"path", "reason"}, ...]`` (empty == clean). Matching is on
    repo-relative POSIX paths.
    """
    protect_pats = [(_glob_to_regex(g), g) for g in (protect_globs or []) if g.strip()]
    allow_pats = [_glob_to_regex(g) for g in (allow_globs or []) if g.strip()]
    violations: List[Dict[str, str]] = []
    for p in paths:
        # Normalize to POSIX separators so globs match regardless of the host's
        # os.sep (snapshot paths use os.sep; treat '\' as a separator too).
        norm = p.replace(os.sep, "/").replace("\\", "/")
        hit = next((g for pat, g in protect_pats if pat.match(norm)), None)
        if hit is not None:
            violations.append({"path": p, "reason": f"matches protected path glob '{hit}'"})
            continue
        if allow_pats and not any(pat.match(norm) for pat in allow_pats):
            violations.append({"path": p, "reason": "outside every --allow-paths glob"})
    return violations


WORKER_PREAMBLE = (
    "You are a coding worker operating inside an existing project repository at the "
    "current working directory. Implement the instruction below by creating and "
    "modifying files DIRECTLY on disk in this directory. Make the changes yourself — "
    "do not ask questions and do not merely describe what to do. Keep the change "
    "minimal and tightly scoped to the instruction, and match the existing code "
    "style. Do NOT run `sudo`. "
    # Process discipline (issue #31): a worker that runs the project's full,
    # long-running test/build gate ITSELF contends with the harness's own
    # --test-cmd verifier on lock-guarded parallel gates, accumulates "stale"
    # jobs, and then "helpfully" pkills them by name — a pattern that matches
    # and kills the orchestrator process running the worker (scope exit-144).
    # The harness owns verification; the worker must neither re-run the heavy
    # gate nor pkill by name.
    "A separate automated harness runs the project's full test/build gate to verify "
    "your work after your turn — do NOT run the full or long-running test suite or "
    "build gate yourself (e.g. `make check`, the complete CI suite); run only the "
    "narrow, fast checks needed to validate your specific change. NEVER kill a "
    "process you did not directly start, and never use `pkill`, `killall`, or `kill` "
    "by process name or pattern — such a pattern can match and take down the "
    "orchestrator that is running you; if a process you spawned hangs, kill only its "
    "exact PID. "
    "When finished, end your reply with a short list of the "
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
    # Per-dispatch token telemetry rollup from usage events.
    tokens: Optional[dict] = None
    # Path-policy guard (#38): change-set entries that violated --protect-paths /
    # --allow-paths. Non-empty forces success=False so a downstream consumer can
    # gate on meta.json instead of a human reading the diff.
    protect_violations: List[Dict[str, str]] = field(default_factory=list)
    # Run-watchdog outcome (#40): set to "stalled" when --run-stall-abort fired
    # (no run-level forward progress within the window). None for a normal run.
    run_outcome: Optional[str] = None


def _as_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        iv = int(value)
    except Exception:
        return None
    return iv if iv >= 0 else None


def _summarize_token_usage(events_path: Path) -> dict:
    per_worker: Dict[str, Dict[str, Any]] = {}
    total_calls = 0
    for raw in events_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if not isinstance(event, dict) or event.get("kind") != "usage":
            continue
        data = event.get("data")
        if not isinstance(data, dict) or data.get("usage_kind") != "call":
            continue
        worker = str(event.get("worker") or data.get("worker") or "unknown")
        model = str(event.get("model") or data.get("model") or "n/a")
        token_source = str(data.get("token_source") or "unavailable")
        token_source = token_source if token_source in {"cli", "unavailable"} else "unavailable"
        row = per_worker.setdefault(
            worker,
            {
                "calls": 0,
                "cli_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "total_tokens": 0,
                "has_input": False,
                "has_output": False,
                "has_cache": False,
                "has_total": False,
                "models": set(),
            },
        )
        total_calls += 1
        row["calls"] += 1
        if token_source == "cli":
            row["cli_calls"] += 1
        row["models"].add(model)
        input_tokens = _as_int(data.get("input_tokens"))
        output_tokens = _as_int(data.get("output_tokens"))
        cache_read_tokens = _as_int(data.get("cache_read_tokens"))
        total_tokens = _as_int(data.get("total_tokens"))
        if input_tokens is not None:
            row["input_tokens"] += input_tokens
            row["has_input"] = True
        if output_tokens is not None:
            row["output_tokens"] += output_tokens
            row["has_output"] = True
        if cache_read_tokens is not None:
            row["cache_read_tokens"] += cache_read_tokens
            row["has_cache"] = True
        if total_tokens is not None:
            row["total_tokens"] += total_tokens
            row["has_total"] = True

    out_per_worker: Dict[str, Dict[str, Any]] = {}
    total_input = 0
    total_output = 0
    total_cache = 0
    total_total = 0
    has_total_input = False
    has_total_output = False
    has_total_cache = False
    has_total_total = False
    for worker in sorted(per_worker):
        row = per_worker[worker]
        calls = int(row["calls"])
        cli_calls = int(row["cli_calls"])
        token_source = "unavailable"
        if cli_calls == calls and calls > 0:
            token_source = "cli"
        elif 0 < cli_calls < calls:
            token_source = "mixed"
        input_tokens = row["input_tokens"] if row["has_input"] else None
        output_tokens = row["output_tokens"] if row["has_output"] else None
        cache_read_tokens = row["cache_read_tokens"] if row["has_cache"] else None
        total_tokens = row["total_tokens"] if row["has_total"] else None
        if input_tokens is not None:
            total_input += int(input_tokens)
            has_total_input = True
        if output_tokens is not None:
            total_output += int(output_tokens)
            has_total_output = True
        if cache_read_tokens is not None:
            total_cache += int(cache_read_tokens)
            has_total_cache = True
        if total_tokens is not None:
            total_total += int(total_tokens)
            has_total_total = True
        out_per_worker[worker] = {
            "calls": calls,
            "token_source": token_source,
            "models": sorted(row["models"]),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "total_tokens": total_tokens,
        }

    return {
        "total_calls": total_calls,
        "per_worker": out_per_worker,
        "grand_total": {
            "input_tokens": total_input if has_total_input else None,
            "output_tokens": total_output if has_total_output else None,
            "cache_read_tokens": total_cache if has_total_cache else None,
            "total_tokens": total_total if has_total_total else None,
        },
    }


def _build_prompt(
    instruction: str, context: Optional[str], spec: Optional[str] = None
) -> str:
    parts = [WORKER_PREAMBLE, "\n## Instruction\n" + instruction.strip()]
    if spec:
        # An approved FloodSpec design doc: the authoritative source of truth.
        # Placed before free-form context so it dominates the worker's framing —
        # and, for master mode, so the planner decomposes THIS design rather than
        # re-inventing one from the bare instruction.
        parts.append(
            "\n## Approved design specification (authoritative)\n"
            "Build the system described by this specification. It has been reviewed "
            "and approved; treat it as the source of truth for architecture, "
            "interfaces, data models, and constraints. Do not redesign it — "
            "implement it.\n\n" + spec.strip()
        )
    if context:
        parts.append("\n## Additional context\n" + context.strip())
    return "\n".join(parts)


def _worker_hint(agent: AgentInstance, fallback_chain: Optional[List[str]]) -> str:
    if fallback_chain:
        return fallback_chain[0]
    if hasattr(agent, "_worker_name"):
        return agent._worker_name()
    return "agy"


def _derive_verifier_delta(
    baseline_result: Optional[VerifierResult], final_verified: bool
) -> Optional[str]:
    """Classify quality movement between pre-run baseline and final verifier signal."""
    if baseline_result is None:
        return None
    baseline_ok = baseline_result.ok
    if baseline_ok and final_verified:
        return "preserved"
    if baseline_ok and not final_verified:
        return "regressed"
    if not baseline_ok and final_verified:
        return "fixed"
    return "unchanged"


def _build_role_agent_compat(
    chain: List[str],
    *,
    prompt: str,
    fallback: bool,
    cycles: int,
    codex_config: Optional[List[str]],
    computer_use_config: Optional[Dict[str, Any]],
    post_construct_hook: Optional[roles.RolePostConstructHook],
) -> AgentInstance:
    """Call roles.build_role_agent while staying compatible with patched tests.

    Some unit tests monkeypatch ``roles.build_role_agent`` with a legacy
    signature that predates ``computer_use_config``. Pass that kwarg only when
    the callee can accept it.
    """
    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "fallback": fallback,
        "cycles": cycles,
        "codex_config": codex_config,
        "post_construct_hook": post_construct_hook,
    }
    accepts_computer_use_config = False
    try:
        sig = inspect.signature(roles.build_role_agent)
        accepts_computer_use_config = (
            "computer_use_config" in sig.parameters
            or any(
                p.kind is inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
        )
    except (TypeError, ValueError):
        accepts_computer_use_config = False
    if accepts_computer_use_config:
        kwargs["computer_use_config"] = computer_use_config
    return roles.build_role_agent(chain, **kwargs)


async def _run_workflow(
    mode: str,
    prompt: str,
    *,
    run_id: str,
    generator_chain: List[str],
    critic_chain: List[str],
    fallback: bool,
    cycles: int,
    max_iterations: int,
    branches: int,
    verifier: Optional[QualityVerifier],
    codex_config: Optional[List[str]],
    computer_use_config: Optional[Dict[str, Any]] = None,
    post_construct_hook: Optional[roles.RolePostConstructHook] = None,
    working_directory: str = ".",
    mission_critical: bool = False,
    baseline_result: Optional[VerifierResult] = None,
    candidate_setup: Optional[str] = None,
    resume_policy: str = "auto",
    plan_only: bool = False,
) -> tuple:
    """Run the workflow; return (output, workflow_or_None) so the caller can read
    the workflow's quality signals for the run ledger."""
    # Plan-only / dry-run (#41): for master/pat, run JUST the planner and stop
    # before any worker mutates the out-dir. pat's "plan" is the master planner,
    # so both modes route through a plan-only MasterWorkflow — no verifier needed
    # (the direct Stage-1 attempt, which would write, is skipped).
    if plan_only and mode in ("master", "pat"):
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
            working_directory=working_directory,
            plan_only=True,
            event_callback=EVENT_BUS.publisher_for(
                run_id, worker="orchestrator", model=model, effort=effort, branch=None,
            ),
        )
        return await wf.execute(prompt), wf

    if mode == "direct":
        gen = _build_role_agent_compat(
            generator_chain,
            prompt=prompt,
            fallback=fallback,
            cycles=cycles,
            codex_config=codex_config,
            computer_use_config=computer_use_config,
            post_construct_hook=post_construct_hook,
        )
        return await gen.run_async(), None

    if mode == "adversarial":
        gen = _build_role_agent_compat(
            generator_chain,
            prompt=prompt,
            fallback=fallback,
            cycles=cycles,
            codex_config=codex_config,
            computer_use_config=computer_use_config,
            post_construct_hook=post_construct_hook,
        )
        critic = _build_role_agent_compat(
            critic_chain,
            prompt="",
            fallback=fallback,
            cycles=cycles,
            codex_config=codex_config,
            computer_use_config=computer_use_config,
            post_construct_hook=post_construct_hook,
        )
        model = str(getattr(gen, "model", None) or "n/a")
        effort = str(getattr(gen, "effort", None) or "n/a")
        wf = AdversarialReview(gen, critic, verifier, max_iterations=max_iterations,
                               working_directory=working_directory,
                               critic_preamble=(CATASTROPHIC_FOCUS_PREAMBLE if mission_critical else ""),
                               event_callback=EVENT_BUS.publisher_for(
                                   run_id,
                                   worker="orchestrator",
                                   model=model,
                                   effort=effort,
                                   branch=None,
                               ))
        return await wf.execute(prompt), wf

    if mode == "feedback":
        # Generator + programmatic verifier loop, no LLM critic. Requires a
        # --test-cmd (the strong oracle) to gate quality on real test results.
        if verifier is None:
            raise ValueError("feedback mode requires --test-cmd (the verifier oracle)")
        gen = _build_role_agent_compat(
            generator_chain,
            prompt=prompt,
            fallback=fallback,
            cycles=cycles,
            codex_config=codex_config,
            computer_use_config=computer_use_config,
            post_construct_hook=post_construct_hook,
        )
        wf = TestFeedbackWorkflow(gen, verifier, max_iterations=max_iterations,
                                  working_directory=working_directory)
        return await wf.execute(prompt), wf

    if mode == "cascade":
        # Cheap-first escalation: each generator-chain token is a STAGE (cheap ->
        # strong); run the test-feedback loop per stage, escalate only on verifier
        # failure. Requires --test-cmd (the gate between stages).
        if verifier is None:
            raise ValueError("cascade mode requires --test-cmd (the escalation gate)")
        stages = [
            _build_role_agent_compat(
                [token],
                prompt=prompt,
                fallback=False,
                cycles=cycles,
                codex_config=codex_config,
                computer_use_config=computer_use_config,
                post_construct_hook=post_construct_hook,
            )
            for token in generator_chain
        ]
        wf = CascadeWorkflow(stages, verifier, max_iterations_per_stage=max_iterations,
                             working_directory=working_directory)
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
            working_directory=working_directory,
            checkpoint_path=_master_checkpoint_path(prompt),
            resume_policy=resume_policy,
            event_callback=EVENT_BUS.publisher_for(
                run_id,
                worker="orchestrator",
                model=model,
                effort=effort,
                branch=None,
            ),
        )
        return await wf.execute(prompt), wf

    if mode == "vote":
        # K-parallel candidates in isolated workspaces; verifier picks the
        # winner. K = `branches` (reusing the existing CLI knob). Each
        # candidate rotates through the generator_chain so K=3 with the
        # default chain (codex,agy,grok) produces one candidate per
        # provider — the heterogeneity gain (arxiv 2602.03794).
        if verifier is None:
            raise ValueError("vote mode requires --test-cmd (the verifier gate)")
        k = max(1, branches)
        vote_generators: List[AgentInstance] = []
        for i in range(k):
            token = generator_chain[i % len(generator_chain)]
            # Each slot is its own single-worker agent (no fallback chain
            # inside a slot — diversity comes from different slots, not
            # from fallback within one slot).
            slot_agent = _build_role_agent_compat(
                [token],
                prompt=prompt,
                fallback=False,
                cycles=cycles,
                codex_config=codex_config,
                computer_use_config=computer_use_config,
                post_construct_hook=post_construct_hook,
            )
            vote_generators.append(slot_agent)
        wf = VoteWorkflow(
            generators=vote_generators,
            verifier=verifier,
            working_directory=working_directory,
            # Reuse the baseline gate the harness just ran on the unchanged base
            # tree (#33) — lets the preflight skip a redundant full-suite re-run.
            baseline_ok=(baseline_result.ok if baseline_result is not None else None),
            # Per-candidate env bootstrap (#34): makes vote isolation sound on
            # editable-install repos (each candidate gets its own venv).
            candidate_setup=candidate_setup,
        )
        return await wf.execute(prompt), wf

    if mode == "pat":
        # Plan-after-Trial: direct generator attempt gated by verifier;
        # on failure, escalate to master mode. Verifier is mandatory.
        if verifier is None:
            raise ValueError("pat mode requires --test-cmd (the Stage 1 verifier gate)")
        direct_gen = _build_role_agent_compat(
            generator_chain,
            prompt=prompt,
            fallback=fallback,
            cycles=cycles,
            codex_config=codex_config,
            computer_use_config=computer_use_config,
            post_construct_hook=post_construct_hook,
        )
        agent_class, model, effort = roles.build_master_agent_class(
            generator_chain, fallback=fallback, cycles=cycles,
            codex_config=codex_config,
            post_construct_hook=post_construct_hook,
        )
        master_wf = MasterWorkflow(
            model=model,
            effort=effort,
            branches=branches,
            max_iterations=max_iterations,
            verifier=verifier,
            agent_class=agent_class,
            working_directory=working_directory,
            checkpoint_path=_master_checkpoint_path(prompt),
            resume_policy=resume_policy,
            event_callback=EVENT_BUS.publisher_for(
                run_id,
                worker="orchestrator",
                model=model,
                effort=effort,
                branch=None,
            ),
        )
        wf = PatWorkflow(
            direct_generator=direct_gen,
            master_workflow=master_wf,
            verifier=verifier,
            working_directory=working_directory,
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
    verifier_mem_max: Optional[str] = None,
    web_search: bool = False,
    mission_critical: bool = False,
    spec: Optional[str] = None,
    dashboard_stream_json: bool = False,
    out_dir: Optional[Union[str, Path]] = None,
    candidate_setup: Optional[str] = None,
    resume_policy: str = "auto",
    protect_paths: Optional[List[str]] = None,
    allow_paths: Optional[List[str]] = None,
    plan_only: bool = False,
    # Run-level watchdog / heartbeat / notify (#40)
    run_stall_abort: Optional[float] = None,
    notify: Optional[str] = None,
    notify_cmd: Optional[str] = None,
    heartbeat_interval: Optional[float] = None,
    # Step 12: computer-use worker params (forwarded to adapter when generator=computer-use)
    # Step 10: real-gui harness wiring (flags only; absent keeps cu_req construction byte-identical)
    computer_use_mode: Optional[str] = None,
    computer_use_task_priority: Optional[str] = None,
    computer_use_budgets: Optional[Dict[str, Any]] = None,
    # real_gui_policy / ask_mode per instruction: bare names (match backend RunRequest/WorkerSession)
    real_gui_policy: Optional[str] = None,
    ask_mode: Optional[str] = None,
    browser_engine: Optional[str] = None,
    browser_display: Optional[str] = None,
) -> DispatchResult:
    """Execute one instruction and capture the run.

    ``spec`` is an approved FloodSpec design document (the text, not a path);
    when present it is injected as the authoritative design the worker must
    implement — for master mode the planner decomposes it instead of the bare
    instruction.

    ``out_dir`` is the directory the worker subprocess runs in (its cwd) and
    the scope of the before/after snapshot diff. Defaults to AgentOrch's own
    repo root, which is the historical behaviour. Set it when invoking
    AgentOrch from another repo so the worker writes there instead of into
    AgentOrch. The ``runs/<id>/`` artifacts always live under AgentOrch
    (they're orchestrator-internal logs, not user data).
    """
    generator_chain = generator_chain or list(roles.GENERATOR_CHAIN)
    critic_chain = critic_chain or list(roles.CRITIC_CHAIN)
    codex_config = ["tools.web_search=true"] if web_search else None
    cu_config: Dict[str, Any] = {
        "mode": computer_use_mode or "ISOLATED",
        "task_priority": computer_use_task_priority or "normal",
        "budgets": computer_use_budgets,
    }
    if real_gui_policy is not None:
        cu_config["real_gui_policy"] = real_gui_policy
    if ask_mode is not None:
        cu_config["ask_mode"] = ask_mode
    if browser_engine is not None:
        cu_config["browser_engine"] = browser_engine
    if browser_display is not None:
        cu_config["browser_display"] = browser_display
    # Step 12: recognize computer-use as standard worker token (no LLM path)
    _is_cu = bool(generator_chain) and generator_chain[0] == roles.COMPUTER_USE_TOKEN
    if mode == "auto":
        from agy_orchestrator.routing.policy import RoutingPolicy, from_dispatch_args
        task = from_dispatch_args(
            instruction=instruction,
            context=context,
            test_cmd=test_cmd,
            branches=branches,
            generator_chain=generator_chain,
        )
        decision = RoutingPolicy().choose(task)
        logger.info("Auto-routing: mode=%s — %s", decision.mode, decision.reason)
        mode = decision.mode
        if decision.branches is not None:
            branches = decision.branches
        if decision.max_iterations is not None:
            max_iterations = decision.max_iterations

    # Where the worker actually writes files. Default = AgentOrch repo root,
    # which preserves the prior behaviour exactly.
    work_dir = Path(out_dir).expanduser().resolve() if out_dir else PROJECT_ROOT
    work_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    events_path.touch()

    # The run monitor (#40) is constructed below but referenced from _sink so it
    # can observe run-level forward progress on every event. Use a 1-slot holder
    # to keep the closure simple without reordering the surrounding setup.
    _monitor_holder: List[RunMonitor] = []

    def _sink(event: dict) -> None:
        if _monitor_holder:
            try:
                _monitor_holder[0].observe(event)
            except Exception:
                pass
        with events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    prompt = _build_prompt(instruction, context, spec)
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    gen_desc = roles.describe_chain(
        generator_chain,
        fallback,
        real_gui_policy=real_gui_policy,
        ask_mode=ask_mode,
        browser_engine=browser_engine,
        browser_display=browser_display,
    )
    crit_desc = (
        roles.describe_chain(
            critic_chain,
            fallback,
            real_gui_policy=real_gui_policy,
            ask_mode=ask_mode,
            browser_engine=browser_engine,
            browser_display=browser_display,
        )
        if mode == "adversarial"
        else None
    )

    # Cross-family verifier guard: warn (don't block) when the critic chain
    # leads with the same provider family as the generator. Only meaningful
    # for adversarial mode — other modes don't use a separate critic chain.
    if mode == "adversarial":
        family_warning = roles.check_chains_cross_family(generator_chain, critic_chain)
        if family_warning:
            logger.warning(family_warning)
    if mode in ("vote", "tot"):
        agy_warning = roles.check_agy_parallelism_warning(mode, generator_chain, branches)
        if agy_warning:
            logger.warning(agy_warning)

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
        # Pin the worker's cwd to the operator-chosen output directory so it
        # doesn't pollute AgentOrch when invoked from elsewhere. Only set when
        # work_dir is not the repo root; staying ``None`` keeps the historical
        # inherit-parent-cwd behaviour for the default case.
        if work_dir != PROJECT_ROOT:
            agent.cwd = str(work_dir)
        # Forward the full computer-use config for shim-based execution paths.
        # This keeps direct adapter short-circuiting unchanged while ensuring
        # --mode workflows that construct ComputerUseShim still receive all
        # CLI-forwarded RunRequest keys.
        if worker == roles.COMPUTER_USE_TOKEN and hasattr(agent, "computer_use_config"):
            setattr(agent, "computer_use_config", dict(cu_config))
            setattr(agent, "_harness_run_id", run_id)
            setattr(agent, "_harness_events_path", str(events_path))

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

    # Run-level watchdog + heartbeat + notify (#40). Heartbeats publish through
    # the bus (so they persist to events.jsonl AND reach the dashboard) on a
    # dedicated publisher; the monitor observes every event via _sink to track
    # run-level forward progress (chatter doesn't count) and the current step.
    if heartbeat_interval is None:
        heartbeat_interval = float(os.environ.get("AGY_HEARTBEAT_SECONDS", "30") or 0)
    notify_url: Optional[str] = None
    notify_command: Optional[str] = notify_cmd
    if notify:
        if notify.startswith(("http://", "https://")):
            notify_url = notify
        else:
            notify_command = notify_command or notify
    notifier = Notifier(webhook=notify_url, command=notify_command, run_id=run_id)
    heartbeat_pub = EVENT_BUS.publisher_for(
        run_id, worker="run-monitor", model="n/a", effort="n/a",
    )
    monitor = RunMonitor(
        run_id=run_id,
        emit=heartbeat_pub,
        notifier=notifier,
        work_dir=str(work_dir),
        heartbeat_interval=heartbeat_interval or 0.0,
        run_stall_abort=run_stall_abort,
    )
    _monitor_holder.append(monitor)
    monitor.notify("start", extra={"mode": mode})

    # Route all tracking logs into this run's stderr.log while still showing them.
    file_handler = logging.FileHandler(run_dir / "stderr.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    logger.info("Dispatch %s | mode=%s | generator=%s%s",
                run_id, mode, gen_desc, f" | critic={crit_desc}" if crit_desc else "")

    if test_cmd:
        # Pass mem_max only when a cap is actually requested, so the common path
        # (and test doubles that swap in a narrower QualityVerifier) keep their
        # original constructor signature.
        _vkwargs = {"test_commands": [test_cmd]}
        if verifier_mem_max:
            _vkwargs["mem_max"] = verifier_mem_max
        verifier = QualityVerifier(**_vkwargs)
    else:
        verifier = None
    baseline_result: Optional[VerifierResult] = None
    if verifier is not None:
        try:
            baseline_result = await verifier.verify(working_directory=str(work_dir))
            if baseline_result.ok:
                logger.info("Baseline verifier: ok=True")
            else:
                logger.info(
                    "Baseline verifier: ok=False (returncode=%s, error_hash=%s)",
                    baseline_result.returncode,
                    baseline_result.error_hash,
                )
        except Exception as exc:
            logger.warning(
                "Baseline verifier failed (continuing): %s: %s",
                type(exc).__name__,
                exc,
            )
            baseline_result = None

    before = take_snapshot(work_dir)
    started = time.monotonic()
    success = True
    error: Optional[str] = None
    output = ""
    workflow = None
    run_outcome: Optional[str] = None
    try:
        if _is_cu:
            # Step 12 minimal glue: exercise the real adapter (writes events.jsonl under runs/<id>/,
            # respects --out-dir via harness snapshot scope, supports cu params).
            # Only direct-style runs are exercised here; complex workflows stay LLM-only.
            from agy_orchestrator.computer_use.adapter import ComputerUseWorkerAdapter
            from agy_orchestrator.computer_use.audit import AuditEventSink as CUAuditSink

            def _cu_sink(rid: str):
                # Append to the harness-prepared events.jsonl and fan out to bus for dashboard.
                s = CUAuditSink(run_id=rid, events_path=events_path)
                pub = EVENT_BUS.publisher_for(rid, worker="computer-use", model="n/a", effort="n/a")
                s.add_callback(lambda d: pub({"kind": "computer_use_event", "data": d}) or None)
                return s

            adapter = ComputerUseWorkerAdapter(audit_sink_factory=_cu_sink)
            cu_req = {
                "run_id": run_id,
                "objective": instruction,  # raw objective for the reasoner; prompt wrapper is in artifacts
                **cu_config,
            }
            h = adapter.start(cu_req)
            output = f"computer-use:{h.status} run_id={h.run_id} events={h.events_path or ''}"
            # workflow stays None (no quality ledger from cu yet)
        else:
            # Supervise the workflow with the run-level watchdog + heartbeat
            # (#40). When neither is active monitor.run awaits the coroutine
            # directly, so the default path is byte-for-byte unchanged.
            output, workflow = await monitor.run(_run_workflow(
                mode, prompt,
                run_id=run_id,
                generator_chain=generator_chain, critic_chain=critic_chain,
                fallback=fallback, cycles=cycles, max_iterations=max_iterations,
                branches=branches, verifier=verifier, codex_config=codex_config,
                computer_use_config=cu_config,
                post_construct_hook=_post_construct_hook,
                working_directory=str(work_dir),
                mission_critical=mission_critical,
                baseline_result=baseline_result,
                candidate_setup=candidate_setup,
                resume_policy=resume_policy,
                plan_only=plan_only,
            ))
    except RunStalled as exc:  # run watchdog aborted: classify, never crash
        success = False
        run_outcome = exc.reason
        error = (
            f"run aborted by watchdog: no run-level forward progress within "
            f"{run_stall_abort:g}s (stuck/stalled)"
        )
        logger.error("Dispatch %s %s", run_id, error)
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

    lead_token = generator_chain[0]
    lead_cfg = roles.AGENT_DEFAULTS.get(lead_token, {})
    effort_val = lead_cfg.get("effort")
    effort = str(effort_val) if effort_val not in (None, "n/a") else None
    final_verified = bool(getattr(workflow, "verified", False))
    verifier_delta = _derive_verifier_delta(baseline_result, final_verified)

    wall_ms_value: Optional[float] = (duration * 1000.0) if duration else None
    out_bytes_value: Optional[int] = None
    watchdog_reason_value: Optional[str] = None
    if workflow is not None:
        # Prefer per-agent telemetry when the workflow exposes it.
        agent_for_telemetry = (
            getattr(workflow, "direct_generator", None)
            or getattr(workflow, "generator", None)
        )
        if agent_for_telemetry is not None:
            out_bytes_value = getattr(agent_for_telemetry, "last_out_bytes", None)
            agent_wall = getattr(agent_for_telemetry, "last_wall_ms", None)
            if agent_wall is not None:
                wall_ms_value = float(agent_wall)
            watchdog_reason_value = getattr(agent_for_telemetry, "_watchdog_reason", None)

    telemetry = {
        "wall_ms": wall_ms_value,
        "out_bytes": out_bytes_value,
        "watchdog_reason": watchdog_reason_value,
        "worker": lead_token,
        "model": str(lead_cfg.get("model", "") or ""),
        "effort": effort,
        "baseline_ok": baseline_result.ok if baseline_result is not None else None,
        "baseline_error_hash": baseline_result.error_hash if baseline_result is not None else None,
        "baseline_duration_ms": baseline_result.duration_ms if baseline_result is not None else None,
        "verifier_delta": verifier_delta,
    }

    # Quality-cost ledger (task #9): how much to trust this run.
    quality = build_ledger(
        workflow, mode=mode, had_verifier=verifier is not None,
        produced_output=bool(output and output.strip()),
        telemetry=telemetry,
    )
    logger.info("Dispatch %s | confidence=%s (%s)", run_id, quality["confidence"], quality["note"])

    after = take_snapshot(work_dir)
    diff = diff_snapshots(before, after)

    # Path-policy guard (#38): fail the run if a worker touched an off-limits path
    # (denylist) or wrote outside the allowed subtree (allowlist). Gate on the
    # change set the snapshot already computed — no new instrumentation.
    protect_violations: List[Dict[str, str]] = []
    if protect_paths or allow_paths:
        protect_violations = evaluate_path_policy(diff.changed, protect_paths, allow_paths)
        if protect_violations:
            success = False
            offenders = ", ".join(v["path"] for v in protect_violations)
            policy_err = f"path-policy violation: {offenders}"
            error = f"{error}; {policy_err}" if error else policy_err
            logger.warning(
                "Dispatch %s VIOLATED path policy (%d file(s)): %s",
                run_id, len(protect_violations),
                "; ".join(f"{v['path']} ({v['reason']})" for v in protect_violations),
            )

    # Notify on anomalies + finish (#40). The stall ping already fired from the
    # watchdog; here we cover OOM / verifier-fail / a clean finish, now that
    # success is final (the path-policy gate above may have flipped it).
    if run_outcome != "stalled" and not success and verifier is not None:
        baseline_oom = bool(baseline_result is not None and getattr(baseline_result, "resource_exceeded", False))
        notifier.fire(
            "oom" if baseline_oom else "verifier_fail",
            monitor.payload(extra={
                "baseline_ok": baseline_result.ok if baseline_result is not None else None,
                "phase": "baseline" if baseline_oom else "run",
            }),
        )
    notifier.fire("finish", monitor.payload(extra={"success": success, "outcome": run_outcome}))

    (run_dir / "stdout.log").write_text(output, encoding="utf-8")
    (run_dir / "changed-files.diff").write_text(
        diff.unified or "(no file changes detected)\n", encoding="utf-8"
    )

    # Plan-only / dry-run (#41): persist the decomposed plan as a structured
    # artifact so it can be reviewed/edited before a real run. The out-dir is
    # untouched in this mode (the diff is empty by construction).
    plan_steps = getattr(workflow, "plan_steps", None) if workflow else None
    if plan_only and plan_steps is not None:
        (run_dir / "plan.json").write_text(
            json.dumps(
                {"instruction": instruction, "n_steps": len(plan_steps), "steps": plan_steps},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    stage_used = getattr(workflow, "stage_used", None) if workflow else None
    n_candidates = getattr(workflow, "n_candidates", None) if workflow else None
    n_passed = getattr(workflow, "n_passed", None) if workflow else None
    winner_index = getattr(workflow, "winner_index", None) if workflow else None
    if stage_used == -1:
        stage_used = None
    if winner_index == -1:
        winner_index = None

    # TODO: Wire the final VerifierResult up from workflows so this can
    # differentiate timeout vs returncode failures for non-verified runs.
    if quality.get("confidence") == "verified":
        verifier_failure_kind = None
    else:
        verifier_failure_kind = None

    # Close the calibration loop: every verified dispatch contributes one
    # observation to the live ledger that CalibrationTable.load() reads on
    # next process start. Only verified runs count — matching the offline
    # sweep's gate, so a critic-approved-but-tests-failed run doesn't
    # pollute the routing baselines. We use the lead generator worker as
    # the key; multi-stage modes (cascade, pat, master) inherit it.
    if quality.get("confidence") == "verified":
        append_live_row(
            worker=lead_token,
            model=str(lead_cfg.get("model", "") or ""),
            effort=effort,
            ok=True,
            out_bytes=out_bytes_value,
            wall_ms=wall_ms_value,
            mode=mode,
            stage_used=stage_used,
            n_candidates=n_candidates,
            n_passed=n_passed,
            winner_index=winner_index,
            verifier_delta=quality.get("verifier_delta"),
            verifier_failure_kind=verifier_failure_kind,
            diff_files_added=len(diff.added),
            diff_files_modified=len(diff.modified),
            diff_files_deleted=len(diff.deleted),
        )

    tokens = _summarize_token_usage(events_path)
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
        tokens=tokens,
        protect_violations=protect_violations,
        run_outcome=run_outcome,
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
    verifier_mem_max: Optional[str] = None,
    web_search: bool = False,
    mission_critical: bool = False,
    spec: Optional[str] = None,
    dashboard_stream_json: bool = False,
    out_dir: Optional[Union[str, Path]] = None,
    candidate_setup: Optional[str] = None,
    resume_policy: str = "auto",
    protect_paths: Optional[List[str]] = None,
    allow_paths: Optional[List[str]] = None,
    plan_only: bool = False,
    run_stall_abort: Optional[float] = None,
    notify: Optional[str] = None,
    notify_cmd: Optional[str] = None,
    heartbeat_interval: Optional[float] = None,
    # Step 12: forwarded for computer-use adapter (see dispatch_async)
    # Step 10: real-gui harness flags (passed through only when present; non-real paths identical)
    computer_use_mode: Optional[str] = None,
    computer_use_task_priority: Optional[str] = None,
    computer_use_budgets: Optional[Dict[str, Any]] = None,
    real_gui_policy: Optional[str] = None,
    ask_mode: Optional[str] = None,
    browser_engine: Optional[str] = None,
    browser_display: Optional[str] = None,
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
            verifier_mem_max=verifier_mem_max,
            web_search=web_search,
            mission_critical=mission_critical,
            spec=spec,
            dashboard_stream_json=dashboard_stream_json,
            out_dir=out_dir,
            candidate_setup=candidate_setup,
            resume_policy=resume_policy,
            protect_paths=protect_paths,
            allow_paths=allow_paths,
            plan_only=plan_only,
            run_stall_abort=run_stall_abort,
            notify=notify,
            notify_cmd=notify_cmd,
            heartbeat_interval=heartbeat_interval,
            computer_use_mode=computer_use_mode,
            computer_use_task_priority=computer_use_task_priority,
            computer_use_budgets=computer_use_budgets,
            real_gui_policy=real_gui_policy,
            ask_mode=ask_mode,
            browser_engine=browser_engine,
            browser_display=browser_display,
        )
    )

"""Operator CLI for the workflow harness.

This is the surface I (the driving assistant) use to instruct agy/codex to write
code, with every run captured under runs/<ts>/ for tracking and review.

  python -m harness do "add a --version flag to the CLI"
  python -m harness do "refactor X" --mode direct --no-fallback
  python -m harness do "build feature Y" --mode master --test-cmd "pytest -q"
  python -m harness runs            # list recent runs
  python -m harness show <run_id>   # print a run's diff + summary
  python -m harness dashboard       # launch dashboard server
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from harness import roles
from harness.dispatch import RUNS_DIR, dispatch

C_RESET, C_BOLD, C_GREEN, C_RED, C_YELLOW, C_CYAN = (
    "\033[0m", "\033[1m", "\033[32m", "\033[31m", "\033[33m", "\033[36m"
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _print_result(result) -> None:
    ok = f"{C_GREEN}OK{C_RESET}" if result.success else f"{C_RED}FAILED{C_RESET}"
    print(f"\n{C_BOLD}── dispatch {result.run_id} [{ok}] ──{C_RESET}")
    print(f"  mode      : {result.mode}")
    print(f"  generator : {result.generator}")
    if result.critic:
        print(f"  critic    : {result.critic}")
    print(f"  duration  : {result.duration_s}s")
    if result.quality:
        conf = result.quality.get("confidence", "?")
        col = C_GREEN if conf == "verified" else (C_YELLOW if conf in ("approved", "unverified") else C_RED)
        iters = result.quality.get("iterations_used")
        extra = f" ({iters} iter)" if iters else ""
        print(f"  confidence: {col}{conf}{C_RESET}{extra} — {result.quality.get('note','')}")
        delta = result.quality.get("verifier_delta")
        if delta:
            dcol = C_GREEN if delta == "fixed" else (C_YELLOW if delta in ("preserved", "unchanged") else C_RED)
            print(f"  verifier  : {dcol}{delta}{C_RESET}")
    # #42: surface the resolved per-agent (model, effort) only when something was
    # actually overridden vs the baked defaults (keeps routine runs uncluttered).
    rc = getattr(result, "resolved_config", None)
    if rc:
        from harness.roles import AGENT_DEFAULTS as _AD

        def _nondefault(cfgmap) -> bool:
            for prov, c in (cfgmap or {}).items():
                d = _AD.get(prov, {})
                if (str(c.get("model")) != str(d.get("model"))
                        or str(c.get("effort")) != str(d.get("effort"))):
                    return True
            return False

        scale = rc.get("watchdog_scale", 1.0)
        if _nondefault(rc.get("generator")) or _nondefault(rc.get("critic")) or (
            scale not in (1.0, None)
        ):
            gen_cfg = rc.get("generator", {})
            parts = [f"{p}={c.get('model')}/{c.get('effort')}" for p, c in gen_cfg.items()]
            tail = f"  (watchdog x{scale})" if scale and scale != 1.0 else ""
            print(f"  {C_CYAN}effort    : {' '.join(parts)}{C_RESET}{tail}")
    rec = getattr(result, "reconciliation", None)
    if rec:
        dead = [f for f in rec.get("findings", [])
                if f.get("classification") == "exists_not_load_bearing"]
        col = C_GREEN if rec.get("reconciled") else C_RED
        print(f"  {col}reconcile : {rec.get('verdict')} "
              f"({len(dead)} exists-but-not-load-bearing){C_RESET}")
        for f in dead[:10]:
            loc = f.get("location") or "?"
            print(f"    {C_RED}⚠ {f.get('name')} [{f.get('sub_kind')}] {loc}{C_RESET}")
    if getattr(result, "run_outcome", None):
        print(f"  {C_RED}outcome   : {result.run_outcome} (run watchdog){C_RESET}")
    if result.error:
        print(f"  {C_RED}error     : {result.error}{C_RESET}")
    if getattr(result, "protect_violations", None):
        print(f"  {C_RED}path policy: {len(result.protect_violations)} violation(s){C_RESET}")
        for v in result.protect_violations:
            print(f"    {C_RED}✗ {v['path']} — {v['reason']}{C_RESET}")
    if result.changed_files:
        print(f"  {C_BOLD}changed   : {len(result.changed_files)} file(s){C_RESET}")
        for f in result.added:
            print(f"    {C_GREEN}+ {f}{C_RESET}")
        for f in result.modified:
            print(f"    {C_YELLOW}~ {f}{C_RESET}")
        for f in result.deleted:
            print(f"    {C_RED}- {f}{C_RESET}")
    else:
        print(f"  {C_YELLOW}changed   : (no files changed on disk){C_RESET}")
    print(f"  artifacts : {result.run_dir}/")
    print("              prompt.txt  stdout.log  stderr.log  changed-files.diff  meta.json")


def _cmd_do(args) -> int:
    gen_chain = args.generator.split(",") if args.generator else None
    crit_chain = args.critic.split(",") if args.critic else None
    spec_text = None
    if args.spec:
        spec_path = Path(args.spec).expanduser()
        if not spec_path.exists():
            print(f"{C_RED}no such spec file: {args.spec}{C_RESET}", file=sys.stderr)
            return 1
        spec_text = spec_path.read_text(encoding="utf-8")

    # #37: master/pat checkpoint resume policy. Default "auto" (resume only if the
    # out-dir still matches the tree the checkpoint was saved against, else start
    # fresh). --fresh forces a clean start; --resume forces resume even if the tree
    # diverged. Mutually exclusive.
    if getattr(args, "fresh", False) and getattr(args, "resume", False):
        print(f"{C_RED}--fresh and --resume are mutually exclusive{C_RESET}", file=sys.stderr)
        return 1
    if getattr(args, "fresh", False):
        resume_policy = "never"
    elif getattr(args, "resume", False):
        resume_policy = "force"
    else:
        resume_policy = "auto"

    # #41: --plan-only is a master/pat dry-run. Warn (don't fail) if used with a
    # mode that has no planner phase — it would otherwise silently run for real.
    if getattr(args, "plan_only", False) and args.mode not in ("master", "pat"):
        print(
            f"{C_YELLOW}warning: --plan-only only applies to --mode master/pat; "
            f"ignoring it for --mode {args.mode}{C_RESET}",
            file=sys.stderr,
        )

    # Plan injection: --plan feeds an edited plan.json back for verbatim execution
    # (the review->revise->execute round-trip). --plan-graph is a STRICT alias that
    # additionally errors if the file is NOT a graph DAG (clear operator intent +
    # a useful error when a flat plan is passed where a graph was expected). Fail
    # fast on misuse and on a malformed plan file so the error is at the CLI, not
    # deep in a dispatch.
    plan_steps = None
    plan_graph = None
    plan_path = getattr(args, "plan", None)
    plan_graph_path = getattr(args, "plan_graph", None)
    if plan_path and plan_graph_path:
        print(
            f"{C_RED}--plan and --plan-graph are mutually exclusive (--plan-graph "
            f"is a strict-graph alias of --plan){C_RESET}",
            file=sys.stderr,
        )
        return 1
    plan_file = plan_path or plan_graph_path
    strict_graph = plan_graph_path is not None
    if plan_file:
        if getattr(args, "plan_only", False):
            print(
                f"{C_RED}--plan/--plan-graph and --plan-only are mutually exclusive "
                f"(--plan-only generates a plan; --plan executes one){C_RESET}",
                file=sys.stderr,
            )
            return 1
        # Graphs are restricted to --mode master for v1 (pat's Stage-1 is a single
        # direct attempt; a DAG only makes sense in the escalation path). A flat
        # plan still runs on master OR pat.
        if args.mode not in ("master", "pat"):
            print(
                f"{C_RED}--plan only applies to --mode master/pat (got "
                f"--mode {args.mode}){C_RESET}",
                file=sys.stderr,
            )
            return 1
        # Load the full Plan so a graph 'nodes' DAG threads plan_graph (the
        # frontier scheduler routes on it); a flat plan keeps the legacy
        # linearization. load_plan validates both shapes (fail-fast at the CLI).
        from harness.dispatch import load_plan
        from agy_orchestrator.execution.graph_plan import GraphPlan
        try:
            plan = load_plan(plan_file)
        except ValueError as exc:
            print(f"{C_RED}{exc}{C_RESET}", file=sys.stderr)
            return 1
        plan_steps = plan.as_steps()
        if isinstance(plan, GraphPlan):
            plan_graph = plan
            shape = "graph DAG"
        else:
            if strict_graph:
                print(
                    f"{C_RED}--plan-graph requires a graph plan (a 'nodes' DAG); "
                    f"{plan_file} is a flat plan — use --plan for it{C_RESET}",
                    file=sys.stderr,
                )
                return 1
            shape = "flat plan"
        # A graph DAG (with non-linear deps) only runs on master (the frontier
        # scheduler is master-only for v1); error for pat rather than silently
        # linearizing it.
        if plan_graph is not None and args.mode != "master":
            print(
                f"{C_RED}a graph DAG plan requires --mode master (got "
                f"--mode {args.mode}); graph execution is master-only for v1{C_RESET}",
                file=sys.stderr,
            )
            return 1
        print(
            f"{C_YELLOW}note: executing supplied {shape} ({len(plan_steps)} "
            f"step(s)) from {plan_file}; the planner is skipped{C_RESET}",
            file=sys.stderr,
        )

    # Graph execution (docs §5 M3): cap concurrent DAG nodes. CLI flag wins over
    # the AGY_MAX_PARALLEL_NODES env; None = unbounded (all-ready nodes run at
    # once). Only affects a graph plan with non-linear deps — linear runs ignore
    # it. (The MasterWorkflow re-resolves the env itself when this is None, so the
    # env still applies on the direct-construction path.)
    max_parallel_nodes = getattr(args, "max_parallel_nodes", None)
    if max_parallel_nodes is None:
        _env_mpn = os.environ.get("AGY_MAX_PARALLEL_NODES", "")
        if _env_mpn.strip().isdigit():
            max_parallel_nodes = int(_env_mpn)

    # #42: validate effort/model overrides up front so a typo'd tier/model fails
    # fast with an enumerated message instead of surfacing deep in a dispatch.
    # Re-resolved (deterministically) inside dispatch; this is the early gate.
    from harness import roles as _roles
    from harness.effort_overrides import OverrideError, resolve_overrides
    _gen_chain = gen_chain or list(_roles.GENERATOR_CHAIN)
    _crit_chain = crit_chain or list(_roles.CRITIC_CHAIN)
    try:
        _resolved = resolve_overrides(
            generator_chain=_gen_chain,
            critic_chain=_crit_chain,
            mode=args.mode,
            profile=args.effort_profile,
            gen_effort=args.gen_effort, gen_model=args.gen_model,
            critic_effort=args.critic_effort, critic_model=args.critic_model,
            architect_effort=args.architect_effort, architect_model=args.architect_model,
            codex_model=args.codex_model,
            effort_map=args.effort_map, model_map=args.model_map,
            watchdog_scale=args.watchdog_scale,
        )
    except OverrideError as exc:
        print(f"{C_RED}{exc}{C_RESET}", file=sys.stderr)
        return 1
    for _note in _resolved.notes:
        print(f"{C_YELLOW}note: {_note}{C_RESET}", file=sys.stderr)

    # Step 12: parse computer-use config (only has effect when generator chain leads with computer-use)
    cu_mode = getattr(args, "computer_use_mode", None)
    cu_priority = getattr(args, "computer_use_task_priority", None)
    real_gui_policy = getattr(args, "real_gui_policy", None)
    ask_mode = getattr(args, "ask_mode", None)
    browser_engine = getattr(args, "browser_engine", None)
    browser_display = getattr(args, "browser_display", None)
    cu_budgets = None
    if getattr(args, "computer_use_budgets", None):
        try:
            import json as _json
            cu_budgets = _json.loads(args.computer_use_budgets)
            if not isinstance(cu_budgets, dict):
                raise ValueError("budgets must be a JSON object")
        except Exception as e:
            print(f"{C_RED}bad --computer-use-budgets JSON: {e}{C_RESET}", file=sys.stderr)
            return 1

    result = dispatch(
        args.instruction,
        mode=args.mode,
        context=args.context,
        generator_chain=gen_chain,
        critic_chain=crit_chain,
        fallback=args.fallback,
        cycles=args.cycles,
        max_iterations=args.max_iterations,
        branches=args.branches,
        test_cmd=args.test_cmd,
        verifier_mem_max=args.verifier_mem_max,
        candidate_setup=args.candidate_setup,
        resume_policy=resume_policy,
        protect_paths=[g for g in (args.protect_paths or "").split(",") if g.strip()] or None,
        allow_paths=[g for g in (args.allow_paths or "").split(",") if g.strip()] or None,
        plan_only=getattr(args, "plan_only", False),
        plan_steps=plan_steps,
        plan_graph=plan_graph,
        max_parallel_nodes=max_parallel_nodes,
        merge_policy=getattr(args, "merge_policy", "reconcile"),
        run_stall_abort=args.run_stall_abort,
        notify=args.notify or os.environ.get("AGY_NOTIFY") or None,
        notify_cmd=args.notify_cmd,
        heartbeat_interval=args.heartbeat_interval,
        telegram_enabled=getattr(args, "telegram_enabled", None),
        telegram_verbosity=getattr(args, "telegram_verbosity", None),
        gen_effort=args.gen_effort,
        gen_model=args.gen_model,
        critic_effort=args.critic_effort,
        critic_model=args.critic_model,
        architect_effort=args.architect_effort,
        architect_model=args.architect_model,
        codex_model=args.codex_model,
        effort_map=args.effort_map,
        model_map=args.model_map,
        effort_profile=args.effort_profile,
        watchdog_scale=args.watchdog_scale,
        max_parallel_workers=args.max_parallel_workers,
        worker_mem_max=args.worker_mem_max,
        baseline_gate=args.baseline_gate,
        reconcile=args.reconcile,
        reconcile_disposition=args.reconcile_disposition,
        web_search=args.web_search,
        mission_critical=args.mission_critical,
        spec=spec_text,
        out_dir=args.out_dir,
        computer_use_mode=cu_mode,
        computer_use_task_priority=cu_priority,
        computer_use_budgets=cu_budgets,
        real_gui_policy=real_gui_policy,
        ask_mode=ask_mode,
        browser_engine=browser_engine,
        browser_display=browser_display,
    )
    _print_result(result)
    if getattr(args, "plan_only", False) and result.success:
        # Graph round-trip hint (docs §5 M5): if the emitted plan.json is a graph
        # DAG (the operator echoed a --plan graph through --plan-only), point them
        # back through --plan, and note they can add deps / split a step into
        # parallel nodes before re-feeding it.
        plan_json = Path(result.run_dir) / "plan.json"
        is_graph_emit = False
        try:
            is_graph_emit = "nodes" in json.loads(plan_json.read_text())
        except Exception:
            is_graph_emit = False
        print(
            f"  {C_BOLD}round-trip{C_RESET}: review/edit {result.run_dir}/plan.json, then\n"
            f"    python -m harness do \"{args.instruction}\" --mode {args.mode} "
            f"--plan {result.run_dir}/plan.json"
        )
        if is_graph_emit:
            print(
                f"  {C_CYAN}graph plan{C_RESET}: edit 'nodes' deps (or split a step "
                f"into parallel nodes), then re-feed via --plan / --plan-graph "
                f"(--merge-policy reconcile|disjoint|fail, --max-parallel-nodes N)."
            )
    return 0 if result.success else 1


def _cmd_spec(args) -> int:
    from harness.spec import generate_spec

    arch_chain = args.architect.split(",") if args.architect else None
    crit_chain = args.critic.split(",") if args.critic else None
    result = generate_spec(
        args.goal,
        constraints=args.constraint,
        architect_chain=arch_chain,
        critic_chain=crit_chain,
        fallback=args.fallback,
        cycles=args.cycles,
        max_iterations=args.max_iterations,
        output_path=args.output,
    )
    ok = f"{C_GREEN}OK{C_RESET}" if result.success else f"{C_RED}FAILED{C_RESET}"
    print(f"\n{C_BOLD}── floodspec {result.run_id} [{ok}] ──{C_RESET}")
    print(f"  architect : {result.generator}")
    print(f"  critic    : {result.critic}")
    print(f"  duration  : {result.duration_s}s")
    conf = "approved" if result.approved else ("stalled" if result.stalled else "max-iter")
    col = C_GREEN if result.approved else C_YELLOW
    print(f"  outcome   : {col}{conf}{C_RESET} ({result.iterations_used} iter, {result.chars} chars)")
    if result.constraints:
        print(f"  {C_BOLD}constraints:{C_RESET} {len(result.constraints)}")
        for c in result.constraints:
            print(f"    - {c}")
    if result.error:
        print(f"  {C_RED}error     : {result.error}{C_RESET}")
    print(f"  {C_BOLD}spec      : {result.spec_path}{C_RESET}")
    print(f"  artifacts : {result.run_dir}/")
    print(f"\n  {C_CYAN}review it, then build:{C_RESET}")
    print(f"    python -m harness do \"<instruction>\" --mode master --spec {result.spec_path}")
    return 0 if result.success else 1


def _cmd_runs(args) -> int:
    if not RUNS_DIR.exists():
        print("(no runs yet)")
        return 0
    rows = sorted((d for d in RUNS_DIR.iterdir() if d.is_dir()), reverse=True)
    for d in rows[: args.limit]:
        meta_path = d / "meta.json"
        if not meta_path.exists():
            print(f"{d.name}  (incomplete)")
            continue
        m = json.loads(meta_path.read_text())
        status = "OK" if m.get("success") else "FAIL"
        n = len(m.get("changed_files", []))
        print(f"{d.name}  {status:4}  {m.get('mode',''):11}  {n} changed  {m.get('duration_s')}s")
    return 0


def _cmd_show(args) -> int:
    run_dir = RUNS_DIR / args.run_id
    if not run_dir.exists():
        print(f"no such run: {args.run_id}", file=sys.stderr)
        return 1
    meta = run_dir / "meta.json"
    if meta.exists():
        print(f"{C_BOLD}meta.json{C_RESET}")
        print(meta.read_text())
    diff = run_dir / "changed-files.diff"
    if diff.exists():
        print(f"\n{C_BOLD}changed-files.diff{C_RESET}")
        print(diff.read_text())
    return 0


def _cmd_dashboard(args) -> int:
    cmd = [sys.executable, "-m", "dashboard", "--port", str(args.port)]
    if args.browser:
        cmd.append("--browser")
    os.execvp(cmd[0], cmd)
    return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="harness", description="Workflow harness: drive agy/codex via the orchestrator."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")
    sub = parser.add_subparsers(dest="command", required=True)

    do = sub.add_parser("do", help="Dispatch one coding instruction to a worker")
    do.add_argument("instruction", type=str, help="The instruction for the worker")
    do.add_argument("--mode", choices=["direct", "adversarial", "feedback", "cascade", "master", "pat", "vote", "auto"],
                    default="adversarial",
                    help="Workflow shape. direct=one shot; adversarial=generate+critic loop "
                         "(default); feedback=generate+run-tests+repair loop (needs --test-cmd); "
                         "cascade=cheap-first escalation across the --generator stages, escalate "
                         "on verifier failure (needs --test-cmd); "
                         "master=plan+ToT+adversarial for whole features; "
                         "pat=Plan-after-Trial: direct attempt first, escalate to master only on "
                         "verifier failure (needs --test-cmd; ~40%% cost savings on easy tasks); "
                         "vote=K parallel candidates in isolated workspaces, verifier picks the "
                         "winner (needs --test-cmd; K=--branches; heterogeneous when chain has "
                         "multiple providers); "
                         "auto=rule-based router picks the right concrete mode based on task "
                         "features (test_cmd presence, prompt scale, ambiguity keywords).")
    do.add_argument("--context", type=str, default=None,
                    help="Extra context appended to the instruction")
    do.add_argument("--generator", type=str, default=None,
                    help=f"Comma-separated generator chain (default: {','.join(roles.GENERATOR_CHAIN)}). "
                         f"Workers: codex, claude, agy, grok, computer-use.")
    do.add_argument("--critic", type=str, default=None,
                    help=f"Comma-separated critic chain (default: {','.join(roles.CRITIC_CHAIN)}). "
                         f"Workers: codex, claude, agy, grok, computer-use.")
    do.add_argument("--fallback", action=argparse.BooleanOptionalAction, default=True,
                    help="Wrap roles in usage-exhaustion fallback (default on)")
    do.add_argument("--cycles", type=int, default=2,
                    help="Times the fallback chain is cycled before giving up")
    do.add_argument("--max-iterations", type=int, default=5,
                    help="Max generator/critic iterations (adversarial/master)")
    do.add_argument("--branches", type=int, default=3, help="ToT branches (master mode)")
    do.add_argument("--test-cmd", type=str, default=None,
                    help="Optional verification command run as a quality gate")
    do.add_argument("--verifier-mem-max", type=str, default=None, metavar="SIZE",
                    help="Run --test-cmd inside its own memory-capped systemd scope "
                         "(e.g. '3G') so a heavy gate is OOM-killed in its own scope "
                         "instead of freezing the host / taking the orchestrator down. "
                         "Opt-in; needs `systemd-run --user` (degrades to uncapped with "
                         "a warning otherwise). Env: AGY_VERIFIER_MEM_MAX.")
    do.add_argument("--candidate-setup", type=str, default=None, metavar="CMD",
                    help="vote mode: shell command run inside each candidate's "
                         "isolated workspace BEFORE its verifier (e.g. "
                         "'python -m venv .venv && .venv/bin/pip install -e .'). "
                         "Makes vote isolation sound on editable-install repos so "
                         "each candidate's verifier imports that candidate's own "
                         "source. Bounded by the verifier-concurrency cap.")
    do.add_argument("--fresh", "--no-resume", dest="fresh", action="store_true",
                    help="master/pat: ignore any salvage checkpoint and start clean "
                         "(don't resume from a prior killed run)")
    do.add_argument("--resume", action="store_true",
                    help="master/pat: force resume from the salvage checkpoint even if "
                         "the out-dir diverged from the tree it was saved against "
                         "(default is to start fresh on divergence; #37)")
    do.add_argument("--plan-only", "--dry-run", dest="plan_only", action="store_true",
                    help="master/pat: run only the planner, emit the decomposed step "
                         "plan (stdout + events + runs/<id>/plan.json), and exit BEFORE "
                         "any worker writes to the out-dir.")
    do.add_argument("--plan", type=str, default=None, metavar="FILE",
                    help="master/pat: execute the steps in this plan file VERBATIM, "
                         "skipping the planner. Accepts a --plan-only plan.json (a "
                         "bare JSON list of step strings, OR a graph 'nodes' DAG), "
                         "closing the round-trip: --plan-only -> review/edit "
                         "plan.json -> --plan <file>. A graph DAG with non-linear "
                         "deps runs the concurrent frontier scheduler. Mutually "
                         "exclusive with --plan-only.")
    do.add_argument("--plan-graph", type=str, default=None, metavar="FILE",
                    help="master: like --plan, but STRICT — the file MUST be a "
                         "graph 'nodes' DAG (errors on a flat plan). Runs the "
                         "concurrent frontier scheduler. Graphs are master-only "
                         "for v1. Mutually exclusive with --plan / --plan-only.")
    do.add_argument("--max-parallel-nodes", type=int, default=None, metavar="N",
                    help="master: cap how many DAG nodes run concurrently when "
                         "executing a graph plan (env AGY_MAX_PARALLEL_NODES; "
                         "default unbounded). 1 serializes a wide layer. Only "
                         "affects a --plan graph with non-linear deps.")
    do.add_argument("--merge-policy", choices=("disjoint", "reconcile", "fail"),
                    default="reconcile",
                    help="master graph mode: how two parallel nodes that write the "
                         "SAME file are reconciled. reconcile (default) auto-applies "
                         "disjoint writes and sends overlaps to the reconcile station "
                         "(then re-verifies the merged tree); disjoint/fail abort on "
                         "any overlap (fail records the conflicting paths in "
                         "meta.json). Only affects a --plan graph with non-linear deps.")
    do.add_argument("--protect-paths", type=str, default=None, metavar="GLOB[,GLOB...]",
                    help="Fail the run if any worker modifies a path matching these "
                         "denylist globs (e.g. 'docs/core/**,**/*.lock,migrations/**'). "
                         "Gated on the change set the harness already computes; "
                         "violations are recorded in meta.json. ** spans directories.")
    do.add_argument("--allow-paths", type=str, default=None, metavar="GLOB[,GLOB...]",
                    help="Allowlist inverse of --protect-paths: fail the run if a worker "
                         "writes any path OUTSIDE these globs (e.g. additive-only into a "
                         "single subtree).")
    do.add_argument("--run-stall-abort", type=float, default=None, metavar="SEC",
                    help="Whole-run watchdog: abort the run if NO run-level forward "
                         "progress (a step/branch/plan milestone, a worker-call "
                         "boundary, a usage tick — i.e. anything but raw token "
                         "chatter) happens within SEC seconds, then classify it "
                         "'stalled' in meta.json and fire --notify. Complements the "
                         "per-agent stall watchdog, which only catches a SILENT "
                         "worker — this catches a stuck-but-chatty run (#40).")
    do.add_argument("--notify", type=str, default=None, metavar="URL|CMD",
                    help="Best-effort, non-fatal notification on lifecycle/anomaly "
                         "events (start/step/stall/oom/verifier-fail/finish). An "
                         "http(s):// value is POSTed a small JSON payload; anything "
                         "else is run as a shell command (payload on stdin + "
                         "AGY_NOTIFY_* env). See also --notify-cmd. Env: AGY_NOTIFY.")
    do.add_argument("--notify-cmd", type=str, default=None, metavar="CMD",
                    help="Shell command form of --notify (e.g. "
                         "'notify-send \"agentorch $AGY_NOTIFY_EVENT\"'). Runs on the "
                         "same events; payload arrives on stdin + AGY_NOTIFY_* env.")
    do.add_argument("--heartbeat-interval", type=float, default=None, metavar="SEC",
                    help="Seconds between run-level 'heartbeat' events written to "
                         "events.jsonl (step, free-mem, elapsed, since-progress) so "
                         "a watcher/dashboard reads one explicit liveness signal "
                         "instead of inferring it from file mtimes. Default 30 "
                         "(env AGY_HEARTBEAT_SECONDS); 0 disables.")
    # Telegram build-progress bot. Auto-ON when TELEGRAM_BOT_KEY is set AND the
    # whitelist (AGY_TELEGRAM_USERS) is non-empty; --no-telegram forces off,
    # --telegram forces on (warns + stays off if key/whitelist missing).
    tg = do.add_mutually_exclusive_group()
    tg.add_argument("--telegram", dest="telegram_enabled", action="store_true", default=None,
                    help="Force-enable Telegram build-progress notifications "
                         "(needs TELEGRAM_BOT_KEY + a non-empty whitelist; warns "
                         "and stays off if missing). Auto-on by default when both "
                         "are present.")
    tg.add_argument("--no-telegram", dest="telegram_enabled", action="store_false",
                    help="Disable Telegram build-progress notifications for this run.")
    do.add_argument("--telegram-verbosity", choices=["quiet", "normal", "verbose", "debug"],
                    default=None,
                    help="Telegram message verbosity (default env "
                         "AGY_TELEGRAM_VERBOSITY else 'normal').")
    # Per-role / per-provider effort + model overrides (#42). The default tier
    # (codex gpt-5.3-codex / high) is right for routine work; crank these for a
    # mission-critical, invariant-touching build that wants every provider at
    # its ceiling. 'max' effort maps to codex reasoning_effort=xhigh.
    eff = do.add_argument_group("effort/model overrides (#42)")
    eff.add_argument("--gen-effort", type=str, default=None, metavar="TIER",
                     help="Generator effort tier (low|medium|high|max); applies to "
                          "every effort-capable provider in the generator chain "
                          "(grok no-ops). 'max' -> codex reasoning_effort=xhigh.")
    eff.add_argument("--critic-effort", type=str, default=None, metavar="TIER",
                     help="Critic effort tier (low|medium|high|max); applies to every "
                          "effort-capable provider in the critic chain.")
    eff.add_argument("--architect-effort", type=str, default=None, metavar="TIER",
                     help="Effort tier for the master/pat architect chain (alias of "
                          "--gen-effort for those modes).")
    eff.add_argument("--gen-model", type=str, default=None, metavar="NAME",
                     help="Model for the generator chain lead (provider-specific).")
    eff.add_argument("--critic-model", type=str, default=None, metavar="NAME",
                     help="Model for the critic chain lead (provider-specific).")
    eff.add_argument("--architect-model", type=str, default=None, metavar="NAME",
                     help="Model for the master/pat architect chain lead.")
    eff.add_argument("--codex-model", type=str, default=None, metavar="NAME",
                     help="Convenience: set the codex model anywhere it appears in any "
                          "chain (e.g. gpt-5.5). Validated against codex's model list.")
    eff.add_argument("--effort", dest="effort_map", type=str, default=None, metavar="MAP",
                     help="Per-provider effort map, e.g. 'codex=max,agy=high' (or a bare "
                          "tier applied to all effort-capable providers). grok=… is dropped.")
    eff.add_argument("--model", dest="model_map", type=str, default=None, metavar="MAP",
                     help="Per-provider model map, e.g. 'codex=gpt-5.5'.")
    eff.add_argument("--effort-profile", choices=["low", "balanced", "max"], default=None,
                     help="One-switch preset: 'max' cranks every effort-capable provider "
                          "to its strongest model + ceiling effort (codex gpt-5.5/xhigh); "
                          "'balanced' == defaults; 'low' = cheap tier. Explicit flags "
                          "override the profile.")
    eff.add_argument("--watchdog-scale", type=float, default=None, metavar="FLOAT",
                     help="Multiply the streaming-watchdog stall/byte budgets (>1.0) for "
                          "known-heavy tiers so a long xhigh run isn't truncated mid-flight "
                          "(env AGY_WATCHDOG_SCALE).")
    eff.add_argument("--max-parallel-workers", type=int, default=None, metavar="N",
                     help="Cap how many candidates run end-to-end at once in vote mode "
                          "(host-safety for --branches>1; env AGY_MAX_PARALLEL_WORKERS).")
    eff.add_argument("--worker-mem-max", type=str, default=None, metavar="SIZE",
                     help="Per-candidate verifier memory cap for vote/tot (e.g. 4G), run "
                          "in its own systemd scope like --verifier-mem-max (#39). Guards "
                          "against an OOM/freeze when --branches>1 verify in parallel.")
    eff.add_argument("--baseline-gate", action="store_true",
                     help="Run the pre-run baseline verifier (the FULL --test-cmd suite on "
                          "the unchanged tree) for non-vote modes too. Off by default: the "
                          "baseline only feeds telemetry outside vote mode, so skipping it "
                          "removes a serial test-suite from the critical path. Set this to "
                          "restore the verifier_delta (preserved/regressed/fixed) telemetry.")
    # Reconciliation / Integration-Skeptic station (#43).
    rec = do.add_argument_group("reconciliation station (#43)")
    rec.add_argument("--reconcile", action="store_true",
                     help="After a converged+green build, run the goal-vs-runtime "
                          "reconciliation station: trace each goal-named mechanism to "
                          "the live execution path and flag 'exists-but-not-load-bearing' "
                          "defects (dead/stubbed/untrained/bypassed code that passes "
                          "tests). Default-ON for --mission-critical; env AGY_RECONCILE=1. "
                          "Writes runs/<id>/reconcile.json. Verdict is distinct — never "
                          "folded into the verifier's pass/fail.")
    rec.add_argument("--reconcile-disposition", choices=["warn", "fail", "open-task"],
                     default=None,
                     help="What a non-reconciled verdict does: 'warn' (default — report "
                          "loudly + artifact, don't fail), 'fail' (flip the run to failed), "
                          "'open-task' (warn + recommend a follow-up build task).")
    do.add_argument("--web-search", action="store_true",
                    help="Enable codex web search (-c tools.web_search=true) for accuracy")
    do.add_argument("--mission-critical", action="store_true",
                    help="Prepend a catastrophic-failure-focused preamble to the "
                         "critic prompt (adversarial mode). Opt-in: more exhaustive, "
                         "severity-prioritized review for code whose failure could "
                         "exhaust resources or crash/hang the host. Off by default.")
    do.add_argument("--out-dir", type=str, default=None, metavar="PATH",
                    help="Directory the worker should write files into (its cwd). "
                         "Default: AgentOrch's own repo root. Set when invoking AgentOrch "
                         "from another repo so workers don't pollute AgentOrch. "
                         "Snapshot diff and changed-files list scope follow this path.")
    # Step 12: computer-use config (forwarded only when --generator contains computer-use)
    do.add_argument("--computer-use-mode", choices=["ISOLATED", "OBSERVE", "REAL"], default=None,
                    help="computer-use: ISOLATED (default: private Xvfb, full perceive+act) or "
                         "OBSERVE (real :0 read-only perception; actions remain isolated-only per FR-03/04) or "
                         "REAL (real :0 perception and owned-child real_act under SafetyKernel policy gates).")
    do.add_argument("--real-gui-policy", choices=["full", "children"], default=None,
                    help="computer-use REAL mode: foreign-target policy ('full' allows prompt-gated foreign act; "
                         "'children' only allows owned-child direct actuation).")
    do.add_argument("--ask-mode", choices=["on", "off"], default=None,
                    help="computer-use REAL mode: GUI confirmation prompting for foreign-target actions.")
    do.add_argument("--browser-engine", choices=["bing", "duckduckgo", "google"], default="bing",
                    help="computer-use browser engine for autonomous navigate/search flows (default: bing).")
    do.add_argument("--browser-display", type=str, default=None,
                    help="computer-use browser display override. Default: :0 in REAL mode, isolated Xvfb otherwise.")
    do.add_argument("--computer-use-task-priority", choices=["normal", "high"], default=None,
                    help="computer-use: 'high' routes reasoner claude→codex; 'normal' (default) codex→claude (FR-14/21).")
    do.add_argument("--computer-use-budgets", type=str, default=None, metavar="JSON",
                    help="computer-use: JSON dict overriding budgets (e.g. '{\"max_steps\": 50, \"max_actions\": 30}').")
    do.add_argument("--spec", type=str, default=None, metavar="PATH",
                    help="Path to an approved FloodSpec design doc (see `harness spec`). "
                         "Injected as the authoritative design the worker must implement; "
                         "in master mode the planner decomposes THIS design instead of "
                         "re-inventing one from the instruction.")
    do.set_defaults(func=_cmd_do)

    spec = sub.add_parser(
        "spec",
        help="FloodSpec: turn a short goal + constraints into a complete design doc",
    )
    spec.add_argument("goal", type=str, help="The short goal to design a system for")
    spec.add_argument("-c", "--constraint", action="append", default=[], metavar="TEXT",
                      help="A constraint the design must honor (repeatable)")
    spec.add_argument("--architect", type=str, default=None,
                      help=f"Comma-separated architect (author) chain "
                           f"(default: {','.join(roles.GENERATOR_CHAIN)}).")
    spec.add_argument("--critic", type=str, default=None,
                      help=f"Comma-separated design-critic chain "
                           f"(default: {','.join(roles.CRITIC_CHAIN)}). Cross-provider "
                           f"from the architect gives stronger gates.")
    spec.add_argument("--fallback", action=argparse.BooleanOptionalAction, default=True,
                      help="Wrap roles in usage-exhaustion fallback (default on)")
    spec.add_argument("--cycles", type=int, default=2,
                      help="Times the fallback chain is cycled before giving up")
    spec.add_argument("--max-iterations", type=int, default=3,
                      help="Max architect/critic refinement rounds (default 3; gains "
                           "flatten fast)")
    spec.add_argument("-o", "--output", type=str, default=None, metavar="PATH",
                      help="Also write the doc here (e.g. a target repo's DESIGN.md). "
                           "The runs/<id>/spec.md artifact is always written regardless.")
    spec.set_defaults(func=_cmd_spec)

    runs = sub.add_parser("runs", help="List recent runs")
    runs.add_argument("--limit", type=int, default=20)
    runs.set_defaults(func=_cmd_runs)

    show = sub.add_parser("show", help="Show a run's diff and metadata")
    show.add_argument("run_id", type=str)
    show.set_defaults(func=_cmd_show)

    dashboard = sub.add_parser("dashboard", help="Launch the AgentOrch control dashboard")
    dashboard.add_argument("--port", type=int, default=8765, help="Dashboard port (default: 8765)")
    dashboard.add_argument("--browser", action="store_true",
                           help="Open the dashboard in the default browser. Off by default — "
                                "the dashboard is dev/automation-driven; auto-opening a tab on "
                                "every boot is a footgun (see dashboard/__main__.py).")
    dashboard.add_argument("--no-browser", action="store_true",
                           help="(deprecated no-op; not opening a browser is now the default)")
    dashboard.set_defaults(func=_cmd_dashboard)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

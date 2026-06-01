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
    if result.error:
        print(f"  {C_RED}error     : {result.error}{C_RESET}")
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
        candidate_setup=args.candidate_setup,
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
    do.add_argument("--candidate-setup", type=str, default=None, metavar="CMD",
                    help="vote mode: shell command run inside each candidate's "
                         "isolated workspace BEFORE its verifier (e.g. "
                         "'python -m venv .venv && .venv/bin/pip install -e .'). "
                         "Makes vote isolation sound on editable-install repos so "
                         "each candidate's verifier imports that candidate's own "
                         "source. Bounded by the verifier-concurrency cap.")
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

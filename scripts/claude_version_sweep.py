#!/usr/bin/env python3
"""Sweep all known Claude model versions head-to-head on the brutal-tier tasks.

Re-validates AgentOrch's empirical claude findings (docs/experiments.md,
memory/verified-worker-models) after a new Opus release. Answers: did the
token/cost/pass-rate ranking change now that Opus 4.8 is out?

Uses the VERIFIED `claude --model` convention (probed 2026-05-28): the full
hyphenated id `claude-opus-4-N`, with an optional `[1m]` suffix for the
1M-context variant on opus 4.7+. The interactive Claude Code `/model` aliases
like `opus4.8` do NOT work on the `--model` flag — only bare opus/sonnet/haiku
or the full form. This sweep always uses the full form so each row pins an
exact version.

Reuses scripts/token_efficiency.py wholesale for the heavy lifting —
``run_config`` (real out_tokens / cost_usd / api_ms from the CLI JSON, in a
per-call tempdir sandbox), ``extract_code`` + ``run_test`` (grade against the
hidden pytest suite), and ``scoreboard`` (ranked table). This script only adds
the claude-version grid + the haiku-effort guard + an account-sharing warning.

Claude-only by design: only codex/claude expose token telemetry, and this is a
cross-VERSION comparison within one provider. For cross-provider wall-clock,
use scripts/model_sweep.py instead.

⚠️  ACCOUNT-SHARING WARNING (memory/account-sharing-rule): claude-opus-4-8 is
the Claude Code default as of 2026-05-28. If your driving Claude Code session is
on 4.8, sweeping claude-opus-4-8 here draws from the SAME Anthropic pool — a
usage wall hits both the sweep AND your session at once. Run this when you're
not mid-session on the same tier, or accept the shared-pool risk for its
duration. (Other versions — 4.7/4.6/sonnet/haiku — are independent of a 4.8
driving session.)

Usage:
    python scripts/claude_version_sweep.py                  # all known models, all brutal tasks
    python scripts/claude_version_sweep.py --dry-run        # print the plan, spawn NOTHING
    python scripts/claude_version_sweep.py --effort high    # override effort (opus/sonnet only)
    python scripts/claude_version_sweep.py --tasks calc3 candy --repeats 2
    python scripts/claude_version_sweep.py --models claude-opus-4-8 claude-opus-4-7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cloud_eval import BRUTAL_TASKS, CLOUD_SUFFIX, extract_code, run_test
from scripts.token_efficiency import label, run_config, scoreboard

# Known claude versions and their default effort. Each model is the FULL
# hyphenated `--model` id (verified convention). Effort defaults differ:
#   - opus/sonnet 4.6+ support --effort (low|medium|high|max; max is opus-tier)
#   - haiku 4.5 REJECTS --effort — it must be None or the CLI errors. This guard
#     is the whole reason the sweep can't just reuse one flat effort value.
KNOWN_CLAUDE_MODELS: list[tuple[str, str | None]] = [
    ("claude-opus-4-8", "high"),
    ("claude-opus-4-7", "high"),
    ("claude-opus-4-6", "high"),
    ("claude-sonnet-4-6", "medium"),
    ("claude-haiku-4-5", None),   # haiku rejects --effort; keep it None
]

# Models that reject the --effort flag entirely. A user-supplied --effort
# override is NOT applied to these (it would error the CLI).
_NO_EFFORT_MODELS = {"claude-haiku-4-5"}


def _resolve_effort(model: str, default_effort: str | None,
                    override: str | None) -> str | None:
    """Effort for a given model: override wins, but never apply effort to a
    model that rejects the flag."""
    if model in _NO_EFFORT_MODELS:
        return None
    return override if override is not None else default_effort


def _build_grid(models: list[str] | None,
                effort_override: str | None) -> list[tuple[str, str, str | None]]:
    """Return [(worker, model, effort), ...] for the requested claude versions."""
    if models:
        # User passed explicit full ids; carry no per-model default effort, so
        # rely on the override (or None). Unknown models still get the haiku guard.
        pairs = [(m, None) for m in models]
    else:
        pairs = list(KNOWN_CLAUDE_MODELS)
    return [("claude", m, _resolve_effort(m, default_eff, effort_override))
            for m, default_eff in pairs]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="Full claude --model ids to sweep (default: all known versions). "
                         "Use the hyphenated form, e.g. claude-opus-4-8.")
    ap.add_argument("--tasks", nargs="*", default=list(BRUTAL_TASKS),
                    choices=list(BRUTAL_TASKS))
    ap.add_argument("--effort", default=None,
                    help="Override effort for all opus/sonnet models "
                         "(haiku always runs without --effort).")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Repeats per (model,task); medians reported.")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--out", default="/tmp/agentorch_research/claude_version_sweep.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan + resolved `claude --model` per row; spawn nothing.")
    args = ap.parse_args()

    grid = _build_grid(args.models, args.effort)
    n_calls = len(grid) * len(args.tasks) * args.repeats

    print("claude version sweep")
    print(f"  models : {len(grid)}   tasks: {len(args.tasks)}   repeats: {args.repeats}"
          f"   = {n_calls} SEQUENTIAL calls")
    print("  ⚠️  account-sharing: claude-opus-4-8 shares the pool of a 4.8 driving "
          "session (see module docstring).")
    print("  rows:")
    for worker, model, effort in grid:
        eff = effort if effort else "(no --effort)"
        print(f"    claude --model {model:24} effort={eff}")
    print(flush=True)

    if args.dry_run:
        print("--dry-run: no workers spawned. Re-run without --dry-run to execute.")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    with open(args.out, "a") as outf:
        for worker, model, effort in grid:
            combo = label(worker, model, effort)
            for task in args.tasks:
                prompt, test_src = BRUTAL_TASKS[task]
                for _ in range(args.repeats):
                    try:
                        m = run_config(worker, model, effort,
                                       prompt + CLOUD_SUFFIX, args.timeout)
                        code = extract_code(m["text"])
                        ok, _ = run_test(code, test_src)
                    except Exception as exc:  # noqa: BLE001 — bench must not abort on one combo
                        m = {"text": "", "out_tokens": None, "reasoning_tokens": None,
                             "in_tokens": None, "cost_usd": None, "api_ms": None,
                             "wall_ms": None}
                        ok, code = False, ""
                        print(f"  ! {combo} {task}: {type(exc).__name__}: {exc}", flush=True)
                    row = {"combo": combo, "worker": worker, "model": model,
                           "effort": effort, "task": task, "ok": ok,
                           "empty": not code.strip(),
                           **{k: m[k] for k in ("out_tokens", "reasoning_tokens",
                                                "in_tokens", "cost_usd", "api_ms",
                                                "wall_ms")}}
                    outf.write(json.dumps(row) + "\n")
                    outf.flush()
                    results.append(row)
                    ot = row["out_tokens"]
                    wm = row["wall_ms"]
                    tail = (f"out_tok={ot if ot is not None else '?':>5} "
                            f"wall={wm/1000:.1f}s" if wm else "")
                    print(f"{combo:30} {task:12} {'P' if ok else 'F'} {tail}", flush=True)
            print(scoreboard(results), flush=True)

    print(scoreboard(results), flush=True)


if __name__ == "__main__":
    main()

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
    python scripts/claude_version_sweep.py --efforts low high max   # version×effort grid (opt-in)
    python scripts/claude_version_sweep.py --include-1m     # also sweep [1m] opus variants (opt-in)
    python scripts/claude_version_sweep.py --tasks calc3 candy --repeats 2
    python scripts/claude_version_sweep.py --models claude-opus-4-8 claude-opus-4-7

On effort & [1m] (see --help): the default sweep holds effort fixed per tier so
the only variable is the model VERSION. On the saturated brutal tasks effort and
[1m] do not move pass-rate — they shift tokens/latency only — so --efforts is a
token/floor probe and --include-1m is near-redundant here (1m matters on
long-context work, not short tasks). Both are opt-in and off by default.
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

# Valid headless `--effort` values (`claude --help`, verified 2026-05-29).
# `ultracode` is deliberately ABSENT: it is an interactive-UI-only effort level
# and the headless flag rejects it ("must be one of: low, medium, high, xhigh,
# max"). Validating against this set fails fast on typos AND documents the limit.
VALID_EFFORTS = {"low", "medium", "high", "xhigh", "max"}

# The `[1m]` 1M-context suffix works on opus 4.7+ only (verified convention).
# WHY IT'S OPT-IN AND USUALLY POINTLESS HERE: on the short brutal tasks 1m is
# near-redundant — same pass-rate and ~same token count as the base model, since
# nothing uses the extra context window. Its real signal is long-context
# workloads (big diff-feedback, master-mode long runs), not this sweep.
_1M_CAPABLE_PREFIXES = ("claude-opus-4-7", "claude-opus-4-8")


def _resolve_effort(model: str, default_effort: str | None,
                    override: str | None) -> str | None:
    """Effort for a given model: override wins, but never apply effort to a
    model that rejects the flag."""
    if model in _NO_EFFORT_MODELS:
        return None
    return override if override is not None else default_effort


def _expand_1m(model: str) -> list[str]:
    """Base model, plus its `[1m]` variant when the model supports it."""
    if "[1m]" in model or not model.startswith(_1M_CAPABLE_PREFIXES):
        return [model]
    return [model, model + "[1m]"]


def _build_grid(models: list[str] | None, effort_override: str | None,
                efforts: list[str] | None, include_1m: bool
                ) -> list[tuple[str, str, str | None]]:
    """Return [(worker, model, effort), ...] for the requested claude versions.

    - ``efforts`` (a list): sweep each model across every listed effort — a
      version×effort grid. Haiku collapses to one no-effort row; dups removed.
    - ``effort_override`` (single): one effort for all (ignored when ``efforts``).
    - ``include_1m``: also add the ``[1m]`` variant of every 1m-capable opus.
    """
    if models:
        # User passed explicit full ids; carry no per-model default effort, so
        # rely on the override/efforts (or None). Unknown models still get the
        # haiku guard via _resolve_effort.
        pairs = [(m, None) for m in models]
    else:
        pairs = list(KNOWN_CLAUDE_MODELS)

    if include_1m:
        pairs = [(variant, eff) for m, eff in pairs for variant in _expand_1m(m)]

    if efforts:
        rows = [("claude", m, _resolve_effort(m, eff, None))
                for m, _default in pairs for eff in efforts]
    else:
        rows = [("claude", m, _resolve_effort(m, default_eff, effort_override))
                for m, default_eff in pairs]

    # Dedup — haiku collapses every effort to a single None row.
    seen: set = set()
    return [r for r in rows if not (r in seen or seen.add(r))]


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
    ap.add_argument("--efforts", nargs="*", default=None,
                    help="Sweep each opus/sonnet model across THESE efforts "
                         "(version×effort grid; e.g. --efforts low high max). "
                         "Overrides --effort. CAVEAT: on the saturated brutal "
                         "tasks effort moves tokens/latency, NOT pass-rate — "
                         "higher effort is wasted, lower effort is a "
                         "quality-floor finder. For the token frontier proper "
                         "use scripts/token_efficiency.py.")
    ap.add_argument("--include-1m", action="store_true",
                    help="Also sweep the [1m] 1M-context variant of every "
                         "1m-capable opus model (4.7+). NEAR-REDUNDANT on the "
                         "short brutal tasks (same pass-rate + tokens as the "
                         "base model); meaningful only on long-context work.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Repeats per (model,task); medians reported.")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--out", default="/tmp/agentorch_research/claude_version_sweep.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan + resolved `claude --model` per row; spawn nothing.")
    args = ap.parse_args()

    bad = [e for e in (args.efforts or []) + ([args.effort] if args.effort else [])
           if e not in VALID_EFFORTS]
    if bad:
        ap.error(f"invalid effort(s) {bad}; valid: {sorted(VALID_EFFORTS)}. "
                 "('ultracode' is interactive-UI-only and rejected by the headless --effort flag.)")

    grid = _build_grid(args.models, args.effort, args.efforts, args.include_1m)
    n_calls = len(grid) * len(args.tasks) * args.repeats

    print("claude version sweep")
    print(f"  models : {len(grid)}   tasks: {len(args.tasks)}   repeats: {args.repeats}"
          f"   = {n_calls} SEQUENTIAL calls")
    print("  ⚠️  account-sharing: claude-opus-4-8 shares the pool of a 4.8 driving "
          "session (see module docstring).")
    print("  rows:")
    for worker, model, effort in grid:
        eff = effort if effort else "(no --effort)"
        print(f"    claude --model {model:30} effort={eff}")
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

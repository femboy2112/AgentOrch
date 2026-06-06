#!/usr/bin/env python3
"""Calibration sweep: completion_time / tokens / quality = f(prompt features, model, effort).

Sweeps configs over tasks from scripts/task_synth.py using a ONE-FACTOR-AT-A-TIME design
(vary each axis from a baseline) so each crossover — e.g. the n_units at which sonnet:medium
overtakes opus:high — is directly visible. Captures real token/cost/api telemetry (codex &
claude only; see token_efficiency.py). Sequential = no contention. Streams JSONL + prints,
per axis, time/tokens/pass for each config.

This is the data behind adaptive start-tier routing: fit best-config = f(features).

  python scripts/calibrate.py --dry-run     # print the plan + feature vectors, no calls
  python scripts/calibrate.py               # run it (spawns codex/claude workers)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agy_orchestrator.core.calibration import research_dir
from scripts.cloud_eval import CLOUD_SUFFIX
from scripts.task_synth import synthesize
from scripts.token_efficiency import run_config

# Configs of interest: the speed/quality frontier incl. the sonnet:medium you observed.
DEFAULT_GRID = [
    ("claude", "opus", "high"),
    ("claude", "sonnet", "medium"),
    ("claude", "sonnet", "low"),
    ("codex", "gpt-5.4-mini", "low"),
    ("codex", "gpt-5.5", "high"),
]


def build_plan() -> list[dict]:
    """One-factor-at-a-time around baseline (n_units=3, mixed, single_file)."""
    plan: list[dict] = []
    seen = set()

    def add(axis, n, diff, struct, seed=0):
        key = (n, diff, struct, seed)
        if key in seen:
            # still tag the baseline under each axis it anchors
            plan.append({"axis": axis, "n_units": n, "difficulty": diff,
                         "structure": struct, "seed": seed})
            return
        seen.add(key)
        plan.append({"axis": axis, "n_units": n, "difficulty": diff,
                     "structure": struct, "seed": seed})

    # Scale axis holds difficulty FIXED (hard) so output-scale isn't confounded by the
    # per-unit difficulty drift that 'mixed' introduces as n_units grows.
    for n in (1, 3, 6, 10):
        add("scale", n, "hard", "single_file")
    for diff in ("easy", "hard", "brutal"):
        add("difficulty", 3, diff, "single_file")
    for struct in ("single_file", "multi_file", "interdependent"):
        add("structure", 3, "mixed", struct)
    return plan


def label(worker, model, effort) -> str:
    return f"{worker}:{model}" + (f":{effort}" if effort else "")


def analyze(rows: list[dict]) -> str:
    """Per-axis tables of wall-time / output-tokens / pass per config, so crossovers
    (e.g. n_units where sonnet:medium overtakes opus:high) are directly visible."""
    from statistics import mean
    combos = sorted({r["combo"] for r in rows})
    out = ["", "=" * 70, "CALIBRATION ANALYSIS — wall_s (out_tok, pass) per config", "=" * 70]
    for axis, key, levels in (
        ("scale", "n_units", sorted({r["n_units"] for r in rows if r["axis"] == "scale"})),
        ("difficulty", "difficulty",
         [d for d in ("easy", "hard", "brutal") if any(r["axis"] == "difficulty" and r["difficulty"] == d for r in rows)]),
        ("structure", "structure",
         [s for s in ("single_file", "multi_file", "interdependent") if any(r["axis"] == "structure" and r["structure"] == s for r in rows)]),
    ):
        out.append(f"\n--- {axis} axis ---")
        out.append(f"{key:>14} | " + " | ".join(f"{c:>22}" for c in combos))
        for lvl in levels:
            cells = []
            for c in combos:
                rs = [r for r in rows if r["axis"] == axis and r[key] == lvl and r["combo"] == c]
                if not rs:
                    cells.append(f"{'-':>22}")
                    continue
                wall = mean(r["wall_ms"] for r in rs if r["wall_ms"]) / 1000
                ots = [r["out_tokens"] for r in rs if r["out_tokens"] is not None]
                ot = int(mean(ots)) if ots else 0
                pas = sum(r["ok"] for r in rs)
                cells.append(f"{f'{wall:.0f}s({ot},{pas}/{len(rs)})':>22}")
            out.append(f"{str(lvl):>14} | " + " | ".join(cells))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="construct tasks + print plan/feature vectors; spawn no workers")
    ap.add_argument("--analyze-only", metavar="JSONL",
                    help="re-print the per-axis analysis from an existing results file")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default=str(research_dir() / "calibrate.jsonl"))
    args = ap.parse_args()

    if args.analyze_only:
        rows = [json.loads(line) for line in open(args.analyze_only)]
        print(analyze(rows))
        return

    plan = build_plan()
    print(f"calibration plan: {len(plan)} task-points x {len(DEFAULT_GRID)} configs "
          f"= {len(plan) * len(DEFAULT_GRID)} calls (sequential)\n")
    for spec in plan:
        t = synthesize(spec["n_units"], spec["difficulty"], spec["structure"], spec["seed"])
        spec["_task"] = t
        print(f"  [{spec['axis']:10}] n={spec['n_units']:2} {spec['difficulty']:6} "
              f"{spec['structure']:14} feats={t.features}")

    if args.dry_run:
        print("\n--dry-run: no workers spawned. Plan validated.")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    outf = open(args.out, "a")
    rows: list[dict] = []
    print()
    for spec in plan:
        task = spec["_task"]
        for worker, model, effort in DEFAULT_GRID:
            combo = label(worker, model, effort)
            t0 = time.time()
            try:
                m = run_config(worker, model, effort, task.prompt + CLOUD_SUFFIX, args.timeout)
                passed, total = task.grade(m["text"])
            except Exception as exc:
                m = {"out_tokens": None, "reasoning_tokens": None, "cost_usd": None,
                     "api_ms": None, "wall_ms": (time.time() - t0) * 1000}
                passed, total = 0, 0
                print(f"  ! {combo} {spec['axis']}: {type(exc).__name__}: {exc}")
            row = {"axis": spec["axis"], "combo": combo, "worker": worker,
                   "model": model, "effort": effort,
                   **{k: spec[k] for k in ("n_units", "difficulty", "structure", "seed")},
                   **task.features, "passed": passed, "total": total,
                   "ok": total > 0 and passed == total,
                   "out_tokens": m["out_tokens"], "reasoning_tokens": m["reasoning_tokens"],
                   "cost_usd": m["cost_usd"], "api_ms": m["api_ms"], "wall_ms": m["wall_ms"]}
            outf.write(json.dumps(row) + "\n")
            outf.flush()
            rows.append(row)
            ot = row["out_tokens"]
            wm = row["wall_ms"]
            print(f"[{spec['axis']:10}] n={spec['n_units']:2} {spec['difficulty']:6} "
                  f"{spec['structure']:14} {combo:22} {passed}/{total} "
                  f"out_tok={ot if ot is not None else '?':>5} wall={wm/1000:.1f}s", flush=True)
    outf.close()
    print(analyze(rows))
    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()

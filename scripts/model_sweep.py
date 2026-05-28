#!/usr/bin/env python3
"""Model/effort sweep — find the fastest/highest-quality worker configuration.

Sweeps a grid of (worker, model, effort) combinations over the same objective,
hidden-pytest-graded code tasks used by cloud_eval, measuring BOTH:
  * quality  — fraction of tasks whose generated function passes the hidden tests
  * latency  — wall-clock seconds per task (single-shot, no critic loop)

Each combo runs its tasks concurrently; combos run sequentially to avoid usage
spikes (and because agy serializes itself via the settings.json model lock).
Results stream to a JSONL file as they complete, and a ranked scoreboard prints
at the end (and after every combo, so partial runs are still useful).

Usage:
    python scripts/model_sweep.py                      # default grid + tasks
    python scripts/model_sweep.py --tasks roman_to_int balanced eval_expr
    python scripts/model_sweep.py --timeout 240 --out /tmp/sweep.jsonl
    python scripts/model_sweep.py --grid codex:gpt-5.5:low claude:sonnet:high
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agy_orchestrator.core.agents.agy_agent import AgyAgent
from agy_orchestrator.core.agents.claude_agent import ClaudeAgent
from agy_orchestrator.core.agents.codex_agent import CodexAgent
from agy_orchestrator.core.agents.grok_agent import GrokAgent
from scripts.cloud_eval import CLOUD_SUFFIX, EASY_TASKS, HARD_TASKS, extract_code, run_test

ALL_TASKS = {**EASY_TASKS, **HARD_TASKS}

# Default sweep grid (worker, model, effort). Kept modest to bound cost; expand
# via --grid. Models/efforts are the empirically-verified ones (verified_models.md).
DEFAULT_GRID = [
    ("codex", "gpt-5.5", "low"),
    ("codex", "gpt-5.5", "high"),
    ("codex", "gpt-5.3-codex", "medium"),
    ("codex", "gpt-5.4-mini", "low"),
    ("claude", "haiku", "low"),
    ("claude", "sonnet", "low"),
    ("claude", "sonnet", "high"),
    ("claude", "opus", "high"),
    ("grok", "grok-build", None),
    ("agy", "flash", "low"),
    ("agy", "flash", "medium"),
    ("agy", "pro", "high"),
    ("agy", "opus", None),
]

DEFAULT_TASKS = ["roman_to_int", "balanced", "merge_intervals", "eval_expr"]


def make_agent(worker: str, model: str, effort, prompt: str):
    if worker == "codex":
        return CodexAgent(prompt=prompt, model=model, effort=effort)
    if worker == "claude":
        return ClaudeAgent(prompt=prompt, model=model, effort=effort)
    if worker == "grok":
        return GrokAgent(prompt=prompt, model=model)
    if worker == "agy":
        return AgyAgent(prompt=prompt, model=model, effort=effort)
    raise ValueError(f"unknown worker: {worker}")


def label_of(worker, model, effort) -> str:
    return f"{worker}:{model}" + (f":{effort}" if effort else "")


async def run_one(worker, model, effort, task, prompt, test_src) -> dict:
    """One generation + grade. Never raises."""
    t0 = time.time()
    try:
        agent = make_agent(worker, model, effort, prompt + CLOUD_SUFFIX)
        raw = await agent.run_async()
        code = extract_code(raw)
        ok, tail = run_test(code, test_src)
        return {"worker": worker, "model": model, "effort": effort, "task": task,
                "ok": ok, "t": round(time.time() - t0, 1),
                "empty": not code.strip(), "tail": "" if ok else tail[:120]}
    except Exception as exc:
        return {"worker": worker, "model": model, "effort": effort, "task": task,
                "ok": False, "t": round(time.time() - t0, 1), "empty": False,
                "tail": f"{type(exc).__name__}: {exc}"[:120]}


def scoreboard(results: list, n_tasks: int) -> str:
    by_combo: dict = {}
    for r in results:
        key = label_of(r["worker"], r["model"], r["effort"])
        by_combo.setdefault(key, []).append(r)
    rows = []
    for key, rs in by_combo.items():
        passed = sum(x["ok"] for x in rs)
        avg = sum(x["t"] for x in rs) / max(len(rs), 1)
        rows.append((passed / len(rs), passed, len(rs), avg, key))
    # Rank: quality desc, then latency asc.
    rows.sort(key=lambda x: (-x[0], x[3]))
    out = ["", "=== scoreboard (rank: pass-rate desc, then avg latency asc) ===",
           f"{'combo':32} {'pass':>7} {'avg s':>7}"]
    out.append("-" * 50)
    for frac, passed, n, avg, key in rows:
        out.append(f"{key:32} {f'{passed}/{n}':>7} {avg:7.1f}")
    return "\n".join(out)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS, choices=list(ALL_TASKS))
    ap.add_argument("--grid", nargs="*", default=None,
                    help="combos as worker:model[:effort] (default: built-in grid)")
    ap.add_argument("--timeout", type=int, default=300, help="per-call ceiling (AGY_TIMEOUT)")
    ap.add_argument("--out", default="/tmp/agentorch_research/sweep_results.jsonl")
    args = ap.parse_args()
    os.environ["AGY_TIMEOUT"] = str(args.timeout)

    if args.grid:
        grid = []
        for spec in args.grid:
            parts = spec.split(":")
            grid.append((parts[0], parts[1], parts[2] if len(parts) > 2 else None))
    else:
        grid = DEFAULT_GRID

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    outf = open(args.out, "a")
    results: list = []

    print(f"sweep: {len(grid)} combos x {len(args.tasks)} tasks "
          f"({len(grid) * len(args.tasks)} calls), timeout={args.timeout}s\n", flush=True)

    for worker, model, effort in grid:
        key = label_of(worker, model, effort)
        rows = await asyncio.gather(*[
            run_one(worker, model, effort, t, ALL_TASKS[t][0], ALL_TASKS[t][1])
            for t in args.tasks
        ])
        for r in rows:
            outf.write(json.dumps(r) + "\n")
            results.append(r)
        outf.flush()
        passed = sum(r["ok"] for r in rows)
        avg = sum(r["t"] for r in rows) / max(len(rows), 1)
        cells = " ".join(f"{r['task'][:10]}={'P' if r['ok'] else 'F'}/{r['t']:.0f}s" for r in rows)
        print(f"{key:32} {passed}/{len(rows)} avg={avg:5.1f}s  {cells}", flush=True)

    outf.close()
    print(scoreboard(results, len(args.tasks)), flush=True)


if __name__ == "__main__":
    asyncio.run(main())

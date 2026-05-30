#!/usr/bin/env python3
"""Prompt-ablation bench for cloud workers on cloud_eval hidden graders.

Measures whether a verbose orchestrator preamble improves quality enough to
justify extra token cost. Reuses cloud_eval tasks + fractional test scoring.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import roles
from scripts.cloud_eval import (
    ALL_TASKS,
    BRUTAL_TASKS,
    CLOUD_SUFFIX,
    HARD_TASKS,
    extract_code,
    run_test_counts,
)

DEFAULT_TASKS = list(HARD_TASKS) + list(BRUTAL_TASKS)
DEFAULT_EFFORTS = ["low", "high"]
PREAMBLES = ("lean", "verbose")

VERBOSE_PREAMBLE = """You are the primary implementation worker for this task.
Operate with engineering excellence and produce a robust, production-quality answer.
Performance matters: avoid avoidable overhead and choose efficient algorithms.
Correctness is mandatory: honor edge cases, tricky inputs, and exact output semantics.
Respect interfaces exactly as requested: function name, signature, and return type.
Do not invent new files, tools, shell commands, or side channels.
Keep the implementation self-contained and deterministic.
Prefer simple, reliable logic over clever-but-fragile shortcuts.
Include only imports that are truly required by the solution.
Double-check boundary behavior before finalizing.
If multiple approaches work, prefer the clearest maintainable one.
Return only final code, with no extra prose or analysis.
"""


def build_condition_prompt(task_prompt: str, preamble: str) -> str:
    if preamble == "verbose":
        return f"{VERBOSE_PREAMBLE.strip()}\n\nTask:\n{task_prompt}{CLOUD_SUFFIX}"
    if preamble == "lean":
        return f"{task_prompt}{CLOUD_SUFFIX}"
    raise ValueError(f"unknown preamble: {preamble}")


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def summarize_ablation(rows: List[dict]) -> dict:
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for row in rows:
        key = (str(row.get("preamble")), str(row.get("effort")))
        grouped.setdefault(key, []).append(row)

    cells: Dict[Tuple[str, str], dict] = {}
    efforts = sorted({effort for _, effort in grouped})
    for key, group in grouped.items():
        fractions = [float(r.get("fraction", 0.0) or 0.0) for r in group]
        token_vals = [r.get("total_tokens") for r in group if r.get("total_tokens") is not None]
        mean_fraction = _mean(fractions) or 0.0
        mean_tokens = _mean([float(v) for v in token_vals]) if token_vals else None
        cells[key] = {
            "mean_fraction": mean_fraction,
            "mean_tokens": mean_tokens,
            "n": len(group),
        }

    deltas: Dict[str, dict] = {}
    for effort in efforts:
        lean = cells.get(("lean", effort))
        verbose = cells.get(("verbose", effort))
        if lean is None or verbose is None:
            continue
        delta_tokens: Optional[float] = None
        if lean["mean_tokens"] is not None and verbose["mean_tokens"] is not None:
            delta_tokens = float(verbose["mean_tokens"]) - float(lean["mean_tokens"])
        deltas[effort] = {
            "delta_fraction": float(verbose["mean_fraction"]) - float(lean["mean_fraction"]),
            "delta_tokens": delta_tokens,
        }
    return {"cells": cells, "deltas": deltas}


def _fmt_num(val: Optional[float], *, digits: int) -> str:
    if val is None:
        return "n/a"
    return f"{val:.{digits}f}"


def _fmt_tokens(val: Optional[float]) -> str:
    if val is None:
        return "n/a"
    return str(int(round(val)))


def render_summary(summary: dict) -> str:
    cells = summary.get("cells", {})
    deltas = summary.get("deltas", {})
    efforts = sorted({effort for _, effort in cells})
    lines = ["", "=== prompt ablation summary ==="]
    lines.append(f"{'effort':8} {'preamble':8} {'n':>4} {'mean_frac':>10} {'mean_tokens':>12}")
    lines.append("-" * 50)
    for effort in efforts:
        for preamble in PREAMBLES:
            row = cells.get((preamble, effort))
            if row is None:
                continue
            lines.append(
                f"{effort:8} {preamble:8} {row['n']:4d} "
                f"{_fmt_num(row['mean_fraction'], digits=3):>10} "
                f"{_fmt_tokens(row['mean_tokens']):>12}"
            )
    lines.append("")
    for effort in efforts:
        lean = cells.get(("lean", effort))
        verbose = cells.get(("verbose", effort))
        if lean is None or verbose is None:
            continue
        d = deltas.get(effort, {})
        lines.append(
            "effort={effort}: lean frac={lean_frac} tok={lean_tok} | "
            "verbose frac={verb_frac} tok={verb_tok} | dquality={dq} dtokens={dt}".format(
                effort=effort,
                lean_frac=_fmt_num(lean["mean_fraction"], digits=2),
                lean_tok=_fmt_tokens(lean["mean_tokens"]),
                verb_frac=_fmt_num(verbose["mean_fraction"], digits=2),
                verb_tok=_fmt_tokens(verbose["mean_tokens"]),
                dq=f"{(d.get('delta_fraction') or 0.0):+.2f}",
                dt=("n/a" if d.get("delta_tokens") is None else f"{int(round(d['delta_tokens'])):+d}"),
            )
        )
    return "\n".join(lines)


async def run_one_cell(*, worker: str, task: str, task_prompt: str, test_src: str,
                       preamble: str, effort: str, repeat_idx: int) -> dict:
    t0 = time.time()
    prompt = build_condition_prompt(task_prompt, preamble)
    row = {
        "worker": worker,
        "task": task,
        "preamble": preamble,
        "effort": effort,
        "repeat": repeat_idx,
        "ok": False,
        "fraction": 0.0,
        "passed_cases": 0,
        "total_cases": 0,
        "total_tokens": None,
        "tail": "",
        "t": 0.0,
    }
    try:
        agent = roles.build_role_agent([worker], prompt=prompt, effort=effort, fallback=False)
        raw = await agent.run_async()
        code = extract_code(raw)
        ok, tail, passed_cases, total_cases = run_test_counts(code, test_src, stop_on_first=False)
        frac = (passed_cases / total_cases) if total_cases > 0 else 0.0
        usage = getattr(agent, "last_usage", None)
        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        row.update({
            "ok": bool(ok),
            "fraction": frac,
            "passed_cases": passed_cases,
            "total_cases": total_cases,
            "total_tokens": tokens,
            "tail": "" if ok else str(tail)[-200:],
        })
    except Exception as exc:
        row["tail"] = f"{type(exc).__name__}: {exc}"[-200:]
    row["t"] = round(time.time() - t0, 2)
    return row


async def run(args: argparse.Namespace) -> List[dict]:
    rows: List[dict] = []
    task_names = args.tasks or list(DEFAULT_TASKS)
    print(
        f"prompt ablation: worker={args.worker}, tasks={len(task_names)}, "
        f"efforts={args.efforts}, repeat={args.repeat}, timeout={args.timeout}s",
        flush=True,
    )
    total = len(task_names) * len(PREAMBLES) * len(args.efforts) * args.repeat
    done = 0
    for task in task_names:
        task_prompt, test_src = ALL_TASKS[task]
        for preamble in PREAMBLES:
            for effort in args.efforts:
                for rep in range(args.repeat):
                    row = await run_one_cell(
                        worker=args.worker,
                        task=task,
                        task_prompt=task_prompt,
                        test_src=test_src,
                        preamble=preamble,
                        effort=effort,
                        repeat_idx=rep,
                    )
                    rows.append(row)
                    done += 1
                    tok = row["total_tokens"] if row["total_tokens"] is not None else "?"
                    print(
                        f"[{done:03d}/{total:03d}] {task:12} {preamble:7} {effort:5} "
                        f"frac={row['fraction']:.2f} tok={tok} t={row['t']:.1f}s",
                        flush=True,
                    )
                    if row["tail"]:
                        print(f"    tail: {row['tail']}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", default="codex", help="worker token (default: codex)")
    ap.add_argument("--tasks", nargs="*", default=list(DEFAULT_TASKS), choices=list(ALL_TASKS))
    ap.add_argument("--efforts", nargs="*", default=list(DEFAULT_EFFORTS),
                    help="effort levels to sweep (default: low high)")
    ap.add_argument("--repeat", type=int, default=3,
                    help="independent repeats per (task,preamble,effort) cell")
    ap.add_argument("--timeout", type=int, default=300,
                    help="per-call timeout seconds (sets AGY_TIMEOUT)")
    ap.add_argument("--out", default="ablation_results.json",
                    help="output JSON path (default: ./ablation_results.json)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["AGY_TIMEOUT"] = str(args.timeout)
    rows = asyncio.run(run(args))
    out_path = Path(args.out)
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary = summarize_ablation(rows)
    print(render_summary(summary), flush=True)
    print(f"\nresults written: {out_path}", flush=True)


if __name__ == "__main__":
    main()

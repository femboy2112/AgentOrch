#!/usr/bin/env python3
"""Efficiency bench for cloud workers on complexity-budgeted tasks.

Generates one solution per worker/task/nudge condition, grades correctness with
the existing hidden pytest oracle, then grades asymptotic behavior with the
subprocess-isolated complexity harness.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agy_orchestrator.core.calibration import research_dir
from harness import roles
from harness.dispatch import COMPLEXITY_MANDATE
from scripts.cloud_eval import (
    ALL_TASKS,
    CLOUD_SUFFIX,
    COMPLEXITY_BUDGETS,
    extract_code,
    grade_complexity,
    run_test,
)

NUDGES = ("off", "on")


def build_prompt(task_prompt: str, nudge: str) -> str:
    if nudge == "off":
        return task_prompt + CLOUD_SUFFIX
    if nudge == "on":
        return task_prompt + COMPLEXITY_MANDATE + CLOUD_SUFFIX
    raise ValueError(f"unknown nudge: {nudge}")


async def run_one_measurement(worker: str, task: str, nudge: str, timeout: int) -> dict:
    """Generate, correctness-grade, and complexity-grade one cell. Never raises."""
    prompt, test_src = ALL_TASKS[task]
    t0 = time.time()
    row = {
        "worker": worker,
        "task": task,
        "nudge": nudge,
        "correct": False,
        "empty": False,
        "tail": "",
        "t": 0.0,
        "complexity": {
            "applicable": True,
            "ok_import": False,
            "within_budget": None,
            "label": "error",
            "exponent": None,
            "timed_out": False,
            "notes": "",
        },
    }
    try:
        agent = roles.build_role_agent([worker], prompt=build_prompt(prompt, nudge), fallback=False)
        raw = await agent.run_async()
        code = extract_code(raw)
        ok, tail = run_test(code, test_src)
        complexity = grade_complexity(code, task, hard_timeout=timeout)
        row.update({
            "correct": bool(ok),
            "empty": not code.strip(),
            "tail": "" if ok else str(tail)[-200:],
            "complexity": complexity,
        })
    except Exception as exc:
        row["tail"] = f"{type(exc).__name__}: {exc}"[-200:]
    row["t"] = round(time.time() - t0, 2)
    return row


def _label(row: dict | None) -> str:
    if row is None:
        return "-"
    cx = row.get("complexity", {})
    label = cx.get("label") or "?"
    exponent = cx.get("exponent")
    if exponent is None:
        return str(label)
    return f"{label}({float(exponent):.2f})"


def _correct(row: dict | None) -> str:
    if row is None:
        return "-"
    return "PASS" if row.get("correct") else "FAIL"


def _within(row: dict | None) -> str:
    if row is None:
        return "-"
    val = row.get("complexity", {}).get("within_budget")
    if val is True:
        return "PASS"
    if val is False:
        return "FAIL"
    return "n/a"


def _delta(off: dict | None, on: dict | None) -> str:
    if off is None or on is None:
        return "-"
    off_ok = off.get("complexity", {}).get("within_budget")
    on_ok = on.get("complexity", {}).get("within_budget")
    if off_ok is False and on_ok is True:
        return "PASS delta"
    if off_ok is True and on_ok is False:
        return "FAIL delta"
    if off_ok == on_ok:
        return "same"
    return "mixed"


def print_table(rows: List[dict], nudge_mode: str) -> None:
    grouped: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault((row["worker"], row["task"]), {})[row["nudge"]] = row

    hdr = (
        f"{'worker':10} {'task':16} {'correct':15} "
        f"{'big-O(off)':18} {'big-O(on)':18} {'within_budget':17} {'delta':10}"
    )
    print("\n" + hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for worker, task in sorted(grouped):
        cells = grouped[(worker, task)]
        off = cells.get("off")
        on = cells.get("on")
        if nudge_mode == "off":
            correct = _correct(off)
            within = _within(off)
        elif nudge_mode == "on":
            correct = _correct(on)
            within = _within(on)
        else:
            correct = f"off:{_correct(off)} on:{_correct(on)}"
            within = f"off:{_within(off)} on:{_within(on)}"
        print(
            f"{worker:10} {task:16} {correct:15} "
            f"{_label(off):18} {_label(on):18} {within:17} {_delta(off, on):10}",
            flush=True,
        )


async def run(args: argparse.Namespace) -> List[dict]:
    task_names = args.tasks or list(COMPLEXITY_BUDGETS)
    nudges = list(NUDGES if args.nudge == "both" else (args.nudge,))
    rows: List[dict] = []
    out_path = research_dir() / "complexity_eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"complexity bench: workers={args.workers}, tasks={len(task_names)}, "
        f"nudge={args.nudge}, timeout={args.timeout}s",
        flush=True,
    )

    with out_path.open("a", encoding="utf-8") as outf:
        total = len(args.workers) * len(task_names) * len(nudges)
        done = 0
        for task in task_names:
            for nudge in nudges:
                batch = await asyncio.gather(*[
                    run_one_measurement(worker, task, nudge, args.timeout)
                    for worker in args.workers
                ])
                for row in batch:
                    done += 1
                    rows.append(row)
                    outf.write(json.dumps(row, sort_keys=True) + "\n")
                    outf.flush()
                    print(
                        f"[{done:03d}/{total:03d}] {row['worker']:8} {task:16} "
                        f"nudge={nudge:3} correct={_correct(row):4} "
                        f"bigO={_label(row):18} budget={_within(row)}",
                        flush=True,
                    )
                    if row["tail"]:
                        print(f"    tail: {row['tail']}", flush=True)
                    notes = row.get("complexity", {}).get("notes")
                    if notes and row.get("complexity", {}).get("within_budget") is not True:
                        print(f"    complexity: {str(notes)[-200:]}", flush=True)

    print_table(rows, args.nudge)
    print(f"\nresults appended: {out_path}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", nargs="*", default=["grok", "claude", "codex", "agy"],
                    help="cloud workers to compare (default: grok claude codex agy)")
    ap.add_argument("--tasks", nargs="*", choices=list(COMPLEXITY_BUDGETS),
                    help="subset of complexity-budgeted tasks")
    ap.add_argument("--timeout", type=int, default=300,
                    help="per-generation and per-complexity wall-clock ceiling")
    ap.add_argument("--nudge", choices=["off", "on", "both"], default="both",
                    help="prompt condition to run (default: both)")
    args = ap.parse_args()
    os.environ["AGY_TIMEOUT"] = str(args.timeout)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

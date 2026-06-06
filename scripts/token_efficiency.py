#!/usr/bin/env python3
"""Token/cost/speed efficiency bench for codex & claude on hidden-pytest tasks.

WHY THIS EXISTS (separate from model_sweep.py): model_sweep measures end-to-end
wall-clock, which on saturated tasks is dominated by per-CLI startup + agentic
overhead + account contention, NOT model work — so it cannot answer "which config
gives correct answers with the FEWEST tokens, fastest". This bench parses the raw
CLI JSON to capture the real signals:

  * output_tokens       — total generated tokens (includes reasoning); the cleanest
                          cross-model "work to produce the answer". LOWER = cheaper/faster.
  * reasoning_tokens    — codex only: the slice spent on hidden reasoning ("effort").
  * cost_usd            — claude only: real dollar cost the CLI reports.
  * api_ms              — claude only: server inference time (excludes CLI overhead).
  * wall_ms             — full subprocess wall-clock (for reference).
  * ok                  — passes the hidden pytest suite.

Only codex and claude expose token telemetry; grok and agy print text only, so they
are out of scope here (compare them on model_sweep wall-clock instead).

Runs SOLO and SEQUENTIALLY (one call at a time) so no concurrent load contaminates
timing. Streams JSONL; prints an efficiency scoreboard ranked by output_tokens.

Usage:
    python scripts/token_efficiency.py
    python scripts/token_efficiency.py --tasks calc3 candy --repeats 2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agy_orchestrator.core.calibration import research_dir
from scripts.cloud_eval import BRUTAL_TASKS, CLOUD_SUFFIX, extract_code, run_test

# (worker, model, effort). ALL currently-valid codex+claude combos (the only two
# workers that expose token telemetry) so a from-scratch run maps the full
# token/cost-vs-effort curve per model. Narrow with --grid to bound cost. Kept in
# sync with the live rosters (codex ~/.codex/models_cache.json; claude --effort):
#   codex  : gpt-5.5 / gpt-5.4 / gpt-5.4-mini / gpt-5.3-codex-spark x low/medium/
#            high/xhigh. ("minimal" is NOT in supported_reasoning_levels anymore.)
#   claude : haiku / sonnet / opus x low/medium/high/xhigh/max (server gates the
#            top tiers per model; unsupported pairs record a failed row).
DEFAULT_GRID = [
    # codex
    ("codex", "gpt-5.5", "low"),
    ("codex", "gpt-5.5", "medium"),
    ("codex", "gpt-5.5", "high"),
    ("codex", "gpt-5.5", "xhigh"),
    ("codex", "gpt-5.4", "low"),
    ("codex", "gpt-5.4", "medium"),
    ("codex", "gpt-5.4", "high"),
    ("codex", "gpt-5.4", "xhigh"),
    ("codex", "gpt-5.4-mini", "low"),
    ("codex", "gpt-5.4-mini", "medium"),
    ("codex", "gpt-5.4-mini", "high"),
    ("codex", "gpt-5.4-mini", "xhigh"),
    ("codex", "gpt-5.3-codex-spark", "low"),
    ("codex", "gpt-5.3-codex-spark", "medium"),
    ("codex", "gpt-5.3-codex-spark", "high"),
    ("codex", "gpt-5.3-codex-spark", "xhigh"),
    # claude
    ("claude", "haiku", "low"),
    ("claude", "haiku", "medium"),
    ("claude", "haiku", "high"),
    ("claude", "haiku", "xhigh"),
    ("claude", "haiku", "max"),
    ("claude", "sonnet", "low"),
    ("claude", "sonnet", "medium"),
    ("claude", "sonnet", "high"),
    ("claude", "sonnet", "xhigh"),
    ("claude", "sonnet", "max"),
    ("claude", "opus", "low"),
    ("claude", "opus", "medium"),
    ("claude", "opus", "high"),
    ("claude", "opus", "xhigh"),
    ("claude", "opus", "max"),
]

# Bench aliases pin bare names to a dated model for reproducibility. Full version
# strings (e.g. "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6") pass
# through unchanged via .get(model, model), so `--grid claude:claude-opus-4-8:high`
# sweeps a specific version. "opus" tracks the current CLI default tier (4.8 as of
# 2026-05-28); pin the full id when you need a fixed version across releases.
CLAUDE_ALIAS = {"haiku": "claude-haiku-4-5", "sonnet": "claude-sonnet-4-6",
                "opus": "claude-opus-4-8", "standard": "claude-sonnet-4-6"}


def _reported_output_tokens(raw, text: str):
    """Return output_tokens, or None when the telemetry is unreliable.

    The claude CLI (and occasionally codex) sometimes reports ``output_tokens=0``
    — or omits it — on a call that plainly produced output (non-empty text, real
    cost/api time). A literal 0 is SILENTLY counted by the efficiency median/min
    (``_med`` skips only None, not 0), so a phantom zero understates the frontier
    and can make a normal run look "free". When the response text is non-empty
    but the count is 0/missing, report None (unknown) so the scoreboard skips it
    rather than trusting the zero. A genuinely empty response keeps its 0/None."""
    if not raw and text.strip():
        return None
    return raw


def _run(cmd: list[str], stdin: str, timeout: float,
         cwd: str | None = None) -> tuple[str, str, float]:
    t0 = time.time()
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True,
                          timeout=timeout, cwd=cwd)
    return proc.stdout, proc.stderr, (time.time() - t0) * 1000.0


def run_claude(model: str, effort, prompt: str, timeout: float,
               cwd: str | None = None) -> dict:
    cmd = ["claude", "-p", "-", "--output-format", "json",
           "--dangerously-skip-permissions", "--model", CLAUDE_ALIAS.get(model, model)]
    if effort:
        cmd += ["--effort", str(effort)]
    out, err, wall = _run(cmd, prompt, timeout, cwd=cwd)
    text, usage, cost, api_ms = "", {}, None, None
    try:
        blobs = json.loads(out)
        for b in (blobs if isinstance(blobs, list) else [blobs]):
            if isinstance(b, dict) and b.get("type") == "result":
                text = b.get("result", "") or ""
                usage = b.get("usage", {}) or {}
                cost = b.get("total_cost_usd")
                api_ms = b.get("duration_api_ms")
    except Exception:
        pass
    return {"text": text,
            "out_tokens": _reported_output_tokens(usage.get("output_tokens"), text),
            "reasoning_tokens": None, "in_tokens": usage.get("input_tokens"),
            "cost_usd": cost, "api_ms": api_ms, "wall_ms": wall}


def run_codex(model: str, effort, prompt: str, timeout: float,
              cwd: str | None = None) -> dict:
    cmd = ["codex", "exec", "--skip-git-repo-check",
           "--dangerously-bypass-approvals-and-sandbox", "--json", "--model", model]
    if effort:
        cmd += ["-c", f'model_reasoning_effort="{effort}"']
    cmd.append("-")
    out, err, wall = _run(cmd, prompt, timeout, cwd=cwd)
    text, usage = "", {}
    msgs: list[str] = []
    for line in out.splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "turn.completed":
            usage = ev.get("usage", {}) or {}
        item = ev.get("item") or ev.get("msg") or {}
        if isinstance(item, dict):
            t = item.get("text") or item.get("message")
            if isinstance(t, str) and ("```" in t or "def " in t):
                msgs.append(t)
    if msgs:
        text = max(msgs, key=len)
    return {"text": text,
            "out_tokens": _reported_output_tokens(usage.get("output_tokens"), text),
            "reasoning_tokens": usage.get("reasoning_output_tokens"),
            "in_tokens": usage.get("input_tokens"), "cost_usd": None,
            "api_ms": None, "wall_ms": wall}


def run_config(worker, model, effort, prompt, timeout) -> dict:
    """Run a single config in a per-call SANDBOX cwd.

    Workers carry --dangerously-* flags that let them write files directly. If
    we don't isolate cwd, multi-file prompts (e.g. task_synth multi_file /
    interdependent) get their `# file: mod_0.py` outputs written into the REPO
    ROOT instead of returned as response text — polluting the tree AND giving
    the sandbox grader 0/0 because the response text contains no file blocks.
    A throwaway tempdir per call fixes both at once.
    """
    with tempfile.TemporaryDirectory(prefix="agentorch-bench-") as cwd:
        if worker == "claude":
            return run_claude(model, effort, prompt, timeout, cwd=cwd)
        if worker == "codex":
            return run_codex(model, effort, prompt, timeout, cwd=cwd)
        raise ValueError(f"token bench supports codex/claude only, not {worker}")


def label(worker, model, effort) -> str:
    return f"{worker}:{model}" + (f":{effort}" if effort else "")


def _med(xs):
    xs = [x for x in xs if x is not None]
    return median(xs) if xs else None


def scoreboard(results: list[dict]) -> str:
    by: dict[str, list[dict]] = {}
    for r in results:
        by.setdefault(r["combo"], []).append(r)
    rows = []
    for combo, rs in by.items():
        passed = sum(r["ok"] for r in rs)
        rows.append({
            "combo": combo, "pass": f"{passed}/{len(rs)}", "passed": passed, "n": len(rs),
            "out_tok": _med([r["out_tokens"] for r in rs]),
            "reason_tok": _med([r["reasoning_tokens"] for r in rs]),
            "cost": _med([r["cost_usd"] for r in rs]),
            "api_ms": _med([r["api_ms"] for r in rs]),
            "wall_ms": _med([r["wall_ms"] for r in rs]),
        })
    # Rank: quality desc, then median output tokens asc (the efficiency frontier).
    rows.sort(key=lambda x: (-x["passed"], x["out_tok"] if x["out_tok"] is not None else 9e9))
    out = ["", "=== efficiency scoreboard (rank: pass desc, then median output_tokens asc) ===",
           f"{'combo':26}{'pass':>6}{'out_tok':>9}{'reason':>8}{'cost$':>9}{'api_s':>7}{'wall_s':>8}",
           "-" * 73]
    for r in rows:
        cost = f"{r['cost']:.4f}" if r["cost"] is not None else "-"
        api = f"{r['api_ms']/1000:.1f}" if r["api_ms"] is not None else "-"
        wall = f"{r['wall_ms']/1000:.1f}" if r["wall_ms"] is not None else "-"
        reason = f"{int(r['reason_tok'])}" if r["reason_tok"] is not None else "-"
        otok = f"{int(r['out_tok'])}" if r["out_tok"] is not None else "-"
        out.append(f"{r['combo']:26}{r['pass']:>6}{otok:>9}{reason:>8}{cost:>9}{api:>7}{wall:>8}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="*", default=list(BRUTAL_TASKS), choices=list(BRUTAL_TASKS))
    ap.add_argument("--grid", nargs="*", default=None, help="worker:model[:effort] combos")
    ap.add_argument("--repeats", type=int, default=1, help="repeats per (combo,task); medians reported")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--out", default=str(research_dir() / "token_efficiency.jsonl"))
    args = ap.parse_args()

    if args.grid:
        grid = []
        for spec in args.grid:
            p = spec.split(":")
            grid.append((p[0], p[1], p[2] if len(p) > 2 else None))
    else:
        grid = DEFAULT_GRID

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    outf = open(args.out, "a")
    results: list[dict] = []
    print(f"token bench: {len(grid)} combos x {len(args.tasks)} tasks x {args.repeats} "
          f"= {len(grid)*len(args.tasks)*args.repeats} SEQUENTIAL calls\n", flush=True)

    for worker, model, effort in grid:
        combo = label(worker, model, effort)
        for task in args.tasks:
            prompt, test_src = BRUTAL_TASKS[task]
            for _ in range(args.repeats):
                try:
                    m = run_config(worker, model, effort, prompt + CLOUD_SUFFIX, args.timeout)
                    code = extract_code(m["text"])
                    ok, _ = run_test(code, test_src)
                except Exception as exc:
                    m = {"text": "", "out_tokens": None, "reasoning_tokens": None,
                         "in_tokens": None, "cost_usd": None, "api_ms": None, "wall_ms": None}
                    ok, code = False, ""
                    print(f"  ! {combo} {task}: {type(exc).__name__}: {exc}", flush=True)
                row = {"combo": combo, "worker": worker, "model": model, "effort": effort,
                       "task": task, "ok": ok, "empty": not code.strip(), **{
                           k: m[k] for k in ("out_tokens", "reasoning_tokens", "in_tokens",
                                             "cost_usd", "api_ms", "wall_ms")}}
                outf.write(json.dumps(row) + "\n")
                outf.flush()
                results.append(row)
                ot = row["out_tokens"]
                wm = row["wall_ms"]
                print(f"{combo:26} {task:12} {'P' if ok else 'F'} "
                      f"out_tok={ot if ot is not None else '?':>5} "
                      f"wall={wm/1000:.1f}s" if wm else f"{combo} {task}", flush=True)
        print(scoreboard(results), flush=True)

    outf.close()
    print(scoreboard(results), flush=True)


if __name__ == "__main__":
    main()

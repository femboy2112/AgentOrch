#!/usr/bin/env python3
"""Turn the raw sweep JSONL into decisions: best quality/effort/speed per model.

Reads the research data written by the sweeps (read-only, spends no usage):
  * sweep_results.jsonl   (model_sweep.py)      — quality + wall-clock per
                                                  (worker, model, effort, task)
  * token_efficiency.jsonl (token_efficiency.py) — output_tokens / cost / api_ms
                                                  for codex + claude

and answers the questions that inform our defaults + fallback ordering:
  1. Per config (worker, model, effort): pass-rate, speed, tokens, cost.
  2. Per model: the BEST effort (quality first, then speed) + the full curve, so
     the quality-vs-effort knee is visible.
  3. Per task and per difficulty tier (easy/hard/brutal): the winning config.
  4. A recommended codex fallback ordering (best-first by measured quality, then
     speed) — paste-ready for AGY_CODEX_MODEL_FALLBACKS / _codex_model_fallbacks.

Usage:
    python scripts/sweep_report.py                       # read research_dir() defaults
    python scripts/sweep_report.py --sweep a.jsonl --tokens b.jsonl
    python scripts/sweep_report.py --min-runs 2          # ignore thin (<2 run) cells
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agy_orchestrator.core.calibration import research_dir
from scripts.cloud_eval import BRUTAL_TASKS, EASY_TASKS, HARD_TASKS

# task name -> difficulty tier (for per-tier aggregation)
TIER_OF = {
    **{t: "easy" for t in EASY_TASKS},
    **{t: "hard" for t in HARD_TASKS},
    **{t: "brutal" for t in BRUTAL_TASKS},
}


def _label(worker, model, effort) -> str:
    return f"{worker}:{model}" + (f":{effort}" if effort else "")


def _read_jsonl(path: Path) -> list:
    rows = []
    if not path or not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except (ValueError, TypeError):
            continue  # tolerate a torn final line from a live append
    return rows


def _key(r) -> tuple:
    return (r.get("worker"), r.get("model"), r.get("effort"))


def _agg(rows: list, *, speed_field, speed_scale=1.0) -> dict:
    """Aggregate rows by (worker, model, effort) config."""
    by: dict = {}
    for r in rows:
        if "ok" not in r:
            continue
        by.setdefault(_key(r), []).append(r)
    out = {}
    for k, rs in by.items():
        passed = sum(1 for x in rs if x.get("ok"))
        speeds = [x[speed_field] * speed_scale for x in rs
                  if x.get(speed_field) is not None]
        toks = [x["out_tokens"] for x in rs if x.get("out_tokens") is not None]
        costs = [x["cost_usd"] for x in rs if x.get("cost_usd") is not None]
        out[k] = {
            "worker": k[0], "model": k[1], "effort": k[2],
            "n": len(rs), "passed": passed, "passrate": passed / max(len(rs), 1),
            "speed_med": median(speeds) if speeds else None,
            "speed_avg": mean(speeds) if speeds else None,
            "tok_med": median(toks) if toks else None,
            "cost_med": median(costs) if costs else None,
        }
    return out


def _rank_key(c):
    """Best-first: quality desc, then speed asc (None speed sorts last)."""
    return (-c["passrate"], c["speed_med"] if c["speed_med"] is not None else 9e9)


def _row_speed_s(r):
    """Per-row wall-clock seconds: sweep `t`, else token `wall_ms`."""
    if r.get("t") is not None:
        return r["t"]
    if r.get("wall_ms") is not None:
        return r["wall_ms"] / 1000.0
    return None


def _cand(key, rows: list) -> dict:
    """Build one candidate config from a list of rows (any bench)."""
    passed = sum(1 for x in rows if x.get("ok"))
    speeds = [s for s in (_row_speed_s(x) for x in rows) if s is not None]
    toks = [x["out_tokens"] for x in rows if x.get("out_tokens") is not None]
    costs = [x["cost_usd"] for x in rows if x.get("cost_usd") is not None]
    return {"worker": key[0], "model": key[1], "effort": key[2],
            "n": len(rows), "passed": passed, "passrate": passed / max(len(rows), 1),
            "speed_med": median(speeds) if speeds else None,
            "tok_med": median(toks) if toks else None,
            "cost_med": median(costs) if costs else None}


def _fmt(c) -> str:
    sp = f"{c['speed_med']:.1f}s" if c["speed_med"] is not None else "-"
    tok = f"{int(c['tok_med'])}" if c["tok_med"] is not None else "-"
    cost = f"${c['cost_med']:.4f}" if c["cost_med"] is not None else "-"
    return (f"{_label(c['worker'], c['model'], c['effort']):30} "
            f"q={c['passed']}/{c['n']} ({c['passrate']*100:4.0f}%)  "
            f"{sp:>7}  tok={tok:>6}  {cost:>9}")


def report(sweep_rows: list, token_rows: list, min_runs: int) -> str:
    # QUALITY combines BOTH benches so the saturated easy/hard tasks (model_sweep)
    # don't mask the brutal-task differentiation (token_efficiency). SPEED uses the
    # sweep `t` (the common easy/hard task set every worker ran -> cross-worker
    # comparable), falling back to the token wall_ms only when a config has no sweep
    # rows. Tokens/cost come only from the token bench.
    sw = _agg(sweep_rows, speed_field="t")
    tk = _agg(token_rows, speed_field="wall_ms", speed_scale=1 / 1000.0)
    cfg: dict = {}
    for k in set(sw) | set(tk):
        a, b = sw.get(k), tk.get(k)
        n = (a["n"] if a else 0) + (b["n"] if b else 0)
        passed = (a["passed"] if a else 0) + (b["passed"] if b else 0)
        speed = (a["speed_med"] if a and a["speed_med"] is not None
                 else (b["speed_med"] if b else None))
        cfg[k] = {
            "worker": k[0], "model": k[1], "effort": k[2],
            "n": n, "passed": passed, "passrate": passed / max(n, 1),
            "speed_med": speed, "speed_avg": a["speed_avg"] if a else None,
            "tok_med": b["tok_med"] if b else None,
            "cost_med": b["cost_med"] if b else None,
        }

    cfgs = [c for c in cfg.values() if c["n"] >= min_runs]
    out = []
    if not cfgs:
        return ("No sweep data found (or all cells below --min-runs). Run "
                "scripts/model_sweep.py / scripts/token_efficiency.py first.")

    # 1. Per-config scoreboard.
    out += ["", "=" * 78, "PER-CONFIG  (rank: quality desc, then median speed asc)", "=" * 78]
    for c in sorted(cfgs, key=_rank_key):
        out.append("  " + _fmt(c))

    # 2. Best effort per model + the curve.
    out += ["", "=" * 78, "BEST EFFORT PER MODEL  (quality first, then speed)", "=" * 78]
    by_model: dict = {}
    for c in cfgs:
        by_model.setdefault((c["worker"], c["model"]), []).append(c)
    for (w, m), cs in sorted(by_model.items()):
        best = sorted(cs, key=_rank_key)[0]
        out.append(f"  {w}:{m:24} BEST -> effort={best['effort'] or 'n/a':6} "
                   f"q={best['passrate']*100:.0f}%  "
                   f"{best['speed_med']:.1f}s" if best['speed_med'] is not None
                   else f"  {w}:{m:24} BEST -> effort={best['effort'] or 'n/a'}")
        for c in sorted(cs, key=lambda x: (x["effort"] or "")):
            sp = f"{c['speed_med']:.1f}s" if c["speed_med"] is not None else "-"
            out.append(f"       {c['effort'] or 'n/a':6}  q={c['passrate']*100:4.0f}%  {sp:>7}")

    # 3. Per-task winner.
    out += ["", "=" * 78, "BEST CONFIG PER TASK", "=" * 78]
    by_task: dict = {}
    for r in sweep_rows + token_rows:
        if "ok" not in r or "task" not in r:
            continue
        by_task.setdefault(r["task"], {}).setdefault(_key(r), []).append(r)
    for task in sorted(by_task):
        cands = [_cand(k, rs) for k, rs in by_task[task].items()]
        win = sorted(cands, key=_rank_key)[0]
        n_win = sum(1 for c in cands if c["passrate"] == 1.0)
        tier = TIER_OF.get(task, "?")
        out.append(f"  {task:16} [{tier:6}] -> {_label(win['worker'], win['model'], win['effort']):28} "
                   f"q={win['passrate']*100:.0f}%  ({n_win} configs at 100%)")

    # 4. Per difficulty tier (combined across both benches).
    out += ["", "=" * 78, "BEST CONFIG PER DIFFICULTY TIER", "=" * 78]
    by_tier: dict = {}
    for r in sweep_rows + token_rows:
        if "ok" not in r:
            continue
        tier = TIER_OF.get(r.get("task"), "?")
        by_tier.setdefault(tier, {}).setdefault(_key(r), []).append(r)
    for tier in ("easy", "hard", "brutal", "?"):
        if tier not in by_tier:
            continue
        cands = [_cand(k, rs) for k, rs in by_tier[tier].items()]
        win = sorted(cands, key=_rank_key)[0]
        out.append(f"  {tier:7} -> {_fmt(win)}")

    # 5. Recommended codex fallback ordering (best model first; one entry per model
    #    using that model's best-measured effort). Drives _codex_model_fallbacks.
    out += ["", "=" * 78, "RECOMMENDED CODEX FALLBACK ORDERING (measured, best-first)", "=" * 78]
    codex_best: dict = {}
    for c in cfgs:
        if c["worker"] != "codex":
            continue
        cur = codex_best.get(c["model"])
        if cur is None or _rank_key(c) < _rank_key(cur):
            codex_best[c["model"]] = c
    ordered = sorted(codex_best.values(), key=_rank_key)
    if ordered:
        for c in ordered:
            out.append("  " + _fmt(c))
        models = ",".join(c["model"] for c in ordered)
        out += ["", "  paste-ready (best-first model order):",
                f"    AGY_CODEX_MODEL_FALLBACKS={models}"]
        # Honesty guard: if the working models all tie on quality, this ordering is
        # SPEED-ONLY on saturated tasks and must NOT override codex's capability
        # priority for fallback (it would demote the strongest model). Flag it.
        working = [c for c in ordered if c["passrate"] > 0]
        if working and len({round(c["passrate"], 3) for c in working}) == 1:
            out += ["",
                    "  ⚠ WARNING: every working codex model scores the same quality on "
                    "these tasks",
                    "    (they are too small/saturated to rank CAPABILITY) — so this order "
                    "is speed-only.",
                    "    Do NOT use it to override codex's priority-based fallback for "
                    "hard/large work;",
                    "    it is valid only for picking a fast tier on easy tasks. Re-run "
                    "with a harder",
                    "    task suite (--tasks / bigger cloud_eval tasks) to rank capability."]
        walled = [c["model"] for c in ordered if c["passrate"] == 0]
        if walled:
            out.append(f"    NOTE: {','.join(walled)} scored 0% — likely a usage wall, "
                       "not low quality; re-bench after reset.")
    else:
        out.append("  (no codex rows in the data yet)")

    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default=str(research_dir() / "sweep_results.jsonl"),
                    help="model_sweep.py output JSONL")
    ap.add_argument("--tokens", default=str(research_dir() / "token_efficiency.jsonl"),
                    help="token_efficiency.py output JSONL")
    ap.add_argument("--min-runs", type=int, default=1,
                    help="ignore (worker,model,effort) cells with fewer than N runs")
    args = ap.parse_args()
    sweep_rows = _read_jsonl(Path(args.sweep))
    token_rows = _read_jsonl(Path(args.tokens))
    print(f"# sweep_results: {len(sweep_rows)} rows ({args.sweep})")
    print(f"# token_efficiency: {len(token_rows)} rows ({args.tokens})")
    print(report(sweep_rows, token_rows, args.min_runs))


if __name__ == "__main__":
    main()

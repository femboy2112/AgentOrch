#!/usr/bin/env python3
"""Single-run latency benchmark for the harness (issue: single-run speed audit).

Measures the wall-clock of ONE dispatch and A/Bs a speed-up knob, attributing time
to harness overhead vs (simulated) model time, WITHOUT issuing a single real call
to the codex/agy/grok pool.

Isolation is structural: the benchmark sets ``AGY_BENCH_MOCK=1`` so
``harness.roles._class_for`` swaps every worker for a hermetic ``MockAgent`` (a real
short-lived ``sh -c 'sleep T; printf ...'`` subprocess). No worker CLI is ever
spawned, so a bench run can never contend with a live mission-critical dispatch.

What it proves out of the box (``--scenario baseline-skip``): gating the pre-run
baseline verifier off for non-vote modes removes one full ``--test-cmd`` suite from
the serial critical path while the real in-loop gate — and thus the success signal —
is unchanged.

Each sample runs in a throwaway git repo so snapshotting/path-policy behave normally.
Reported per metric: mean, p50, p95 across N samples, plus the arm-to-arm delta.

Usage:
    python scripts/speed_bench.py --scenario baseline-skip --samples 8
    python scripts/speed_bench.py --scenario baseline-skip --mode adversarial --test-sleep 3
    python scripts/speed_bench.py --scenario cache --samples 8   # models a prefix-cache hit

Token-count metrics are noise-free (derived from the actual prompt each mock call
received), so prompt-bloat / projection findings can be A/B'd on summed input_tokens
even when wall-clock is jittery.
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Make the worktree copy of the package win over any editable install on sys.path.
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _init_repo(path: str) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"],
                   cwd=path, check=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "bench", "GIT_AUTHOR_EMAIL": "b@b",
                        "GIT_COMMITTER_NAME": "bench", "GIT_COMMITTER_EMAIL": "b@b"})


def _sample(dispatch_kwargs: Dict[str, Any], env: Dict[str, str]) -> Dict[str, Any]:
    """Run one dispatch in an isolated repo; return measured metrics."""
    # Apply per-arm env (mock latency / cache ratio) before the import-light call.
    for k, v in env.items():
        os.environ[k] = v
    from harness.dispatch import dispatch  # imported here so AGY_BENCH_MOCK is set

    with tempfile.TemporaryDirectory() as d:
        _init_repo(d)
        t0 = time.monotonic()
        r = dispatch(out_dir=d, heartbeat_interval=0, **dispatch_kwargs)
        wall = time.monotonic() - t0

    tokens = r.tokens or {}
    gt = (tokens.get("grand_total") or {}) if isinstance(tokens, dict) else {}
    return {
        "wall_s": wall,
        "success": 1.0 if r.success else 0.0,
        "worker_calls": float((tokens.get("total_calls") or 0) if isinstance(tokens, dict) else 0),
        "input_tokens": float(gt.get("input_tokens") or 0),
        "cache_read_tokens": float(gt.get("cache_read_tokens") or 0),
        "total_tokens": float(gt.get("total_tokens") or 0),
    }


def _agg(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    s = sorted(values)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    return {"mean": statistics.fmean(values), "p50": statistics.median(values), "p95": p95}


def run_arm(name: str, dispatch_kwargs: Dict[str, Any], env: Dict[str, str],
            samples: int) -> Dict[str, Dict[str, float]]:
    print(f"  arm {name!r}: {samples} samples ...", flush=True)
    rows: List[Dict[str, Any]] = []
    for i in range(samples):
        rows.append(_sample(dispatch_kwargs, env))
    metrics = ("wall_s", "input_tokens", "cache_read_tokens", "worker_calls", "success")
    return {m: _agg([row[m] for row in rows]) for m in metrics}


SCENARIOS: Dict[str, Callable[[argparse.Namespace], Dict[str, Any]]] = {}


def scenario(fn):
    SCENARIOS[fn.__name__.replace("_", "-")] = fn
    return fn


def _base_kwargs(args) -> Dict[str, Any]:
    kw: Dict[str, Any] = {
        "instruction": (_HERE / "scripts/bench_fixtures/instruction.txt").read_text().strip(),
        "mode": args.mode,
        "generator_chain": ["codex"],
        "critic_chain": ["agy"],
        "fallback": False,
        "test_cmd": f"sleep {args.test_sleep}",
    }
    if args.spec:
        kw["spec"] = (_HERE / "scripts/bench_fixtures/spec.md").read_text()
    return kw


@scenario
def baseline_skip(args) -> Dict[str, Any]:
    """A/B the rank-1 win: pre-run baseline verifier gated on vs off."""
    base = _base_kwargs(args)
    env = {"AGY_BENCH_MOCK": "1", "AGY_BENCH_MOCK_SLEEP": str(args.mock_sleep)}
    return {
        "env": env,
        "arms": {
            "baseline-on (today)": {**base, "baseline_gate": True},
            "baseline-skip (rank-1)": {**base, "baseline_gate": False},
        },
    }


@scenario
def cache(args) -> Dict[str, Any]:
    """A/B a prefix-cache hit: cache_read_ratio 0 vs 0.6 (models warm-session reuse)."""
    base = _base_kwargs(args)
    base = {**base, "baseline_gate": False}
    return {
        "env": {"AGY_BENCH_MOCK": "1", "AGY_BENCH_MOCK_SLEEP": str(args.mock_sleep)},
        "arms": {
            "cold (ratio 0.0)": (base, {"AGY_BENCH_MOCK_CACHE_RATIO": "0.0"}),
            "warm (ratio 0.6)": (base, {"AGY_BENCH_MOCK_CACHE_RATIO": "0.6"}),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="baseline-skip")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--mode", default="direct",
                    help="dispatch mode (direct/adversarial/master/...). direct isolates "
                         "the baseline-skip cleanly; adversarial adds the critic loop.")
    ap.add_argument("--test-sleep", type=float, default=2.0,
                    help="seconds the hermetic --test-cmd suite takes (stands in for a "
                         "real pytest/make-check)")
    ap.add_argument("--mock-sleep", type=float, default=0.3,
                    help="simulated model latency per worker call (seconds)")
    ap.add_argument("--spec", action="store_true",
                    help="inject the FloodSpec fixture (for projection/bloat scenarios)")
    args = ap.parse_args()

    spec = SCENARIOS[args.scenario](args)
    base_env = spec["env"]
    print(f"scenario={args.scenario} mode={args.mode} samples={args.samples} "
          f"test_sleep={args.test_sleep}s mock_sleep={args.mock_sleep}s (HERMETIC — no real workers)")

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for arm_name, spec_arm in spec["arms"].items():
        if isinstance(spec_arm, tuple):
            kwargs, arm_env = spec_arm
        else:
            kwargs, arm_env = spec_arm, {}
        results[arm_name] = run_arm(arm_name, kwargs, {**base_env, **arm_env}, args.samples)

    # ---- report ----
    metrics = ["wall_s", "input_tokens", "cache_read_tokens", "worker_calls", "success"]
    arm_names = list(results)
    w = max(len(a) for a in arm_names) + 2
    print("\n" + "=" * 72)
    print(f"{'metric':<20}" + "".join(f"{a:>{w}}" for a in arm_names) + f"{'Δ (p50)':>14}")
    print("-" * 72)
    for m in metrics:
        cells = "".join(f"{results[a][m]['p50']:>{w}.2f}" for a in arm_names)
        if len(arm_names) == 2:
            d = results[arm_names[1]][m]["p50"] - results[arm_names[0]][m]["p50"]
            delta = f"{d:>+14.2f}"
        else:
            delta = " " * 14
        print(f"{m:<20}{cells}{delta}")
    print("-" * 72)
    print("(values are p50 across samples; wall_s in seconds, tokens absolute)")

    # Headline for the default scenario.
    if args.scenario == "baseline-skip" and len(arm_names) == 2:
        on = results[arm_names[0]]["wall_s"]["p50"]
        off = results[arm_names[1]]["wall_s"]["p50"]
        ok_on = results[arm_names[0]]["success"]["p50"]
        ok_off = results[arm_names[1]]["success"]["p50"]
        print(f"\nRANK-1: baseline-skip saved {on - off:.2f}s/run (p50 {on:.2f}s -> {off:.2f}s); "
              f"success preserved ({ok_on:.0f}->{ok_off:.0f}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

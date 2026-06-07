"""Empirical wall-time complexity probe.

Correctness tests cannot distinguish an efficient implementation from a slow one
when both return the right answer. This module measures the *minimum* wall time
(timeit best practice: the least-contended run is the truest cost and yields a far
more stable slope than the mean/median, which absorb positive noise spikes) across a
geometric size ladder and fits a log-log slope as a practical big-O oracle. The
signal is wall time, not line or bytecode counts, because candidate code may hide
real work behind C builtins such as ``sorted`` or set operations.

The n-vs-nlogn boundary is close to normal timing resolution, so classification is
intentionally tolerant and budget checks are lenient when the regression quality is
too low to trust.

``gen(n)`` may return either a tuple of positional arguments, which is splatted into
``fn(*args)``, or one value, which is passed as ``fn(value)``.
"""

import logging
import math
import statistics
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_LOW_CONFIDENCE_R2 = 0.5


@dataclass
class ComplexityResult:
    samples: list[tuple[int, float]]
    exponent: float
    r2: float
    label: str
    within_budget: Optional[bool]
    budget_exponent: Optional[float]
    notes: str


def _call(fn, generated):
    if isinstance(generated, tuple):
        return fn(*generated)
    return fn(generated)


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    if len(xs) < 2:
        return 0.0, ys[0] if ys else 0.0, 0.0

    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    ss_x = sum((x - mean_x) ** 2 for x in xs)
    if ss_x == 0:
        return 0.0, mean_y, 0.0

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / ss_x
    intercept = mean_y - slope * mean_x
    residual = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    total = sum((y - mean_y) ** 2 for y in ys)
    if total == 0:
        r2 = 1.0 if residual == 0 else 0.0
    else:
        r2 = max(0.0, min(1.0, 1.0 - residual / total))
    return slope, intercept, r2


def _single_predictor_r2(xs: list[float], ys: list[float]) -> float:
    """R2 for y = x + c, used for the explicit t = c * n log n check."""
    if len(xs) < 2:
        return 0.0
    intercept = statistics.fmean(y - x for x, y in zip(xs, ys))
    residual = sum((y - (x + intercept)) ** 2 for x, y in zip(xs, ys))
    mean_y = statistics.fmean(ys)
    total = sum((y - mean_y) ** 2 for y in ys)
    if total == 0:
        return 1.0 if residual == 0 else 0.0
    return max(0.0, min(1.0, 1.0 - residual / total))


def classify(samples) -> tuple[float, float, str]:
    """Return ``(exponent, r2, label)`` from ``(n, median_seconds)`` samples."""
    usable = [(int(n), float(seconds)) for n, seconds in samples if n > 1 and seconds > 0]
    if len(usable) < 2:
        return 0.0, 0.0, "O(1)"

    xs = [math.log2(n) for n, _seconds in usable]
    ys = [math.log2(seconds) for _n, seconds in usable]
    exponent, _intercept, r2 = _fit(xs, ys)

    if exponent < 0.3:
        label = "O(1)"
    elif exponent < 0.85:
        label = "O(log n)"
    elif exponent < 1.4:
        label = "O(n)"
        if 1.05 <= exponent < 1.4:
            nlogn_xs = [math.log2(n * math.log2(n)) for n, _seconds in usable]
            if _single_predictor_r2(nlogn_xs, ys) > r2:
                label = "O(n log n)"
    elif exponent < 2.5:
        label = "O(n^2)"
    elif exponent < 3.5:
        label = "O(n^3)"
    else:
        label = "super-polynomial"

    return exponent, r2, label


def check_budget(result, budget_exponent, margin=0.35) -> bool:
    """Return True when the measured exponent fits the budget.

    Low-confidence fits pass with warning semantics; callers should inspect
    ``result.notes`` rather than treating noisy measurements as hard failures.
    """
    if result.r2 < _LOW_CONFIDENCE_R2:
        return True
    return result.exponent <= budget_exponent + margin


def measure_scaling(fn, gen, sizes, *, repeats=5, warmup=1,
                    per_call_timeout=None, budget_exponent=None,
                    margin=0.35) -> ComplexityResult:
    samples: list[tuple[int, float]] = []
    notes: list[str] = []

    for n in sizes:
        stop = False

        for _index in range(warmup):
            try:
                generated = gen(n)
                start = time.perf_counter()
                _call(fn, generated)
                elapsed = time.perf_counter() - start
            except Exception as exc:
                logger.debug("complexity probe warmup failed at n=%s", n, exc_info=True)
                notes.append(f"candidate raised during warmup at n={n}: {exc!r}")
                stop = True
                break
            if per_call_timeout is not None and elapsed > per_call_timeout:
                notes.append(
                    f"per-call timeout at n={n} during warmup: "
                    f"{elapsed:.6f}s > {per_call_timeout:.6f}s"
                )
                stop = True
                break
        if stop:
            break

        timings: list[float] = []
        for _index in range(repeats):
            try:
                generated = gen(n)
                start = time.perf_counter()
                _call(fn, generated)
                elapsed = time.perf_counter() - start
            except Exception as exc:
                logger.debug("complexity probe call failed at n=%s", n, exc_info=True)
                notes.append(f"candidate raised at n={n}: {exc!r}")
                stop = True
                break

            if per_call_timeout is not None and elapsed > per_call_timeout:
                notes.append(
                    f"per-call timeout at n={n}: "
                    f"{elapsed:.6f}s > {per_call_timeout:.6f}s"
                )
                stop = True
                break
            timings.append(elapsed)

        if stop:
            break
        if timings:
            # min, not median: the fastest of the repeats is the run least disturbed
            # by GC/scheduler contention, so the across-size slope is far more stable.
            samples.append((int(n), min(timings)))

    exponent, r2, label = classify(samples)
    if len(samples) < 2:
        notes.append("insufficient samples for a reliable regression")
    elif r2 < _LOW_CONFIDENCE_R2:
        notes.append(f"low-confidence fit (r2={r2:.3f}); treating budget as warning-only")

    within_budget = None
    if budget_exponent is not None:
        within_budget = check_budget(
            ComplexityResult(samples, exponent, r2, label, None, budget_exponent, "; ".join(notes)),
            budget_exponent,
            margin=margin,
        )

    return ComplexityResult(
        samples=samples,
        exponent=exponent,
        r2=r2,
        label=label,
        within_budget=within_budget,
        budget_exponent=budget_exponent,
        notes="; ".join(notes),
    )

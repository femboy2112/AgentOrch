from agy_orchestrator.execution import complexity_probe
from agy_orchestrator.execution.complexity_probe import (
    ComplexityResult,
    check_budget,
    measure_scaling,
)


def constant_index(values):
    return values[len(values) // 2]


def loop_sum(values):
    total = 0
    for value in values:
        total += value
    return total


def builtin_sort(values):
    # This must stay as the C-backed builtin sort reference: the probe is based on
    # wall time specifically so one-line C work is still visible to the oracle.
    return sorted(values)


def count_pairs(values):
    total = 0
    for left in values:
        for right in values:
            total += left <= right
    return total


def add_many(left, right):
    total = 0
    for value in left:
        total += value
    for value in right:
        total += value
    return total


def shuffled_values(n):
    return [((i * 1_103_515_245 + 12_345) & 0x7FFFFFFF) for i in range(n)]


def test_constant_reference_is_constant_or_logarithmic():
    result = measure_scaling(
        constant_index,
        lambda n: list(range(n)),
        [1_000, 2_000, 4_000, 8_000, 16_000],
        repeats=7,
        warmup=1,
    )

    assert result.label in {"O(1)", "O(log n)"}
    assert result.exponent < 0.85


def test_linear_reference_lands_in_linear_bucket():
    result = measure_scaling(
        loop_sum,
        lambda n: list(range(n)),
        [8_000, 16_000, 32_000, 64_000, 128_000],
        repeats=7,
        warmup=1,
    )

    assert result.label == "O(n)"
    assert 0.7 <= result.exponent <= 1.4


def test_builtin_sort_exposes_c_backed_nlogn_work():
    result = measure_scaling(
        builtin_sort,
        shuffled_values,
        [16_000, 32_000, 64_000, 128_000, 256_000],
        repeats=9,
        warmup=1,
    )

    # The point: a one-line C builtin's hidden work IS visible to a wall-time probe
    # (a line/bytecode counter would call sorted() O(1)). So the measured growth must
    # be clearly super-constant. We do NOT pin it to exactly n log n: in this size
    # range cache/memory-hierarchy effects push the measured wall-time exponent up
    # toward ~1.4, which is honest — and the budget margin (0.35) absorbs it.
    assert result.exponent >= 0.9
    assert result.label in {"O(n)", "O(n log n)", "O(n^2)"}


def test_quadratic_reference_lands_in_quadratic_bucket():
    result = measure_scaling(
        count_pairs,
        lambda n: list(range(n)),
        [90, 135, 202, 303, 454],
        repeats=5,
        warmup=1,
    )

    assert result.label == "O(n^2)"
    assert 1.6 <= result.exponent <= 2.4


def test_budget_gate_passes_linear_and_fails_quadratic():
    linear = measure_scaling(
        loop_sum,
        lambda n: list(range(n)),
        [2_000, 4_000, 8_000, 16_000, 32_000],
        repeats=5,
        warmup=1,
    )
    quadratic = measure_scaling(
        count_pairs,
        lambda n: list(range(n)),
        [90, 135, 202, 303, 454],
        repeats=5,
        warmup=1,
    )

    assert check_budget(linear, budget_exponent=1.2)
    assert not check_budget(quadratic, budget_exponent=1.2)


def test_tuple_and_single_value_generators_work():
    tuple_result = measure_scaling(
        add_many,
        lambda n: (list(range(n)), list(range(n))),
        [20_000, 40_000, 80_000, 160_000],
        repeats=7,
        warmup=1,
    )
    single_result = measure_scaling(
        loop_sum,
        lambda n: list(range(n)),
        [20_000, 40_000, 80_000, 160_000],
        repeats=7,
        warmup=1,
    )

    # Primary purpose: both gen shapes (tuple -> splatted args, single value -> one
    # arg) actually drive the candidate. Secondary: both are linear-ish. We assert an
    # exponent band rather than the exact "O(n)" label, because finite-n constant
    # overhead can flatten the slope just below the 0.85 label boundary.
    assert tuple_result.samples
    assert single_result.samples
    assert 0.75 <= tuple_result.exponent <= 1.35
    assert 0.75 <= single_result.exponent <= 1.35


def test_per_call_timeout_stops_ladder_and_records_note():
    def slow(values):
        total = 0
        for value in values:
            total += value * value
        return total

    result = measure_scaling(
        slow,
        lambda n: list(range(n)),
        [20_000, 40_000, 80_000],
        repeats=3,
        warmup=0,
        per_call_timeout=0.0,
    )

    assert len(result.samples) < 3
    assert "timeout" in result.notes


def test_candidate_exception_returns_partial_result_with_note():
    def errors_at_large_n(values):
        if len(values) > 2_000:
            raise ValueError("too large")
        return loop_sum(values)

    result = measure_scaling(
        errors_at_large_n,
        lambda n: list(range(n)),
        [1_000, 2_000, 4_000, 8_000],
        repeats=3,
        warmup=0,
    )

    assert result.samples
    assert "raised" in result.notes


def test_low_confidence_budget_path_is_warning_only(monkeypatch):
    def fake_classify(samples):
        return 4.0, 0.1, "super-polynomial"

    monkeypatch.setattr(complexity_probe, "classify", fake_classify)

    result = measure_scaling(
        loop_sum,
        lambda n: list(range(n)),
        [10, 20],
        repeats=1,
        warmup=0,
        budget_exponent=1.0,
    )

    assert result.within_budget is True
    assert "low-confidence" in result.notes


def test_low_confidence_check_budget_does_not_fail_noise():
    result = ComplexityResult(
        samples=[(10, 1.0), (20, 1.1)],
        exponent=9.0,
        r2=0.1,
        label="super-polynomial",
        within_budget=None,
        budget_exponent=None,
        notes="low-confidence fit",
    )

    assert check_budget(result, budget_exponent=1.0)

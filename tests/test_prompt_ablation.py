import pytest

from scripts.prompt_ablation import (
    VERBOSE_PREAMBLE,
    build_condition_prompt,
    summarize_ablation,
)


def test_summarize_ablation_means_and_deltas() -> None:
    rows = [
        {"preamble": "lean", "effort": "low", "fraction": 0.4, "total_tokens": 100},
        {"preamble": "lean", "effort": "low", "fraction": 0.8, "total_tokens": 300},
        {"preamble": "verbose", "effort": "low", "fraction": 0.7, "total_tokens": 400},
        {"preamble": "verbose", "effort": "low", "fraction": 0.9, "total_tokens": 600},
        {"preamble": "lean", "effort": "high", "fraction": 1.0, "total_tokens": 900},
        {"preamble": "verbose", "effort": "high", "fraction": 0.5, "total_tokens": 1100},
    ]

    summary = summarize_ablation(rows)
    cells = summary["cells"]
    deltas = summary["deltas"]

    assert cells[("lean", "low")]["mean_fraction"] == pytest.approx(0.6)
    assert cells[("lean", "low")]["mean_tokens"] == pytest.approx(200.0)
    assert cells[("lean", "low")]["n"] == 2

    assert cells[("verbose", "low")]["mean_fraction"] == pytest.approx(0.8)
    assert cells[("verbose", "low")]["mean_tokens"] == pytest.approx(500.0)
    assert cells[("verbose", "low")]["n"] == 2

    assert deltas["low"]["delta_fraction"] == pytest.approx(0.2)
    assert deltas["low"]["delta_tokens"] == pytest.approx(300.0)

    assert deltas["high"]["delta_fraction"] == pytest.approx(-0.5)
    assert deltas["high"]["delta_tokens"] == pytest.approx(200.0)


def test_verbose_preamble_and_lean_prompt_shape() -> None:
    assert VERBOSE_PREAMBLE.strip()

    lean_prompt = build_condition_prompt("Write f()", "lean")
    verbose_prompt = build_condition_prompt("Write f()", "verbose")

    assert VERBOSE_PREAMBLE.strip() not in lean_prompt
    assert VERBOSE_PREAMBLE.strip() in verbose_prompt

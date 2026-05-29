"""Telemetry-guard tests for scripts/token_efficiency.py.

The efficiency scoreboard ranks by median output_tokens, and `_med` skips only
None (not 0). The claude CLI occasionally reports output_tokens=0 on a call that
plainly produced output — a phantom zero that would silently understate the
frontier. `_reported_output_tokens` converts that case to None so the median
ignores it. These tests pin that behavior without spawning any worker.
"""
from scripts.token_efficiency import _med, _reported_output_tokens


def test_zero_tokens_with_real_text_is_unreported():
    # The bug we observed: opus-4-8 decode_ways / haiku calc3 — passed, non-empty
    # code, real cost, but output_tokens came back 0. Must become None (unknown).
    assert _reported_output_tokens(0, "def f(): return 1") is None


def test_missing_tokens_with_real_text_is_unreported():
    assert _reported_output_tokens(None, "def f(): return 1") is None


def test_real_count_passes_through():
    assert _reported_output_tokens(200, "def f(): return 1") == 200


def test_genuinely_empty_response_keeps_its_zero():
    # An empty/whitespace response is a real failure, not a telemetry gap — keep
    # the reported value as-is (the `empty` row flag is what marks these).
    assert _reported_output_tokens(0, "") == 0
    assert _reported_output_tokens(0, "   \n ") == 0
    assert _reported_output_tokens(None, "") is None


def test_phantom_zero_no_longer_drags_the_median():
    # Before the guard, a 0 was counted: median([812,200,243,0,181]) == 200 but the
    # min/mean were corrupted and a row could read "0 tokens". After the guard the
    # phantom 0 is None and skipped entirely.
    raw = [812, 200, 243, 0, 181]
    texts = ["x"] * 5  # all non-empty -> the 0 is a phantom
    guarded = [_reported_output_tokens(v, t) for v, t in zip(raw, texts)]
    assert guarded == [812, 200, 243, None, 181]
    assert _med(guarded) == 221.5  # median of [181,200,243,812]; phantom 0 excluded
    assert min(x for x in guarded if x is not None) == 181

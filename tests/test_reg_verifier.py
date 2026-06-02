"""Regression tests for execution/verifier.py — QualityVerifier subprocess/parse/caps/process-group.

Hermetic: no real worker CLI, no subprocess spawn, no network. Drives the pure
helpers (`_clamp_xdist`, `_head_tail`) directly with monkeypatched cpu_count.

Covers two confirmed defects:
  * verifier-1: ``_clamp_xdist`` corrupted commands when an embedded ``-n<digits>``
    substring (a test-file path or ``-k`` filter) was mistaken for an xdist flag
    and silently rewritten — running the WRONG verification.
  * verifier-2: ``_head_tail(text, 1)`` returned the ENTIRE input plus the elision
    marker (``text[-0:]`` is the whole string), defeating AGY_VERIFY_FULL_LOG_MAX.
"""
import os

import pytest

from agy_orchestrator.execution.verifier import QualityVerifier


# ---------------------------------------------------------------------------
# verifier-1: _clamp_xdist must only touch a genuine xdist worker-count flag,
#             never an embedded `-n<digits>` substring in a path / -k filter.
# ---------------------------------------------------------------------------
@pytest.fixture
def verifier(monkeypatch):
    # Resource bound ON (clamp active) and a small fixed core count so any
    # number above it would (buggily) trigger a rewrite.
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    return QualityVerifier(test_commands=["x"])


def test_clamp_xdist_leaves_embedded_path_substring_untouched(verifier):
    # `-n200` here is part of a test-FILE PATH, not an xdist flag. The old
    # boundary-less regex rewrote it to `-n8`, corrupting the path so the
    # verifier ran a different (nonexistent) file. Must be left untouched.
    cmd = "pytest tests/test_issue99-n200.py"
    assert verifier._clamp_xdist(cmd) == cmd


def test_clamp_xdist_leaves_embedded_k_filter_substring_untouched(verifier):
    # `-n99` inside a `-k` selection expression is a test-name substring, not
    # an xdist worker count. The old regex corrupted the selection to `-n8`.
    cmd = "pytest -k test-n99-x"
    assert verifier._clamp_xdist(cmd) == cmd


def test_clamp_xdist_leaves_hyphenated_token_untouched(verifier):
    # A bare hyphenated token like `foo-n500` (e.g. a marker / arbitrary arg)
    # is not a flag start, so it must not be clamped.
    cmd = "pytest -m smoke-n500"
    assert verifier._clamp_xdist(cmd) == cmd


def test_clamp_xdist_still_clamps_genuine_flags(verifier):
    # The fix must NOT break the real clamp: all genuine xdist flag forms whose
    # count exceeds the (8) core budget are still clamped down.
    assert verifier._clamp_xdist("pytest -n 200") == "pytest -n 8"
    assert verifier._clamp_xdist("pytest -n200") == "pytest -n8"
    assert verifier._clamp_xdist("pytest -n200 -q") == "pytest -n8 -q"
    assert verifier._clamp_xdist("pytest --numprocesses 200") == "pytest --numprocesses 8"
    assert verifier._clamp_xdist("pytest --numprocesses=200") == "pytest --numprocesses=8"
    # Flag at the very start of the string (pre == "") is still anchored.
    assert verifier._clamp_xdist("-n 200 pytest") == "-n 8 pytest"


def test_clamp_xdist_leaves_within_budget_and_auto(verifier):
    # Sanity: under-budget and `-n auto` are left alone (intended semantics).
    assert verifier._clamp_xdist("pytest -n 4") == "pytest -n 4"
    assert verifier._clamp_xdist("pytest -n auto") == "pytest -n auto"


# ---------------------------------------------------------------------------
# verifier-2: _head_tail(text, 1) must honour the cap, not return the whole
#             string via the `text[-0:]` slice quirk.
# ---------------------------------------------------------------------------
def test_head_tail_max_chars_one_honours_cap():
    text = "x" * 100000
    out = QualityVerifier._head_tail(text, 1)
    # Before the fix: len(out) == 100048 (> len(text)). After: a plain head cut.
    assert len(out) <= 1
    assert out == "x"


def test_head_tail_does_not_exceed_input_for_tiny_caps():
    # For EVERY tiny positive cap, the output must never be LONGER than the input
    # (the contract the bound exists to enforce). The verifier-2 fix only special-
    # cased max_chars==1, but the fixed-size elision marker (~44 chars) defeats the
    # bound for a whole band of small caps too — the framed head+tail came out
    # longer than the input there. The root fix degrades to a head cut whenever the
    # framing would not actually shrink the text.
    text = "y" * 30
    for cap in range(1, 48):
        out = QualityVerifier._head_tail(text, cap)
        assert len(out) <= len(text), f"cap {cap} produced longer output ({len(out)})"


def test_head_tail_unbounded_and_passthrough_preserved():
    # Intended behaviour preserved: max_chars <= 0 means unbounded, and text
    # already under the cap is returned verbatim.
    text = "z" * 1000
    assert QualityVerifier._head_tail(text, 0) == text
    assert QualityVerifier._head_tail(text, 5000) == text

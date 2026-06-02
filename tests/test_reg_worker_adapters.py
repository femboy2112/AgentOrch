"""Hermetic regression tests for confirmed worker-adapter defects.

Subsystem: worker adapters (codex/claude/grok/agy/mock): command build, stdin,
postprocess, JSON-event/usage/session parsing.

These NEVER spawn a real worker CLI and NEVER touch the network. They drive the
pure parse/postprocess methods directly with the exact inputs that reproduced
each bug, and assert the fixed (bounded / non-crashing / lossless) behaviour.

Covered:
  - adapters-grok-1: GrokAgent._extract_payload regex {.*} backtracked O(n^2)
    -> multi-second hang on large brace-heavy non-balanced stdout. Now bounded,
    linear find()/rfind() slice.
  - adapters-codex-1: CodexAgent._extract_usage crashed (ValueError/OverflowError)
    on NaN/Inf token counts, silently dropping a valid usage row.
"""
import json
import time

from agy_orchestrator.core.agents.grok_agent import GrokAgent
from agy_orchestrator.core.agents.codex_agent import CodexAgent


# --------------------------------------------------------------------------- #
# adapters-grok-1
# --------------------------------------------------------------------------- #

def test_grok_postprocess_large_unbalanced_braces_is_fast():
    """A ~560KB truncated, brace-heavy, non-balanced grok stdout must NOT hang.

    Pre-fix this took ~16s at this size (and scaled O(n^2)); the linear slice
    completes in well under a second. Generous ceiling so the test is robust on
    a loaded CI box while still catching the quadratic regression (which would
    blow past it by an order of magnitude).
    """
    s = "INFO {a {b {c " * 40000  # ~560KB, many '{', no late balanced '}'
    t0 = time.time()
    out = GrokAgent("p")._postprocess(s)
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"_postprocess took {elapsed:.2f}s (quadratic regression)"
    # No JSON object recoverable -> raw text returned unchanged (intended behaviour).
    assert out == s


def test_grok_extract_payload_recovers_embedded_object():
    """Linear fallback still recovers a real {...} object wrapped in log noise."""
    s = 'INFO booting\n{"text": "hello", "sessionId": "abc"}\nINFO done'
    obj = GrokAgent._extract_payload(s)
    assert isinstance(obj, dict)
    assert obj["text"] == "hello"
    assert GrokAgent._extract_text(s) == "hello"


def test_grok_extract_payload_recovers_object_with_trailing_noise():
    """First '{' .. last '}' slice still matches the object even with leading and
    trailing junk lines around it."""
    s = 'log line\n{"text": "ok"}\ntrailing {garbage with no close'
    obj = GrokAgent._extract_payload(s)
    assert isinstance(obj, dict) and obj["text"] == "ok"


def test_grok_extract_payload_plain_text_returns_none():
    s = "just some plain text, no braces at all"
    assert GrokAgent._extract_payload(s) is None
    assert GrokAgent._extract_text(s) == s


# --------------------------------------------------------------------------- #
# adapters-codex-1
# --------------------------------------------------------------------------- #

def test_codex_extract_usage_nan_input_does_not_crash():
    """json.loads accepts 'NaN'; int(nan) -> ValueError. Must not crash, and the
    rest of the (valid) usage row must survive."""
    raw = json.dumps({
        "type": "turn.completed",
        "usage": {
            "input_tokens": float("nan"),
            "cache_read_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 100,
        },
    })
    row = CodexAgent("p")._extract_usage(raw, "")
    assert row["token_source"] == "cli"          # row not discarded
    assert row["output_tokens"] == 10            # valid fields preserved
    assert row["total_tokens"] == 100
    assert row["cache_read_tokens"] == 5
    # fresh-input subtraction skipped on a non-finite input; raw value passes
    # through untouched rather than crashing.
    assert row["input_tokens"] != row["input_tokens"]  # NaN passthrough


def test_codex_extract_usage_inf_input_does_not_crash():
    """'1e400' -> inf; int(inf) -> OverflowError. Must not crash."""
    raw2 = '{"type":"turn.completed","usage":{"input_tokens":1e400,"cache_read_tokens":5,"output_tokens":7}}'
    row = CodexAgent("p")._extract_usage(raw2, "")
    assert row["token_source"] == "cli"
    assert row["cache_read_tokens"] == 5
    assert row["output_tokens"] == 7
    assert row["input_tokens"] == float("inf")   # passthrough, not crashed


def test_codex_extract_usage_finite_values_still_normalized():
    """Regression guard: finite values still get the fresh = total - cached
    normalization (intended behaviour preserved)."""
    raw = json.dumps({
        "type": "turn.completed",
        "usage": {
            "input_tokens": 100,
            "cache_read_tokens": 30,
            "output_tokens": 12,
            "total_tokens": 112,
        },
    })
    row = CodexAgent("p")._extract_usage(raw, "")
    assert row["input_tokens"] == 70   # 100 - 30
    assert row["cache_read_tokens"] == 30
    assert row["output_tokens"] == 12

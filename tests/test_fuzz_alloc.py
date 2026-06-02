"""Hermetic fuzz / property / edge-case tests for the allocation-adjacent core:

  * core/profile.py          (effort downgrade from plan tier)
  * core/calibration.py      (per-config watchdog budgets + JSONL persistence)
  * core/model_discovery.py  (never-raise discovery + static fallback)
  * execution/ledger.py      (confidence scoring on missing/odd signals)
  * execution/execution_selector.py (pytest-count parsing, ranking)

Every test here asserts CORRECT, robust behaviour and MUST pass. Genuine defects
found while writing these are reported separately (not committed as failing
tests). No network, no real worker CLI, no credentials — subprocess use is
limited to the in-process pytest sandbox of ExecutionSelector (hermetic).
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
from pathlib import Path

import pytest

from agy_orchestrator.core.profile import UserProfile
from agy_orchestrator.core.calibration import (
    CalibrationTable,
    append_live_row,
    DEFAULT_MAX_BYTES,
    DEFAULT_STALL_SECONDS,
    DEFAULT_STALL_SECONDS_BY_WORKER,
)
from agy_orchestrator.core.model_discovery import discover_models, reset_cache
from agy_orchestrator.execution.ledger import build_ledger
from agy_orchestrator.execution.execution_selector import ExecutionSelector


# --------------------------------------------------------------------------- #
# UserProfile.get_baseline_effort
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "plan,expected",
    [
        ("free", "low"),
        ("", "low"),
        ("   ", "low"),
        ("Plus $20", "medium"),
        ("pro", "high"),
        ("Max 20x", "max"),
        ("$100 plan", "max"),
        ("CLAUDE MAX", "max"),       # case-insensitive
        ("garbage-tier", "low"),     # unknown -> safe floor
    ],
)
def test_baseline_effort_tiers(plan, expected):
    up = UserProfile(claude_plan=plan)
    assert up.get_baseline_effort("claude") == expected


def test_baseline_effort_unknown_agent_defaults_low():
    # Unknown agent_name -> the getattr default ("free") -> "low". Never KeyError.
    up = UserProfile()
    assert up.get_baseline_effort("grok") == "low"
    assert up.get_baseline_effort("does_not_exist") == "low"


def test_baseline_effort_precedence_max_over_lower():
    # A plan string that matches several markers resolves to the highest tier.
    up = UserProfile(codex_plan="pro plus max $20 $50 $100")
    assert up.get_baseline_effort("codex") == "max"


def test_baseline_effort_always_known_value():
    # Property: result is always one of the four known tiers, for many inputs.
    valid = {"low", "medium", "high", "max"}
    for plan in ["", "x", "20x", "plus", "PRO", "weird unicode ☃", "a" * 5000]:
        up = UserProfile(agy_plan=plan)
        assert up.get_baseline_effort("agy") in valid


# --------------------------------------------------------------------------- #
# CalibrationTable.budget_for / ingest
# --------------------------------------------------------------------------- #

def test_empty_table_returns_conservative_defaults():
    t = CalibrationTable()
    mb, ss = t.budget_for("codex", "gpt", "high")
    assert mb == DEFAULT_MAX_BYTES
    assert ss == DEFAULT_STALL_SECONDS_BY_WORKER["codex"]
    assert not t.has_data_for("codex", "gpt", "high")


def test_unknown_worker_uses_global_stall_default():
    t = CalibrationTable()
    _, ss = t.budget_for("mystery_worker", "m", "e")
    assert ss == DEFAULT_STALL_SECONDS


def test_fewer_than_three_rows_stays_default():
    t = CalibrationTable()
    t.ingest([{"ok": True, "worker": "codex", "model": "m", "effort": "e",
               "out_tokens": 100, "wall_ms": 500}] * 2)
    mb, ss = t.budget_for("codex", "m", "e")
    assert mb == DEFAULT_MAX_BYTES
    assert ss == DEFAULT_STALL_SECONDS_BY_WORKER["codex"]


def test_calibrated_stall_never_tighter_than_default():
    # Issue #29 invariant: a calibrated config must never get a TIGHTER stall
    # window than an uncalibrated one. Small wall times must floor at default.
    t = CalibrationTable()
    t.ingest([{"ok": True, "worker": "claude", "model": "m", "effort": "e",
               "out_tokens": 10, "wall_ms": 5}] * 5)  # tiny wall
    _, ss = t.budget_for("claude", "m", "e")
    assert ss >= DEFAULT_STALL_SECONDS_BY_WORKER["claude"]


def test_max_bytes_floored_at_8kb():
    # Even tiny successful runs get at least 8KB of headroom.
    t = CalibrationTable()
    t.ingest([{"ok": True, "worker": "codex", "model": "m", "effort": "e",
               "out_tokens": 1, "wall_ms": 100}] * 5)
    mb, _ = t.budget_for("codex", "m", "e")
    assert mb >= 8_000


def test_ingest_skips_non_ok_rows():
    t = CalibrationTable()
    used = t.ingest([
        {"ok": False, "worker": "c", "model": "m", "effort": "e", "out_tokens": 1, "wall_ms": 1},
        {"worker": "c", "model": "m", "effort": "e", "out_tokens": 1, "wall_ms": 1},  # ok missing
    ])
    assert used == 0


def test_ingest_skips_rows_with_no_telemetry():
    t = CalibrationTable()
    used = t.ingest([{"ok": True, "worker": "c", "model": "m", "effort": "e"}])
    assert used == 0


def test_ingest_none_effort_key_distinct_from_string():
    t = CalibrationTable()
    t.ingest([{"ok": True, "worker": "c", "model": "m", "effort": None,
               "out_tokens": 10, "wall_ms": 10}] * 3)
    assert t.has_data_for("c", "m", None)
    assert not t.has_data_for("c", "m", "n/a")


def test_negative_and_zero_telemetry_filtered_not_crashing():
    # Negative/zero token+wall values are corrupt; they must be filtered out
    # (r[0] > 0 / r[1] > 0) and fall back to conservative defaults, not crash.
    t = CalibrationTable()
    t.ingest([{"ok": True, "worker": "c", "model": "m", "effort": "e",
               "out_tokens": -5, "wall_ms": -10}] * 3)
    mb, ss = t.budget_for("c", "m", "e")
    assert mb == DEFAULT_MAX_BYTES
    assert ss == DEFAULT_STALL_SECONDS


def test_budget_for_returns_finite_positive_on_normal_data():
    # Property: well-formed telemetry yields finite, positive budgets.
    t = CalibrationTable()
    t.ingest([{"ok": True, "worker": "codex", "model": "m", "effort": "high",
               "out_tokens": 500 + i, "wall_ms": 1000 + i} for i in range(30)])
    mb, ss = t.budget_for("codex", "m", "high")
    assert mb > 0 and math.isfinite(mb)
    assert ss > 0 and math.isfinite(ss)


# --------------------------------------------------------------------------- #
# CalibrationTable.load — corrupt / partial / missing persistence
# --------------------------------------------------------------------------- #

@pytest.fixture
def no_default_live(monkeypatch):
    # Disable the default live-ledger source so load() reads only our temp file.
    monkeypatch.setenv("AGY_LIVE_LEDGER", "off")


def test_load_missing_file_returns_empty(no_default_live, tmp_path):
    t = CalibrationTable.load(path=tmp_path / "does_not_exist.jsonl")
    assert not t.has_data_for("codex", "m", "high")


def test_load_zero_byte_file(no_default_live, tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    t = CalibrationTable.load(path=p)
    assert isinstance(t, CalibrationTable)


def test_load_garbage_json_lines_does_not_crash(no_default_live, tmp_path):
    # A truncated / non-JSON file must be swallowed (warn) and yield an empty table.
    p = tmp_path / "garbage.jsonl"
    p.write_text("{not json at all\n\x00\x01garbage\n][\n")
    t = CalibrationTable.load(path=p)
    assert isinstance(t, CalibrationTable)
    assert not t.has_data_for("codex", "m", "high")


def test_load_blank_lines_ignored(no_default_live, tmp_path):
    p = tmp_path / "blanks.jsonl"
    p.write_text("\n   \n\n")
    t = CalibrationTable.load(path=p)
    assert isinstance(t, CalibrationTable)


def test_load_path_that_is_a_directory(no_default_live, tmp_path):
    d = tmp_path / "is_a_dir.jsonl"
    d.mkdir()
    # Opening a directory raises; load() must swallow it and return a table.
    t = CalibrationTable.load(path=d)
    assert isinstance(t, CalibrationTable)


def test_load_wellformed_rows(no_default_live, tmp_path):
    p = tmp_path / "good.jsonl"
    rows = [{"ok": True, "worker": "codex", "model": "m", "effort": "high",
             "out_tokens": 100 + i, "wall_ms": 1000 + i} for i in range(5)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    t = CalibrationTable.load(path=p)
    assert t.has_data_for("codex", "m", "high")


# --------------------------------------------------------------------------- #
# append_live_row — robust persistence
# --------------------------------------------------------------------------- #

def test_append_live_row_roundtrip(tmp_path, monkeypatch):
    monkeypatch.delenv("AGY_LIVE_LEDGER", raising=False)
    p = tmp_path / "sub" / "live.jsonl"  # parent dir does not exist yet
    append_live_row("codex", "m", "high", ok=True, out_bytes=4000, wall_ms=1234.0, path=p)
    line = p.read_text().strip()
    row = json.loads(line)
    assert row["worker"] == "codex"
    assert row["ok"] is True
    assert row["out_tokens"] == 1000  # 4000 bytes / 4 bytes-per-token
    assert row["wall_ms"] == 1234.0


def test_append_live_row_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_LIVE_LEDGER", "off")
    p = tmp_path / "live.jsonl"
    append_live_row("codex", "m", "high", ok=True, out_bytes=4000, path=p)
    assert not p.exists()


def test_append_live_row_noop_when_no_metric(tmp_path, monkeypatch):
    monkeypatch.delenv("AGY_LIVE_LEDGER", raising=False)
    p = tmp_path / "live.jsonl"
    append_live_row("codex", "m", "high", ok=True, path=p)
    assert not p.exists()


def test_append_live_row_swallows_disk_error(tmp_path, monkeypatch):
    # Target parent is a file, so mkdir/open fails — must be swallowed, never raise.
    monkeypatch.delenv("AGY_LIVE_LEDGER", raising=False)
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    p = blocker / "live.jsonl"  # blocker is a file, can't be a parent dir
    append_live_row("codex", "m", "high", ok=True, out_bytes=4000, path=p)  # no raise


def test_append_then_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("AGY_LIVE_LEDGER", "off")  # don't pull the default ledger
    p = tmp_path / "live.jsonl"
    for i in range(4):
        # AGY_LIVE_LEDGER=off would no-op append; flip it just for the write.
        monkeypatch.setenv("AGY_LIVE_LEDGER", "on")
        append_live_row("codex", "m", "high", ok=True,
                        out_bytes=4000 + i, wall_ms=1000.0 + i, path=p)
    monkeypatch.setenv("AGY_LIVE_LEDGER", "off")
    t = CalibrationTable.load(path=p)
    assert t.has_data_for("codex", "m", "high")


# --------------------------------------------------------------------------- #
# model_discovery — never raises, static fallback, memoization, concurrency
# --------------------------------------------------------------------------- #

def test_discover_none_argv_returns_fallback():
    reset_cache()
    out = asyncio.run(discover_models("k_none", list_argv=None,
                                      parse=lambda s: ["ignored"], fallback=["a", "b"]))
    assert out == ["a", "b"]


def test_discover_parser_exception_falls_back():
    reset_cache()
    out = asyncio.run(discover_models(
        "k_parse_raise", list_argv=["true"],  # real /bin/true exits 0, empty stdout
        parse=lambda s: (_ for _ in ()).throw(RuntimeError("boom")),
        fallback=["fb"]))
    assert out == ["fb"]


def test_discover_missing_binary_falls_back():
    reset_cache()
    out = asyncio.run(discover_models(
        "k_missing", list_argv=["this_binary_does_not_exist_zzz_12345"],
        parse=lambda s: ["x"], fallback=["fb"]))
    assert out == ["fb"]


def test_discover_nonzero_exit_falls_back():
    reset_cache()
    out = asyncio.run(discover_models(
        "k_false", list_argv=["false"],  # exits 1
        parse=lambda s: ["x"], fallback=["fb"]))
    assert out == ["fb"]


def test_discover_returned_list_is_isolated_copy():
    reset_cache()

    async def go():
        a = await discover_models("k_copy", list_argv=None,
                                  parse=lambda s: [], fallback=["a", "b"])
        a.append("MUTATED")
        b = await discover_models("k_copy", list_argv=None,
                                  parse=lambda s: [], fallback=["zzz"])
        return b

    out = asyncio.run(go())
    assert out == ["a", "b"]  # cache not corrupted by caller mutation


def test_discover_concurrent_same_key_no_crash():
    reset_cache()

    async def go():
        results = await asyncio.gather(*[
            discover_models("k_concurrent", list_argv=None,
                            parse=lambda s: [], fallback=["a"])
            for _ in range(25)
        ])
        return results

    results = asyncio.run(go())
    assert all(r == ["a"] for r in results)


def test_discover_empty_fallback_is_fine():
    reset_cache()
    out = asyncio.run(discover_models("k_empty_fb", list_argv=None,
                                      parse=lambda s: [], fallback=[]))
    assert out == []


# --------------------------------------------------------------------------- #
# build_ledger — confidence scoring on missing / odd signals
# --------------------------------------------------------------------------- #

class _WF:
    pass


def test_ledger_verified_beats_everything():
    wf = _WF()
    wf.verified = True
    wf.approved = True
    row = build_ledger(wf, mode="adversarial", had_verifier=True, produced_output=True)
    assert row["confidence"] == "verified"


def test_ledger_approved_when_no_verifier():
    wf = _WF()
    wf.approved = True
    row = build_ledger(wf, mode="adversarial", had_verifier=False, produced_output=True)
    assert row["confidence"] == "approved"


def test_ledger_unverified_with_verifier_note():
    wf = _WF()
    row = build_ledger(wf, mode="feedback", had_verifier=True, produced_output=True)
    assert row["confidence"] == "unverified"
    assert "configured" in row["note"]


def test_ledger_failed_when_no_output():
    wf = _WF()
    row = build_ledger(wf, mode="direct", had_verifier=False, produced_output=False)
    assert row["confidence"] == "failed"


def test_ledger_missing_all_attrs_defaults_gracefully():
    # A workflow exposing NO signals at all (master/direct) must not crash.
    row = build_ledger(object(), mode="master", had_verifier=False, produced_output=True)
    assert row["confidence"] == "unverified"
    assert row["verified"] is False
    assert row["iterations_used"] is None


def test_ledger_drops_none_telemetry_keeps_present():
    wf = _WF()
    row = build_ledger(wf, mode="direct", had_verifier=False, produced_output=True,
                       telemetry={"wall_ms": 100, "out_bytes": None, "worker": "codex"})
    assert row["wall_ms"] == 100
    assert row["worker"] == "codex"
    assert "out_bytes" not in row  # None dropped


def test_ledger_empty_telemetry_dict():
    wf = _WF()
    row = build_ledger(wf, mode="direct", had_verifier=False, produced_output=True,
                       telemetry={})
    assert row["confidence"] == "unverified"


# --------------------------------------------------------------------------- #
# ExecutionSelector — code extraction, pytest-count parsing, ranking
# --------------------------------------------------------------------------- #

def test_extract_code_fenced():
    txt = "blah\n```python\nx = 1\n```\ntrailing"
    assert ExecutionSelector.extract_code(txt) == "x = 1"


def test_extract_code_strips_think_block():
    txt = "<think>reasoning</think>\nactual = 2"
    assert ExecutionSelector.extract_code(txt) == "actual = 2"


def test_extract_code_empty_string():
    assert ExecutionSelector.extract_code("") == ""


def test_extract_code_with_control_chars():
    txt = "\x00\x01 noise ```python\nv = 3\n```"
    assert ExecutionSelector.extract_code(txt) == "v = 3"


def test_select_empty_candidates_raises_valueerror():
    sel = ExecutionSelector(test_source="def test_x():\n    assert True\n")
    with pytest.raises(ValueError):
        asyncio.run(sel.select([]))


def test_select_ranks_passing_above_failing():
    test_src = (
        "import solution\n"
        "def test_answer():\n"
        "    assert solution.answer() == 42\n"
    )
    sel = ExecutionSelector(test_source=test_src, timeout=30.0)
    good = "```python\ndef answer():\n    return 42\n```"
    bad = "```python\ndef answer():\n    return 0\n```"
    best, scores = asyncio.run(sel.select([bad, good]))
    assert best == 1
    assert scores[1][0] == 1  # the good candidate passed 1 test


def test_select_single_candidate():
    test_src = "def test_t():\n    assert 1 == 1\n"
    sel = ExecutionSelector(test_source=test_src, timeout=30.0)
    best, scores = asyncio.run(sel.select(["```python\nx = 1\n```"]))
    assert best == 0
    assert len(scores) == 1


def test_select_all_garbage_candidates_picks_index_zero():
    # Candidates that are syntactically broken score (0,0); max() still returns a
    # valid index (the first), never crashes.
    test_src = "import solution\ndef test_t():\n    assert solution.f()\n"
    sel = ExecutionSelector(test_source=test_src, timeout=30.0)
    best, scores = asyncio.run(sel.select([
        "```python\ndef (((:\n```",
        "not even code at all",
    ]))
    assert best in (0, 1)
    assert all(s == (0, 0) for s in scores)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

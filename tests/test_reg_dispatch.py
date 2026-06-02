"""Regression tests for harness/dispatch.py plan-load / path-policy defects.

Subsystem: harness/dispatch.py — orchestration/meta/events/plan-load/
path-policy/pipeline wiring.

dispatch-plan-1: load_plan() and plan_file_sha256() both document that any
malformed plan path raises an operator-facing ValueError. They formerly gated
only on Path.exists() before read_text()/read_bytes(), so a *directory* (which
satisfies exists()==True) or a permission-denied regular file raised a raw
IsADirectoryError / PermissionError instead — escaping the CLI's `except
ValueError` fail-fast path and surfacing as a traceback to the operator. The
fix gates on p.is_file() and wraps the read in try/except OSError -> ValueError.

Hermetic: no network, no worker CLI, pure-function calls only.
"""
import os
import stat

import pytest

from harness.dispatch import load_plan, plan_file_sha256


def test_directory_plan_path_raises_valueerror(tmp_path):
    """A directory passed to --plan must raise the documented ValueError,
    not a raw IsADirectoryError (dispatch-plan-1)."""
    d = tmp_path / "plan.json"
    d.mkdir()  # operator slip: passed a runs/<id>/ directory
    with pytest.raises(ValueError):
        load_plan(d)
    with pytest.raises(ValueError):
        plan_file_sha256(d)


def test_missing_plan_path_raises_valueerror(tmp_path):
    """The pre-existing missing-file contract must remain ValueError."""
    missing = tmp_path / "nope.json"
    with pytest.raises(ValueError):
        load_plan(missing)
    with pytest.raises(ValueError):
        plan_file_sha256(missing)


def test_valid_plan_still_loads_and_hashes(tmp_path):
    """The fix must not regress the happy path."""
    p = tmp_path / "plan.json"
    p.write_text('["step one", "step two"]', encoding="utf-8")
    plan = load_plan(p)
    assert plan.as_steps() == ["step one", "step two"]
    digest = plan_file_sha256(p)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_bad_json_plan_still_raises_valueerror(tmp_path):
    """Invalid JSON must still surface as the documented ValueError."""
    p = tmp_path / "plan.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_plan(p)


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permissions")
def test_unreadable_plan_file_raises_valueerror(tmp_path):
    """A permission-denied regular file must raise ValueError, not a raw
    PermissionError (same root cause as the directory case)."""
    p = tmp_path / "plan.json"
    p.write_text('["x"]', encoding="utf-8")
    p.chmod(0)
    try:
        with pytest.raises(ValueError):
            load_plan(p)
        with pytest.raises(ValueError):
            plan_file_sha256(p)
    finally:
        p.chmod(stat.S_IRUSR | stat.S_IWUSR)

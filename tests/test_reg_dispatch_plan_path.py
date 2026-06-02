"""Hermetic regression: load_plan / plan_file_sha256 must raise ValueError
(the documented operator-facing failure) for non-regular-file plan paths,
not a raw IsADirectoryError / PermissionError (dispatch-plan-1)."""
import os
import stat
import tempfile
from pathlib import Path

import pytest

from harness.dispatch import load_plan, plan_file_sha256


def test_directory_path_raises_valueerror(tmp_path):
    d = tmp_path / "plan.json"
    d.mkdir()  # operator slip: passed a runs/<id>/ dir to --plan
    with pytest.raises(ValueError):
        load_plan(d)
    with pytest.raises(ValueError):
        plan_file_sha256(d)


def test_missing_path_still_raises_valueerror(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(ValueError):
        load_plan(missing)
    with pytest.raises(ValueError):
        plan_file_sha256(missing)


def test_valid_plan_still_loads(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text('["step one", "step two"]', encoding="utf-8")
    plan = load_plan(p)
    assert plan.as_steps() == ["step one", "step two"]
    # sha is the hex digest of the raw bytes
    assert len(plan_file_sha256(p)) == 64


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permissions")
def test_unreadable_file_raises_valueerror(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text('["x"]', encoding="utf-8")
    p.chmod(0)  # permission-denied unreadable file
    try:
        with pytest.raises(ValueError):
            load_plan(p)
        with pytest.raises(ValueError):
            plan_file_sha256(p)
    finally:
        p.chmod(stat.S_IRUSR | stat.S_IWUSR)

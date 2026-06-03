"""Issue #71 — --mission-critical hard-gates reconcile by default.

A run dispatched with --mission-critical but no explicit
--reconcile-disposition must default to "fail" (a confirmed dead-wiring
finding blocks the run), not the historical "warn". An EXPLICIT disposition is
always honored: explicit "warn" under --mission-critical is still respected, but
logs a loud non-gating warning.

Hermetic: exercises the pure helper ``_resolve_reconcile_disposition`` directly;
no dispatch, worker CLI, or network.
"""
from __future__ import annotations

import logging

from harness.dispatch import _resolve_reconcile_disposition


def test_unspecified_mission_critical_defaults_to_fail():
    # Case 1
    assert _resolve_reconcile_disposition(None, True) == "fail"


def test_unspecified_non_mission_critical_defaults_to_warn():
    # Case 2 (UNCHANGED)
    assert _resolve_reconcile_disposition(None, False) == "warn"


def test_explicit_warn_mission_critical_is_honored_with_warning(caplog):
    # Case 3 — honor "warn" but log a loud warning.
    with caplog.at_level(logging.WARNING, logger="harness"):
        result = _resolve_reconcile_disposition("warn", True)
    assert result == "warn"
    assert any(
        record.levelno == logging.WARNING
        and "will NOT gate the run" in record.getMessage()
        for record in caplog.records
    ), "expected a loud non-gating warning for explicit warn under mission-critical"


def test_explicit_warn_non_mission_critical_no_warning(caplog):
    # Explicit warn without mission-critical: honored, no warning noise.
    with caplog.at_level(logging.WARNING, logger="harness"):
        result = _resolve_reconcile_disposition("warn", False)
    assert result == "warn"
    assert not any(
        "will NOT gate the run" in record.getMessage() for record in caplog.records
    )


def test_explicit_fail_used_unchanged():
    # Case 4
    assert _resolve_reconcile_disposition("fail", True) == "fail"
    assert _resolve_reconcile_disposition("fail", False) == "fail"


def test_explicit_open_task_used_unchanged():
    # Case 4
    assert _resolve_reconcile_disposition("open-task", True) == "open-task"
    assert _resolve_reconcile_disposition("open-task", False) == "open-task"

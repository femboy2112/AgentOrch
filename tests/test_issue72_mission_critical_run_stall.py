"""Issue #72 (Layer 2) — --mission-critical default-arms the run-level stall net.

Layer 1 (worker transport bound) lands the core fix; this layer ensures a
coarse run-level safety net is ARMED under --mission-critical even if Layer 1
is somehow defeated. The run-level stall watchdog (run_stall_abort) was OPT-IN
(--run-stall-abort) and was NOT armed in the incident, so the run-level net
never engaged.

When --mission-critical is set and the operator did NOT explicitly pass
--run-stall-abort, a generous run-level stall is DEFAULT-ARMED. An explicit
--run-stall-abort value (including explicit 0/off) always WINS. Without
--mission-critical the default is NOT applied (unchanged behaviour).

Hermetic: exercises the pure helper ``_resolve_run_stall_abort`` directly; no
dispatch, worker CLI, or network.
"""
from __future__ import annotations

import logging

from harness.dispatch import (
    MISSION_CRITICAL_RUN_STALL_DEFAULT,
    _resolve_run_stall_abort,
)


def test_unspecified_mission_critical_default_arms_positive():
    # Operator passed --mission-critical but NOT --run-stall-abort (None).
    # The run-level net must be armed: non-None, positive.
    result = _resolve_run_stall_abort(None, True)
    assert result is not None
    assert result > 0
    assert result == MISSION_CRITICAL_RUN_STALL_DEFAULT


def test_unspecified_non_mission_critical_stays_none():
    # No --mission-critical, no explicit value -> unchanged behaviour (None).
    assert _resolve_run_stall_abort(None, False) is None


def test_explicit_value_preserved_under_mission_critical():
    # Explicit --run-stall-abort wins over the mission-critical default.
    assert _resolve_run_stall_abort(600.0, True) == 600.0


def test_explicit_value_preserved_without_mission_critical():
    assert _resolve_run_stall_abort(600.0, False) == 600.0


def test_explicit_zero_off_preserved_under_mission_critical(caplog):
    # Explicit 0/off DISABLES the run-level net even under --mission-critical;
    # the operator's explicit choice always wins (no silent re-arm).
    with caplog.at_level(logging.WARNING, logger="harness"):
        result = _resolve_run_stall_abort(0.0, True)
    assert result == 0.0


def test_default_logs_when_applied(caplog):
    # When the mission-critical default is applied, log clearly.
    with caplog.at_level(logging.INFO, logger="harness"):
        _resolve_run_stall_abort(None, True)
    assert any(
        "mission-critical" in record.getMessage().lower()
        and "run-stall" in record.getMessage().lower()
        for record in caplog.records
    ), "expected a clear log line when the mission-critical run-stall default is armed"


def test_env_override_of_default(monkeypatch):
    # AGY_MISSION_CRITICAL_RUN_STALL overrides the baked default; 0 disables.
    monkeypatch.setenv("AGY_MISSION_CRITICAL_RUN_STALL", "900")
    assert _resolve_run_stall_abort(None, True) == 900.0
    monkeypatch.setenv("AGY_MISSION_CRITICAL_RUN_STALL", "0")
    assert _resolve_run_stall_abort(None, True) is None

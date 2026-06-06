"""Issue #83 — DECOUPLE the watchdog byte budget from the stall budget.

``--watchdog-scale`` multiplies BOTH the verbose byte budget AND the per-worker
stall timeout with one coupled multiplier. A read-heavy primary generator (codex
high-effort told to read a design doc + several modules) blows the ~200KB byte
budget during the legitimate file-reading phase, gets SIGKILLed as
``watchdog:verbose``, and is SILENTLY demoted off the operator's chosen primary
generator onto a fallback worker. The only lever (``--watchdog-scale 5.0``)
over-inflates the stall timeout as collateral. Bytes and stall-time are
independent failure modes and must be independently overridable.

This is the CONTRACT (RED) suite: it pins the two NEW absolute, independent
overrides plus the LOUD lead-verbose-demotion warning, and MUST fail on the
current (un-patched) code.

Fix #1 — two NEW absolute overrides, applied INDEPENDENTLY in
``harness/roles.py::_arm_watchdog`` AFTER the calibrated+scaled budgets:
  * ``--watchdog-max-bytes`` / env ``AGY_WATCHDOG_MAX_BYTES`` -> REPLACE max_bytes.
  * ``--watchdog-stall``     / env ``AGY_WATCHDOG_STALL``     -> REPLACE stall.
An absolute override WINS over ``--watchdog-scale`` for its dimension; the OTHER
dimension still reflects the scale. The legacy AGY_MAX_OUTPUT_BYTES /
AGY_STALL_SECONDS early-return short-circuit (#28/#49 lineage) is untouched.

Fix #2 — a DISTINCT, un-missable WARNING when the LEAD/primary generator
(attempts == 1) is verbose-killed and the chain advances/reroutes, naming the
primary-generator override + the ``--watchdog-max-bytes`` remedy.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

import harness.roles as roles
from agy_orchestrator.core.agent import (
    WATCHDOG_MARKER,
    WATCHDOG_STALLED,
    WATCHDOG_VERBOSE,
    AgentInstance,
)
from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent
from agy_orchestrator.core.calibration import (
    DEFAULT_MAX_BYTES,
    DEFAULT_STALL_SECONDS_BY_WORKER,
    CalibrationTable,
)

# Calibrated baselines for an EMPTY table + the "codex" worker: a fresh
# CalibrationTable returns the conservative per-worker defaults (no data rows).
CAL_BYTES = DEFAULT_MAX_BYTES
CAL_STALL = DEFAULT_STALL_SECONDS_BY_WORKER["codex"]


@pytest.fixture(autouse=True)
def _empty_calibration(monkeypatch):
    """Pin a deterministic, dataless CalibrationTable so the calibrated budgets
    are exactly (CAL_BYTES, CAL_STALL) for codex, and clear every watchdog env
    var so a stray AGY_WATCHDOG_SCALE in the shell can't perturb the math."""
    monkeypatch.setattr(roles, "_get_calibration", lambda: CalibrationTable())
    for var in (
        "AGY_WATCHDOG",
        "AGY_WATCHDOG_SCALE",
        "AGY_WATCHDOG_MAX_BYTES",
        "AGY_WATCHDOG_STALL",
        "AGY_MAX_OUTPUT_BYTES",
        "AGY_STALL_SECONDS",
    ):
        monkeypatch.delenv(var, raising=False)


def _fresh_agent():
    """A minimal stand-in: _arm_watchdog only reads/sets these two attributes."""
    return SimpleNamespace(max_output_bytes=0, stall_seconds=0)


CFG = {"model": "gpt-5.5", "effort": "high"}


# --------------------------------------------------------------------------- #
# Fix #1 — independent absolute overrides in _arm_watchdog
# --------------------------------------------------------------------------- #
def test_max_bytes_override_alone_leaves_stall_calibrated():
    """watchdog_max_bytes (arg) REPLACES bytes; stall stays calibrated (no scale)."""
    agent = _fresh_agent()
    roles._arm_watchdog(agent, "codex", CFG, scale=1.0, watchdog_max_bytes=987_654)
    assert agent.max_output_bytes == 987_654          # overridden
    assert agent.stall_seconds == CAL_STALL           # independent — untouched


def test_stall_override_alone_leaves_bytes_calibrated():
    """watchdog_stall (arg) REPLACES stall; bytes stay calibrated (reverse)."""
    agent = _fresh_agent()
    roles._arm_watchdog(agent, "codex", CFG, scale=1.0, watchdog_stall=4242.0)
    assert agent.stall_seconds == 4242.0              # overridden
    assert agent.max_output_bytes == CAL_BYTES        # independent — untouched


def test_env_vars_apply_independent_overrides(monkeypatch):
    """Env AGY_WATCHDOG_MAX_BYTES / AGY_WATCHDOG_STALL produce the same
    independent overrides as the args."""
    monkeypatch.setenv("AGY_WATCHDOG_MAX_BYTES", "555000")
    agent = _fresh_agent()
    roles._arm_watchdog(agent, "codex", CFG, scale=1.0)
    assert agent.max_output_bytes == 555_000
    assert agent.stall_seconds == CAL_STALL

    monkeypatch.delenv("AGY_WATCHDOG_MAX_BYTES", raising=False)
    monkeypatch.setenv("AGY_WATCHDOG_STALL", "1234.5")
    agent2 = _fresh_agent()
    roles._arm_watchdog(agent2, "codex", CFG, scale=1.0)
    assert agent2.stall_seconds == 1234.5
    assert agent2.max_output_bytes == CAL_BYTES


def test_bad_env_value_is_ignored(monkeypatch):
    """A non-numeric env override is tolerated (ValueError -> ignore), no crash;
    the calibrated value stands."""
    monkeypatch.setenv("AGY_WATCHDOG_MAX_BYTES", "not-a-number")
    monkeypatch.setenv("AGY_WATCHDOG_STALL", "garbage")
    agent = _fresh_agent()
    roles._arm_watchdog(agent, "codex", CFG, scale=1.0)
    assert agent.max_output_bytes == CAL_BYTES
    assert agent.stall_seconds == CAL_STALL


def test_absolute_override_wins_over_scale_per_dimension():
    """An absolute override WINS over a coexisting --watchdog-scale for ITS
    dimension; the OTHER dimension still reflects the scale."""
    agent = _fresh_agent()
    roles._arm_watchdog(
        agent, "codex", CFG, scale=3.0, watchdog_max_bytes=12_345,
    )
    assert agent.max_output_bytes == 12_345           # override beats scale
    assert agent.stall_seconds == CAL_STALL * 3.0     # scale still applies here

    agent2 = _fresh_agent()
    roles._arm_watchdog(
        agent2, "codex", CFG, scale=3.0, watchdog_stall=99.0,
    )
    assert agent2.stall_seconds == 99.0               # override beats scale
    assert agent2.max_output_bytes == int(CAL_BYTES * 3.0)  # scale still applies


# --------------------------------------------------------------------------- #
# Fix #1 — BACK-COMPAT
# --------------------------------------------------------------------------- #
def test_no_overrides_arms_calibrated_scaled_budgets():
    """With NO new overrides, _arm_watchdog arms exactly today's calibrated
    (x scale) budgets — byte-identical behavior."""
    agent = _fresh_agent()
    roles._arm_watchdog(agent, "codex", CFG, scale=2.0)
    assert agent.max_output_bytes == int(CAL_BYTES * 2.0)
    assert agent.stall_seconds == CAL_STALL * 2.0

    agent2 = _fresh_agent()
    roles._arm_watchdog(agent2, "codex", CFG)  # default scale=1.0
    assert agent2.max_output_bytes == CAL_BYTES
    assert agent2.stall_seconds == CAL_STALL


def test_legacy_env_shortcircuit_untouched_by_new_overrides():
    """The legacy AGY_MAX_OUTPUT_BYTES / AGY_STALL_SECONDS early-return
    short-circuit (agent already has explicit budgets) must STILL win — the NEW
    finer overrides do not bypass it."""
    agent = SimpleNamespace(max_output_bytes=999, stall_seconds=0)
    roles._arm_watchdog(
        agent, "codex", CFG, scale=5.0,
        watchdog_max_bytes=5000, watchdog_stall=10.0,
    )
    # Early return: nothing re-armed, the legacy explicit budget stands.
    assert agent.max_output_bytes == 999
    assert agent.stall_seconds == 0


# --------------------------------------------------------------------------- #
# Fix #1 — CLI flags
# --------------------------------------------------------------------------- #
def test_cli_accepts_decouple_flags_and_forwards(monkeypatch):
    """argparse accepts --watchdog-max-bytes / --watchdog-stall and forwards them
    into dispatch kwargs."""
    from harness import cli

    calls = {}

    def fake_dispatch(instruction, **kwargs):
        calls["kwargs"] = kwargs

        class _R:
            success = True
            run_id = None
            error = None
        return _R()

    monkeypatch.setattr(cli, "dispatch", fake_dispatch)
    monkeypatch.setattr(cli.broker_client, "is_running", lambda *a, **k: False)

    rc = cli.main([
        "do", "read the design doc then edit", "--direct",
        "--watchdog-max-bytes", "1500000", "--watchdog-stall", "900",
    ])
    assert rc == 0
    assert calls["kwargs"].get("watchdog_max_bytes") == 1_500_000
    assert calls["kwargs"].get("watchdog_stall") == 900.0


def test_cli_nonpositive_watchdog_max_bytes_fails_fast(monkeypatch):
    """A non-positive --watchdog-max-bytes fails fast (return 1), never dispatches."""
    from harness import cli

    def boom_dispatch(*a, **k):
        raise AssertionError("dispatch must not run on a bad override")

    monkeypatch.setattr(cli, "dispatch", boom_dispatch)
    monkeypatch.setattr(cli.broker_client, "is_running", lambda *a, **k: False)

    rc = cli.main(["do", "x", "--direct", "--watchdog-max-bytes", "-5"])
    assert rc == 1


def test_cli_nonpositive_watchdog_stall_fails_fast(monkeypatch):
    """A non-positive --watchdog-stall fails fast (return 1), never dispatches."""
    from harness import cli

    def boom_dispatch(*a, **k):
        raise AssertionError("dispatch must not run on a bad override")

    monkeypatch.setattr(cli, "dispatch", boom_dispatch)
    monkeypatch.setattr(cli.broker_client, "is_running", lambda *a, **k: False)

    rc = cli.main(["do", "x", "--direct", "--watchdog-stall", "0"])
    assert rc == 1


# --------------------------------------------------------------------------- #
# Fix #2 — LOUD lead-verbose-demotion warning in FallbackAgent
# --------------------------------------------------------------------------- #
class _BaseStub(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]


class _LeadVerbose(_BaseStub):
    """Lead provider killed by the verbose BYTE-budget watchdog (attempts == 1)."""
    async def run_async(self, piped_input=None):
        self.stderr = f"{WATCHDOG_MARKER}{WATCHDOG_VERBOSE}] bytes>budget"
        raise RuntimeError(self.stderr)


class _LeadStall(_BaseStub):
    """Lead provider killed by the STALL watchdog — must NOT trigger the
    verbose-specific warning."""
    async def run_async(self, piped_input=None):
        self.stderr = f"{WATCHDOG_MARKER}{WATCHDOG_STALLED}] no output"
        raise RuntimeError(self.stderr)


class _LeadGenericFail(_BaseStub):
    """Lead provider fails with a plain error (no watchdog marker) -> chain
    advances; used to push a verbose trip onto a NON-lead attempt."""
    async def run_async(self, piped_input=None):
        self.stderr = "boom: plain error"
        raise RuntimeError(self.stderr)


class _MidVerbose(_BaseStub):
    """A NON-lead provider (attempts >= 2) that trips verbose."""
    async def run_async(self, piped_input=None):
        self.stderr = f"{WATCHDOG_MARKER}{WATCHDOG_VERBOSE}] bytes>budget"
        raise RuntimeError(self.stderr)


class _Success(_BaseStub):
    async def run_async(self, piped_input=None):
        self.stdout = "ok"
        self.returncode = 0
        return self.stdout


def _lead_verbose_warning_records(records):
    """A WARNING that names the PRIMARY/lead generator override AND the
    --watchdog-max-bytes remedy (the issue-#83 distinct demotion signal)."""
    out = []
    for r in records:
        if r.levelno != logging.WARNING:
            continue
        msg = r.getMessage().lower()
        if "primary" in msg and "--watchdog-max-bytes" in msg:
            out.append(r)
    return out


def test_lead_verbose_kill_emits_distinct_primary_demotion_warning(caplog):
    Agent = make_fallback_agent(chain=[_LeadVerbose, _Success], cycles=1)
    with caplog.at_level(logging.WARNING):
        out = asyncio.run(Agent(prompt="x").run_async())
    assert out == "ok"
    hits = _lead_verbose_warning_records(caplog.records)
    assert hits, (
        "lead verbose-kill must emit a DISTINCT warning naming the primary "
        "generator override + the --watchdog-max-bytes remedy"
    )


def test_non_lead_verbose_kill_does_not_emit_lead_warning(caplog):
    """A verbose trip on a NON-lead provider (attempts >= 2) is an ordinary
    fallback, NOT the operator-primary override — the special warning must not fire."""
    Agent = make_fallback_agent(chain=[_LeadGenericFail, _MidVerbose, _Success], cycles=1)
    with caplog.at_level(logging.WARNING):
        out = asyncio.run(Agent(prompt="x").run_async())
    assert out == "ok"
    assert not _lead_verbose_warning_records(caplog.records)


def test_lead_stall_kill_does_not_emit_lead_verbose_warning(caplog):
    """A lead STALL trip (hang, not a byte-budget kill) must NOT emit the
    verbose-specific primary-demotion warning — the two failure modes are
    distinct (the whole point of issue #83)."""
    Agent = make_fallback_agent(chain=[_LeadStall, _Success], cycles=1)
    with caplog.at_level(logging.WARNING):
        out = asyncio.run(Agent(prompt="x").run_async())
    assert out == "ok"
    assert not _lead_verbose_warning_records(caplog.records)

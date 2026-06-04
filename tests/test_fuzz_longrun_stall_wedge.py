"""Fuzz / long-run resilience probes — DIMENSION: stall / wedge / liveness.

These probes target failure modes that only bite on multi-step / multi-hour /
mission-critical runs: a chatty-but-wedged provider whose transport-error cadence
just dodges the #72/#73 degradation tracker, the absence of any CUMULATIVE
degradation bound, an active-but-slow worker bounded only by the 160-min default
ceiling, the run-level watchdog defeated by adapter/usage chatter, and the
per-cancelled-branch teardown tax on an unkillable grandchild.

All hermetic + fast: the agent-level probes drive a REAL short-lived subprocess
with TINY real timings engineered to exploit the recovery-window reset (no
multi-hour waits); the run-monitor probe uses an INJECTED clock; the teardown
probe uses a fake unkillable child + a no-op killpg + a short outer timeout.

Each probe asserts the DESIRED resilient behaviour. Where the orchestrator
currently misbehaves, the probe is decorated @pytest.mark.xfail(strict=True) so
the committed suite stays green while the assertion stands as a RED contract for
the fix (it flips to XPASS->fail once fixed). Verdicts (resilient / bug) were
established by RUNNING each assertion WITHOUT the xfail first; see the per-probe
docstrings for the observed baseline.
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from agy_orchestrator.core.agent import (
    ABSOLUTE_TIMEOUT_FACTOR,
    WATCHDOG_TRANSPORT_STALL,
    AgentInstance,
    is_transport_error,
)
from harness.run_monitor import RunMonitor

pytestmark = pytest.mark.not_slow


# The literal codex websocket-reset line from the wedged real run (#72/#73); a
# genuine transport error per is_transport_error.
_RESET_LINE = (
    "codex_api::endpoint::responses_websocket: failed to connect to websocket: "
    "IO error: Connection reset by peer (os error 104)"
)


def test_reset_line_is_a_transport_error():
    """Sanity anchor: the failure line used by the probes below is counted as a
    genuine transport error by the COUNTING helper (so the spell logic sees it)."""
    assert is_transport_error(_RESET_LINE)


class _Base(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]


# --------------------------------------------------------------------------- #
# PROBE 1 (high) — transport spell RESETS forever when errors arrive just OVER
# the recovery window, so neither bound ever fires (chatty-but-wedged at a slow
# cadence: the #72/#73 wedge at a different period).
# --------------------------------------------------------------------------- #
def test_slow_flap_just_over_recovery_window_still_trips():
    """A provider flaps: one websocket-reset stderr line, then a clean gap slightly
    LONGER than transport_recovery_window, repeated indefinitely. The recovery-clear
    branch (agent.py:713-715) closes the spell and zeroes transport_err_count on
    every cycle, then the next error opens a BRAND-NEW spell with count=1. So the
    error count never approaches transport_max_errors and the per-spell duration
    never approaches transport_max_seconds — the tracker never trips for this
    cadence even though the provider is genuinely wedged.

    DESIRED (resilient): a provider that emits N>max_errors genuine transport
    errors over the call — even spaced just over the recovery window — must
    eventually trip WATCHDOG_TRANSPORT_STALL (it is the slow-cadence twin of the
    #72/#73 wedge). Tiny bounds keep it fast.

    OBSERVED BASELINE (assertion run WITHOUT xfail): 8 transport errors spaced
    ~0.7s apart with recovery_window=0.5 over ~5.7s -> _watchdog_reason is None,
    returncode 0. RED -> classified BUG.
    """

    class _SlowFlap(_Base):
        def build_command(self, piped_input=None):
            # 8 cycles: a transport-error stderr line + a clean stdout line, then a
            # 0.7s gap (> recovery_window 0.5). Each gap clears the spell, so the
            # cumulative count (8) never accumulates in a single spell.
            return [
                "bash", "-c",
                "for i in $(seq 1 8); do "
                "echo '" + _RESET_LINE + "' >&2; "
                "echo \"chunk $i\"; "
                "sleep 0.7; done; echo done",
            ]

    agent = _SlowFlap(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 0           # not the trip under test
    agent.timeout = 60
    agent.max_retries = 1
    agent.transport_max_errors = 3    # 8 cumulative errors >> 3
    agent.transport_max_seconds = 0   # disable the per-spell seconds bound
    agent.transport_recovery_window = 0.5

    try:
        asyncio.run(agent.run_async())
    except RuntimeError:
        pass  # a trip surfaces as RuntimeError — acceptable for the resilient path
    # DESIRED: the sustained slow flap is detected as a transport wedge.
    assert agent._watchdog_reason == WATCHDOG_TRANSPORT_STALL


# --------------------------------------------------------------------------- #
# PROBE 2 (medium) — no CUMULATIVE degradation bound: a provider that degrades
# below max_seconds, recovers, and re-degrades burns unbounded aggregate
# wall-clock while no single contiguous spell ever exceeds a bound.
# --------------------------------------------------------------------------- #
def test_cumulative_degradation_across_recoveries_trips():
    """Independent of cadence: even when errors arrive in bursts, the spell-DURATION
    bound (now - transport_spell_start) measures only the CURRENT contiguous spell,
    and the error-count bound is reset to 0 on every recovery. A provider that
    degrades for nearly max_seconds, recovers for one window, then degrades again
    burns cumulative degraded time of many multiples of max_seconds while NO single
    spell ever exceeds either bound — operationally a 2-hour mostly-broken provider
    invisible to the watchdog.

    DESIRED (resilient): total time-in-degradation across recoveries (here >2x
    max_seconds with >2x max_errors total transport errors) must eventually trip.

    OBSERVED BASELINE (assertion run WITHOUT xfail): 3 spells x 3 errors over
    ~7.7s, each spell under max_seconds=2.5 and under max_errors=5 ->
    _watchdog_reason is None, returncode 0. RED -> classified BUG.
    """

    class _RecoverReDegrade(_Base):
        def build_command(self, piped_input=None):
            # 3 cycles. Each cycle: 3 transport errors over ~1.8s (< max_seconds 2.5
            # and < max_errors 5), then a 0.7s clean recovery (> recovery_window 0.5)
            # that clears the spell. Cumulative degraded time ~5.4s >> 2.5.
            return [
                "bash", "-c",
                "for c in 1 2 3; do "
                "  for i in 1 2 3; do echo '" + _RESET_LINE + "' >&2; echo work; sleep 0.6; done; "
                "  echo 'clean recover'; sleep 0.7; "
                "done; echo done",
            ]

    agent = _RecoverReDegrade(prompt="x")
    agent.max_output_bytes = 0
    agent.stall_seconds = 0
    agent.timeout = 120
    agent.max_retries = 1
    agent.transport_max_errors = 5    # no single 3-error spell reaches this
    agent.transport_max_seconds = 2.5  # no single ~1.8s spell exceeds this
    agent.transport_recovery_window = 0.5

    try:
        asyncio.run(agent.run_async())
    except RuntimeError:
        pass
    # DESIRED: cumulative degradation is bounded.
    assert agent._watchdog_reason == WATCHDOG_TRANSPORT_STALL


# --------------------------------------------------------------------------- #
# PROBE 3 (medium) — an active-but-slow worker is bounded ONLY by the 160-min
# default ceiling: the idle window resets on every line, and the absolute cap
# defaults to timeout * 4 with AGY_WORKER_CMD_TIMEOUT / AGY_ABSOLUTE_TIMEOUT off.
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    reason="fuzz-longrun stall_wedge: with AGY_WORKER_CMD_TIMEOUT / "
    "AGY_ABSOLUTE_TIMEOUT unset (the default), an active-but-slow worker is bounded "
    "only by timeout*ABSOLUTE_TIMEOUT_FACTOR = 2400*4 = 9600s (160 min); the idle "
    "window never fires while output dribbles (agent.py:887-901,919-927)",
    strict=True,
)
def test_active_but_slow_worker_has_a_sane_default_ceiling(monkeypatch):
    """In streaming mode `timeout` is the IDLE window and is reset on EVERY output
    line, so a worker emitting one line per <timeout never trips it (#49: a worker
    ran `pytest -m slow` 44+ min). The only backstop is _absolute_cap(), which with
    BOTH opt-in knobs unset defaults to timeout * ABSOLUTE_TIMEOUT_FACTOR. At the
    production default timeout=2400 that is 9600s = 160 minutes PER CALL — and a
    master run with ~20 such steps wedges for the better part of a day on
    accumulated per-step ceilings while the idle watchdog provably never fires.

    DESIRED (resilient): the out-of-the-box per-call ceiling for an active-but-slow
    worker should be operationally sane (well under an hour) without requiring the
    operator to set an opt-in env var.

    OBSERVED BASELINE (assertion run WITHOUT xfail): with the knobs unset and
    timeout=2400, _absolute_cap() == 9600.0 (== timeout*4) -> the only ceiling is
    160 min. RED -> classified BUG (a default-OFF protection is no protection on an
    unattended run).
    """
    monkeypatch.delenv("AGY_WORKER_CMD_TIMEOUT", raising=False)
    monkeypatch.delenv("AGY_ABSOLUTE_TIMEOUT", raising=False)

    agent = _Base(prompt="x")
    agent.worker_cmd_timeout = 0.0
    agent.absolute_timeout = 0.0
    agent.timeout = 2400  # production default

    cap = agent._absolute_cap()
    # Sanity: with neither knob set the cap IS the loose multiple (documents the gap).
    assert cap == 2400 * ABSOLUTE_TIMEOUT_FACTOR == 9600.0
    # DESIRED: a sane default ceiling — under an hour, not 160 minutes.
    assert cap <= 3600, (
        "default per-call ceiling for an active-but-slow worker is %.0fs (%.0f min); "
        "the idle window never fires on a dribbling worker, so this is the ONLY guard"
        % (cap, cap / 60)
    )


def test_active_but_slow_worker_only_stopped_by_absolute_cap(monkeypatch):
    """Companion REGRESSION GUARD (resilient): prove mechanically that the idle
    window does NOT fire for an active worker — only the absolute cap stops it.
    Drive _await_with_liveness with a coroutine that bumps _last_progress every
    50ms forever; with idle window 0.3s and cap 0.3*4=1.2s, the call is killed by
    the WALLCLOCK cap (not the idle window). This passes today and locks in the
    observed mechanism so the gap in PROBE 3 can't silently change shape.
    """
    monkeypatch.delenv("AGY_WORKER_CMD_TIMEOUT", raising=False)
    monkeypatch.delenv("AGY_ABSOLUTE_TIMEOUT", raising=False)

    agent = _Base(prompt="x")
    agent.worker_cmd_timeout = 0.0
    agent.absolute_timeout = 0.0
    agent.timeout = 0.3              # idle window 0.3s; cap = 1.2s
    agent._current_process = None   # bare-coroutine teardown path

    async def _active_forever():
        while True:
            agent._last_progress = time.monotonic()
            await asyncio.sleep(0.05)

    async def _go():
        t0 = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await agent._await_with_liveness(_active_forever(), time.monotonic())
        return time.monotonic() - t0, getattr(agent, "_timeout_kind", None)

    elapsed, kind = asyncio.run(_go())
    # The idle window (0.3s) NEVER fired (output kept bumping it); the wallclock cap
    # (1.2s) did. Proves an active worker is bounded ONLY by the absolute cap.
    assert kind == "wallclock"
    assert elapsed >= 1.1


# --------------------------------------------------------------------------- #
# PROBE 4 (medium) — run-level stall watchdog defeated when a wedged step still
# emits usage / adapter-synthesized 'progress' events (kind not in the
# non-progress allowlist).
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    reason="fuzz-longrun stall_wedge: RunMonitor._NON_PROGRESS_KINDS is only "
    "{stderr,stdout,heartbeat}; per-call 'usage' events (agent.py:485) and "
    "adapter-synthesized lifecycle/reasoning events reset the run progress clock, "
    "so a chatty-but-wedged step defeats --run-stall-abort (run_monitor.py:52,200-206)",
    strict=True,
)
def test_run_watchdog_not_defeated_by_usage_chatter():
    """The run-level watchdog (--run-stall-abort, default-armed to 1800s under
    --mission-critical) is the operator's last line of defence for an unattended
    overnight run. But observe() treats ANY event whose kind is not in
    {stderr,stdout,heartbeat} as run-level forward progress. A wedged step that
    retries internally still emits a per-call 'usage' event (kind='usage') and, when
    a dashboard adapter is importable, codex stdout lines are parsed into
    lifecycle/reasoning events — none in the non-progress set. So since_progress()
    keeps resetting and should_abort() never returns True: the mission-critical run
    that should abort at its window instead wedges to the per-call / whole-run wall.

    DESIRED (resilient): a flood of usage / adapter-synthesized chatter with NO real
    step milestone must NOT mask a wedge — should_abort() fires once the no-real-
    progress window elapses.

    OBSERVED BASELINE (assertion run WITHOUT xfail): 100 'usage' events over 100
    injected-clock seconds with run_stall_abort=10 -> since_progress()==0.0,
    should_abort()==False. RED -> classified BUG.
    """
    clk = [0.0]
    m = RunMonitor(run_id="r", heartbeat_interval=0, run_stall_abort=10,
                   clock=lambda: clk[0])

    # The wedged step emits a per-call usage tick every second (a retry loop's
    # cadence) plus an adapter-synthesized reasoning event — no real step milestone.
    for t in range(1, 101):
        clk[0] = float(t)
        m.observe({"kind": "usage", "data": {"usage_kind": "call"}})
        m.observe({"kind": "reasoning", "data": {"delta": "thinking..."}})

    # DESIRED: 100s of no REAL step progress > the 10s window -> the watchdog fires.
    assert m.should_abort() is True, (
        "should_abort stayed False after 100s of usage/adapter chatter with a 10s "
        "window: since_progress=%.1f" % m.since_progress()
    )


def test_run_watchdog_real_milestone_still_resets_progress():
    """Companion REGRESSION GUARD (resilient): a genuine step milestone DOES reset
    the run progress clock (the watchdog must not be so aggressive it trips on real
    progress). Passes today; pins the intended distinction PROBE 4 relies on."""
    clk = [0.0]
    m = RunMonitor(run_id="r", heartbeat_interval=0, run_stall_abort=10,
                   clock=lambda: clk[0])
    clk[0] = 8.0
    m.observe({"kind": "lifecycle",
               "data": {"orchestration": {"phase": "step", "step_index": 2,
                                          "step_total": 5}}})
    assert m.since_progress() == 0.0
    clk[0] = 9.0
    assert m.should_abort() is False


# --------------------------------------------------------------------------- #
# PROBE 5 (low) — teardown of a cancelled branch blocks on the bounded shield
# wait when a grandchild escaped the process group (killpg misses it, returncode
# never flips), imposing a per-cancelled-branch teardown tax.
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    reason="fuzz-longrun stall_wedge: on a cancelled branch whose child escaped the "
    "process group, _killpg_tree misses it, returncode never flips, and "
    "_await_with_liveness teardown awaits asyncio.shield(task) for the full 10s "
    "bound — a per-cancelled-branch tax exercised on every streaming cancellation "
    "now the watchdog is default-on (agent.py:954-960)",
    strict=True,
)
def test_cancelled_branch_teardown_is_prompt_on_escaped_grandchild():
    """Tree-of-thought / ParallelSwarm cancel losing branches routinely. Post-#72/#73
    the teardown KILLPGs the child then awaits the shielded streaming task with a 10s
    bound. If a grandchild escaped the group (its own setsid — the case
    _post_exit_drain_grace exists for), the killpg misses it, the task's returncode
    never flips, and the teardown blocks for the FULL 10s on EVERY such cancelled
    branch. On a long master run with many ToT steps this compounds.

    DESIRED (resilient): a cancelled branch tears down promptly (small grace) even
    when the child is unkillable — it must not pay a 10s synchronous tax per branch.

    OBSERVED BASELINE (assertion run WITHOUT xfail): with a fake unkillable child
    (returncode stays None, wait() never resolves), a no-op killpg, and a never-
    finishing task, _await_with_liveness teardown does NOT complete within a 2s
    outer bound (it is stuck in the 10s shield wait). RED -> classified BUG.
    """

    class _UnkillableProc:
        returncode = None

        async def wait(self):
            await asyncio.Event().wait()

    agent = _Base(prompt="x")
    agent.timeout = 0.2              # tiny idle window -> enters teardown fast
    agent.absolute_timeout = 0.0
    agent.worker_cmd_timeout = 0.0
    agent._current_process = _UnkillableProc()
    # Grandchild escaped the process group: killpg is a no-op and does NOT flip
    # returncode, so the shielded task never resolves.
    agent._killpg_tree = lambda proc: None

    async def _never():
        await asyncio.Event().wait()

    async def _go():
        t0 = time.monotonic()
        timed_out = False
        try:
            # A prompt teardown returns (idle TimeoutError) well within 2s. A teardown
            # stuck in the 10s shield does not -> outer wait_for raises.
            await asyncio.wait_for(
                agent._await_with_liveness(_never(), time.monotonic()), timeout=2.0)
        except asyncio.TimeoutError:
            timed_out = True
        return timed_out, time.monotonic() - t0

    timed_out, elapsed = asyncio.run(_go())
    # DESIRED: teardown completes promptly (no 10s-per-branch tax) -> no outer timeout.
    assert timed_out is False, (
        "cancelled-branch teardown did not complete within 2s (stuck in the 10s "
        "shield wait on an escaped grandchild); elapsed=%.1fs" % elapsed
    )

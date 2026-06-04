"""Hermetic long-run resilience probes for the cross-provider FallbackAgent.

DIMENSION: Fallback / usage-wall cascade mid-run.

These probes simulate the conditions that only bite on a MULTI-HOUR / mission-
critical run — every provider walling within the same window, a link that flaps
across cycles, a provider that is transport-STALLED (chatty-but-stuck) for the
full watchdog window, and the generator-rotation hand-off after K verify fails.
NOTHING here spawns a real worker CLI or touches the network: every sub-agent is
an in-process AgentInstance stub with a scripted ``run_async``; time is injected
via a fake monotonic clock; backoff sleeps are spied (never real-slept). Each
probe asserts the DESIRED RESILIENT behaviour.

Probes that assert behaviour the orchestrator does NOT yet have are decorated
``@pytest.mark.xfail(strict=True)`` so they document the long-run defect as a RED
contract while keeping the committed suite green (xfail => reported pass today,
flips to a hard failure once the fix lands).
"""
from __future__ import annotations

import asyncio

import pytest

from agy_orchestrator.core.agent import (
    WATCHDOG_MARKER,
    WATCHDOG_TRANSPORT_STALL,
    AgentInstance,
)
from agy_orchestrator.core.agents import fallback_agent as fa
from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent

pytestmark = pytest.mark.not_slow


# --------------------------------------------------------------------------- #
# Hermetic stub sub-agents (no subprocess, no network).
# --------------------------------------------------------------------------- #
class _Base(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["stub-model"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]


def _run(agent):
    return asyncio.run(agent.run_async())


def _spy_sleep(monkeypatch):
    """Replace asyncio.sleep with a non-sleeping spy that records every duration.

    Returns the recording list. Real wall-clock stays sub-second because the spy
    awaits a zero-second sleep instead of the requested duration."""
    durations: list[float] = []
    real_sleep = asyncio.sleep

    async def _fake(delay, *a, **k):  # noqa: ANN001
        durations.append(float(delay))
        return await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _fake)
    return durations


# =========================================================================== #
# PROBE 1 — every provider walls within the same window: the cycles>1 design
# must insert SOME backoff between re-attempts so a rate-limit window can clear,
# instead of spinning the whole cycles*providers budget in seconds.
# =========================================================================== #
def test_probe1_usage_wall_cascade_backs_off_between_cycles(monkeypatch):
    """chain=[A,B,C] cycles=3, all three usage-walled.

    The cycles>1 design exists for the long-run case where a walled provider's
    quota RECOVERS over time (per the docstring). For that to mean anything the
    loop must wait between cycles. DESIRED: at least one inter-attempt backoff
    sleep is issued before the run gives up — so a 60s rate-limit window has a
    chance to clear on a multi-hour build. Currently the usage-wall branch does
    ``last_error = exc; continue`` with NO sleep, burning all 9 attempts in the
    time it takes the CLIs to emit their quota errors."""
    monkeypatch.setenv("AGY_TRANSPORT_BACKOFF", "0")
    sleeps = _spy_sleep(monkeypatch)

    def _wall(name):
        class W(_Base):
            async def run_async(self, piped_input=None):
                self.stderr = "usage limit reached"
                raise RuntimeError("wall")

        W.__name__ = name
        return W

    A, B, C = _wall("A"), _wall("B"), _wall("C")
    Agent = make_fallback_agent(chain=[A, B, C], cycles=3)
    with pytest.raises(RuntimeError, match="exhausted"):
        _run(Agent(prompt="x"))

    # DESIRED resilient behaviour: a usage-wall cascade across cycles applies some
    # backoff so the recover-by-next-cycle premise can actually hold. Observed:
    # zero sleeps — the budget is spent instantly (RED contract for the fix).
    assert sleeps, (
        "usage-wall cascade across cycles applied NO backoff (sleeps=[]); the "
        "cycles>1 recovery premise is defeated — all attempts burn in seconds"
    )


# =========================================================================== #
# PROBE 2 — transport same-provider retry budget never resets across cycles, so
# a flapping link spends its whole budget in cycle 1 and gets none later.
# =========================================================================== #
def test_probe2_transport_retry_budget_resets_each_cycle(monkeypatch):
    """A link that blips once per cycle should get its bounded same-provider
    transport retries AGAIN in a later cycle (a genuinely transient reset).

    chain=[Flaky, Walled] cycles=2: Flaky always emits a transport blip; Walled
    always usage-walls so the chain advances into cycle 2. In cycle 1 Flaky is
    tried 1 + max_transport_retries(=2) = 3 times. DESIRED: cycle 2 grants the
    SAME bounded budget again => 6 total. Observed: the per-provider counter is
    initialised ONCE before the while-loop and never reset, so cycle 2 advances
    after a SINGLE Flaky call => 4 total (RED contract)."""
    monkeypatch.setenv("AGY_TRANSPORT_RETRIES", "2")
    monkeypatch.setenv("AGY_TRANSPORT_BACKOFF", "0")
    _spy_sleep(monkeypatch)

    class Flaky(_Base):
        calls = 0

        async def run_async(self, piped_input=None):
            type(self).calls += 1
            self.stderr = "connection reset by peer"
            raise RuntimeError("blip")

    class Walled(_Base):
        async def run_async(self, piped_input=None):
            self.stderr = "usage limit reached"
            raise RuntimeError("wall")

    Flaky.calls = 0
    Agent = make_fallback_agent(chain=[Flaky, Walled], cycles=2)
    with pytest.raises(RuntimeError, match="exhausted"):
        _run(Agent(prompt="x"))

    # 2 cycles * (1 initial + 2 bounded retries) = 6 when the budget resets.
    assert Flaky.calls == 6, (
        f"transport retry budget did not reset across cycles: Flaky called "
        f"{Flaky.calls}x (expected 6 = 2 cycles x 3). The per-provider counter "
        f"is spent permanently in cycle 1, so later transient blips get no retry"
    )


# =========================================================================== #
# PROBE 3 — stacked transport-stall windows: each bounded same-provider retry of
# a watchdog-stalled provider can re-incur the full AGY_TRANSPORT_MAX_SECONDS,
# with NO aggregate wall-clock cap per provider.
# =========================================================================== #
def test_probe3_transport_stall_retries_have_aggregate_walltime_cap(monkeypatch):
    """A chatty-but-stuck provider trips WATCHDOG_TRANSPORT_STALL after the full
    stall window (~300s by default). FallbackAgent classifies it as transport and
    takes the bounded same-provider retry path; each retry can again run a full
    300s window before re-tripping. With max_transport_retries=2 that is up to
    ~3 x 300s = 900s stuck on ONE wedged provider.

    Simulated with an INJECTED monotonic clock that advances 300s per call (real
    wall-clock stays sub-second). DESIRED: an aggregate wall-clock cap bounds the
    time a single stalled provider can consume across its retries to well under
    the 3-window worst case. Observed: no aggregate cap — total simulated elapsed
    reaches ~900s (RED contract)."""
    monkeypatch.setenv("AGY_TRANSPORT_RETRIES", "2")
    monkeypatch.setenv("AGY_TRANSPORT_BACKOFF", "0")
    _spy_sleep(monkeypatch)

    STALL_WINDOW = 300.0
    clock = {"t": 0.0}

    class Stalled(_Base):
        calls = 0

        async def run_async(self, piped_input=None):
            type(self).calls += 1
            clock["t"] += STALL_WINDOW  # this call burned a full stall window
            self.stderr = f"{WATCHDOG_MARKER}{WATCHDOG_TRANSPORT_STALL}] stuck"
            raise RuntimeError("stall")

    Stalled.calls = 0
    Agent = make_fallback_agent(chain=[Stalled], cycles=1)
    with pytest.raises(RuntimeError, match="exhausted"):
        _run(Agent(prompt="x"))

    # Worst case today is 1 + 2 retries = 3 calls x 300s = 900s on ONE provider.
    # A resilient design would cap aggregate same-provider stall time so a single
    # wedged provider can't silently eat ~15 minutes of a mission-critical run.
    assert clock["t"] <= STALL_WINDOW * 2, (
        f"a single transport-STALLED provider consumed ~{clock['t']:.0f}s of "
        f"simulated wall-clock across {Stalled.calls} retries with no aggregate "
        f"cap (worst case 3 x {STALL_WINDOW:.0f}s = 900s on one dead provider)"
    )


# =========================================================================== #
# PROBE 4 — a permanently transport-stalled provider is NOT fast-failed the way
# a wedged session is; it wastes the same-provider retry budget on futile retries.
# =========================================================================== #
def test_probe4_persistent_transport_stall_fast_fails_like_wedged(monkeypatch):
    """A transport-STALL is, by construction (#72/#73), a provider producing noise
    but no progress for the full watchdog window — effectively wedged at the
    transport layer. ``is_wedged_session`` is explicitly excluded from the futile
    same-provider retry path; a persistent transport-stall is NOT, so it burns
    1 + max_transport_retries calls before advancing.

    DESIRED (symmetry with the wedged-session fast-fail): a provider that trips
    WATCHDOG_TRANSPORT_STALL on EVERY attempt is called once, then the loop
    advances to the next provider — same as a wedged session. Observed: it is
    called 1 + 2 = 3 times (RED contract). A control assertion confirms a
    genuinely wedged session IS called exactly once, proving the asymmetry."""
    monkeypatch.setenv("AGY_TRANSPORT_RETRIES", "2")
    monkeypatch.setenv("AGY_TRANSPORT_BACKOFF", "0")
    _spy_sleep(monkeypatch)

    # --- control: a wedged session already fast-fails (called exactly once). ---
    class Wedged(_Base):
        calls = 0

        async def run_async(self, piped_input=None):
            type(self).calls += 1
            self.stderr = (
                "stdin is closed for this session; rerun exec_command with tty=true"
            )
            raise RuntimeError("wedged")

    class NextW(_Base):
        async def run_async(self, piped_input=None):
            self.stdout = "ok"
            self.returncode = 0
            return "ok"

    Wedged.calls = 0
    WedgedAgent = make_fallback_agent(chain=[Wedged, NextW], cycles=1)
    assert _run(WedgedAgent(prompt="x")) == "ok"
    assert Wedged.calls == 1, "control: wedged session must fast-fail once"

    # --- subject: a persistent transport-stall SHOULD fast-fail the same way. ---
    class Stalled(_Base):
        calls = 0

        async def run_async(self, piped_input=None):
            type(self).calls += 1
            self.stderr = f"{WATCHDOG_MARKER}{WATCHDOG_TRANSPORT_STALL}] stuck"
            raise RuntimeError("stall")

    class NextS(_Base):
        async def run_async(self, piped_input=None):
            self.stdout = "ok"
            self.returncode = 0
            return "ok"

    Stalled.calls = 0
    StalledAgent = make_fallback_agent(chain=[Stalled, NextS], cycles=1)
    assert _run(StalledAgent(prompt="x")) == "ok"
    assert Stalled.calls == 1, (
        f"a permanently transport-stalled provider was retried {Stalled.calls}x "
        f"(1 + transport retries) instead of fast-failing once like a wedged "
        f"session — it disproportionately wastes the mid-run budget"
    )


# =========================================================================== #
# PROBE 5 — generator rotation (#65) onto a usage-walled provider must NOT strand
# the iteration budget; the working provider must still produce real output.
# =========================================================================== #
def test_probe5_rotation_onto_walled_provider_does_not_strand_budget(monkeypatch):
    """After K consecutive verify-failures the adversarial loop bumps the
    generator's rotate_offset to lead with the next configured provider (#65). If
    that next provider has walled MID-RUN, rotation leads with a dead provider.

    DESIRED resilient behaviour: the run must NOT be stranded — the working
    provider's real (failing-but-nonempty) output must still reach the loop on
    every iteration, because FallbackAgent recovers per-call (a walled lead falls
    through to the next available provider WITHIN the same run_async call).

    This probe wires a REAL FallbackAgent generator into the real AdversarialReview
    with a verifier stub that always fails, so the rotation actually fires. It
    asserts the working provider (A) keeps being reached after rotation and the
    final output stays non-empty — i.e. a mid-run-walled rotation target is
    absorbed by fallback recovery rather than wasting the remaining iterations.
    Verdict expectation: RESILIENT (per-call fallback recovery covers it)."""
    monkeypatch.setenv("AGY_GENERATOR_ROTATE_AFTER", "2")
    monkeypatch.setenv("AGY_TRANSPORT_BACKOFF", "0")
    _spy_sleep(monkeypatch)

    from agy_orchestrator.execution.verifier import VerifierResult
    from agy_orchestrator.workflows.adversarial import AdversarialReview

    class A(_Base):
        calls = 0

        async def run_async(self, piped_input=None):
            type(self).calls += 1
            self.stdout = "real-but-failing-change"
            self.returncode = 0
            return self.stdout

    class Walled(_Base):
        calls = 0

        async def run_async(self, piped_input=None):
            type(self).calls += 1
            self.stderr = "usage limit reached"
            raise RuntimeError("wall")

    A.calls = 0
    Walled.calls = 0
    Gen = make_fallback_agent(chain=[A, Walled], cycles=1)
    generator = Gen(prompt="x")

    class _AlwaysFailVerifier:
        async def verify(self, working_directory="."):
            return VerifierResult(ok=False, message="nope")

    class _Critic(_Base):
        async def run_async(self, piped_input=None):
            self.stdout = "not approved; revise"
            self.returncode = 0
            return self.stdout

    review = AdversarialReview(
        generator_instance=generator,
        critic_instance=_Critic(prompt="c"),
        verifier=_AlwaysFailVerifier(),
        max_iterations=4,
    )
    out = asyncio.run(review.execute("do the thing"))

    # Rotation fired (offset advanced to lead with the walled provider)...
    assert review.generator_rotations >= 1, "rotation should have fired after K fails"
    # ...yet the working provider A was still reached on EVERY iteration (4), and
    # the final output is the real change — fallback recovery absorbed the walled
    # lead instead of stranding the remaining iterations on a dead provider.
    assert A.calls == review.max_iterations, (
        f"working provider A reached only {A.calls}/{review.max_iterations} "
        f"iterations after rotation onto a walled provider — budget stranded"
    )
    assert out == "real-but-failing-change", (
        "final generator output went empty/None after rotation onto a walled "
        "provider — the working provider was stranded behind a dead one"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

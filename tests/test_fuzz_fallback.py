"""Hermetic fuzz / property / edge-case tests for the cross-provider FallbackAgent.

NOTHING here spawns a real worker CLI or touches the network: every sub-agent is a
pure in-process stub subclassing AgentInstance with a fake ``run_async``. These
tests assert the *robust* contract of ``make_fallback_agent`` and its helper
functions and must all stay GREEN. Genuine defects found while writing these
(e.g. the unbounded watchdog-rule ping-pong) are reported separately as findings
with their own repros, NOT committed here as failing tests.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from agy_orchestrator.core.agent import (
    WATCHDOG_MARKER,
    WATCHDOG_TRANSPORT_STALL,
    AgentInstance,
)
from agy_orchestrator.core.agents.fallback_agent import (
    _reason_category,
    _transport_backoff_seconds,
    _transport_retries,
    _watchdog_reason_in,
    make_fallback_agent,
)


# --------------------------------------------------------------------------- #
# Stub sub-agents (hermetic, no subprocess, no network).
# --------------------------------------------------------------------------- #
class _BaseStub(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["stub-model"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]


def _ok_stub(name, output):
    class _Ok(_BaseStub):
        async def run_async(self, piped_input=None):
            self.stdout = output
            self.returncode = 0
            return output
    _Ok.__name__ = name
    return _Ok


def _fail_stub(name, stderr=""):
    class _Fail(_BaseStub):
        async def run_async(self, piped_input=None):
            self.stderr = stderr
            self.returncode = 1
            raise RuntimeError(stderr or "fail")
    _Fail.__name__ = name
    return _Fail


def _run(agent):
    return asyncio.run(agent.run_async())


# --------------------------------------------------------------------------- #
# Constructor guards / boundary inputs.
# --------------------------------------------------------------------------- #
def test_empty_chain_rejected():
    with pytest.raises(ValueError):
        make_fallback_agent(chain=[], cycles=1)


@pytest.mark.parametrize("cycles", [0, -1, -1000])
def test_non_positive_cycles_rejected(cycles):
    with pytest.raises(ValueError):
        make_fallback_agent(chain=[_ok_stub("A", "a")], cycles=cycles)


def test_single_element_chain_success():
    A = _ok_stub("A", "out-a")
    Agent = make_fallback_agent(chain=[A], cycles=1)
    assert _run(Agent(prompt="x")) == "out-a"


def test_single_element_chain_exhaustion_message_is_clean():
    F = _fail_stub("F", "boom")
    Agent = make_fallback_agent(chain=[F], cycles=3)
    with pytest.raises(RuntimeError) as ei:
        _run(Agent(prompt="x"))
    msg = str(ei.value)
    # Bounded, informative, mentions the provider/cycle accounting.
    assert "exhausted" in msg
    assert "1 providers x 3 cycles" in msg
    assert "boom" in msg


def test_cycles_one_tries_each_provider_exactly_once():
    calls = {"a": 0, "b": 0}

    class A(_BaseStub):
        async def run_async(self, piped_input=None):
            calls["a"] += 1
            self.stderr = "wall"
            raise RuntimeError("a-fail")

    class B(_BaseStub):
        async def run_async(self, piped_input=None):
            calls["b"] += 1
            self.stdout = "b-ok"
            self.returncode = 0
            return "b-ok"

    Agent = make_fallback_agent(chain=[A, B], cycles=1)
    assert _run(Agent(prompt="x")) == "b-ok"
    assert calls == {"a": 1, "b": 1}


def test_chain_repeats_for_each_cycle_until_recovery():
    # A fails the first two passes, succeeds on the third — cycles=3 must reach it.
    state = {"n": 0}

    class Flaky(_BaseStub):
        async def run_async(self, piped_input=None):
            state["n"] += 1
            if state["n"] < 3:
                self.stderr = "wall"
                raise RuntimeError("not yet")
            self.stdout = "finally"
            self.returncode = 0
            return "finally"

    Agent = make_fallback_agent(chain=[Flaky], cycles=3)
    assert _run(Agent(prompt="x")) == "finally"
    assert state["n"] == 3


# --------------------------------------------------------------------------- #
# rotate_offset behaviour.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "off,expected_leader",
    [
        (0, "P1"),
        (1, "P2"),
        (2, "P3"),
        (3, "P1"),     # wraps
        (6, "P1"),     # wraps multiple times
        (-1, "P3"),    # negative -> python modulo wraps to last
        (-3, "P1"),
        (True, "P2"),  # bool is an int subclass
        (False, "P1"),
        (None, "P1"),  # falsy -> no rotation
        (0.0, "P1"),   # falsy float -> no rotation
    ],
)
def test_rotate_offset_picks_leader_robustly(off, expected_leader):
    P1, P2, P3 = _ok_stub("P1", "p1"), _ok_stub("P2", "p2"), _ok_stub("P3", "p3")
    Agent = make_fallback_agent(chain=[P1, P2, P3], cycles=1)
    a = Agent(prompt="x")
    a.rotate_offset = off
    _run(a)
    assert a.last_provider == expected_leader


def test_rotate_offset_default_absent_is_no_rotation():
    P1, P2 = _ok_stub("P1", "p1"), _ok_stub("P2", "p2")
    Agent = make_fallback_agent(chain=[P1, P2], cycles=1)
    a = Agent(prompt="x")  # never sets rotate_offset
    assert _run(a) == "p1"
    assert a.last_provider == "P1"


# --------------------------------------------------------------------------- #
# Construction-failure resilience.
# --------------------------------------------------------------------------- #
def test_construction_failure_skips_provider_not_crash():
    class BadCtor(_BaseStub):
        def __init__(self, *a, **k):
            raise ValueError("ctor boom")

        async def run_async(self, piped_input=None):
            return "never"

    Good = _ok_stub("Good", "good")
    Agent = make_fallback_agent(chain=[BadCtor, Good], cycles=1)
    # BadCtor can't be built; the chain must fall through to Good, not crash.
    assert _run(Agent(prompt="x")) == "good"


def test_all_construction_failures_raise_bounded_runtimeerror():
    class BadCtor(_BaseStub):
        def __init__(self, *a, **k):
            raise ValueError("ctor boom")

        async def run_async(self, piped_input=None):
            return "never"

    Agent = make_fallback_agent(chain=[BadCtor], cycles=2)
    with pytest.raises(RuntimeError) as ei:
        _run(Agent(prompt="x"))
    assert "ctor boom" in str(ei.value)


# --------------------------------------------------------------------------- #
# Watchdog-rule re-routing: a WELL-FORMED rule pre-empts the normal sequence
# once, then re-enters the normal sequence and terminates.
# --------------------------------------------------------------------------- #
def test_well_formed_watchdog_rule_reroutes_then_terminates():
    class Verbose(_BaseStub):
        async def run_async(self, piped_input=None):
            self.stderr = f"{WATCHDOG_MARKER}verbose] bytes>budget"
            raise RuntimeError("verbose")

    Terse = _ok_stub("Terse", "terse-ok")
    Normal = _ok_stub("Normal", "normal-ok")
    Agent = make_fallback_agent(
        chain=[Verbose, Normal], cycles=1, watchdog_rules={"verbose": [Terse]}
    )
    assert _run(Agent(prompt="x")) == "terse-ok"


def test_watchdog_rule_target_failure_falls_back_to_normal():
    class Verbose(_BaseStub):
        async def run_async(self, piped_input=None):
            self.stderr = f"{WATCHDOG_MARKER}verbose] x"
            raise RuntimeError("verbose")

    # The rule target also fails (but with a PLAIN error, no watchdog reason),
    # so after the rule pre-empt the normal sequence must still be reached.
    BadTarget = _fail_stub("BadTarget", "plain-fail")
    Normal = _ok_stub("Normal", "normal-ok")
    Agent = make_fallback_agent(
        chain=[Verbose, Normal], cycles=1, watchdog_rules={"verbose": [BadTarget]}
    )
    assert _run(Agent(prompt="x")) == "normal-ok"


def test_empty_watchdog_rule_target_list_is_noop():
    class Verbose(_BaseStub):
        async def run_async(self, piped_input=None):
            self.stderr = f"{WATCHDOG_MARKER}verbose] x"
            raise RuntimeError("verbose")

    Normal = _ok_stub("Normal", "normal-ok")
    Agent = make_fallback_agent(
        chain=[Verbose, Normal], cycles=1, watchdog_rules={"verbose": []}
    )
    assert _run(Agent(prompt="x")) == "normal-ok"


# --------------------------------------------------------------------------- #
# Transport-blip same-provider bounded retry (issue #57).
# --------------------------------------------------------------------------- #
def test_transport_blip_retries_same_provider_then_advances(monkeypatch):
    monkeypatch.setenv("AGY_TRANSPORT_RETRIES", "2")
    monkeypatch.setenv("AGY_TRANSPORT_BACKOFF", "0")  # no real sleep
    calls = {"t": 0, "n": 0}

    class Transport(_BaseStub):
        async def run_async(self, piped_input=None):
            calls["t"] += 1
            self.stderr = "connection reset by peer"
            raise RuntimeError("blip")

    class Next(_BaseStub):
        async def run_async(self, piped_input=None):
            calls["n"] += 1
            self.stdout = "next-ok"
            self.returncode = 0
            return "next-ok"

    Agent = make_fallback_agent(chain=[Transport, Next], cycles=1)
    assert _run(Agent(prompt="x")) == "next-ok"
    # Transport retried at most (1 + max_retries) = 3 times, never unboundedly.
    assert calls["t"] <= 1 + 2
    assert calls["t"] >= 2  # at least one bounded retry happened
    assert calls["n"] == 1


def test_transport_retry_count_is_bounded_per_provider(monkeypatch):
    monkeypatch.setenv("AGY_TRANSPORT_RETRIES", "2")
    monkeypatch.setenv("AGY_TRANSPORT_BACKOFF", "0")
    calls = {"t": 0}

    class Transport(_BaseStub):
        async def run_async(self, piped_input=None):
            calls["t"] += 1
            self.stderr = "websocket stream error"
            raise RuntimeError("blip")

    Agent = make_fallback_agent(chain=[Transport], cycles=1)
    with pytest.raises(RuntimeError):
        _run(Agent(prompt="x"))
    # 1 initial + at most 2 bounded same-provider retries.
    assert calls["t"] <= 3


def test_wedged_session_does_not_take_transport_retry_path(monkeypatch):
    # A wedged session whose accumulated stderr ALSO carries transport noise must
    # NOT loop on the same provider; it must advance to the next provider.
    monkeypatch.setenv("AGY_TRANSPORT_RETRIES", "2")
    monkeypatch.setenv("AGY_TRANSPORT_BACKOFF", "0")
    calls = {"w": 0, "n": 0}

    class Wedged(_BaseStub):
        async def run_async(self, piped_input=None):
            calls["w"] += 1
            # both a transport marker and the wedged marker present
            self.stderr = (
                "connection reset\n"
                "stdin is closed for this session; rerun exec_command with tty=true"
            )
            raise RuntimeError("wedged")

    class Next(_BaseStub):
        async def run_async(self, piped_input=None):
            calls["n"] += 1
            self.stdout = "next-ok"
            self.returncode = 0
            return "next-ok"

    Agent = make_fallback_agent(chain=[Wedged, Next], cycles=1)
    assert _run(Agent(prompt="x")) == "next-ok"
    assert calls["w"] == 1  # NOT retried on the same provider
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# Session-id reuse only within the same provider class.
# --------------------------------------------------------------------------- #
def test_session_id_only_reused_by_same_provider_class():
    from agy_orchestrator.core.agents.claude_agent import ClaudeAgent
    from agy_orchestrator.core.agents.grok_agent import GrokAgent

    seen_session = {}

    class FakeClaude(ClaudeAgent):
        async def run_async(self, piped_input=None):
            # First claude run produces a session id.
            self.session_id = "claude-sid-1"
            self.stdout = "c-ok"
            self.returncode = 0
            return "c-ok"

    class FakeGrok(GrokAgent):
        async def run_async(self, piped_input=None):
            # Grok must NEVER be handed claude's session id.
            seen_session["grok"] = getattr(self, "session_id", None)
            self.stdout = "g-ok"
            self.returncode = 0
            return "g-ok"

    # First call: claude succeeds, records session owner.
    Agent = make_fallback_agent(chain=[FakeClaude], cycles=1)
    a = Agent(prompt="x")
    assert _run(a) == "c-ok"
    assert a.session_id == "claude-sid-1"

    # Second wrapper carrying claude's session, but routing to a grok sub: the
    # grok sub must NOT receive the claude session id.
    Agent2 = make_fallback_agent(chain=[FakeGrok], cycles=1)
    b = Agent2(prompt="x")
    b.session_id = "claude-sid-1"
    b._session_owner = FakeClaude  # a different class than FakeGrok
    assert _run(b) == "g-ok"
    # Whatever grok saw, it must NOT be claude's session id (cross-class leak).
    assert seen_session["grok"] != "claude-sid-1"


# --------------------------------------------------------------------------- #
# _watchdog_reason_in parsing — adversarial stderr.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "stderr,expected",
    [
        ("", None),
        ("no marker here", None),
        (f"{WATCHDOG_MARKER}stalled]", "stalled"),
        (f"{WATCHDOG_MARKER}verbose] extra text", "verbose"),
        (f"prefix {WATCHDOG_MARKER}{WATCHDOG_TRANSPORT_STALL}] suffix", WATCHDOG_TRANSPORT_STALL),
        (f"{WATCHDOG_MARKER}]", None),       # empty slug -> None
        (f"{WATCHDOG_MARKER}   ]", None),    # whitespace-only slug -> None
        (f"{WATCHDOG_MARKER}weird\x00null]", "weird\x00null"),  # NUL doesn't crash
    ],
)
def test_watchdog_reason_in_parsing(stderr, expected):
    assert _watchdog_reason_in(stderr) == expected


def test_watchdog_reason_in_marker_without_close_bracket():
    # A truncated marker with no closing ']' must not crash; the whole tail is
    # treated as the slug.
    out = _watchdog_reason_in(f"{WATCHDOG_MARKER}stalled-no-close")
    assert out == "stalled-no-close"


def test_watchdog_reason_in_none_input():
    assert _watchdog_reason_in(None) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# _reason_category precedence (issue #47 / #57 / #62).
# --------------------------------------------------------------------------- #
def test_reason_category_precedence():
    # context_overflow wins over everything.
    assert _reason_category(
        looked_like_usage=True, watchdog_reason="stalled",
        context_overflow=True, transport=True, wedged=True,
    ) == "context_overflow"
    # wedged wins over usage/transport/stalled.
    assert _reason_category(
        looked_like_usage=True, watchdog_reason="stalled",
        context_overflow=False, transport=True, wedged=True,
    ) == "wedged"
    # usage wins over transport.
    assert _reason_category(
        looked_like_usage=True, watchdog_reason=None, transport=True,
    ) == "usage"
    # transport flag.
    assert _reason_category(
        looked_like_usage=False, watchdog_reason=None, transport=True,
    ) == "transport"
    # watchdog transport-stall funnels to transport.
    assert _reason_category(
        looked_like_usage=False, watchdog_reason=WATCHDOG_TRANSPORT_STALL,
    ) == "transport"
    assert _reason_category(
        looked_like_usage=False, watchdog_reason="verbose",
    ) == "verbose"
    assert _reason_category(
        looked_like_usage=False, watchdog_reason="stalled",
    ) == "stalled"
    # nothing -> generic error.
    assert _reason_category(
        looked_like_usage=False, watchdog_reason=None,
    ) == "error"


# --------------------------------------------------------------------------- #
# Transport-retry env parsing robustness (malformed env must not crash).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "val,expected",
    [("0", 0), ("1", 1), ("2", 2), ("5", 2), ("-3", 0), ("", 2), ("garbage", 2)],
)
def test_transport_retries_env_parsing(monkeypatch, val, expected):
    monkeypatch.setenv("AGY_TRANSPORT_RETRIES", val)
    assert _transport_retries() == expected


def test_transport_retries_unset(monkeypatch):
    monkeypatch.delenv("AGY_TRANSPORT_RETRIES", raising=False)
    assert _transport_retries() == 2


@pytest.mark.parametrize("val", ["", "garbage", "nan-ish"])
def test_transport_backoff_malformed_env_defaults(monkeypatch, val):
    monkeypatch.setenv("AGY_TRANSPORT_BACKOFF", val)
    # Must not raise; falls back to a sane base.
    assert _transport_backoff_seconds(1) == pytest.approx(0.5)


def test_transport_backoff_scales_with_attempt(monkeypatch):
    monkeypatch.setenv("AGY_TRANSPORT_BACKOFF", "0.25")
    assert _transport_backoff_seconds(1) == pytest.approx(0.25)
    assert _transport_backoff_seconds(2) == pytest.approx(0.50)


# --------------------------------------------------------------------------- #
# Observability callbacks are best-effort: a raising callback must not break a run.
# --------------------------------------------------------------------------- #
def test_raising_event_callback_does_not_break_run():
    F = _fail_stub("F", "wall")
    Good = _ok_stub("Good", "good")
    Agent = make_fallback_agent(chain=[F, Good], cycles=1)
    a = Agent(prompt="x")

    def boom(_ev):
        raise RuntimeError("callback exploded")

    a.event_callback = boom
    # The reroute emit uses event_callback; a raising one must be swallowed.
    assert _run(a) == "good"


def test_raising_post_construct_hook_does_not_block_call():
    Good = _ok_stub("Good", "good")

    def bad_hook(sub, cls):
        raise RuntimeError("hook boom")

    Agent = make_fallback_agent(chain=[Good], cycles=1, post_construct_hook=bad_hook)
    assert _run(Agent(prompt="x")) == "good"


def test_raising_event_callback_factory_falls_back_to_plain_callback():
    Good = _ok_stub("Good", "good")
    Agent = make_fallback_agent(chain=[Good], cycles=1)
    a = Agent(prompt="x")
    a.event_callback = lambda ev: None
    a.event_callback_factory = lambda cls, sub: (_ for _ in ()).throw(RuntimeError("factory boom"))
    # Factory raising must not crash sub construction.
    assert _run(a) == "good"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

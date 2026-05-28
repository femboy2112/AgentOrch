"""Harness must arm the streaming watchdog on every agent it builds:

  - single-agent path: agent gets per-config (max_output_bytes, stall_seconds)
    from the CalibrationTable.
  - FallbackAgent path: each sub built by _make_sub gets armed via the
    post_construct_hook installed by roles.build_role_agent.
  - AGY_WATCHDOG=off cleanly disables arming so the legacy fail-fast-only
    behaviour is one env-var away.
"""
from __future__ import annotations

import asyncio

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.core.calibration import (
    DEFAULT_MAX_BYTES,
    DEFAULT_STALL_SECONDS,
)
from harness import roles


@pytest.fixture(autouse=True)
def _isolate_calibration(monkeypatch):
    """Each test sees a fresh CalibrationTable (no stale data leaking in)."""
    roles._reset_calibration()
    yield
    roles._reset_calibration()


def test_single_agent_path_arms_safe_defaults(monkeypatch):
    monkeypatch.delenv("AGY_WATCHDOG", raising=False)
    agent = roles.build_role_agent(["codex"], fallback=False)
    # With no calibration data on disk, safe defaults apply (never false-trip).
    assert agent.max_output_bytes == DEFAULT_MAX_BYTES
    assert agent.stall_seconds == DEFAULT_STALL_SECONDS


def test_agy_watchdog_off_skips_arming(monkeypatch):
    monkeypatch.setenv("AGY_WATCHDOG", "off")
    agent = roles.build_role_agent(["codex"], fallback=False)
    # Disabled means the agent keeps its constructor defaults (0/0).
    assert agent.max_output_bytes == 0
    assert agent.stall_seconds == 0


def test_fallback_path_arms_each_sub_via_hook(monkeypatch):
    monkeypatch.delenv("AGY_WATCHDOG", raising=False)
    fb = roles.build_role_agent(["codex", "agy"], fallback=True, cycles=1)
    # Sub-agents are built lazily by _make_sub on demand; call it manually
    # and check the sub was armed.
    sub_codex = fb._make_sub(roles.AGENT_CLASSES["codex"])
    sub_agy = fb._make_sub(roles.AGENT_CLASSES["agy"])
    assert isinstance(sub_codex, AgentInstance)
    assert isinstance(sub_agy, AgentInstance)
    assert sub_codex.max_output_bytes == DEFAULT_MAX_BYTES
    assert sub_codex.stall_seconds == DEFAULT_STALL_SECONDS
    assert sub_agy.max_output_bytes == DEFAULT_MAX_BYTES
    assert sub_agy.stall_seconds == DEFAULT_STALL_SECONDS


def test_env_var_override_is_respected(monkeypatch):
    """If the operator explicitly sets a budget via env var, the harness must
    NOT clobber it from the calibration table."""
    monkeypatch.delenv("AGY_WATCHDOG", raising=False)
    monkeypatch.setenv("AGY_MAX_OUTPUT_BYTES", "12345")
    monkeypatch.setenv("AGY_STALL_SECONDS", "67")
    agent = roles.build_role_agent(["codex"], fallback=False)
    assert agent.max_output_bytes == 12345
    assert agent.stall_seconds == 67


def test_grok_with_na_effort_does_not_crash():
    """grok carries effort='n/a' (the CLI rejects --effort). The arming path
    treats 'n/a' as no effort key — must not raise or break the lookup."""
    agent = roles.build_role_agent(["grok"], fallback=False)
    assert agent.max_output_bytes == DEFAULT_MAX_BYTES


def test_post_construct_hook_swallows_exceptions(monkeypatch):
    """A bad hook must never block a real call — the wrapper logs and continues."""
    from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent

    class _Stub(AgentInstance):
        @classmethod
        async def get_available_models(cls):
            return ["x"]

        @classmethod
        async def get_model_usage(cls, model):
            return 100.0

        def build_command(self, piped_input=None):
            return ["true"]

    def bad_hook(sub, cls):
        raise RuntimeError("boom")

    Agent = make_fallback_agent([_Stub], cycles=1, post_construct_hook=bad_hook)
    out = asyncio.run(Agent(prompt="x").run_async())
    # _Stub runs `true` -> returncode 0 -> empty stdout, but the call completes.
    assert out == ""

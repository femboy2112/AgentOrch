import os

from agy_orchestrator.core.agents.agy_agent import AgyAgent


def test_agy_agent_sets_browser_noop_env_override():
    agent = AgyAgent(prompt="x")
    assert agent.extra_env == {"BROWSER": "/bin/true"}


def test_child_env_merges_parent_env_and_extra_env():
    agent = AgyAgent(prompt="x")
    env = agent._child_env()
    assert env is not None
    assert env["BROWSER"] == "/bin/true"
    assert "PATH" in os.environ
    assert env["PATH"] == os.environ["PATH"]


def test_child_env_none_when_extra_env_empty_and_bounds_disabled(monkeypatch):
    # With worker resource bounding off and no extra_env, the inherit-unchanged
    # path still returns None (prior behavior).
    monkeypatch.setenv("AGY_WORKER_RESOURCE_BOUND", "0")
    agent = AgyAgent(prompt="x")
    agent.extra_env = {}
    assert agent._child_env() is None


def test_build_command_includes_print_timeout_default_and_override(monkeypatch):
    monkeypatch.delenv("AGY_PRINT_TIMEOUT", raising=False)
    agent = AgyAgent(prompt="x")
    cmd = agent.build_command()
    assert "agy" in cmd
    assert "--print" in cmd
    assert "--dangerously-skip-permissions" in cmd
    assert "--print-timeout" in cmd
    idx = cmd.index("--print-timeout")
    assert cmd[idx + 1] == "300s"

    monkeypatch.setenv("AGY_PRINT_TIMEOUT", "120s")
    cmd_override = AgyAgent(prompt="x").build_command()
    idx_override = cmd_override.index("--print-timeout")
    assert cmd_override[idx_override + 1] == "120s"

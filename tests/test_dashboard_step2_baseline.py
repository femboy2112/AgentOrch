from agy_orchestrator.core.agents.claude_agent import ClaudeAgent
from harness import roles


def test_claude_normal_cli_path_uses_json_output_format():
    agent = ClaudeAgent(prompt="baseline", model="sonnet", effort="high")
    cmd = agent.build_command()
    assert "--output-format" in cmd
    idx = cmd.index("--output-format")
    assert cmd[idx + 1] == "json"
    assert "stream-json" not in cmd


def test_default_harness_chains_exclude_claude():
    assert roles.GENERATOR_CHAIN == ["codex", "agy", "grok"]
    assert roles.CRITIC_CHAIN == ["agy", "codex", "grok"]
    assert "claude" not in roles.GENERATOR_CHAIN
    assert "claude" not in roles.CRITIC_CHAIN

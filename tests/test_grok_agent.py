"""GrokAgent command construction + output parsing.

Grok-build rejects the reasoning-effort parameter, so the command must NEVER
carry --effort/--reasoning-effort, and the JSON envelope must be unwrapped to
its "text" field. These are pure-construction tests — no network/CLI calls.
"""
from __future__ import annotations

from agy_orchestrator.core.agents.grok_agent import GrokAgent
from harness import roles


def test_command_omits_effort_and_uses_prompt_file():
    agent = GrokAgent(prompt="do the thing", model="grok-build", effort="high")
    cmd = agent._build_cmd("/tmp/p.txt")
    assert "--effort" not in cmd
    assert "--reasoning-effort" not in cmd
    assert cmd[:1] == ["grok"]
    assert "--prompt-file" in cmd and "/tmp/p.txt" in cmd
    assert "--output-format" in cmd and "json" in cmd
    assert "bypassPermissions" in cmd
    assert "--always-approve" in cmd
    assert "-m" in cmd and "grok-build" in cmd


def test_web_search_toggle():
    on = GrokAgent(prompt="x", model="grok-build")._build_cmd("/tmp/p.txt")
    assert "--disable-web-search" not in on
    off = GrokAgent(prompt="x", model="grok-build", web_search=False)._build_cmd("/tmp/p.txt")
    assert "--disable-web-search" in off


def test_resume_session_when_known():
    agent = GrokAgent(prompt="x", model="grok-build", session_id="abc-123")
    cmd = agent._build_cmd("/tmp/p.txt")
    assert "--resume" in cmd and "abc-123" in cmd


def test_extract_text_from_json_envelope():
    raw = '{"text": "PONG", "stopReason": "EndTurn", "sessionId": "s1", "thought": "..."}'
    assert GrokAgent._extract_text(raw) == "PONG"
    assert GrokAgent._extract_session_id(raw) == "s1"


def test_extract_text_tolerates_leading_log_noise():
    raw = 'WARN something\n{"text": "hello", "sessionId": "s2"}\n'
    assert GrokAgent._extract_text(raw) == "hello"


def test_extract_text_falls_back_to_raw_on_plain_output():
    assert GrokAgent._extract_text("just plain text") == "just plain text"


def test_filter_stderr_drops_known_noise():
    agent = GrokAgent(prompt="x")
    noisy = (
        "ERROR failed to watch root recursively: PermissionDenied\n"
        "real error here\n"
        "Shell cwd was reset to /home/leah/Agent64"
    )
    cleaned = agent.filter_stderr(noisy)
    assert "real error here" in cleaned
    assert "watch root recursively" not in cleaned
    assert "Shell cwd was reset" not in cleaned


def test_registered_in_roles():
    assert roles.AGENT_CLASSES["grok"] is GrokAgent
    name, cfg = roles._cfg_for_token("grok")
    assert name == "grok"
    assert cfg["model"] == "grok-build"
    # describe_chain must not KeyError (needs model + effort keys present).
    assert "grok" in roles.describe_chain(["grok"], fallback=False)

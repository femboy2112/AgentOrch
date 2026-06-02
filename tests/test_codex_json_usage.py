"""Hermetic tests for the codex --json usage/observability wiring.

No real ``codex`` process is spawned: we exercise build_command (flag presence),
_extract_usage (parsing + cache-inclusive normalization), and _postprocess
(output-last-message file primary, JSONL message fallback) directly.

Why this matters: without ``--json`` codex reports only a "tokens used: N" total
on stderr, so per-call cache_read is invisible and W2's cache_read_ratio is always
None for the dominant worker. --json surfaces turn.completed usage incl.
cached_input_tokens; -o/--output-last-message keeps clean agent-text capture even
though stdout is now a JSONL event stream.
"""
from __future__ import annotations

import json
import os

from agy_orchestrator.core.agents.codex_agent import CodexAgent
from harness.dispatch import _cache_read_ratio


def test_build_command_has_json_and_output_file() -> None:
    agent = CodexAgent(prompt="hi")
    cmd = agent.build_command()
    assert "--json" in cmd
    assert "--output-last-message" in cmd
    # the flag is followed by a concrete path, stable per instance
    path = cmd[cmd.index("--output-last-message") + 1]
    assert path and agent._last_message_path() == path
    # still reads the prompt from stdin and keeps the sandbox-bypass flags
    assert cmd[-1] == "-"
    assert "--skip-git-repo-check" in cmd


def test_standard_model_maps_to_account_valid_spark() -> None:
    # The harness default codex model is the "standard" alias. It MUST map to the
    # ChatGPT-account-valid "gpt-5.3-codex-spark" — the bare "gpt-5.3-codex" 400s
    # ("model is not supported when using Codex with a ChatGPT account"), which
    # silently fell codex back off every call.
    cmd = CodexAgent(prompt="hi", model="standard").build_command()
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "gpt-5.3-codex-spark"
    assert "gpt-5.3-codex" not in cmd  # the bare (rejected) name must not appear


def test_explicit_model_passes_through_and_none_omits_flag() -> None:
    # An explicit override is honored verbatim (operator's choice / #42).
    cmd = CodexAgent(prompt="hi", model="gpt-5.5").build_command()
    assert cmd[cmd.index("--model") + 1] == "gpt-5.5"
    # No model => no --model flag => codex uses the account default (also valid).
    assert "--model" not in CodexAgent(prompt="hi").build_command()


def test_extract_usage_normalizes_cache_inclusive() -> None:
    # codex/OpenAI: input_tokens is INCLUSIVE of cached. Numbers from the real probe.
    line = json.dumps({
        "type": "turn.completed",
        "usage": {"input_tokens": 34406, "cached_input_tokens": 21888,
                  "output_tokens": 6, "total_tokens": 34412},
    })
    res = CodexAgent(prompt="x")._extract_usage(line + "\n", "")
    assert res["token_source"] == "cli"
    assert res["cache_read_tokens"] == 21888
    # normalized to FRESH input = total - cached, matching the Anthropic convention
    assert res["input_tokens"] == 34406 - 21888  # 12518
    assert res["output_tokens"] == 6
    # and the W2 hit-rate now computes correctly (cached / total input)
    ratio = _cache_read_ratio(res["cache_read_tokens"], res["input_tokens"])
    assert ratio == round(21888 / 34406, 4)  # ~0.6362


def test_extract_usage_cold_call_no_cache() -> None:
    line = json.dumps({
        "type": "turn.completed",
        "usage": {"input_tokens": 5000, "output_tokens": 10},
    })
    res = CodexAgent(prompt="x")._extract_usage(line + "\n", "")
    # no cached field -> fresh input passes through unchanged, cache_read None
    assert res["input_tokens"] == 5000
    assert res["cache_read_tokens"] is None


def test_postprocess_prefers_output_file(tmp_path) -> None:
    agent = CodexAgent(prompt="x")
    p = tmp_path / "lastmsg.txt"
    p.write_text("FINAL ANSWER from -o file")
    agent._codex_last_msg_path = str(p)
    # stdout is JSONL noise; the clean text must come from the file
    out = agent._postprocess('{"type":"turn.completed","usage":{}}\n')
    assert out == "FINAL ANSWER from -o file"
    # file is consumed (unlinked) so it can't leak into a later call
    assert not p.exists()


def test_postprocess_jsonl_fallback() -> None:
    agent = CodexAgent(prompt="x")
    # no -o file set -> reconstruct from message events
    agent._codex_last_msg_path = "/nonexistent/codex_lastmsg_test.txt"
    stdout = "\n".join([
        json.dumps({"type": "reasoning", "text": "thinking..."}),
        json.dumps({"type": "message", "text": "Hello "}),
        json.dumps({"item": {"type": "message", "text": "world"}}),
        json.dumps({"type": "turn.completed", "usage": {}}),
    ])
    assert agent._postprocess(stdout) == "Hello world"


def test_postprocess_raw_fallback_on_no_message() -> None:
    agent = CodexAgent(prompt="x")
    agent._codex_last_msg_path = "/nonexistent/codex_lastmsg_test2.txt"
    # a pre-turn error with no message events -> return raw unchanged
    raw = "codex: fatal error before any turn\n"
    assert agent._postprocess(raw) == raw

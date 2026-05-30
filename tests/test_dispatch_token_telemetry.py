import json

from harness.dispatch import _summarize_token_usage
from agy_orchestrator.core.agents.codex_agent import CodexAgent


def test_summarize_token_usage_rolls_up_cli_and_unavailable(tmp_path) -> None:
    events_path = tmp_path / "events.jsonl"
    rows = [
        {
            "kind": "usage",
            "worker": "codex",
            "model": "gpt-5.5",
            "data": {
                "usage_kind": "call",
                "token_source": "cli",
                "input_tokens": 100,
                "output_tokens": 40,
                "cache_read_tokens": 7,
                "total_tokens": 254,
            },
        },
        {
            "kind": "usage",
            "worker": "agy",
            "model": "Gemini 3.1 Pro (High)",
            "data": {
                "usage_kind": "call",
                "token_source": "unavailable",
                "input_tokens": None,
                "output_tokens": None,
                "cache_read_tokens": None,
                "total_tokens": None,
            },
        },
        {
            "kind": "usage",
            "worker": "codex",
            "model": "gpt-5.5",
            "data": {
                "usage_kind": "stream",
                "token_source": "cli",
                "input_tokens": 1,
                "output_tokens": 1,
            },
        },
    ]
    events_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    summary = _summarize_token_usage(events_path)

    assert summary["total_calls"] == 2
    assert summary["per_worker"]["codex"]["token_source"] == "cli"
    assert summary["per_worker"]["codex"]["input_tokens"] == 100
    assert summary["per_worker"]["codex"]["output_tokens"] == 40
    assert summary["per_worker"]["codex"]["total_tokens"] == 254
    assert summary["per_worker"]["agy"]["token_source"] == "unavailable"
    assert summary["per_worker"]["agy"]["input_tokens"] is None
    assert summary["per_worker"]["agy"]["total_tokens"] is None
    assert summary["grand_total"]["input_tokens"] == 100
    assert summary["grand_total"]["output_tokens"] == 40
    assert summary["grand_total"]["total_tokens"] == 254


def test_codex_extract_usage_fallback() -> None:
    agent = CodexAgent(prompt="test")
    
    # Test plain newline format
    res1 = agent._extract_usage("...assistant text...\ntokens used\n254\n", "")
    assert res1["token_source"] == "cli"
    assert res1["total_tokens"] == 254
    assert res1["input_tokens"] is None
    assert res1["output_tokens"] is None

    # Test inline format with comma
    res2 = agent._extract_usage("...assistant text...\ntokens used: 1,234\n", "")
    assert res2["token_source"] == "cli"
    assert res2["total_tokens"] == 1234
    
    # Real codex prints "tokens used\n<N>" on STDERR, not stdout — must be parsed.
    res_err = agent._extract_usage("hi", "session id: x\ncodex\nhi\ntokens used\n257\n")
    assert res_err["token_source"] == "cli"
    assert res_err["total_tokens"] == 257

    # Multiple "tokens used" lines: take the last (final per-call total).
    res_multi = agent._extract_usage("", "tokens used\n10\nmore\ntokens used\n42\n")
    assert res_multi["total_tokens"] == 42

    # Test unavailable when no match and no JSON
    res3 = agent._extract_usage("just some output", "")
    assert res3["token_source"] == "unavailable"
    assert res3["total_tokens"] is None

import json

from harness.dispatch import _summarize_token_usage


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
    assert summary["per_worker"]["agy"]["token_source"] == "unavailable"
    assert summary["per_worker"]["agy"]["input_tokens"] is None
    assert summary["grand_total"]["input_tokens"] == 100
    assert summary["grand_total"]["output_tokens"] == 40

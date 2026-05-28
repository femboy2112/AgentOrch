from __future__ import annotations

import json

from dashboard.adapters import (
    AdapterCtx,
    parse_agy_stderr,
    parse_agy_stdout,
    parse_claude_stream_line,
    parse_codex_stream_line,
    parse_grok_stream_line,
)


def test_codex_adapter_maps_core_event_kinds():
    ctx = AdapterCtx()

    started = parse_codex_stream_line('{"type":"turn.started"}', ctx)
    assert started and started[0]["kind"] == "lifecycle"

    reasoning = parse_codex_stream_line('{"type":"reasoning","text":"thinking"}', ctx)
    assert reasoning == [{"kind": "reasoning", "text": "thinking"}]

    tool = parse_codex_stream_line('{"type":"tool_call","name":"read_file","args":{"path":"x"}}', ctx)
    assert tool[0]["kind"] == "tool_call"
    assert tool[0]["data"]["name"] == "read_file"

    completed = parse_codex_stream_line(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 11, "output_tokens": 5, "reasoning_tokens": 3},
            }
        ),
        ctx,
    )
    assert any(ev["kind"] == "usage" for ev in completed)
    assert any(ev["kind"] == "lifecycle" for ev in completed)


def test_claude_stream_adapter_maps_reasoning_messages_tools_and_usage():
    ctx = AdapterCtx()

    start = parse_claude_stream_line('{"type":"message_start"}', ctx)
    assert start[0]["kind"] == "lifecycle"

    think = parse_claude_stream_line(
        '{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"hmm"}}', ctx
    )
    assert think == [{"kind": "reasoning", "text": "hmm"}]

    text = parse_claude_stream_line(
        '{"type":"content_block_delta","delta":{"type":"text_delta","text":"done"}}', ctx
    )
    assert text == [{"kind": "message", "text": "done"}]

    call = parse_claude_stream_line(
        '{"type":"content_block_start","content_block":{"type":"tool_use","name":"grep","input":{"q":"x"}}}',
        ctx,
    )
    assert call[0]["kind"] == "tool_call"
    assert call[0]["data"]["name"] == "grep"

    usage = parse_claude_stream_line(
        '{"type":"message_delta","usage":{"input_tokens":7,"output_tokens":9,"reasoning_tokens":2}}',
        ctx,
    )
    assert usage[0]["kind"] == "usage"
    assert usage[0]["data"]["in_tokens"] == 7


def test_agy_adapter_maps_stderr_watchdog_and_stdout_message():
    events = parse_agy_stderr("[watchdog:verbose] stream exceeded")
    assert any(ev["kind"] == "stderr" for ev in events)
    assert any(ev["kind"] == "watchdog" for ev in events)

    out = parse_agy_stdout("final answer")
    assert out == [{"kind": "message", "text": "final answer"}]


def test_grok_adapter_maps_thought_text_and_stop_reason():
    ctx = AdapterCtx()
    events = parse_grok_stream_line(
        '{"thought":"trace","text":"result","stopReason":"EndTurn","usage":{"input_tokens":2,"output_tokens":4}}',
        ctx,
    )
    kinds = [ev["kind"] for ev in events]
    assert "reasoning" in kinds
    assert "message" in kinds
    assert "usage" in kinds
    assert "lifecycle" in kinds

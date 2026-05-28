from __future__ import annotations

from typing import List

from dashboard.adapters.base import AdapterCtx, load_json_line


def parse_stream_line(line: str, ctx: AdapterCtx) -> List[dict]:
    payload = load_json_line(line)
    if payload is None:
        return []

    out: List[dict] = []
    thought = payload.get("thought")
    if thought:
        out.append({"kind": "reasoning", "text": str(thought)})

    text = payload.get("text")
    if text:
        out.append({"kind": "message", "text": str(text)})

    usage = payload.get("usage")
    if isinstance(usage, dict):
        out.append(
            {
                "kind": "usage",
                "data": {
                    "in_tokens": usage.get("input_tokens"),
                    "out_tokens": usage.get("output_tokens"),
                    "reasoning_tokens": usage.get("reasoning_tokens"),
                    "cost_usd": usage.get("cost_usd"),
                    "api_ms": usage.get("api_ms"),
                },
            }
        )

    stop_reason = payload.get("stopReason")
    if stop_reason:
        out.append({"kind": "lifecycle", "data": {"event": "agent_finished", "detail": str(stop_reason)}})

    return out

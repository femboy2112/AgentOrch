from __future__ import annotations

from typing import List

from dashboard.adapters.base import AdapterCtx, load_json_line

_REASONING_TYPES = {
    "reasoning",
    "item.reasoning",
    "response.reasoning.delta",
}
_MESSAGE_TYPES = {
    "message",
    "item.message",
    "response.output_text.delta",
}


def parse_stream_line(line: str, ctx: AdapterCtx) -> List[dict]:
    payload = load_json_line(line)
    if payload is None:
        return []

    out: List[dict] = []
    event_type = str(payload.get("type") or payload.get("event") or "")

    if event_type == "turn.started":
        out.append({"kind": "lifecycle", "data": {"event": "agent_started", "detail": "turn.started"}})
        return out

    if event_type == "turn.completed":
        usage = payload.get("usage") or payload.get("data", {}).get("usage") or {}
        if isinstance(usage, dict) and usage:
            out.append(
                {
                    "kind": "usage",
                    "data": {
                        "in_tokens": usage.get("in_tokens") or usage.get("input_tokens"),
                        "out_tokens": usage.get("out_tokens") or usage.get("output_tokens"),
                        "reasoning_tokens": usage.get("reasoning_tokens"),
                        "cost_usd": usage.get("cost_usd"),
                        "api_ms": usage.get("api_ms"),
                    },
                }
            )
        out.append({"kind": "lifecycle", "data": {"event": "agent_finished", "detail": "turn.completed"}})
        return out

    if event_type in _REASONING_TYPES:
        text = payload.get("text") or payload.get("delta") or ""
        if text:
            out.append({"kind": "reasoning", "text": str(text)})
        return out

    if event_type in _MESSAGE_TYPES:
        text = payload.get("text") or payload.get("delta") or payload.get("message") or ""
        if text:
            out.append({"kind": "message", "text": str(text)})
        return out

    if event_type in {"tool_call", "item.tool_call"}:
        out.append(
            {
                "kind": "tool_call",
                "data": {
                    "name": payload.get("name") or payload.get("tool"),
                    "args": payload.get("args") or payload.get("input") or {},
                },
            }
        )
        return out

    if event_type in {"tool_result", "item.tool_result"}:
        out.append(
            {
                "kind": "tool_result",
                "data": {
                    "name": payload.get("name") or payload.get("tool"),
                    "summary": payload.get("summary") or payload.get("result") or payload.get("output"),
                },
            }
        )
        return out

    item = payload.get("item")
    if isinstance(item, dict):
        item_type = item.get("type")
        if item_type == "reasoning" and item.get("text"):
            out.append({"kind": "reasoning", "text": str(item["text"])})
        elif item_type == "message" and item.get("text"):
            out.append({"kind": "message", "text": str(item["text"])})
        elif item_type == "tool_call":
            out.append({"kind": "tool_call", "data": {"name": item.get("name"), "args": item.get("args") or {}}})
        elif item_type == "tool_result":
            out.append(
                {
                    "kind": "tool_result",
                    "data": {"name": item.get("name"), "summary": item.get("summary") or item.get("output")},
                }
            )
    return out

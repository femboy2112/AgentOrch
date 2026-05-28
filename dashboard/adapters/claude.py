from __future__ import annotations

from typing import List

from dashboard.adapters.base import AdapterCtx, load_json_line


def parse_stream_line(line: str, ctx: AdapterCtx) -> List[dict]:
    payload = load_json_line(line)
    if payload is None:
        return []

    out: List[dict] = []
    event_type = str(payload.get("type") or "")

    if event_type == "message_start":
        out.append({"kind": "lifecycle", "data": {"event": "agent_started", "detail": "message_start"}})
        return out

    if event_type == "content_block_start":
        block = payload.get("content_block") or {}
        if block.get("type") == "tool_use":
            name = block.get("name")
            ctx.state["last_tool"] = name
            out.append({"kind": "tool_call", "data": {"name": name, "args": block.get("input") or {}}})
        elif block.get("type") == "tool_result":
            out.append(
                {
                    "kind": "tool_result",
                    "data": {
                        "name": ctx.state.get("last_tool"),
                        "summary": block.get("content") or block.get("text") or "",
                    },
                }
            )
        return out

    if event_type == "content_block_delta":
        delta = payload.get("delta") or {}
        delta_type = delta.get("type")
        if delta_type in {"thinking_delta", "reasoning_delta"}:
            text = delta.get("thinking") or delta.get("text") or ""
            if text:
                out.append({"kind": "reasoning", "text": str(text)})
        elif delta_type == "text_delta":
            text = delta.get("text") or ""
            if text:
                out.append({"kind": "message", "text": str(text)})
        return out

    if event_type == "message_delta":
        usage = payload.get("usage")
        if isinstance(usage, dict):
            out.append(
                {
                    "kind": "usage",
                    "data": {
                        "in_tokens": usage.get("input_tokens"),
                        "out_tokens": usage.get("output_tokens"),
                        "reasoning_tokens": usage.get("reasoning_tokens"),
                    },
                }
            )
        return out

    if event_type == "message_stop":
        out.append({"kind": "lifecycle", "data": {"event": "agent_finished", "detail": "message_stop"}})
        return out

    if event_type == "result":
        text = payload.get("result")
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
                    },
                }
            )
        out.append({"kind": "lifecycle", "data": {"event": "agent_finished", "detail": "result"}})
    return out

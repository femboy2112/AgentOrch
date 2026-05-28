import json
from typing import List, Optional

from agy_orchestrator.core.agent import AgentInstance
from dashboard.adapters import AdapterCtx, parse_codex_stream_line


class CodexAgent(AgentInstance):
    @staticmethod
    def _parse_stream_line(line: str) -> List[dict]:
        text = line.strip()
        if not text:
            return []
        try:
            obj = json.loads(text)
        except Exception:
            return []
        if not isinstance(obj, dict):
            return []

        typ = str(obj.get("type") or "")
        item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
        out: List[dict] = []

        if typ in {"turn.started", "turn.completed"}:
            out.append({"kind": "lifecycle", "data": {"event": typ, "detail": obj}})
            if typ == "turn.completed":
                usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else payload.get("usage")
                if isinstance(usage, dict):
                    out.append({
                        "kind": "usage",
                        "data": {
                            "in_tokens": usage.get("input_tokens"),
                            "out_tokens": usage.get("output_tokens"),
                            "reasoning_tokens": usage.get("reasoning_tokens"),
                            "cost_usd": usage.get("cost_usd"),
                            "api_ms": usage.get("api_ms"),
                        },
                    })
            return out

        if "reasoning" in typ or item.get("type") == "reasoning":
            delta = (
                obj.get("delta")
                or item.get("text")
                or payload.get("text")
                or obj.get("text")
                or ""
            )
            if delta:
                return [{"kind": "reasoning", "text": str(delta), "data": {}}]
            return []

        if "tool_call" in typ or item.get("type") in {"tool_call", "function_call"}:
            name = item.get("name") or obj.get("name") or payload.get("name")
            args = item.get("args") or obj.get("args") or payload.get("args")
            return [{
                "kind": "tool_call",
                "data": {"name": name or "tool", "args": args or {}},
            }]

        if "tool_result" in typ or item.get("type") in {"tool_result", "function_result"}:
            name = item.get("name") or obj.get("name") or payload.get("name")
            summary = item.get("summary") or obj.get("summary") or payload.get("summary")
            return [{
                "kind": "tool_result",
                "data": {"name": name or "tool", "summary": summary or ""},
            }]

        if "message" in typ or item.get("type") == "message":
            msg = (
                item.get("text")
                or obj.get("text")
                or payload.get("text")
                or obj.get("message")
                or ""
            )
            if msg:
                return [{"kind": "message", "text": str(msg), "data": {}}]

        return []

    def _events_from_stdout_line(self, line: str) -> List[dict]:
        return self._parse_stream_line(line)

    def _events_from_stderr_line(self, line: str) -> List[dict]:
        parsed = self._parse_stream_line(line)
        if parsed:
            return parsed
        return super()._events_from_stderr_line(line)

    @classmethod
    async def get_available_models(cls) -> List[str]:
        return ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-5.3-codex-spark", "gpt-5.2"]

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        # Codex exposes no machine-readable remaining-usage value, so report
        # "full" and let real quota exhaustion be handled by the fallback layer
        # (a usage wall surfaces as a non-zero exit -> fail over). Spawning
        # `codex --usage` only to discard its output was pure overhead.
        return 100.0

    def filter_stderr(self, stderr: str) -> str:
        lines = stderr.splitlines()
        filtered = [ln for ln in lines if "network" not in ln.lower() and "timeout" not in ln.lower()]
        return "\n".join(filtered)

    def _full_prompt(self, piped_input: Optional[str] = None) -> str:
        full_prompt = self.prompt
        if piped_input:
            full_prompt += f"\n\n[Context]:\n{piped_input}"
        return full_prompt

    def _stdin_bytes(self, piped_input: Optional[str] = None) -> bytes:
        # Deliver the prompt on stdin (codex exec reads it when '-' is the prompt
        # arg) so large diff-feedback prompts can't hit MAX_ARG_STRLEN (128 KiB).
        return self._full_prompt(piped_input).encode()

    def _events_from_stdout_line(self, line: str) -> List[dict]:
        ctx = getattr(self, "_adapter_ctx", None)
        if ctx is None:
            ctx = AdapterCtx()
            self._adapter_ctx = ctx
        return parse_codex_stream_line(line, ctx)

    def build_command(self, piped_input: Optional[str] = None) -> List[str]:
        cmd = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
        ]

        if self.model and self.model != "standard":
            cmd.extend(["--model", self.model])
        elif self.model == "standard":
            cmd.extend(["--model", "gpt-5.3-codex"])

        if hasattr(self, "effort") and self.effort:
            effort = "xhigh" if self.effort == "max" else self.effort
            cmd.extend(["-c", f"model_reasoning_effort=\"{effort}\""])

        # Arbitrary `-c key=value` config overrides (e.g. "tools.web_search=true").
        for entry in getattr(self, "config_overrides", None) or []:
            cmd.extend(["-c", entry])

        for k, v in self.additional_flags.items():
            cmd.extend([f"--{k}", str(v)])

        # '-' tells codex exec to read the prompt from stdin (see _stdin_bytes).
        cmd.append("-")
        return cmd

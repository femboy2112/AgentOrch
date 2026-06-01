import json
import os
import tempfile
from typing import List, Optional

from agy_orchestrator.core.agent import AgentInstance


class CodexAgent(AgentInstance):
    # Single source of truth for codex model names (used by the async API below
    # and synchronously by harness/effort_overrides.py for --codex-model
    # validation without spinning an event loop).
    AVAILABLE_MODELS: List[str] = [
        "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-5.3-codex-spark", "gpt-5.2",
    ]

    @classmethod
    async def get_available_models(cls) -> List[str]:
        return list(cls.AVAILABLE_MODELS)

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

    def _postprocess(self, raw_stdout: str) -> str:
        # Under --json the agent's answer is no longer plain stdout (that's now a
        # JSONL event stream). Prefer codex's --output-last-message file (the clean
        # final message); fall back to concatenating message-event text from the
        # JSONL; last resort, return raw_stdout unchanged (e.g. a pre-turn error).
        path = getattr(self, "_codex_last_msg_path", None)
        if path:
            try:
                text = ""
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                if text.strip():
                    return text
            except Exception:
                pass
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        msg = self._message_text_from_jsonl(raw_stdout)
        return msg if msg is not None else raw_stdout

    @staticmethod
    def _message_text_from_jsonl(raw_stdout: str) -> Optional[str]:
        # Reconstruct the assistant message from codex --json events. Mirrors the
        # message shapes the dashboard codex adapter handles: a top-level
        # {"type":"message"|"item.message"|"response.output_text.delta", "text"|"delta"|"message"}
        # or a nested {"item": {"type":"message", "text": ...}}.
        parts: List[str] = []
        saw_message = False
        for line in raw_stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if not isinstance(ev, dict):
                continue
            etype = str(ev.get("type") or ev.get("event") or "")
            if etype in ("message", "item.message", "response.output_text.delta"):
                txt = ev.get("text") or ev.get("delta") or ev.get("message")
                if txt:
                    parts.append(str(txt))
                    saw_message = True
                continue
            item = ev.get("item")
            if isinstance(item, dict) and item.get("type") == "message" and item.get("text"):
                parts.append(str(item["text"]))
                saw_message = True
        if not saw_message:
            return None
        return "".join(parts)

    def _events_from_stdout_line(self, line: str) -> List[dict]:
        try:
            from dashboard.adapters import AdapterCtx, parse_codex_stream_line
        except Exception:
            return []
        ctx = getattr(self, "_adapter_ctx", None)
        if ctx is None:
            ctx = AdapterCtx()
            self._adapter_ctx = ctx
        return parse_codex_stream_line(line, ctx)

    def _events_from_stderr_line(self, line: str) -> List[dict]:
        try:
            from dashboard.adapters import AdapterCtx, parse_codex_stream_line
        except Exception:
            return super()._events_from_stderr_line(line)
        ctx = getattr(self, "_adapter_ctx", None)
        if ctx is None:
            ctx = AdapterCtx()
            self._adapter_ctx = ctx
        parsed = parse_codex_stream_line(line, ctx)
        if parsed:
            return parsed
        return super()._events_from_stderr_line(line)

    def _extract_usage(self, raw_stdout: str, raw_stderr: str) -> dict:
        # codex usage is available only when JSON turn events are present.
        usage = {}
        for line in raw_stdout.splitlines():
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            event_type = payload.get("type") or payload.get("event")
            if event_type == "turn.completed":
                row = payload.get("usage") or payload.get("data", {}).get("usage") or {}
                if isinstance(row, dict):
                    usage = row
        if not usage:
            import re
            # codex prints the "tokens used\n<N>" summary on STDERR, not stdout;
            # scan both and take the last match (the final per-call total).
            matches = re.findall(
                r"tokens used(?:[:\s]+)([\d,]+)",
                f"{raw_stdout}\n{raw_stderr}",
                re.IGNORECASE,
            )
            if matches:
                try:
                    total = int(matches[-1].replace(",", ""))
                    return {
                        "token_source": "cli",
                        "input_tokens": None,
                        "output_tokens": None,
                        "cache_read_tokens": None,
                        "total_tokens": total,
                    }
                except Exception:
                    pass
            return {
                "token_source": "unavailable",
                "input_tokens": None,
                "output_tokens": None,
                "cache_read_tokens": None,
                "total_tokens": None,
            }
        raw_input = usage.get("input_tokens") or usage.get("in_tokens")
        cache_read = (
            usage.get("cache_read_tokens")
            or usage.get("cached_input_tokens")
            or usage.get("cached_tokens")
        )
        # codex/OpenAI report ``input_tokens`` INCLUSIVE of the cached prompt
        # tokens (the probe: total input stays ~constant while cached grows on a
        # warm prefix). The rest of the stack — and dispatch._cache_read_ratio —
        # treats input_tokens as the FRESH, reprocessed input (the Anthropic /
        # claude convention, cache-exclusive). Normalize codex to match so the
        # hit-rate is correct and cross-provider sums line up: fresh = total - cached.
        fresh_input = raw_input
        if isinstance(raw_input, (int, float)) and isinstance(cache_read, (int, float)):
            fresh_input = max(int(raw_input) - int(cache_read), 0)
        return {
            "token_source": "cli",
            "input_tokens": fresh_input,
            "output_tokens": usage.get("output_tokens") or usage.get("out_tokens"),
            "cache_read_tokens": cache_read,
            "total_tokens": usage.get("total_tokens"),
        }

    def _last_message_path(self) -> str:
        # Stable per-instance path so build_command and _postprocess agree. Unique
        # per (pid, instance) so concurrent ToT branches (separate instances) and
        # parallel processes don't collide. codex overwrites it each run; we read
        # and unlink it in _postprocess.
        path = getattr(self, "_codex_last_msg_path", None)
        if not path:
            path = os.path.join(
                tempfile.gettempdir(), f"codex_lastmsg_{os.getpid()}_{id(self)}.txt"
            )
            self._codex_last_msg_path = path
        return path

    def build_command(self, piped_input: Optional[str] = None) -> List[str]:
        cmd = [
            "codex",
            "exec",
            # --json makes codex emit JSONL events to stdout, which is the ONLY way
            # it reports structured per-call usage (turn.completed -> input/output/
            # cached_input_tokens). Without it we get only a "tokens used: N" total on
            # stderr (cache_read invisible). It also feeds the dashboard codex adapter
            # real stream events. The agent's final text no longer lands on stdout as
            # plain text, so -o/--output-last-message captures it cleanly instead (see
            # _postprocess).
            "--json",
            "--output-last-message",
            self._last_message_path(),
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

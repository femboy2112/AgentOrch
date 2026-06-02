import json
import logging
import re
from typing import List, Optional

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.core.model_discovery import discover_models

logger = logging.getLogger(__name__)

# Static fallback model list (claude has no documented models subcommand; bare
# names resolve to the CLI's current dated default).
_CLAUDE_FALLBACK_MODELS = ["haiku", "sonnet", "opus"]

# Map orchestrator-internal names to claude --model values. "opus"/"sonnet"/
# "haiku" pass through unchanged — the CLI resolves them to its CURRENT default
# dated model (as of 2026-05-28: opus->claude-opus-4-8, sonnet->claude-sonnet-4-6,
# haiku->claude-haiku-4-5). To PIN a specific version, pass the full hyphenated id
# as the model (e.g. "claude-opus-4-7", "claude-opus-4-8", with an optional "[1m]"
# suffix for the 1M-context variant on opus 4.7+): it falls through
# MODEL_ALIASES.get(model, model) unchanged to `claude --model`.
# VERIFIED 2026-05-28: the `--model` FLAG accepts only bare "opus"/"sonnet"/"haiku"
# or the full "claude-opus-4-N" form. The interactive Claude Code `/model` aliases
# ("opus4.8", "opus4.7") do NOT work as `--model` values — they error.
MODEL_ALIASES = {
    "standard": "sonnet",
}


class ClaudeAgent(AgentInstance):
    """
    Claude CLI agent that pipes prompts via stdin to avoid ARG_MAX limits,
    and supports session reuse via --resume / --fork-session for warm cache hits.

    Session behaviour:
      - session_id=None (default): starts a fresh session, captures session ID
        from JSON output and stores it on self.session_id for callers to reuse.
      - session_id=<uuid>: resumes that session (warm prompt cache).
      - fork_session=True: resumes the given session_id but forks it into a new
        session (used for ToT branches so they don't pollute the main thread).
    """

    def __init__(self, *args, session_id: Optional[str] = None,
                 fork_session: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id: Optional[str] = session_id
        self.fork_session: bool = fork_session
        self.dashboard_stream_json: bool = False

    @classmethod
    async def get_available_models(cls) -> List[str]:
        # claude has no documented model-list subcommand, so we don't shell out
        # (``list_argv=None``) and the static list is the fallback. Routed through
        # discover_models to keep the path uniform + memoized across adapters.
        return await discover_models(
            "claude",
            list_argv=None,
            parse=lambda _stdout: [],
            fallback=_CLAUDE_FALLBACK_MODELS,
        )

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        # No machine-readable remaining-usage value is exposed, so report "full"
        # and let real quota exhaustion be handled by the fallback layer. Spawning
        # `claude --usage` only to discard its output was pure overhead.
        return 100.0

    def _build_base_cmd(self) -> List[str]:
        """Build the claude command. Prompt is always passed via stdin."""
        # Protected surface: this is the normal orchestrator path and must stay on
        # `--output-format json`. Dashboard-only dispatch wiring may opt into
        # stream-json separately, but must not change this default path.
        output_format = "stream-json" if self.dashboard_stream_json else "json"
        cmd = ["claude", "-p", "-", "--output-format", output_format, "--dangerously-skip-permissions"]

        model = MODEL_ALIASES.get(self.model, self.model) if self.model else None
        if model:
            cmd.extend(["--model", model])

        if hasattr(self, "effort") and self.effort:
            cmd.extend(["--effort", self.effort])

        # Session reuse
        if self.session_id:
            cmd.extend(["--resume", self.session_id])
            if self.fork_session:
                cmd.append("--fork-session")

        for k, v in self.additional_flags.items():
            cmd.extend([f"--{k}", str(v)])

        return cmd

    def build_command(self, piped_input: Optional[str] = None) -> List[str]:
        # Prompt is handled via stdin in run_async; this satisfies the ABC.
        return self._build_base_cmd()

    def _build_full_prompt(self, piped_input: Optional[str] = None) -> str:
        full_prompt = self.prompt
        if piped_input:
            full_prompt += f"\n\n[Context]:\n{piped_input}"
        return full_prompt

    @staticmethod
    def _extract_session_id(raw_stdout: str) -> Optional[str]:
        """Pull session_id from --output-format json output.

        Parse the whole payload as one JSON value first (it may be a single object
        or a list of objects) — the previous line-by-line scan failed on a
        pretty-printed multi-line object, silently dropping the session id and
        defeating warm-cache reuse. Regex is the genuine last-resort fallback.
        """
        try:
            payload = json.loads(raw_stdout)
            blobs = payload if isinstance(payload, list) else [payload]
            for blob in blobs:
                if isinstance(blob, dict) and blob.get("session_id"):
                    return blob["session_id"]
        except Exception:
            pass
        m = re.search(
            r'"session_id"\s*:\s*"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"',
            raw_stdout,
        )
        if m:
            return m.group(1)
        for line in raw_stdout.splitlines():
            try:
                blob = json.loads(line)
            except Exception:
                continue
            if isinstance(blob, dict) and blob.get("session_id"):
                return blob["session_id"]
        return None

    @staticmethod
    def _extract_result_text(raw_stdout: str) -> str:
        """Extract the plain text result from --output-format json output."""
        try:
            blobs = json.loads(raw_stdout)
            if not isinstance(blobs, list):
                blobs = [blobs]
            for blob in blobs:
                if isinstance(blob, dict) and blob.get("type") == "result":
                    return blob.get("result", raw_stdout)
        except Exception:
            pass
        chunks: List[str] = []
        for line in raw_stdout.splitlines():
            try:
                blob = json.loads(line)
            except Exception:
                continue
            if not isinstance(blob, dict):
                continue
            if blob.get("type") == "result":
                return str(blob.get("result", "")).strip() or raw_stdout
            if blob.get("type") == "text":
                txt = blob.get("text")
                if txt:
                    chunks.append(str(txt))
        if chunks:
            return "".join(chunks)
        return raw_stdout

    def _events_from_stdout_line(self, line: str) -> List[dict]:
        if not self.dashboard_stream_json:
            return []
        try:
            from dashboard.adapters import AdapterCtx, parse_claude_stream_line
        except Exception:
            return []
        ctx = getattr(self, "_adapter_ctx", None)
        if ctx is None:
            ctx = AdapterCtx()
            self._adapter_ctx = ctx
        return parse_claude_stream_line(line, ctx)

    def _events_from_stdout_complete(self, raw_stdout: str) -> List[dict]:
        if self.dashboard_stream_json:
            return []
        text = self._extract_result_text(raw_stdout).strip()
        if not text:
            return []
        return [{"kind": "message", "text": text, "data": {}}]

    def _stdin_bytes(self, piped_input: Optional[str] = None) -> bytes:
        """Deliver the prompt via stdin (dodges ARG_MAX on large diff-feedback)."""
        return self._build_full_prompt(piped_input).encode()

    def _postprocess(self, raw_stdout: str) -> str:
        """Capture the resumable session id, then return the JSON result text."""
        sid = self._extract_session_id(raw_stdout)
        if sid:
            if not self.session_id:
                logger.info("New session established: %s", sid)
            self.session_id = sid
        return self._extract_result_text(raw_stdout)

    def _extract_usage(self, raw_stdout: str, raw_stderr: str) -> dict:
        """Parse usage from Claude's normal JSON/stream-json output."""
        usage = {}
        try:
            payload = json.loads(raw_stdout)
            blobs = payload if isinstance(payload, list) else [payload]
            for blob in blobs:
                if isinstance(blob, dict) and isinstance(blob.get("usage"), dict):
                    usage = blob.get("usage") or usage
        except Exception:
            pass
        if not usage:
            for line in raw_stdout.splitlines():
                try:
                    blob = json.loads(line)
                except Exception:
                    continue
                if isinstance(blob, dict) and isinstance(blob.get("usage"), dict):
                    usage = blob.get("usage") or usage
        if not usage:
            return {
                "token_source": "unavailable",
                "input_tokens": None,
                "output_tokens": None,
                "cache_read_tokens": None,
                "total_tokens": None,
            }
        return {
            "token_source": "cli",
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_read_tokens": (
                usage.get("cache_read_tokens")
                or usage.get("cache_read_input_tokens")
                or usage.get("cached_input_tokens")
            ),
            "total_tokens": usage.get("total_tokens"),
        }

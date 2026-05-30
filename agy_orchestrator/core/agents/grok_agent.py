import asyncio
import json
import logging
import os
import re
import tempfile
from typing import List, Optional

from agy_orchestrator.core.agent import AgentInstance

logger = logging.getLogger(__name__)

# grok-build is the only model surfaced by `grok models` today, and it REJECTS
# the reasoningEffort parameter (API 400: "Model grok-build does not support
# parameter reasoningEffort"). So we never pass --effort/--reasoning-effort for
# it. If a future model supports effort, add it to this set to re-enable.
_EFFORT_CAPABLE_MODELS: set = set()


class GrokAgent(AgentInstance):
    """xAI Grok agentic CLI (`grok`) as a worker.

    Grok is a Claude-Code-style agentic coder: in single-turn headless mode it
    can read/edit files directly. We drive it through `grok -p`/`--prompt-file`:

      grok --prompt-file <tmp> --output-format json \
           --always-approve --permission-mode bypassPermissions [-m MODEL]

    Notes:
      - Prompt goes through a temp file (`--prompt-file`), not an argv string,
        so large diff-feedback prompts can't hit MAX_ARG_STRLEN (128 KiB).
      - `--output-format json` returns a single object
        {"text", "stopReason", "sessionId", "requestId", "thought"}; we return
        "text" and stash sessionId for warm `--resume` reuse (like ClaudeAgent).
      - Effort is intentionally NOT sent for grok-build (it 400s); see above.
      - Web search/fetch are ON by default; `web_search=False` disables them.
    """

    def __init__(self, *args, session_id: Optional[str] = None,
                 web_search: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id: Optional[str] = session_id
        self.web_search: bool = web_search

    @classmethod
    async def get_available_models(cls) -> List[str]:
        """Query `grok models`; fall back to the known default on any error."""
        try:
            process = await asyncio.create_subprocess_exec(
                "grok", "models",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            if process.returncode == 0:
                models = []
                for line in stdout.decode().splitlines():
                    m = re.match(r"\s*\*?\s*([A-Za-z0-9._-]+)", line)
                    # Only the indented bullet list under "Available models:".
                    if m and ("*" in line or line.startswith("  ")) and \
                            "model" not in line.lower() and "logged" not in line.lower():
                        models.append(m.group(1))
                if models:
                    return models
        except Exception:
            pass
        return ["grok-build"]

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        return 100.0

    def _build_cmd(self, prompt_file: str) -> List[str]:
        cmd = [
            "grok",
            "--prompt-file", prompt_file,
            "--output-format", "json",
            "--always-approve",
            "--permission-mode", "bypassPermissions",
        ]
        if self.model and self.model not in ("standard", "n/a"):
            cmd.extend(["-m", self.model])

        # Effort only for models that accept it (none today — grok-build 400s).
        effort = getattr(self, "effort", None)
        if effort and effort not in ("n/a", "high") and self.model in _EFFORT_CAPABLE_MODELS:
            cmd.extend(["--effort", effort])

        if not self.web_search:
            cmd.append("--disable-web-search")

        if self.session_id:
            cmd.extend(["--resume", self.session_id])

        for k, v in self.additional_flags.items():
            cmd.extend([f"--{k}", str(v)])
        return cmd

    def build_command(self, piped_input: Optional[str] = None) -> List[str]:
        # Deliver the prompt via a temp file (--prompt-file) so large diff-feedback
        # prompts can't hit MAX_ARG_STRLEN. The file is removed in _cleanup().
        full_prompt = self._full_prompt(piped_input)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".grok-prompt.txt", delete=False, encoding="utf-8"
        )
        tmp.write(full_prompt)
        tmp.close()
        self._tmp_prompt_path = tmp.name
        logger.info("grok prompt -> %s (%d bytes, session=%s, model=%s)",
                    tmp.name, len(full_prompt), self.session_id or "new", self.model)
        return self._build_cmd(tmp.name)

    def _cleanup(self) -> None:
        path = getattr(self, "_tmp_prompt_path", None)
        if path:
            try:
                os.unlink(path)
            except Exception:
                pass
            self._tmp_prompt_path = None

    def _postprocess(self, raw_stdout: str) -> str:
        """Capture the resumable session id, then return the unwrapped text."""
        sid = self._extract_session_id(raw_stdout)
        if sid:
            if not self.session_id:
                logger.info("grok session established: %s", sid)
            self.session_id = sid
        return self._extract_text(raw_stdout)

    def _full_prompt(self, piped_input: Optional[str]) -> str:
        full = self.prompt
        if piped_input:
            full += f"\n\n[Context]:\n{piped_input}"
        return full

    def filter_stderr(self, stderr: str) -> str:
        """Drop grok's benign tracing noise (e.g. the /tmp inotify warning and
        ANSI-coloured INFO/ERROR trace lines) so a real failure stands out."""
        kept = []
        for line in stderr.splitlines():
            low = line.lower()
            if "failed to watch root recursively" in low:
                continue
            if "shell cwd was reset" in low:
                continue
            kept.append(line)
        return "\n".join(kept)

    def _events_from_stdout_line(self, line: str) -> List[dict]:
        try:
            from dashboard.adapters import AdapterCtx, parse_grok_stream_line
        except Exception:
            return []
        ctx = getattr(self, "_adapter_ctx", None)
        if ctx is None:
            ctx = AdapterCtx()
            self._adapter_ctx = ctx
        return parse_grok_stream_line(line, ctx)

    def _events_from_stdout_complete(self, raw_stdout: str) -> List[dict]:
        text = self._extract_text(raw_stdout).strip()
        if not text:
            return []
        return [{"kind": "message", "text": text, "data": {}}]

    @staticmethod
    def _extract_payload(raw_stdout: str) -> Optional[dict]:
        s = raw_stdout.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            m = re.search(r"\{.*\}", s, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass
        return None

    @staticmethod
    def _extract_text(raw_stdout: str) -> str:
        """Pull "text" from the JSON object; tolerate plain output / extra logs."""
        obj = GrokAgent._extract_payload(raw_stdout)
        if obj and "text" in obj:
            return str(obj["text"])
        return raw_stdout

    @staticmethod
    def _extract_session_id(raw_stdout: str) -> Optional[str]:
        m = re.search(r'"sessionId"\s*:\s*"([^"]+)"', raw_stdout)
        return m.group(1) if m else None

    def _extract_usage(self, raw_stdout: str, raw_stderr: str) -> dict:
        obj = self._extract_payload(raw_stdout)
        usage = obj.get("usage") if isinstance(obj, dict) else None
        if not isinstance(usage, dict):
            return {
                "token_source": "unavailable",
                "input_tokens": None,
                "output_tokens": None,
                "cache_read_tokens": None,
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
        }

import asyncio as _asyncio
import json
import logging
import os
import re
import tempfile
from typing import Optional, List

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
            process = await _asyncio.create_subprocess_exec(
                "grok", "models",
                stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.PIPE,
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
        # Prompt is delivered via a temp file in run_async; satisfy the ABC by
        # pointing --prompt-file at the (yet-unwritten) path is not possible
        # here, so this is only used by callers that don't write the file.
        return self._build_cmd("/dev/stdin")

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

    @staticmethod
    def _extract_text(raw_stdout: str) -> str:
        """Pull "text" from the JSON object; tolerate plain output / extra logs."""
        s = raw_stdout.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "text" in obj:
                return obj["text"]
        except Exception:
            # JSON may be embedded among stray log lines — grab the last {...}.
            m = re.search(r"\{.*\}", s, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    if isinstance(obj, dict) and "text" in obj:
                        return obj["text"]
                except Exception:
                    pass
        return raw_stdout

    @staticmethod
    def _extract_session_id(raw_stdout: str) -> Optional[str]:
        m = re.search(r'"sessionId"\s*:\s*"([^"]+)"', raw_stdout)
        return m.group(1) if m else None

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        full_prompt = self._full_prompt(piped_input)

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".grok-prompt.txt", delete=False, encoding="utf-8"
        )
        try:
            tmp.write(full_prompt)
            tmp.close()
            cmd = self._build_cmd(tmp.name)

            logger.info("Executing grok (prompt=%d bytes, session=%s, model=%s)",
                        len(full_prompt), self.session_id or "new", self.model)

            for attempt in range(1, self.max_retries + 1):
                process = None
                try:
                    process = await _asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=_asyncio.subprocess.PIPE,
                        stderr=_asyncio.subprocess.PIPE,
                    )
                    comm = process.communicate()
                    try:
                        if self.timeout and self.timeout > 0:
                            stdout_bytes, stderr_bytes = await _asyncio.wait_for(comm, self.timeout)
                        else:
                            stdout_bytes, stderr_bytes = await comm
                    except _asyncio.TimeoutError:
                        logger.error("grok subprocess exceeded %.0fs; killing and failing over.", self.timeout)
                        try:
                            process.kill()
                            await process.wait()
                        except Exception:
                            pass
                        self.stderr = f"timed out after {self.timeout:.0f}s"
                        raise RuntimeError(self.stderr)

                    raw_stdout = stdout_bytes.decode()
                    self.stderr = self.filter_stderr(stderr_bytes.decode())
                    self.returncode = process.returncode

                    if self.returncode == 0:
                        sid = self._extract_session_id(raw_stdout)
                        if sid:
                            if not self.session_id:
                                logger.info("grok session established: %s", sid)
                            self.session_id = sid
                        self.stdout = self._extract_text(raw_stdout)
                        return self.stdout

                    logger.warning("grok attempt %d/%d failed (code %d):\n%s",
                                   attempt, self.max_retries, self.returncode, self.stderr[:1000])
                except Exception as e:
                    logger.warning("grok attempt %d/%d exception: %s", attempt, self.max_retries, e)
                    self.stderr = str(e)

                if attempt < self.max_retries:
                    await _asyncio.sleep(2 ** attempt)

            raise RuntimeError(f"GrokAgent failed after {self.max_retries} attempts: {self.stderr}")
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

import asyncio as _asyncio
import json
import logging
import re
from typing import Optional, List

from agy_orchestrator.core.agent import AgentInstance

logger = logging.getLogger(__name__)

MODEL_ALIASES = {
    "standard": "sonnet",
    "opus": "default",
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

    @classmethod
    async def get_available_models(cls) -> List[str]:
        return ["haiku", "sonnet", "opus"]

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        try:
            process = await _asyncio.create_subprocess_exec(
                "claude", "--usage", model,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            if process.returncode == 0:
                pass
        except Exception:
            pass
        return 100.0

    def _build_base_cmd(self) -> List[str]:
        """Build the claude command. Prompt is always passed via stdin."""
        cmd = ["claude", "-p", "-", "--output-format", "json", "--dangerously-skip-permissions"]

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
        """Pull session_id from the JSON stream output."""
        try:
            for line in raw_stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Output may be a JSON array or a stream of objects
                blobs = [line]
                if line.startswith("["):
                    blobs = json.loads(line)
                for blob in blobs:
                    if isinstance(blob, dict) and "session_id" in blob:
                        return blob["session_id"]
        except Exception:
            # Fallback: regex scan
            m = re.search(
                r'"session_id"\s*:\s*"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"',
                raw_stdout
            )
            if m:
                return m.group(1)
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
        return raw_stdout

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        """Execute via stdin, parse JSON output, track session_id for reuse."""
        cmd = self._build_base_cmd()
        full_prompt = self._build_full_prompt(piped_input)
        stdin_data = full_prompt.encode()

        logger.info(
            "Executing agent command (stdin=%d bytes, session=%s, fork=%s)",
            len(stdin_data), self.session_id or "new", self.fork_session
        )

        self._current_process = None
        try:
            for attempt in range(1, self.max_retries + 1):
                try:
                    process = await _asyncio.create_subprocess_exec(
                        *cmd,
                        stdin=_asyncio.subprocess.PIPE,
                        stdout=_asyncio.subprocess.PIPE,
                        stderr=_asyncio.subprocess.PIPE
                    )
                    self._current_process = process
                    try:
                        comm = process.communicate(input=stdin_data)
                        if self.timeout and self.timeout > 0:
                            stdout_bytes, stderr_bytes = await _asyncio.wait_for(comm, self.timeout)
                        else:
                            stdout_bytes, stderr_bytes = await comm
                    except _asyncio.TimeoutError:
                        logger.error("claude subprocess exceeded %.0fs; killing and failing over.", self.timeout)
                        try:
                            process.kill()
                            await process.wait()
                        except Exception:
                            pass
                        self.stderr = f"timed out after {self.timeout:.0f}s"
                        raise RuntimeError(self.stderr)
                    finally:
                        self._current_process = None

                    raw_stdout = stdout_bytes.decode()
                    raw_stderr = stderr_bytes.decode()
                    self.stderr = self.filter_stderr(raw_stderr)
                    self.returncode = process.returncode

                    if self.returncode == 0:
                        # Capture session ID for caller to reuse
                        sid = self._extract_session_id(raw_stdout)
                        if sid:
                            if not self.session_id:
                                logger.info("New session established: %s", sid)
                            self.session_id = sid

                        self.stdout = self._extract_result_text(raw_stdout)
                        return self.stdout

                    logger.warning(
                        "Attempt %d/%d failed (code %d):\nSTDERR: %s\nSTDOUT: %s",
                        attempt, self.max_retries, self.returncode,
                        self.stderr[:1000], raw_stdout[:1000]
                    )

                except Exception as e:
                    logger.warning("Attempt %d/%d exception: %s", attempt, self.max_retries, e)
                    self.stderr = str(e)

                if attempt < self.max_retries:
                    backoff_time = 2 ** attempt
                    logger.info("Retrying in %d seconds...", backoff_time)
                    await _asyncio.sleep(backoff_time)

            logger.error("All %d attempts failed.", self.max_retries)
            raise RuntimeError(f"AgentInstance failed after {self.max_retries} attempts: {self.stderr}")
        finally:
            if self._current_process:
                try:
                    self._current_process.kill()
                except Exception:
                    pass
                self._current_process = None

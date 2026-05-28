import asyncio
import logging
import os
import random
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Substrings (lowercased) in stderr that indicate a usage/quota wall rather than
# a transient error. Such a failure will NOT clear within a few seconds, so the
# base run loop fails fast (no retry/backoff) and lets the fallback layer roll to
# the next provider immediately. Imported by core.agents.fallback_agent too.
USAGE_MARKERS = (
    "usage limit", "rate limit", "rate_limit", "quota", "out of credits",
    "insufficient_quota", "too many requests", "429", "plan limit",
    "exceeded", "balance",
)


def is_usage_wall(stderr: str) -> bool:
    s = (stderr or "").lower()
    return any(marker in s for marker in USAGE_MARKERS)

class AgentInstance(ABC):
    """
    Abstract base class representing a single execution of an AI agent CLI.
    """
    def __init__(
        self,
        prompt: str,
        model: Optional[str] = None,
        additional_flags: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        self.prompt = prompt
        self.model = model
        self.additional_flags = additional_flags or {}
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.stdout = ""
        self.stderr = ""
        self.returncode: Optional[int] = None
        self.max_retries = 3
        # The live child process, tracked so it is always killed on timeout,
        # exception, or task cancellation (a losing ToT/swarm branch) instead of
        # leaking an orphaned worker CLI.
        self._current_process = None
        # Wall-clock ceiling for a single subprocess call. A worker CLI that
        # stalls on a network read (no timeout of its own) would otherwise hang
        # the whole dispatch indefinitely. On timeout we kill it and fail fast so
        # the fallback chain can roll to the next provider. 0/unset disables.
        # Override per run with AGY_TIMEOUT (seconds).
        self.timeout = float(os.environ.get("AGY_TIMEOUT", "2400") or 0)

    @classmethod
    @abstractmethod
    async def get_available_models(cls) -> List[str]:
        """
        Dynamically query the CLI for available models.
        """
        pass

    @classmethod
    @abstractmethod
    async def get_model_usage(cls, model: str) -> float:
        """
        Dynamically query the CLI for remaining usage percentage of a model.
        Returns a float between 0.0 and 100.0.
        """
        pass

    @abstractmethod
    def build_command(self, piped_input: Optional[str] = None) -> List[str]:
        """
        Build the command line arguments for the agent CLI execution.
        """
        pass

    def filter_stderr(self, stderr: str) -> str:
        """
        Filter out expected network errors or verbosely piped logs.
        Can be overridden by subclasses.
        """
        return stderr

    # --- template-method hooks (override per agent) ----------------------------
    # These let every agent share the one hardened run loop below (retry/timeout/
    # backoff/kill/cancel) while customizing only how the prompt is delivered and
    # how the raw stdout is interpreted.

    def _stdin_bytes(self, piped_input: Optional[str] = None) -> Optional[bytes]:
        """Bytes to feed the child on stdin, or None to pass nothing on stdin.

        Default: None (prompt is carried in argv / a file by build_command).
        Override (e.g. ClaudeAgent) to deliver the prompt via stdin and dodge
        ARG_MAX limits on large prompts."""
        return None

    def _postprocess(self, raw_stdout: str) -> str:
        """Transform decoded stdout into the final result string.

        Default: return it unchanged. Override (claude/grok) to unwrap a JSON
        envelope and capture a resumable session id for warm-cache reuse."""
        return raw_stdout

    def _cleanup(self) -> None:
        """Release any per-call resources (e.g. a temp prompt file). No-op by default."""
        pass

    @staticmethod
    async def _stream_communicate(process, stdin_bytes: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Drain stdout+stderr concurrently, echoing stderr live to our stderr.

        Returns the full (stdout_bytes, stderr_bytes) so callers behave exactly
        as with ``process.communicate()`` — the only difference is that the
        child's stderr is mirrored to our own stderr line-by-line as it arrives,
        which makes a long build's progress tailable instead of buffered. Feeds
        ``stdin_bytes`` (if any) concurrently so stdin-delivered prompts also stream.
        """

        async def _feed() -> None:
            if stdin_bytes is None or process.stdin is None:
                return
            try:
                process.stdin.write(stdin_bytes)
                await process.stdin.drain()
            except Exception:
                pass
            finally:
                try:
                    process.stdin.close()
                except Exception:
                    pass

        async def _drain(reader, chunks, echo: bool) -> None:
            if reader is None:
                return
            while True:
                line = await reader.readline()
                if not line:
                    break
                chunks.append(line)
                if echo:
                    sys.stderr.buffer.write(line)
                    sys.stderr.buffer.flush()

        out_chunks: List[bytes] = []
        err_chunks: List[bytes] = []
        await asyncio.gather(
            _feed(),
            _drain(process.stdout, out_chunks, echo=False),
            _drain(process.stderr, err_chunks, echo=True),
        )
        await process.wait()
        return b"".join(out_chunks), b"".join(err_chunks)

    async def _kill_current(self) -> None:
        """Kill and reap the tracked child if it is still running. Best-effort,
        bounded so a wedged process can't hang the reaping itself."""
        proc = self._current_process
        self._current_process = None
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.kill()
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), 5)
        except Exception:
            pass

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        """Run the agent CLI, retrying only on transient errors.

        Fails fast (no retry, no backoff) on a timeout or a usage/quota wall —
        neither recovers within a few seconds, so retrying just burns time before
        the fallback layer can advance to the next provider. Transient errors get
        a short jittered backoff. The child is always killed on exit (timeout,
        exception, or cancellation) so no orphaned worker CLI leaks.
        """
        cmd = self.build_command(piped_input)
        stdin_bytes = self._stdin_bytes(piped_input)
        logger.info("Executing agent command: %s", cmd[0] if cmd else "?")

        try:
            for attempt in range(1, self.max_retries + 1):
                try:
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else None,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    self._current_process = process

                    # Live mode: drain both pipes line-by-line, echoing the child's
                    # stderr as it arrives (tailable progress) while accumulating
                    # the full streams. (codex writes work to stderr live; claude
                    # --output-format json emits only at the end.) Stream mode does
                    # not support stdin, so agents that feed stdin use communicate().
                    if os.environ.get("AGY_STREAM"):
                        comm = self._stream_communicate(process, stdin_bytes)
                    else:
                        comm = process.communicate(input=stdin_bytes)

                    try:
                        if self.timeout and self.timeout > 0:
                            stdout_bytes, stderr_bytes = await asyncio.wait_for(comm, self.timeout)
                        else:
                            stdout_bytes, stderr_bytes = await comm
                    except asyncio.TimeoutError:
                        logger.error("Agent subprocess exceeded %.0fs; killing and failing over.",
                                     self.timeout)
                        await self._kill_current()
                        self.stderr = f"timed out after {self.timeout:.0f}s"
                        raise RuntimeError(self.stderr)  # fail fast, no retry

                    raw_stdout = stdout_bytes.decode()
                    self.stderr = self.filter_stderr(stderr_bytes.decode())
                    self.returncode = process.returncode
                    self._current_process = None

                    if self.returncode == 0:
                        self.stdout = self._postprocess(raw_stdout)
                        return self.stdout

                    logger.warning("Attempt %d/%d failed with code %s:\n%s",
                                   attempt, self.max_retries, self.returncode, self.stderr)
                    if is_usage_wall(self.stderr):
                        # A quota wall won't clear in seconds — let the caller fail over now.
                        raise RuntimeError(f"usage wall: {self.stderr[:200]}")

                except asyncio.CancelledError:
                    await self._kill_current()
                    raise
                except RuntimeError:
                    raise  # timeout / usage wall: fail fast
                except Exception as e:
                    logger.warning("Attempt %d/%d encountered an exception: %s",
                                   attempt, self.max_retries, e)
                    self.stderr = str(e)

                if attempt < self.max_retries:
                    backoff_time = min(8.0, 2 ** attempt) * (0.5 + random.random())
                    logger.info("Retrying in %.1f seconds...", backoff_time)
                    await asyncio.sleep(backoff_time)

            logger.error("All %d attempts failed.", self.max_retries)
            raise RuntimeError(f"AgentInstance failed after {self.max_retries} attempts: {self.stderr}")
        finally:
            await self._kill_current()
            self._cleanup()

    def run(self, piped_input: Optional[str] = None) -> str:
        """Synchronous wrapper for run_async."""
        return asyncio.run(self.run_async(piped_input))

import asyncio
import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

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

    @staticmethod
    async def _stream_communicate(process) -> Tuple[bytes, bytes]:
        """Drain stdout+stderr concurrently, echoing stderr live to our stderr.

        Returns the full (stdout_bytes, stderr_bytes) so callers behave exactly
        as with ``process.communicate()`` — the only difference is that the
        child's stderr is mirrored to our own stderr line-by-line as it arrives,
        which makes a long build's progress tailable instead of buffered.
        """

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
            _drain(process.stdout, out_chunks, echo=False),
            _drain(process.stderr, err_chunks, echo=True),
        )
        await process.wait()
        return b"".join(out_chunks), b"".join(err_chunks)

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        """Executes the instance asynchronously with exponential backoff for network smoothing."""
        cmd = self.build_command(piped_input)
        
        logger.info(f"Executing agent command")
        
        for attempt in range(1, self.max_retries + 1):
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                if os.environ.get("AGY_STREAM"):
                    # Live mode: drain both pipes line-by-line, echoing the child's
                    # stderr to our own stderr as it arrives (tailable progress),
                    # while still accumulating the full streams for the caller.
                    # (codex writes its work to stderr live; claude --output-format
                    # json still emits only at the end, so prefer codex in this mode.)
                    comm = self._stream_communicate(process)
                else:
                    comm = process.communicate()

                try:
                    if self.timeout and self.timeout > 0:
                        stdout_bytes, stderr_bytes = await asyncio.wait_for(comm, self.timeout)
                    else:
                        stdout_bytes, stderr_bytes = await comm
                except asyncio.TimeoutError:
                    # Hung worker: kill the subprocess and fail fast (no retry on
                    # the same stalled provider) so the fallback chain advances.
                    logger.error("Agent subprocess exceeded %.0fs; killing and failing over.", self.timeout)
                    try:
                        process.kill()
                        await process.wait()
                    except Exception:
                        pass
                    self.stderr = f"timed out after {self.timeout:.0f}s"
                    raise RuntimeError(self.stderr)

                self.stdout = stdout_bytes.decode()
                raw_stderr = stderr_bytes.decode()
                self.stderr = self.filter_stderr(raw_stderr)
                self.returncode = process.returncode
                
                if self.returncode == 0:
                    return self.stdout
                
                logger.warning(f"Attempt {attempt}/{self.max_retries} failed with code {self.returncode}:\n{self.stderr}")
                
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{self.max_retries} encountered an exception: {e}")
                self.stderr = str(e)
            
            if attempt < self.max_retries:
                backoff_time = 2 ** attempt
                logger.info(f"Retrying in {backoff_time} seconds...")
                await asyncio.sleep(backoff_time)
                
        logger.error(f"All {self.max_retries} attempts failed.")
        raise RuntimeError(f"AgentInstance failed after {self.max_retries} attempts: {self.stderr}")

    def run(self, piped_input: Optional[str] = None) -> str:
        """Synchronous wrapper for run_async."""
        return asyncio.run(self.run_async(piped_input))

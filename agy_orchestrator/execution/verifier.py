import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

@dataclass
class VerifierResult:
    ok: bool
    message: str = ""
    returncode: int = 0
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_ms: int = 0
    timeout: bool = False
    cmd: str = ""
    error_hash: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok

    def __iter__(self):
        # Back-compat unpacking: success, error_msg = await verifier.verify(...)
        yield self.ok
        yield self.message


class QualityVerifier:
    """
    Executes programmatic tests to verify the quality of generated output.
    Enforces the 100% quality guarantee by checking build scripts, linters, or unit tests.
    """
    def __init__(self, test_commands: List[str], timeout: float = None):
        self.test_commands = test_commands
        # Wall-clock ceiling per test command. An operator-supplied --test-cmd that
        # hangs (a server that never exits, an interactive prompt) would otherwise
        # hang the whole workflow forever. Defaults to AGY_TEST_TIMEOUT or 600s.
        if timeout is None:
            timeout = float(os.environ.get("AGY_TEST_TIMEOUT", "600") or 0)
        self.timeout = timeout

    async def verify(self, working_directory: str) -> VerifierResult:
        """
        Runs the configured test commands in the specified directory.

        Returns:
            VerifierResult: structured verification outcome.
        """
        if not self.test_commands:
            return VerifierResult(
                ok=True,
                message="No verification commands configured",
                returncode=0,
                duration_ms=0,
            )

        total_duration_ms = 0
        for cmd in self.test_commands:
            logger.info(f"Running verification: {cmd} in {working_directory}")
            start = time.monotonic()
            process = await asyncio.create_subprocess_shell(
                cmd,
                cwd=working_directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout = b""
            stderr = b""
            timed_out = False

            try:
                comm = process.communicate()
                if self.timeout and self.timeout > 0:
                    stdout, stderr = await asyncio.wait_for(comm, self.timeout)
                else:
                    stdout, stderr = await comm
            except asyncio.TimeoutError:
                logger.warning("Verification command exceeded %.0fs; killing: %s", self.timeout, cmd)
                timed_out = True
                try:
                    process.kill()
                    await asyncio.wait_for(process.wait(), 5)
                except Exception:
                    pass
            duration_ms = round((time.monotonic() - start) * 1000)
            total_duration_ms += duration_ms

            stdout_tail = stdout.decode(errors="replace")[-2000:] if stdout else ""
            stderr_tail = stderr.decode(errors="replace")[-2000:] if stderr else ""

            if timed_out:
                result = VerifierResult(
                    ok=False,
                    message=f"Verification command timed out after {self.timeout:.0f}s: {cmd}",
                    returncode=process.returncode if process.returncode is not None else -1,
                    stdout_tail=stdout_tail,
                    stderr_tail=stderr_tail,
                    duration_ms=duration_ms,
                    timeout=True,
                    cmd=cmd,
                )
                if result.stderr_tail:
                    result.error_hash = hashlib.sha256(
                        result.stderr_tail.encode()
                    ).hexdigest()[:16]
                return result

            if process.returncode != 0:
                result = VerifierResult(
                    ok=False,
                    message=f"Command failed with exit code {process.returncode}: {cmd}",
                    returncode=process.returncode,
                    stdout_tail=stdout_tail,
                    stderr_tail=stderr_tail,
                    duration_ms=duration_ms,
                    timeout=False,
                    cmd=cmd,
                )
                if result.stderr_tail:
                    result.error_hash = hashlib.sha256(
                        result.stderr_tail.encode()
                    ).hexdigest()[:16]
                logger.warning("Verification failed: %s", result.message)
                return result

        logger.info("All verifications passed successfully.")
        final_cmd = "<multi>" if len(self.test_commands) > 1 else self.test_commands[0]
        return VerifierResult(
            ok=True,
            message="All tests passed",
            returncode=0,
            duration_ms=total_duration_ms,
            cmd=final_cmd,
        )

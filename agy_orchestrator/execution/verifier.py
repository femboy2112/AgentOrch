import asyncio
import logging
import os
from typing import List, Tuple

logger = logging.getLogger(__name__)

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

    async def verify(self, working_directory: str) -> Tuple[bool, str]:
        """
        Runs the configured test commands in the specified directory.

        Returns:
            Tuple[bool, str]: (Success boolean, Error details string)
        """
        if not self.test_commands:
            return True, "No verification commands configured."

        for cmd in self.test_commands:
            logger.info(f"Running verification: {cmd} in {working_directory}")
            process = await asyncio.create_subprocess_shell(
                cmd,
                cwd=working_directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                comm = process.communicate()
                if self.timeout and self.timeout > 0:
                    stdout, stderr = await asyncio.wait_for(comm, self.timeout)
                else:
                    stdout, stderr = await comm
            except asyncio.TimeoutError:
                logger.warning("Verification command exceeded %.0fs; killing: %s", self.timeout, cmd)
                try:
                    process.kill()
                    await asyncio.wait_for(process.wait(), 5)
                except Exception:
                    pass
                return False, f"Verification command timed out after {self.timeout:.0f}s: {cmd}"

            if process.returncode != 0:
                error_msg = f"Command failed with exit code {process.returncode}: {cmd}\n"
                error_msg += f"-- STDOUT --\n{stdout.decode()}\n"
                error_msg += f"-- STDERR --\n{stderr.decode()}"

                logger.warning(f"Verification failed:\n{error_msg}")
                return False, error_msg

        logger.info("All verifications passed successfully.")
        return True, "All tests passed successfully."

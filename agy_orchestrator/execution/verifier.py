import asyncio
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class QualityVerifier:
    """
    Executes programmatic tests to verify the quality of generated output.
    Enforces the 100% quality guarantee by checking build scripts, linters, or unit tests.
    """
    def __init__(self, test_commands: List[str]):
        self.test_commands = test_commands
        
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
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = f"Command failed with exit code {process.returncode}: {cmd}\n"
                error_msg += f"-- STDOUT --\n{stdout.decode()}\n"
                error_msg += f"-- STDERR --\n{stderr.decode()}"
                
                logger.warning(f"Verification failed:\n{error_msg}")
                return False, error_msg
                
        logger.info("All verifications passed successfully.")
        return True, "All tests passed successfully."

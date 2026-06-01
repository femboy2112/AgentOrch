import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
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
    # #39: the verifier was killed by its own resource cap (OOM in its cgroup
    # scope), NOT a genuine test failure. Lets callers report "verifier exceeded
    # resource budget" distinctly and avoid misreading it as a code failure.
    resource_exceeded: bool = False

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
    def __init__(self, test_commands: List[str], timeout: float = None,
                 mem_max: Optional[str] = None):
        self.test_commands = test_commands
        # Wall-clock ceiling per test command. An operator-supplied --test-cmd that
        # hangs (a server that never exits, an interactive prompt) would otherwise
        # hang the whole workflow forever. Defaults to AGY_TEST_TIMEOUT or 600s.
        if timeout is None:
            timeout = float(os.environ.get("AGY_TEST_TIMEOUT", "600") or 0)
        self.timeout = timeout
        # #39: opt-in memory cap on the verifier subprocess. A heavy --test-cmd
        # (pytest -n auto + mypy + ruff ≈ several GB) can OOM a small host and,
        # because the verifier is a child of the orchestrator, take the harness
        # down with it. When set (e.g. "3G"), each test command runs inside its
        # OWN transient systemd scope with MemoryMax/MemorySwapMax so a spike is
        # OOM-killed in that scope and the orchestrator survives to record it.
        self.mem_max = mem_max or os.environ.get("AGY_VERIFIER_MEM_MAX") or None
        # MemorySwapMax: default 0 (hard ceiling, no swap-thrash → fail fast
        # instead of freezing the host). Override via AGY_VERIFIER_SWAP_MAX.
        self.swap_max = os.environ.get("AGY_VERIFIER_SWAP_MAX", "0")
        # Probe once: only cap if a usable user systemd scope manager exists.
        # Otherwise degrade to uncapped (opt-in feature must not break non-systemd
        # hosts), warning once so the operator knows the cap didn't take.
        self._cap_active = bool(self.mem_max) and self._systemd_scope_available()
        if self.mem_max and not self._cap_active:
            logger.warning(
                "Verifier memory cap requested (%s) but `systemd-run --user --scope` "
                "is unavailable; running the verifier UNCAPPED. Wrap the whole "
                "`harness do` in an external scope if you need a hard cap here.",
                self.mem_max,
            )

    @staticmethod
    def _systemd_scope_available() -> bool:
        if not shutil.which("systemd-run"):
            return False
        try:
            probe = subprocess.run(
                ["systemd-run", "--user", "--scope", "--quiet", "--collect", "true"],
                capture_output=True, timeout=15,
            )
            return probe.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def _exec_argv(self, cmd: str) -> Optional[List[str]]:
        """systemd-run argv that runs `cmd` in a memory-capped transient scope,
        or None when the cap isn't active (caller falls back to a plain shell)."""
        if not self._cap_active:
            return None
        return [
            "systemd-run", "--user", "--scope", "--quiet", "--collect",
            "-p", f"MemoryMax={self.mem_max}",
            "-p", f"MemorySwapMax={self.swap_max}",
            "/bin/sh", "-c", cmd,
        ]

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
            logger.info(
                "Running verification: %s in %s%s",
                cmd, working_directory,
                f" (mem cap {self.mem_max})" if self._cap_active else "",
            )
            start = time.monotonic()
            argv = self._exec_argv(cmd)
            if argv is not None:
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=working_directory,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    cwd=working_directory,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
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
                # #39: under a memory cap, a SIGKILL (rc 137 / -9) almost always
                # means the cgroup OOM-killer fired — the verifier exceeded its
                # budget, NOT a genuine test failure. Classify it distinctly so a
                # downstream consumer doesn't misread it as a code failure.
                resource_exceeded = (
                    self._cap_active and process.returncode in (137, -9, 9)
                )
                if resource_exceeded:
                    message = (
                        f"Verifier exceeded its resource budget (memory cap "
                        f"{self.mem_max}); OOM-killed in its own scope: {cmd}"
                    )
                else:
                    message = f"Command failed with exit code {process.returncode}: {cmd}"
                result = VerifierResult(
                    ok=False,
                    message=message,
                    returncode=process.returncode,
                    stdout_tail=stdout_tail,
                    stderr_tail=stderr_tail,
                    duration_ms=duration_ms,
                    timeout=False,
                    cmd=cmd,
                    resource_exceeded=resource_exceeded,
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

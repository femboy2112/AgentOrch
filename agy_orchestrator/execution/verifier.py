import asyncio
import hashlib
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

# Shared with the worker-child resource bounds (issue #48): ONE thread-pin var
# list + ONE opt-out, so the verifier subprocess (#54) and the worker child env
# stay symmetric. Imported (not re-declared) so a future change to either lands
# in both. agent.py does not import verifier, so this import introduces no cycle.
from agy_orchestrator.core.agent import (
    _THREAD_PIN_VARS,
    _resource_bound_disabled,
    looks_like_infra,
)

logger = logging.getLogger(__name__)

# Matches the failing-test identity lines pytest -q writes to STDOUT, both in the
# live "FAILED test::name - ..." stream and the "=== short test summary ===" block
# ("FAILED test::name" / "ERROR test::name"). Used to surface a thrashing step's
# failing tests in the event stream (issue #60). Anchored at line start; the
# nodeid is the first whitespace-delimited token after the keyword.
_PYTEST_FAILED_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)


def _safe_getpgid(process) -> Optional[int]:
    """Process-group id of a freshly-spawned child (== its pid when it was started
    with ``start_new_session=True``), or None if it already exited / pgid is
    unavailable. Captured right after spawn so the group can be signalled later
    even once the direct child has been reaped."""
    try:
        return os.getpgid(process.pid)
    except Exception:
        return None


def _kill_process_group(pgid: Optional[int], *, fallback_process=None) -> None:
    """SIGKILL an entire process group, best-effort (#61).

    On a plain (non-scoped) verifier spawn the test process is its own group
    leader, so signalling the group reaps the WHOLE pool — the multiprocessing /
    pytest-xdist workers that block on ``sys.stdin.readline()`` and which a bare
    ``process.kill()`` (direct child only) would orphan to PID 1 forever. Safe to
    call after normal completion too: the leader is already gone, so this just
    sweeps any leaked daemon descendants (``ProcessLookupError`` => nothing left).

    Under the #39 memory-cap path the command runs inside a transient systemd
    ``--scope``; those descendants live in the scope's cgroup (child of the user
    manager), not our group, so the scope's own ``--collect`` GC reaps them — here
    we still kill the ``systemd-run`` client group, which is strictly better than
    the previous leak-everything behaviour."""
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return  # group already gone — nothing to reap
        except Exception:
            pass
    if fallback_process is not None:
        try:
            fallback_process.kill()
        except Exception:
            pass


def _parse_failed_tests(stdout_tail: str, limit: int = 25) -> List[str]:
    """Extract failing-test nodeids from a pytest -q stdout tail (issue #60).

    pytest writes "FAILED test::name" to STDOUT (not stderr), so the failing-test
    identity is otherwise persisted nowhere. De-duplicated, order-preserving, and
    bounded so a suite that fails hundreds of tests can't unbound the event."""
    if not stdout_tail:
        return []
    seen: List[str] = []
    for nodeid in _PYTEST_FAILED_RE.findall(stdout_tail):
        if nodeid not in seen:
            seen.append(nodeid)
        if len(seen) >= limit:
            break
    return seen


def _full_log_cap() -> int:
    """Max chars of full verifier output persisted to verify_step<N>_iter<M>.full.log
    (#64). Generous (1 MiB default) so pytest tracebacks survive; head+tail is kept
    when exceeded. Tunable via AGY_VERIFY_FULL_LOG_MAX (0 = unbounded)."""
    try:
        return int(os.environ.get("AGY_VERIFY_FULL_LOG_MAX", str(1024 * 1024)) or 0)
    except Exception:
        return 1024 * 1024


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
    # #58: a NON-timeout, normal-returncode failure whose stderr matches an
    # infra-class marker (build tool missing, half-installed venv, disk full,
    # permission flip, conftest import error) — a broken HOST env, not a code
    # defect. Heuristic, so the adversarial loop treats it as a SOFT signal (one
    # regenerate then bail) rather than ground truth. Only ever set on a genuine
    # failure (ok=False, not timeout/resource_exceeded); False otherwise so a
    # consumer that never inspects it sees byte-identical behaviour.
    infra_suspected: bool = False

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
        # The most recent VerifierResult, exposed so the harness can surface the
        # FINAL converged verifier's wall-clock + timeout margin (issue #55)
        # without re-running the suite. None until verify() has run once.
        self.last_result: Optional[VerifierResult] = None
        # Optional observability sinks (issue #60), all opt-in (set by the harness
        # after construction). When unset the verifier behaves byte-identically:
        #   * run_dir         — directory to persist a per-iteration verify log to.
        #   * event_callback  — bus publisher; failing-test nodeids are emitted so a
        #                       thrashing step is visible in events.jsonl.
        #   * step_index      — the build step this verifier gates (artifact naming).
        # The per-step iteration counter advances on every FAILED verify so repeated
        # failures within one step land in distinct artifacts.
        self.run_dir: Optional[str] = None
        self.event_callback: Optional[Callable[[dict], None]] = None
        self.step_index: int = 0
        self._iter_index: int = 0

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

    def _verifier_env(self) -> Optional[dict]:
        """Child env for the verifier subprocess with BLAS/xdist thread pins (#54).

        Symmetric gap from #48: ``apply_worker_resource_bounds`` pins the thread
        pools ONLY in the WORKER child env, but the orchestrator's OWN verifier ran
        the raw ``--test-cmd`` with no env injection — inheriting the parent env, so
        a ``--test-cmd`` running its own ``pytest -n auto`` plus default BLAS pools
        nested-oversubscribed the host. Pin the SAME set here, guarded by the SAME
        ``AGY_WORKER_RESOURCE_BOUND=0`` opt-out. Each var is only set when ABSENT, so
        an operator/parent value is respected. Returns None (inherit parent env)
        when bounding is disabled, keeping the historical behaviour byte-identical."""
        if _resource_bound_disabled():
            return None
        env = dict(os.environ)
        for var in _THREAD_PIN_VARS:
            env.setdefault(var, "1")
        return env

    def _clamp_xdist(self, cmd: str) -> str:
        """Clamp any verifier ``-n K`` xdist worker count to the host core count (#55b).

        A ``--test-cmd ... -n 8`` on a 4-core box requests more xdist workers than
        cores, oversubscribing every numeric library underneath. Rewrite ``-n K`` to
        ``-n min(K, cpu_count())`` and log the clamp. ``-n auto``/``-n logical`` (and a
        bound disabled via the resource opt-out) are left untouched: ``auto`` already
        tracks the core count, and the opt-out means the operator wants raw control."""
        if _resource_bound_disabled():
            return cmd
        cpu = os.cpu_count() or 1

        def _sub(m: "re.Match") -> str:
            k = int(m.group("k"))
            if k <= cpu:
                return m.group(0)
            logger.warning(
                "Verifier -n %d exceeds host core count %d; clamping to -n %d "
                "to avoid oversubscription.", k, cpu, cpu,
            )
            return f"{m.group('flag')}{m.group('sep')}{cpu}"

        # Match `-n 8` and `-n8` and `--numprocesses 8`, but NOT `-n auto`/`logical`.
        pattern = re.compile(
            r"(?P<flag>--numprocesses|-n)(?P<sep>=|\s+|)(?P<k>\d+)"
        )
        return pattern.sub(_sub, cmd)

    async def verify(self, working_directory: str) -> VerifierResult:
        """
        Runs the configured test commands in the specified directory.

        Returns:
            VerifierResult: structured verification outcome.
        """
        if not self.test_commands:
            return self._finalize(VerifierResult(
                ok=True,
                message="No verification commands configured",
                returncode=0,
                duration_ms=0,
            ))

        # Thread-pin child env (#54), shared with the worker-child bounds (#48).
        env = self._verifier_env()

        total_duration_ms = 0
        for cmd in self.test_commands:
            # Clamp any -n K xdist worker count to the host core count (#55b).
            cmd = self._clamp_xdist(cmd)
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
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    # #61: own session/process group so a timeout (or any teardown)
                    # can SIGKILL the WHOLE group — reaping forked pool workers that
                    # a direct-child kill would orphan to PID 1.
                    start_new_session=True,
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    cwd=working_directory,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,  # #61: see above
                )
            # #61: capture the group id now, while the child is alive, so a timeout
            # kill (or post-completion sweep) can signal the whole group even after
            # the direct child has been reaped.
            pgid = _safe_getpgid(process)
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
                    # #61: kill the whole group, not just the direct child, so
                    # forked pool workers don't outlive the timeout as PID-1 orphans.
                    _kill_process_group(pgid, fallback_process=process)
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
                return self._finalize(result)

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
                # #58: a non-timeout, non-OOM failure whose stderr matches an
                # infra-class marker looks like a broken HOST env (missing build
                # tool, half-installed venv, full disk, permission flip, conftest
                # import error), not a code defect. Flag it so the adversarial loop
                # surfaces the real cause and bails after one regenerate instead of
                # regenerating code against a problem the code can't fix. Heuristic,
                # so only a SOFT signal — never override a resource_exceeded verdict.
                infra_suspected = not resource_exceeded and looks_like_infra(stderr_tail)
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
                    infra_suspected=infra_suspected,
                )
                if result.stderr_tail:
                    result.error_hash = hashlib.sha256(
                        result.stderr_tail.encode()
                    ).hexdigest()[:16]
                # #60: a non-timeout failure used to log ONLY result.message ("Command
                # failed with exit code N: cmd"). pytest -q writes the failing-test
                # identity ("FAILED test::name", "=== short test summary ===") to
                # STDOUT, so without surfacing stdout_tail the actual failure was
                # persisted nowhere. Log a bounded stdout tail (+ stderr tail when
                # non-empty), persist the full tails to a per-iteration artifact, and
                # emit the parsed failing-test nodeids into the event stream.
                self._report_failure(result, stdout_bytes=stdout or b"",
                                     stderr_bytes=stderr or b"")
                # #61: a non-zero exit can still leave forked pool workers blocked
                # on stdin — sweep the group before returning.
                _kill_process_group(pgid)
                return self._finalize(result)

            # #61: clean (rc 0) completion can still leak daemon pool workers on a
            # buggy test suite — sweep the group at the end of each iteration.
            _kill_process_group(pgid)

        logger.info("All verifications passed successfully.")
        final_cmd = "<multi>" if len(self.test_commands) > 1 else self.test_commands[0]
        return self._finalize(VerifierResult(
            ok=True,
            message="All tests passed",
            returncode=0,
            duration_ms=total_duration_ms,
            cmd=final_cmd,
        ))

    def _finalize(self, result: VerifierResult) -> VerifierResult:
        """Record the result for FINAL-verifier telemetry (#55) and return it."""
        self.last_result = result
        return result

    @staticmethod
    def _stdout_log_tail(stdout_tail: str, max_lines: int = 25,
                         max_chars: int = 1500) -> str:
        """Last ~25 lines / ~1500 chars of a pytest stdout tail — the region that
        holds the "=== short test summary ===" / "FAILED ..." block (#60)."""
        if not stdout_tail:
            return ""
        tail = "\n".join(stdout_tail.splitlines()[-max_lines:])
        return tail[-max_chars:]

    @staticmethod
    def _head_tail(text: str, max_chars: int) -> str:
        """Keep both the HEAD and TAIL of ``text`` within ``max_chars`` (#64).

        pytest writes per-failure tracebacks BEFORE the short summary, so a pure
        tail cut loses the assertion detail. When the output exceeds the budget,
        retain the first and last halves (tracebacks + summary) with an elision
        marker between; ``max_chars <= 0`` means unbounded."""
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        half = max_chars // 2
        elided = len(text) - 2 * half
        return (f"{text[:half]}\n"
                f"... [{elided} bytes elided — head+tail kept] ...\n"
                f"{text[-half:]}")

    def _report_failure(self, result: VerifierResult, *,
                        stdout_bytes: bytes = b"", stderr_bytes: bytes = b"") -> None:
        """Surface a non-timeout verifier failure (#60): WARNING-log the stdout tail,
        persist a quick-glance artifact AND the FULL output (#64), and emit the
        parsed failing-test nodeids into the event stream. All best-effort/bounded.

        ``stdout_bytes``/``stderr_bytes`` are the FULL captured streams (the result
        only carries a 2 KB tail, which drops pytest tracebacks — the #64 gap)."""
        # Advance the per-step iteration counter so repeated failures within one
        # build step land in distinct artifacts (verify_step<N>_iter<M>.log).
        self._iter_index += 1
        # Full decoded streams (fall back to the result tails if not supplied).
        full_out = stdout_bytes.decode(errors="replace") if stdout_bytes else result.stdout_tail
        full_err = stderr_bytes.decode(errors="replace") if stderr_bytes else result.stderr_tail

        out_tail = self._stdout_log_tail(result.stdout_tail)
        if out_tail:
            logger.warning(
                "Verification failed: %s\n--- captured stdout (tail) ---\n%s",
                result.message, out_tail,
            )
        else:
            logger.warning("Verification failed: %s", result.message)
        if result.stderr_tail.strip():
            logger.warning(
                "--- captured stderr (tail) ---\n%s",
                result.stderr_tail[-1500:],
            )

        # (2) Persist a per-iteration artifact so post-hoc diagnosis needs no
        # re-run. Opt-in on run_dir being set. TWO files (#64):
        #   * verify_step<N>_iter<M>.log      — the 2 KB quick-glance tails;
        #   * verify_step<N>_iter<M>.full.log — the FULL output (head+tail bounded
        #     by AGY_VERIFY_FULL_LOG_MAX) so the per-failure pytest TRACEBACKS,
        #     which precede the summary and were dropped by the tail cut, survive.
        if self.run_dir:
            from pathlib import Path
            base = Path(self.run_dir)
            stem = f"verify_step{self.step_index}_iter{self._iter_index}"
            header = (
                f"$ {result.cmd}\n"
                f"exit code: {result.returncode}\n"
                f"duration_ms: {result.duration_ms}\n"
                f"message: {result.message}\n"
            )
            try:
                (base / f"{stem}.log").write_text(
                    f"{header}=== STDOUT (tail) ===\n{result.stdout_tail}\n"
                    f"=== STDERR (tail) ===\n{result.stderr_tail}\n",
                    encoding="utf-8",
                )
            except Exception as exc:  # never let observability fail a run
                logger.debug("Could not persist verify artifact: %s", exc)
            try:
                cap = _full_log_cap()
                (base / f"{stem}.full.log").write_text(
                    f"{header}=== STDOUT (full) ===\n{self._head_tail(full_out, cap)}\n"
                    f"=== STDERR (full) ===\n{self._head_tail(full_err, cap)}\n",
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.debug("Could not persist full verify artifact: %s", exc)

        # (3) Parse "FAILED " lines from the FULL stdout (not just the 2 KB tail,
        # which could miss failing tests listed earlier) and emit the names so a
        # thrashing step is visible in events.jsonl.
        failed = _parse_failed_tests(full_out)
        if failed and self.event_callback is not None:
            try:
                self.event_callback({
                    "kind": "lifecycle",
                    "data": {
                        "event": "verifier_failed",
                        "detail": {
                            "step": self.step_index,
                            "iteration": self._iter_index,
                            "returncode": result.returncode,
                            "failed_tests": failed,
                        },
                    },
                })
            except Exception:
                pass

import asyncio
import logging
import os
import random
import signal
import sys
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Watchdog trip reasons exposed via AgentInstance._watchdog_reason after run_async
# returns or raises. FallbackAgent reads these to choose rule-based re-route targets.
WATCHDOG_VERBOSE = "verbose"   # output bytes > per-config budget (rambling/looping)
WATCHDOG_STALLED = "stalled"   # no output for > stall_seconds and nothing produced
WATCHDOG_MARKER = "[watchdog:"  # carried in self.stderr for downstream pattern matching

# Default absolute-timeout multiple of `timeout` when AGY_ABSOLUTE_TIMEOUT is
# unset: a streaming worker that keeps emitting output can extend its idle
# deadline up to this many `timeout` windows before the hard wall-clock cap
# fires. Bounds a worker that streams forever while still letting a legitimately
# long slow-gate turn run several idle-windows past the base ceiling. (Issue #28)
ABSOLUTE_TIMEOUT_FACTOR = 4.0

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
        #
        # In streaming mode `timeout` is reinterpreted as the IDLE ceiling: the
        # deadline is extended as long as the child keeps emitting output, so a
        # legitimately long turn on a slow-test-gate repo (which streams edit/
        # verify traces the whole time) is no longer discarded mid-progress. The
        # extension is hard-capped by `absolute_timeout` below, and a full
        # `timeout` window with NO new output is still treated as a kill. The
        # non-streaming path keeps the flat wall-clock semantics. See issue #28.
        self.timeout = float(os.environ.get("AGY_TIMEOUT", "2400") or 0)
        # Absolute wall-clock ceiling for a single streaming call — the hard cap
        # on liveness-based extension. 0/unset => default to ABSOLUTE_TIMEOUT_FACTOR
        # x timeout. Override with AGY_ABSOLUTE_TIMEOUT (seconds).
        self.absolute_timeout = float(os.environ.get("AGY_ABSOLUTE_TIMEOUT", "0") or 0)
        # Last time the child emitted output on either stream (monotonic). Shared
        # between the stream drain, the watchdog, and the liveness-aware wait.
        self._last_progress: float = 0.0

        # Streaming watchdog budgets (opt-in; 0/unset disables that signal).
        # max_output_bytes: kill if the child has emitted more than this on stdout.
        # stall_seconds: kill if no output has been observed for this many seconds.
        # Per-call callers can also set self.max_output_bytes / self.stall_seconds
        # directly (e.g. orchestration picks per-config budgets from a CalibrationTable).
        self.max_output_bytes = int(os.environ.get("AGY_MAX_OUTPUT_BYTES", "0") or 0)
        self.stall_seconds = float(os.environ.get("AGY_STALL_SECONDS", "0") or 0)
        # Set by the watchdog when it trips so callers (and FallbackAgent) can
        # distinguish a runaway from a normal failure. Cleared at each run start.
        self._watchdog_reason: Optional[str] = None
        # Telemetry recorded for every completed call (wall_ms always; out_bytes
        # only when streaming). Sinks (ledger, calibration) read these.
        self.last_wall_ms: Optional[float] = None
        self.last_out_bytes: Optional[int] = None
        self.last_usage: Optional[Dict[str, object]] = None
        # Optional dashboard/event-bus callback. Best-effort only: failures in
        # observability must never affect execution.
        self.event_callback: Optional[Callable[[dict], None]] = None
        # Working directory the worker child process runs in. None = inherit
        # whatever cwd the parent (harness/dashboard) was launched in. The
        # harness sets this to its --out-dir so a worker invocation from
        # another repo doesn't pollute AgentOrch itself.
        self.cwd: Optional[str] = None
        # Extra environment variables to inject into the child subprocess. Empty
        # by default (child inherits the parent env unchanged). Subclasses set
        # this to override/add vars for their CLI (e.g. AgyAgent neuters BROWSER
        # so an expired-token OAuth flow can't pop a browser inside a headless run).
        self.extra_env: Dict[str, str] = {}

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

    def _worker_name(self) -> str:
        name = self.__class__.__name__.lower()
        for worker in ("codex", "claude", "agy", "grok"):
            if worker in name:
                return worker
        return "agy"

    def _emit_event(self, event: dict) -> None:
        cb = self.event_callback
        if cb is None:
            return
        try:
            cb(dict(event or {}))
        except Exception:
            pass

    def _events_from_stdout_line(self, line: str) -> List[dict]:
        return []

    def _events_from_stderr_line(self, line: str) -> List[dict]:
        txt = line.rstrip("\n")
        if not txt:
            return []
        return [{"kind": "stderr", "text": txt, "data": {}}]

    def _events_from_stdout_complete(self, raw_stdout: str) -> List[dict]:
        return []

    def _child_env(self) -> Optional[Dict[str, str]]:
        """Environment for the child subprocess: parent env plus self.extra_env,
        or None to inherit the parent env unchanged (the default — preserves
        prior behavior exactly when no overrides are set)."""
        if not self.extra_env:
            return None
        return {**os.environ, **self.extra_env}

    @staticmethod
    def _to_int_token(value: object) -> Optional[int]:
        if value is None or isinstance(value, bool):
            return None
        try:
            iv = int(value)
        except Exception:
            return None
        return iv if iv >= 0 else None

    def _extract_usage(self, raw_stdout: str, raw_stderr: str) -> Dict[str, object]:
        """Best-effort token usage extraction from normal CLI output.

        Subclasses override this when their CLI emits usage counts in
        non-interactive output. Failures must degrade to "unavailable".
        """
        return {
            "token_source": "unavailable",
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_tokens": None,
            "total_tokens": None,
        }

    def _emit_usage_event(self, raw_stdout: str, raw_stderr: str, *,
                          attempt: int, success: bool) -> None:
        usage: Dict[str, object]
        try:
            usage = dict(self._extract_usage(raw_stdout, raw_stderr) or {})
        except Exception:
            usage = {}
        token_source = str(usage.get("token_source") or "unavailable")
        if token_source not in {"cli", "unavailable"}:
            token_source = "unavailable"
        usage_data = {
            "usage_kind": "call",
            "token_source": token_source,
            "input_tokens": self._to_int_token(usage.get("input_tokens")),
            "output_tokens": self._to_int_token(usage.get("output_tokens")),
            "cache_read_tokens": self._to_int_token(usage.get("cache_read_tokens")),
            "total_tokens": self._to_int_token(usage.get("total_tokens")),
            "attempt": attempt,
            "success": bool(success),
            "worker": self._worker_name(),
            "model": self.model,
        }
        self.last_usage = dict(usage_data)
        self._emit_event({"kind": "usage", "data": usage_data})

    async def _stream_communicate(self, process, stdin_bytes: Optional[bytes] = None,
                                  *, max_output_bytes: int = 0,
                                  stall_seconds: float = 0) -> Tuple[bytes, bytes]:
        """Drain stdout+stderr concurrently, echoing stderr live to our stderr.

        Returns the full (stdout_bytes, stderr_bytes) so callers behave exactly
        as with ``process.communicate()`` — the only difference is that the
        child's stderr is mirrored to our own stderr line-by-line as it arrives,
        which makes a long build's progress tailable instead of buffered. Feeds
        ``stdin_bytes`` (if any) concurrently so stdin-delivered prompts also stream.

        When ``max_output_bytes`` or ``stall_seconds`` is set, a streaming watchdog
        kills the child if the corresponding budget is exceeded and records the
        trip reason on ``self._watchdog_reason`` so the caller can distinguish a
        runaway from a hard failure. The watchdog is conservative — defaults are
        disabled, thresholds are intended to come from a per-config CalibrationTable.
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

        # Mutable state shared with the watchdog: byte tally on stdout (for the
        # verbose-runaway budget), and a single last-progress timestamp that
        # advances on output from EITHER stream. Stderr is where workers do
        # most of their visible work (codex exec lines + apply_patch traces,
        # claude/agy/grok status, network-retry messages); treating its silence
        # as "stalled" was a false-positive farm — codex's normal recovery from
        # network blips looks like a stall under stdout-only progress tracking.
        out_total = [0]
        any_output = [False]
        # Reset the shared last-progress clock at the start of this attempt so a
        # stale value from a prior attempt can't make the liveness wait think the
        # child is already idle.
        self._last_progress = time.monotonic()

        def _emit_line(line: bytes, echo: bool, stream: str) -> None:
            if echo:
                sys.stderr.buffer.write(line)
                sys.stderr.buffer.flush()
            text = line.decode(errors="replace")
            try:
                if stream == "stdout":
                    events = self._events_from_stdout_line(text)
                else:
                    events = self._events_from_stderr_line(text)
            except Exception:
                events = []
            for event in events:
                self._emit_event(event)

        async def _drain(reader, chunks, echo: bool, count: bool, stream: str) -> None:
            if reader is None:
                return
            # Read fixed-size chunks instead of readline(): readline() on the
            # default 64 KiB StreamReader raises LimitOverrunError ("Separator is
            # not found, and chunk exceed the limit") on any single line longer
            # than 64 KiB with no newline — common for claude emitting one long
            # JSON payload — failing the attempt deterministically on every retry.
            # Chunked reads are immune to line length. We still re-split on '\n'
            # before dispatching to the per-line event consumers, because the
            # stream-json adapters (load_json_line) require exactly one JSON
            # object per line. The full byte stream is reconstructed identically
            # from `chunks`, so raw_stdout / usage extraction are byte-for-byte
            # unchanged. See issue #30.
            buf = bytearray()
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                chunks.append(data)
                # Either stream resets the stall clock — the watchdog cares
                # about "is the worker doing anything", not "is stdout flowing".
                self._last_progress = time.monotonic()
                any_output[0] = True
                if count:
                    # Byte budget tallies stdout only: runaway-verbose tax
                    # (e.g. claude:haiku spitting 4528 tokens on calc3) lands
                    # on stdout. Stderr volume is normal worker chatter.
                    out_total[0] += len(data)
                buf.extend(data)
                # Emit every complete (newline-terminated) line; keep the partial
                # remainder buffered for the next read.
                nl = buf.find(b"\n")
                while nl >= 0:
                    _emit_line(bytes(buf[: nl + 1]), echo, stream)
                    del buf[: nl + 1]
                    nl = buf.find(b"\n")
            # Flush any trailing newline-less final segment at EOF (matches the
            # final non-terminated chunk readline() used to return).
            if buf:
                _emit_line(bytes(buf), echo, stream)

        async def _watchdog() -> None:
            # Poll every 2s; cheap relative to subprocess scheduling, fine-grained
            # enough that a runaway emitting 10KB/s trips within one tick of budget.
            if max_output_bytes <= 0 and stall_seconds <= 0:
                return
            while process.returncode is None:
                await asyncio.sleep(2.0)
                if max_output_bytes > 0 and out_total[0] > max_output_bytes:
                    self._watchdog_reason = WATCHDOG_VERBOSE
                    logger.warning("watchdog: VERBOSE trip — %d bytes > budget %d; killing",
                                   out_total[0], max_output_bytes)
                    self._killpg_tree(process)
                    return
                if stall_seconds > 0 and (time.monotonic() - self._last_progress) > stall_seconds:
                    # Only trip stall if NOTHING has been produced on EITHER stream
                    # yet. A worker that emitted, then went silent for one stall
                    # window, may be reasoning between bursts — the hard wall-timeout
                    # (AGY_TIMEOUT, default 2400s) still catches genuine hangs.
                    if not any_output[0]:
                        self._watchdog_reason = WATCHDOG_STALLED
                        logger.warning("watchdog: STALLED trip — no output for %.0fs; killing",
                                       stall_seconds)
                        self._killpg_tree(process)
                        return

        out_chunks: List[bytes] = []
        err_chunks: List[bytes] = []
        await asyncio.gather(
            _feed(),
            _drain(process.stdout, out_chunks, echo=False, count=True, stream="stdout"),
            _drain(process.stderr, err_chunks, echo=True, count=False, stream="stderr"),
            _watchdog(),
        )
        await process.wait()
        self.last_out_bytes = out_total[0]
        return b"".join(out_chunks), b"".join(err_chunks)

    @staticmethod
    def _killpg_tree(proc: "asyncio.subprocess.Process") -> None:
        """SIGKILL the child's ENTIRE process group, not just the child.

        The worker is spawned with start_new_session=True, so it leads its own
        process group and every descendant (e.g. a `codex exec` grandchild
        running `make check`) shares that pgid. Killing the group reaps the whole
        tree; killing only the child lets the grandchild escape, reparent to init,
        and keep running — the runaway-codex failure mode. Falls back to a plain
        child kill if the group lookup/kill fails (e.g. it already exited)."""
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.kill()
        except Exception:
            pass

    async def _kill_current(self) -> None:
        """Kill and reap the tracked child TREE if it is still running.
        Best-effort, bounded so a wedged process can't hang the reaping itself."""
        proc = self._current_process
        self._current_process = None
        if proc is None or proc.returncode is not None:
            return
        self._killpg_tree(proc)
        try:
            await asyncio.wait_for(proc.wait(), 5)
        except Exception:
            pass

    def _absolute_cap(self) -> float:
        """Hard wall-clock ceiling for a streaming call (0 = uncapped). Explicit
        AGY_ABSOLUTE_TIMEOUT wins; otherwise default to a multiple of timeout."""
        if self.absolute_timeout and self.absolute_timeout > 0:
            return self.absolute_timeout
        if self.timeout and self.timeout > 0:
            return self.timeout * ABSOLUTE_TIMEOUT_FACTOR
        return 0.0

    async def _await_with_liveness(self, comm, attempt_start: float):
        """Await a streaming call, extending the deadline while the child keeps
        emitting output. Raises asyncio.TimeoutError when the child has produced
        NO new output for a full `timeout` (idle) window, or when the absolute
        wall-clock cap is reached — whichever comes first. This replaces the flat
        `wait_for(comm, timeout)` for streaming runs so a legitimately long turn
        that is still making visible progress isn't discarded mid-flight. (#28)"""
        idle_window = self.timeout if self.timeout and self.timeout > 0 else 0.0
        hard_cap = self._absolute_cap()
        if idle_window <= 0 and hard_cap <= 0:
            return await comm  # no ceiling at all
        task = asyncio.ensure_future(comm)
        try:
            while True:
                now = time.monotonic()
                last = self._last_progress or attempt_start
                if hard_cap > 0 and (now - attempt_start) >= hard_cap:
                    raise asyncio.TimeoutError
                if idle_window > 0 and (now - last) >= idle_window:
                    raise asyncio.TimeoutError
                slice_s = 2.0  # poll cadence; matches the watchdog tick
                if idle_window > 0:
                    slice_s = min(slice_s, max(idle_window - (now - last), 0.05))
                if hard_cap > 0:
                    slice_s = min(slice_s, max(hard_cap - (now - attempt_start), 0.05))
                # asyncio.wait does NOT cancel the task on timeout (unlike
                # wait_for), so the streaming call survives each poll slice.
                done, _pending = await asyncio.wait({task}, timeout=max(slice_s, 0.05))
                if task in done:
                    return task.result()
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except BaseException:
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
        # Force streaming mode when any watchdog budget is set — the watchdog
        # needs line-by-line visibility to count bytes and detect stalls. Stays
        # opt-in: with no budgets set, behaviour is identical to the prior path.
        watchdog_armed = self.max_output_bytes > 0 or self.stall_seconds > 0
        stream_mode = bool(os.environ.get("AGY_STREAM")) or watchdog_armed or self.event_callback is not None

        self._emit_event({"kind": "lifecycle", "data": {"event": "agent_started", "detail": {}}})
        finished: Optional[bool] = None
        try:
            for attempt in range(1, self.max_retries + 1):
                self._watchdog_reason = None
                attempt_start = time.monotonic()
                self._last_progress = attempt_start
                try:
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else None,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        # Raise the StreamReader buffer well above the 64 KiB
                        # default so a long single-line worker payload never
                        # overflows even on the rare code path that still buffers
                        # a whole line (the chunked _drain is immune regardless).
                        limit=8 * 1024 * 1024,
                        cwd=self.cwd,
                        env=self._child_env(),
                        # Own session/process group so the WHOLE worker tree is
                        # reapable on timeout/watchdog kill. Without this, killing
                        # the wrapper (e.g. `codex`) leaves its heavy grandchild
                        # (`codex exec` running `make check`) to reparent to init
                        # and keep burning CPU, starving the real verifier. See
                        # _kill_current()/_killpg_tree().
                        start_new_session=True,
                    )
                    self._current_process = process

                    # Live mode: drain both pipes line-by-line, echoing the child's
                    # stderr as it arrives (tailable progress) while accumulating
                    # the full streams. (codex writes work to stderr live; claude
                    # --output-format json emits only at the end.) The watchdog
                    # rides this path and trips on per-config budgets.
                    if stream_mode:
                        comm = self._stream_communicate(
                            process, stdin_bytes,
                            max_output_bytes=self.max_output_bytes,
                            stall_seconds=self.stall_seconds,
                        )
                    else:
                        comm = process.communicate(input=stdin_bytes)

                    try:
                        if stream_mode:
                            # Liveness-aware: extend while the child streams output,
                            # capped by the absolute ceiling (#28).
                            stdout_bytes, stderr_bytes = await self._await_with_liveness(
                                comm, attempt_start)
                        elif self.timeout and self.timeout > 0:
                            stdout_bytes, stderr_bytes = await asyncio.wait_for(comm, self.timeout)
                        else:
                            stdout_bytes, stderr_bytes = await comm
                    except asyncio.TimeoutError:
                        waited = time.monotonic() - attempt_start
                        logger.error("Agent subprocess exceeded liveness/abs ceiling "
                                     "(%.0fs elapsed, idle>%.0fs or cap %.0fs); killing and "
                                     "failing over.", waited, self.timeout, self._absolute_cap())
                        await self._kill_current()
                        self.stderr = f"timed out after {waited:.0f}s"
                        self._emit_usage_event("", self.stderr, attempt=attempt, success=False)
                        raise RuntimeError(self.stderr)  # fail fast, no retry

                    raw_stdout = stdout_bytes.decode()
                    self.stderr = self.filter_stderr(stderr_bytes.decode())
                    self.returncode = process.returncode
                    self._current_process = None
                    self.last_wall_ms = (time.monotonic() - attempt_start) * 1000
                    self._emit_usage_event(
                        raw_stdout, self.stderr, attempt=attempt, success=bool(self.returncode == 0)
                    )

                    # Watchdog trip: the child was killed mid-flight. Surface the
                    # reason in stderr so FallbackAgent (and human readers) can
                    # see it; raise fast so the chain can re-route on the rule.
                    if self._watchdog_reason:
                        marker = f"{WATCHDOG_MARKER}{self._watchdog_reason}]"
                        self._emit_event({
                            "kind": "watchdog",
                            "text": marker,
                            "data": {"reason": self._watchdog_reason},
                        })
                        self.stderr = f"{marker} {self.stderr}".strip()
                        raise RuntimeError(self.stderr)

                    if self.returncode == 0:
                        self.stdout = self._postprocess(raw_stdout)
                        for event in self._events_from_stdout_complete(raw_stdout):
                            self._emit_event(event)
                        finished = True
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
            finished = False
            raise RuntimeError(f"AgentInstance failed after {self.max_retries} attempts: {self.stderr}")
        finally:
            if finished is None:
                finished = False
            self._emit_event({
                "kind": "lifecycle",
                "data": {
                    "event": "agent_finished",
                    "detail": {"success": finished, "returncode": self.returncode},
                },
            })
            await self._kill_current()
            self._cleanup()

    def run(self, piped_input: Optional[str] = None) -> str:
        """Synchronous wrapper for run_async."""
        return asyncio.run(self.run_async(piped_input))

"""C2: the singleton broker — drain loop, concurrency cap, pool coordination.

A single long-lived process owns a persistent :class:`~harness.job_queue.JobQueue`
and drains it with an ``asyncio.Semaphore(MAX_CONCURRENT)`` cap of 2 live jobs.
The broker coordinates generator/critic *pools* across the two live lines so they
don't collide on one provider account (the account-sharing rule): when a slot is
free it prefers a queued job whose pools are disjoint from the pools already in
flight. If the only remaining candidates collide, it waits a bounded window
(``AGY_BROKER_POOL_WAIT``, default ~5s) and then runs the oldest queued job
anyway — never starving the queue — logging a pool-overlap warning.

The :class:`Broker` is usable WITHOUT any socket: the IPC server + singleton
guard (C3) live in the same module but are layered on top. Construct ``Broker``
with a :class:`JobQueue` and (optionally) an injected ``dispatch_fn`` and call
``await broker.run()``; ``broker.stop()`` ends the loop. Every status transition
is persisted through the queue's atomic writer.
"""
from __future__ import annotations

import asyncio
import atexit
import errno
import json
import logging
import os
import signal
import socket
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

from harness.job_queue import Job, JobQueue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- queue paths
# The broker's state lives under runs/.queue/ (gitignored, like runs/), mirroring
# harness.dispatch.RUNS_DIR resolution so a plain `harness serve` and a `harness
# do` agree on where the socket / queue / pidfile are.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUEUE_DIR = PROJECT_ROOT / "runs" / ".queue"
QUEUE_PATH = QUEUE_DIR / "queue.json"
SOCK_PATH = QUEUE_DIR / "broker.sock"
PID_PATH = QUEUE_DIR / "broker.pid"


# Default anti-deadlock wait before a pool-colliding job is run anyway.
_DEFAULT_POOL_WAIT = 5.0
# Idle poll interval when nothing is runnable and we are waiting for either a
# slot to free or the queue to gain work. Kept small so newly-submitted jobs and
# the pool-wait escape are picked up promptly without a busy loop.
_IDLE_POLL = 0.05


def _pool_wait_seconds(explicit: Optional[float]) -> float:
    """Resolve the anti-deadlock wait: explicit arg > env > default."""
    if explicit is not None:
        return max(0.0, float(explicit))
    raw = os.environ.get("AGY_BROKER_POOL_WAIT")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            logger.warning("AGY_BROKER_POOL_WAIT=%r is not a number; using default", raw)
    return _DEFAULT_POOL_WAIT


def _default_dispatch_fn() -> Callable[..., Awaitable]:
    """Late import of the real dispatcher so importing this module is cheap and
    so tests that inject a fake never load the heavy dispatch stack."""
    from harness.dispatch import dispatch_async

    return dispatch_async


def _normalize_result(raw) -> dict:
    """Build the compact, json-safe result summary the queue stores.

    ``dispatch_async`` returns a :class:`DispatchResult` dataclass; a fake
    dispatch_fn may return a dict, an arbitrary object, or ``None``. Read
    ``run_id`` / ``success`` / ``error`` defensively from either shape.
    """
    if raw is None:
        return {"success": True, "run_id": None, "error": None}

    def _field(name, default=None):
        if isinstance(raw, dict):
            return raw.get(name, default)
        return getattr(raw, name, default)

    success = _field("success", True)
    return {
        "success": bool(success),
        "run_id": _field("run_id"),
        "error": _field("error"),
    }


class Broker:
    """Drains a :class:`JobQueue` with a concurrency cap and pool coordination.

    Usable without a socket — the IPC layer (C3) wraps this. One instance per
    broker process; it owns the queue and mutates it from a single event loop.
    """

    MAX_CONCURRENT = 2

    def __init__(
        self,
        queue: JobQueue,
        *,
        dispatch_fn: Optional[Callable[..., Awaitable]] = None,
        cap: Optional[int] = None,
        pool_wait: Optional[float] = None,
    ) -> None:
        self.queue = queue
        self._dispatch_fn = dispatch_fn  # None -> resolved lazily to dispatch_async
        self.cap = int(cap) if cap is not None else self.MAX_CONCURRENT
        self._pool_wait = _pool_wait_seconds(pool_wait)
        self._sem = asyncio.Semaphore(self.cap)
        # Pools currently in flight (union across running jobs) + bookkeeping.
        self._active_pools: dict[str, list] = {}  # job_id -> that job's pools
        self._running_tasks: "set[asyncio.Task]" = set()
        self._stopping = False
        # Set when something changes that might make a job runnable (a job
        # finished, or a new job was submitted via _wake()). Lets the drain loop
        # block instead of busy-polling.
        self._wakeup = asyncio.Event()
        # Timestamp we first noticed only-colliding candidates with a free slot;
        # None when not in a pool-wait window. Drives the anti-deadlock escape.
        self._collide_since: Optional[float] = None

    # ----------------------------------------------------------- dispatch_fn
    @property
    def dispatch_fn(self) -> Callable[..., Awaitable]:
        if self._dispatch_fn is None:
            self._dispatch_fn = _default_dispatch_fn()
        return self._dispatch_fn

    # ------------------------------------------------------------- active set
    def _active_pool_union(self) -> set:
        out: set = set()
        for pools in self._active_pools.values():
            out.update(pools)
        return out

    def wake(self) -> None:
        """Nudge the drain loop to re-check for runnable work (e.g. after a new
        submit arrives over IPC). Safe to call from the same event loop."""
        self._wakeup.set()

    # ---------------------------------------------------------------- run/stop
    async def run(self) -> None:
        """Drain the queue until :meth:`stop` is called.

        Holds at most ``cap`` jobs in flight. On each iteration: if a slot is
        free, pick the oldest queued job whose pools are disjoint from those in
        flight. If none is disjoint but queued work exists and a slot is free,
        start a bounded pool-wait; once it elapses, run the oldest queued job
        regardless of pools and warn. Otherwise sleep until woken.
        """
        self._stopping = False
        while not self._stopping:
            launched = await self._try_launch_one()
            if launched:
                # A slot may still be free; loop again immediately to fill it.
                continue
            if self._stopping:
                break
            # Nothing launched this pass: either no free slot, no queued work, or
            # only colliding candidates still inside the pool-wait window. Block
            # on the wakeup event with a short timeout so the pool-wait deadline
            # and external submits are both observed promptly.
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=_IDLE_POLL)
            except asyncio.TimeoutError:
                pass
            self._wakeup.clear()

        # Drain in-flight jobs on stop so status is persisted before exit.
        if self._running_tasks:
            await asyncio.gather(*list(self._running_tasks), return_exceptions=True)

    def stop(self) -> None:
        """Request the drain loop to exit after in-flight jobs settle."""
        self._stopping = True
        self._wakeup.set()

    # ------------------------------------------------------------ scheduling
    async def _try_launch_one(self) -> bool:
        """Try to start exactly one job. Returns True iff one was launched."""
        # Respect the concurrency cap without blocking the loop: only acquire the
        # semaphore once we have a concrete job to run.
        if len(self._running_tasks) >= self.cap:
            return False

        active = self._active_pool_union()
        job = self.queue.next_runnable(active)
        if job is not None:
            # A pool-disjoint job is available -> clear any pool-wait window.
            self._collide_since = None
            await self._launch(job)
            return True

        # No disjoint job. Is there queued work at all (only-colliding case)?
        oldest = self.queue.oldest_queued()
        if oldest is None:
            # Queue empty (of queued jobs) -> nothing to wait on.
            self._collide_since = None
            return False

        # There IS queued work but it collides on pools with the in-flight set.
        # If no job is in flight at all, there is nothing to collide WITH, so run
        # it now (defensive: next_runnable already covers this, but keep the
        # invariant explicit).
        if not self._active_pools:
            self._collide_since = None
            await self._launch(oldest)
            return True

        # Start / continue the bounded anti-deadlock wait.
        now = time.monotonic()
        if self._collide_since is None:
            self._collide_since = now
            return False
        if now - self._collide_since >= self._pool_wait:
            logger.warning(
                "Broker pool-overlap: only pool-colliding jobs queued for %.1fs "
                "(>=%.1fs); running oldest job %s anyway (pools=%s, active=%s)",
                now - self._collide_since,
                self._pool_wait,
                oldest.id,
                oldest.pools,
                sorted(active),
            )
            self._collide_since = None
            await self._launch(oldest)
            return True
        return False

    async def _launch(self, job: Job) -> None:
        """Acquire a slot, mark the job running, and spawn its task."""
        await self._sem.acquire()
        # Reserve the job's pools and mark running BEFORE the task starts so a
        # subsequent _try_launch_one in the same loop pass sees them in flight.
        self._active_pools[job.id] = list(job.pools)
        self.queue.mark_running(job.id, job.run_id)
        task = asyncio.ensure_future(self._run_job(job))
        self._running_tasks.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: "asyncio.Task") -> None:
        self._running_tasks.discard(task)
        # A slot freed up -> re-evaluate scheduling promptly.
        self._wakeup.set()

    async def _run_job(self, job: Job) -> None:
        """Run one job through dispatch_fn, persisting done/failed + freeing pools."""
        try:
            try:
                raw = await self.dispatch_fn(**job.kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - any dispatch failure -> failed job
                logger.exception("Broker job %s failed in dispatch", job.id)
                self.queue.mark_failed(job.id, f"{type(exc).__name__}: {exc}")
                return
            result = _normalize_result(raw)
            if result.get("run_id") and not self.queue.get(job.id).run_id:
                # Record the dispatch's run_id if it minted one after mark_running.
                self.queue.mark_running(job.id, result["run_id"])
            if result.get("success"):
                self.queue.mark_done(job.id, result)
            else:
                self.queue.mark_failed(job.id, result.get("error") or "dispatch reported failure")
        finally:
            self._active_pools.pop(job.id, None)
            self._sem.release()
            self._wakeup.set()


# ===========================================================================
# C3 — IPC (unix socket, line-delimited JSON) + singleton guard
# ===========================================================================

class BrokerAlreadyRunning(RuntimeError):
    """Raised when a live broker already answers ping on the target socket."""

    def __init__(self, pid, sock_path) -> None:
        self.pid = pid
        self.sock_path = str(sock_path)
        super().__init__(f"broker already running, pid {pid}")


def _read_pidfile(pid_path) -> Optional[int]:
    """Best-effort read of broker.pid; None if absent/garbage."""
    try:
        raw = Path(pid_path).read_text().strip()
    except OSError:
        return None
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None


def _probe_ping(sock_path, timeout: float = 0.5) -> Optional[dict]:
    """Connect to an existing socket and send a single ``ping``.

    Returns the parsed response dict if a live broker answers, else ``None``
    (no socket, connection refused, stale file, timeout, garbage reply). Never
    raises and never hangs longer than ``timeout``.
    """
    sock_path = str(sock_path)
    if not os.path.exists(sock_path):
        return None
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
        s.sendall(json.dumps({"op": "ping"}).encode("utf-8") + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
        if not buf:
            return None
        line = buf.split(b"\n", 1)[0]
        resp = json.loads(line.decode("utf-8"))
        return resp if isinstance(resp, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    finally:
        try:
            s.close()
        except OSError:
            pass


class BrokerServer:
    """Asyncio unix-socket IPC server wrapping a :class:`Broker`.

    Binds ``sock_path`` (default :data:`SOCK_PATH`), speaks line-delimited JSON,
    and dispatches the ops in §C3 onto the broker's queue. The singleton guard
    lives in :meth:`bind`: if the socket path already exists and a live broker
    answers ``ping``, binding refuses with :class:`BrokerAlreadyRunning`; a stale
    socket file (no answer) is unlinked and reclaimed. Writes ``broker.pid`` and
    cleans up the socket + pidfile on shutdown (atexit + SIGTERM/SIGINT).
    """

    def __init__(
        self,
        broker: Broker,
        *,
        sock_path=None,
        pid_path=None,
    ) -> None:
        self.broker = broker
        self.queue = broker.queue
        self.sock_path = str(sock_path) if sock_path is not None else str(SOCK_PATH)
        self.pid_path = str(pid_path) if pid_path is not None else str(PID_PATH)
        self._server: Optional[asyncio.AbstractServer] = None
        self._cleaned = False
        self._atexit_registered = False
        self._prev_handlers: dict = {}

    # ----------------------------------------------------------- singleton guard
    def _refuse_if_live(self) -> None:
        """Honor the singleton guard. Refuse if a live broker answers ping;
        unlink a stale socket so we can rebind."""
        if not os.path.exists(self.sock_path):
            return
        resp = _probe_ping(self.sock_path)
        if resp is not None and resp.get("ok"):
            pid = resp.get("pid")
            if pid is None:
                pid = _read_pidfile(self.pid_path)
            raise BrokerAlreadyRunning(pid, self.sock_path)
        # Stale socket file (no live answer) -> reclaim it.
        logger.info("Broker: reclaiming stale socket %s", self.sock_path)
        try:
            os.unlink(self.sock_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Broker: could not unlink stale socket %s (%s)", self.sock_path, exc)

    def _write_pidfile(self) -> None:
        Path(self.pid_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.pid_path + f".tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            fh.write(str(os.getpid()))
        os.replace(tmp, self.pid_path)

    def cleanup(self) -> None:
        """Remove socket + pidfile. Idempotent; safe from a signal handler."""
        if self._cleaned:
            return
        self._cleaned = True
        for p in (self.sock_path, self.pid_path):
            try:
                os.unlink(p)
            except (FileNotFoundError, OSError):
                pass
        # Restore any signal handlers we replaced so we don't leak global state
        # (matters for in-process tests sharing the interpreter).
        for sig, prev in list(self._prev_handlers.items()):
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError, TypeError):
                pass
        self._prev_handlers.clear()

    def _install_cleanup_handlers(self) -> None:
        if not self._atexit_registered:
            atexit.register(self.cleanup)
            self._atexit_registered = True

        def _handler(signum, frame):
            self.cleanup()
            self.broker.stop()
            # Re-raise default behavior so the process actually terminates.
            prev = self._prev_handlers.get(signum)
            if callable(prev):
                prev(signum, frame)
            else:
                raise KeyboardInterrupt()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._prev_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # Not on the main thread (e.g. under a test loop) -> skip; atexit
                # still covers normal interpreter exit.
                pass

    # --------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        """Honor the singleton guard, bind the socket, write the pidfile, and
        begin accepting connections. Raises :class:`BrokerAlreadyRunning` if a
        live broker already owns the socket."""
        Path(self.sock_path).parent.mkdir(parents=True, exist_ok=True)
        # Probe an existing socket off the event loop: if THIS loop already hosts
        # a server (e.g. an in-process test binding a second one), a synchronous
        # probe would deadlock waiting for an accept it is itself blocking.
        await asyncio.to_thread(self._refuse_if_live)
        self._server = await asyncio.start_unix_server(self._handle_client, path=self.sock_path)
        self._write_pidfile()
        self._install_cleanup_handlers()
        logger.info("Broker IPC listening on %s (pid %d)", self.sock_path, os.getpid())

    async def serve_forever(self) -> None:
        """Run the IPC server and the broker drain loop together until stopped."""
        if self._server is None:
            await self.start()
        drain = asyncio.ensure_future(self.broker.run())
        try:
            await drain
        finally:
            await self.close()

    async def close(self) -> None:
        """Stop accepting, drain the broker, and clean up socket + pidfile."""
        self.broker.stop()
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:  # noqa: BLE001
                pass
            self._server = None
        self.cleanup()

    # ----------------------------------------------------------- request loop
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line.decode("utf-8"))
                    if not isinstance(req, dict):
                        raise ValueError("request is not an object")
                except (ValueError, json.JSONDecodeError) as exc:
                    resp = {"ok": False, "error": f"bad request: {exc}"}
                else:
                    resp = self._handle_op(req)
                writer.write(json.dumps(resp).encode("utf-8") + b"\n")
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception:  # noqa: BLE001
            logger.exception("Broker IPC: error serving client")
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    def _handle_op(self, req: dict) -> dict:
        op = req.get("op")
        try:
            if op == "ping":
                return self._op_ping()
            if op == "submit":
                return self._op_submit(req)
            if op == "list":
                return self._op_list()
            if op == "status":
                return self._op_status(req)
            if op == "cancel":
                return self._op_cancel(req)
            return {"ok": False, "error": f"unknown op: {op!r}"}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Broker IPC: op %r failed", op)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------- ops
    def _counts(self):
        running = queued = 0
        for rec in self.queue.snapshot():
            st = rec.get("status")
            if st == "running":
                running += 1
            elif st == "queued":
                queued += 1
        return running, queued

    def _op_ping(self) -> dict:
        running, queued = self._counts()
        return {
            "ok": True,
            "pid": os.getpid(),
            "cap": self.broker.cap,
            "running": running,
            "queued": queued,
        }

    def _op_submit(self, req: dict) -> dict:
        instruction = req.get("instruction", "")
        kwargs = req.get("kwargs") or {}
        pools = req.get("pools") or []
        if not isinstance(kwargs, dict):
            return {"ok": False, "error": "kwargs must be an object"}
        if not isinstance(pools, list):
            return {"ok": False, "error": "pools must be a list"}
        job = self.queue.submit(instruction, kwargs, pools)
        # Nudge the drain loop so the new job is considered promptly.
        self.broker.wake()
        return {"ok": True, "job_id": job.id}

    def _op_list(self) -> dict:
        return {"ok": True, "jobs": self.queue.snapshot()}

    def _op_status(self, req: dict) -> dict:
        job_id = req.get("job_id")
        job = self.queue.get(job_id) if job_id else None
        return {"ok": True, "job": job.to_dict() if job is not None else None}

    def _op_cancel(self, req: dict) -> dict:
        job_id = req.get("job_id")
        canceled = self.queue.cancel(job_id) if job_id else False
        return {"ok": True, "canceled": bool(canceled)}


def make_server(
    *,
    queue_path=None,
    sock_path=None,
    pid_path=None,
    dispatch_fn=None,
    cap=None,
) -> BrokerServer:
    """Convenience: build a JobQueue + Broker + BrokerServer from paths.

    Used by ``harness serve``. All paths default to the runs/.queue/ locations.
    The queue is loaded (running -> queued reset applied) before serving.
    """
    queue = JobQueue(queue_path if queue_path is not None else QUEUE_PATH)
    queue.load()
    broker = Broker(queue, dispatch_fn=dispatch_fn, cap=cap)
    return BrokerServer(broker, sock_path=sock_path, pid_path=pid_path)

"""Run-level watchdog, heartbeat, and notification hook (issue #40).

The orchestrator already has a *per-agent* streaming watchdog
(``harness/roles.py`` arms ``stall_seconds`` per worker), but that only catches
a *silent* worker. It does nothing for:

  * a run making **no run-level forward progress** while a worker keeps
    producing tokens (the per-agent stall never trips — the run is
    "stuck but chatty"), or a verifier wedged just under its timeout;
  * an unattended/overnight run that wedges or thrashes with **no signal**
    until it fails to finish.

This module adds three first-class, opt-in (heartbeat: on-by-default) signals
at *run* scope, complementing the per-agent watchdog:

  * **Run watchdog** — abort the run if no run-level forward-progress *event*
    (a step/branch/plan milestone, a worker-call boundary, a usage tick — i.e.
    anything that is NOT raw token chatter) occurs within a window.
  * **Heartbeat** — a periodic ``heartbeat`` event into ``events.jsonl`` with
    the current step, free memory, elapsed, and seconds-since-progress, so an
    external watcher (or the dashboard) reads *one explicit* liveness signal
    instead of inferring it from file mtimes.
  * **Notify** — a best-effort, non-fatal hook fired on lifecycle / anomaly
    events (start / step / stall / oom / verifier-fail / finish) to a webhook
    URL and/or a shell command.

Everything here is intentionally cheap and dependency-free (stdlib only). The
progress/heartbeat math takes an injectable ``clock`` so the logic is unit
testable without real sleeping.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import threading
import time
import urllib.request
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Event kinds that are raw worker token chatter (or our own heartbeat), NOT
# run-level forward progress. Everything else — lifecycle (agent start/finish,
# orchestration step/branch/plan transitions), usage (a call completed),
# watchdog, and computer-use reasoning/tool events — counts as the run having
# advanced. This is the exact distinction that lets the run watchdog catch a
# "stuck but chatty" run that the per-agent stall watchdog (which only watches a
# single worker's stream liveness) misses.
_NON_PROGRESS_KINDS = frozenset({"stderr", "stdout", "heartbeat"})

HEARTBEAT_EVENT_KIND = "heartbeat"


def read_free_mem_mb() -> Optional[int]:
    """Available system memory in MiB from /proc/meminfo, or None off-Linux."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return None
    return None


class RunStalled(Exception):
    """Raised by :meth:`RunMonitor.run` when the run watchdog aborted the run."""

    def __init__(self, reason: str = "stalled", *, since_progress_s: Optional[float] = None):
        self.reason = reason
        self.since_progress_s = since_progress_s
        super().__init__(f"run aborted by watchdog: {reason}")


class Notifier:
    """Best-effort, non-fatal lifecycle/anomaly notifier.

    A webhook POSTs the JSON payload; a command runs in a shell with the
    payload on stdin and ``AGY_NOTIFY_*`` in the environment. Delivery is
    fire-and-forget on a daemon thread with a short timeout — a slow or broken
    endpoint never blocks (or fails) the run.
    """

    def __init__(
        self,
        *,
        webhook: Optional[str] = None,
        command: Optional[str] = None,
        run_id: str = "",
        timeout: float = 5.0,
    ):
        self.webhook = webhook or None
        self.command = command or None
        self.run_id = run_id
        self.timeout = float(timeout)

    @property
    def enabled(self) -> bool:
        return bool(self.webhook or self.command)

    def build_body(self, event: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        body = {"event": str(event), "run_id": self.run_id}
        if payload:
            body.update(payload)
        return body

    def command_env(self, event: str, body: Dict[str, Any]) -> Dict[str, str]:
        env = dict(os.environ)
        env["AGY_NOTIFY_EVENT"] = str(event)
        env["AGY_NOTIFY_RUN_ID"] = str(self.run_id)
        env["AGY_NOTIFY_PAYLOAD"] = json.dumps(body, ensure_ascii=False)
        return env

    def fire(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Schedule best-effort delivery of one event. Never raises."""
        if not self.enabled:
            return
        body = self.build_body(event, payload)
        t = threading.Thread(target=self.deliver, args=(event, body), daemon=True)
        t.start()

    def deliver(self, event: str, body: Dict[str, Any]) -> None:
        """Synchronous delivery to webhook and/or command. Never raises."""
        if self.webhook:
            try:
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(
                    self.webhook,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=self.timeout).close()
            except Exception as exc:  # best-effort
                logger.debug("notify webhook failed (%s): %s", event, exc)
        if self.command:
            try:
                subprocess.run(
                    self.command,
                    shell=True,
                    env=self.command_env(event, body),
                    input=json.dumps(body, ensure_ascii=False),
                    text=True,
                    timeout=self.timeout,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as exc:  # best-effort
                logger.debug("notify command failed (%s): %s", event, exc)


class RunMonitor:
    """Supervises one dispatch with a heartbeat, a stall watchdog, and notify.

    ``emit`` is a callable that publishes one event dict (e.g. an EventBus
    publisher); it is used only for heartbeat events. ``observe`` is fed every
    event the run produces so the monitor can track run-level forward progress
    and the current step. The watchdog and heartbeat math use ``clock`` (a
    monotonic clock by default) so the supervision logic is unit-testable.
    """

    def __init__(
        self,
        *,
        run_id: str,
        emit: Optional[Callable[[dict], None]] = None,
        notifier: Optional[Notifier] = None,
        work_dir: Optional[str] = None,
        heartbeat_interval: float = 30.0,
        run_stall_abort: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.run_id = run_id
        self._emit = emit
        self.notifier = notifier or Notifier(run_id=run_id)
        self.work_dir = work_dir
        self.heartbeat_interval = float(heartbeat_interval or 0.0)
        self.run_stall_abort = float(run_stall_abort) if run_stall_abort else None
        self._clock = clock

        now = self._clock()
        self._start = now
        self._last_progress = now
        self._last_heartbeat = now
        self._step: Optional[str] = None
        self._step_index: Optional[int] = None
        self.abort_reason: Optional[str] = None
        self.heartbeats_emitted = 0

        self._task: Optional["asyncio.Future"] = None
        self._loop_task: Optional["asyncio.Future"] = None
        self._stop_evt: Optional[asyncio.Event] = None

    # -- progress tracking (synchronous, unit-testable) -------------------- #

    def observe(self, event: Optional[dict]) -> None:
        """Record one run event; resets the progress clock unless it's chatter."""
        event = event or {}
        self._track_step(event)
        if event.get("kind") in _NON_PROGRESS_KINDS:
            return
        self._last_progress = self._clock()

    def _track_step(self, event: dict) -> None:
        data = event.get("data")
        if not isinstance(data, dict):
            return
        orch = data.get("orchestration")
        if not isinstance(orch, dict) or orch.get("phase") != "step":
            return
        idx = orch.get("step_index")
        total = orch.get("step_total")
        if idx is not None:
            self._step_index = idx
            self._step = f"{idx}/{total}" if total else str(idx)
        if orch.get("action") == "completed":
            self.notifier.fire(
                "step",
                self.payload(extra={"phase": "step", "action": "completed"}),
            )

    def since_progress(self, now: Optional[float] = None) -> float:
        now = self._clock() if now is None else now
        return now - self._last_progress

    def elapsed(self, now: Optional[float] = None) -> float:
        now = self._clock() if now is None else now
        return now - self._start

    def should_abort(self, now: Optional[float] = None) -> bool:
        if not self.run_stall_abort:
            return False
        return self.since_progress(now) > self.run_stall_abort

    # -- payloads / heartbeat --------------------------------------------- #

    def payload(self, *, now: Optional[float] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        now = self._clock() if now is None else now
        body: Dict[str, Any] = {
            "run_id": self.run_id,
            "step": self._step,
            "elapsed_s": round(self.elapsed(now), 1),
            "since_progress_s": round(self.since_progress(now), 1),
            "free_mem_mb": read_free_mem_mb(),
        }
        if extra:
            body.update(extra)
        return body

    def heartbeat_event(self, now: Optional[float] = None) -> dict:
        return {"kind": HEARTBEAT_EVENT_KIND, "data": self.payload(now=now)}

    def emit_heartbeat(self, now: Optional[float] = None) -> None:
        if self._emit is None:
            return
        try:
            self._emit(self.heartbeat_event(now))
            self.heartbeats_emitted += 1
        except Exception as exc:
            logger.debug("heartbeat emit failed: %s", exc)

    # -- notify convenience ----------------------------------------------- #

    def notify(self, event: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        self.notifier.fire(event, self.payload(extra=extra))

    # -- async supervision ------------------------------------------------ #

    @property
    def _supervision_needed(self) -> bool:
        return bool(self.run_stall_abort) or self.heartbeat_interval > 0

    async def run(self, coro):
        """Run ``coro`` under heartbeat + watchdog supervision; return its result.

        If the watchdog trips, ``coro`` is cancelled and :class:`RunStalled` is
        raised. When neither heartbeat nor watchdog is active the coroutine is
        awaited directly (zero added supervision overhead / behaviour identical
        to the pre-#40 path).
        """
        if not self._supervision_needed:
            return await coro

        self._stop_evt = asyncio.Event()
        self._task = asyncio.ensure_future(coro)
        self._loop_task = asyncio.ensure_future(self._supervise())
        try:
            return await self._task
        except asyncio.CancelledError:
            if self.abort_reason:
                raise RunStalled(self.abort_reason, since_progress_s=self.since_progress())
            raise
        finally:
            if self._stop_evt is not None:
                self._stop_evt.set()
            if self._loop_task is not None:
                try:
                    await self._loop_task
                except Exception:
                    pass

    async def _supervise(self) -> None:
        poll = self.heartbeat_interval if self.heartbeat_interval > 0 else None
        if self.run_stall_abort:
            wd_poll = max(1.0, self.run_stall_abort / 4.0)
            poll = wd_poll if poll is None else min(poll, wd_poll)
        poll = poll or 5.0

        assert self._stop_evt is not None
        while not self._stop_evt.is_set():
            try:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=poll)
                return  # run finished -> stop requested
            except asyncio.TimeoutError:
                pass

            now = self._clock()
            if self.heartbeat_interval > 0 and (now - self._last_heartbeat) >= self.heartbeat_interval - 1e-6:
                self._last_heartbeat = now
                self.emit_heartbeat(now)

            if self.should_abort(now):
                self.abort_reason = "stalled"
                logger.warning(
                    "run watchdog: no run-level progress in %.1fs (> %.1fs) — aborting %s",
                    self.since_progress(now), self.run_stall_abort, self.run_id,
                )
                self.notifier.fire(
                    "stall",
                    self.payload(now=now, extra={"reason": "no run-level forward progress"}),
                )
                if self._task is not None and not self._task.done():
                    self._task.cancel()
                return

"""C1: persistent build-queue data layer for the singleton broker.

Pure data layer only — NO IPC, NO asyncio. The broker (C2) drives this and
adds the loop + sockets. Persistence is crash-safe: every mutation writes a
temp file and ``os.replace``s it into place atomically.

Loading a missing or garbage file degrades to an empty queue and logs — it
never raises. On load, any job left in ``status=="running"`` is reset to
``"queued"`` because the broker that owned it is dead, so the job re-runs.

stdlib-only by contract.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

JOB_STATUSES = ("queued", "running", "done", "failed", "canceled")

# status groups for ordering / liveness checks
_TERMINAL_STATUSES = ("done", "failed", "canceled")
_ACTIVE_STATUSES = ("queued", "running")

# Monotonic per-process tie-break counter. A *random* suffix kept ids unique
# but NOT sortable: two ids minted in the same millisecond could sort in either
# order, so a same-ms burst drained out of submission order (every ordering
# path here sorts by id). A monotonically increasing counter, zero-padded so it
# sorts lexicographically, breaks same-ms ties in insertion order.
_id_counter = itertools.count()


def _new_job_id() -> str:
    """Sortable id: timestamp to the millisecond + monotonic counter suffix.

    The clock is read EXACTLY ONCE (``time.time_ns()``); both the second-level
    ``strftime`` part and the millisecond part are derived from that single
    value. Reading the clock twice (one ``strftime`` call for the seconds, a
    separate ``time_ns`` for the sub-second) could straddle a second boundary
    and emit a stamp that sorts *before* the previous id — and because the id
    sorts lexicographically by stamp first, that reordered same-ms bursts even
    though the counter suffix was monotonic. The zero-padded monotonic counter
    breaks exact same-millisecond ties in insertion order.
    """
    ns = time.time_ns()
    secs, millis = divmod(ns // 1_000_000, 1000)  # ms since epoch -> (s, ms-in-s)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(secs)) + ("-%03d" % millis)
    suffix = "%012d" % next(_id_counter)
    return f"{stamp}-{suffix}"


@dataclass
class Job:
    id: str
    instruction: str
    kwargs: dict  # exact keyword args forwarded to dispatch_async (json-safe)
    status: str = "queued"
    submitted_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    run_id: Optional[str] = None
    pools: list = field(default_factory=list)
    result: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        return cls(
            id=d["id"],
            instruction=d.get("instruction", ""),
            kwargs=dict(d.get("kwargs") or {}),
            status=d.get("status", "queued"),
            submitted_at=float(d.get("submitted_at") or 0.0),
            started_at=d.get("started_at"),
            finished_at=d.get("finished_at"),
            run_id=d.get("run_id"),
            pools=list(d.get("pools") or []),
            result=d.get("result"),
        )


class JobQueue:
    """Crash-safe, persistent FIFO build queue.

    Not thread/async-safe by itself; the broker owns a single instance and
    mutates it from one event loop.
    """

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._jobs: "dict[str, Job]" = {}

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        """Read the queue file; tolerate missing/garbage by going empty."""
        self._jobs = {}
        try:
            raw = self.path.read_text()
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("JobQueue: could not read %s (%s); starting empty", self.path, exc)
            return
        try:
            data = json.loads(raw)
            records = data["jobs"] if isinstance(data, dict) else data
            if not isinstance(records, list):
                raise ValueError("queue file 'jobs' is not a list")
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "JobQueue: garbage queue file %s (%s); starting empty", self.path, exc
            )
            return
        for rec in records:
            try:
                job = Job.from_dict(rec)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("JobQueue: skipping malformed job record (%s)", exc)
                continue
            # broker that owned a running job died -> re-run it
            if job.status == "running":
                job.status = "queued"
                job.started_at = None
                job.run_id = None
            self._jobs[job.id] = job

    # --------------------------------------------------------------- persist
    def _persist(self) -> None:
        """Atomic write: dump to a tmp sibling then os.replace into place."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"jobs": [j.to_dict() for j in self._ordered()]}
        tmp = self.path.with_name(self.path.name + f".tmp.{os.getpid()}.{id(self)}")
        text = json.dumps(payload, indent=2)
        with open(tmp, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    # ----------------------------------------------------------------- order
    def _ordered(self) -> "list[Job]":
        """Insertion order is FIFO because ids are time-sortable; sort by id."""
        return [self._jobs[k] for k in sorted(self._jobs.keys())]

    # --------------------------------------------------------------- mutators
    def submit(self, instruction: str, kwargs: dict, pools: list) -> Job:
        job = Job(
            id=_new_job_id(),
            instruction=instruction,
            kwargs=dict(kwargs or {}),
            status="queued",
            submitted_at=time.time(),
            pools=list(pools or []),
        )
        # extremely unlikely id collision in the same ms; regenerate if so
        while job.id in self._jobs:
            job.id = _new_job_id()
        self._jobs[job.id] = job
        self._persist()
        return job

    def next_runnable(self, active_pools) -> Optional[Job]:
        """FIFO-oldest queued job whose pools are disjoint from active_pools.

        Returns None if the queue has no queued job, or every queued job's
        pools intersect ``active_pools`` (caller decides to wait).
        """
        active = set(active_pools or ())
        for job in self._ordered():
            if job.status != "queued":
                continue
            if active.isdisjoint(set(job.pools)):
                return job
        return None

    def oldest_queued(self) -> Optional[Job]:
        """FIFO-oldest queued job regardless of pools (anti-deadlock escape)."""
        for job in self._ordered():
            if job.status == "queued":
                return job
        return None

    def mark_running(self, job_id: str, run_id) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = "running"
        job.started_at = time.time()
        job.run_id = run_id
        self._persist()

    def mark_done(self, job_id: str, result: dict) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = "done"
        job.finished_at = time.time()
        job.result = result
        if isinstance(result, dict) and result.get("run_id") and not job.run_id:
            job.run_id = result.get("run_id")
        self._persist()

    def mark_failed(self, job_id: str, error: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = "failed"
        job.finished_at = time.time()
        job.result = {"success": False, "error": str(error), "run_id": job.run_id}
        self._persist()

    def cancel(self, job_id: str) -> bool:
        """Cancel only if still queued. Returns whether it was canceled."""
        job = self._jobs.get(job_id)
        if job is None or job.status != "queued":
            return False
        job.status = "canceled"
        job.finished_at = time.time()
        self._persist()
        return True

    # ----------------------------------------------------------------- reads
    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def snapshot(self) -> list:
        """All jobs as dicts, queued+running first, then terminal; FIFO within."""

        def sort_key(job: Job):
            group = 0 if job.status in _ACTIVE_STATUSES else 1
            return (group, job.id)

        return [j.to_dict() for j in sorted(self._jobs.values(), key=sort_key)]

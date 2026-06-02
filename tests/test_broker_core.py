"""C2 tests: Broker drain loop, cap=2, pool coordination (hermetic).

No real worker CLIs, no network, no sockets — a fake async ``dispatch_fn``
records observed concurrency and per-call ordering. Temp queue paths only.
The anti-deadlock pool wait is kept tiny via the ``pool_wait`` constructor arg
so colliding-pool tests run fast.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from harness.broker import Broker, _normalize_result
from harness.job_queue import JobQueue


@pytest.fixture()
def qpath(tmp_path):
    return tmp_path / "queue.json"


class ConcurrencyRecorder:
    """A fake dispatch_fn that tracks live concurrency + the order jobs ran."""

    def __init__(self, *, hold=0.02):
        self.live = 0
        self.max_live = 0
        self.hold = hold
        self.started_order: list[str] = []
        self.overlaps: list[tuple[str, ...]] = []  # snapshot of running set per start
        self._running: set[str] = set()

    async def __call__(self, **kwargs):
        ident = kwargs.get("instruction", "?")
        self.live += 1
        self.max_live = max(self.max_live, self.live)
        self.started_order.append(ident)
        self._running.add(ident)
        self.overlaps.append(tuple(sorted(self._running)))
        try:
            await asyncio.sleep(self.hold)
            return {"success": True, "run_id": f"run-{ident}", "error": None}
        finally:
            self._running.discard(ident)
            self.live -= 1


async def _run_until_drained(broker: Broker, queue: JobQueue, *, timeout=5.0):
    """Run the broker until every job is terminal, then stop it."""
    task = asyncio.ensure_future(broker.run())

    async def _wait_terminal():
        while True:
            statuses = {j["status"] for j in queue.snapshot()}
            if statuses and statuses <= {"done", "failed", "canceled"}:
                return
            await asyncio.sleep(0.005)

    try:
        await asyncio.wait_for(_wait_terminal(), timeout=timeout)
    finally:
        broker.stop()
        await asyncio.wait_for(task, timeout=timeout)


# --------------------------------------------------------------------------- C2

def test_normalize_result_handles_object_dict_and_none():
    class R:
        run_id = "r1"
        success = True
        error = None

    assert _normalize_result(R()) == {"success": True, "run_id": "r1", "error": None}
    assert _normalize_result({"success": False, "run_id": "r2", "error": "boom"}) == {
        "success": False,
        "run_id": "r2",
        "error": "boom",
    }
    assert _normalize_result(None) == {"success": True, "run_id": None, "error": None}


def test_default_dispatch_fn_is_dispatch_async():
    # Resolved lazily; must point at the real dispatcher per the contract.
    from harness.dispatch import dispatch_async

    b = Broker(JobQueue("/tmp/does-not-matter.json"))
    assert b.dispatch_fn is dispatch_async


def test_cap_two_three_jobs_all_done(qpath):
    async def _go():
        q = JobQueue(qpath)
        rec = ConcurrencyRecorder(hold=0.05)
        b = Broker(q, dispatch_fn=rec, pool_wait=0.01)
        # Three distinct, non-colliding pools so only the cap bounds concurrency.
        q.submit("a", {"instruction": "a"}, ["codex"])
        q.submit("b", {"instruction": "b"}, ["agy"])
        q.submit("c", {"instruction": "c"}, ["grok"])
        await _run_until_drained(b, q)
        assert rec.max_live == 2, f"cap violated: observed {rec.max_live}"
        statuses = {j["status"] for j in q.snapshot()}
        assert statuses == {"done"}
        # All three actually ran.
        assert sorted(rec.started_order) == ["a", "b", "c"]

    asyncio.run(_go())


def test_run_ids_recorded_from_result(qpath):
    async def _go():
        q = JobQueue(qpath)
        rec = ConcurrencyRecorder(hold=0.0)
        b = Broker(q, dispatch_fn=rec, pool_wait=0.01)
        q.submit("only", {"instruction": "only"}, [])
        await _run_until_drained(b, q)
        job = q.snapshot()[0]
        assert job["status"] == "done"
        assert job["run_id"] == "run-only"
        assert job["result"]["run_id"] == "run-only"

    asyncio.run(_go())


def test_pool_colliding_jobs_do_not_run_together_until_wait(qpath):
    async def _go():
        q = JobQueue(qpath)
        # Long hold so the two colliding jobs would overlap IF scheduled together.
        rec = ConcurrencyRecorder(hold=0.15)
        pool_wait = 0.25
        b = Broker(q, dispatch_fn=rec, pool_wait=pool_wait)
        # Both want the SAME pool 'codex' -> must NOT run concurrently while the
        # first holds it; the second only starts after the anti-deadlock wait.
        q.submit("p1", {"instruction": "p1"}, ["codex"])
        q.submit("p2", {"instruction": "p2"}, ["codex"])
        await _run_until_drained(b, q, timeout=8.0)
        # Both completed.
        assert {j["status"] for j in q.snapshot()} == {"done"}
        # They were never in flight at the same time (no overlap snapshot had both).
        for snap in rec.overlaps:
            assert not ({"p1", "p2"} <= set(snap)), f"colliding jobs overlapped: {snap}"
        # p1 ran first; p2 only after the wait.
        assert rec.started_order == ["p1", "p2"]

    asyncio.run(_go())


def test_anti_deadlock_runs_oldest_after_wait(qpath):
    """With one slot held by a long colliding job, the second colliding job must
    still eventually run (after pool_wait) rather than starve forever."""
    async def _go():
        q = JobQueue(qpath)
        rec = ConcurrencyRecorder(hold=0.2)
        pool_wait = 0.1
        b = Broker(q, dispatch_fn=rec, pool_wait=pool_wait)
        q.submit("x1", {"instruction": "x1"}, ["agy"])
        q.submit("x2", {"instruction": "x2"}, ["agy"])  # collides
        await _run_until_drained(b, q, timeout=8.0)
        assert {j["status"] for j in q.snapshot()} == {"done"}
        assert sorted(rec.started_order) == ["x1", "x2"]

    asyncio.run(_go())


def test_status_persisted_at_each_transition(qpath):
    """The queue.json file reflects running and done states (atomic persistence)."""
    async def _go():
        q = JobQueue(qpath)
        seen_running = asyncio.Event()

        async def dispatch_fn(**kwargs):
            # While we're in here the on-disk status for this job must be running.
            data = json.loads(qpath.read_text())
            statuses = {j["status"] for j in data["jobs"]}
            assert "running" in statuses
            seen_running.set()
            return {"success": True, "run_id": "rid", "error": None}

        b = Broker(q, dispatch_fn=dispatch_fn, pool_wait=0.01)
        q.submit("s", {"instruction": "s"}, [])
        await _run_until_drained(b, q)
        assert seen_running.is_set()
        # Final on-disk state is done.
        data = json.loads(qpath.read_text())
        assert {j["status"] for j in data["jobs"]} == {"done"}

    asyncio.run(_go())


def test_dispatch_exception_marks_failed(qpath):
    async def _go():
        q = JobQueue(qpath)

        async def boom(**kwargs):
            raise RuntimeError("kaboom")

        b = Broker(q, dispatch_fn=boom, pool_wait=0.01)
        q.submit("f", {"instruction": "f"}, [])
        await _run_until_drained(b, q)
        job = q.snapshot()[0]
        assert job["status"] == "failed"
        assert "kaboom" in (job["result"]["error"] or "")

    asyncio.run(_go())


def test_unsuccessful_result_marks_failed(qpath):
    async def _go():
        q = JobQueue(qpath)

        async def sad(**kwargs):
            return {"success": False, "run_id": "r", "error": "verifier red"}

        b = Broker(q, dispatch_fn=sad, pool_wait=0.01)
        q.submit("u", {"instruction": "u"}, [])
        await _run_until_drained(b, q)
        job = q.snapshot()[0]
        assert job["status"] == "failed"
        assert "verifier red" in (job["result"]["error"] or "")

    asyncio.run(_go())


def test_pool_wait_env_default(monkeypatch):
    monkeypatch.setenv("AGY_BROKER_POOL_WAIT", "2.5")
    b = Broker(JobQueue("/tmp/x.json"))
    assert b._pool_wait == 2.5
    monkeypatch.setenv("AGY_BROKER_POOL_WAIT", "not-a-number")
    b2 = Broker(JobQueue("/tmp/x.json"))
    assert b2._pool_wait == 5.0  # falls back to default on garbage

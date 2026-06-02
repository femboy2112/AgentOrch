"""C5 — in-process end-to-end integration for the singleton broker layer.

Hermetic by contract: a fake ``dispatch_fn`` (no real worker CLI, no network),
tmp socket/queue/pid paths under ``tmp_path``, no real sockets to the internet.
This exercises the full plane together — :class:`JobQueue` persistence,
:class:`Broker` drain loop + cap, the :class:`BrokerServer` IPC layer, and the
:mod:`harness.broker_client` over a unix socket — then a simulated broker
restart that recovers persisted state and re-queues any interrupted job.

What it asserts:
  * submit 3 jobs over IPC -> all reach ``done``;
  * the concurrency cap of 2 is honored (observed max-in-flight <= 2, and with
    enough disjoint-pool work it actually reaches 2);
  * ``queue.json`` persists across a simulated broker restart — a fresh
    ``JobQueue`` built from the same path recovers state, and an interrupted
    ``running`` job is reset to ``queued`` so it re-runs.
"""
from __future__ import annotations

import asyncio

from harness import broker_client
from harness.broker import Broker, BrokerServer
from harness.job_queue import JobQueue


# --------------------------------------------------------------------- fixtures
def _paths(tmp_path):
    return {
        "queue": tmp_path / "queue.json",
        "sock": tmp_path / "broker.sock",
        "pid": tmp_path / "broker.pid",
    }


class _ConcurrencyTracker:
    """A fake dispatch_fn that records peak concurrency and overlaps jobs.

    Each invocation increments a live counter, awaits a small barrier so that
    independent jobs actually run at the same time (otherwise a too-fast fake
    could finish before the next even starts, masking the cap), records the peak
    live count, then returns a success result shaped like a DispatchResult.
    """

    def __init__(self) -> None:
        self.live = 0
        self.max_live = 0
        self.calls = 0
        self.instructions: list[str] = []
        # Gate the broker holds jobs behind until released, so >=2 jobs are
        # concurrently in flight when we measure the peak.
        self._hold = asyncio.Event()

    def release(self) -> None:
        self._hold.set()

    async def __call__(self, **kwargs):
        self.calls += 1
        self.instructions.append(kwargs.get("instruction", ""))
        self.live += 1
        self.max_live = max(self.max_live, self.live)
        try:
            # Stay in flight until released (or a short ceiling) so concurrency
            # is observable; never hangs the test.
            try:
                await asyncio.wait_for(self._hold.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            return {"success": True, "run_id": f"RID-{self.calls}", "error": None}
        finally:
            self.live -= 1


async def _submit(sock, instruction, pools):
    return await asyncio.to_thread(
        lambda: broker_client.submit(
            instruction, {"instruction": instruction, "mode": "direct"}, pools, sock_path=str(sock)
        )
    )


async def _wait_all_terminal(queue: JobQueue, *, deadline_s=8.0):
    loop = asyncio.get_event_loop()
    start = loop.time()
    while loop.time() - start < deadline_s:
        await asyncio.sleep(0.02)
        queue.load()  # re-read persisted state (cheap, hermetic)
        snap = queue.snapshot()
        if snap and all(j["status"] in ("done", "failed", "canceled") for j in snap):
            return snap
    queue.load()
    return queue.snapshot()


def test_end_to_end_cap_honored_all_done(tmp_path):
    """Submit 3 jobs over IPC; cap=2 honored; all reach done."""
    paths = _paths(tmp_path)
    tracker = _ConcurrencyTracker()

    async def _go():
        queue = JobQueue(paths["queue"])
        queue.load()
        broker = Broker(queue, dispatch_fn=tracker, cap=2)
        srv = BrokerServer(broker, sock_path=paths["sock"], pid_path=paths["pid"])
        await srv.start()
        drain = asyncio.ensure_future(broker.run())
        try:
            # Three jobs. Two have fully disjoint pools so the broker can run
            # them together (reaching the cap of 2); the third shares a pool
            # with the first so it waits for a free slot rather than exceeding
            # the cap.
            j1 = await _submit(paths["sock"], "job-1 codex", ["codex"])
            j2 = await _submit(paths["sock"], "job-2 agy", ["agy"])
            j3 = await _submit(paths["sock"], "job-3 grok", ["grok"])
            assert j1 and j2 and j3

            # Let the broker pick up two and reach peak concurrency, then release
            # everything so all three complete.
            for _ in range(100):
                await asyncio.sleep(0.01)
                if tracker.max_live >= 2:
                    break
            tracker.release()

            snap = await _wait_all_terminal(queue)
            # All three completed.
            assert len(snap) == 3
            assert all(j["status"] == "done" for j in snap), snap
            ids = {j["id"] for j in snap}
            assert ids == {j1, j2, j3}
            assert all(j["result"]["success"] for j in snap)

            # Cap honored: never more than 2 dispatches in flight at once, and
            # with two disjoint-pool jobs available the cap was actually reached.
            assert tracker.max_live <= 2, f"cap exceeded: {tracker.max_live}"
            assert tracker.max_live == 2, f"expected to reach cap of 2, got {tracker.max_live}"
            assert tracker.calls == 3
        finally:
            broker.stop()
            await asyncio.wait_for(drain, timeout=5.0)
            await srv.close()

    asyncio.run(_go())


def test_queue_persists_and_recovers_across_broker_restart(tmp_path):
    """queue.json survives a simulated restart; an interrupted running job
    is re-queued and then completes under a fresh broker."""
    paths = _paths(tmp_path)

    # --- First broker generation: submit jobs, let some finish, then simulate a
    #     crash mid-flight by persisting one job in the 'running' state. ---
    async def _gen1():
        queue = JobQueue(paths["queue"])
        queue.load()
        tracker = _ConcurrencyTracker()
        broker = Broker(queue, dispatch_fn=tracker, cap=2)
        srv = BrokerServer(broker, sock_path=paths["sock"], pid_path=paths["pid"])
        await srv.start()
        drain = asyncio.ensure_future(broker.run())
        try:
            j_done = await _submit(paths["sock"], "finish-me", ["codex"])
            j_crash = await _submit(paths["sock"], "interrupt-me", ["agy"])
            tracker.release()
            # Wait for at least the first job to reach done.
            for _ in range(200):
                await asyncio.sleep(0.01)
                queue.load()
                jd = queue.get(j_done)
                if jd is not None and jd.status == "done":
                    break
        finally:
            broker.stop()
            await asyncio.wait_for(drain, timeout=5.0)
            await srv.close()
        return j_done, j_crash

    j_done, j_crash = asyncio.run(_gen1())

    # Forcibly simulate a broker that died WHILE j_crash was running: mutate the
    # persisted queue file so j_crash.status == 'running' (no broker owns it).
    pre = JobQueue(paths["queue"])
    pre.load()
    assert pre.get(j_done).status == "done"
    pre.mark_running(j_crash, run_id="RID-CRASHED")  # persists 'running' to disk
    assert pre.get(j_crash).status == "running"

    # --- Reload from the SAME path: this is what a restarted broker does. The
    #     interrupted 'running' job must be reset to 'queued' on load. ---
    recovered = JobQueue(paths["queue"])
    recovered.load()
    assert recovered.get(j_done).status == "done", "done job must survive restart"
    crash_job = recovered.get(j_crash)
    assert crash_job is not None
    assert crash_job.status == "queued", "interrupted running job must re-queue on load"
    assert crash_job.run_id is None, "re-queued job clears its stale run_id"

    # --- Second broker generation over the SAME paths picks up the re-queued
    #     job and drives it to done. ---
    async def _gen2():
        queue = JobQueue(paths["queue"])
        queue.load()  # running -> queued reset applied again here
        tracker2 = _ConcurrencyTracker()
        tracker2.release()  # let it complete immediately
        broker = Broker(queue, dispatch_fn=tracker2, cap=2)
        srv = BrokerServer(broker, sock_path=paths["sock"], pid_path=paths["pid"])
        await srv.start()
        drain = asyncio.ensure_future(broker.run())
        try:
            snap = await _wait_all_terminal(queue)
            assert all(j["status"] == "done" for j in snap), snap
            # The crashed job actually re-ran under the new broker.
            assert "interrupt-me" in tracker2.instructions
        finally:
            broker.stop()
            await asyncio.wait_for(drain, timeout=5.0)
            await srv.close()

    asyncio.run(_gen2())

"""C3 tests — IPC unix-socket server + client + singleton guard.

Hermetic: a fake ``dispatch_fn`` (never invoked here — these tests exercise the
IPC plane only), tmp socket/queue/pid paths under ``tmp_path``, no real worker
calls, no network. Every op round-trips; the singleton guard refuses a second
bind while the first answers ``ping``; a stale socket file is reclaimed.
"""
from __future__ import annotations

import asyncio
import os
import socket

import pytest

from harness import broker_client
from harness.broker import (
    Broker,
    BrokerAlreadyRunning,
    BrokerServer,
    _probe_ping,
)
from harness.job_queue import JobQueue


# --------------------------------------------------------------------- fixtures
@pytest.fixture()
def paths(tmp_path):
    return {
        "queue": tmp_path / "queue.json",
        "sock": tmp_path / "broker.sock",
        "pid": tmp_path / "broker.pid",
    }


async def _noop_dispatch(**kwargs):  # never invoked by these tests
    return {"success": True, "run_id": "RID", "error": None}


def _make_server(paths, *, cap=2):
    queue = JobQueue(paths["queue"])
    queue.load()
    broker = Broker(queue, dispatch_fn=_noop_dispatch, cap=cap)
    return BrokerServer(broker, sock_path=paths["sock"], pid_path=paths["pid"])


def _raw_request(sock_path, req: dict, timeout=2.0) -> dict:
    """Raw line-delimited JSON round-trip, bypassing broker_client wrappers.

    Synchronous/blocking — call via :func:`_client_request` (a thread) when the
    server shares this test's event loop, so the loop stays free to accept.
    """
    import json

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(sock_path))
        s.sendall(json.dumps(req).encode("utf-8") + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
    finally:
        s.close()


async def _client_request(sock_path, req: dict, timeout=2.0) -> dict:
    """Run the blocking round-trip off the event loop so the server can accept."""
    return await asyncio.to_thread(_raw_request, sock_path, req, timeout)


# ------------------------------------------------------------------ op round-trips
def test_ping_roundtrip_reports_pid_cap_counts(paths):
    async def _go():
        srv = _make_server(paths, cap=2)
        await srv.start()
        try:
            resp = await _client_request(paths["sock"], {"op": "ping"})
            assert resp["ok"] is True
            assert resp["pid"] == os.getpid()
            assert resp["cap"] == 2
            assert resp["running"] == 0
            assert resp["queued"] == 0
        finally:
            await srv.close()

    asyncio.run(_go())


def test_submit_list_status_cancel_roundtrip(paths):
    async def _go():
        srv = _make_server(paths)
        await srv.start()
        try:
            # submit
            sub = await _client_request(
                paths["sock"],
                {
                    "op": "submit",
                    "instruction": "build a thing",
                    "kwargs": {"instruction": "build a thing", "mode": "direct"},
                    "pools": ["codex", "agy"],
                },
            )
            assert sub["ok"] is True
            jid = sub["job_id"]
            assert isinstance(jid, str) and jid

            # list shows it
            lst = await _client_request(paths["sock"], {"op": "list"})
            assert lst["ok"] is True
            ids = [j["id"] for j in lst["jobs"]]
            assert jid in ids
            job_rec = next(j for j in lst["jobs"] if j["id"] == jid)
            assert job_rec["pools"] == ["codex", "agy"]

            # status returns the job
            st = await _client_request(paths["sock"], {"op": "status", "job_id": jid})
            assert st["ok"] is True
            assert st["job"] is not None
            assert st["job"]["id"] == jid
            assert st["job"]["instruction"] == "build a thing"

            # status for unknown id -> null
            st_unknown = await _client_request(paths["sock"], {"op": "status", "job_id": "nope"})
            assert st_unknown["ok"] is True
            assert st_unknown["job"] is None

            # cancel a still-queued job
            can = await _client_request(paths["sock"], {"op": "cancel", "job_id": jid})
            assert can["ok"] is True
            assert can["canceled"] is True

            # second cancel is a no-op
            can2 = await _client_request(paths["sock"], {"op": "cancel", "job_id": jid})
            assert can2["ok"] is True
            assert can2["canceled"] is False
        finally:
            await srv.close()

    asyncio.run(_go())


def test_unknown_op_and_bad_json_are_handled(paths):
    async def _go():
        srv = _make_server(paths)
        await srv.start()
        try:
            bad = await _client_request(paths["sock"], {"op": "frobnicate"})
            assert bad["ok"] is False
            assert "unknown op" in bad["error"]

            # raw garbage (not JSON) still gets a structured error
            def _send_garbage():
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(2.0)
                s.connect(str(paths["sock"]))
                try:
                    s.sendall(b"this is not json\n")
                    buf = b""
                    while b"\n" not in buf:
                        chunk = s.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                    import json

                    return json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
                finally:
                    s.close()

            resp = await asyncio.to_thread(_send_garbage)
            assert resp["ok"] is False
            assert "bad request" in resp["error"]
        finally:
            await srv.close()

    asyncio.run(_go())


# ----------------------------------------------------------------- broker_client
def test_broker_client_full_surface(paths):
    async def _go():
        srv = _make_server(paths, cap=3)
        await srv.start()
        try:
            sp = str(paths["sock"])
            assert await asyncio.to_thread(broker_client.is_running, sp) is True

            pong = await asyncio.to_thread(lambda: broker_client.ping(sock_path=sp))
            assert pong is not None and pong["cap"] == 3

            jid = await asyncio.to_thread(
                lambda: broker_client.submit("do x", {"instruction": "do x"}, ["grok"], sock_path=sp)
            )
            assert isinstance(jid, str)

            jobs = await asyncio.to_thread(lambda: broker_client.list_jobs(sock_path=sp))
            assert any(j["id"] == jid for j in jobs)

            job = await asyncio.to_thread(lambda: broker_client.status(jid, sock_path=sp))
            assert job is not None and job["pools"] == ["grok"]
            assert await asyncio.to_thread(lambda: broker_client.status("missing", sock_path=sp)) is None

            assert await asyncio.to_thread(lambda: broker_client.cancel(jid, sock_path=sp)) is True
            assert await asyncio.to_thread(lambda: broker_client.cancel(jid, sock_path=sp)) is False
        finally:
            await srv.close()

    asyncio.run(_go())


def test_is_running_false_when_no_socket(tmp_path):
    # No server bound -> never hangs, returns False.
    assert broker_client.is_running(tmp_path / "absent.sock") is False
    assert broker_client.ping(sock_path=tmp_path / "absent.sock") is None


# --------------------------------------------------------------- singleton guard
def test_singleton_guard_refuses_second_bind_while_first_alive(paths):
    async def _go():
        first = _make_server(paths)
        await first.start()
        try:
            # A live broker answers ping -> a second bind must refuse.
            assert await asyncio.to_thread(broker_client.is_running, str(paths["sock"])) is True
            second = _make_server(paths)
            with pytest.raises(BrokerAlreadyRunning) as ei:
                await second.start()
            assert ei.value.pid == os.getpid()
        finally:
            await first.close()

    asyncio.run(_go())


def test_stale_socket_is_reclaimed(paths):
    async def _go():
        # Create a stale socket FILE that nothing is listening on: bind a raw
        # unix socket, close the listener but leave the path on disk.
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(str(paths["sock"]))
        stale.close()  # path remains, but no one accepts -> ping fails
        assert os.path.exists(paths["sock"])
        assert _probe_ping(paths["sock"]) is None  # confirmed stale

        # The server must unlink the stale file and bind successfully.
        srv = _make_server(paths)
        await srv.start()
        try:
            assert await asyncio.to_thread(broker_client.is_running, str(paths["sock"])) is True
        finally:
            await srv.close()

    asyncio.run(_go())


def test_cleanup_removes_socket_and_pidfile(paths):
    async def _go():
        srv = _make_server(paths)
        await srv.start()
        assert os.path.exists(paths["sock"])
        assert os.path.exists(paths["pid"])
        with open(paths["pid"]) as fh:
            assert fh.read().strip() == str(os.getpid())
        await srv.close()
        assert not os.path.exists(paths["sock"])
        assert not os.path.exists(paths["pid"])

    asyncio.run(_go())


def test_serve_forever_drains_and_cleans_up(paths):
    """End-to-end: serve_forever runs the drain loop; stop() ends it and cleans up."""
    async def _go():
        srv = _make_server(paths)
        # Submit one job directly to the queue before serving so the drain loop
        # has work; the fake dispatch returns success immediately.
        srv.queue.submit("x", {"instruction": "x"}, ["codex"])
        serve = asyncio.ensure_future(srv.serve_forever())
        # Let the loop bind + drain.
        for _ in range(200):
            await asyncio.sleep(0.01)
            jobs = srv.queue.snapshot()
            if jobs and all(j["status"] in ("done", "failed", "canceled") for j in jobs):
                break
        srv.broker.stop()
        await asyncio.wait_for(serve, timeout=5.0)
        assert not os.path.exists(paths["sock"])
        assert all(j["status"] == "done" for j in srv.queue.snapshot())

    asyncio.run(_go())

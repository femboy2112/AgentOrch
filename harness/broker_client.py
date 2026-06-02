"""C3: thin, stdlib-only unix-socket client for the singleton broker.

The CLI (C4) uses this to talk to a running :class:`~harness.broker.BrokerServer`
over ``runs/.queue/broker.sock``. Every call is a single line-delimited JSON
request/response round-trip on a fresh connection. All calls use a short connect
timeout and **never hang**: any socket/parse error degrades to a safe default
(``is_running`` -> ``False``; the others raise :class:`BrokerError` only on an
explicit protocol error, returning ``None``/``[]`` when the broker is simply
unreachable).

stdlib-only by contract — no third-party deps, no asyncio.
"""
from __future__ import annotations

import json
import os
import socket
from typing import Optional

from harness.broker import SOCK_PATH

# Default short timeout for a round-trip. Auto-detection in the CLI must be cheap
# and never block a `harness do` invocation, so keep this tight.
_DEFAULT_TIMEOUT = 2.0
_PING_TIMEOUT = 0.5


class BrokerError(RuntimeError):
    """The broker answered but reported an error, or the protocol was violated."""


def _resolve(sock_path) -> str:
    return str(sock_path) if sock_path is not None else str(SOCK_PATH)


def _request(op: str, payload: Optional[dict] = None, *, sock_path=None, timeout: float = _DEFAULT_TIMEOUT) -> dict:
    """Send one op and return the parsed response dict.

    Raises :class:`OSError` if the broker is unreachable (caller decides whether
    that is fatal) and :class:`BrokerError` on a malformed reply.
    """
    path = _resolve(sock_path)
    req = {"op": op}
    if payload:
        req.update(payload)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(path)
        s.sendall(json.dumps(req).encode("utf-8") + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        if not buf:
            raise BrokerError("broker closed connection without replying")
        line = buf.split(b"\n", 1)[0]
        try:
            resp = json.loads(line.decode("utf-8"))
        except (ValueError, json.JSONDecodeError) as exc:
            raise BrokerError(f"malformed broker reply: {exc}") from exc
        if not isinstance(resp, dict):
            raise BrokerError("broker reply was not an object")
        return resp
    finally:
        try:
            s.close()
        except OSError:
            pass


def is_running(sock_path=None) -> bool:
    """Cheap liveness check: ``True`` iff a broker answers ``ping`` with ok.

    Never raises, never hangs longer than the ping timeout. Any error (no socket,
    connection refused, stale file, timeout, garbage reply) -> ``False``.
    """
    path = _resolve(sock_path)
    if not os.path.exists(path):
        return False
    try:
        resp = _request("ping", sock_path=path, timeout=_PING_TIMEOUT)
    except (OSError, BrokerError):
        return False
    return bool(resp.get("ok"))


def ping(sock_path=None) -> Optional[dict]:
    """Return the broker's ping payload (pid/cap/running/queued) or ``None``."""
    path = _resolve(sock_path)
    if not os.path.exists(path):
        return None
    try:
        resp = _request("ping", sock_path=path, timeout=_PING_TIMEOUT)
    except (OSError, BrokerError):
        return None
    return resp if resp.get("ok") else None


def submit(instruction: str, kwargs: dict, pools, *, sock_path=None, timeout: float = _DEFAULT_TIMEOUT) -> str:
    """Submit a job; return its broker-assigned id. Raises on failure."""
    resp = _request(
        "submit",
        {"instruction": instruction, "kwargs": dict(kwargs or {}), "pools": list(pools or [])},
        sock_path=sock_path,
        timeout=timeout,
    )
    if not resp.get("ok"):
        raise BrokerError(resp.get("error") or "submit failed")
    return resp["job_id"]


def list_jobs(*, sock_path=None, timeout: float = _DEFAULT_TIMEOUT) -> list:
    """Return the broker's queue snapshot (list of job dicts)."""
    resp = _request("list", sock_path=sock_path, timeout=timeout)
    if not resp.get("ok"):
        raise BrokerError(resp.get("error") or "list failed")
    return resp.get("jobs") or []


def status(job_id: str, *, sock_path=None, timeout: float = _DEFAULT_TIMEOUT) -> Optional[dict]:
    """Return the job dict for ``job_id`` or ``None`` if unknown."""
    resp = _request("status", {"job_id": job_id}, sock_path=sock_path, timeout=timeout)
    if not resp.get("ok"):
        raise BrokerError(resp.get("error") or "status failed")
    return resp.get("job")


def cancel(job_id: str, *, sock_path=None, timeout: float = _DEFAULT_TIMEOUT) -> bool:
    """Cancel a queued job; return whether it was canceled."""
    resp = _request("cancel", {"job_id": job_id}, sock_path=sock_path, timeout=timeout)
    if not resp.get("ok"):
        raise BrokerError(resp.get("error") or "cancel failed")
    return bool(resp.get("canceled"))

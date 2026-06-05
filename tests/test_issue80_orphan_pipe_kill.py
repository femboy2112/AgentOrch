"""Issue #80: a worker-spawned `make check` (-> `pytest -n auto`) ran in its OWN
session, so it escaped `killpg(worker_pgid)`, orphaned past the worker's exit, and
#66 merely ABANDONED the drain instead of killing it — it then ran ~43 min unbounded
and OOM-killed an unrelated job.

`AgentInstance._kill_pipe_fd_holders` finds the actual fd-holder by the SHARED pipe
inode (regardless of session/group escaping) and SIGKILLs it. These tests spawn a
real orphan in its own session holding the inherited pipe write-end and prove it is
terminated — and that an UNRELATED process is never touched.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from agy_orchestrator.core.agent import AgentInstance

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc"), reason="needs /proc (Linux)")


def _inode_of_readfd(read_fd: int) -> str:
    return os.readlink(f"/proc/self/fd/{read_fd}")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        # Reap if it's our child so it doesn't linger as a zombie (still 'alive').
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def test_kills_orphan_holding_pipe_in_its_own_session():
    r, w = os.pipe()
    # A child in its OWN session (start_new_session) — exactly the escaped-group
    # orphan #80 describes — inheriting the write-end and sleeping on it.
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        pass_fds=(w,), start_new_session=True,
    )
    try:
        os.close(w)  # only the orphan now holds the write-end
        inode = _inode_of_readfd(r)
        # Sanity: the orphan really is in a different process group (escaped).
        assert os.getpgid(child.pid) != os.getpgid(os.getpid())

        n = AgentInstance._kill_pipe_fd_holders([inode], reason="test")
        assert n >= 1, "must signal the orphan holding the worker pipe fd"
        assert _wait_dead(child.pid), "orphan must be SIGKILLed, not left running"
    finally:
        try:
            os.close(r)
        except OSError:
            pass
        if _alive(child.pid):
            try:
                os.kill(child.pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            child.wait(timeout=2)
        except Exception:
            pass


def test_does_not_kill_unrelated_process():
    # A process that does NOT hold our pipe inode must be untouched.
    r, w = os.pipe()
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
    try:
        inode = _inode_of_readfd(r)
        AgentInstance._kill_pipe_fd_holders([inode], reason="test")
        assert _alive(bystander.pid), "a process not holding our pipe must survive"
    finally:
        for fd in (r, w):
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            bystander.wait(timeout=4)
        except Exception:
            bystander.kill()


def test_noop_on_empty_or_bad_inodes():
    assert AgentInstance._kill_pipe_fd_holders([]) == 0
    assert AgentInstance._kill_pipe_fd_holders([None, ""]) == 0
    # A plausible-but-unheld inode matches nothing.
    assert AgentInstance._kill_pipe_fd_holders(["pipe:[999999999]"]) == 0


def test_pipe_inode_helper_reads_subprocess_stream():
    import asyncio

    async def _run():
        p = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(0.2)",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        inode = AgentInstance._pipe_inode(p.stdout)
        await p.wait()
        return inode

    inode = asyncio.run(_run())
    assert inode is not None and inode.startswith("pipe:[")

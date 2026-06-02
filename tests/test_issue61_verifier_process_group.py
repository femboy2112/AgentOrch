"""Issue #61: the verifier must contain its test subprocess in its own process
group and reap the WHOLE group on timeout / completion, so forked pool workers
(multiprocessing / pytest-xdist) are never orphaned to PID 1.

Two layers:
  * a fast hermetic check that the spawn requests a new session, and that the
    group-kill / pgid helpers behave; and
  * a real integration check that a grandchild spawned by the test command does
    NOT survive a verifier timeout.
"""
from __future__ import annotations

import asyncio
import os
import signal
import sys
import time

import pytest

from agy_orchestrator.execution import verifier as verifier_mod
from agy_orchestrator.execution.verifier import (
    QualityVerifier,
    _kill_process_group,
    _safe_getpgid,
)


# --------------------------------------------------------------------------- #
# Fast hermetic checks
# --------------------------------------------------------------------------- #
@pytest.mark.not_slow
def test_spawn_requests_new_session(monkeypatch, tmp_path):
    """Both spawn paths must pass start_new_session=True so there is a group to
    signal. We intercept the shell spawn (the no-mem-cap default path)."""
    captured = {}

    class _FakeProc:
        pid = 424242
        returncode = 0

        async def communicate(self):
            return (b"", b"")

        async def wait(self):
            return 0

        def kill(self):
            pass

    async def _fake_shell(cmd, **kwargs):
        captured.update(kwargs)
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _fake_shell)
    # _safe_getpgid would raise on the fake pid; stub it so verify() completes.
    monkeypatch.setattr(verifier_mod, "_safe_getpgid", lambda p: None)
    monkeypatch.setattr(verifier_mod, "_kill_process_group", lambda *a, **k: None)

    v = QualityVerifier(test_commands=["true"], timeout=5)
    res = asyncio.run(v.verify(str(tmp_path)))

    assert res.ok is True
    assert captured.get("start_new_session") is True


@pytest.mark.not_slow
def test_kill_process_group_swallows_missing_group():
    """A pgid whose group is already gone must not raise (ProcessLookupError)."""
    # Pick a pgid extremely unlikely to exist.
    _kill_process_group(999_999_999)  # should be a no-op, never raise


@pytest.mark.not_slow
def test_kill_process_group_falls_back_to_direct_child():
    """When pgid is None, fall back to killing the direct child process."""
    killed = {"n": 0}

    class _P:
        def kill(self):
            killed["n"] += 1

    _kill_process_group(None, fallback_process=_P())
    assert killed["n"] == 1


@pytest.mark.not_slow
def test_safe_getpgid_none_on_bad_pid():
    class _P:
        pid = 999_999_999

    assert _safe_getpgid(_P()) is None


# --------------------------------------------------------------------------- #
# Real integration: a grandchild must not outlive a verifier timeout
# --------------------------------------------------------------------------- #
def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.not_slow
@pytest.mark.skipif(
    not hasattr(os, "killpg") or sys.platform == "win32",
    reason="POSIX process-group semantics required",
)
def test_timeout_reaps_grandchild(tmp_path):
    """The test command backgrounds a long-lived grandchild (in the same process
    group) then blocks past the timeout. After the verifier times out, the
    grandchild must be dead — proving the whole group was killed, not just the
    direct child."""
    pidfile = tmp_path / "grandchild.pid"
    # `sleep 300 &` is a grandchild in the shell's process group; record its pid,
    # then the shell itself blocks long enough to trip the 2s timeout.
    cmd = f"sleep 300 & echo $! > {pidfile}; sleep 30"

    v = QualityVerifier(test_commands=[cmd], timeout=2)
    res = asyncio.run(v.verify(str(tmp_path)))

    assert res.timeout is True, f"expected a timeout result, got {res!r}"
    assert pidfile.exists(), "test command never recorded the grandchild pid"
    gc_pid = int(pidfile.read_text().strip())

    # Give the group-kill a brief moment to land.
    deadline = time.monotonic() + 5.0
    while _alive(gc_pid) and time.monotonic() < deadline:
        time.sleep(0.05)

    if _alive(gc_pid):
        # Clean up the orphan we just proved leaked, then fail loudly.
        try:
            os.kill(gc_pid, signal.SIGKILL)
        except Exception:
            pass
        pytest.fail(f"grandchild {gc_pid} survived the verifier timeout (orphaned)")

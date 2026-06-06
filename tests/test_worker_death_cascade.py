"""Worker death-cascade hardening: when the orchestrator is killed from OUTSIDE
(kill/pkill/kill -9), its spawned worker trees must not be orphaned.

Two independent layers, tested here:
  * PR_SET_PDEATHSIG in the worker preexec_fn — the kernel SIGKILLs the worker the
    instant the orchestrator dies, the ONLY thing that survives an un-catchable
    SIGKILL of the orchestrator (no Python __del__/atexit/handler can run then).
  * a SIGTERM/SIGHUP/atexit hook in the entrypoint that killpg's every LIVE worker
    group — covers the common `kill`/`pkill` (default SIGTERM) path with an orderly
    full-tree reap.

SAFETY: every test here only ever signals process groups it spawned itself in the
test (explicit pids). Nothing matches by name; nothing can touch another
orchestrator instance or unrelated process.
"""
import os
import signal
import sys
import textwrap
import time

import pytest

from agy_orchestrator.core import agent as agent_mod


# --------------------------------------------------------------------------- #
# Layer B: live-worker registry + kill_all_live_workers
# --------------------------------------------------------------------------- #

@pytest.fixture
def clean_registry():
    """Snapshot/restore the process-wide registry so a test can't leak into others."""
    saved = set(agent_mod._LIVE_WORKER_PGIDS)
    agent_mod._LIVE_WORKER_PGIDS.clear()
    try:
        yield
    finally:
        agent_mod._LIVE_WORKER_PGIDS.clear()
        agent_mod._LIVE_WORKER_PGIDS.update(saved)


def test_kill_all_live_workers_empty_is_noop(clean_registry):
    assert agent_mod.kill_all_live_workers() == 0


def test_register_unregister_roundtrip(clean_registry):
    agent_mod._register_worker_pgid(424242)
    assert 424242 in agent_mod._LIVE_WORKER_PGIDS
    agent_mod._unregister_worker_pgid(424242)
    assert 424242 not in agent_mod._LIVE_WORKER_PGIDS
    # None is ignored on both sides
    agent_mod._register_worker_pgid(None)
    agent_mod._unregister_worker_pgid(None)
    assert agent_mod._LIVE_WORKER_PGIDS == set()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="POSIX process groups")
def test_kill_all_live_workers_reaps_a_real_registered_group(clean_registry):
    """A real child in its OWN session, once registered, is reaped by the hook."""
    import subprocess

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    pgid = os.getpgid(proc.pid)
    agent_mod._register_worker_pgid(pgid)
    try:
        n = agent_mod.kill_all_live_workers(signal.SIGKILL)
        assert n == 1
        proc.wait(timeout=5)
        assert proc.returncode is not None  # it actually died
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


# --------------------------------------------------------------------------- #
# Layer B: entrypoint handler installation
# --------------------------------------------------------------------------- #

def test_install_handlers_is_idempotent_and_arms_sigterm():
    saved_term = signal.getsignal(signal.SIGTERM)
    saved_flag = agent_mod._CLEANUP_HANDLERS_INSTALLED
    try:
        agent_mod._CLEANUP_HANDLERS_INSTALLED = False
        agent_mod.install_process_cleanup_handlers()
        first = signal.getsignal(signal.SIGTERM)
        assert callable(first)
        assert first not in (signal.SIG_DFL, signal.SIG_IGN, saved_term)
        # second call is a no-op (idempotent) — handler unchanged
        agent_mod.install_process_cleanup_handlers()
        assert signal.getsignal(signal.SIGTERM) is first
    finally:
        signal.signal(signal.SIGTERM, saved_term)
        agent_mod._CLEANUP_HANDLERS_INSTALLED = saved_flag


def test_entrypoints_install_cleanup_handlers():
    """Both CLIs must wire the teardown so a killed dispatch reaps its workers."""
    import harness.cli as hcli
    import agy_orchestrator.cli as acli

    import inspect

    assert "install_process_cleanup_handlers" in inspect.getsource(hcli.main)
    assert "install_process_cleanup_handlers" in inspect.getsource(acli.main)


# --------------------------------------------------------------------------- #
# Layer A: PR_SET_PDEATHSIG preexec
# --------------------------------------------------------------------------- #

def test_build_preexec_respects_env(monkeypatch):
    if not sys.platform.startswith("linux") or agent_mod._PRCTL is None:
        monkeypatch.setenv("AGY_WORKER_PDEATHSIG", "1")
        assert agent_mod._build_worker_preexec() is None  # unavailable off-linux
        return
    monkeypatch.setenv("AGY_WORKER_PDEATHSIG", "1")
    assert callable(agent_mod._build_worker_preexec())
    monkeypatch.setenv("AGY_WORKER_PDEATHSIG", "0")
    assert agent_mod._build_worker_preexec() is None


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or agent_mod._PRCTL is None,
    reason="PR_SET_PDEATHSIG is Linux-only",
)
def test_pdeathsig_kills_worker_when_orchestrator_is_sigkilled(tmp_path):
    """End-to-end: a stand-in 'orchestrator' launches a sleeper with our preexec,
    then is SIGKILLed. The kernel must SIGKILL the sleeper too — the case no
    Python cleanup could ever handle. Uses only pids spawned here."""
    import subprocess

    marker = tmp_path / "grandchild.pid"
    # Stand-in orchestrator: spawn a long sleeper with the SAME preexec the real
    # worker spawn uses, record its pid, then sleep forever itself.
    script = textwrap.dedent(
        f"""
        import os, sys, time, subprocess
        sys.path.insert(0, {os.getcwd()!r})
        from agy_orchestrator.core.agent import _build_worker_preexec
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            preexec_fn=_build_worker_preexec(),
            start_new_session=True,
        )
        open({str(marker)!r}, "w").write(str(child.pid))
        time.sleep(120)
        """
    )
    parent = subprocess.Popen([sys.executable, "-c", script], env={**os.environ, "AGY_WORKER_PDEATHSIG": "1"})
    try:
        # Wait for the grandchild pid to be recorded.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.05)
        assert marker.exists(), "stand-in never recorded the worker pid"
        worker_pid = int(marker.read_text().strip())
        assert _pid_alive(worker_pid), "worker should be alive before the kill"

        # Hard-kill the stand-in orchestrator. No Python cleanup runs.
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=5)

        # The kernel must have delivered SIGKILL to the worker via PDEATHSIG.
        died = False
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if not _pid_alive(worker_pid):
                died = True
                break
            time.sleep(0.05)
        assert died, "worker was orphaned — PDEATHSIG did not fire"
    finally:
        # Belt-and-suspenders cleanup of anything we spawned.
        for pid in (parent.pid,):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            wp = int(marker.read_text().strip()) if marker.exists() else None
            if wp is not None and _pid_alive(wp):
                os.kill(wp, signal.SIGKILL)
        except (OSError, ValueError):
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

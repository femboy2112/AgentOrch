import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from agy_orchestrator.core import agent as agent_mod
from agy_orchestrator.core import reaper


@pytest.fixture
def clean_reaper_state():
    saved = (
        reaper._REAP_CGROUP_PATH,
        reaper._REAP_PIPE_WRITE_FD,
        reaper._REAPER_PROCESS,
        reaper._INSTALL_ATTEMPTED,
        reaper._DEGRADATION_LOGGED,
    )
    reaper._REAP_CGROUP_PATH = None
    reaper._REAP_PIPE_WRITE_FD = None
    reaper._REAPER_PROCESS = None
    reaper._INSTALL_ATTEMPTED = False
    reaper._DEGRADATION_LOGGED = False
    try:
        yield
    finally:
        if reaper._REAP_PIPE_WRITE_FD is not None:
            try:
                os.close(reaper._REAP_PIPE_WRITE_FD)
            except OSError:
                pass
        proc = reaper._REAPER_PROCESS
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        reaper._REAP_CGROUP_PATH = saved[0]
        reaper._REAP_PIPE_WRITE_FD = saved[1]
        reaper._REAPER_PROCESS = saved[2]
        reaper._INSTALL_ATTEMPTED = saved[3]
        reaper._DEGRADATION_LOGGED = saved[4]


def test_cgroup_reap_enabled_env(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_CGROUP_REAP", raising=False)
    assert reaper.cgroup_reap_enabled() is True
    monkeypatch.setenv("AGY_WORKER_CGROUP_REAP", "0")
    assert reaper.cgroup_reap_enabled() is False
    monkeypatch.setenv("AGY_WORKER_CGROUP_REAP", "false")
    assert reaper.cgroup_reap_enabled() is False


def test_cgroup_v2_available_false_when_controllers_absent(monkeypatch):
    monkeypatch.setattr(reaper.os.path, "exists", lambda path: False)
    assert reaper.cgroup_v2_available() is False


def test_join_reap_cgroup_in_child_no_path_is_noop(clean_reaper_state):
    reaper.join_reap_cgroup_in_child()
    assert reaper.reap_cgroup_path() is None


def test_reaper_noops_on_missing_cgroup(tmp_path):
    missing = tmp_path / "missing-cgroup"
    reaper._reap_cgroup(str(missing))


def test_install_hardkill_reaper_disabled_is_noop(monkeypatch, clean_reaper_state):
    monkeypatch.setenv("AGY_WORKER_CGROUP_REAP", "0")
    monkeypatch.setenv("AGY_WORKER_PDEATHSIG", "0")
    reaper.install_hardkill_reaper()
    assert reaper.reap_cgroup_path() is None
    assert reaper._REAPER_PROCESS is None
    assert reaper._REAP_PIPE_WRITE_FD is None
    assert agent_mod._build_worker_preexec() is None


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or not reaper.cgroup_v2_available(),
    reason="needs cgroup v2 delegation",
)
def test_hardkill_reaps_setsid_escaped_grandchild(tmp_path):
    marker = tmp_path / "grandchild.pid"
    worker_marker = tmp_path / "worker.pid"
    cgroup_marker = tmp_path / "cgroup.path"
    script = textwrap.dedent(
        f"""
        import os
        import subprocess
        import sys
        import time

        sys.path.insert(0, {os.getcwd()!r})
        from agy_orchestrator.core.agent import _build_worker_preexec
        from agy_orchestrator.core import reaper

        reaper.install_hardkill_reaper()
        open({str(cgroup_marker)!r}, "w").write(reaper.reap_cgroup_path() or "")

        worker_code = {worker_code(marker)!r}
        worker = subprocess.Popen(
            [sys.executable, "-c", worker_code],
            preexec_fn=_build_worker_preexec(),
            start_new_session=True,
        )
        open({str(worker_marker)!r}, "w").write(str(worker.pid))
        time.sleep(120)
        """
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
    try:
        grandchild_pid = _read_pid(marker)
        assert _pid_alive(grandchild_pid)

        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=5)

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if not _pid_alive(grandchild_pid):
                break
            time.sleep(0.05)
        assert not _pid_alive(grandchild_pid), "setsid grandchild survived orchestrator SIGKILL"
    finally:
        _kill_pid_if_alive(parent.pid)
        for path in (worker_marker, marker):
            if path.exists():
                try:
                    _kill_pid_if_alive(int(path.read_text().strip()))
                except ValueError:
                    pass
        if cgroup_marker.exists():
            cgroup_path = cgroup_marker.read_text().strip()
            if cgroup_path:
                reaper._reap_cgroup(cgroup_path)


def worker_code(marker) -> str:
    return textwrap.dedent(
        f"""
        import os
        import time

        pid = os.fork()
        if pid == 0:
            os.setsid()
            open({str(marker)!r}, "w").write(str(os.getpid()))
            time.sleep(120)
        else:
            time.sleep(120)
        """
    )


def _read_pid(path):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.exists():
            text = path.read_text().strip()
            if text:
                return int(text)
        time.sleep(0.05)
    raise AssertionError(f"pid file was not written: {path}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _kill_pid_if_alive(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass

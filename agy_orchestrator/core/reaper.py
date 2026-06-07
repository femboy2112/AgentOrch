"""Cgroup-v2 hard-kill reaper for worker subtrees.

The orchestrator's in-process cleanup cannot run after SIGKILL. This module
creates a cgroup umbrella for worker children and a detached helper that waits
for pipe EOF from the orchestrator, then kills everything in that cgroup.
"""
import argparse
import logging
import os
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)

_REAP_CGROUP_PATH: Optional[str] = None
_REAP_PIPE_WRITE_FD: Optional[int] = None
_REAPER_PROCESS: Optional[subprocess.Popen] = None
_INSTALL_ATTEMPTED = False
_DEGRADATION_LOGGED = False


def cgroup_reap_enabled() -> bool:
    """Default-ON; ``AGY_WORKER_CGROUP_REAP=0`` opts out."""
    return os.environ.get("AGY_WORKER_CGROUP_REAP", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _own_cgroup_dir() -> Optional[str]:
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("0::"):
                    rel = line[3:] or "/"
                    if rel == "/":
                        return "/sys/fs/cgroup"
                    return os.path.join("/sys/fs/cgroup", rel.lstrip("/"))
    except OSError:
        return None
    return None


def cgroup_v2_available() -> bool:
    """Return True when a delegated unified cgroup-v2 child can be created."""
    if not sys.platform.startswith("linux"):
        return False
    if not os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
        return False
    base = _own_cgroup_dir()
    if not base:
        return False
    probe = os.path.join(base, f".agentorch-probe-{os.getpid()}")
    try:
        os.mkdir(probe)
    except OSError:
        return False
    try:
        os.rmdir(probe)
    except OSError:
        pass
    return True


def create_reap_cgroup() -> Optional[str]:
    """Create and remember this orchestrator process's reap cgroup."""
    global _REAP_CGROUP_PATH
    if _REAP_CGROUP_PATH is not None:
        return _REAP_CGROUP_PATH
    base = _own_cgroup_dir()
    if not base:
        return None
    path = os.path.join(base, f"agentorch-reap-{os.getpid()}")
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    except OSError:
        return None
    _REAP_CGROUP_PATH = path
    return path


def reap_cgroup_path() -> Optional[str]:
    return _REAP_CGROUP_PATH


def join_reap_cgroup_in_child() -> None:
    """Move the forked worker child into the reap cgroup, if one is active."""
    path = _REAP_CGROUP_PATH
    if not path:
        return
    try:
        with open(os.path.join(path, "cgroup.procs"), "w", encoding="ascii") as f:
            f.write(str(os.getpid()))
    except OSError:
        pass


def reap_created_cgroup() -> None:
    """Best-effort synchronous reap for orderly shutdown paths."""
    path = _REAP_CGROUP_PATH
    if path:
        _reap_cgroup(path)


def install_hardkill_reaper() -> None:
    """Install the detached EOF-triggered cgroup reaper once per process."""
    global _INSTALL_ATTEMPTED, _REAP_PIPE_WRITE_FD, _REAPER_PROCESS
    if _INSTALL_ATTEMPTED:
        return
    _INSTALL_ATTEMPTED = True
    if not cgroup_reap_enabled():
        return
    if not cgroup_v2_available():
        _log_degradation_once(
            "Worker cgroup hard-kill reaper unavailable; using PDEATHSIG/killpg cleanup only."
        )
        return
    cgroup_path = create_reap_cgroup()
    if not cgroup_path:
        _log_degradation_once(
            "Worker cgroup hard-kill reaper could not create cgroup; "
            "using PDEATHSIG/killpg cleanup only."
        )
        return
    try:
        r_fd, w_fd = os.pipe()
    except OSError:
        _log_degradation_once(
            "Worker cgroup hard-kill reaper could not create pipe; "
            "using PDEATHSIG/killpg cleanup only."
        )
        return
    env = dict(os.environ)
    env["AGY_REAP_PIPE_FD"] = str(r_fd)
    try:
        _REAPER_PROCESS = subprocess.Popen(
            [sys.executable, "-m", "agy_orchestrator.core.reaper", "--reap", cgroup_path],
            pass_fds=(r_fd,),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            env=env,
        )
        _REAP_PIPE_WRITE_FD = w_fd
    except OSError:
        try:
            os.close(w_fd)
        except OSError:
            pass
        _log_degradation_once(
            "Worker cgroup hard-kill reaper could not spawn; using PDEATHSIG/killpg cleanup only."
        )
    finally:
        try:
            os.close(r_fd)
        except OSError:
            pass


def _log_degradation_once(message: str) -> None:
    global _DEGRADATION_LOGGED
    if _DEGRADATION_LOGGED:
        return
    _DEGRADATION_LOGGED = True
    logger.info(message)


def _reap_cgroup(cgroup_path: str) -> None:
    try:
        with open(os.path.join(cgroup_path, "cgroup.kill"), "w", encoding="ascii") as f:
            f.write("1\n")
    except OSError:
        pass
    try:
        os.rmdir(cgroup_path)
    except OSError:
        pass


def _reaper_main(cgroup_path: str) -> int:
    try:
        fd = int(os.environ.get("AGY_REAP_PIPE_FD", ""))
    except ValueError:
        return 0
    try:
        while os.read(fd, 1):
            pass
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    _reap_cgroup(cgroup_path)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reap")
    args = parser.parse_args(argv)
    if args.reap:
        return _reaper_main(args.reap)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException:
        raise SystemExit(0)

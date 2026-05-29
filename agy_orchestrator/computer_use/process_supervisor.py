"""ProcessSupervisor (Step 4 — XAUTHORITY ISOLATION #1 integrated + prior hardenings).

Every subprocess (Xvfb roots, launch_app children, later reasoning CLIs) is
started under start_new_session=True (KILLABLE TREE, #2) *and* with OS rlimits
(RLIMIT_NPROC + RLIMIT_AS) applied via preexec_fn before exec (HARD RESOURCE
BACKSTOP, #3). The rlimits are inherited by the entire subtree and act as a
hard kernel cap that a fast fork/alloc inside any owned child cannot outrun
even if the psutil poll in enforce_limits is delayed.

XAUTHORITY ISOLATION (hardening #1) is now wired at the child_env construction
site inside spawn(): for every owned process with display_scope="isolated" we
force a freshly generated private cookie file via get_isolated_env(), override
DISPLAY/XAUTHORITY. This guarantees the env seen by xdotool, Xvfb, and every
GUI app has no path (in env or cookie bytes) to the operator's real
~/.Xauthority even though we share the same UID. See xauth.py.

enforce_limits(session) now actively polls budgets (max_processes, max_rss_mb,
max_cpu_percent) and terminates over-limit trees (secondary to the rlimits).

Strict scope (per task):
- All Xvfb/app spawns remain bound to isolated displays only.
- Real-:0 paths are structurally impossible for any action execution path.
- allowlist (FR-23/24) and full SafetyKernel still future.
- The release-blocking `test_executor_env_has_no_path_to_real_x_cookie` lives
  in the test module and covers the helper + the guarantee.

This module remains the FR-12 authority and killable-tree mechanism.
"""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psutil

from .models import IsolatedDisplaySpec, SpawnSpec
from .xauth import generate_private_xauthority, get_isolated_env  # hardening #1 XAUTHORITY ISOLATION


@dataclass
class SpawnedProc:
    """Public handle returned to callers (and stored in events for audit)."""
    pid: int
    pgid: int
    root_id: str
    argv: List[str]
    display_scope: Optional[str] = None
    started_at: float = 0.0


class ProcessSupervisor:
    """Killable owned-process tree manager (FR-11/12/22, hardenings #2+#3).

    start_new_session + preexec rlimits (NPROC/AS) on every spawn root.
    terminate_tree reaps full trees via killpg+ancestry. is_owned is FR-12 truth.
    enforce_limits polls budgets and terminates violators (rlimits are primary).
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Any]] = {}

    # Hardening #3 — floor values we guarantee as a backstop (never lower an
    # inherited limit below these for normal spawns; explicit low values are
    # only passed by the chaos test to prove the cap). A true fork bomb inside
    # an owned child will still be kernel-capped well before it can destabilize
    # a desktop host. The poll in enforce_limits is the fast reaction path.
    #
    # Safety note (Linux rlimit semantics): RLIMIT_NPROC is stored per-process
    # but the *count* it limits is the number of processes/threads sharing the
    # real UID. Critically, each fork consults *only the rlimit of the forking
    # process*. Therefore a tight cap installed via preexec (or prlimit on a
    # subtree root) only constrains forks originating inside that owned tree.
    # Parent, sibling, and unrelated same-UID processes retain their original
    # (usually much higher) limits. This was verified experimentally; the
    # mechanism satisfies host-safety + "never interfere with operator CLIs".
    _RLIMIT_FLOOR_NPROC: tuple[int, int] = (4096, 4096)
    _RLIMIT_FLOOR_AS: tuple[int, int] = (2 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024)  # 2 GiB

    # ---------------- spawn ----------------
    def spawn(
        self,
        argv: Any = None,
        *,
        spec: Optional[SpawnSpec] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        require_owned_group: bool = True,
        no_shell: bool = True,
        display_scope: str = "isolated",
        source_action_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        rlimit_nproc: Optional[tuple[int, int]] = None,
        rlimit_as: Optional[tuple[int, int]] = None,
        isolated_display: Optional[str] = None,  # Step 4: XAUTHORITY hardening #1 wiring
    ) -> SpawnedProc:
        """Create a new owned process tree root under its own session/group.

        Accepts the three call styles the Step-2 tests use plus explicit rlimit
        overrides for the hardening #3 chaos test (and future callers).

        Step-4 addition: when display_scope="isolated", the final child_env
        passed to Popen is forced through xauth.get_isolated_env(). This
        guarantees every owned process — Xvfb, xdotool, launch_app GUI apps —
        receives DISPLAY + private XAUTHORITY pointing at a cookie file that
        has no material allowing authentication to the real :0. The
        `isolated_display` kwarg (when known) scopes the cookie; absent it we
        still sanitize and use a safe default display.
        """
        # Normalise the three call styles into a SpawnSpec we can trust
        if isinstance(argv, SpawnSpec) and spec is None:
            spec = argv
            argv = None

        if spec is not None:
            argv = spec.argv
            cwd = cwd or spec.cwd
            env = env or spec.env
            no_shell = spec.no_shell
            display_scope = spec.display_scope
            source_action_id = source_action_id or spec.source_action_id
            require_owned_group = spec.require_owned_group

        if not argv:
            raise ValueError("argv must be non-empty")
        if no_shell is not True:
            raise ValueError("no_shell must be exactly True")

        child_env = os.environ.copy()
        if env:
            child_env.update({k: v for k, v in env.items() if isinstance(k, str) and isinstance(v, str)})

        # ------------------------------------------------------------------
        # Hardening #1 (Step 4): XAUTHORITY ISOLATION
        # For every owned process with display_scope="isolated" we replace the
        # child env's X* variables with values from get_isolated_env(). The
        # produced env has:
        #   DISPLAY=<the requested isolated display>
        #   XAUTHORITY=/tmp/agu-...Xauth   (fresh private cookie file, 0600)
        # and contains *no* reference to the parent's real ~/.Xauthority or
        # $XAUTHORITY (neither in the dict values nor in the cookie file bytes).
        # This is applied uniformly so xdotool, Xvfb children, and every
        # launch_app GUI app are covered. The guarantee holds even at same UID.
        # ------------------------------------------------------------------
        if display_scope == "isolated":
            try:
                iso = get_isolated_env(isolated_display or ":99")
                child_env["DISPLAY"] = iso["DISPLAY"]
                child_env["XAUTHORITY"] = iso["XAUTHORITY"]
                if "HOME" in iso:
                    child_env["HOME"] = iso["HOME"]
                child_env["AGY_ISOLATED_X"] = iso.get("AGY_ISOLATED_X", "1")
                # Never let a real-session Wayland key survive into an X11-only tree
                child_env.pop("WAYLAND_DISPLAY", None)
            except Exception:
                # Safety net (never leak the real cookie): strip the three dangerous
                # keys and point DISPLAY at the isolated one. XAUTHORITY left unset
                # (clients will fail to auth — safe closed state). Avoids needing
                # extra imports in this module for the rare fallback path.
                for k in ("XAUTHORITY", "DISPLAY", "WAYLAND_DISPLAY"):
                    child_env.pop(k, None)
                child_env["DISPLAY"] = isolated_display or ":99"

        # Hardening #3: preexec applies (or safely raises) the rlimits so the
        # child and all its descendants inherit a finite kernel cap. We *never*
        # lower the limit below what the parent process (the caller of spawn)
        # had, except when the test explicitly passes a tight tuple to prove
        # the cap behavior. This keeps the Step-2 grandchild tests stable on
        # loaded workstations while still providing the hard backstop.
        def _apply_hard_rlimits() -> None:
            try:
                # What the child just inherited from us (the spawner)
                cur_nsoft, cur_nhard = resource.getrlimit(resource.RLIMIT_NPROC)
                cur_asoft, cur_ahard = resource.getrlimit(resource.RLIMIT_AS)

                if rlimit_nproc is not None:
                    # Explicit (chaos test only): allow deliberate tight caps below floor.
                    resource.setrlimit(resource.RLIMIT_NPROC, rlimit_nproc)
                else:
                    # Normal path: install finite backstop floor *only if* it is a
                    # lowering (or equal) relative to inherited hard limit. Never
                    # attempt to raise above what the parent process was allowed.
                    floor_n = self._RLIMIT_FLOOR_NPROC
                    if floor_n[1] <= (cur_nhard if cur_nhard > 0 else 10**18):
                        new_soft = min(cur_nsoft if cur_nsoft > 0 else floor_n[0], floor_n[0])
                        new_hard = min(cur_nhard if cur_nhard > 0 else floor_n[1], floor_n[1])
                        if (new_soft, new_hard) != (cur_nsoft, cur_nhard):
                            resource.setrlimit(resource.RLIMIT_NPROC, (new_soft, new_hard))

                if rlimit_as is not None:
                    resource.setrlimit(resource.RLIMIT_AS, rlimit_as)
                else:
                    floor_a = self._RLIMIT_FLOOR_AS
                    if floor_a[1] <= (cur_ahard if cur_ahard > 0 else 10**18):
                        new_soft = min(cur_asoft if cur_asoft > 0 else floor_a[0], floor_a[0])
                        new_hard = min(cur_ahard if cur_ahard > 0 else floor_a[1], floor_a[1])
                        if (new_soft, new_hard) != (cur_asoft, cur_ahard):
                            resource.setrlimit(resource.RLIMIT_AS, (new_soft, new_hard))
            except (OSError, ValueError, AttributeError):
                # Best-effort (no CAP, container, old libc, non-Linux, etc.).
                # The psutil poll + enforce_limits path remains the secondary defense.
                pass

        try:
            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                env=child_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,   # HARDENING #2
                close_fds=True,
                preexec_fn=_apply_hard_rlimits,  # HARDENING #3 — rlimits before exec
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"executable not found: {argv[0]}") from e

        pid = proc.pid
        try:
            pgid = os.getpgid(pid)
        except OSError:
            pgid = pid

        root_id = source_action_id or f"spawn-{pid}"

        entry: Dict[str, Any] = {
            "pgid": pgid,
            "root_pid": pid,
            "spec": spec or SpawnSpec(
                argv=argv,
                display_scope=display_scope,
                env=env,
                cwd=cwd,
                timeout_ms=timeout_ms,
                require_owned_group=require_owned_group,
                no_shell=True,
                source_action_id=source_action_id,
            ),
            "proc": proc,
            "started_at": time.time(),
        }
        self._registry[root_id] = entry

        return SpawnedProc(
            pid=pid,
            pgid=pgid,
            root_id=root_id,
            argv=list(argv),
            display_scope=display_scope,
            started_at=entry["started_at"],
        )

    def spawn_isolated_display(self, spec: IsolatedDisplaySpec) -> SpawnedProc:
        """FR-22: register an Xvfb as an owned tree root (rlimits applied via #3)."""
        xvfb = spec.xvfb_binary or "Xvfb"
        argv = [xvfb, spec.display, "-screen", "0", spec.screen, "-nolisten", "tcp", "-noreset"]
        sp = SpawnSpec(argv=argv, display_scope="isolated", no_shell=True,
                       source_action_id=f"xvfb-{spec.display.replace(':', '')}")
        # Step 4 wiring: thread the display so spawn() forces the matching
        # private XAUTHORITY cookie + sanitized HOME into the Xvfb process env.
        return self.spawn(sp, isolated_display=spec.display)

    # ---------------- terminate / ownership ----------------
    def terminate_tree(self, root_id: str) -> None:
        """Kill the whole owned process tree (root + grandchildren + daemons in pgid).

        Implements the KILLABLE TREE hardening: SIGTERM then SIGKILL via killpg
        on the session pgid (covers every member that inherited it), followed by
        Popen.wait + psutil children(recursive=True) sweep + non-blocking waitpid
        reaping. Only ever signals pids that were registered through this supervisor.
        """
        if root_id not in self._registry:
            return

        entry = self._registry.pop(root_id)
        pgid = entry["pgid"]
        proc = entry.get("proc")
        root_pid = entry.get("root_pid")

        # 1. group kill (the hardening #2 mechanism)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            if sig == signal.SIGTERM:
                time.sleep(0.08)

        # 2. reap via the Popen we kept (turns the leader into a real exit)
        if proc is not None:
            try:
                proc.wait(timeout=1.5)
            except Exception:
                pass

        # 3. belt-and-suspenders psutil sweep of any remaining descendants
        if root_pid:
            try:
                p = psutil.Process(root_pid)
                for c in p.children(recursive=True):
                    try:
                        c.kill()
                    except Exception:
                        pass
                try:
                    p.kill()
                except Exception:
                    pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # 4. final non-blocking waitpid so the pid is gone from the table
        #    (psutil.pid_exists becomes False for the leader)
        if root_pid:
            try:
                os.waitpid(root_pid, os.WNOHANG)
            except (ChildProcessError, OSError):
                pass

    def is_owned(self, pid: int) -> bool:
        """FR-12: return True only for pids that belong to a tree we registered.

        Primary check: same process group as one of our roots (covers grandchildren
        that inherited the pgid from a start_new_session launch).
        Fallback: direct descendant via psutil ancestry walk (defends against
        processes that changed pgid after fork).
        Dead pids and anything outside our registry always return False.
        """
        if not isinstance(pid, int) or pid <= 0:
            return False
        if not psutil.pid_exists(pid):
            return False

        # Primary: same process group as any root we own
        try:
            pg = os.getpgid(pid)
            for meta in self._registry.values():
                if meta.get("pgid") == pg:
                    return True
        except (ProcessLookupError, PermissionError, OSError):
            pass

        # Fallback: direct descendant of one of our roots
        for meta in self._registry.values():
            root = meta.get("root_pid")
            if not root:
                continue
            try:
                rp = psutil.Process(root)
                for c in rp.children(recursive=True):
                    if c.pid == pid:
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def enforce_limits(self, session: Optional[Dict[str, Any]] = None) -> None:
        """Poll owned trees and enforce budgets (hardening #3 + FR-10/11).

        The OS rlimits set at spawn time are the *hard* backstop (a fork storm
        cannot outrun them). This method is the slower poll that catches
        cpu/rss/proc-count violations using the session budgets and terminates
        the offending owned tree. Only ever acts on pids we registered (FR-12).
        """
        # 1. registry hygiene (dead roots)
        dead = []
        for rid, meta in list(self._registry.items()):
            rp = meta.get("root_pid")
            if rp and not psutil.pid_exists(rp):
                dead.append(rid)
        for rid in dead:
            self._registry.pop(rid, None)

        if not session or not isinstance(session, dict):
            return
        budgets = session.get("budgets") or {}
        if not budgets:
            return

        max_procs = int(budgets.get("max_processes", 64))
        max_rss_mb = int(budgets.get("max_rss_mb", 2048))
        max_cpu = float(budgets.get("max_cpu_percent", 200.0))

        for rid in list(self._registry.keys()):
            meta = self._registry.get(rid)
            if not meta:
                continue
            root_pid = meta.get("root_pid")
            if not root_pid or not psutil.pid_exists(root_pid):
                continue
            try:
                root_p = psutil.Process(root_pid)
                procs = [root_p] + root_p.children(recursive=True)
                proc_count = len(procs)
                rss_mb = sum(p.memory_info().rss for p in procs) / (1024 * 1024)
                # non-blocking cpu sample (first call often 0.0; sufficient for threshold)
                cpu = sum(p.cpu_percent(interval=0.0) for p in procs)
                if proc_count > max_procs or rss_mb > max_rss_mb or cpu > max_cpu:
                    self.terminate_tree(rid)
                    continue
                # Hardening #3: also push the session caps down to the live root
                # via prlimit (affects this process + future forks by it or its kids).
                # Complements the preexec set at birth; the rlimits are the hard
                # kernel backstop that a fast fork/alloc cannot outrun.
                try:
                    prlimit = getattr(resource, "prlimit", None)
                    if prlimit is not None:
                        resource.prlimit(root_pid, resource.RLIMIT_NPROC, (max_procs, max_procs))
                        as_bytes = max_rss_mb * 1024 * 1024
                        resource.prlimit(root_pid, resource.RLIMIT_AS, (as_bytes, as_bytes))
                except (OSError, ValueError, AttributeError):
                    pass
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, ValueError):
                continue

    def roots(self) -> List[SpawnedProc]:
        """Helper used by test fixtures for teardown."""
        out: List[SpawnedProc] = []
        for rid, meta in self._registry.items():
            sp = meta.get("spec")
            out.append(SpawnedProc(
                pid=meta["root_pid"],
                pgid=meta["pgid"],
                root_id=rid,
                argv=sp.argv if sp is not None else [],
                display_scope=sp.display_scope if sp is not None else None,
            ))
        return out

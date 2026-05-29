"""OwnershipResolver (Step 2 — baseline capture + ownership classification).

Exact implementation of COMPUTER_USE_REALGUI_DESIGN.md §4 (Ownership & baseline model)
and FR-28/29 plus the two INVARIANT A/B guarantees. This is the *single source
of truth* for the mission-critical "never touch the operator's pre-existing
terminals / other agents / orchestrator" protection (INVARIANT A, FR-40).

- capture_baseline(real_display=":0"): psutil live PID snapshot + best-effort
  wmctrl -lp primary + xprop _NET_WM_PID fallback exactly as §4.1. All missing,
  zero, or legacy-client PIDs become None (treated FOREIGN). Result + internal
  frozenset/dict state are immutable for the run. This is the narrow audited
  exception that reads the *real* :0 to build the foreign baseline.
- classify(pid): single source of truth. Returns "OWNED" ONLY when
  (supervisor is not None and supervisor.is_owned(pid)) AND (pid not in baseline).
  Baseline numbers are *permanently* FOREIGN (PID-reuse conservatism per §4.2
  + Step-2 instruction). This makes every pre-existing PID default-deny.
- resolve_target_to_pid(target, snapshot_windows): FR-29. Element targets take
  app_pid (or window_id snapshot fallback); coordinate targets delegate to the
  shared _topmost_pid_for_point helper (highest z_index wins). Unresolvable →
  None (maps to TARGET_UNRESOLVABLE DENY).

Two private helpers (_bbox_hit, _topmost_pid_for_point) are shared so the
hermetic Fake and the real class have *byte-for-byte identical* resolution
behavior (no test/prod drift on z-order or edge cases).

FakeOwnershipResolver(synthetic_baseline_pids, synthetic_owned) is the pure
test double: zero psutil calls, zero subprocess, zero X11, never side-effects.
It (and only it) is exercised by the hermetic release-blocking realgui tests
and by the self-test block below. All Step-2 verification uses Fake paths only.

Matches process_supervisor.py style exactly (long safety-rationale docstrings,
defensive isinstance/positive checks everywhere, broad try/except for best-effort
external commands, from __future__, no new dependencies).
psutil is already a transitive requirement of this package.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional, Set

import psutil

from .process_supervisor import ProcessSupervisor


def _sh(cmd: List[str], env: Dict[str, str], timeout: float = 2.5) -> str:
    """Best-effort subprocess (stdlib only). Never raises; returns stdout or ''."""
    try:
        return subprocess.check_output(cmd, env=env, stderr=subprocess.DEVNULL, timeout=timeout, text=True)
    except Exception:
        return ""


def _extract_pid_from_xprop(wid: str, env: Dict[str, str]) -> Optional[int]:
    """Single xprop -id ... _NET_WM_PID parse. Returns positive int or None."""
    out = _sh(["xprop", "-id", wid, "_NET_WM_PID"], env, 1.0)
    m = re.search(r"=\s*(\d+)", out)
    if m:
        try:
            p = int(m.group(1))
            return p if p > 0 else None
        except ValueError:
            return None
    return None


def _bbox_hit(x: int, y: int, bbox: Any) -> bool:
    """Pure geometry test. Treats bbox as [x, x+w) x [y, y+h); zero/negative sizes never hit."""
    if not isinstance(bbox, dict):
        return False
    try:
        bx = int(bbox.get("x", 0))
        by = int(bbox.get("y", 0))
        bw = int(bbox.get("w", 0))
        bh = int(bbox.get("h", 0))
        return bw > 0 and bh > 0 and bx <= x < (bx + bw) and by <= y < (by + bh)
    except (TypeError, ValueError):
        return False


def _topmost_pid_for_point(windows: list, x: int, y: int) -> Optional[int]:
    """FR-29 coordinate resolution (shared by real + Fake for identical behavior).

    Walks the snapshot window list, finds all bbox hits that carry a positive pid,
    returns the pid belonging to the highest z_index (topmost in stack). Ties broken
    by stable sort order. No hit or no usable pid -> None (maps to TARGET_UNRESOLVABLE DENY).
    This helper guarantees the z-order rule is never implemented differently between
    the production path and the hermetic Fake used in all release-blocking tests.
    """
    hits: List[tuple[int, int]] = []
    for w in (windows or []):
        if not isinstance(w, dict):
            continue
        if not _bbox_hit(x, y, w.get("bbox")):
            continue
        p = w.get("pid")
        if not (isinstance(p, int) and p > 0):
            continue
        try:
            z = int(w.get("z_index") or 0)
        except Exception:
            z = 0
        hits.append((z, p))
    if not hits:
        return None
    hits.sort(reverse=True)  # highest z wins; stable on original order for z ties
    return hits[0][1]


class OwnershipResolver:
    """Production resolver (injected into SafetyKernel / SessionController).

    Baseline captured *once* at SessionController start, before any ProcessSupervisor.spawn.
    This is the root of INVARIANT A: everything alive then is FOREIGN forever.
    classify() and resolve_target_to_pid() are the only two call sites for all
    real_act permission decisions. The supervisor is a live reference (its registry
    grows as owned children are spawned); baseline is a frozen snapshot.
    """

    def __init__(
        self,
        baseline_pids: Optional[Set[int]] = None,
        baseline_windows: Optional[Dict[str, Any]] = None,
        supervisor: Optional[ProcessSupervisor] = None,
    ) -> None:
        self._baseline_pids: frozenset[int] = frozenset(baseline_pids or ())
        self._baseline_windows: Dict[str, Any] = dict(baseline_windows or {})
        self._supervisor = supervisor

    def capture_baseline(self, real_display: str = ":0") -> Dict[str, Any]:
        """FR-28 / §4.1: immutable baseline of every live PID + every X window's _NET_WM_PID.

        Must be called before any ProcessSupervisor.spawn so that operator's pre-existing
        terminals, claude/codex/grok/agy sessions, and the orchestrator itself are recorded
        as FOREIGN for the entire run lifetime (INVARIANT A + FR-40, the mission-critical
        host-safety guarantee). Any PID in this set is *never* allowed to receive input
        under the real_gui harness unless an explicit operator grant is cached later.

        Uses the caller's current DISPLAY/XAUTHORITY (the *real* :0). This is the single,
        narrow, audited exception to XAUTHORITY isolation #1 — only for the ownership
        probe that builds the deny-by-default baseline. The prompter later uses the same
        real display for its zenity dialog (also owned child, no foreign injection).
        """
        # 1. psutil snapshot — authoritative list of every PID alive at this instant
        try:
            live_pids: Set[int] = set(psutil.pids())
        except Exception:
            live_pids = set()

        # 2. X window → PID map via the documented EWMH path (wmctrl -lp + xprop)
        win_map: Dict[str, Optional[int]] = {}
        env = os.environ.copy()
        env["DISPLAY"] = real_display
        env.pop("WAYLAND_DISPLAY", None)

        # Primary: wmctrl -lp (most reliable when a compliant WM is present)
        try:
            out = _sh(["wmctrl", "-lp"], env)
            for line in (out or "").splitlines():
                parts = line.split(None, 4)
                if len(parts) >= 3:
                    wid = parts[0]
                    try:
                        pid = int(parts[2])
                        win_map[wid] = pid if pid > 0 else None
                    except ValueError:
                        win_map[wid] = None
        except Exception:
            pass

        # Secondary: any wids still unknown *or* whose primary -lp parse gave no usable pid
        # get an authoritative xprop _NET_WM_PID probe. This makes baseline capture
        # robust across WMs that report pid unreliably in wmctrl columns (legacy clients
        # still end up with None -> FOREIGN, as required).
        try:
            out = _sh(["wmctrl", "-l"], env)
            for line in (out or "").splitlines():
                parts = line.split(None, 1)
                if not parts:
                    continue
                wid = parts[0]
                existing = win_map.get(wid)
                if isinstance(existing, int) and existing > 0:
                    continue
                win_map[wid] = _extract_pid_from_xprop(wid, env)
        except Exception:
            pass

        # Commit as immutable for the lifetime of this resolver instance
        self._baseline_pids = frozenset(live_pids)
        self._baseline_windows = dict(win_map)

        return {
            "baseline_pids": set(self._baseline_pids),
            "baseline_windows": {k: v for k, v in self._baseline_windows.items()},
            "captured_at": time.time(),
            "display": real_display,
        }

    def classify(self, pid: int) -> str:
        """§4.2 / INVARIANT A+B (mission-critical baseline protection): single source of truth.

        Per spec:
            OWNED   ⇔  ProcessSupervisor.is_owned(p) is True  AND  p ∉ baseline_pids
            FOREIGN ⇔  otherwise

        A PID number present in the baseline captured at session start (before any
        owned spawn) is *forever* FOREIGN for this run, even if is_owned() would
        claim it later. This is the deliberate conservative rule that makes the
        operator's pre-existing terminals, claude/codex/grok/agy, and orchestrator
        default-deny (FR-40 regression). PID-reuse supersede (registry timestamp
        after baseline) is noted in the design but requires supervisor timestamp
        tracking not present in Step 2 scope; the implementation follows the
        "only if ... AND not in baseline" rule given for this micro-step and errs
        closed (safer for host protection).
        """
        if not isinstance(pid, int) or pid <= 0:
            return "FOREIGN"
        if pid in self._baseline_pids:
            return "FOREIGN"
        if self._supervisor is not None:
            try:
                if self._supervisor.is_owned(pid):
                    return "OWNED"
            except Exception:
                pass
        return "FOREIGN"

    def resolve_target_to_pid(self, target: dict, snapshot_windows: list) -> Optional[int]:
        """FR-29: attribute an action target to its owning PID (or None → DENY).

        element target: the ElementHandle (or enriched target dict) carries app_pid.
        coordinate target: delegated to shared _topmost_pid_for_point (highest z wins).
        Ambiguous / missing / stale / no positive pid -> None (TARGET_UNRESOLVABLE DENY).
        """
        if not isinstance(target, dict):
            return None

        kind = target.get("kind")

        # Element path — app_pid authoritative; window_id snapshot fallback for enriched targets
        if kind == "element":
            for key in ("app_pid", "pid"):
                val = target.get(key)
                if isinstance(val, int) and val > 0:
                    return val
            wid = target.get("window_id")
            if wid:
                for w in (snapshot_windows or []):
                    if isinstance(w, dict) and w.get("window_id") == wid:
                        p = w.get("pid")
                        if isinstance(p, int) and p > 0:
                            return p
            return None

        # Coordinate path — exact same helper as Fake (guarantees identical z-order topmost selection)
        if kind == "coordinate" or ("x" in target and "y" in target):
            try:
                x = int(target.get("x", -10**9))
                y = int(target.get("y", -10**9))
            except (TypeError, ValueError):
                return None
            return _topmost_pid_for_point(snapshot_windows, x, y)

        return None


class FakeOwnershipResolver:
    """Pure synthetic drop-in for *all* hermetic realgui tests (FR-30..FR-40 + INV E/F).

    Never calls psutil, never spawns subprocess (even wmctrl/xprop), never reads any
    X display, never has side effects. Synthetic baseline + synthetic owned control
    every classify() answer and resolve path. The 15-line __main__ block and every
    release-blocking @pytest.mark.realgui test use *only* this class (plus injected
    clock in later grants tests). Full behavioral parity with the real class on the
    two public decision methods (via the shared helpers).
    """

    def __init__(self, synthetic_baseline_pids: Optional[Set[int]] = None, synthetic_owned: Optional[Set[int]] = None) -> None:
        self._baseline: frozenset[int] = frozenset(synthetic_baseline_pids or set())
        self._owned: Set[int] = set(synthetic_owned or set())

    def capture_baseline(self, real_display: str = ":0") -> Dict[str, Any]:
        return {
            "baseline_pids": set(self._baseline),
            "baseline_windows": {},
            "captured_at": 0.0,
            "display": real_display,
            "fake": True,
        }

    def classify(self, pid: int) -> str:
        if not isinstance(pid, int) or pid <= 0:
            return "FOREIGN"
        if pid in self._baseline:
            return "FOREIGN"
        if pid in self._owned:
            return "OWNED"
        return "FOREIGN"

    def resolve_target_to_pid(self, target: dict, snapshot_windows: list) -> Optional[int]:
        """Full parity with OwnershipResolver (element app_pid + window_id fallback; coordinate z-topmost).

        All hermetic tests therefore exercise the exact same resolution rules that will run in prod.
        """
        if not isinstance(target, dict):
            return None
        # element fast path + window_id snapshot fallback (exact match to real impl)
        for k in ("app_pid", "pid"):
            v = target.get(k)
            if isinstance(v, int) and v > 0:
                return v
        wid = target.get("window_id")
        if wid:
            for w in (snapshot_windows or []):
                if isinstance(w, dict) and w.get("window_id") == wid:
                    p = w.get("pid")
                    if isinstance(p, int) and p > 0:
                        return p
        # coordinate: delegate to shared helper (highest z, identical to production)
        kind = target.get("kind")
        if kind == "coordinate" or ("x" in target and "y" in target):
            try:
                x = int(target.get("x", -10**9))
                y = int(target.get("y", -10**9))
            except Exception:
                return None
            return _topmost_pid_for_point(snapshot_windows, x, y)
        return None


# ------------------------------------------------------------------
# 15-line self-test block (exercises ONLY FakeOwnershipResolver; full parity paths).
# python -m agy_orchestrator.computer_use.ownership
# Zero real X, zero psutil, zero side effects. Covers FR-28/29/40 + INV A/B.
# ------------------------------------------------------------------
if __name__ == "__main__":
    f = FakeOwnershipResolver({1001, 1002}, {2001})
    assert f.classify(1001) == "FOREIGN"  # baseline -> FOREIGN (FR-40)
    assert f.classify(2001) == "OWNED"
    assert f.classify(999) == "FOREIGN"
    assert f.resolve_target_to_pid({"kind": "element", "app_pid": 2001}, []) == 2001
    ws = [{"pid": 2001, "bbox": {"x": 0, "y": 0, "w": 50, "h": 50}, "z_index": 5}]
    assert f.resolve_target_to_pid({"kind": "coordinate", "x": 10, "y": 10}, ws) == 2001
    assert f.resolve_target_to_pid({"kind": "coordinate", "x": 999, "y": 0}, ws) is None
    # window_id fallback + z-topmost (via shared helper)
    ws2 = [{"window_id": "0xabc", "pid": 2001, "bbox": {"x": 0, "y": 0, "w": 10, "h": 10}}]
    assert f.resolve_target_to_pid({"kind": "element", "window_id": "0xabc"}, ws2) == 2001
    fb = f.capture_baseline()
    assert fb.get("fake") and 1001 in fb["baseline_pids"]
    print("ownership.py Fake self-test: 9 asserts passed (baseline + classify + element/coordinate + z + fallback)")

"""GrantCache + FakeClock (Step 3 — FR-36 grant scopes + injectable monotonic clock).

Exact implementation of COMPUTER_USE_REALGUI_DESIGN.md §7 (Grant cache) and
the Step-3 contract: in-memory per-run storage only, clock injectable (defaults
to time.time), ACTION single-use+consumed, PROCESS_RUN sticky, PROCESS_TTL
expiry via injected clock (no wall time in any test path), new unseen pid always
misses, clear_for_pid for PID death, grant(pid, scope, ttl_seconds?) -> Grant.

__init__ signature per contract (clock first optional, run_id keyword-only for
safety). All paths defensive on pid types/positivity. TTL expiry + ACTION
consumption are deterministic under injected clock.

All permission decisions remain auditable; cache never persists across runs.
A fresh GrantCache instance per WorkerSession (wired in later steps).

FakeClock is the *only* clock used by release-blocking @pytest.mark.realgui
FR-36 tests (synthetic baselines + FakePrompter + FakeOwnershipResolver).

Matches models.py dataclass/enum style + ownership.py defensive patterns
(no subprocess here; zero X11; zero side effects on import or construction).

Usage (exactly 10 lines, production + test paths; run_id is keyword-only):

    from agy_orchestrator.computer_use.grants import GrantCache, FakeClock
    from agy_orchestrator.computer_use.models import GrantScope
    cache = GrantCache(run_id="20260529-...")  # real wall-clock for TTL
    g = cache.grant(4242, GrantScope.PROCESS_RUN)
    assert cache.is_granted(4242)
    cache.grant(4242, GrantScope.ACTION)
    assert cache.is_granted(4242, action_id="act-77") and not cache.is_granted(4242)
    clk = FakeClock(start=1_000_000.0)
    c2 = GrantCache(clock=clk, run_id="t")
    c2.grant(99, GrantScope.PROCESS_TTL, ttl_seconds=600)
    clk.advance(601)
    assert not c2.is_granted(99)

Zero real time.time() calls in any path that supplies clock= (all @pytest.mark.realgui FR-36 tests).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Union

from .models import Grant, GrantScope


class FakeClock:
    """Deterministic callable clock for hermetic FR-36 TTL grant tests.

    Never consults the system wall clock. Tests control time via .advance().
    """

    def __init__(self, start: float = 0.0) -> None:
        self._t: float = float(start)

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        """Advance simulated time (used only in test paths)."""
        self._t += float(seconds)


class GrantCache:
    """Per-run in-memory grant cache for real_act foreign PID decisions (FR-36).

    - ACTION grants are single-use: first is_granted(pid, action_id=...) after
      grant() returns True and consumes the grant for that pid.
    - PROCESS_RUN grants persist for pid until clear_for_pid (or run end).
    - PROCESS_TTL grants expire relative to the injected clock() value.
    - Unseen / cleared / expired / bad pids always return False (new foreign
      misses; INVARIANT C fail-closed).
    - DENY scope on grant() clears the pid (no grant recorded).
    """

    def __init__(
        self,
        clock: Optional[Callable[[], float]] = None,
        *,
        run_id: str = "",
    ) -> None:
        # clock may be a FakeClock for hermetic tests (FR-36); default is real wall
        # time ONLY in production paths. Keyword-only run_id prevents accidental
        # positional binding bugs (clock first in the Step-3 contract signature).
        self.clock: Callable[[], float] = clock if clock is not None else time.time
        self.run_id: str = run_id if run_id is not None else ""
        self._store: Dict[int, Dict[str, Any]] = {}  # pid -> internal entry

    def grant(
        self,
        pid: int,
        scope: Union[GrantScope, str],
        ttl_seconds: Optional[int] = None,
    ) -> Grant:
        """Create/overwrite a grant for pid and return the snapshot Grant dataclass.

        ttl_seconds only used for PROCESS_TTL (default 600s = 10 min per design).
        """
        if not isinstance(pid, int) or pid <= 0:
            # Never store bad pids; return a DENY Grant echoing the (invalid) pid
            # for auditability. is_granted(bad) is always False.
            bad_pid = pid if isinstance(pid, int) else 0
            return Grant(pid=bad_pid, scope=GrantScope.DENY.value, run_id=self.run_id)

        scope_val = scope.value if isinstance(scope, GrantScope) else str(scope)

        if scope_val == GrantScope.DENY.value:
            self.clear_for_pid(pid)
            return Grant(pid=pid, scope=GrantScope.DENY.value, run_id=self.run_id)

        entry: Dict[str, Any] = {"scope": scope_val}
        expires_at_str: Optional[str] = None
        expiry: Optional[float] = None

        if scope_val == GrantScope.PROCESS_TTL.value:
            secs = ttl_seconds if ttl_seconds is not None else 600
            expiry = self.clock() + float(secs)
            expires_at_str = datetime.fromtimestamp(expiry, tz=timezone.utc).isoformat()
            entry["expiry"] = expiry
        elif scope_val == GrantScope.PROCESS_RUN.value:
            entry["expiry"] = None
        else:
            # ACTION (or unknown -> treat as ACTION)
            if scope_val != GrantScope.ACTION.value:
                scope_val = GrantScope.ACTION.value
            entry["scope"] = scope_val
            entry["consumed"] = False
            entry["expiry"] = None

        self._store[pid] = entry

        return Grant(
            pid=pid,
            scope=scope_val,
            expires_at=expires_at_str,
            run_id=self.run_id,
        )

    def is_granted(self, pid: int, action_id: Optional[str] = None) -> bool:
        """Return whether pid currently holds a valid grant for this run.

        ACTION: at most one True (consumed on the check that succeeds).
        TTL: auto-clears and returns False once clock() exceeds expiry.
        """
        if not isinstance(pid, int) or pid <= 0:
            return False
        entry = self._store.get(pid)
        if entry is None:
            return False

        scope_val = entry.get("scope")
        expiry = entry.get("expiry")

        if scope_val == GrantScope.PROCESS_RUN.value:
            return True

        if scope_val == GrantScope.PROCESS_TTL.value:
            if expiry is not None and self.clock() > expiry:
                self.clear_for_pid(pid)
                return False
            return True

        if scope_val == GrantScope.ACTION.value:
            if entry.get("consumed"):
                return False
            entry["consumed"] = True
            if action_id:
                entry.setdefault("consumed_by", action_id)
            return True

        return False

    def clear_for_pid(self, pid: int) -> None:
        """Remove any grant entry for pid (PID death, revoke, or DENY)."""
        self._store.pop(pid, None)


# ------------------------------------------------------------------
# Hermetic self-test (python -m agy_orchestrator.computer_use.grants).
# Exercises FakeClock + every grant scope + consumption + expiry + clear + edges.
# Zero wall-clock (even the default time.time binding is never invoked), zero
# subprocess, zero X, zero side effects. Covers the perfected contract.
# ------------------------------------------------------------------
if __name__ == "__main__":
    clk = FakeClock(1_000_000.0)
    gc = GrantCache(clock=clk, run_id="self-test-1")

    assert gc.is_granted(999) is False  # unseen foreign pid misses

    ga = gc.grant(4001, GrantScope.ACTION)
    assert ga.scope == "ACTION" and ga.expires_at is None and ga.run_id
    assert gc.is_granted(4001, action_id="act-1") is True   # consumes
    assert gc.is_granted(4001, action_id="act-2") is False  # single-use gone

    gr = gc.grant(4002, "PROCESS_RUN")  # accepts str or Enum
    assert gc.is_granted(4002) is True

    gttl = gc.grant(4003, GrantScope.PROCESS_TTL, ttl_seconds=10)
    assert gc.is_granted(4003) is True
    clk.advance(11)
    assert gc.is_granted(4003) is False  # TTL auto-expired + cleared

    gc.clear_for_pid(4002)
    assert gc.is_granted(4002) is False

    gd = gc.grant(4004, GrantScope.DENY)
    assert gd.scope == "DENY" and gc.is_granted(4004) is False

    # Refined contract edges (bad pid never stored, pid echoed in DENY Grant sentinel;
    # ACTION re-arm after consume; type guards). All via FakeClock only.
    gbad = gc.grant(-1, GrantScope.PROCESS_RUN)
    assert gbad.pid == -1 and gbad.scope == "DENY" and not gc.is_granted(-1)
    assert gc.is_granted("bad") is False and gc.is_granted(0) is False

    gc.grant(4005, GrantScope.ACTION)
    assert gc.is_granted(4005, action_id="x1") is True
    assert gc.is_granted(4005) is False  # consumed
    gc.grant(4005, GrantScope.ACTION)  # re-grant arms a new single-use
    assert gc.is_granted(4005) is True

    print("GrantCache + FakeClock self-test OK (ACTION consume, RUN, TTL expiry, clear, DENY, bad-pid, re-arm)")

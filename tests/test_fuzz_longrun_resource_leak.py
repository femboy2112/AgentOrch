"""Long-run resource-exhaustion probes (dimension: accumulation across many steps).

HERMETIC + FAST. No live workers, no network, no multi-hour waits. These probes
simulate a long-lived dispatcher / broker process servicing MANY dispatches and
assert the DESIRED resilient behavior (bounded memory / threads / git admin
state). A single one-shot CLI run frees everything on exit and hides these
leaks; they only bite a Phase-3 singleton broker that runs hundreds of jobs in
ONE process.

Each probe asserts the resilient contract directly. Probes whose assertion FAILS
on today's source (a real long-run defect) are marked xfail(strict=True) so the
committed suite stays green while the test serves as a RED contract for the fix.
The xfail flips to XPASS->fail once the leak is fixed.
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.not_slow


# --------------------------------------------------------------------------- #
# Probe 1: EventBus never evicts completed-run state -> unbounded growth in a
# long-lived dispatcher (the module-global EVENT_BUS in harness/dispatch.py).
# --------------------------------------------------------------------------- #
def test_event_bus_evicts_finished_run_state():
    """A long-lived bus that has fully finished N runs must not retain all N
    runs' event histories + closed-set entries forever.

    Mimics the dispatch teardown exactly: per run -> publisher_for, push events,
    bus.close(run_id), bus.sinks.pop(run_id) (dispatch.py:1965-1966). A resilient
    bus would evict (or cap to a small retention window) the per-run queues once
    a run is closed and its sink removed, so memory stays bounded regardless of
    how many runs the process has serviced.
    """
    from dashboard.event_bus import EventBus

    bus = EventBus()
    N = 50
    EVENTS_PER_RUN = 20

    for i in range(N):
        run_id = f"run{i}"
        pub = bus.publisher_for(run_id, worker="codex", model="m", effort="high")
        for j in range(EVENTS_PER_RUN):
            pub({"kind": "message", "text": f"event {j}"})
        # Exactly what dispatch's finally block does at run end:
        bus.close(run_id)
        bus.sinks.pop(run_id, None)

    # DESIRED: a finished+closed+sink-popped run is fully evicted, so the bus
    # holds at most a small bounded retention window — NOT all N runs. We allow a
    # generous window (a fixed bus might keep the last few for late SSE replay),
    # but it must not be O(N).
    RETENTION_CAP = 8
    assert len(bus.queues) <= RETENTION_CAP, (
        f"EventBus retained queues for {len(bus.queues)} finished runs "
        f"(expected <= {RETENTION_CAP}); a long-lived broker leaks every run's "
        f"full event history forever"
    )
    assert len(bus.closed) <= RETENTION_CAP, (
        f"EventBus.closed grew to {len(bus.closed)} entries (expected <= "
        f"{RETENTION_CAP}); the closed-set is never pruned"
    )


# --------------------------------------------------------------------------- #
# Probe 2: TelegramNotifier daemon delivery thread is never stopped -> one
# leaked thread (+ its BuildState/queue closure) per dispatch.
# --------------------------------------------------------------------------- #
class _FakeTelegramClient:
    """A TelegramClient stand-in: configured=True (so notifier.active), but every
    send is a no-op, no network."""

    configured = True

    def send_message(self, *a, **k):
        return {"result": {"message_id": 1}}

    def pin_chat_message(self, *a, **k):
        return {"ok": True}

    def edit_message_text(self, *a, **k):
        return {"ok": True}


def test_telegram_notifier_reaps_worker_thread():
    """Each dispatch builds a fresh TelegramNotifier and spins a daemon delivery
    thread on first event. A long-lived broker must reap that thread when the run
    finishes, or every job leaks a live thread until thread creation fails.

    A resilient notifier exposes a stop()/close() that poison-pills the worker;
    here we look for one, call it, and assert the thread exits — and that the
    live thread count returns to baseline across K runs.
    """
    from harness.telegram import TelegramNotifier

    baseline = threading.active_count()
    K = 6
    notifiers = []

    for i in range(K):
        notifier = TelegramNotifier(
            run_id=f"run{i}",
            mode="adversarial",
            verbosity="normal",
            client=_FakeTelegramClient(),
            chat_ids=[123],
            max_queue=64,
            instruction="do a thing",
        )
        # Drive a synthetic event so _ensure_worker() spins the daemon thread.
        notifier({"kind": "message", "text": "hello", "ts": 1.0,
                  "data": {"event": "agent_started"}})
        notifier.finished({"success": True})  # renders summary + flush (NO stop)
        notifiers.append(notifier)

    # DESIRED: the notifier exposes a way to stop its worker; finished() or an
    # explicit stop()/close() should poison-pill it. Probe for the API and use it.
    stop_method = None
    for name in ("stop", "close", "shutdown"):
        if hasattr(notifiers[0], name) and callable(getattr(notifiers[0], name)):
            stop_method = name
            break
    assert stop_method is not None, (
        "TelegramNotifier exposes no stop()/close()/shutdown() to reap its daemon "
        "delivery thread; the worker blocks on queue.get() forever and leaks per "
        "dispatch"
    )

    for n in notifiers:
        getattr(n, stop_method)()
    # Give the (now poison-pilled) threads a moment to exit.
    for n in notifiers:
        w = getattr(n, "_worker", None)
        if w is not None:
            w.join(0.5)

    leaked = threading.active_count() - baseline
    assert leaked <= 1, (
        f"{leaked} delivery threads still alive after stopping {K} notifiers "
        f"(expected ~0); worker threads are not reaped"
    )


# --------------------------------------------------------------------------- #
# Probe 3: events.jsonl is reopened (open/write/close) on EVERY event -> syscall
# churn + flush backpressure under an event flood.
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    reason="fuzz-longrun resource_leak: dispatch _sink opens+writes+closes "
    "events.jsonl per event (no held-open handle) -> O(M) syscall churn on the "
    "synchronous producing coroutine path under an event flood",
    strict=True,
)
def test_events_jsonl_not_reopened_per_event(tmp_path, monkeypatch):
    """The dispatch _sink does `with events_path.open('a') ...` for EACH event.

    On a chatty mission-critical master run (thousands of events) that is one
    open()+write()+close() syscall sequence per event, on the synchronous
    producing coroutine path. A resilient sink reuses a held-open handle (or a
    buffered writer) so open() is called O(1), not O(M).

    We replicate the EXACT sink body from dispatch.py:1603-1610 against a counting
    Path.open wrapper and feed it M events.
    """
    events_path = tmp_path / "events.jsonl"

    real_open = Path.open
    open_calls = {"n": 0}

    def _counting_open(self, *args, **kwargs):
        # Count only opens of the events file (ignore unrelated tmp I/O).
        if Path(self) == events_path:
            open_calls["n"] += 1
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _counting_open)

    import json as _json

    # Faithful copy of the dispatch _sink hot-path (no RunMonitor here).
    def _sink(event: dict) -> None:
        with events_path.open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps(event, ensure_ascii=False) + "\n")

    M = 2000
    for j in range(M):
        _sink({"kind": "stderr", "text": f"chatter {j}"})

    # DESIRED resilient behavior: the sink does NOT pay an open()/close() per
    # event. A single held-open handle (or a small number of reopens) keeps this
    # bounded well below M.
    assert open_calls["n"] < M, (
        f"events.jsonl was reopened {open_calls['n']} times for {M} events "
        f"(one open/write/close per event); a chatty long run pays this syscall "
        f"churn on the producing coroutine's hot path"
    )


# --------------------------------------------------------------------------- #
# Probe 4: git worktree admin entries accumulate in the base repo when teardown
# partially fails across many DAG nodes (no `git worktree prune` is ever run).
# --------------------------------------------------------------------------- #
def _git(args, cwd):
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, timeout=30,
    )


def _git_available() -> bool:
    try:
        return subprocess.run(["git", "--version"], capture_output=True,
                              timeout=5).returncode == 0
    except Exception:
        return False


def test_worktree_admin_entries_pruned_on_partial_teardown_failure(tmp_path, monkeypatch):
    """Simulate the killed-mid-teardown / locked-worktree case: _remove_git_worktree
    is a no-op, so only shutil.rmtree(tmp_root) runs. The base repo's
    .git/worktrees/<name> admin dir then survives. A resilient teardown would
    still reconcile the base registry (e.g. `git worktree prune`) so stale entries
    don't accumulate across many nodes.
    """
    if not _git_available():
        pytest.skip("git not available")

    import asyncio
    from agy_orchestrator.execution import workspace as ws

    base = tmp_path / "baserepo"
    base.mkdir()
    assert _git(["init"], base).returncode == 0
    _git(["config", "user.email", "t@t.t"], base)
    _git(["config", "user.name", "t"], base)
    (base / "f.txt").write_text("hi\n")
    _git(["add", "-A"], base)
    assert _git(["commit", "-m", "init"], base).returncode == 0

    # Simulate the partial-teardown failure mode: the best-effort worktree remove
    # is skipped (process killed before it ran, or `git worktree remove` failed).
    async def _noop_remove(base_, target_):
        return None

    monkeypatch.setattr(ws, "_remove_git_worktree", _noop_remove)

    worktrees_admin = base / ".git" / "worktrees"
    K = 8

    async def _churn():
        for i in range(K):
            async with ws.candidate_workspace(base, candidate_id=f"node{i}") as (_wp, backend):
                assert backend == "worktree", "expected worktree backend on a clean git repo"

    asyncio.run(_churn())

    # Count stale admin entries left behind in the base repo's registry.
    stale = (
        sorted(p.name for p in worktrees_admin.iterdir())
        if worktrees_admin.exists() else []
    )

    # DESIRED: teardown reconciles the base registry so stale entries do NOT
    # accumulate (a resilient impl runs `git worktree prune` or removes the admin
    # dir even when the per-node remove is skipped).
    assert len(stale) == 0, (
        f"{len(stale)} stale .git/worktrees admin entries leaked across {K} nodes "
        f"({stale}); no `git worktree prune` reconciles the base registry, so a "
        f"long graph run inflates .git and slows every subsequent worktree add"
    )

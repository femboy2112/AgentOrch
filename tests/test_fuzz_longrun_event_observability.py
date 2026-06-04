"""Long-run resilience probes: event bus / observability backpressure.

DIMENSION: Event bus / observability backpressure (the DEFERRED coverage gap
from the 8e21495 fuzz-harden — event_bus / run_monitor / telegram sink were
never long-run-fuzzed).

Each test below encodes the *desired* resilient behavior for a hypothesized
LONG-RUN failure mode. A passing test = the orchestrator already handles it
(regression guard). A test decorated @pytest.mark.xfail(strict=True) = a REAL
long-run defect: the assertion FAILS today (a RED contract for the eventual
fix), and xfail keeps the committed suite green until the fix flips it to XPASS.

Everything is HERMETIC + sub-second: an injectable fake monotonic clock drives
all stall math, synthetic events are fabricated in-process, no live worker, no
network, no real sleeping for long-run conditions.
"""
from __future__ import annotations

import threading
import time
from typing import Any, List

import pytest

from dashboard.event_bus import EventBus
from harness.run_monitor import Notifier, RunMonitor

pytestmark = pytest.mark.not_slow


# --------------------------------------------------------------------------- #
# Probe 1 — EventBus per-run replay buffer grows unbounded (no DROP policy).
# --------------------------------------------------------------------------- #
# A long/chatty master run streams tens of thousands of token events; every
# _publish appends to self.queues[run_id] and NOTHING trims it for the whole
# run. Memory is O(total events emitted). The desired resilient behavior is a
# bounded ring-buffer (a DROP-oldest cap) so a multi-hour run can't OOM the box
# it shares with the worker it supervises. subscribe() must still advance by
# _event_id past dropped rows.
#
# OBSERVED (pre-fix): with 50_000 events pushed, len(bus.queues['r']) == 50_000
# — no cap whatsoever; the bounded-buffer assertion FAILS. -> verdict: bug.
def test_eventbus_replay_buffer_is_bounded():
    bus = EventBus()
    pub = bus.publisher_for("r", worker="codex", model="m", effort="high")

    N = 50_000
    CAP = 20_000  # a sane upper bound a long-run-safe ring buffer would enforce
    for i in range(N):
        pub({"kind": "stderr", "text": "x" * 256})

    # Resilient contract: the in-memory replay buffer must NOT retain every
    # event forever — it must cap (drop-oldest) so memory is bounded.
    assert len(bus.queues["r"]) <= CAP, (
        f"replay buffer retained {len(bus.queues['r'])} events with no cap — "
        f"unbounded growth OOMs a long run"
    )


# A companion *resilient* guard for the part that already works: even if rows
# were dropped, subscribe()'s floor logic keys off _event_id (a monotonically
# increasing counter), so a late subscriber resumes correctly past old ids.
def test_eventbus_event_ids_are_monotonic_for_replay_floor():
    bus = EventBus()
    pub = bus.publisher_for("r", worker="codex", model="m", effort="high")
    for i in range(2000):
        pub({"kind": "stderr", "text": "tok"})
    ids = [e["_event_id"] for e in bus.queues["r"]]
    # strictly increasing -> a ring-buffer fix could drop the head and a
    # resuming subscriber would still advance correctly by id.
    assert ids == sorted(ids)
    assert ids[-1] == len(ids) - 1


# --------------------------------------------------------------------------- #
# Probe 2 — Telegram sink does per-event blocking file I/O on the publish path.
# --------------------------------------------------------------------------- #
# TelegramNotifier.__call__ is an EventBus sink invoked synchronously on the
# dispatch event-loop thread. Per progress event it calls _suppressed_for ->
# _live_state(), which open()+json.load()s bot_state.json EVERY event. Under
# event flood this serializes the publish path on a filesystem stat/open/parse;
# on a slow/contended/NFS mount each event blocks the loop, starving the
# workers' own stream pumps. The desired behavior: bot_state.json is read at
# most ~once (or TTL/mtime-cached), not once per event.
#
# OBSERVED (pre-fix): for 1000 events, state_path()'s file is opened ~1000
# times (one per event); the "<= a small constant" assertion FAILS. -> bug.
def test_telegram_sink_does_not_reopen_state_file_per_event(tmp_path, monkeypatch):
    import json as _json
    from harness import telegram as tg

    state_file = tmp_path / "bot_state.json"
    # A non-suppressing state with at least one chat (so _suppressed_for takes
    # the real path that reads the file).
    state_file.write_text(_json.dumps({"chats": {"42": {}}, "verbosity": "normal"}))

    monkeypatch.setattr(tg, "state_path", lambda path=None: str(state_file))

    opens = {"n": 0}
    real_open = open

    def counting_open(file, *a, **k):
        if str(file) == str(state_file):
            opens["n"] += 1
        return real_open(file, *a, **k)

    monkeypatch.setattr("builtins.open", counting_open)

    class _FakeClient:
        configured = True

        def send_message(self, *a, **k):
            return {"result": {"message_id": 1}}

        def pin_chat_message(self, *a, **k):
            return None

        def edit_message_text(self, *a, **k):
            return None

    notifier = tg.TelegramNotifier(
        run_id="r",
        mode="direct",
        verbosity="normal",
        client=_FakeClient(),
        chat_ids=[42],
    )

    N = 1000
    for i in range(N):
        # agent_started renders a 🤖 spin-up line at 'normal' -> a real progress
        # event that flows through _suppressed_for -> _live_state().
        notifier(
            {
                "kind": "lifecycle",
                "worker": "codex",
                "model": "m",
                "effort": "high",
                "data": {"event": "agent_started", "detail": {}},
            }
        )

    # Resilient contract: the persisted state file is read at most a small
    # constant number of times across the whole flood (cached), NOT per-event.
    assert opens["n"] <= 5, (
        f"bot_state.json opened {opens['n']}x for {N} events — per-event "
        f"blocking I/O on the publish hot path"
    )


# --------------------------------------------------------------------------- #
# Probe 3 — Run watchdog never fires when a worker thrashes (chatty lifecycle).
# --------------------------------------------------------------------------- #
# RunMonitor.observe resets the progress clock on ANY event whose kind is not in
# _NON_PROGRESS_KINDS = {stderr,stdout,heartbeat}. A worker stuck in a
# retry/fallback loop emits a 'lifecycle' agent_started + a 'usage' call event
# on EVERY attempt — neither kind is non-progress — so a run making ZERO real
# milestone progress keeps _last_progress fresh and should_abort() stays False
# forever. The desired behavior: only true milestones (step/plan/branch
# transitions) count as forward progress, so a thrash loop is still abortable.
#
# OBSERVED (pre-fix): feeding agent_started every 50s of fake-clock time for
# 500s total keeps should_abort() == False the whole time; the "must abort"
# assertion FAILS. -> bug.
@pytest.mark.xfail(
    reason="fuzz-longrun event_observability: bare agent_started/usage churn resets the run-stall clock — a chatty-but-stuck fallback thrash escapes the run watchdog (harness/run_monitor.py:52,204-206)",
    strict=True,
)
def test_run_watchdog_aborts_chatty_thrash_with_no_milestone():
    clk = [0.0]
    m = RunMonitor(
        run_id="r",
        heartbeat_interval=0,
        run_stall_abort=100.0,
        clock=lambda: clk[0],
    )

    aborted = False
    # 10 iterations of a worker re-spin: agent_started churn, no step/plan
    # milestone, 50s of wall-clock between each -> 500s of real no-progress.
    for _ in range(10):
        m.observe({"kind": "lifecycle", "data": {"event": "agent_started", "detail": {}}})
        m.observe({"kind": "usage", "data": {"event": "call"}})
        clk[0] += 50.0
        if m.should_abort():
            aborted = True
            break

    # Resilient contract: 500s of no real milestone progress (only re-spin
    # chatter) MUST trip the run-stall watchdog.
    assert aborted, (
        "run watchdog never aborted a 500s thrash loop emitting only "
        "agent_started/usage churn — 'stuck but chatty' escapes the watchdog"
    )


# A companion *resilient* guard: the watchdog DOES fire when the run emits only
# true raw chatter (stderr) — proving the watchdog itself works, isolating the
# defect to the over-broad progress classifier.
def test_run_watchdog_aborts_pure_stderr_chatter():
    clk = [0.0]
    m = RunMonitor(
        run_id="r",
        heartbeat_interval=0,
        run_stall_abort=100.0,
        clock=lambda: clk[0],
    )
    aborted = False
    for _ in range(10):
        m.observe({"kind": "stderr", "text": "still thinking..."})
        clk[0] += 50.0
        if m.should_abort():
            aborted = True
            break
    assert aborted, "pure-stderr stall must trip the run watchdog"


# --------------------------------------------------------------------------- #
# Probe 4 — Telegram flush() does not revive a dead drain worker.
# --------------------------------------------------------------------------- #
# _ensure_worker (re)starts the single drain daemon only when a NEW action is
# enqueued. finished() enqueues the summary card THEN calls flush(). flush()
# polls _queue.unfinished_tasks but does NOT call _ensure_worker — so the
# enqueue inside finished() is the last chance to revive the worker. If the
# worker thread has died (e.g. after an earlier poison-pill / interpreter
# teardown race) the enqueue should still revive it before flush polls. The
# desired behavior: after finished(), the guaranteed final-summary card is
# actually delivered.
#
# We simulate the long-run "drain thread died mid-stream" by force-killing the
# worker (poison pill + join) after a delivery, then send the end-of-run
# summary. A resilient notifier revives the worker and delivers the card.
def test_telegram_final_summary_survives_dead_drain_worker(tmp_path, monkeypatch):
    import json as _json
    from harness import telegram as tg

    state_file = tmp_path / "bot_state.json"
    state_file.write_text(_json.dumps({"chats": {}, "verbosity": "normal"}))
    monkeypatch.setattr(tg, "state_path", lambda path=None: str(state_file))

    sends: List[Any] = []
    lock = threading.Lock()

    class _RecordingClient:
        configured = True

        def send_message(self, chat_id, text, reply_markup=None):
            with lock:
                sends.append(("send", text))
            return {"result": {"message_id": len(sends)}}

        def pin_chat_message(self, *a, **k):
            return None

        def edit_message_text(self, chat_id, mid, text, reply_markup=None):
            with lock:
                sends.append(("edit", text))
            return None

    notifier = tg.TelegramNotifier(
        run_id="run-longhaul",
        mode="master",
        verbosity="normal",
        client=_RecordingClient(),
        chat_ids=[7],
    )

    # Bring the worker up and deliver one message, then deterministically kill
    # the drain thread (the long-run "thread died between enqueues" condition).
    notifier._broadcast("first")
    notifier.flush(timeout=2.0)
    notifier._queue.put(None)  # poison pill -> worker returns
    if notifier._worker is not None:
        notifier._worker.join(timeout=2.0)
    assert notifier._worker is None or not notifier._worker.is_alive(), (
        "precondition: drain worker should be dead before the final summary"
    )

    sends.clear()

    # End-of-run: must deliver the guaranteed final-summary card even though the
    # drain worker died mid-run.
    notifier.finished({"success": True, "run_id": "run-longhaul", "mode": "master"})
    # give the (hopefully revived) worker a moment beyond flush()'s own poll.
    notifier.flush(timeout=3.0)
    deadline = time.monotonic() + 2.0
    while not sends and time.monotonic() < deadline:
        time.sleep(0.02)

    with lock:
        delivered = list(sends)
    assert delivered, (
        "end-of-run summary card was NOT delivered after the drain worker died "
        "mid-run — finished()->flush() did not revive a dead worker"
    )


# --------------------------------------------------------------------------- #
# Probe 5 — Notify spawns an unbounded thread per step completion.
# --------------------------------------------------------------------------- #
# RunMonitor._track_step fires notifier.fire('step', ...) on every completed
# step, and Notifier.fire spawns a NEW daemon thread per call. A many-step
# graph build with retries can churn hundreds of step-completions; if the
# webhook/command is slow each thread lives up to its timeout, so a burst of
# rapid completions briefly holds many live delivery threads. The desired
# behavior: notify delivery is bounded (a pool / coalescing), not one unbounded
# thread per completion.
#
# OBSERVED (pre-fix): feeding 60 step-'completed' events while delivery blocks
# spawns ~60 concurrent threads; the "bounded thread count" assertion FAILS.
# -> bug.
@pytest.mark.xfail(
    reason="fuzz-longrun event_observability: Notifier.fire spawns one unbounded daemon thread per step completion — a burst holds dozens of live threads (harness/run_monitor.py:117-123,220-224)",
    strict=True,
)
def test_notify_thread_fanout_is_bounded_under_step_burst():
    release = threading.Event()
    started = threading.Semaphore(0)
    live = {"n": 0}
    live_lock = threading.Lock()

    class _BlockingNotifier(Notifier):
        def __init__(self):
            super().__init__(command="true", run_id="r")  # enabled

        def deliver(self, event, body):  # block so threads pile up
            with live_lock:
                live["n"] += 1
            started.release()
            release.wait(timeout=5.0)
            with live_lock:
                live["n"] -= 1

    notifier = _BlockingNotifier()
    m = RunMonitor(run_id="r", heartbeat_interval=0, notifier=notifier)

    BURST = 60
    BOUND = 16  # a pooled/coalesced notifier would cap concurrent deliveries
    try:
        for i in range(BURST):
            m.observe(
                {
                    "kind": "lifecycle",
                    "data": {
                        "orchestration": {
                            "phase": "step",
                            "action": "completed",
                            "step_index": i,
                            "step_total": BURST,
                        }
                    },
                }
            )
        # wait until all spawned deliveries have entered deliver()
        for _ in range(BURST):
            assert started.acquire(timeout=3.0), "not all notify threads started"
        with live_lock:
            peak = live["n"]
        assert peak <= BOUND, (
            f"{peak} concurrent notify threads for {BURST} step completions — "
            f"unbounded thread-per-event fan-out"
        )
    finally:
        release.set()

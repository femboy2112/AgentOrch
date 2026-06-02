# Singleton orchestrator layer — design

**Status:** built (Phase 3). **Audience:** the build agents + future Phase-4 work.

## Goal

A single long-lived **broker** process owns a **persistent build queue**. Dispatch
requests become **jobs** that the broker drains with a **concurrency cap of 2**
(at most two orchestration "lines" run at once). The broker coordinates
generator/critic provider choice across the two live lines so they don't collide
on one provider pool (the [account-sharing rule]). A second `harness serve`
**attaches/refuses** rather than spinning up a rival.

Why: today every `harness do` is an independent, uncoordinated process. Several
at once can each fan out worker chains and collectively wall a shared provider
pool or thrash the host. One broker with a bounded pool makes concurrency, usage,
and resource pressure observable and controllable in one place.

## Hard invariant — backward compatibility

The existing `dispatch()` / `dispatch_async()` in `harness/dispatch.py` are **not
changed in behavior**. The broker **wraps** them. When **no broker is running and
no `--queue` flag is given, `harness do` behaves byte-identically to today** —
same direct call, same `runs/<id>/` artifacts, same stdout. Every new capability
is **opt-in**. This invariant is tested explicitly.

## Default routing (operator decision: backward-compatible opt-in)

`python -m harness do "..."`:
- `--direct` → always run locally in-process (today's path), even if a broker is up.
- `--queue` → submit to the broker; error clearly if none is running.
- neither flag → **auto**: if a broker is reachable, submit to it and stream the
  result; otherwise run directly (today's path). Auto-detection must be cheap and
  never hang (short connect timeout; any error ⇒ fall back to direct).

## Components & file layout

```
harness/
  job_queue.py      # C1: Job model + persistent JobQueue (no IPC, no asyncio loop)
  broker.py         # C2+C3: Broker (drain loop, cap=2, pool coordination) + IPC server + singleton guard
  broker_client.py  # C3: thin unix-socket client used by the CLI
  cli.py            # C4: `serve`, `queue`, and `do` routing (additive only)
docs/
  singleton-layer-design.md   # this file
runs/.queue/        # broker state (gitignored, like runs/)
  queue.json        # persisted queue (atomic write)
  broker.sock       # unix domain socket
  broker.pid        # pidfile (singleton guard)
```

## Interface contracts (PIN THESE EXACTLY — clusters must match)

### C1 — `harness/job_queue.py`

```python
JOB_STATUSES = ("queued", "running", "done", "failed", "canceled")

@dataclass
class Job:
    id: str                 # broker-assigned, sortable: "%Y%m%d-%H%M%S-%f"[:−3] + short rand suffix
    instruction: str
    kwargs: dict            # the exact keyword args forwarded to dispatch_async (json-safe)
    status: str = "queued"
    submitted_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    run_id: str | None = None       # the dispatch run_id once it starts
    pools: list[str] = field(default_factory=list)  # provider pools this job will use
    result: dict | None = None      # compact summary on completion (success, run_id, error)
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "Job": ...

class JobQueue:
    def __init__(self, path: str | Path): ...        # path = runs/.queue/queue.json
    def load(self) -> None: ...                       # tolerate missing/garbage -> empty
    def _persist(self) -> None: ...                   # atomic: write tmp + os.replace
    def submit(self, instruction: str, kwargs: dict, pools: list[str]) -> Job: ...
    def next_runnable(self, active_pools: set[str]) -> Job | None: ...
        # FIFO over status=="queued"; prefer a job whose pools are DISJOINT from
        # active_pools; if none disjoint, return None (caller decides to wait).
    def mark_running(self, job_id: str, run_id: str | None) -> None: ...
    def mark_done(self, job_id: str, result: dict) -> None: ...
    def mark_failed(self, job_id: str, error: str) -> None: ...
    def cancel(self, job_id: str) -> bool: ...        # only if still queued
    def get(self, job_id: str) -> Job | None: ...
    def snapshot(self) -> list[dict]: ...             # all jobs as dicts, queued+running first
```
Persistence is crash-safe (atomic replace). Loading a half-written/garbage file
must not raise — degrade to empty and log. On load, any `status=="running"` job
is reset to `"queued"` (the broker that owned it died), so it re-runs.

### C2 — `harness/broker.py :: Broker`

```python
class Broker:
    MAX_CONCURRENT = 2
    def __init__(self, queue: JobQueue, *, dispatch_fn=None, cap=None, pool_wait=None): ...
        # dispatch_fn defaults to harness.dispatch.dispatch_async; injectable for tests.
        # cap (keyword-only, additive): overrides MAX_CONCURRENT for the
        #   Semaphore bound; None -> MAX_CONCURRENT (=2). `harness serve --cap N`
        #   and the C5 test both pass cap=N.
        # pool_wait (keyword-only, additive): overrides the anti-deadlock wait;
        #   None -> AGY_BROKER_POOL_WAIT env or ~5s.
    async def run(self) -> None: ...     # drain loop until stop()
    async def _run_job(self, job: Job) -> None: ...  # calls dispatch_fn(**job.kwargs), updates status
    def stop(self) -> None: ...
```
Drain loop: an `asyncio.Semaphore(MAX_CONCURRENT)` bounds live jobs. When a slot
is free, pick `queue.next_runnable(active_pools)`; if it returns None (queue empty
OR only pool-colliding jobs remain) wait on a short poll / wakeup event. Track
`active_pools` = union of running jobs' pools; clear a job's pools when it
finishes. **Anti-deadlock:** if the queue is non-empty, a slot is free, and the
only candidates collide on pools, after a bounded wait (`AGY_BROKER_POOL_WAIT`,
default ~5s) run the oldest queued job anyway and log a pool-overlap warning —
never starve. Every status transition persists via the JobQueue.

### C3 — IPC (unix socket, line-delimited JSON) + singleton guard

Request: `{"op": "...", ...}\n`  →  Response: `{"ok": bool, ...}\n`. Ops:
- `ping` → `{ok, pid, cap, running, queued}`
- `submit` `{instruction, kwargs, pools}` → `{ok, job_id}`
- `list` → `{ok, jobs:[...]}`  (queue.snapshot())
- `status` `{job_id}` → `{ok, job:{...}|null}`
- `cancel` `{job_id}` → `{ok, canceled:bool}`

The IPC layer is exposed as a `BrokerServer` class plus a `make_server()` factory
(both in `harness/broker.py`); the C5 integration test imports both:

```python
class BrokerServer:
    def __init__(self, broker: Broker, *, sock_path=None, pid_path=None): ...
        # sock_path/pid_path default to runs/.queue/{broker.sock,broker.pid}.
    async def start(self) -> None: ...        # singleton guard + bind + pidfile
    async def serve_forever(self) -> None: ... # start() + broker.run() together
    async def close(self) -> None: ...         # stop broker, unbind, cleanup
    def cleanup(self) -> None: ...             # idempotent socket+pidfile removal

def make_server(*, queue_path=None, sock_path=None, pid_path=None,
                dispatch_fn=None, cap=None) -> BrokerServer: ...
    # Convenience builder used by `harness serve`: JobQueue+Broker+BrokerServer
    # from paths; loads the queue (running -> queued reset) before serving.
```

Singleton guard: on `serve`, bind `runs/.queue/broker.sock`. If the path exists,
`ping` it — if a live broker answers, **refuse** ("broker already running, pid N")
and exit non-zero; if it's stale (no answer), unlink and bind. Write `broker.pid`.
Clean up socket+pidfile on shutdown (atexit + SIGTERM/SIGINT handler).

`broker_client.py`: `is_running(sock_path)->bool` (cheap ping), `submit(...)`,
`list_jobs()`, `status(id)`, `cancel(id)`. Short connect timeout; never hangs.

### C4 — CLI (`harness/cli.py`, additive only)

- `harness serve [--cap N]` → construct JobQueue + Broker, run the IPC server +
  drain loop (foreground; operator daemonizes via nohup/systemd). Refuses a rival.
- `harness queue` → client `list`, pretty table (id, status, mode, instruction head).
- `harness do ...` → add `--queue` / `--direct` (mutually exclusive) per the
  routing rules above. The auto path uses `broker_client.is_running()`. When
  routed to the broker, default is to **submit and wait** (poll status until
  terminal, print the same result card); `--detach` submits and prints the job id.

Resolving `pools` for a job: derive from the resolved generator/critic chains
(`harness/roles.py`) — the set of provider names (`codex`,`agy`,`claude`,`grok`)
the job can touch. Reuse existing chain-resolution; do not duplicate it.

## Testing requirements

- C1: persistence round-trip; atomic write; garbage-file ⇒ empty; running⇒queued
  reset on load; `next_runnable` FIFO + pool-disjoint preference.
- C2: with a fake `dispatch_fn`, submitting 3 jobs runs **≤2 concurrently**
  (assert observed max), all reach `done`; pool-colliding jobs don't run together
  until the anti-deadlock wait elapses; status persisted at each transition.
- C3: server+client round-trip over a tmp socket for every op; singleton guard
  refuses a second bind while the first answers `ping`; stale socket reclaimed.
- C4: arg parsing for `serve`/`queue`/`do --queue/--direct/--detach`; **the
  backward-compat path** — `do` with neither flag and no broker running calls the
  real `dispatch` exactly as before (mock dispatch, assert called with same args).
- C5: in-process end-to-end — start a broker, submit 3 jobs via the client,
  assert cap honored + all complete + queue.json persists across a simulated
  broker restart (reload picks up state).

All tests hermetic (no real worker calls, no real network; fake dispatch_fn, tmp
socket/queue paths). `runs/.queue/` is gitignored.

## Usage

The broker is **opt-in**. With no broker running and no `--queue` flag, `harness
do` behaves exactly as before (direct in-process dispatch).

```bash
# Start the singleton broker in the foreground (operator daemonizes via
# nohup/systemd). Owns runs/.queue/{queue.json,broker.sock,broker.pid}, drains
# with a concurrency cap of 2, and refuses to start if one is already running.
python -m harness serve
python -m harness serve --cap 3        # raise the concurrency cap

# Inspect the broker's build queue (id, status, mode, instruction head).
python -m harness queue

# Dispatch routing on `do` (additive flags):
python -m harness do "INSTRUCTION"            # auto: submit to broker if one is
                                              #   reachable, else run locally
python -m harness do "INSTRUCTION" --direct   # always run locally (today's path)
python -m harness do "INSTRUCTION" --queue    # require the broker; error if none
python -m harness do "INSTRUCTION" --queue --detach   # submit and print job id,
                                                      #   don't wait (--detach needs
                                                      #   broker routing)
```

`--queue` and `--direct` are mutually exclusive. With neither flag, auto-detection
is cheap and never hangs: if the broker socket answers `ping`, the job is
submitted and the CLI waits for it (printing the same result card as a local run);
otherwise it falls back to running locally. `--detach` submits and prints only the
job id (it requires broker routing, so it's incompatible with `--direct`).

## Out of scope (later phases)

- Dashboard rewire to read broker state live — **Phase 4** (the wiring surface
  this builds is the prerequisite). Do not touch the dashboard now.
- Multi-host / distributed queue. Single-host, single-broker only.

# AgentOrch Control Dashboard — design spec (v1)

This is the source-of-truth design for the AgentOrch control dashboard. It is
intentionally complete enough that a worker CLI can build the v1 directly from
this file plus the existing repo. Anything not in this spec is implementer's
choice as long as it doesn't contradict what is.

## 1. Purpose & scope

A single-user, localhost-only web dashboard that gives the operator complete
manual control of the orchestrator: launching dispatches with full mode /
chain / verifier control, watching the underlying workers **think in real
time**, reviewing past runs, and replaying captured thinking-streams. v1 ships
four pages — Dispatch, Live, Runs, Run-detail. Workers / Calibration / Bench
/ Settings are explicitly deferred to v2.

The dashboard is the marquee operator UX; the existing CLI continues to work
unchanged and shares all the same primitives (harness.dispatch, runs/, etc.).

## 2. Architecture

```
Browser  ◄──── SSE ─────  FastAPI server (in-process asyncio)
   ▲                          │
   │   REST                   ├── harness.dispatch.dispatch()
   └────────────────────────► │   wired with event_callback that
                              │   publishes WorkerEvents to per-run
                              │   asyncio.Queues read by the SSE endpoint
                              │
                              └── AgentInstance._stream_communicate()
                                  per-worker adapters parse the native
                                  stream and emit WorkerEvents
```

**Tech stack** (locked):
- Backend: FastAPI + uvicorn (Python 3.10+, async)
- Frontend: vanilla HTML / CSS / JS, no framework, no build step
- Transport: REST for state, SSE for live streams
- Bind: `127.0.0.1` only, no auth
- Launch: `python -m dashboard` (also reachable via `python -m harness dashboard`)

**Exec model** (locked): in-process async tasks. Each dispatch runs as a
background `asyncio.Task` in the same event loop as the SSE producer. Restart
kills in-flight runs — acceptable for a single-user dev tool.

**Concurrency** (locked): unlimited parallel dispatches. The Live page shows N
parallel streams; account contention is the operator's call, and the streaming
watchdog (already shipped) catches runaway cases.

## 3. The WorkerEvent — uniform stream shape

The single unifying primitive. Every worker's native stream output gets parsed
by its adapter into this shape; everything downstream (SSE, persisted JSONL,
frontend renderer) sees only this:

```json
{
  "ts": 1700000000.123,           // float, epoch seconds
  "run_id": "20260527-225930-001",
  "worker": "codex",              // codex | claude | agy | grok
  "model": "gpt-5.5",
  "effort": "high",               // or "n/a"
  "branch": null,                 // for ToT/master multi-branch (int) or null
  "kind": "reasoning",            // see below
  "text": "...",                  // for reasoning|message|stderr|watchdog
  "data": { ... }                 // for usage|tool_call|tool_result|lifecycle
}
```

`kind` enum:

| kind | meaning | populates |
|---|---|---|
| `lifecycle` | dispatch started / agent started / agent finished / dispatch finished | `data.event`, `data.detail` |
| `reasoning` | streamed model reasoning text (continuous chunks) | `text` |
| `message` | model's final / intermediate user-visible message | `text` |
| `tool_call` | the model invoked a tool | `data.name`, `data.args` (summary) |
| `tool_result` | result returned to the model | `data.name`, `data.summary` |
| `usage` | token / cost accounting (mid- or end-of-call) | `data.in_tokens`, `data.out_tokens`, `data.reasoning_tokens`, `data.cost_usd?`, `data.api_ms?` |
| `stderr` | raw stderr line (fallback / opaque workers) | `text` |
| `watchdog` | streaming watchdog tripped | `text` (marker), `data.reason` |

Adapters are best-effort: a worker without rich telemetry (agy) emits mostly
`stderr` + a final `message` + a `lifecycle` event. The renderer must handle
sparse streams gracefully.

## 4. Backend

### 4.1 Hook in `agy_orchestrator/core/agent.py`

Add to `AgentInstance.__init__`:
```python
self.event_callback: Optional[Callable[[dict], None]] = None
```

Add a private helper:
```python
def _emit_event(self, event: dict) -> None:
    cb = self.event_callback
    if cb is None: return
    try:
        cb(event)
    except Exception:    # never let UI plumbing break execution
        pass
```

The existing `_stream_communicate` already drains stdout line-by-line; per-worker
adapters call `_emit_event` from inside their override or from a shared
`_postprocess_line` hook. Default `kind="stderr"` for unparsed stderr lines.

### 4.2 Per-worker adapters (`dashboard/adapters/*.py` OR extend each agent file)

Each adapter exposes one function:

```python
def parse_stream_line(line: str, ctx: AdapterCtx) -> list[dict]:
    """Parse one stdout line into 0+ WorkerEvent dicts (sans ts/run_id/worker)."""
```

Adapter `ctx` carries running state (e.g. current tool_call awaiting result).
The harness's event-bus wrapper stamps `ts`, `run_id`, `worker`, `model`,
`effort`, `branch` onto each emitted event before pushing.

**codex**: parse JSONL events; map `turn.started → lifecycle`, reasoning
chunks → `reasoning`, `item.tool_call → tool_call`, results → `tool_result`,
`item.message → message`, `turn.completed → usage + lifecycle`.

**claude**: switch CLI from `--output-format json` to `--output-format
stream-json` **for dashboard dispatches only** (the existing CLI path stays
on json). Parse per-message blocks; thinking blocks → `reasoning`, text
blocks → `message`, tool_use → `tool_call`, tool_result → `tool_result`,
final result → `usage + lifecycle`.

**agy**: line-by-line stderr → `stderr` events, final stdout → `message`,
end → `lifecycle`. No tool/reasoning visibility (agy doesn't expose it).

**grok**: parse JSON envelope (already does in GrokAgent), emit `thought` →
`reasoning`, message → `message`, `stopReason` → `lifecycle`.

### 4.3 Event bus (`dashboard/event_bus.py`)

```python
class EventBus:
    """Per-run-id asyncio.Queue. Publishers push WorkerEvents; subscribers
    drain them. Closes on dispatch completion. Also fans out to a sink list
    (e.g. JSONL writer) so persistence is decoupled from SSE consumers."""

    queues: dict[str, asyncio.Queue]
    sinks:  dict[str, list[Callable[[dict], None]]]
    closed: set[str]

    def publisher_for(self, run_id, *, worker, model, effort, branch=None) -> Callable[[dict], None]:
        """Returns a callback suitable for AgentInstance.event_callback. The
        callback stamps the trailing fields and enqueues."""

    async def subscribe(self, run_id) -> AsyncIterator[dict]:
        """Drain events until the queue is closed."""

    def add_sink(self, run_id, sink: Callable[[dict], None]) -> None: ...

    def close(self, run_id) -> None: ...
```

A single bus instance lives at app startup. The dispatch path:

1. Generates `run_id` (existing behaviour).
2. Calls `bus.publisher_for(run_id, worker=..., model=..., effort=...)` for
   each agent it constructs, sets `agent.event_callback = publisher`.
3. Registers a JSONL sink writing to `runs/{run_id}/events.jsonl`.
4. Runs the workflow in a background task.
5. On completion, calls `bus.close(run_id)`.

### 4.4 REST + SSE surface

All routes under `/api/`.

```
POST /api/dispatch
  body: {
    instruction: str,
    mode: "direct"|"adversarial"|"feedback"|"cascade"|"master",
    generator_chain: ["codex","agy"],
    critic_chain: ["agy","codex"]   // optional, only adversarial uses
    test_cmd?: str,
    web_search?: bool,
    fallback?: bool = true,
    cycles?: int = 2,
    max_iterations?: int = 5,
    branches?: int = 3,
  }
  returns: { run_id: str }
  side effect: dispatch starts as bg asyncio.Task

GET  /api/runs?limit=50&before=<run_id>&q=<search>&mode=&success=&worker=
  returns: { runs: [{ run_id, mode, generator, critic?, duration_s, success,
                       quality, changed_files_count, instruction_preview,
                       started_at }], next_before: str|null }

GET  /api/runs/{run_id}
  returns: full meta.json + artifact paths

GET  /api/runs/{run_id}/artifact/{name}
  name in: prompt | stdout | stderr | diff | events
  returns: raw text (or NDJSON for events). Honors Range header for tailing.

DELETE /api/runs/{run_id}
  returns: { deleted: true }

GET  /api/live
  returns: { running: [{ run_id, started_at, mode, generator, critic? }] }

POST /api/live/{run_id}/kill
  returns: { killed: bool }

GET  /api/sse/{run_id}
  SSE stream of WorkerEvents. Events:
    event: worker_event
    data:  <JSON of WorkerEvent>
  When the dispatch finishes:
    event: done
    data:  <final meta.json>
  Reconnect protocol: client passes Last-Event-ID; server replays from
  events.jsonl past that ID then resumes live (use array index as ID).

GET  /api/sse/recent
  SSE stream of completion notices for the "recently completed" pinning on Live.

GET  /healthz   → "ok"
```

### 4.5 File / module layout

```
dashboard/
  __init__.py
  __main__.py            # uvicorn launcher (default 127.0.0.1:8765, --port, --no-browser)
  server.py              # FastAPI app factory; lifespan = start/stop event_bus
  event_bus.py           # EventBus class
  routers/
    dispatch.py
    runs.py
    live.py
    health.py
  adapters/
    __init__.py
    base.py              # AdapterCtx, helpers
    codex.py
    claude.py
    agy.py
    grok.py
  static/
    index.html
    css/{tokens.css, app.css}
    js/{app.js, api.js, sse.js, router.js,
        pages/{dispatch.js, runs.js, run_detail.js, live.js},
        components/{stream.js, run_row.js, chip.js, toast.js, button.js}}
harness/
  cli.py                 # +`dashboard` subcommand → exec dashboard.__main__
agy_orchestrator/core/
  agent.py               # +event_callback hook (~10 LOC)
agy_orchestrator/core/agents/
  codex_agent.py         # +emit events from JSONL parse
  claude_agent.py        # +stream-json mode for dashboard dispatches
  agy_agent.py           # +per-line stderr emit
  grok_agent.py          # +emit thought events
tests/
  test_dashboard_api.py        # FastAPI TestClient smoke
  test_event_adapters.py       # per-worker parse fixtures
  test_dashboard_event_bus.py  # publish/subscribe/close/replay
  test_dashboard_e2e.py        # stub-agent dispatch → drain SSE → assert
```

### 4.6 Persistence

Every dispatch creates `runs/<id>/events.jsonl` in addition to the existing
artifacts. Each line is a complete WorkerEvent. The file is **append-only
during the run** (the JSONL sink writes synchronously per event). The
Run-detail page replays this through the same renderer used by Live.

## 5. Frontend

### 5.1 Sitemap

```
/                      → redirect to /dispatch
/dispatch              → form, fires POST /api/dispatch
/live                  → list of running dispatches; each is a collapsible card
                         with embedded thinking-stream. Recently-completed
                         dispatches pin here for 60s.
/runs                  → filterable list
/runs/<id>             → tabs: Stream | Prompt | Stdout | Stderr | Diff | Quality
```

Hash-based routing in vanilla JS (no library). The static index.html is the
only HTML served; pages are JS-rendered.

### 5.2 Visual tokens (`static/css/tokens.css`)

```css
:root {
  /* color (dark mode default; light is a class on <html>) */
  --bg-0: #0a0a0c;  --bg-1: #131316;  --bg-2: #1c1c21;
  --line: #2a2a31;
  --fg-0: #e8e8ea;  --fg-1: #9c9ca3;  --fg-2: #5c5c64;
  --accent: #7cc4ff;
  --good: #6fcf97;  --warn: #f2c94c;  --bad: #eb5757;
  /* worker tags */
  --codex:  #6c9eff;  --claude: #ff9b6c;
  --agy:    #7fd49a;  --grok:   #b48cff;
  /* type */
  --font-ui:   ui-sans-serif, system-ui, -apple-system, sans-serif;
  --font-mono: ui-monospace, "JetBrains Mono", Consolas, monospace;
  /* spacing 4px ramp */
  --s-1: 4px; --s-2: 8px; --s-3: 12px; --s-4: 16px; --s-6: 24px; --s-8: 32px;
  /* radii */
  --r-1: 4px; --r-2: 8px; --r-3: 12px;
}
html.light { /* same tokens, light values; toggle via header button */ }
```

Reasoning text: `--font-mono`, 14px, line-height 1.55, max-width ~80ch.
Body chrome: `--font-ui`.

### 5.3 Hybrid stream renderer (`components/stream.js`)

The **default view** is DeepSeek-style continuous reasoning text. Non-reasoning
events appear inline as compact one-line markers in the same flow:

```
14:23:01  codex / gpt-5.5 / high  ·  reasoning ▾
Looking at the existing roles.py to understand how agents are constructed.
The fallback wrapper builds sub-agents via _make_sub, so the watchdog
arming needs to thread through there. Let me read the current shape…

  ▸ read_file  harness/roles.py                            (134 lines)

…The post_construct_hook pattern is right. Now to set up the lookup…

  ▸ apply_patch  harness/roles.py                          (+34, -7)
  ▴ usage  in=1,204  out=287  reasoning=12

14:23:23  agy / pro / high  ·  reasoning ▾
Reviewing the proposed change. The arm_hook closure captures the per-class
config from the outer scope, so each provider gets its own budget lookup…
```

Rules:
- Reasoning events from the same agent in a short window (≤1.5s gap) merge
  into one continuous paragraph (no double timestamps).
- Tool-call markers are single lines with `▸`; results with `◂`; usage with
  `▴`; watchdog trips with a red `⚠ [watchdog:verbose]` banner.
- Auto-scroll to bottom unless the user scrolled up (then a "▼ jump to live"
  pill appears in the bottom-right).
- Filter pills at top of stream: `reasoning` `messages` `tools` `usage`
  `stderr`. Default all on.
- **Toggle to Cards view**: same events, rendered as discrete cards (one per
  event). Same renderer module, just different layout class.

### 5.4 Multi-branch layout (ToT / master)

Default: **vertical timelines**, branches stacked, each with a `branch=N` tag
prefix on every event card / paragraph. Scales to any N.

Toggle: **Split view** — branches in side-by-side columns. Auto-disable
when branches > 3 (column width becomes unreadable).

### 5.5 Page-specific layouts

**/dispatch** — see §6 sketch in the design conversation; concretely:

- Instruction textarea (14 rows, mono)
- Mode select; generator/critic chain text inputs with provider suggest
- Test cmd input; checkboxes for web-search, no-fallback, "Stay on this page"
- Live budget panel showing what the watchdog will arm for the selected chain
- Big DISPATCH button
- Last-3-dispatches summary clickable to the run-detail

Submit posts `/api/dispatch`. On 200:
- "Stay" unchecked → window.location.hash = `#/live` (and Live auto-expands the new card)
- "Stay" checked → form stays put, embedded thinking-stream renders below

**/live** — list of currently-running dispatches as collapsible accordion cards.
Each card auto-expanded if there's only one. Header shows mode, gen/critic,
elapsed, token count, watchdog budget %. Body is the stream renderer. Pinned
recently-completed cards (60s) have a 📌 toggle for indefinite pinning.

**/runs** — table with search box + filter chips. Server-paginated 50/page;
"load more" button. Each row clicks to `/runs/<id>`.

**/runs/<id>** — header strip (mode, chains, duration, quality, files); body
is a tab bar [Stream | Prompt | Stdout | Stderr | Diff | Quality]. The Stream
tab uses the same renderer as Live, fed from `runs/<id>/events.jsonl` instead
of an SSE feed.

### 5.6 SSE handling (`js/sse.js`)

Thin EventSource wrapper:
- Auto-reconnect with exponential backoff on disconnect.
- Tracks Last-Event-ID for replay-on-reconnect (server reads it).
- Exposes `subscribe(run_id, onEvent, onDone, onError)` returning a `cancel()`.

## 6. Build phases (the worker's commit milestones)

Each phase is a green commit. The worker is instructed to commit at each
boundary (small batches are easier to review). The verifier runs at the end
of the dispatch; intra-phase the worker is trusted to keep tests green.

**Phase 1 — Event bus + per-worker adapters + persistence**
- `agy_orchestrator/core/agent.py` event_callback hook
- Per-worker adapters extending each `*_agent.py`
- `dashboard/event_bus.py`
- JSONL persistence into `runs/<id>/events.jsonl`
- Tests: `test_event_adapters.py` (per-worker fixtures → expected events),
  `test_dashboard_event_bus.py` (publish/subscribe/close/replay).
- Acceptance: `pytest -q` green; an in-process stub dispatch writes a valid
  events.jsonl.

**Phase 2 — FastAPI server skeleton + runs / dispatch / sse endpoints**
- `dashboard/server.py`, all routers, `__main__.py`.
- Wire `POST /api/dispatch` to `harness.dispatch.dispatch()` in a background
  task; wire `GET /api/sse/{run_id}` to the event bus.
- Tests: `test_dashboard_api.py` (TestClient: dispatch with stub agent →
  drain SSE → assert events arrive in order).
- Acceptance: `pytest -q` green; `python -m dashboard` boots, `/healthz` 200.

**Phase 3 — Frontend shell + Dispatch + Live**
- `static/index.html`, `tokens.css`, `app.css`, `app.js`, `router.js`,
  `api.js`, `sse.js`.
- Pages: dispatch.js, live.js. Component: stream.js (hybrid renderer).
- Manual: open browser, dispatch a stub instruction, watch the stream.
- Acceptance: `pytest -q` green; smoke documented in a one-paragraph README.

**Phase 4 — Runs + Run-detail + dark/light toggle**
- pages: runs.js, run_detail.js.
- run_detail.js Stream tab replays from `runs/<id>/events.jsonl` via the
  same renderer.
- Theme toggle in the header.
- Acceptance: `pytest -q` green; manually open a few past runs, exercise tabs.

**Phase 5 — Polish + harness CLI integration + docs**
- `harness/cli.py` adds `dashboard` subcommand that execs
  `python -m dashboard`.
- `dashboard/README.md` with: launch, screenshots/wireframes, the v2 backlog
  (workers / calibration / bench / settings).
- Acceptance: `python -m harness dashboard` opens the browser to the working
  dashboard.

## 7. Test strategy

Hard gate: `--test-cmd "python -m pytest -q"` on the dispatch. The verifier
must pass for the dispatch to be considered done. New tests required:

| File | Covers |
|---|---|
| `test_event_adapters.py` | Each worker's adapter on a captured-stream fixture → expected list of WorkerEvents |
| `test_dashboard_event_bus.py` | publish/subscribe/close; reconnect replay via Last-Event-ID |
| `test_dashboard_api.py` | TestClient: GET /healthz, POST /api/dispatch (stub agent), GET /api/runs, GET /api/runs/{id} |
| `test_dashboard_e2e.py` | Full in-process dispatch with stub agent → drain SSE → assert events arrive ordered + a `done` terminator |

Existing 70-test suite must remain green.

## 8. Implementation gotchas the worker must respect

1. **AgentInstance.event_callback must never raise back into the run loop**.
   The `_emit_event` helper already swallows exceptions; adapters must too.
2. **claude `--output-format stream-json` is dashboard-only**. The existing
   CLI path still uses `--output-format json` (so existing tests keep
   passing). Gate the stream-json mode behind an attribute / flag set by the
   dispatch path when wiring the callback.
3. **Streaming watchdog already armed by the harness** (`harness/roles.py`
   `_arm_watchdog`). Dashboard dispatches go through the same path, so the
   watchdog Just Works — but: when it trips, surface the trip as a
   `kind="watchdog"` event so the frontend can render the red banner.
4. **Multi-branch master/ToT**: each branch's agent gets a separate
   `event_callback` whose `branch=N` field is pre-stamped. The bus keys
   events by `run_id`; the frontend separates by `branch`.
5. **events.jsonl is APPEND-ONLY** during the run. Reading it for the
   Run-detail Stream tab must use a streaming JSON parser that tolerates a
   partial last line (don't `json.load(open(f).read())`).
6. **Bind to 127.0.0.1**, not 0.0.0.0. Single-user local tool.
7. **The dispatch chain for the BUILD itself excludes claude** —
   `codex,agy,grok` cycled. Reason: the operator's driving agent is
   claude; sharing the pool would cascade a usage wall. Documented in
   `CLAUDE.md` "Account-sharing rule".

## 9. Out of v1 (explicit non-goals)

- `/workers` page (per-provider model/usage cards) — v2.
- `/calibration` page (budget table editor) — v2.
- `/bench` page (cloud_eval/model_sweep/calibrate launcher) — v2.
- `/settings` page (env vars, default chains) — v2.
- Multi-user auth / network exposure.
- Persistent task queue across server restarts.
- Mobile responsive layout (desktop-first; reasonable shrinkage to ~900px
  is fine, below that is acceptable degradation).
- Editing past runs.
- Diffing two runs against each other.

## 10. Acceptance checklist

A dispatch is considered complete when:

- [ ] `python -m pytest -q` green on the new + existing tests
- [ ] `python -m dashboard` boots on `127.0.0.1:8765`, `/healthz` returns 200
- [ ] `python -m harness dashboard` opens the browser to the dashboard
- [ ] POST `/api/dispatch` with a stub instruction returns a `run_id` and
      `runs/<id>/events.jsonl` is created
- [ ] `/live` shows the running dispatch with a streaming reasoning view
- [ ] `/runs` lists prior runs; `/runs/<id>` shows tabs incl. Stream replay
- [ ] Dark mode is default; light mode toggle works
- [ ] `dashboard/README.md` documents launch + v2 backlog

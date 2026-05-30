## 1. Overview
AgentOrch’s existing dashboard will be refactored into a public-release-quality UI that keeps the current pages (`Dispatch`, `Live`, `Runs`, `Run-detail`) and theme system, while adding a runtime presentation toggle with two views over one shared live stream:

- **PROFESSIONAL mode**: dense, complete operational telemetry (runs, events, tokens/cost, diffs).
- **USER-FRIENDLY mode**: an animated, low-clutter plan/progress diagram for non-experts.

The redesign stays on the current stack (FastAPI + vanilla JS SPA + SSE EventBus) and uses the existing `WorkerEvent` stream as the single source for live state [goal-derived]. Value is delivered by making the same orchestration truth understandable to both expert and non-expert operators without changing backend transport or operational model [goal-derived].

## 2. Goals & Non-Goals
**Goals**
- Provide two runtime-switchable dashboard modes backed by the same run/event data stream [goal-derived].
- Visualize master workflow progression and detours live: planner, ToT branch selection, adversarial iteration `k/N`, step `N/M`, fallback/reroute [goal-derived].
- Keep UI responsive under sustained event flow (no blocking interactions) [goal-derived].
- Preserve existing pages, routing, dark/light theme, and no-build SPA architecture [goal-derived].
- Add `--out-dir` support to Dispatch page UX and API contract [goal-derived].
- Add computer-use dogfood validation (OBSERVE as oracle, REAL as interactive driver) as a first-class acceptance mechanism [goal-derived].

**Non-Goals**
- Replacing SSE with WebSockets or introducing additional transports.
- Rewriting frontend in React/Vue/Svelte or introducing a build toolchain.
- Adding auth/RBAC/multi-tenant access control (dashboard remains localhost-only).
- Changing orchestrator core workflow semantics beyond additive event metadata needed for visualization compatibility.

## 3. Requirements
### Functional Requirements
- **FR-01**: WHEN any dashboard page renders THEN the system SHALL expose a global presentation-mode toggle (`professional|friendly`) without changing route structure.
- **FR-02**: WHEN the operator toggles presentation mode THEN the visible run visualization SHALL switch in-place without reconnecting SSE or losing buffered events [goal-derived].
- **FR-03**: WHEN `WorkerEvent` items arrive THEN the system SHALL append them to a shared per-run in-memory event store used by both modes [goal-derived].
- **FR-04**: WHEN PROFESSIONAL mode is active THEN the system SHALL render complete stream details including lifecycle, reasoning, message, tool events, usage (tokens/cost), stderr/watchdog, and diff links.
- **FR-05**: WHEN USER-FRIENDLY mode is active THEN the system SHALL render a dynamic plan/progress diagram driven only by `WorkerEvent` data [goal-derived].
- **FR-06**: WHEN a master-mode run emits orchestration transitions THEN the diagram SHALL animate the sequence: `master plan -> ToT branch pick -> adversarial iteration k/N -> step N/M -> fallback/reroute`.
- **FR-07**: IF a run replay (`runs/<id>/events.jsonl`) lacks new orchestration metadata THEN the system SHALL render a degraded but valid inferred timeline and label inferred segments.
- **FR-08**: WHEN the frontend consumes live updates THEN it SHALL use existing SSE endpoints only and SHALL NOT introduce periodic polling loops for stream updates [goal-derived].
- **FR-09**: WHEN the Dispatch form is submitted with `output directory` THEN `/api/dispatch` SHALL accept and forward `out_dir` to harness dispatch.
- **FR-10**: IF `out_dir` is invalid (non-existent and non-creatable, non-directory, or permission denied) THEN `/api/dispatch` SHALL return `400` with machine-parseable error code and human-readable message [industry-default].
- **FR-11**: WHEN new orchestration fields are added to events THEN they SHALL be additive/optional so existing event consumers and replay behavior remain valid.
- **FR-12**: WHEN a master dispatch is running for dogfood validation THEN a Phase-1 computer-use worker in **OBSERVE** mode SHALL read dashboard state on display `:0` via DOM/AT-SPI/OCR and produce oracle observations [goal-derived].
- **FR-13**: WHEN dogfood interactivity validation runs THEN a Phase-1 computer-use worker in **REAL** mode SHALL switch dashboard presentation modes, navigate tabs, and trigger a dispatch to verify interactive behavior [goal-derived].
- **FR-14**: WHEN the dashboard server starts THEN it SHALL bind localhost only (`127.0.0.1`) and remain unauthenticated, matching current behavior.
- **FR-15**: WHEN SSE reconnects with `Last-Event-ID`/`after_id` THEN state reconstruction in both modes SHALL be deterministic and gap-free for persisted events.
- **FR-16**: WHEN a fallback/reroute occurs in worker/provider execution THEN a structured lifecycle event SHALL be emitted so both modes can represent the detour [goal-derived].

### Non-Functional Requirements
- **Responsiveness**: UI input-to-response latency p95 `<100ms` under 50 events/sec burst per visible run [assumption].
- **Render cadence**: diagram layout/animation recompute cadence `2–4 Hz`; event ingestion remains continuous [assumption].
- **Scale**: support at least 8 concurrent live runs in one browser session without UI lockups [industry-default].
- **Reliability**: SSE reconnect/replay must preserve ordering and avoid duplicate state transitions.
- **Security posture**: localhost-only/no-auth, no broadened network exposure, no new privileged browser capabilities.
- **Compatibility**: existing pytest suite remains green; architecture remains dependency-light and no-build.

## 4. Architecture
Single-stream architecture is retained and extended:

- **Source of truth**: `WorkerEvent` stream (live SSE + replay JSONL) [goal-derived].
- **Backend**: existing FastAPI routers + EventBus + harness dispatch; add additive orchestration lifecycle emission points.
- **Frontend**: one shared `RunEventStore` and two renderers (`ProfessionalRenderer`, `FriendlyPlanRenderer`) selected by `PresentationModeController`.

```text
harness/workflows + fallback agent
  -> WorkerEvent (existing kinds, additive lifecycle detail)
  -> EventBus (existing)
  -> /api/sse/{run_id} (existing transport)
  -> RunEventStore (shared in browser)
      -> ProfessionalRenderer (dense stream/metrics)
      -> FriendlyPlanRenderer (animated plan graph)
```

Key architectural decisions:
- Add orchestration semantics as **optional lifecycle payload fields**, not a new transport [goal-derived].
- Keep one subscription per run and fan out to both renderers in-memory [goal-derived].
- Keep current pages/routes/theme; mode toggle changes view composition only [goal-derived].

## 5. Components & Interfaces
### Backend Components
- **Dispatch API (`POST /api/dispatch`)**
  - Responsibility: start dispatch tasks.
  - Input (JSON): existing fields + `out_dir?: string`.
  - Output: `{ run_id: string }` or `400 { code, message }`.
  - Interaction: called by Dispatch page; forwards `out_dir` to `dispatch(...)`.

- **Orchestration Event Emitter (new additive logic in workflows/fallback)**
  - Responsibility: emit structured lifecycle transitions for master-mode visualization.
  - Interface:
    - Emits `WorkerEvent.kind="lifecycle"` with:
      - `data.event: "orchestration_transition"`
      - `data.orchestration: OrchestrationTransition`
  - Interaction: consumed by EventBus unchanged; ignored safely by legacy renderers.

- **SSE Live/Replay (`GET /api/sse/{run_id}`)**
  - Responsibility: replay persisted events and stream live events.
  - Interface: unchanged SSE event type `worker_event` + `done`.
  - Interaction: browser SSE client; supports `after_id` resume.

### Frontend Components
- **PresentationModeController**
  - Responsibility: global mode state (`professional|friendly`) persisted in `localStorage`.
  - Interface:
    - `getMode(): Mode`
    - `setMode(mode: Mode): void`
    - Emits `modechange`.
  - Interaction: used by `Live`, `Run-detail` stream tab, and optionally Dispatch embedded stream.

- **RunEventStore (shared)**
  - Responsibility: normalized per-run event buffer and derived counters.
  - Interface:
    - `append(runId, WorkerEvent, eventId)`
    - `snapshot(runId): RunProjection`
  - Interaction: fed by SSE client; read by both renderers.

- **ProfessionalRenderer**
  - Responsibility: dense expert telemetry presentation.
  - Input: `RunProjection`.
  - Output: DOM subtree for stream/cards/metrics.
  - Interaction: existing `StreamRenderer` evolution; no data ownership.

- **FriendlyPlanRenderer**
  - Responsibility: animated uncluttered plan/progress diagram.
  - Input: `RunProjection` + orchestration transitions.
  - Output: DOM/SVG diagram with animated nodes/edges/status.
  - Interaction: reads same projection; no direct SSE calls.

- **RenderScheduler**
  - Responsibility: non-blocking UI updates with micro-batching and frame/time slicing.
  - Interface:
    - `scheduleEventBatch(runId)`
    - `flush(maxMsBudget)`
  - Interaction: decouples event ingestion from paint cadence.

- **DispatchForm Extension**
  - Responsibility: include output directory entry.
  - UI field: `Output Directory (--out-dir)` text input.
  - Contract: sends `out_dir` in payload when non-empty.

### Dogfood Acceptance Components
- **Dogfood Orchestrator (test harness)**
  - Responsibility: coordinate real master dispatch + OBSERVE/REAL computer-use checks.
  - Interface: pytest/e2e entry with run id and expected transition stream.
  - Interaction:
    - OBSERVE worker reads rendered dashboard as oracle.
    - REAL worker drives UI operations.
    - Comparator asserts UI state matches event-stream-derived truth with bounded lag.

## 6. Data Models
```ts
type WorkerEventKind =
  | "lifecycle" | "reasoning" | "message" | "tool_call"
  | "tool_result" | "usage" | "stderr" | "watchdog";
```

```ts
interface WorkerEvent {
  ts: number;
  run_id: string;
  worker: string;
  model: string;
  effort: string;
  branch: number | null;
  kind: WorkerEventKind;
  text: string;
  data: Record<string, unknown>; // additive-compatible
}
```

Additive lifecycle extension:
```ts
interface OrchestrationTransition {
  workflow: "master";
  phase:
    | "plan"
    | "tot"
    | "adversarial"
    | "step"
    | "fallback";
  action:
    | "started"
    | "completed"
    | "branch_selected"
    | "iteration_started"
    | "iteration_completed"
    | "reroute";
  step_index?: number;      // 1-based
  step_total?: number;
  iteration?: number;       // 1-based
  iteration_total?: number;
  selected_branch?: number; // 1-based
  from_worker?: string;
  to_worker?: string;
  reason?: string;
}
```

Lifecycle payload:
```ts
interface LifecycleData {
  event: string;                 // existing
  detail?: unknown;              // existing
  orchestration?: OrchestrationTransition; // new optional
}
```

Dispatch request extension:
```ts
interface DispatchRequest {
  instruction: string;
  mode: "direct" | "adversarial" | "feedback" | "cascade" | "master" | "pat" | "vote" | "auto";
  generator_chain?: string[];
  critic_chain?: string[];
  test_cmd?: string;
  web_search?: boolean;
  fallback?: boolean;
  cycles?: number;
  max_iterations?: number;
  branches?: number;
  out_dir?: string; // new
}
```

Frontend projection:
```ts
interface PlanNode {
  id: string;
  type: "plan" | "tot" | "step" | "iteration" | "fallback";
  label: string;
  status: "pending" | "active" | "done" | "failed";
  started_ts?: number;
  ended_ts?: number;
  meta?: Record<string, unknown>;
}
interface RunProjection {
  run_id: string;
  events: WorkerEvent[];
  nodes: PlanNode[];
  edges: Array<{from: string; to: string;}>;
  counters: {in_tokens: number; out_tokens: number; cost_usd?: number;};
  inferred: boolean;
}
```

Dogfood oracle record:
```ts
interface DashboardObservation {
  ts: number;
  run_id: string;
  mode: "professional" | "friendly";
  route: "#/dispatch" | "#/live" | "#/runs" | `#/runs/${string}`;
  visible_plan_signature: string; // normalized extracted sequence
}
```

## 7. Error Handling & Failure Modes
- **Missing orchestration metadata in older runs**
  - Response: fallback to inferred timeline; mark `inferred=true`; no hard failure.
- **SSE disconnect/reconnect**
  - Response: resume with `after_id`; deduplicate by event id; continue rendering.
- **High event burst**
  - Response: ingestion queue remains lossless; renderer time-slices per frame; low-priority visuals may skip intermediate animation frames but not final state [industry-default].
- **`out_dir` invalid**
  - Response: reject dispatch with `400`; form remains populated; inline error shown.
- **Malformed event rows in replay**
  - Response: skip bad row (existing tolerant behavior), keep stream alive.
- **Orchestration emitter missing for a workflow path**
  - Response: professional mode unaffected; friendly mode shows partial graph and explicit “partial telemetry” badge.
- **Dogfood REAL gating denial**
  - Response: test records explicit permission/gate failure reason; marks interactivity validation inconclusive, not silently pass.

## 8. Testing Strategy
- **Unit strategy**
  - Validate event-to-projection logic (including branch selection, iteration counters, reroute transitions).
  - Validate backward compatibility with legacy events lacking `data.orchestration`.
  - Validate dispatch API contract extension for `out_dir` validation and pass-through.
  - Validate render scheduler non-blocking behavior under synthetic burst streams.

- **Integration strategy**
  - FastAPI `TestClient` verifies `/api/dispatch` + `/api/sse/{run_id}` replay continuity and additive lifecycle payload tolerance.
  - Browserless JS tests validate shared store feeds both renderers identically from one event stream.
  - Existing dashboard routes/theme tests remain unchanged and must pass.

- **End-to-end strategy**
  - Live run E2E with real master mode:
    - PROFESSIONAL mode correctness against raw event stream.
    - USER-FRIENDLY diagram transition correctness and animation state.
    - Mode switching during active stream without reconnect or data loss.
    - Dispatch with `out_dir` targeting a non-AgentOrch repo path.

- **Computer-use dogfood (first-class acceptance)**
  - OBSERVE pass: Phase-1 worker reads dashboard on display `:0` via DOM/AT-SPI/OCR and emits normalized observed plan/progress signatures.
  - REAL pass: Phase-1 worker drives mode switches, tab navigation, and dispatch trigger.
  - Comparator pass: observed signatures must match event-stream-derived expected sequence with bounded lag; UI actions remain responsive during stream.
  - This path is a release gate for interactivity + truthfulness of friendly diagram [goal-derived].

## 9. Constraints & Guardrails
- Must remain FastAPI + vanilla JS + no build tooling.
- Must reuse existing SSE/EventBus transport; no second live transport.
- Must preserve existing pages and dark/light theme behavior.
- Must keep localhost-only/no-auth hosting behavior.
- Must keep WorkerEvent compatibility with existing `runs/<id>/events.jsonl` replay.
- Must keep UI non-blocking under continuous live updates.
- Must not create polling storms; live updates are SSE-driven.
- Must add `out_dir` in Dispatch UX/API as part of redesign, not separate feature work.
- Must keep current pytest suite green.

## 10. Alternatives Considered
- **Full frontend rewrite with React + graph library**: rejected due to no-build/dependency-light constraint and risk to existing dashboard pages.
- **WebSocket transport for live updates**: rejected because SSE/EventBus already satisfies one-way live stream needs and transport change adds migration risk.
- **Separate backend endpoints for friendly mode projection**: rejected; duplicates logic and violates “one shared data source” requirement.
- **New WorkerEvent kinds for orchestration**: rejected in favor of additive lifecycle payload fields to maximize replay/consumer compatibility [goal-derived].

## 11. Out of Scope
- Authentication/authorization, remote multi-user access, TLS termination.
- Major CLI workflow redesign beyond additive event emission.
- Replacing current run artifact format (`meta.json`, `events.jsonl`, logs, diff).
- New dashboard pages beyond existing four.
- Historical analytics warehouse/reporting beyond current run artifacts.
- Mobile-native app packaging.

## 12. Assumptions
- **A-01**: The dashboard must stay responsive for burst rates up to 50 events/sec per visible run; this is sufficient for public-release expectations in current AgentOrch usage.
- **A-02**: “Calm cadence” is satisfied by recomputing friendly diagram layout/animation at 2–4 Hz while ingesting events continuously.
- **A-03**: Dogfood environments provide an accessible `:0` display and browser session so OBSERVE/REAL computer-use checks can run against rendered UI.
- **A-04**: For automated dogfood runs requiring REAL actions, the existing computer-use test harness can provide deterministic approval behavior (e.g., fake/auto grant path) without manual operator intervention.

## 13. Open Questions
- None currently block buildability under the stated constraints; unresolved cosmetic preferences (exact iconography, color accents, animation easing style) are intentionally left to implementation-time defaults [industry-default].

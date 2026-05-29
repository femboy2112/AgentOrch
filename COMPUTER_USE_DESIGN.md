## 1. Overview
This design defines a new AgentOrch worker that performs GUI computer-use by reconstructing screen state from structured signals, not pixel vision: AT-SPI accessibility tree, window geometry, OCR, and browser DOM. It uses existing `claude`/`codex` CLI workers for reasoning and decision-making, and executes actions through a constrained action runtime.

The system solves three problems simultaneously:
- Reliable GUI perception without a vision model or API keys.
- Safe execution with strict isolation and host-protection boundaries.
- Full auditability for every action and rationale, integrated into AgentOrch run logs.

Core value delivery:
- Better deterministic behavior from structured UI state [goal-derived].
- Safe testing-first operation via default isolated display mode [goal-derived].
- Operational trust from hard capability boundaries, process ownership rules, spawn-policy validation, isolated-display environment binding, and action-level audit trails [goal-derived].

## 2. Goals & Non-Goals
Goals:
- Provide a buildable AgentOrch worker that can perceive GUI state programmatically and perform actions in a private display.
- Support two perception modes: `ISOLATED` (private Xvfb, full perceive+act) and `OBSERVE` (real `:0` read-only perception, no real-session input injection).
- Enforce structural incapability of real-session control in `OBSERVE`.
- Ensure host/process safety with bounded steps/actions, timeouts, resource/process caps, and strict spawn-policy validation.
- Integrate as a standard AgentOrch worker with `is_available()` capability probing, graceful AT-SPI degradation, and event streaming.
- Guarantee auditable actions and explicit gating for destructive/irreversible operations.
- Ensure the reasoner can consume simultaneous multi-scope state when `OBSERVE` mode requires real-screen monitoring plus isolated-screen control [goal-derived].
- Ensure deterministic reasoner routing based on explicit task priority and documented defaults.

Non-goals:
- Pixel-vision model inference.
- Use of Anthropic SDK, OpenAI SDK, or API-key based inference.
- Controlling real `:0` input devices.
- Global process orchestration or killing non-worker processes.
- Wayland-native compositor automation beyond X11/Xvfb/Xwayland surfaces [assumption].

## 3. Requirements
Functional requirements (testable):
- `FR-01`: WHEN a run starts without explicit mode THEN the worker SHALL run in `ISOLATED` mode.
- `FR-02`: WHEN in `ISOLATED` mode THEN the worker SHALL perceive and act only within its private Xvfb display.
- `FR-03`: WHEN in `OBSERVE` mode THEN the worker SHALL be able to read real `:0` screen state (screenshot, AT-SPI if available, geometry, OCR) and SHALL reject all action attempts targeting real `:0`.
- `FR-04`: IF an action request lacks explicit `display_scope` THEN the action SHALL be hard-failed; IF `display_scope != "isolated"` at execution boundary THEN the Action Executor SHALL hard-reject.
- `FR-05`: WHEN PyGObject/AT-SPI is unavailable THEN `is_available()` SHALL report degraded capability and runtime SHALL continue with OCR+geometry perception.
- `FR-06`: WHEN browser context is detected THEN the worker SHALL attempt DOM snapshot acquisition via Playwright/CDP and merge it into perception state.
- `FR-07`: WHEN a model proposes a spatial action with an element handle THEN `SafetyKernel` target resolution SHALL convert that handle into concrete coordinates before execution.
- `FR-08`: WHEN any action executes THEN the system SHALL append an audit event including element handle (if used), resolved coordinates (if spatial), rationale, outcome, timestamps.
- `FR-09`: IF an action is classified destructive or irreversible THEN the system SHALL require explicit confirmation after a dry-run preview event before execution.
- `FR-10`: WHEN run-level action budget or step budget is exceeded THEN the system SHALL stop action execution and terminate owned subprocesses.
- `FR-11`: WHEN any subprocess exceeds timeout or resource cap THEN the system SHALL terminate only owned process tree members and continue/abort per policy.
- `FR-12`: IF a process is not in the owned-process registry THEN the worker SHALL NOT signal, kill, ptrace, or otherwise interfere with it.
- `FR-13`: WHEN integrated in AgentOrch THEN action/perception lifecycle events SHALL stream into `runs/<id>/events.jsonl`.
- `FR-14`: WHEN `task_priority="high"` THEN `claude` CLI SHALL be used as lead engine; IF unavailable/failing THEN `codex` CLI SHALL be used as primary fallback.
- `FR-15`: WHEN availability probing runs THEN it SHALL return a capability matrix (`atspi`, `ocr`, `geometry`, `dom`, `action_exec`) and a readiness state.
- `FR-16`: WHEN CLI reasoning output cannot be parsed/validated into `ActionIntent` THEN the system SHALL perform bounded repair retries and fallback engine routing; IF still invalid THEN it SHALL abort the step with no action execution and emit auditable parse-error events.
- `FR-17`: WHEN `OBSERVE` mode is active THEN each reasoning turn SHALL include multi-scope snapshots (`observe_real` and `isolated`) when available, in one `ReasoningInput`.
- `FR-18`: WHEN an external confirmation is submitted via adapter API THEN the worker SHALL inject that token into the next gated decision path and SHALL allow execution only if token validation succeeds against the pending high-risk intent.
- `FR-19`: WHEN a high-risk intent is gated for confirmation THEN the run loop SHALL enter a suspended wait state (no reasoner busy-polling, no action execution) until valid confirmation arrives or confirmation timeout occurs.
- `FR-20`: WHEN `ReasonerBridge.decide()` exceeds `reasoning_timeout_ms` THEN the worker SHALL terminate the owned reasoning subprocess, emit timeout events, and perform bounded retry/fallback; IF all attempts fail THEN it SHALL abort the step with no action execution.
- `FR-21`: WHEN `task_priority` is absent or `normal` THEN engine routing SHALL default to `codex` lead with `claude` fallback; if lead/fallback are unavailable, the step SHALL fail closed with no action execution.
- `FR-22`: WHEN `ISOLATED` mode session is created THEN the worker SHALL spawn and register an owned Xvfb process via `ProcessSupervisor`; on session close/failure it SHALL deterministically tear down that owned Xvfb subtree.
- `FR-23`: WHEN `action.type="launch_app"` THEN `SafetyKernel.validate()` SHALL accept only an allowlisted executable identity and argument schema; shell command strings, metacharacters, and non-allowlisted executables SHALL be rejected with no spawn.
- `FR-24`: WHEN any app/browser process is spawned for action execution THEN `ProcessSupervisor` SHALL force isolated-display environment binding (including `DISPLAY=<isolated_display>`) and SHALL reject spawn requests that attempt to override it to real-session displays.
- `FR-25`: WHEN a reasoning CLI emits auth-expired/interactive-login indicators or blocks on interactive OAuth prompt THEN `ReasonerBridge` SHALL classify as `auth_required`, terminate the owned subprocess, emit auth-failure events, and apply bounded fallback routing; IF route exhausted THEN fail closed with no action execution.

Non-functional requirements:
- Performance: Perception snapshot assembly p95 under 1500 ms in `ISOLATED` mode for typical desktop scenes [assumption].
- Performance: Action dispatch-to-result p95 under 500 ms excluding target app latency [assumption].
- Security/Safety: No global X input grabs, no writes/signals to foreign processes, no real-session input injection in `OBSERVE`.
- Security/Safety: Resource governance via CPU, memory, process-count limits and per-action/per-reasoning timeouts.
- Security/Safety: Spawn path is non-shell, allowlist-validated, and isolated-display-bound to prevent arbitrary command execution and host destabilization [goal-derived].
- Reliability: Crash-safe event logging (append-only JSONL) and deterministic teardown of owned process trees.
- Reliability: Degraded mode operation when optional sensors are unavailable.

## 4. Architecture
High-level structure:
- `ComputerUseWorkerAdapter` (AgentOrch role adapter, including confirmation ingress API).
- `SessionController` (run lifecycle + mode policy + suspended confirmation wait state + isolated display bootstrap/teardown orchestration).
- `CapabilityBroker` (`is_available`, sensor/action capability matrix).
- `PerceptionPipeline` (`ATSPICollector`, `GeometryCollector`, `OCRCollector`, `BrowserDOMCollector`, `FusionEngine`).
- `ReasonerBridge` (`claude` lead for high priority, `codex` lead for normal priority, bounded reasoning timeout, OAuth/interactive-prompt failure detection).
- `SafetyKernel` (policy checks, destructive gate, budget/time/resource enforcement, process-ownership enforcement, target resolution, display-scope intent validation, launch-app allowlist validation).
- `ActionExecutor` (`xdotool`-based input injection on isolated X display only; app/browser launches delegated through `ProcessSupervisor` using forced isolated display environment).
- `ProcessSupervisor` (owned subprocess tree lifecycle, killable root, registry authority, including Xvfb ownership for isolated sessions, and strict non-shell spawn contract).
- `AuditEventSink` (`runs/<id>/events.jsonl` stream).

Data flow:
1. Adapter receives AgentOrch work item, creates run session, mode, budgets, and `task_priority`.
2. In `ISOLATED`, `SessionController` requests `ProcessSupervisor` to spawn/register Xvfb and returns the allocated display handle; in `OBSERVE`, isolated display may still be provisioned for actuation.
3. Capability probe evaluates available sensors and action runtime.
4. Perception collectors gather signals from configured scope(s); fusion emits per-scope `PerceptionSnapshot` objects.
5. Snapshot set + objective + pending orchestrator messages go to `ReasonerBridge`; model emits `ActionIntent` plus rationale, under `reasoning_timeout_ms`.
6. `SafetyKernel` validates intended `display_scope`, ownership policy, risk classification, token validity when needed, target requirements by action type, and `launch_app` payload policy.
7. If action is high-risk without valid confirmation, dry-run event is emitted and loop transitions to suspended wait state.
8. Suspended wait exits only when a matching valid confirmation token is received or confirmation timeout elapses; no reasoner busy-loop is performed while waiting.
9. If intent is spatial and targets an element handle, `SafetyKernel` resolves it to `CoordinateTarget`.
10. `ActionExecutor` executes only allowed actions against isolated display; spatial actions require resolved coordinates, non-spatial actions (`launch_app`, `navigate`, `hotkey`, `wait`) do not.
11. For `launch_app`/browser spawns, `ProcessSupervisor` executes non-shell command arrays only, injects forced isolated display env (`DISPLAY=<isolated_display>`), strips/disallows unsafe env overrides, and registers owned process roots.
12. Result is audited and streamed; loop continues until done/budget/failure.
13. Session termination tears down owned subprocess trees, including owned Xvfb when present.

Key structural safety decision:
- In `OBSERVE`, perception and actuation are capability-split: observer side can read configured real-display channels; action side is bound to isolated-display X authority only and cannot authenticate to real `:0` [goal-derived].

## 5. Components & Interfaces
| Component | Responsibility | Public Interface / Contract | Interaction Pattern |
|---|---|---|---|
| `ComputerUseWorkerAdapter` | AgentOrch worker integration + external approvals | `start(run_request) -> RunHandle`; `stop(run_id) -> StopResult`; `submit_confirmation(run_id, token, intent_id?) -> Ack`; `is_available() -> CapabilityReport` | Synchronous control API + async event stream [industry-default] |
| `SessionController` | Mode selection, run lifecycle, policy binding, confirmation wait orchestration | `create_session(config: RunRequest) -> Session`; `close_session(session_id)`; `enqueue_orchestrator_message(run_id, msg)`; `await_confirmation(run_id, pending_intent_id, timeout_ms) -> ConfirmationOutcome` | Called once per run; owns session context |
| `CapabilityBroker` | Runtime capability detection and degrade state | `probe() -> CapabilityReport` with fields `{atspi: bool, ocr: bool, geometry: bool, dom: bool, action_exec: bool, degraded: bool}` | Called at run start and on sensor faults |
| `PerceptionPipeline` | Build unified textual scene model | `snapshot(scope: Scope) -> PerceptionSnapshot`; `snapshot_set(scopes: Scope[]) -> Record<Scope, PerceptionSnapshot>` where `Scope = "isolated" | "observe_real"` | Pull model each loop; emits provenance per node |
| `ReasonerBridge` | CLI model orchestration + priority-based routing | `decide(input: ReasoningInput, timeout_ms: number) -> ActionIntent`; `engine_status() -> EngineHealth`; parse contract `ACTION_INTENT_JSON_V1`; auth-state contract `AUTH_STATE_V1` (`ready`/`auth_required`) | High priority route: `claude -> codex`; normal/default route: `codex -> claude` [goal-derived] |
| `SafetyKernel` | Enforce hard guardrails + resolve targets | `validate(intent, session) -> ValidationResult`; `classify_risk(intent) -> RiskLevel`; `require_confirmation(intent) -> GateDecision`; `resolve_target(intent, snapshots) -> CoordinateTarget`; `validate_launch_app(payload, policy) -> AppLaunchValidation` | Mandatory pre-action gate |
| `ActionExecutor` | Execute allowed GUI actions on isolated display | `execute(action: ActionSpec) -> ActionResult` with preconditions: `action.display_scope == "isolated"`; for spatial action types (`click`, `double_click`, `type`, `scroll`, `drag`) `action.target.kind == "coordinate"` is required; for non-spatial types (`launch_app`, `navigate`, `hotkey`, `wait`) target is optional | Uses `xdotool` on isolated display; delegates process spawns to `ProcessSupervisor` |
| `ProcessSupervisor` | Owned subprocess tree management and limits | `spawn(spec: SpawnSpec) -> SpawnedProc`; `spawn_isolated_display(spec: IsolatedDisplaySpec) -> SpawnedProc`; `terminate_tree(root_id)`; `enforce_limits(session)`; `is_owned(pid) -> bool` | Periodic watchdog + on-demand kill |
| `AuditEventSink` | Append auditable events | `emit(event: WorkerEvent) -> void` | Append-only JSONL to run stream [industry-default] |

Action contract (core):
- Input `ActionSpec`:
  - `action_id: string`
  - `type: "click" | "double_click" | "type" | "hotkey" | "scroll" | "drag" | "wait" | "launch_app" | "navigate"`
  - `display_scope: "isolated"`
  - `target?: CoordinateTarget` (required only for spatial action types; optional for `launch_app`, `navigate`, `hotkey`, `wait`)
  - `source_handle_id?: string` (for audit traceability when target originated from an element handle)
  - `text?: string`
  - `hotkey?: string[]`
  - `scroll_delta?: {dx: number; dy: number}`
  - `drag_to?: CoordinateTarget`
  - `wait_ms?: number`
  - `url?: string`
  - `app?: string` (allowlisted executable identity only; not a shell command)
  - `app_args?: string[]` (optional; schema-validated per allowlisted app policy)
  - `rationale: string`
  - `risk_level: "low" | "medium" | "high" | "irreversible"`
  - `confirmation_token?: string`
- Output `ActionResult`:
  - `status: "ok" | "rejected" | "timeout" | "failed"`
  - `executed_at: timestamp`
  - `resolved_target?: {x:number,y:number}` (present for spatial actions)
  - `resolved_source_bbox?: {x:number,y:number,w:number,h:number}` (optional audit geometry when available)
  - `spawned_process_ids?: number[]`
  - `error_code?: string`
  - `artifacts?: string[]`

## 6. Data Models
Typed schemas (logical):

```ts
type RunMode = "ISOLATED" | "OBSERVE";
type Scope = "isolated" | "observe_real";
type TaskPriority = "normal" | "high";

interface RunRequest {
  run_id: string;
  objective: string;
  mode?: RunMode;
  task_priority?: TaskPriority; // default "normal"
  budgets?: Partial<WorkerSession["budgets"]>;
  observe_display?: string;     // default ":0" in OBSERVE if permitted
}
```

```ts
interface WorkerSession {
  run_id: string;
  mode: RunMode;
  task_priority: TaskPriority;
  created_at: string; // ISO-8601
  budgets: {
    max_steps: number;
    max_actions: number;
    action_timeout_ms: number;
    reasoning_timeout_ms: number;
    confirmation_wait_timeout_ms: number;
    max_cpu_percent: number;
    max_rss_mb: number;
    max_processes: number;
  };
  displays: {
    isolated_display: string;   // e.g. ":99"
    isolated_xvfb_root_pid?: number; // owned process root when worker-spawned
    observe_display?: string;   // ":0" only in OBSERVE
  };
  capabilities: CapabilityReport;
}
```

```ts
interface CapabilityReport {
  atspi: boolean;
  ocr: boolean;
  geometry: boolean;
  dom: boolean;
  action_exec: boolean;
  degraded: boolean;
  readiness: "ready" | "degraded" | "unavailable";
  notes?: string[];
}
```

```ts
interface ElementHandle {
  handle_id: string;            // stable within snapshot epoch
  source: "ATSPI" | "DOM" | "OCR" | "GEOMETRY";
  window_id?: string;
  app_pid?: number;
  role?: string;
  name?: string;
  bbox: { x: number; y: number; w: number; h: number };
  confidence: number;           // 0..1
  provenance: string[];         // source collector ids
}
```

```ts
interface CoordinateTarget {
  kind: "coordinate";
  x: number;
  y: number;
  coordinate_space: "display" | "window";
  window_id?: string;           // required when coordinate_space="window"
  tolerance_px?: number;        // default 3
}
```

```ts
interface PerceptionSnapshot {
  snapshot_id: string;
  run_id: string;
  mode: RunMode;
  scope: Scope;
  captured_at: string;
  windows: Array<{
    window_id: string;
    title: string;
    pid?: number;
    bbox: { x: number; y: number; w: number; h: number };
    z_index?: number;
  }>;
  elements: ElementHandle[];
  raw_text_blocks: Array<{ text: string; bbox: {x:number;y:number;w:number;h:number}; source: "OCR" | "ATSPI" | "DOM" }>;
}
```

```ts
interface SnapshotSummary {
  snapshot_id: string;
  captured_at: string;
  scope: Scope;
  windows: PerceptionSnapshot["windows"];
  elements: Array<{
    handle_id: string;
    source: ElementHandle["source"];
    role?: string;
    name?: string;
    bbox: ElementHandle["bbox"];
    confidence: number;
  }>;
  raw_text_blocks: PerceptionSnapshot["raw_text_blocks"];
}
```

```ts
type OrchestratorMessage =
  | {
      kind: "confirmation_token";
      token: string;
      intent_id?: string;
      issued_at: string;
    }
  | {
      kind: "operator_note";
      text: string;
      issued_at: string;
    };
```

```ts
interface ReasoningInput {
  run_id: string;
  session_mode: RunMode;
  task_priority: TaskPriority;
  objective: string;
  constraints: {
    must_use_display_scope: "isolated";
    max_actions_remaining: number;
    max_steps_remaining: number;
    disallowed_ops: string[];
  };
  snapshots: Partial<Record<Scope, SnapshotSummary>>;
  orchestrator_messages?: OrchestratorMessage[];
  prior_step_context?: {
    last_action?: string;
    last_result?: string;
    recent_errors?: string[];
  };
  output_contract: "ACTION_INTENT_JSON_V1";
}
```

```ts
type IntentTarget =
  | { kind: "element"; handle_id: string }
  | CoordinateTarget;

interface ActionIntent {
  intent_id: string;
  snapshot_id: string;
  action: {
    type: "click" | "double_click" | "type" | "hotkey" | "scroll" | "drag" | "wait" | "launch_app" | "navigate";
    display_scope: Scope;       // model-declared intent scope; must validate to "isolated" for execution
    target?: IntentTarget;      // required for spatial actions; optional for launch_app/navigate/hotkey/wait
    text?: string;              // required for type
    hotkey?: string[];          // e.g. ["CTRL","L"]
    scroll_delta?: { dx: number; dy: number };
    drag_to?: CoordinateTarget; // required when type=drag
    wait_ms?: number;           // required when type=wait
    url?: string;               // for navigate
    app?: string;               // allowlisted executable identity
    app_args?: string[];        // optional per app policy
  };
  rationale: string;
  risk_level: "low" | "medium" | "high" | "irreversible";
  requires_confirmation: boolean;
  confirmation_token?: string;
  confidence: number;           // 0..1
}
```

```ts
interface ActionSpec {
  action_id: string;
  type: ActionIntent["action"]["type"];
  display_scope: "isolated";
  target?: CoordinateTarget;
  source_handle_id?: string;
  text?: string;
  hotkey?: string[];
  scroll_delta?: { dx: number; dy: number };
  drag_to?: CoordinateTarget;
  wait_ms?: number;
  url?: string;
  app?: string;
  app_args?: string[];
  rationale: string;
  risk_level: ActionIntent["risk_level"];
  confirmation_token?: string;
}
```

```ts
interface SpawnSpec {
  argv: string[];                     // required; argv[0] must resolve to allowlisted executable for action-driven launches
  env?: Record<string, string>;       // optional caller env additions, merged under supervisor policy
  cwd?: string;                       // optional working directory (must be policy-allowed path)
  timeout_ms?: number;                // optional process-specific timeout
  display_scope: "isolated";          // required for action-driven spawn
  require_owned_group?: boolean;      // default true
  no_shell: true;                     // required invariant
  source_action_id?: string;          // audit linkage
}
```

```ts
interface IsolatedDisplaySpec {
  display: string;                    // e.g. ":99"
  screen: string;                     // e.g. "1920x1080x24"
  xvfb_binary?: string;               // default "Xvfb"
  timeout_ms?: number;                // startup readiness timeout
}
```

```ts
interface AppLaunchPolicy {
  allowed_apps: Record<string, {
    exec_path: string;
    allowed_args_pattern?: string;    // regex string applied per arg, optional
    default_args?: string[];
    blocked_args?: string[];
  }>;
  blocked_env_keys: string[];         // e.g. ["DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "LD_PRELOAD"]
}
```

```ts
interface ValidationResult {
  valid: boolean;
  violations?: Array<{
    code:
      | "display_scope_invalid"
      | "target_missing"
      | "target_unresolvable"
      | "foreign_process"
      | "budget_exceeded"
      | "resource_limit"
      | "schema_invalid"
      | "confirmation_required"
      | "confirmation_invalid"
      | "launch_app_not_allowlisted"
      | "launch_app_args_invalid"
      | "spawn_env_override_forbidden";
    message: string;
    field?: string;
  }>;
  normalized_action?: ActionSpec;
}
```

```ts
interface GateDecision {
  gate: "allow" | "require_confirmation" | "deny";
  reason: string;
  pending_intent_id?: string;
  required_token_scope?: {
    run_id: string;
    intent_id: string;
    expires_at: string;
  };
}
```

```ts
interface EngineHealth {
  claude: "ready" | "degraded" | "unavailable";
  codex: "ready" | "degraded" | "unavailable";
  routing_default: "codex_then_claude";
  routing_high_priority: "claude_then_codex";
  last_error?: string;
}
```

```ts
interface ConfirmationOutcome {
  outcome: "accepted" | "rejected" | "timeout" | "cancelled";
  token?: string;
  intent_id?: string;
  received_at?: string;
  reason?: string;
}
```

```ts
interface ActionResult {
  status: "ok" | "rejected" | "timeout" | "failed";
  executed_at: string;
  resolved_target?: { x: number; y: number };
  resolved_source_bbox?: { x: number; y: number; w: number; h: number };
  spawned_process_ids?: number[];
  error_code?: string;
  artifacts?: string[];
}
```

```ts
interface WorkerEvent {
  ts: string;
  run_id: string;
  event_type:
    | "capability.probe"
    | "perception.snapshot"
    | "reasoner.intent"
    | "reasoner.parse_error"
    | "reasoner.timeout"
    | "reasoner.auth_required"
    | "confirmation.received"
    | "confirmation.wait_started"
    | "confirmation.wait_timeout"
    | "action.dry_run"
    | "action.executed"
    | "action.rejected"
    | "safety.violation"
    | "resource.limit"
    | "session.terminated";
  payload: Record<string, unknown>;
}
```

Reasoner I/O formatting contract:
- The worker serializes `ReasoningInput` as deterministic JSON (stable key ordering, UTF-8, no comments) embedded in a fixed prompt envelope.
- Header contains task and safety constraints.
- Body contains fenced JSON with `ReasoningInput`.
- Footer contains strict output instructions requiring `ACTION_INTENT_JSON_V1`.
- The CLI response parser extracts the first fenced `json` block that contains top-level key `action_intent`.
- Required output payload shape:

```json
{
  "action_intent": {
    "intent_id": "string",
    "snapshot_id": "string",
    "action": {
      "type": "click",
      "display_scope": "isolated",
      "target": { "kind": "element", "handle_id": "h1" }
    },
    "rationale": "string",
    "risk_level": "low",
    "requires_confirmation": false,
    "confidence": 0.82
  }
}
```

- Parsed JSON is schema-validated.
- Any extra free text outside the JSON block is ignored for execution and kept only in audit artifacts.
- Auth/error detector scans raw CLI output and stderr for interactive login markers (for example OAuth expiry prompts, browser-login prompts, TTY-required auth text) and maps to `reasoner.auth_required` before parse phase.

Persistence/lifecycle:
- Session state and in-flight counters are in-memory with periodic checkpoint to run-local state file [industry-default].
- Events are persisted as append-only JSONL in `runs/<id>/events.jsonl`.
- Snapshots can be stored as compact JSON artifacts per run for replay/debug [industry-default].

## 7. Error Handling & Failure Modes
| Failure mode | Detection | System response | Degradation behavior |
|---|---|---|---|
| AT-SPI unavailable/import error | Capability probe/import failure | Mark `atspi=false`, emit `capability.probe` degraded | Continue with OCR+geometry (`FR-05`) |
| Playwright/CDP unavailable | Collector init failure | Mark `dom=false`, emit warning event | Continue without DOM signal |
| OCR binary missing | Probe failure | Mark `ocr=false`; if no other text channel, fail run as `unavailable` | If AT-SPI/DOM present, continue |
| Intended or requested action scope is real display | `ActionIntent.action.display_scope != "isolated"` or any real-scope target mapping | Hard reject, emit `safety.violation` | Continue run or abort per policy |
| Spatial action missing target | Action type in spatial set and target absent/unresolvable | Reject action, emit `action.rejected` | Continue with refreshed perception turn |
| Non-spatial action missing required payload | `wait` without `wait_ms`, `navigate` without `url`, `launch_app` without `app`, `scroll` without `scroll_delta`, `drag` without `drag_to`, `hotkey` without `hotkey` | Reject action, emit `action.rejected` with schema/detail code | Continue with refreshed perception turn |
| `launch_app` payload policy violation | `app` not in allowlist, arg schema violation, shell/metacharacter attempt, executable resolution mismatch | Reject action, emit `safety.violation` / `action.rejected` | Continue with refreshed perception turn |
| Spawn env attempts real-display override | Spawn request contains forbidden env keys/values (`DISPLAY=:0`, conflicting X authority, etc.) | Reject spawn at supervisor boundary, emit `safety.violation` | No process launch; continue/abort per policy |
| CLI output parse/validation failure | Missing/invalid `ACTION_INTENT_JSON_V1`, schema mismatch, unresolved enum/target fields | Emit `reasoner.parse_error`; retry same engine with parse-repair prompt (bounded, e.g., 1 retry); on repeated failure fallback per priority route; if fallback also fails, abort current step with `no_action` and continue/terminate per policy (`FR-16`) | Never execute action from unparseable output; maintain safe no-op progression |
| CLI reasoning timeout/hang | `decide()` exceeds `reasoning_timeout_ms` | Terminate owned reasoning subprocess; emit `reasoner.timeout`; bounded retry/fallback; if exhausted, abort current step with `no_action` | No unsafe retries; resources released |
| CLI OAuth expiration or interactive auth prompt | Output/stderr marker match for auth-required state, or interactive prompt detector before timeout | Terminate owned reasoning subprocess; emit `reasoner.auth_required`; route to fallback engine if available; if route exhausted, abort run/step fail-closed with no action | No interactive login attempt inside worker loop |
| High-risk action pending confirmation | Risk gate triggered and no valid token available | Emit `action.dry_run`; enter suspended wait state and emit `confirmation.wait_started` | No reasoner busy-polling, no action execution while waiting |
| Confirmation wait timeout | No valid matching token before `confirmation_wait_timeout_ms` | Emit `confirmation.wait_timeout`; reject/skip pending intent per policy | Continue with next reasoning turn or terminate per policy |
| Action timeout/hung app | Action timer exceeds `action_timeout_ms` | Kill owned child action process via `ProcessSupervisor`; mark action timeout | Continue next step if safe |
| Resource cap exceeded | `psutil` watchdog threshold breach | Terminate owned process subtree, emit `resource.limit` | Abort run safely |
| Foreign process targeted | PID not in owned registry | Reject request, emit violation | Continue run |
| CLI reasoner unavailable | CLI health check fail | Route using configured priority fallback; if route exhausted, fail run deterministically | No unsafe retries |
| Event sink write error | I/O error on JSONL append | Retry bounded times; if persistent, fail closed (stop actions) [goal-derived] | Preserve safety over progress |
| Confirmation missing or invalid for high-risk action | Missing token, malformed token, intent-token mismatch, or expired token | Emit `action.dry_run` and/or `action.rejected`; keep action blocked | Await valid confirmation or timeout path |
| Handle resolution failure | Target handle missing/stale in current snapshots | Reject action with explicit resolution error; emit safety event | Continue with refreshed perception turn |
| Xvfb startup failure in `ISOLATED` | Spawn failure or readiness probe timeout | Emit session initialization error and terminate run safely | No fallback to real display actuation |

## 8. Testing Strategy
Validation strategy is requirement-driven, not implementation-order-driven.

Unit testing:
- Policy and guardrail units:
  - `FR-03/04/12`: reject real-display actions and foreign-process operations.
  - `FR-07`: handle-to-coordinate resolver behavior, including stale-handle rejection.
  - `FR-09/18/19`: risk classifier + confirmation gate + token validation + suspended wait behavior.
  - `FR-10/11`: budget/timeout/resource threshold transitions.
  - `FR-23/24`: `launch_app` allowlist validation, argument schema validation, forbidden shell/metacharacter rejection, forced isolated-display env enforcement.
- Data model validators:
  - schema validation for `PerceptionSnapshot`, `ReasoningInput`, `ActionIntent`, `CoordinateTarget`, `ActionSpec`, `SpawnSpec`, `IsolatedDisplaySpec`, `ValidationResult`, `GateDecision`, `EngineHealth`, `ConfirmationOutcome`, `WorkerEvent`.
- Reasoner routing and parsing logic:
  - `FR-14` high-priority engine selection and fallback correctness.
  - `FR-21` normal/default priority routing correctness.
  - `FR-16` parse failure retry/fallback/abort behavior.
  - `FR-20` reasoning-timeout kill/retry/fallback behavior.
  - `FR-25` auth-required detection and fallback/abort behavior.
- Adapter ingress:
  - `submit_confirmation(run_id, token, intent_id?)` acceptance, queuing, replay protection, and audit event emission.
- Action type contracts:
  - target optionality for `launch_app`, `navigate`, `hotkey`, `wait`.
  - target requirement for spatial actions only.
  - payload requirement validation for `hotkey`, `scroll_delta`, `drag_to`, `wait_ms`, `url`, `app`.
  - `launch_app` app identity and optional `app_args` policy validation.

Integration testing:
- Mode behavior:
  - `FR-01/02/03` across `ISOLATED` and `OBSERVE`.
- Capability degradation:
  - simulate missing PyGObject and verify `FR-05`.
- Sensor fusion:
  - mixed ATSPI/DOM/OCR/geometry snapshots produce consistent element handles.
  - `FR-17` multi-scope snapshot assembly in `OBSERVE`.
- Action execution:
  - verify `xdotool` injection only targets isolated display.
  - verify hard rejection paths for real-display targeting, including intent scope mismatch.
  - verify non-spatial actions execute without coordinate targets and with required non-spatial payload fields.
- Process ownership:
  - `launch_app`/`navigate` spawn paths must register through `ProcessSupervisor.spawn()`.
  - isolated display bootstrap must register Xvfb through `ProcessSupervisor.spawn_isolated_display()`.
  - create foreign dummy process and assert no signal/interference (`FR-12`).
- Spawn isolation:
  - verify `DISPLAY=<isolated_display>` is enforced for every spawned GUI app.
  - verify forbidden env overrides are blocked and audited.
- CLI contract conformance:
  - malformed/multi-block CLI outputs, schema mismatch outputs, valid contract outputs, reasoning timeout cases, and OAuth interactive-auth failure signatures.

End-to-end testing:
- Full AgentOrch dispatch with worker adapter:
  - `is_available()` report, reasoning loop, action loop, event stream to `runs/<id>/events.jsonl` (`FR-13`).
- Audit completeness:
  - each action contains handle reference (when applicable), coordinates for spatial actions, rationale, and outcome (`FR-08`).
- Destructive-action gate:
  - dry-run emitted and execution blocked until explicit confirmation (`FR-09`, `FR-18`, `FR-19`).
- Parse-error safety:
  - intentionally corrupted CLI outputs result in no action execution and traceable parse-error events (`FR-16`).
- Reasoner-timeout safety:
  - hung CLI process is terminated and step proceeds via retry/fallback policy (`FR-20`).
- Reasoner-auth safety:
  - expired OAuth / interactive auth output causes `reasoner.auth_required`, subprocess termination, and fallback/abort per policy (`FR-25`).
- Priority routing behavior:
  - `task_priority=high` route uses `claude -> codex`; `task_priority=normal/default` route uses `codex -> claude`.
- Host-protection safety:
  - malicious `launch_app` intents (for example command-like payloads) are rejected with no subprocess spawn and no host-side side effects (`FR-23`).

Reliability and chaos testing:
- Inject collector crashes/timeouts; assert controlled degradation and clean teardown.
- Inject reasoner hangs; assert timeout enforcement and no leaked subprocesses.
- Inject reasoner auth-expiry prompt patterns; assert `auth_required` path and deterministic fallback.
- Inject Xvfb startup and runtime faults; assert fail-closed startup and deterministic owned-tree teardown.
- Stress with configured concurrency profile to validate isolation and cap enforcement [assumption].
- Replay events for determinism checks (same inputs -> same policy decisions).

Acceptance mapping:
- A run is acceptable only if all mandatory FRs pass and no guardrail violation results in unsafe side effects.
- Any failure of `FR-03`, `FR-04`, `FR-09`, `FR-12`, `FR-23`, or `FR-24` is release-blocking by policy.

## 9. Constraints & Guardrails
`C1` Two modes:
- `ISOLATED`: private Xvfb perceive+act.
- `OBSERVE`: real `:0` read-only perception; structurally no real input injection.
- Guardrail: action interfaces only accept execution on `display_scope="isolated"`; intent scope is validated and any non-isolated intent is rejected pre-execution.

`C2` Non-interference with foreign processes:
- Must never read/signal/kill non-owned processes.
- No global X input grabs.
- Owned-process registry is authoritative; all process actions are registry-gated.

`C3` Host stability:
- Hard caps on steps/actions/time/resources/process-count.
- Self-contained subprocess tree with deterministic group kill.
- Xvfb for isolated sessions is worker-owned and must be in that killable owned tree.
- Fail-closed on watchdog violations.

`C4` Perception modality limits:
- Allowed: AT-SPI (PyGObject), geometry tooling, OCR (tesseract), DOM (Playwright/CDP).
- Forbidden: pixel-vision models, Anthropic SDK, API keys.

`C5` Reasoning + AgentOrch integration:
- Reason through `claude`/`codex` CLI OAuth sessions.
- Priority-aware routing: high priority `claude -> codex`; normal/default `codex -> claude`.
- Standard worker adapter/role, `is_available()` with graceful degradation.
- Stream events to `runs/<id>/events.jsonl`.
- Accept orchestrator confirmation ingress and propagate it into gating logic for high-risk intents.
- Detect auth-required/interactive OAuth states and fail over or fail closed; no interactive login in-loop.

`C6` Audit + destructive gating:
- Every action must log target handle reference when applicable, resolved coordinates when applicable, rationale, outcome.
- Irreversible/destructive actions require explicit confirmation after dry-run preview.
- While waiting for confirmation, loop must suspend/yield rather than re-query reasoner.

`C7` Timeout completeness:
- Reasoning and action phases must both be time-bounded (`reasoning_timeout_ms`, `action_timeout_ms`).
- Timeout handling must terminate only owned subprocesses and preserve auditability.

`C8` Spawn hardening:
- `launch_app` must be allowlist-validated; arbitrary command strings are forbidden.
- Process spawning must be non-shell (`argv` only) and policy-validated.
- Spawned action apps must be forced onto isolated display (`DISPLAY=<isolated_display>`), with forbidden env overrides blocked.
- Spawn contract must not permit host-level destructive command execution via action payloads.

Hard must-never rules:
- Never inject keyboard/mouse into real `:0`.
- Never signal or kill foreign processes.
- Never continue actions after event-sink integrity failure.
- Never execute high-risk action without valid confirmation token.
- Never execute an action from unparseable or schema-invalid reasoner output.
- Never execute `launch_app` from non-allowlisted executable identity or shell command text.
- Never allow spawned action process env to target real-session displays.

## 10. Alternatives Considered
Pixel-vision GUI agent:
- Rejected because it violates modality constraints and increases nondeterminism/cost [goal-derived].

Single-process combined observer+actor:
- Rejected because it weakens structural separation between real-display observation and isolated action capabilities [goal-derived].

Direct API SDK-based reasoning:
- Rejected because requirement mandates CLI OAuth engines and no API keys [goal-derived].

Always-on DOM-only browser automation:
- Rejected because scope includes general desktop GUI, not browser-only surfaces [goal-derived].

No OCR fallback:
- Rejected because AT-SPI may be absent and graceful degradation is required [goal-derived].

Externally provisioned shared Xvfb:
- Rejected as default because it weakens per-run ownership/teardown guarantees and complicates kill-scope safety [goal-derived].

Unrestricted `launch_app` command passthrough:
- Rejected because it violates process non-interference and host-stability constraints by enabling arbitrary command execution [goal-derived].

## 11. Out of Scope
- Wayland-native privileged automation APIs.
- Human UI tooling for manual approvals (only event/confirmation contract is defined).
- Cross-machine distributed execution.
- Automated policy-learning for destructive-action classification.
- Long-term analytics warehouse for GUI telemetry.

## 12. Assumptions
- The deployment host is Linux with X11/Xwayland support, and `Xvfb` is installable/available.
- AgentOrch’s worker adapter can register this worker and append custom events to `runs/<id>/events.jsonl`.
- `claude` and `codex` CLIs are available on PATH and support non-interactive prompt/response usage suitable for worker orchestration.
- Default guardrail limits are configurable and initially set to: `max_steps=200`, `max_actions=200`, `action_timeout_ms=10000`, `reasoning_timeout_ms=45000`, `confirmation_wait_timeout_ms=300000`, `max_cpu_percent=200`, `max_rss_mb=2048`, `max_processes=64`.
- Performance targets (`snapshot p95 <= 1500 ms`, `action dispatch p95 <= 500 ms` excluding app latency) are acceptable baseline SLOs.
- OCR default language is English (`eng`) unless run config overrides it.
- `OBSERVE` mode use is authorized for reading visible real-session content, including terminals.
- Current scope targets X11 surfaces; native Wayland-only applications outside Xwayland are not required to be actionable.
- No long-term cross-run UI data store is required beyond run-local artifacts and retention policy controls [assumption].
- If multi-run concurrency is enabled in deployment, each run is provisioned with independent isolated display/session/process state and per-run limits; this is deployment-configured rather than a mandatory product guarantee [assumption].
- DOM perception in `OBSERVE` mode requires the target real-session browser to already expose a reachable CDP endpoint (for example, `--remote-debugging-port=9222`) that policy permits the collector to connect to [assumption].
- Deployment provides an operator-maintained `AppLaunchPolicy` allowlist for `launch_app`; absent policy, `launch_app` capability is treated as unavailable [assumption].
- CLI auth-state detection relies on stable textual markers emitted by `claude`/`codex` CLIs for expired OAuth or interactive login requirements [assumption].

## 13. Open Questions
- In `OBSERVE` mode, should terminal/window content be redacted before being sent to reasoning CLIs, or is full-text forwarding explicitly approved?
- What is the authoritative policy source for “destructive/irreversible” classification: static rule set, repo-provided policy file, or operator-provided runtime policy?
- Should confirmation for high-risk actions be human-only, or may orchestrator policy auto-confirm based on run flags?
- When multi-run concurrency is enabled, should isolation be enforced as one OS process per run or as multi-session isolation within one host process boundary?


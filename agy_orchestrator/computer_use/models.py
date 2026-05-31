"""Data models for computer-use worker.

Exact 1:1 translation of every TypeScript interface in the authoritative
COMPUTER_USE_DESIGN.md §6 Data Models (and supporting types from the spec).

- Enums for all closed string literal sets (fidelity tested).
- dataclasses for every interface/record.
- TypedDict for variant/union shapes and inline complex objects.
- __post_init__ validators on key contracts (display_scope, no_shell, etc.).
- to_dict/from_dict for clean roundtrip serialization (primitives only).

Pure stdlib, no Pydantic, no GUI/X11 code. Matches style of core/agent.py
(typing imports, dataclasses like routing/policy.py and execution/verifier.py).

Release-blocking FRs and hardening #4 redaction tests depend on these shapes.
Phase 1.x (design addendum): DOMTargetDict(TypedDict) + extended IntentTarget union (dom kind) + browser_engine/browser_display on RunRequest/WorkerSession + BROWSER_NOT_OPEN (additive only; zero behavior change; INV F / B4 preserved).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union

# =============================================================================
# Enums (string literal fidelity — every value matches the spec exactly)
# =============================================================================

class RunMode(str, Enum):
    ISOLATED = "ISOLATED"
    OBSERVE = "OBSERVE"
    REAL = "REAL"


class Scope(str, Enum):
    ISOLATED = "isolated"
    OBSERVE_REAL = "observe_real"
    REAL_ACT = "real_act"


class RealGuiPolicy(str, Enum):
    FULL = "full"
    CHILDREN = "children"


class AskMode(str, Enum):
    ON = "on"
    OFF = "off"


class GrantScope(str, Enum):
    ACTION = "ACTION"
    PROCESS_RUN = "PROCESS_RUN"
    PROCESS_TTL = "PROCESS_TTL"
    DENY = "DENY"


class TaskPriority(str, Enum):
    NORMAL = "normal"
    HIGH = "high"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IRREVERSIBLE = "irreversible"


class ActionType(str, Enum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    TYPE = "type"
    HOTKEY = "hotkey"
    SCROLL = "scroll"
    DRAG = "drag"
    WAIT = "wait"
    LAUNCH_APP = "launch_app"
    NAVIGATE = "navigate"


class ElementSource(str, Enum):
    ATSPI = "ATSPI"
    DOM = "DOM"
    OCR = "OCR"
    GEOMETRY = "GEOMETRY"


class TextBlockSource(str, Enum):
    OCR = "OCR"
    ATSPI = "ATSPI"
    DOM = "DOM"


class CoordinateSpace(str, Enum):
    DISPLAY = "display"
    WINDOW = "window"


class Readiness(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ConfirmationOutcomeType(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ActionStatus(str, Enum):
    OK = "ok"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    FAILED = "failed"


class GateType(str, Enum):
    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    DENY = "deny"


class EngineStatus(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class WorkerEventType(str, Enum):
    CAPABILITY_PROBE = "capability.probe"
    PERCEPTION_SNAPSHOT = "perception.snapshot"
    REASONER_INTENT = "reasoner.intent"
    REASONER_PARSE_ERROR = "reasoner.parse_error"
    REASONER_TIMEOUT = "reasoner.timeout"
    REASONER_AUTH_REQUIRED = "reasoner.auth_required"
    CONFIRMATION_RECEIVED = "confirmation.received"
    CONFIRMATION_WAIT_STARTED = "confirmation.wait_started"
    CONFIRMATION_WAIT_TIMEOUT = "confirmation.wait_timeout"
    ACTION_DRY_RUN = "action.dry_run"
    ACTION_EXECUTED = "action.executed"
    ACTION_REJECTED = "action.rejected"
    SAFETY_VIOLATION = "safety.violation"
    RESOURCE_LIMIT = "resource.limit"
    SESSION_TERMINATED = "session.terminated"
    BASELINE_CAPTURED = "baseline.captured"
    PERMISSION_PROMPT_SHOWN = "permission.prompt_shown"
    PERMISSION_GRANTED = "permission.granted"
    PERMISSION_DENIED = "permission.denied"
    FOREIGN_INTERACTION_BLOCKED = "foreign_interaction_blocked"
    OPERATOR_NOTE_RECEIVED = "operator_note.received"


class ViolationCode(str, Enum):
    DISPLAY_SCOPE_INVALID = "display_scope_invalid"
    TARGET_MISSING = "target_missing"
    TARGET_UNRESOLVABLE = "target_unresolvable"
    FOREIGN_PROCESS = "foreign_process"
    BUDGET_EXCEEDED = "budget_exceeded"
    RESOURCE_LIMIT = "resource_limit"
    SCHEMA_INVALID = "schema_invalid"
    CONFIRMATION_REQUIRED = "confirmation_required"
    CONFIRMATION_INVALID = "confirmation_invalid"
    LAUNCH_APP_NOT_ALLOWLISTED = "launch_app_not_allowlisted"
    LAUNCH_APP_ARGS_INVALID = "launch_app_args_invalid"
    SPAWN_ENV_OVERRIDE_FORBIDDEN = "spawn_env_override_forbidden"
    REAL_ACT_NOT_PERMITTED = "real_act_not_permitted"
    ASK_MODE_DISABLED = "ask_mode_disabled"
    GRANT_REQUIRED = "grant_required"
    GRANT_EXPIRED = "grant_expired"
    CLEARANCE_TOKEN_INVALID = "clearance_token_invalid"
    BROWSER_NOT_OPEN = "browser_not_open"


# =============================================================================
# TypedDicts for variant/union shapes and inline anonymous objects (exact spec)
# =============================================================================

class BBoxDict(TypedDict):
    x: int
    y: int
    w: int
    h: int


class WindowDict(TypedDict, total=False):
    window_id: str
    title: str
    pid: Optional[int]
    bbox: BBoxDict
    z_index: Optional[int]


class RawTextBlockDict(TypedDict):
    text: str
    bbox: BBoxDict
    source: str  # TextBlockSource value


class SimplifiedElementDict(TypedDict, total=False):
    handle_id: str
    source: str  # ElementSource value
    role: Optional[str]
    name: Optional[str]
    bbox: BBoxDict
    confidence: float


class ElementTargetDict(TypedDict):
    kind: str  # "element"
    handle_id: str


class CoordinateTargetDict(TypedDict, total=False):
    kind: str  # "coordinate"
    x: int
    y: int
    coordinate_space: str
    window_id: Optional[str]
    tolerance_px: Optional[int]


class DOMTargetDict(TypedDict, total=False):
    """Target for browser/DOM actuation (click/type on agent-owned Chromium via CDP).
    kind="dom"; index is 1-based (matches prototype and reasoner plans).
    A dom target with no open browser yields REJECTED browser_not_open (fail-closed).
    """
    kind: str  # "dom"
    selector: Optional[str]
    index: Optional[int]  # 1-based


# IntentTarget union (Step 1 extended): Element|Coordinate|DOMTargetDict (kind="dom", selector, 1-based index for browser/DOM click/type). Dict fallback for forward compat.
IntentTarget = Union[ElementTargetDict, CoordinateTargetDict, DOMTargetDict, Dict[str, Any]]


class ConfirmationTokenMessageDict(TypedDict, total=False):
    kind: str  # "confirmation_token"
    token: str
    intent_id: Optional[str]
    issued_at: str


class OperatorNoteMessageDict(TypedDict, total=False):
    kind: str  # "operator_note"
    text: str
    issued_at: str


OrchestratorMessage = Union[ConfirmationTokenMessageDict, OperatorNoteMessageDict, Dict[str, Any]]


class PriorStepContextDict(TypedDict, total=False):
    last_action: Optional[str]
    last_result: Optional[str]
    recent_errors: Optional[List[str]]


class ActionPayloadDict(TypedDict, total=False):
    """Inline 'action' object inside ActionIntent (and mirrored in ActionSpec).
    target may be ElementTargetDict | CoordinateTargetDict | DOMTargetDict (kind="dom" for browser).
    """
    type: str
    display_scope: str
    target: Optional[IntentTarget]
    text: Optional[str]
    hotkey: Optional[List[str]]
    scroll_delta: Optional[Dict[str, int]]
    drag_to: Optional[CoordinateTargetDict]
    wait_ms: Optional[int]
    url: Optional[str]
    app: Optional[str]
    app_args: Optional[List[str]]


class ViolationDict(TypedDict, total=False):
    code: str  # ViolationCode value
    message: str
    field: Optional[str]


class RequiredTokenScopeDict(TypedDict):
    run_id: str
    intent_id: str
    expires_at: str


class ResolvedTargetDict(TypedDict, total=False):
    x: int
    y: int


class ResolvedSourceBBoxDict(TypedDict, total=False):
    x: int
    y: int
    w: int
    h: int


# =============================================================================
# Dataclasses — every interface from the spec, exact field names + defaults
# =============================================================================

@dataclass
class RunRequest:
    run_id: str
    objective: str
    mode: Optional[str] = None  # RunMode value or None -> default ISOLATED
    task_priority: Optional[str] = None  # TaskPriority value
    budgets: Optional[Dict[str, Any]] = None  # Partial<WorkerSession["budgets"]>
    observe_display: Optional[str] = None  # ":0" default in OBSERVE
    real_gui_policy: Optional[str] = None  # RealGuiPolicy value (validated only when mode==REAL)
    ask_mode: Optional[str] = None  # AskMode value (validated only when mode==REAL)
    browser_engine: Optional[str] = None  # "bing" | "duckduckgo" | "google" (default bing for bot tolerance)
    browser_display: Optional[str] = None  # ":0" in REAL else isolated Xvfb; forwarded to BrowserController

    def __post_init__(self) -> None:
        if self.mode is not None and self.mode not in {m.value for m in RunMode}:
            raise ValueError(f"mode must be one of {[m.value for m in RunMode]}")
        if self.task_priority is not None and self.task_priority not in {p.value for p in TaskPriority}:
            raise ValueError(f"task_priority must be one of {[p.value for p in TaskPriority]}")
        # real_gui_policy / ask_mode forwarded; validated only when mode==REAL (per Step 1 spec)
        # browser_* allowed unconditionally (no validation here; used by adapter + BrowserController)
        if self.mode == RunMode.REAL.value:
            if self.real_gui_policy is not None and self.real_gui_policy not in {p.value for p in RealGuiPolicy}:
                raise ValueError(f"real_gui_policy must be one of {[p.value for p in RealGuiPolicy]} when mode=REAL")
            if self.ask_mode is not None and self.ask_mode not in {a.value for a in AskMode}:
                raise ValueError(f"ask_mode must be one of {[a.value for a in AskMode]} when mode=REAL")


@dataclass
class WorkerSession:
    run_id: str
    mode: str  # RunMode value
    task_priority: str  # TaskPriority value
    created_at: str  # ISO-8601
    budgets: Dict[str, int]  # full required shape
    displays: Dict[str, Any]  # {isolated_display, isolated_xvfb_root_pid?, observe_display?}
    capabilities: CapabilityReport  # forward ref resolved at runtime via string
    real_gui_policy: Optional[str] = None  # RealGuiPolicy value (only for REAL mode)
    ask_mode: Optional[str] = None  # AskMode value
    browser_engine: Optional[str] = None  # echoed from RunRequest for the run
    browser_display: Optional[str] = None  # echoed; ":0" or isolated Xvfb for agent-owned browser
    baseline_pids: Optional[List[int]] = None
    baseline_windows: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.mode not in {m.value for m in RunMode}:
            raise ValueError("invalid RunMode")
        if self.task_priority not in {p.value for p in TaskPriority}:
            raise ValueError("invalid TaskPriority")
        # budgets shape check (minimal)
        required_budget_keys = {
            "max_steps", "max_actions", "action_timeout_ms",
            "reasoning_timeout_ms", "confirmation_wait_timeout_ms",
            "max_cpu_percent", "max_rss_mb", "max_processes",
        }
        if not required_budget_keys.issubset(self.budgets.keys()):
            raise ValueError(f"budgets missing keys: {required_budget_keys - set(self.budgets.keys())}")


@dataclass
class CapabilityReport:
    atspi: bool
    ocr: bool
    geometry: bool
    dom: bool
    action_exec: bool
    degraded: bool
    readiness: str  # Readiness value
    notes: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.readiness not in {r.value for r in Readiness}:
            raise ValueError(f"readiness must be one of {[r.value for r in Readiness]}")


@dataclass
class ElementHandle:
    handle_id: str
    source: str  # ElementSource value
    window_id: Optional[str] = None
    app_pid: Optional[int] = None
    role: Optional[str] = None
    name: Optional[str] = None
    bbox: Dict[str, int] = field(default_factory=dict)  # {x,y,w,h}
    confidence: float = 0.0
    provenance: List[str] = field(default_factory=list)
    # DOM elements only: a stable, actuation-ready selector (+ 1-based index within
    # querySelectorAll(selector)) so the reasoner can emit a precise
    # target {kind:'dom', selector, index} click/type instead of guessing
    # coordinates from the bbox. None for non-DOM (geometry/AT-SPI/OCR) sources.
    dom_selector: Optional[str] = None
    dom_index: Optional[int] = None

    def __post_init__(self) -> None:
        if self.source not in {s.value for s in ElementSource}:
            raise ValueError("invalid ElementSource")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")


@dataclass
class CoordinateTarget:
    kind: str = "coordinate"
    x: int = 0
    y: int = 0
    coordinate_space: str = "display"  # CoordinateSpace value
    window_id: Optional[str] = None
    tolerance_px: Optional[int] = 3

    def __post_init__(self) -> None:
        if self.kind != "coordinate":
            raise ValueError('CoordinateTarget kind must be "coordinate"')
        if self.coordinate_space not in {c.value for c in CoordinateSpace}:
            raise ValueError("invalid CoordinateSpace")


@dataclass
class PerceptionSnapshot:
    snapshot_id: str
    run_id: str
    mode: str  # RunMode
    scope: str  # Scope
    captured_at: str
    windows: List[Dict[str, Any]] = field(default_factory=list)  # list of WindowDict
    elements: List[ElementHandle] = field(default_factory=list)
    raw_text_blocks: List[Dict[str, Any]] = field(default_factory=list)  # list of RawTextBlockDict

    def __post_init__(self) -> None:
        if self.mode not in {m.value for m in RunMode}:
            raise ValueError("invalid RunMode in snapshot")
        if self.scope not in {s.value for s in Scope}:
            raise ValueError("invalid Scope")


@dataclass
class SnapshotSummary:
    snapshot_id: str
    captured_at: str
    scope: str  # Scope
    windows: List[Dict[str, Any]] = field(default_factory=list)
    elements: List[Dict[str, Any]] = field(default_factory=list)  # SimplifiedElementDict
    raw_text_blocks: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.scope not in {s.value for s in Scope}:
            raise ValueError("invalid Scope in summary")


@dataclass
class ReasoningInput:
    run_id: str
    session_mode: str  # RunMode
    task_priority: str  # TaskPriority
    objective: str
    constraints: Dict[str, Any]  # {must_use_display_scope, max_*, disallowed_ops}
    snapshots: Dict[str, SnapshotSummary]  # Partial<Record<Scope, SnapshotSummary>>
    orchestrator_messages: Optional[List[OrchestratorMessage]] = None
    prior_step_context: Optional[Dict[str, Any]] = None
    output_contract: str = "ACTION_INTENT_JSON_V1"

    def __post_init__(self) -> None:
        if self.session_mode not in {m.value for m in RunMode}:
            raise ValueError("invalid session_mode")
        if self.task_priority not in {p.value for p in TaskPriority}:
            raise ValueError("invalid task_priority")
        if self.constraints.get("must_use_display_scope") != "isolated":
            # Per spec this is the hard constraint; soft here, hard in SafetyKernel
            pass


@dataclass
class ActionIntent:
    intent_id: str
    snapshot_id: str
    action: Dict[str, Any]  # ActionPayloadDict shape (kept dict for variant target flexibility)
    rationale: str
    risk_level: str  # RiskLevel value
    requires_confirmation: bool
    confirmation_token: Optional[str] = None
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.risk_level not in {r.value for r in RiskLevel}:
            raise ValueError("invalid RiskLevel")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0,1]")


@dataclass
class ActionSpec:
    action_id: str
    type: str  # ActionType value
    display_scope: str  # must be "isolated" at execution boundary (FR-04)
    target: Optional[Dict[str, Any]] = None  # Element|Coordinate|DOMTargetDict or None for non-spatial
    source_handle_id: Optional[str] = None
    text: Optional[str] = None
    hotkey: Optional[List[str]] = None
    scroll_delta: Optional[Dict[str, int]] = None
    drag_to: Optional[Dict[str, Any]] = None
    wait_ms: Optional[int] = None
    url: Optional[str] = None
    app: Optional[str] = None
    app_args: Optional[List[str]] = None
    rationale: str = ""
    risk_level: str = "low"  # RiskLevel
    confirmation_token: Optional[str] = None
    clearance_token: Optional[str] = None  # REQUIRED for real_act (kernel-issued); absent for isolated

    def __post_init__(self) -> None:
        # Relaxed for real_act (Step 1): accept "isolated" (byte-identical behavior) or "real_act" (kernel-gated only).
        # Invalid scopes (e.g. observe_real) still rejected for safety.
        if self.display_scope not in {"isolated", "real_act"}:
            raise ValueError(
                f"ActionSpec.display_scope must be exactly 'isolated' (got {self.display_scope}; real_act requires clearance_token)"
            )
        if self.display_scope == "real_act":
            if not self.clearance_token or not str(self.clearance_token).strip():
                raise ValueError(
                    "ActionSpec with display_scope='real_act' requires a non-empty clearance_token "
                    "(SafetyKernel ownership gate only; reasoner cannot fabricate)"
                )
        if self.type not in {a.value for a in ActionType}:
            raise ValueError("invalid ActionType")
        if self.risk_level not in {r.value for r in RiskLevel}:
            raise ValueError("invalid RiskLevel")


@dataclass
class SpawnSpec:
    argv: List[str]
    display_scope: str  # must be "isolated"
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    timeout_ms: Optional[int] = None
    require_owned_group: bool = True
    no_shell: bool = True  # required invariant — __post_init__ enforces
    source_action_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.display_scope != "isolated":
            raise ValueError("SpawnSpec.display_scope must be 'isolated'")
        if not self.no_shell:
            raise ValueError("SpawnSpec.no_shell must be exactly True (non-shell argv required)")
        if not self.argv:
            raise ValueError("SpawnSpec.argv must be non-empty")


@dataclass
class IsolatedDisplaySpec:
    display: str  # e.g. ":99"
    screen: str  # e.g. "1920x1080x24"
    xvfb_binary: Optional[str] = "Xvfb"
    timeout_ms: Optional[int] = None


@dataclass
class AppLaunchPolicy:
    allowed_apps: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    blocked_env_keys: List[str] = field(
        default_factory=lambda: ["DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "LD_PRELOAD"]
    )


@dataclass
class ValidationResult:
    valid: bool
    violations: Optional[List[Dict[str, Any]]] = None  # list of ViolationDict
    normalized_action: Optional[ActionSpec] = None


@dataclass
class GateDecision:
    gate: str  # GateType value
    reason: str
    pending_intent_id: Optional[str] = None
    required_token_scope: Optional[Dict[str, str]] = None  # RequiredTokenScopeDict

    def __post_init__(self) -> None:
        if self.gate not in {g.value for g in GateType}:
            raise ValueError("invalid GateType")


@dataclass
class EngineHealth:
    claude: str  # EngineStatus
    codex: str
    routing_default: str = "codex_then_claude"
    routing_high_priority: str = "claude_then_codex"
    last_error: Optional[str] = None

    def __post_init__(self) -> None:
        for status in (self.claude, self.codex):
            if status not in {e.value for e in EngineStatus}:
                raise ValueError("invalid EngineStatus")


@dataclass
class ConfirmationOutcome:
    outcome: str  # ConfirmationOutcomeType
    token: Optional[str] = None
    intent_id: Optional[str] = None
    received_at: Optional[str] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.outcome not in {c.value for c in ConfirmationOutcomeType}:
            raise ValueError("invalid ConfirmationOutcomeType")


@dataclass
class ActionResult:
    status: str  # ActionStatus
    executed_at: str
    resolved_target: Optional[Dict[str, int]] = None
    resolved_source_bbox: Optional[Dict[str, int]] = None
    spawned_process_ids: Optional[List[int]] = None
    error_code: Optional[str] = None
    artifacts: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.status not in {s.value for s in ActionStatus}:
            raise ValueError("invalid ActionStatus")


@dataclass
class WorkerEvent:
    ts: str
    run_id: str
    event_type: str  # WorkerEventType
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_type not in {e.value for e in WorkerEventType}:
            raise ValueError(f"invalid WorkerEventType: {self.event_type}")


@dataclass
class PromptContext:
    """Context for GuiPrompter.ask (FR-33/34/37). Injected; no real GUI in tests."""
    run_id: str
    pid: int
    action_type: str  # ActionType value
    policy: str  # RealGuiPolicy value
    ask_mode: str  # AskMode value
    target_app_title: Optional[str] = None
    window_id: Optional[str] = None
    rationale: str = ""


@dataclass
class PromptResult:
    """Result from operator via GUI prompter (or FakePrompter in CI)."""
    decision: str
    grant_scope: Optional[str] = None  # GrantScope value or None
    operator_text: str = ""
    granted: bool = False


@dataclass
class Grant:
    """In-memory grant entry (per-run, never persisted)."""
    pid: int
    scope: str  # GrantScope value
    expires_at: Optional[str] = None  # ISO-8601 or None for ACTION/PROCESS_RUN
    run_id: str = ""


# =============================================================================
# Serialization helpers (roundtrip contract for tests + event sink + reasoner)
# __all__ updated minimally in package surface (DOMTargetDict etc. listed in __init__.py); no __all__ here to preserve from .models import * semantics (additive Phase 1.x, INV F).
# =============================================================================

def _normalize_for_dict(obj: Any) -> Any:
    """Recursively turn Enums into .value and dataclasses into plain dicts."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _normalize_for_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_dict(i) for i in obj]
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return obj


def to_dict(obj: Any) -> Dict[str, Any]:
    """Public deterministic to-dict (used by envelope + audit)."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_dict(i) for i in obj]
    if isinstance(obj, dict):
        return _normalize_for_dict(obj)
    return asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj


def from_dict(cls: type, data: Dict[str, Any]) -> Any:
    """Reconstruct dataclass from plain dict (runs __post_init__ validators)."""
    if not isinstance(data, dict):
        raise TypeError("from_dict expects a dict")
    return cls(**data)


def as_json(obj: Any, *, indent: Optional[int] = None) -> str:
    """Deterministic JSON for reasoner payloads and events.jsonl (sort_keys)."""
    d = to_dict(obj)
    return json.dumps(d, sort_keys=True, separators=(",", ":") if indent is None else None, indent=indent)


def from_json(cls: type, json_str: str) -> Any:
    """Parse JSON and reconstruct via from_dict (validates)."""
    data = json.loads(json_str)
    return from_dict(cls, data)


# =============================================================================
# Reasoner I/O contract helpers (spec §6)
# =============================================================================

def make_reasoning_input_envelope(ri: ReasoningInput) -> str:
    """Return the exact prompt body fragment containing the fenced ReasoningInput JSON.

    The full envelope (header + this + footer) lives in ReasonerBridge (later steps).
    Deterministic: stable key order via sort_keys + compact separators.
    """
    return as_json(ri)


ACTION_INTENT_JSON_V1 = "ACTION_INTENT_JSON_V1"

# The exact output shape the reasoner CLIs must emit (first fenced json block)
# {
#   "action_intent": { ...ActionIntent shape... }
# }


# =============================================================================
# Adapter / SessionController support types (Step 11, minimal per spec table §5)
# =============================================================================

@dataclass
class RunHandle:
    """Returned by ComputerUseWorkerAdapter.start()."""
    run_id: str
    status: str = "running"  # running | completed | stopped | failed | errored
    session_id: Optional[str] = None
    events_path: Optional[str] = None


@dataclass
class StopResult:
    """Returned by stop()."""
    run_id: str
    status: str
    terminated: bool = True
    reason: Optional[str] = None


@dataclass
class Ack:
    """Acknowledgement for submit_confirmation and similar ingress APIs."""
    ok: bool = True
    run_id: Optional[str] = None
    intent_id: Optional[str] = None
    message: str = "ok"


# =============================================================================
# Late attachment of serialization helpers (must be after all dataclass defs)
# =============================================================================

# Attach to_dict / from_dict / as_json / from_json to every dataclass for convenience
# (includes Step 11 adapter support types)
_SERIALIZABLE = (
    RunRequest, WorkerSession, CapabilityReport, ElementHandle, CoordinateTarget,
    PerceptionSnapshot, SnapshotSummary, ReasoningInput, ActionIntent, ActionSpec,
    SpawnSpec, IsolatedDisplaySpec, AppLaunchPolicy, ValidationResult, GateDecision,
    EngineHealth, ConfirmationOutcome, ActionResult, WorkerEvent,
    PromptContext, PromptResult, Grant,
    RunHandle, StopResult, Ack,
)

for _cls in _SERIALIZABLE:
    # Capture cls in closure to avoid loop-var bugs
    def _make_to_dict(c: type) -> Any:
        def _to_dict(self: Any) -> Dict[str, Any]:
            d = asdict(self)
            return _normalize_for_dict(d)
        return _to_dict

    def _make_from_dict(c: type) -> Any:
        def _from_dict(cls: type, data: Dict[str, Any]) -> Any:
            return from_dict(c, data)
        return classmethod(_from_dict)

    def _make_as_json(c: type) -> Any:
        def _as_json(self: Any, *, indent: Optional[int] = None) -> str:
            return as_json(self, indent=indent)
        return _as_json

    def _make_from_json(c: type) -> Any:
        def _from_json(cls: type, data: str) -> Any:
            return from_json(c, data)
        return classmethod(_from_json)

    _cls.to_dict = _make_to_dict(_cls)  # type: ignore[attr-defined]
    _cls.from_dict = _make_from_dict(_cls)  # type: ignore[attr-defined]
    _cls.as_json = _make_as_json(_cls)  # type: ignore[attr-defined]
    _cls.from_json = _make_from_json(_cls)  # type: ignore[attr-defined]

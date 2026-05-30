"""Computer-use worker for AgentOrch (scaffolded in Step 1).

Full implementation through Step 11:
- models + utils (Step 1)
- ProcessSupervisor + killable tree + rlimits (Steps 2-3)
- xauth isolation hardening #1 (Step 4)
- SafetyKernel + FR-03/04/09/12/23/24 (Step 5)
- AuditEventSink (Step 6)
- CapabilityBroker (Step 7)
- PerceptionPipeline + collectors + OBSERVE redaction #4 (Step 8)
- ActionExecutor (Step 9)
- ReasonerBridge + routing/timeout/auth safety (Step 10)
- SessionController + ComputerUseWorkerAdapter + core run loop (Step 11)

See COMPUTER_USE_DESIGN.md for authoritative spec.

Step 14 final pass (imports/exports/docs/verification only, no logic):
- from agy_orchestrator.computer_use import OwnershipResolver, FakeOwnershipResolver, Prompter, FakePrompter, GuiPrompter, GrantCache, FakeClock confirms clean surface (three new modules exported + __all__)
- All public classes re-exported cleanly via this __init__
- docs/computer_use.md has 6-line Real-GUI Security Harness summary at top
- Full hermetic test suite (release_blocking + realgui + INV F) green on exact Step-13 commands; zero real-:0/zenity
- Grep verification + full test runs + constraint review completed (operator review next)
"""

# Re-exports for Step 1 (models + utils only)
from .models import *  # noqa: F401,F403
from .utils import redact_secrets, is_redaction_enabled_for_run  # noqa: F401
from .process_supervisor import ProcessSupervisor, SpawnedProc  # noqa: F401
from .xauth import generate_private_xauthority, get_isolated_env  # noqa: F401  # Step 4 hardening #1 XAUTHORITY ISOLATION
from .safety import SafetyKernel, DEFAULT_APP_LAUNCH_POLICY, get_default_app_launch_policy  # noqa: F401  # Step 5: SafetyKernel + AppLaunchPolicy (FR-03/04/09/12/23/24)
from .audit import AuditEventSink  # noqa: F401  # Step 6: append-only JSONL audit sink for runs/<id>/events.jsonl (FR-13)
from .capability import CapabilityBroker, probe_capabilities, is_available  # noqa: F401  # Step 7: CapabilityBroker.probe() + is_available() delegation point for adapter (FR-05/15)
from .perception import (  # noqa: F401  # Step 8: PerceptionPipeline + collectors (FR-05/06/17 + OBSERVE redaction hardening #4)
    ATSPICollector,
    BrowserDOMCollector,
    GeometryCollector,
    OCRCollector,
    PerceptionPipeline,
)
from .action_executor import ActionExecutor  # noqa: F401  # Step 9: isolated-display xdotool executor (FR-04/24 + hardening #1)
from .reasoner import ReasonerBridge, ReasonerError, build_reasoner_prompt_envelope  # noqa: F401  # Step 10: ReasonerBridge + exact CLI contract + FR-14/16/20/21/25 + events
from .session import SessionController  # noqa: F401  # Step 11: lifecycle + Xvfb ownership (FR-22) + suspended confirmation (FR-18/19)
from .adapter import (  # noqa: F401  # Step 11: public contract + core run loop
    Ack,
    ComputerUseWorkerAdapter,
    RunHandle,
    StopResult,
)
# Step 14: Real-GUI Security Harness (exports for the three new modules; zero functional change)
from .ownership import OwnershipResolver, FakeOwnershipResolver  # noqa: F401
from .gui_prompt import Prompter, FakePrompter, GuiPrompter  # noqa: F401
from .grants import GrantCache, FakeClock  # noqa: F401
# Phase 1.x Step 3: BrowserController + FakeBrowserController (additive export only; INV F / B4 preserved exactly)
from .browser import BrowserController, FakeBrowserController  # noqa: F401  # Step 3: new module (real + Fake double; refined+finalized)
# Make the Step-2 SpawnedProc the canonical one on the package namespace
# (the "from .models import *" above must not shadow it).
globals()["SpawnedProc"] = SpawnedProc

__all__ = [  # explicit for clean surface (Step 1)
    # Enums (exact spec fidelity)
    "RunMode",
    "Scope",
    "TaskPriority",
    "RiskLevel",
    "ActionType",
    "ElementSource",
    "TextBlockSource",
    "CoordinateSpace",
    "Readiness",
    "ConfirmationOutcomeType",
    "ActionStatus",
    "GateType",
    "EngineStatus",
    "WorkerEventType",
    "ViolationCode",
    # Core models (dataclasses, every TS interface from spec §6)
    "RunRequest",
    "WorkerSession",
    "CapabilityReport",
    "ElementHandle",
    "CoordinateTarget",
    "PerceptionSnapshot",
    "SnapshotSummary",
    "ReasoningInput",
    "ActionIntent",
    "ActionSpec",
    "SpawnSpec",
    "IsolatedDisplaySpec",
    "SpawnedProc",
    "AppLaunchPolicy",
    "ValidationResult",
    "GateDecision",
    "EngineHealth",
    "ConfirmationOutcome",
    "ActionResult",
    "WorkerEvent",
    "OrchestratorMessage",
    "IntentTarget",
    "DOMTargetDict",  # Phase 1.x (addendum): DOM target kind for autonomous browser/DOM actuation (additive; B4 INV F)
    # Serialization (deterministic, used by events.jsonl + reasoner envelope)
    "to_dict",
    "from_dict",
    "as_json",
    "from_json",
    "make_reasoning_input_envelope",
    "ACTION_INTENT_JSON_V1",
    # utils (OBSERVE redaction hardening #4 + opt-out)
    "redact_secrets",
    "is_redaction_enabled_for_run",
    # xauth (Step 4: XAUTHORITY isolation hardening #1)
    "generate_private_xauthority",
    "get_isolated_env",
    # ProcessSupervisor (Step 2: killable tree foundation, hardening #2)
    "ProcessSupervisor",
    # SafetyKernel (Step 5: mandatory pre-action gate + AppLaunchPolicy, FR-03/04/09/12/23/24)
    "SafetyKernel",
    "DEFAULT_APP_LAUNCH_POLICY",
    "get_default_app_launch_policy",
    # AuditEventSink (Step 6: mandatory append-only JSONL to runs/<id>/events.jsonl + callbacks, FR-13)
    "AuditEventSink",
    # CapabilityBroker (Step 7: probe() -> CapabilityReport, graceful AT-SPI degrade FR-05/15)
    "CapabilityBroker",
    "probe_capabilities",
    "is_available",
    # PerceptionPipeline (Step 8: collectors + fusion + OBSERVE redaction #4 + FR-17 multi-scope)
    "PerceptionPipeline",
    "ATSPICollector",
    "GeometryCollector",
    "OCRCollector",
    "BrowserDOMCollector",
    # Step 9: ActionExecutor (isolated display only, FR-04/24 + hardening #1)
    "ActionExecutor",
    # Step 10: ReasonerBridge (CLI contract, priority routing, timeout/auth/parse safety, FR-14/16/20/21/25)
    "ReasonerBridge",
    "ReasonerError",
    "build_reasoner_prompt_envelope",
    # Step 11
    "SessionController",
    "RunHandle",
    "StopResult",
    "Ack",
    "ComputerUseWorkerAdapter",
    # Step 14: Real-GUI Security Harness (ownership+baseline, prompter, grants — clean re-exports)
    "OwnershipResolver",
    "FakeOwnershipResolver",
    "Prompter",
    "FakePrompter",
    "GuiPrompter",
    "GrantCache",
    "FakeClock",
    # Phase 1.x Step 3: BrowserController + Fake (additive only; hermetic double for all browser tests; finalized)
    "BrowserController",
    "FakeBrowserController",
]

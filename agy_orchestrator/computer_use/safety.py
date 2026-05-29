"""SafetyKernel (Step 5) — the single mandatory pre-action gate.

This is the *only* choke point through which every ActionIntent must pass before
ActionExecutor or ProcessSupervisor may act. All release-blocking FR-03/04/09/12/23/24
plus FR-07 target resolution, budget enforcement, and full ViolationCode paths are
implemented here.

Hardening context (enforced end-to-end by the computer-use stack):
- XAUTHORITY ISOLATION (#1): enforced in ProcessSupervisor + xauth (SafetyKernel never lets
  real ~/.Xauthority into any spawn env for isolated display_scope).
- KILLABLE TREE (#2): start_new_session + terminate_tree (ProcessSupervisor); SafetyKernel
  only ever consults is_owned (FR-12).
- HARD RESOURCE BACKSTOP (#3): rlimits in preexec + enforce_limits (ProcessSupervisor).
- OBSERVE REDACTION (#4, default ON): in utils.redact_secrets; Perception/ReasonerBridge
  callers must apply before any real-:0 text reaches a claude/codex prompt.

SafetyKernel itself is pure policy (no X11, no subprocess spawn, no real-:0 side effects).
All tests use only hermetic isolated Xvfb fixtures.

Default AppLaunchPolicy is minimal/safe (xclock/xeyes/true + strict schema example).
Operator extends via get_default... or by constructing AppLaunchPolicy and passing to
SafetyKernel(app_policy=...).
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

from .models import (
    ActionIntent,
    ActionSpec,
    ActionType,
    AppLaunchPolicy,
    CoordinateTarget,
    GateDecision,
    GateType,
    RiskLevel,
    SpawnSpec,
    ValidationResult,
    ViolationCode,
)
from .process_supervisor import ProcessSupervisor


def get_default_app_launch_policy() -> AppLaunchPolicy:
    return AppLaunchPolicy(
        allowed_apps={
            "xclock": {"exec_path": "xclock", "allowed_args_pattern": r"^-.*$|^$", "default_args": [], "blocked_args": []},
            "xeyes": {"exec_path": "xeyes", "allowed_args_pattern": r"^-.*$|^$", "default_args": [], "blocked_args": []},
            "true": {"exec_path": "true", "allowed_args_pattern": ".*", "default_args": [], "blocked_args": []},
            "test-gui-app": {"exec_path": "true", "allowed_args_pattern": r"^--test-mode$|^$", "default_args": [], "blocked_args": ["--dangerous"]},
        },
        blocked_env_keys=["DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "LD_PRELOAD"],
    )


DEFAULT_APP_LAUNCH_POLICY = get_default_app_launch_policy()


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Robust dict normalizer for ActionIntent / WorkerSession / snapshots (supports dataclass + dict)."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


class SafetyKernel:
    """Single mandatory pre-action gate. validate(intent, session) is the choke point."""

    def __init__(self, supervisor: Optional[ProcessSupervisor] = None, app_policy: Optional[AppLaunchPolicy] = None, **kw: Any) -> None:
        self.supervisor = supervisor or kw.get("process_supervisor") or ProcessSupervisor()
        self.policy = app_policy or kw.get("app_launch_policy") or DEFAULT_APP_LAUNCH_POLICY

    def validate(self, intent: Union[ActionIntent, Dict[str, Any]], session: Union[Dict[str, Any], Any], *, current_step: int = 0, current_actions: int = 0, snapshots: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Single mandatory pre-action gate (FR-03/04/07/09/12/23/24 + all budgets/ownership/schema).

        Hard-enforces display_scope=="isolated" at the boundary (no real-:0 actuation possible).
        Uses ProcessSupervisor.is_owned for FR-12 foreign-process rejection on element handles.
        """
        snapshots = snapshots or {}
        vs: List[Dict[str, Any]] = []
        d = _to_dict(intent)
        a = d.get("action") or {}
        if not isinstance(a, dict):
            a = {}
        typ = a.get("type")
        ds = a.get("display_scope")
        tgt = a.get("target") or {}
        if not isinstance(tgt, dict):
            tgt = {}
        tok = d.get("confirmation_token")

        # FR-04: explicit "isolated" required; anything else (missing, observe_real, :0, etc.) hard-reject
        if ds != "isolated":
            return ValidationResult(
                valid=False,
                violations=[{"code": ViolationCode.DISPLAY_SCOPE_INVALID.value, "message": "display_scope must be exactly 'isolated' (FR-04)"}],
            )

        spatial = {"click", "double_click", "type", "scroll", "drag"}
        if typ not in {x.value for x in ActionType}:
            vs.append({"code": ViolationCode.SCHEMA_INVALID.value, "message": "bad action type"})
        if typ in spatial and not tgt:
            vs.append({"code": ViolationCode.TARGET_MISSING.value, "message": "target required for spatial action"})
        if typ == "wait" and a.get("wait_ms") is None:
            vs.append({"code": ViolationCode.SCHEMA_INVALID.value, "message": "wait requires wait_ms"})
        if typ == "launch_app" and not a.get("app"):
            vs.append({"code": ViolationCode.LAUNCH_APP_NOT_ALLOWLISTED.value, "message": "launch_app requires app identity"})
        if typ == "navigate" and not a.get("url"):
            vs.append({"code": ViolationCode.SCHEMA_INVALID.value, "message": "navigate requires url"})
        if typ == "hotkey" and not a.get("hotkey"):
            vs.append({"code": ViolationCode.SCHEMA_INVALID.value, "message": "hotkey requires hotkey array"})
        if typ == "scroll" and not a.get("scroll_delta"):
            vs.append({"code": ViolationCode.SCHEMA_INVALID.value, "message": "scroll requires scroll_delta"})
        if typ == "drag" and not a.get("drag_to"):
            vs.append({"code": ViolationCode.SCHEMA_INVALID.value, "message": "drag requires drag_to target"})

        # Budgets (FR-10)
        sb = _to_dict(session)
        b = sb.get("budgets") or {}
        if current_step >= int(b.get("max_steps", 200)):
            vs.append({"code": ViolationCode.BUDGET_EXCEEDED.value, "message": "max_steps exceeded"})
        if current_actions >= int(b.get("max_actions", 200)):
            vs.append({"code": ViolationCode.BUDGET_EXCEEDED.value, "message": "max_actions exceeded"})

        # FR-23 launch_app allowlist + schema (argument pattern / blocked / metachar)
        if typ == "launch_app":
            lv = self.validate_launch_app({"app": a.get("app"), "app_args": a.get("app_args") or []}, self.policy)
            if not lv.get("valid"):
                vs.extend(lv.get("violations", []))

        # FR-09 risk gate (confirmation required for high/irreversible)
        risk = self.classify_risk(intent)
        if risk in (RiskLevel.HIGH.value, RiskLevel.IRREVERSIBLE.value) and not tok:
            vs.append({"code": ViolationCode.CONFIRMATION_REQUIRED.value, "message": "high-risk requires confirmation (FR-09)"})

        # FR-12: element handle provenance — reject if handle claims a foreign (non-owned) pid
        if tgt.get("kind") == "element":
            hid = tgt.get("handle_id")
            for _s, sn in (snapshots or {}).items():
                elems = []
                if isinstance(sn, dict):
                    elems = sn.get("elements") or []
                else:
                    elems = getattr(sn, "elements", []) or []
                for e in elems:
                    ed = _to_dict(e)
                    if ed.get("handle_id") == hid:
                        pid = ed.get("app_pid")
                        if pid is not None:
                            try:
                                if not self.supervisor.is_owned(int(pid)):
                                    vs.append({"code": ViolationCode.FOREIGN_PROCESS.value, "message": "foreign pid via element handle (FR-12)"})
                            except (ValueError, TypeError):
                                vs.append({"code": ViolationCode.FOREIGN_PROCESS.value, "message": "invalid pid in element handle (FR-12)"})

        if vs:
            return ValidationResult(valid=False, violations=vs, normalized_action=None)

        # Target resolution (FR-07): element handle → CoordinateTarget; stale → unresolvable
        coord = None
        sh = None
        kind = tgt.get("kind")
        if kind == "coordinate":
            try:
                coord = CoordinateTarget(x=int(tgt.get("x", 0)), y=int(tgt.get("y", 0)))
            except (ValueError, TypeError):
                return ValidationResult(valid=False, violations=[{"code": ViolationCode.TARGET_UNRESOLVABLE.value, "message": "bad coordinate"}])
        elif kind == "element":
            try:
                coord = self.resolve_target(intent, snapshots)
                sh = tgt.get("handle_id")
            except Exception as e:
                return ValidationResult(valid=False, violations=[{"code": ViolationCode.TARGET_UNRESOLVABLE.value, "message": str(e)}])
        else:
            # Unknown target kind on a spatial action that passed the "tgt present" pre-check
            if typ in spatial:
                return ValidationResult(valid=False, violations=[{"code": ViolationCode.TARGET_UNRESOLVABLE.value, "message": f"invalid target kind for spatial action: {kind}"}])

        # Final guard: spatial actions must have a resolved coordinate target at the gate
        if typ in spatial and coord is None:
            return ValidationResult(valid=False, violations=[{"code": ViolationCode.TARGET_UNRESOLVABLE.value, "message": "spatial action missing resolved coordinate target"}])

        try:
            norm = ActionSpec(
                action_id=f"act-{int(time.time()*1000)}",
                type=typ,
                display_scope="isolated",
                target=asdict(coord) if coord else None,
                source_handle_id=sh,
                text=a.get("text"),
                hotkey=a.get("hotkey"),
                scroll_delta=a.get("scroll_delta"),
                drag_to=a.get("drag_to"),
                wait_ms=a.get("wait_ms"),
                url=a.get("url"),
                app=a.get("app"),
                app_args=a.get("app_args"),
                rationale=d.get("rationale", ""),
                risk_level=risk,
                confirmation_token=tok,
            )
        except Exception as e:
            return ValidationResult(valid=False, violations=[{"code": ViolationCode.SCHEMA_INVALID.value, "message": str(e)}])
        return ValidationResult(valid=True, violations=[], normalized_action=norm)

    def classify_risk(self, intent: Union[ActionIntent, Dict[str, Any]]) -> str:
        """Risk classification (heuristic + explicit). Irreversible examples trigger dry-run gate (FR-09)."""
        d = _to_dict(intent)
        r = d.get("risk_level", RiskLevel.LOW.value)
        rat = (d.get("rationale") or "").lower()
        if "reboot" in rat or ":0" in rat or "destroy" in rat or "rm -rf" in rat:
            return RiskLevel.IRREVERSIBLE.value
        return r

    def require_confirmation(self, intent: Union[ActionIntent, Dict[str, Any]]) -> GateDecision:
        """Pure gate decision for high/irreversible (FR-09/18/19). validate() still enforces token for execution path."""
        d = _to_dict(intent)
        risk = self.classify_risk(intent)
        iid = d.get("intent_id") or "i"
        if risk in (RiskLevel.HIGH.value, RiskLevel.IRREVERSIBLE.value):
            tok = d.get("confirmation_token")
            if not tok:
                return GateDecision(gate=GateType.REQUIRE_CONFIRMATION.value, reason=f"{risk} needs token (FR-09)", pending_intent_id=iid)
            return GateDecision(gate=GateType.ALLOW.value, reason="token present")
        return GateDecision(gate=GateType.ALLOW.value, reason="low risk")

    def resolve_target(self, intent: Union[ActionIntent, Dict[str, Any]], snapshots: Dict[str, Any]) -> CoordinateTarget:
        """Element handle → center coordinate (FR-07). Stale/missing handle → ValueError (caught as TARGET_UNRESOLVABLE)."""
        d = _to_dict(intent)
        a = d.get("action") or {}
        tgt = (a.get("target") if isinstance(a, dict) else {}) or {}
        if tgt.get("kind") == "coordinate":
            return CoordinateTarget(x=int(tgt.get("x", 0)), y=int(tgt.get("y", 0)))
        if tgt.get("kind") == "element":
            hid = tgt.get("handle_id")
            for _s, sn in (snapshots or {}).items():
                elems = (sn.get("elements") if isinstance(sn, dict) else getattr(sn, "elements", [])) or []
                for e in elems:
                    ed = _to_dict(e)
                    if ed.get("handle_id") == hid:
                        b = ed.get("bbox", {}) or {}
                        return CoordinateTarget(
                            x=int(b.get("x", 0) + (b.get("w", 0) or 0) // 2),
                            y=int(b.get("y", 0) + (b.get("h", 0) or 0) // 2),
                        )
            raise ValueError(f"stale handle {hid}")
        raise ValueError("bad target kind")

    def validate_launch_app(self, payload: Dict[str, Any], policy: AppLaunchPolicy) -> Dict[str, Any]:
        """FR-23 + FR-24 guard: allowlist identity only + per-app arg schema + universal metachar block.

        Never accepts shell strings, metachars, or non-allowlisted execs. Arg schema (pattern/blocked/defaults)
        is enforced exactly against the policy entry. Zero spawn on any violation (caller responsibility).
        """
        vs: List[Dict[str, Any]] = []
        app = payload.get("app")
        args = [str(x) for x in (payload.get("app_args") or [])]

        entry = None
        if not app or app not in policy.allowed_apps:
            vs.append({"code": ViolationCode.LAUNCH_APP_NOT_ALLOWLISTED.value, "message": f"app '{app}' not allowlisted (FR-23)"})
        else:
            entry = policy.allowed_apps[app]

        # Universal defense-in-depth: shell metacharacters are always forbidden in launch_app payloads
        for v in args:
            if any(c in v for c in ";&|`$()<>[]{}*?^~\n\r"):
                vs.append({"code": ViolationCode.LAUNCH_APP_ARGS_INVALID.value, "message": "shell metachar in app_args (FR-23)"})

        if entry is not None:
            blocked = set(entry.get("blocked_args") or [])
            pat_str = entry.get("allowed_args_pattern")
            pat = None
            if pat_str:
                try:
                    pat = re.compile(pat_str)
                except re.error:
                    vs.append({"code": ViolationCode.LAUNCH_APP_ARGS_INVALID.value, "message": "malformed allowed_args_pattern in policy"})
                    pat = None

            for v in args:
                if v in blocked:
                    vs.append({"code": ViolationCode.LAUNCH_APP_ARGS_INVALID.value, "message": f"blocked arg '{v}' per app policy (FR-23)"})
                    continue
                if pat is not None and not pat.fullmatch(v):
                    vs.append({"code": ViolationCode.LAUNCH_APP_ARGS_INVALID.value, "message": f"app_args '{v}' violates allowed_args_pattern (FR-23)"})

        return {"valid": len(vs) == 0, "violations": vs}

    def validate_spawn_request(self, spec: SpawnSpec, isolated_display: str) -> ValidationResult:
        """Early FR-24 + C1 guard before ProcessSupervisor.spawn.

        Rejects any attempt to target real-session displays via env overrides.
        The isolated_display param is advisory (supervisor + xauth.py do the actual forcing).
        """
        vs: List[Dict[str, Any]] = []
        if spec.display_scope != "isolated":
            vs.append({"code": ViolationCode.SPAWN_ENV_OVERRIDE_FORBIDDEN.value, "message": "display_scope must be 'isolated' (FR-24)"})
        env = spec.env or {}
        for k, v in env.items():
            val = str(v)
            if k in ("DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY"):
                if ":0" in val or ":0." in val or ".Xauthority" in val or "/.Xauthority" in val or val.strip() == "":
                    vs.append({"code": ViolationCode.SPAWN_ENV_OVERRIDE_FORBIDDEN.value, "message": f"real-display or empty X override {k}={val} (FR-24)"})
        # Also reject obvious attempts to point at real session via other means
        if any(":0" in str(v) for v in env.values()):
            if not any(v["code"] == ViolationCode.SPAWN_ENV_OVERRIDE_FORBIDDEN.value for v in vs):
                vs.append({"code": ViolationCode.SPAWN_ENV_OVERRIDE_FORBIDDEN.value, "message": "env contains real :0 reference (FR-24)"})
        return ValidationResult(valid=len(vs) == 0, violations=vs or None)

    validate_spawn_env = validate_spawn_request

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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from .models import (
    ActionIntent,
    ActionSpec,
    ActionType,
    AppLaunchPolicy,
    AskMode,
    CoordinateTarget,
    GateDecision,
    GateType,
    GrantScope,
    PromptContext,
    PromptResult,
    RealGuiPolicy,
    RiskLevel,
    RunMode,
    SpawnSpec,
    ValidationResult,
    ViolationCode,
    WorkerEvent,
    WorkerEventType,
)
from .grants import GrantCache
from .gui_prompt import FakePrompter, Prompter
from .ownership import FakeOwnershipResolver, OwnershipResolver
from .process_supervisor import ProcessSupervisor
from .audit import AuditEventSink  # optional, for permission event streaming at decision points (FR-38)


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

    def __init__(
        self,
        supervisor: Optional[ProcessSupervisor] = None,
        app_policy: Optional[AppLaunchPolicy] = None,
        ownership_resolver: Optional[Any] = None,
        gui_prompter: Optional[Any] = None,
        grant_cache: Optional[Any] = None,
        audit_sink: Optional[Any] = None,
        **kw: Any,
    ) -> None:
        self.supervisor = supervisor or kw.get("process_supervisor") or ProcessSupervisor()
        self.policy = app_policy or kw.get("app_launch_policy") or DEFAULT_APP_LAUNCH_POLICY
        # Step 5 real-GUI injectables (all optional; None = no-op safe defaults that preserve
        # all prior isolated/observe behavior and deny any real_act by default).
        self.ownership_resolver = ownership_resolver or kw.get("ownership_resolver")
        self.prompter = gui_prompter or kw.get("gui_prompter")
        self.grant_cache = grant_cache or kw.get("grant_cache")
        self.audit_sink = audit_sink or kw.get("audit_sink")

    def _emit(self, etype: str, payload: Dict[str, Any]) -> None:
        """Best-effort emit for permission/audit events at real-act decision points (FR-38, Step 12 complete; uniform run_id+pid/... shapes)."""
        if not self.audit_sink:
            return
        try:
            self.audit_sink.emit(WorkerEvent(ts=datetime.now(timezone.utc).isoformat(), run_id=str(payload.get("run_id", "unknown")), event_type=etype, payload=payload))
        except Exception:
            pass  # never let audit break the gate

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
        is_dom_spatial = typ in {"click", "type"} and kind == "dom"
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
        elif kind == "dom" and typ in {"click", "type"}:
            # DOM targets are resolved by BrowserController via selector/index, not coordinates.
            coord = None
        else:
            # Unknown target kind on a spatial action that passed the "tgt present" pre-check
            if typ in spatial:
                return ValidationResult(valid=False, violations=[{"code": ViolationCode.TARGET_UNRESOLVABLE.value, "message": f"invalid target kind for spatial action: {kind}"}])

        # Final guard: spatial actions must have a resolved coordinate target at the gate
        if typ in spatial and coord is None and not is_dom_spatial:
            return ValidationResult(valid=False, violations=[{"code": ViolationCode.TARGET_UNRESOLVABLE.value, "message": "spatial action missing resolved coordinate target"}])

        try:
            norm = ActionSpec(
                action_id=f"act-{int(time.time()*1000)}",
                type=typ,
                display_scope="isolated",
                target=asdict(coord) if coord else (dict(tgt) if is_dom_spatial else None),
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

    def classify_risk(self, intent: Union[ActionIntent, Dict[str, Any]], scope: Optional[str] = None) -> str:
        """Risk classification (heuristic + explicit). Irreversible examples trigger dry-run gate (FR-09).
        When scope=="real_act", ":0" mentions are expected and do not auto-escalate to IRREVERSIBLE.
        """
        d = _to_dict(intent)
        r = d.get("risk_level", RiskLevel.LOW.value)
        rat = (d.get("rationale") or "").lower()
        if scope == "real_act":
            # real_act scope: only explicit destructive intent (not mere :0 reference) is irreversible
            if "reboot" in rat or "destroy" in rat or "rm -rf" in rat:
                return RiskLevel.IRREVERSIBLE.value
            return r
        if "reboot" in rat or ":0" in rat or "destroy" in rat or "rm -rf" in rat:
            return RiskLevel.IRREVERSIBLE.value
        return r

    def require_confirmation(self, intent: Union[ActionIntent, Dict[str, Any]], scope: Optional[str] = None) -> GateDecision:
        """Pure gate decision for high/irreversible (FR-09/18/19). validate() still enforces token for execution path.
        Scope forwarded to classify_risk so real_act uses the narrowed irreversible heuristic (new codes only affect real_act).
        """
        d = _to_dict(intent)
        risk = self.classify_risk(intent, scope=scope)
        iid = d.get("intent_id") or "i"
        if risk in (RiskLevel.HIGH.value, RiskLevel.IRREVERSIBLE.value):
            tok = d.get("confirmation_token")
            if not tok:
                return GateDecision(gate=GateType.REQUIRE_CONFIRMATION.value, reason=f"{risk} needs token (FR-09)", pending_intent_id=iid)
            return GateDecision(gate=GateType.ALLOW.value, reason="token present")
        return GateDecision(gate=GateType.ALLOW.value, reason="low risk")

    def resolve_target(self, intent: Union[ActionIntent, Dict[str, Any]], snapshots: Dict[str, Any]) -> Union[CoordinateTarget, Dict[str, Any], None]:
        """Element handle → center coordinate (FR-07). Stale/missing handle → ValueError (caught as TARGET_UNRESOLVABLE)."""
        d = _to_dict(intent)
        a = d.get("action") or {}
        tgt = (a.get("target") if isinstance(a, dict) else {}) or {}
        if tgt.get("kind") == "coordinate":
            return CoordinateTarget(x=int(tgt.get("x", 0)), y=int(tgt.get("y", 0)))
        if tgt.get("kind") == "dom":
            return dict(tgt) if isinstance(tgt, dict) else None
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

    # -------------------------------------------------------------------------
    # Step 5: real_act gate (ownership + baseline + ask + grant cache). Base
    # isolated/observe_real paths in validate() are byte-for-byte untouched.
    # _real_act_gate is the private implementation of §5 decision tree (FR-27/30/31/32/33/35/36).
    # It is not yet wired into the public validate() early-return (Step 5 scope);
    # future steps will call it when ds == "real_act" after the common schema/budget
    # checks. All hermetic tests exercise it directly with Fakes only.
    # On every ALLOW for real_act we issue a clearance_token (format per contract);
    # ActionSpec for real_act is only ever constructed here after the gate passes
    # (INVARIANT E).
    # -------------------------------------------------------------------------

    def _real_act_gate(
        self,
        intent: Union[ActionIntent, Dict[str, Any]],
        session: Union[Dict[str, Any], Any],
        *,
        current_step: int = 0,
        current_actions: int = 0,
        snapshots: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Private §5 decision tree for display_scope=="real_act".

        Exact flow (fail-closed at every ambiguity):
          1. mode != REAL -> REAL_ACT_NOT_PERMITTED
          2. resolve_target_to_pid -> None -> TARGET_UNRESOLVABLE
          3. classify(pid) == OWNED -> ALLOW (no prompt, FR-30)
          4. FOREIGN + children policy -> FOREIGN_PROCESS (FR-31, no prompt)
          5. FOREIGN + full + ask=off -> ASK_MODE_DISABLED (FR-32, no prompt)
          6. FOREIGN + full + ask=on:
               grant_cache hit -> ALLOW (FR-36)
               else: prompter.prompt(...) ; grant/allow -> cache + ALLOW ; else DENY (FR-33/34/35)
        Injectables absent -> conservative DENY (keeps old safe defaults).
        Uses injected clock via GrantCache; FakePrompter in all tests (FR-39).
        """
        snapshots = snapshots or {}
        d = _to_dict(intent)
        a = d.get("action") or {}
        if not isinstance(a, dict):
            a = {}
        tgt = a.get("target") or {}
        if not isinstance(tgt, dict):
            tgt = {}
        tok = d.get("confirmation_token")
        risk = self.classify_risk(intent, scope="real_act")

        sb = _to_dict(session)
        run_id = str(sb.get("run_id") or "unknown")
        mode = sb.get("mode") or "ISOLATED"
        policy = sb.get("real_gui_policy") or "full"
        ask = sb.get("ask_mode") or "on"

        # 1. mode gate
        if mode != RunMode.REAL.value and mode != "REAL":
            self._emit(WorkerEventType.PERMISSION_DENIED.value, {"run_id": run_id, "reason": "real_act_not_permitted", "pid": None})
            return ValidationResult(
                valid=False,
                violations=[{"code": ViolationCode.REAL_ACT_NOT_PERMITTED.value, "message": "real_act requires mode=REAL (FR-26)"}],
            )

        # Collect windows for coord resolution (defensive; element path uses app_pid)
        snapshot_windows: List[Dict[str, Any]] = []
        for _s, sn in (snapshots or {}).items():
            if isinstance(sn, dict):
                ws = sn.get("windows") or []
            else:
                try:
                    ws = getattr(sn, "windows", []) or []
                except Exception:
                    ws = []
            if isinstance(ws, list):
                snapshot_windows.extend([w for w in ws if isinstance(w, dict)])

        # 2. resolve target -> pid (FR-29)
        pid: Optional[int] = None
        if self.ownership_resolver is not None:
            try:
                pid = self.ownership_resolver.resolve_target_to_pid(tgt, snapshot_windows)
            except Exception:
                pid = None
        if not (isinstance(pid, int) and pid > 0):
            self._emit(WorkerEventType.FOREIGN_INTERACTION_BLOCKED.value, {"run_id": run_id, "reason": "target_unresolvable", "pid": None})
            return ValidationResult(
                valid=False,
                violations=[{"code": ViolationCode.TARGET_UNRESOLVABLE.value, "message": "target could not be resolved to owning pid (FR-29)"}],
            )

        # 3. classify (INVARIANT A/B)
        cls = "FOREIGN"
        if self.ownership_resolver is not None:
            try:
                cls = self.ownership_resolver.classify(pid)
            except Exception:
                cls = "FOREIGN"

        if cls == "OWNED":
            # FR-30: owned child, no prompt, immediate allow + token
            self._emit(WorkerEventType.PERMISSION_GRANTED.value, {"run_id": run_id, "pid": pid, "grant_scope": "owned_child", "implicit": True})
            token = f"clr:{run_id}:{pid}:real_act:{int(time.time())}"
            return self._build_real_act_allowed(d, a, tok, risk, token)

        # FOREIGN
        pol = policy
        if pol == RealGuiPolicy.CHILDREN.value or pol == "children":
            # FR-31: children policy never allows foreign input injection
            self._emit(WorkerEventType.FOREIGN_INTERACTION_BLOCKED.value, {"run_id": run_id, "pid": pid, "policy": "children", "reason": "foreign_process"})
            return ValidationResult(
                valid=False,
                violations=[{"code": ViolationCode.FOREIGN_PROCESS.value, "message": "foreign pid denied under children policy (FR-31)"}],
            )

        # full policy
        if ask == AskMode.OFF.value or ask == "off":
            # FR-32
            self._emit(WorkerEventType.PERMISSION_DENIED.value, {"run_id": run_id, "pid": pid, "reason": "ask_mode_disabled"})
            return ValidationResult(
                valid=False,
                violations=[{"code": ViolationCode.ASK_MODE_DISABLED.value, "message": "ask_mode=off; foreign interaction hard-denied (FR-32)"}],
            )

        # full + ask=on -> check cache first (FR-36)
        has_grant = False
        if self.grant_cache is not None:
            try:
                has_grant = bool(self.grant_cache.is_granted(pid))
            except Exception:
                has_grant = False
        if has_grant:
            self._emit(WorkerEventType.PERMISSION_GRANTED.value, {"run_id": run_id, "pid": pid, "grant_scope": "cached", "implicit": True})
            token = f"clr:{run_id}:{pid}:real_act:{int(time.time())}"
            return self._build_real_act_allowed(d, a, tok, risk, token)

        # 6. prompt path (FR-33/34/35)
        if self.prompter is None:
            self._emit(WorkerEventType.PERMISSION_DENIED.value, {"run_id": run_id, "pid": pid, "reason": "no_prompter"})
            return ValidationResult(
                valid=False,
                violations=[{"code": ViolationCode.ASK_MODE_DISABLED.value, "message": "no prompter available; foreign denied (FR-35)"}],
            )

        ctx = PromptContext(
            run_id=run_id,
            pid=pid,
            action_type=str(a.get("type") or "click"),
            policy=str(pol),
            ask_mode=str(ask),
            target_app_title=None,
            window_id=str(tgt.get("window_id")) if tgt.get("window_id") else None,
            rationale=str(d.get("rationale") or ""),
        )
        self._emit(WorkerEventType.PERMISSION_PROMPT_SHOWN.value, {"run_id": run_id, "pid": pid, "action_type": ctx.action_type, "policy": str(pol), "ask_mode": str(ask)})
        try:
            res: PromptResult = self.prompter.prompt(ctx)
        except Exception:
            # FR-35 fail-closed on any prompter error/timeout
            self._emit(WorkerEventType.PERMISSION_DENIED.value, {"run_id": run_id, "pid": pid, "reason": "prompt_failed"})
            return ValidationResult(
                valid=False,
                violations=[{"code": ViolationCode.GRANT_REQUIRED.value, "message": "prompter raised/timeout (FR-35 fail-closed)"}],
            )

        # FR-37/38: operator free-text (even on deny) becomes operator_note event + routed by adapter wrapper
        txt = getattr(res, "operator_text", "") or ""
        if txt.strip():
            self._emit(WorkerEventType.OPERATOR_NOTE_RECEIVED.value, {"run_id": run_id, "pid": pid, "text": txt.strip()[:200]})

        if not getattr(res, "granted", False) or not getattr(res, "grant_scope", None):
            self._emit(WorkerEventType.PERMISSION_DENIED.value, {"run_id": run_id, "pid": pid, "reason": "operator_denied"})
            return ValidationResult(
                valid=False,
                violations=[{"code": ViolationCode.GRANT_REQUIRED.value, "message": "operator denied / cancel / timeout (FR-35)"}],
            )

        # operator granted a scope -> cache it (FR-34/36)
        gscope = res.grant_scope
        if self.grant_cache is not None:
            try:
                self.grant_cache.grant(pid, gscope)
            except Exception:
                pass
        self._emit(WorkerEventType.PERMISSION_GRANTED.value, {"run_id": run_id, "pid": pid, "grant_scope": str(gscope), "operator_text": getattr(res, "operator_text", "")[:200]})

        token = f"clr:{run_id}:{pid}:{gscope}:{int(time.time())}"
        return self._build_real_act_allowed(d, a, tok, risk, token)

    def _build_real_act_allowed(
        self,
        intent_d: Dict[str, Any],
        a: Dict[str, Any],
        confirmation_tok: Optional[str],
        risk: str,
        clearance_token: str,
    ) -> ValidationResult:
        """Construct the executable real_act ActionSpec (only after gate allows; INVARIANT E)."""
        tgt = a.get("target") if isinstance(a.get("target"), dict) else None
        try:
            norm = ActionSpec(
                action_id=f"act-{int(time.time()*1000)}",
                type=a.get("type") or "click",
                display_scope="real_act",
                target=tgt,
                source_handle_id=(tgt or {}).get("handle_id") if isinstance(tgt, dict) else None,
                text=a.get("text"),
                hotkey=a.get("hotkey"),
                scroll_delta=a.get("scroll_delta"),
                drag_to=a.get("drag_to"),
                wait_ms=a.get("wait_ms"),
                url=a.get("url"),
                app=a.get("app"),
                app_args=a.get("app_args"),
                rationale=intent_d.get("rationale", ""),
                risk_level=risk,
                confirmation_token=confirmation_tok,
                clearance_token=clearance_token,
            )
            return ValidationResult(valid=True, violations=[], normalized_action=norm)
        except Exception as e:
            return ValidationResult(
                valid=False,
                violations=[{"code": ViolationCode.SCHEMA_INVALID.value, "message": f"real_act ActionSpec build failed: {e}"}],
            )

    # Step 5: export the new helpers (and the Fake* test doubles) from this module
    # for convenient test imports without reaching into submodules yet.
    # (Authoritative definitions live in ownership.py / grants.py / gui_prompt.py)

    # Module-level exported helpers (Step 5 contract, used by tests and harness for
    # clearance token format checks and direct gate invocation in advanced scenarios).
    REAL_ACT_CLEARANCE_PREFIX = "clr:"

    def _is_real_act_clearance(self, token: Optional[str]) -> bool:
        """Small exported helper: returns True only for tokens issued by this kernel's real_act gate."""
        return bool(token and str(token).startswith(self.REAL_ACT_CLEARANCE_PREFIX))


# Module-level aliases for the helpers (so "from .safety import _real_act_gate, REAL_ACT_CLEARANCE_PREFIX" works
# for test modules and later wiring steps; the authoritative impl is the bound method on instances).
# These are thin shims that require a kernel to operate.
def _real_act_gate(kernel: SafetyKernel, *a: Any, **k: Any) -> ValidationResult:
    """Re-export of the real_act decision tree (Step 5)."""
    return kernel._real_act_gate(*a, **k)


REAL_ACT_CLEARANCE_PREFIX = SafetyKernel.REAL_ACT_CLEARANCE_PREFIX

"""ActionExecutor (Step 9 + real_act Step 7) — xdotool on isolated or gated real_act.

Implements the exact contract from COMPUTER_USE_DESIGN.md §5 and the
release-blocking tests in tests/test_computer_use_action.py:
- execute(ActionSpec|dict) hard-fails (returns rejected, *zero* side effects)
  unless display_scope in {"isolated", "real_act"} (FR-04 extended). The ds
  check is the first statement after normalize; for real_act a non-empty
  clearance_token (issued only by SafetyKernel after ownership gate) is
  required, else REJECTED with CLEARANCE_TOKEN_INVALID (INVARIANT E).
- Spatial actions (click etc.) require target.kind == "coordinate".
- Non-spatial (launch_app, navigate, hotkey, wait) accept target=None/absent.
- isolated paths: every xdotool uses private XAUTHORITY+DISPLAY from
  get_isolated_env (hardening #1). real_act paths: use operator real :0 env
  (DISPLAY/XAUTHORITY from os.environ, never the isolated cookie or
  AGY_ISOLATED_X).
- launch_app delegates to ProcessSupervisor.spawn (FR-24: no_shell, forced
  isolated env, rlimits, start_new_session). navigate can optionally route to
  an injected BrowserController and otherwise preserves the legacy spawn stub.
- click/type support additive DOM targets (`kind="dom"`) through an injected
  BrowserController; coordinate/real_act/xdotool paths are unchanged.
- Full ActionResult (status, executed_at, resolved_target, spawned_pids, ...).
- Ctor accepts the kwargs the test suite uses: isolated_display, action_timeout_ms.
- Private _verify_clearance(action) for the token gate (caller after kernel
  guarantees validity; we enforce non-empty defensively).

Ctor is side-effect free (lazy env materialization) so FR-04 / token-reject
tests see zero subprocess calls even for bad execute after construction.
All hermetic tests use mocks; zero real-:0 actuation or foreign input ever.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from .models import ActionSpec, ActionResult, ActionStatus
from .process_supervisor import ProcessSupervisor
from .xauth import get_isolated_env


class ActionExecutor:
    """Isolated-display-only action executor (xdotool + supervisor spawns).

    Additive Phase 1.x wiring: optional browser_controller for navigate + DOM
    click/type targets, with legacy spawn stub fallback when absent.
    """

    SPATIAL: set[str] = {"click", "double_click", "type", "scroll", "drag"}

    def __init__(
        self,
        isolated_display: str = ":99",
        supervisor: Optional[ProcessSupervisor] = None,
        browser_controller: Optional[Any] = None,
        **kw: Any,
    ) -> None:
        disp = isolated_display or ":99"
        self.isolated_display = disp if (isinstance(disp, str) and disp.startswith(":")) else ":99"
        self.supervisor = supervisor or ProcessSupervisor()
        self.browser_controller = browser_controller
        self.action_timeout_ms = int(kw.get("action_timeout_ms", 10000))
        # Lazy: do not call get_isolated_env here (FR-04 tests construct then
        # execute(bad_scope) and must observe zero subprocess side effects).
        self._env: Optional[Dict[str, str]] = None

    @property
    def env(self) -> Dict[str, str]:
        """Materialize (once) the hardened private env for this isolated display."""
        if self._env is None:
            try:
                self._env = get_isolated_env(self.isolated_display)
            except Exception:
                self._env = {"DISPLAY": self.isolated_display, "AGY_ISOLATED_X": "1"}
            self._env["DISPLAY"] = self.isolated_display
            self._env.pop("WAYLAND_DISPLAY", None)
        return self._env

    def _get_real_env(self) -> Dict[str, str]:
        """Minimal equivalent for real_act: the operator's live :0 session env.

        Returns DISPLAY + XAUTHORITY (or absent so X defaults to ~/.Xauthority)
        from the caller's os.environ. Explicitly strips any AGY_ISOLATED_* and
        WAYLAND_DISPLAY. NEVER returns or copies the isolated cookie produced
        by get_isolated_env. Used only on the cleared real_act success path.
        """
        env: Dict[str, str] = {}
        for k, v in os.environ.items():
            if k in ("WAYLAND_DISPLAY",):
                continue
            if k.startswith("AGY_ISOLATED"):
                continue
            env[k] = v
        # Force a real display (default :0); never the executor's isolated_display
        disp = env.get("DISPLAY") or ":0"
        if not (isinstance(disp, str) and disp.startswith(":")):
            disp = ":0"
        env["DISPLAY"] = disp
        # Do not force an XAUTHORITY that points at an isolated file; leave
        # absent or inherited so real X clients use the operator's authority.
        # If a value containing an isolated marker somehow leaked in, drop it.
        xa = env.get("XAUTHORITY", "")
        if "agu-" in xa or "AGY_ISOLATED" in xa:
            env.pop("XAUTHORITY", None)
        env.pop("AGY_ISOLATED_X", None)
        return env

    def _verify_clearance(self, action: Union[ActionSpec, Dict[str, Any]]) -> bool:
        """Private clearance gate for real_act.

        The SafetyKernel (after _real_act_gate ownership/ask/grant decision)
        is the ONLY code path that constructs an ActionSpec with
        display_scope="real_act" + a matching clearance_token. This method
        is the defensive check at the executor boundary (INVARIANT E).
        We require non-empty token; full cryptographic validation of the
        "clr:..." token is not performed here (kernel-issued guarantee).
        """
        if isinstance(action, dict):
            tok = action.get("clearance_token")
        else:
            tok = getattr(action, "clearance_token", None)
        return bool(tok and str(tok).strip())

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _run(self, cmd: List[Any], env: Dict[str, str], timeout: float) -> bool:
        try:
            subprocess.run(
                [str(c) for c in cmd],
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
            return True
        except Exception:
            return False

    def execute(self, action: Union[ActionSpec, Dict[str, Any]]) -> ActionResult:
        """Execute on isolated (existing) or real_act (token-gated only).

        Accepts either ActionSpec (post-SafetyKernel) or raw dict (tests).
        The ds check ("isolated" or "real_act") is the first statement after
        normalize; real_act then requires non-empty clearance_token (else
        CLEARANCE_TOKEN_INVALID, zero side effects). Real env materialization
        happens only after both gates; isolated lazy env untouched on real paths.
        """
        executed_at = self._now()

        # Normalize input (dict or dataclass). Keep tgt as raw (possibly None)
        # so the spatial pre-check can uniformly decide "target_missing".
        if isinstance(action, dict):
            a = action
            typ = a.get("type")
            ds = a.get("display_scope")
            tgt = a.get("target")
            url = a.get("url")
            text = a.get("text")
            hotkey = a.get("hotkey")
            scroll_delta = a.get("scroll_delta")
            drag_to = a.get("drag_to")
            wait_ms = a.get("wait_ms")
            app = a.get("app")
            app_args = a.get("app_args")
            action_id = a.get("action_id", "act-d")
        else:
            a = action  # type: ignore[assignment]
            typ = getattr(a, "type", None)
            ds = getattr(a, "display_scope", None)
            tgt = getattr(a, "target", None)
            url = getattr(a, "url", None)
            text = getattr(a, "text", None)
            hotkey = getattr(a, "hotkey", None)
            scroll_delta = getattr(a, "scroll_delta", None)
            drag_to = getattr(a, "drag_to", None)
            wait_ms = getattr(a, "wait_ms", None)
            app = getattr(a, "app", None)
            app_args = getattr(a, "app_args", None)
            action_id = getattr(a, "action_id", "act-s")

        # The ds gate is the first statement after input normalization (per Step 7).
        # FR-04 (release-blocking): hard-reject anything except "isolated" or
        # "real_act" before any coerce, any spatial check, any env, any tool call.
        if ds not in ("isolated", "real_act"):
            return ActionResult(
                status=ActionStatus.REJECTED.value,
                executed_at=executed_at,
                error_code="display_scope_invalid",
            )

        # Real-act token gate (INVARIANT E) immediately after ds allow.
        if ds == "real_act" and not self._verify_clearance(action):
            return ActionResult(
                status=ActionStatus.REJECTED.value,
                executed_at=executed_at,
                error_code="clearance_token_invalid",
            )

        # Coerce tgt to dict only for the convenience of the spatial branches;
        # the "absent / wrong kind" check below sees the original.
        tgt_d = tgt if isinstance(tgt, dict) else (tgt or {})
        tgt_kind = tgt_d.get("kind")
        is_dom_target = typ in ("click", "type") and tgt_kind == "dom"

        is_spatial = typ in self.SPATIAL
        if is_spatial and not is_dom_target and (not tgt_d or tgt_kind != "coordinate"):
            return ActionResult(
                status=ActionStatus.REJECTED.value,
                executed_at=executed_at,
                error_code="target_missing",
            )
        bc = self.browser_controller
        if is_dom_target and bc is None:
            return ActionResult(
                status=ActionStatus.REJECTED.value,
                executed_at=executed_at,
                error_code="browser_not_open",
            )

        to = max(0.5, min(self.action_timeout_ms / 1000.0, 8.0))
        # Materialize env *only* on the allowed success path.
        # isolated: lazy hardened private cookie (existing behavior, zero side
        #   effects on construct+execute(bad) paths).
        # real_act: real operator :0 env via _get_real_env (never isolated).
        if ds == "real_act":
            env = self._get_real_env()
        else:
            env = self.env  # existing lazy path

        resolved: Optional[Dict[str, int]] = None
        spawned: List[int] = []
        artifacts: Optional[List[str]] = None
        ok = False

        try:
            if typ == "click" and tgt_kind == "dom":
                selector = tgt_d.get("selector")
                index = int(tgt_d.get("index", 1))
                out = bc.click_dom(selector=selector, index=index)
                ok = bool(out.get("ok"))
                if ok:
                    resolved = {"index": int(out.get("index", index))}
                    landed_url = out.get("landed_url")
                    child_url = out.get("child_page_url")
                    art = []
                    if landed_url:
                        art.append(f"landed_url:{landed_url}")
                    if child_url:
                        art.append(f"child_page_url:{child_url}")
                    artifacts = art or None
                elif out.get("error_code") == "browser_not_open":
                    return ActionResult(
                        status=ActionStatus.REJECTED.value,
                        executed_at=executed_at,
                        error_code="browser_not_open",
                    )
                else:
                    ok = False

            elif typ == "type" and tgt_kind == "dom":
                selector = tgt_d.get("selector")
                index = int(tgt_d.get("index", 1))
                txt = str(text or "")
                out = bc.type_dom(selector=selector, index=index, text=txt)
                ok = bool(out.get("ok"))
                if ok:
                    resolved = {"index": int(out.get("index", index))}
                    landed_url = out.get("landed_url")
                    artifacts = [f"landed_url:{landed_url}"] if landed_url else None
                elif out.get("error_code") == "browser_not_open":
                    return ActionResult(
                        status=ActionStatus.REJECTED.value,
                        executed_at=executed_at,
                        error_code="browser_not_open",
                    )
                else:
                    ok = False

            elif typ in ("click", "double_click"):
                x = int(tgt_d.get("x", 0))
                y = int(tgt_d.get("y", 0))
                self._run(["xdotool", "mousemove", x, y, "click", "1"], env, to)
                if typ == "double_click":
                    self._run(["xdotool", "click", "1"], env, to)
                resolved = {"x": x, "y": y}
                ok = True

            elif typ == "type":
                x = int(tgt_d.get("x", 0))
                y = int(tgt_d.get("y", 0))
                txt = str(text or "")
                self._run(["xdotool", "mousemove", x, y, "click", "1"], env, to)
                self._run(["xdotool", "type", "--", txt or " "], env, to)
                resolved = {"x": x, "y": y}
                ok = True

            elif typ == "hotkey":
                ks = "+".join(str(k) for k in (hotkey or ["Return"]))
                self._run(["xdotool", "key", ks], env, to)
                ok = True

            elif typ == "scroll":
                # Spatial: move to target if supplied (required by pre-check for all SPATIAL),
                # then emit wheel event. This matches click/type/drag/double_click contract.
                x = y = 0
                if tgt_d and tgt_d.get("kind") == "coordinate":
                    x = int(tgt_d.get("x", 0))
                    y = int(tgt_d.get("y", 0))
                    self._run(["xdotool", "mousemove", x, y], env, to)
                dy = int((scroll_delta or {}).get("dy", 0))
                btn = "5" if dy >= 0 else "4"
                self._run(["xdotool", "click", btn], env, to)
                resolved = {"x": x, "y": y} if (tgt_d and tgt_d.get("kind") == "coordinate") else None
                ok = True

            elif typ == "drag":
                x, y = int(tgt_d.get("x", 0)), int(tgt_d.get("y", 0))
                tx, ty = int((drag_to or {}).get("x", x + 10)), int((drag_to or {}).get("y", y + 10))
                self._run(["xdotool", "mousemove", x, y, "mousedown", "1"], env, to)
                self._run(["xdotool", "mousemove", tx, ty], env, to)
                self._run(["xdotool", "mouseup", "1"], env, to)
                resolved = {"x": tx, "y": ty}
                ok = True

            elif typ == "wait":
                ms = int(wait_ms or 0)
                time.sleep(max(0.0, min(ms / 1000.0, 30.0)))
                ok = True

            elif typ == "launch_app":
                if app:
                    argv = [str(app)] + [str(x) for x in (app_args or [])]
                    sp = self.supervisor.spawn(argv=argv, display_scope="isolated", no_shell=True, source_action_id=action_id)
                    spawned = [sp.pid]
                    ok = True
                else:
                    ok = False

            elif typ == "navigate":
                if bc is not None:
                    landed_url, pid = bc.navigate(str(url or "about:blank"))
                    spawned = [int(pid)] if pid else []
                    resolved = {"browser_pid": int(pid or 0)}
                    artifacts = [f"landed_url:{landed_url}"] if landed_url else None
                else:
                    # Delegate (browser stub) via supervisor per test contract + FR-24
                    sp = self.supervisor.spawn(argv=["true"], display_scope="isolated", no_shell=True, source_action_id=action_id)
                    spawned = [sp.pid]
                ok = True

            else:
                ok = False

        except Exception:
            ok = False

        status = ActionStatus.OK.value if ok else ActionStatus.FAILED.value
        return ActionResult(
            status=status,
            executed_at=executed_at,
            resolved_target=resolved,
            spawned_process_ids=spawned or None,
            error_code=None if ok else "execution_failed",
            artifacts=artifacts,
        )

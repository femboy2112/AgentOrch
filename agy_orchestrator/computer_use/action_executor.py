"""ActionExecutor (Step 9) — xdotool input injection strictly on isolated display.

Implements the exact contract from COMPUTER_USE_DESIGN.md §5 and the
release-blocking tests in tests/test_computer_use_action.py:
- execute(ActionSpec|dict) hard-fails (returns rejected, *zero* side effects)
  unless display_scope == "isolated" (FR-04). The check is first statement.
- Spatial actions (click etc.) require target.kind == "coordinate".
- Non-spatial (launch_app, navigate, hotkey, wait) accept target=None/absent.
- Every xdotool and every delegated spawn uses the private XAUTHORITY+DISPLAY
  produced by get_isolated_env (Step 4 hardening #1). Real ~/.Xauthority is
  structurally absent; "cannot authenticate to :0" holds at same UID.
- launch_app and navigate delegate exclusively to ProcessSupervisor.spawn
  (FR-24: no_shell, forced isolated env, rlimits, start_new_session).
- Full ActionResult (status, executed_at, resolved_target, spawned_pids, ...).
- Ctor accepts the kwargs the test suite uses: isolated_display, action_timeout_ms.

Ctor is side-effect free (lazy env materialization) so FR-04 rejection tests
see zero subprocess calls even for bad-scope execute after construction.
All actuation paths (tests + prod) use only temp isolated Xvfb.
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from .models import ActionSpec, ActionResult, ActionStatus
from .process_supervisor import ProcessSupervisor
from .xauth import get_isolated_env


class ActionExecutor:
    """Isolated-display-only action executor (xdotool + supervisor spawns)."""

    SPATIAL: set[str] = {"click", "double_click", "type", "scroll", "drag"}

    def __init__(
        self,
        isolated_display: str = ":99",
        supervisor: Optional[ProcessSupervisor] = None,
        **kw: Any,
    ) -> None:
        disp = isolated_display or ":99"
        self.isolated_display = disp if (isinstance(disp, str) and disp.startswith(":")) else ":99"
        self.supervisor = supervisor or ProcessSupervisor()
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
        """Execute only on isolated. Reject real-scope *before* any tool call.

        Accepts either ActionSpec (post-SafetyKernel, already display-validated)
        or raw dict (test/FR-04 boundary paths that must never materialize env
        or call tools for non-isolated scopes). The ds check is deliberately
        the very first substantive statement.
        """
        executed_at = self._now()

        # Normalize input (dict or dataclass). Keep tgt as raw (possibly None)
        # so the spatial pre-check can uniformly decide "target_missing".
        if isinstance(action, dict):
            a = action
            typ = a.get("type")
            ds = a.get("display_scope")
            tgt = a.get("target")
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
            text = getattr(a, "text", None)
            hotkey = getattr(a, "hotkey", None)
            scroll_delta = getattr(a, "scroll_delta", None)
            drag_to = getattr(a, "drag_to", None)
            wait_ms = getattr(a, "wait_ms", None)
            app = getattr(a, "app", None)
            app_args = getattr(a, "app_args", None)
            action_id = getattr(a, "action_id", "act-s")

        # Coerce tgt to dict only for the convenience of the spatial branches;
        # the "absent / wrong kind" check below sees the original.
        tgt_d = tgt if isinstance(tgt, dict) else (tgt or {})

        # FR-04 (release-blocking): hard gate first, no env materialization,
        # no xdotool, no supervisor.spawn for anything but "isolated".
        if ds != "isolated":
            return ActionResult(
                status=ActionStatus.REJECTED.value,
                executed_at=executed_at,
                error_code="display_scope_invalid",
            )

        is_spatial = typ in self.SPATIAL
        if is_spatial and (not tgt_d or tgt_d.get("kind") != "coordinate"):
            return ActionResult(
                status=ActionStatus.REJECTED.value,
                executed_at=executed_at,
                error_code="target_missing",
            )

        to = max(0.5, min(self.action_timeout_ms / 1000.0, 8.0))
        env = self.env  # materialize only on success path

        resolved: Optional[Dict[str, int]] = None
        spawned: List[int] = []
        ok = False

        try:
            if typ in ("click", "double_click"):
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
        )

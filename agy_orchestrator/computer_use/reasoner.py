"""ReasonerBridge (Step 10) — exact CLI contract + safety behaviors (FR-14,16,20,21,25).

Implements the authoritative prompt envelope, ACTION_INTENT_JSON_V1 parse, bounded repair,
priority routing (high=claude->codex; normal=codex->claude), reasoning timeout with owned
subprocess kill, auth-required marker detection + kill + fallback, and guaranteed "no
ActionIntent from bad output".

Emits correct WorkerEvents to optional audit_sink. Accepts pre-built agents or builds
direct CodexAgent/ClaudeAgent. Supports the test harness call styles (positional run_id,
claude_agent= / codex_agent=, _kill_agent_process hook).

No X11, no real displays, no GUI, pure mock-friendly unit-testable logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import (
    ActionIntent,
    EngineHealth,
    EngineStatus,
    ReasoningInput,
    WorkerEvent,
    WorkerEventType,
    as_json,
    from_dict,
)

logger = logging.getLogger(__name__)


class ReasonerError(RuntimeError):
    """Exhausted all routes/repairs without a valid ActionIntent (safety: never execute)."""
    pass


# Auth-state markers per spec (FR-25): scan raw output + stderr for interactive / OAuth / TTY login prompts.
# Specific auth/interactive-login phrases only. Bare tokens like "browser",
# "interactive", "expired", and "tty" were removed: they collide with the
# reasoner's own action rationales (e.g. "use the agent-owned browser to
# navigate"), which would false-positive as an auth-required state and kill a
# perfectly valid reasoning turn. Real interactive-login prompts contain these
# fuller phrases. (Deeper hardening would scan only the CLI's stderr, never the
# model's structured answer, for these markers.)
AUTH_MARKERS: tuple[str, ...] = (
    "oauth",
    "sign in",
    "log in",
    "authentication",
    "login required",
    "re-authenticate",
    "please authenticate",
    "session expired",
    "browser to re-auth",
)

REPAIR_NUDGE = (
    "PREVIOUS OUTPUT WAS INVALID OR DID NOT CONTAIN A SINGLE WELL-FORMED action_intent. "
    "Emit EXACTLY ONE fenced ```json block containing the key \"action_intent\" matching ACTION_INTENT_JSON_V1. "
    "No other JSON. No prose outside the block."
)


def _is_auth(text: str) -> bool:
    """Detect auth-required / interactive-login states in CLI stdout or stderr (FR-25)."""
    t = (text or "").lower()
    return any(m in t for m in AUTH_MARKERS)


def _extract(raw: str) -> Optional[Dict[str, Any]]:
    """Extract the first action_intent dict from fenced json blocks or bare top-level json.
    Per spec: first fenced json block that contains the top-level key "action_intent".
    """
    if not raw:
        return None
    # Prefer fenced blocks (```json or ```)
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE):
        block = m.group(1).strip()
        try:
            j = json.loads(block)
            if isinstance(j, dict) and "action_intent" in j:
                ai = j["action_intent"]
                return ai if isinstance(ai, dict) else None
        except Exception:
            continue
    # Bare JSON fallback (whole output is a single object)
    try:
        j = json.loads(raw.strip())
        if isinstance(j, dict) and "action_intent" in j:
            ai = j["action_intent"]
            return ai if isinstance(ai, dict) else None
    except Exception:
        pass
    return None


# =============================================================================
# Precise prompt envelope builder (spec §6 Reasoner I/O formatting contract)
# =============================================================================

REASONER_HEADER = """You are a precise, safety-first GUI computer-use reasoning engine for AgentOrch.

MISSION
Analyze the structured perception snapshots (windows, AT-SPI elements, geometry, OCR text blocks, optional DOM) from the current session and output exactly one next ActionIntent that advances the operator objective while obeying all hard constraints.

navigate opens an agent-owned CDP browser (registered via ProcessSupervisor); use target {kind:'dom', selector:'...', index:N (1-based)} on click/type to actuate DOM elements and auto-follow new tabs (ctx.pages[-1]). The browser is always the agent's child — no real_act prompt required.

HARD SAFETY CONSTRAINTS (VIOLATIONS ARE NEVER ACCEPTABLE)
- action.display_scope MUST be the literal string "isolated". Never target real :0 or any other display.
- For spatial actions (click, double_click, type, scroll, drag) a target is REQUIRED: either a valid element handle_id from the current snapshots or explicit {kind:"coordinate", x, y, ...}.
- Non-spatial actions (launch_app, navigate, hotkey, wait) have their own required payload fields; target is optional.
- launch_app "app" must be an allowlisted executable identity (e.g. "firefox"), never a raw shell command or metacharacter string.
- Destructive or irreversible actions (rm, delete, format, poweroff, close critical windows, etc.) MUST use risk_level="irreversible" and requires_confirmation=true.
- You may only act inside the displays and process trees owned by this run.
- If the snapshots contain no actionable elements for the objective, emit a small wait action rather than hallucinating coordinates.

"""

REASONER_FOOTER = """
OUTPUT CONTRACT — ACTION_INTENT_JSON_V1 (MANDATORY)
Respond with EXACTLY ONE fenced JSON code block using one of:
```json
{ "action_intent": { ... } }
```
or a bare top-level object containing the key. No other top-level keys at the JSON root for execution purposes.

The "action_intent" value MUST contain (at minimum):
- intent_id: string (unique for this decision turn)
- snapshot_id: string (must reference a snapshot_id present in the REASONING_INPUT)
- action: object with:
    - type: one of click | double_click | type | hotkey | scroll | drag | wait | launch_app | navigate
    - display_scope: "isolated"   (literal, required)
    - target?: element handle or coordinate (required for all spatial types)
    - text?, hotkey?, scroll_delta?, drag_to?, wait_ms?, url?, app?, app_args? as required by the action type
- rationale: short string explaining the choice
- risk_level: "low" | "medium" | "high" | "irreversible"
- requires_confirmation: boolean
- confidence: number in [0.0, 1.0]

Extra free text outside the JSON block is retained only in the audit log and is ignored for execution.

If you cannot produce a safe, schema-valid ActionIntent, emit a low-confidence wait action targeting the isolated display rather than guessing or violating constraints.

Begin.
"""


def build_reasoner_prompt_envelope(ri: ReasoningInput) -> str:
    """Return the complete fixed prompt envelope for a reasoning turn.

    Header (safety + task) + body (deterministic fenced ReasoningInput JSON) + footer (strict output contract).
    Uses as_json for stable key ordering + compact separators (no comments, UTF-8 safe).
    This is the exact text handed to the claude/codex CLI worker for this decide() call.
    """
    # Body: the serialized ReasoningInput under a clear label
    body_json = as_json(ri)
    body = f"REASONING_INPUT:\n```json\n{body_json}\n```\n"
    return REASONER_HEADER + body + REASONER_FOOTER


class ReasonerBridge:
    """Step 10 ReasonerBridge — exact CLI contract + all required safety behaviors.

    - Builds the authoritative fixed prompt envelope (header + stable ReasoningInput JSON + footer).
    - Priority routing (FR-14/21): high -> claude then codex; normal/absent -> codex then claude.
    - Bounded repair (1) on same engine for parse/schema errors, then fallback.
    - Enforces reasoning_timeout_ms via asyncio.wait_for; on timeout: kill owned subprocess,
      emit reasoner.timeout, fallback (FR-20).
    - Auth-required detection on stdout+stderr (FR-25): kill, emit reasoner.auth_required, fallback.
    - Never returns ActionIntent (or executes) from unparseable, schema-invalid, auth, or timeout paths.
    - Emits reasoner.* events to optional AuditEventSink (FR-13).
    - Flexible construction for tests (mocks) and prod (real CodexAgent/ClaudeAgent instances
      or auto-build via direct agent classes when none supplied).
    """

    # Default persona prompt used only when we auto-instantiate agents internally.
    # In real adapter usage the caller typically supplies pre-built agents (with their own
    # base prompt) and the per-turn envelope is delivered as the piped_input/context.
    DEFAULT_PERSONA = "You are the GUI computer-use reasoner for an isolated-display AgentOrch worker."

    def __init__(
        self,
        run_id: str = "r",
        audit_sink: Any = None,
        *,
        claude: Any = None,
        codex: Any = None,
        claude_agent: Any = None,
        codex_agent: Any = None,
        max_repair: int = 1,
        auto_build_agents: bool = False,
        **kw: Any,
    ) -> None:
        # Flexible test / harness call patterns (positional run_id, alias kwargs, etc.)
        if not isinstance(run_id, str) or not run_id:
            run_id = "r"
        if audit_sink is None and kw.get("audit_sink"):
            audit_sink = kw.get("audit_sink")

        self.run_id = str(run_id)
        self.audit = audit_sink
        self.max_repair = max(0, int(max_repair))
        self._last_error: Optional[str] = None

        # Accept either kwarg style (claude= / codex= or the _agent aliases used in tests)
        self.claude = claude or claude_agent or kw.get("claude")
        self.codex = codex or codex_agent or kw.get("codex")

        self._auto_build = bool(auto_build_agents)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if not self.audit:
            return
        try:
            ev = WorkerEvent(
                ts=datetime.now(timezone.utc).isoformat(),
                run_id=self.run_id,
                event_type=event_type,
                payload=payload or {},
            )
            self.audit.emit(ev)
        except Exception:
            # Audit must never break the reasoner control flow (reliability NFR)
            pass

    def _route(self, task_priority: Optional[str]) -> List[str]:
        """FR-14 + FR-21 routing matrix."""
        if (task_priority or "").lower() == "high":
            return ["claude", "codex"]
        return ["codex", "claude"]

    def _ensure_agent(self, name: str) -> Optional[Any]:
        """Return the requested engine, auto-instantiating a plain agent if allowed and missing."""
        if name == "claude":
            if self.claude is None and self._auto_build:
                try:
                    from agy_orchestrator.core.agents.claude_agent import ClaudeAgent

                    self.claude = ClaudeAgent(prompt=self.DEFAULT_PERSONA)
                except Exception:
                    self.claude = None
            return self.claude
        else:
            if self.codex is None and self._auto_build:
                try:
                    from agy_orchestrator.core.agents.codex_agent import CodexAgent

                    self.codex = CodexAgent(prompt=self.DEFAULT_PERSONA)
                except Exception:
                    self.codex = None
            return self.codex

    def _kill_agent_process(self, agent: Any = None) -> None:
        """Best-effort kill of the agent's tracked _current_process (FR-20 / FR-25).
        Safe to call from sync decide() context.
        """
        tgt = agent or self.claude or self.codex
        if tgt is None or not hasattr(tgt, "_kill_current"):
            return
        try:
            coro = tgt._kill_current()
            if asyncio.iscoroutine(coro):
                # decide() is sync; run the kill in a fresh loop when no loop is running.
                try:
                    asyncio.get_running_loop()
                    # Loop is running — best effort: create task (fire-and-forget)
                    asyncio.create_task(coro)  # type: ignore
                except RuntimeError:
                    # No running loop — safe to asyncio.run
                    asyncio.run(coro)
        except Exception:
            pass

    def _call_agent(self, agent: Any, prompt: str, timeout_ms: int) -> str:
        """Call agent.run_async (preferred) or .run under a hard timeout.
        Returns the raw text output. Raises on timeout or agent failure.
        The caller is responsible for killing on timeout paths.
        """
        timeout_s = max(0.5, timeout_ms / 1000.0)

        # Prefer the async path (all real agents + our test mocks implement it)
        if hasattr(agent, "run_async"):
            async def _invoke() -> str:
                # Some agents accept the per-turn text as positional / piped_input
                return await agent.run_async(prompt)  # type: ignore

            def _run_blocking() -> str:
                return asyncio.run(asyncio.wait_for(_invoke(), timeout=timeout_s))

            try:
                # decide() is synchronous but is driven on the adapter's perceive->
                # reason->act loop thread, which ALSO owns the BrowserController's
                # Playwright SYNC API. Two hazards if we run asyncio.run() on that
                # thread directly:
                #   (a) if the harness's outer loop is running on it, asyncio.run()
                #       refuses to nest and raises immediately; and
                #   (b) even with no outer loop, creating+tearing down an event loop
                #       on the Playwright-owner thread corrupts Playwright's own loop
                #       state, so the NEXT page.evaluate() (next-step perception) hangs
                #       forever. (Playwright sync leaves a live loop pinned to the thread.)
                # Both are avoided by ALWAYS running the agent coroutine on a dedicated
                # worker thread (its own fresh loop), never inline. The owner thread's
                # asyncio/Playwright state is then never disturbed.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(_run_blocking).result()
            except asyncio.TimeoutError:
                raise

        # Sync fallback (rare in practice; real agents expose run() wrapper)
        if hasattr(agent, "run"):
            # No easy way to timeout a blocking sync call without threads.
            # For unit tests that supply sync callables we just invoke (they are fast mocks).
            return str(agent.run(prompt))  # type: ignore

        # Last resort — the agent object itself may be a stub that returns text when called
        return str(agent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, ri: Any, timeout_ms: int = 45000) -> ActionIntent:
        """Main entry point. Returns a validated ActionIntent or raises ReasonerError.

        Guarantees (per FR-16/20/25 and hardening):
        - No ActionIntent is ever returned from unparseable output, schema errors,
          auth-required responses, or timeout paths.
        - Owned subprocess is terminated on timeout / auth before falling back.
        """
        if isinstance(ri, dict):
            ri = from_dict(ReasoningInput, ri)
        if not isinstance(ri, ReasoningInput):
            # Last-ditch: try to coerce via to_dict roundtrip (defensive)
            try:
                ri = from_dict(ReasoningInput, ri.to_dict() if hasattr(ri, "to_dict") else ri)
            except Exception:
                ri = ri  # will likely fail later; let it surface as parse error path

        engines = self._route(getattr(ri, "task_priority", None) or getattr(ri, "task_priority", None))
        last_err: Optional[str] = None

        for eng in engines:
            agent = self._ensure_agent(eng)
            if agent is None:
                last_err = f"no-agent:{eng}"
                continue

            for attempt in range(self.max_repair + 1):
                envelope = build_reasoner_prompt_envelope(ri)
                if attempt > 0:
                    envelope = envelope + "\n\n" + REPAIR_NUDGE

                raw = ""
                try:
                    raw = self._call_agent(agent, envelope, timeout_ms)
                except asyncio.TimeoutError:
                    self._emit(WorkerEventType.REASONER_TIMEOUT.value, {"engine": eng, "attempt": attempt})
                    self._kill_agent_process(agent)
                    last_err = f"timeout:{eng}"
                    break  # do not attempt repair on this engine; fall through to next engine
                except RuntimeError as e:
                    msg = str(e)
                    if "timeout" in msg.lower() or "exceeded" in msg.lower() or "watchdog" in msg.lower():
                        self._emit(WorkerEventType.REASONER_TIMEOUT.value, {"engine": eng, "attempt": attempt})
                        self._kill_agent_process(agent)
                        last_err = f"timeout:{eng}"
                        break
                    # Transient error — treat as empty for parse purposes
                    raw = ""
                except Exception:
                    raw = ""

                # Auth detection (FR-25) — must happen before any parse attempt
                stderr = getattr(agent, "stderr", "") or ""
                combined = f"{raw}\n{stderr}"
                if _is_auth(combined):
                    self._emit(WorkerEventType.REASONER_AUTH_REQUIRED.value, {"engine": eng, "attempt": attempt})
                    self._kill_agent_process(agent)
                    last_err = f"auth:{eng}"
                    break  # no repair on auth failure; fall to next engine

                ai = _extract(raw)
                if ai is None:
                    self._emit(WorkerEventType.REASONER_PARSE_ERROR.value, {"engine": eng, "attempt": attempt})
                    last_err = f"parse:{eng}"
                    if attempt < self.max_repair:
                        continue
                    break  # repair budget exhausted on this engine

                # Schema validation via dataclass (runs __post_init__ for RiskLevel, confidence range, etc.)
                try:
                    intent = ActionIntent(
                        intent_id=ai.get("intent_id", f"i-{eng}-{attempt}"),
                        snapshot_id=ai.get("snapshot_id", "s1"),
                        action=ai.get("action", {}),
                        rationale=ai.get("rationale", ""),
                        risk_level=ai.get("risk_level", "low"),
                        requires_confirmation=bool(ai.get("requires_confirmation", False)),
                        confirmation_token=ai.get("confirmation_token"),
                        confidence=float(ai.get("confidence", 0.5)),
                    )
                except Exception as ve:
                    self._emit(
                        WorkerEventType.REASONER_PARSE_ERROR.value,
                        {"engine": eng, "attempt": attempt, "schema_error": str(ve)[:120]},
                    )
                    last_err = f"schema:{eng}"
                    if attempt < self.max_repair:
                        continue
                    break

                # Success path — emit and return (SafetyKernel will do final display_scope / risk gating)
                self._emit(WorkerEventType.REASONER_INTENT.value, {"engine": eng, "intent_id": intent.intent_id})
                self._last_error = None
                return intent

        # All engines + repairs exhausted without producing a valid ActionIntent
        self._last_error = last_err or "exhausted"
        self._emit(WorkerEventType.REASONER_PARSE_ERROR.value, {"final": True, "last_error": self._last_error})
        raise ReasonerError(
            f"ReasonerBridge exhausted all routes/repairs with no valid ActionIntent "
            f"(last_error={self._last_error}). Safety invariant: no execution from bad output."
        )

    def engine_status(self) -> EngineHealth:
        """Return current engine health + documented routing policy (used by adapter + tests)."""
        cl = EngineStatus.READY.value if self.claude is not None else EngineStatus.DEGRADED.value
        co = EngineStatus.READY.value if self.codex is not None else EngineStatus.DEGRADED.value
        return EngineHealth(
            claude=cl,
            codex=co,
            last_error=self._last_error,
            # The defaults on the dataclass already document the policy; we just surface them explicitly.
        )


__all__ = ["ReasonerBridge", "ReasonerError", "AUTH_MARKERS", "build_reasoner_prompt_envelope"]

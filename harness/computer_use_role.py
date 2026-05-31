"""Minimal wiring for 'computer-use' as a standard harness-selectable worker token (Step 12).

Exposes a thin shim that satisfies the AgentInstance surface used by --mode direct
(and is safe no-op for other workflow shapes). Delegates the actual GUI loop to
ComputerUseWorkerAdapter using harness run context (run_id, events.jsonl, out_dir cwd)
and optional computer_use_config for mode/task_priority/budgets/real_gui_policy/ask_mode/browser_engine/browser_display.

Does not alter any LLM worker code paths. All actuation stays on isolated Xvfb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

# Lazy to avoid import side-effects when harness is used without computer-use.
_ComputerUseWorkerAdapter = None
_RunRequest = None
_AuditEventSink = None


def _get_cu_imports():
    global _ComputerUseWorkerAdapter, _RunRequest, _AuditEventSink
    if _ComputerUseWorkerAdapter is None:
        from agy_orchestrator.computer_use.adapter import ComputerUseWorkerAdapter
        from agy_orchestrator.computer_use.audit import AuditEventSink
        from agy_orchestrator.computer_use.models import RunRequest
        _ComputerUseWorkerAdapter = ComputerUseWorkerAdapter
        _RunRequest = RunRequest
        _AuditEventSink = AuditEventSink
    return _ComputerUseWorkerAdapter, _RunRequest, _AuditEventSink


def _extract_objective(prompt: str) -> str:
    """Best-effort: pull the original instruction when harness wrapped it."""
    if "## Instruction\n" in prompt:
        tail = prompt.split("## Instruction\n", 1)[1]
        if "\n## " in tail:
            tail = tail.split("\n## ", 1)[0]
        return tail.strip()
    return prompt.strip()


class ComputerUseShim:
    """Duck-typed shim for roles.build_role_agent when token=='computer-use'.

    Supports the post_construct_hook injections (event_callback, cwd, and our
    _harness_* for run linkage) and the direct-mode run_async() call.
    """

    def __init__(self, prompt: str = "", model: Optional[str] = None, effort: Optional[str] = None, **kwargs: Any) -> None:
        self.prompt = prompt
        self.model = model or "computer-use"
        self.effort = effort or "n/a"
        self.stdout = ""
        self.stderr = ""
        self.returncode: Optional[int] = None
        self.cwd: Optional[str] = kwargs.get("cwd")
        self.event_callback: Optional[Any] = kwargs.get("event_callback")
        # Harness run linkage (set by post_construct_hook or direct construction for run_async)
        self._harness_run_id: Optional[str] = kwargs.get("_harness_run_id") or kwargs.get("harness_run_id")
        self._harness_events_path: Optional[str] = kwargs.get("_harness_events_path") or kwargs.get("harness_events_path")
        self.computer_use_config: Optional[Dict[str, Any]] = kwargs.get("computer_use_config")
        # Also accept flat for convenience (CLI / dispatch forwarding)
        for k in (
            "computer_use_mode",
            "computer_use_task_priority",
            "task_priority",
            "budgets",
            "real_gui_policy",
            "ask_mode",
            "browser_engine",
            "browser_display",
        ):
            if k in kwargs and self.computer_use_config is None:
                self.computer_use_config = {}
            if k in kwargs:
                self.computer_use_config = self.computer_use_config or {}
                self.computer_use_config[k.replace("computer_use_", "")] = kwargs[k]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        Adapter, RunRequest, AuditEventSink = _get_cu_imports()

        objective = _extract_objective(self.prompt)
        cfg = self.computer_use_config or {}

        run_id = getattr(self, "_harness_run_id", None)
        events_path = getattr(self, "_harness_events_path", None)

        # Build sink factory so cu shares the exact harness-created events.jsonl (FR-13)
        def _sink_factory(rid: str):
            if events_path:
                return AuditEventSink(run_id=rid, events_path=Path(events_path))
            return AuditEventSink(run_id=rid)

        rid = run_id or f"cu-{int(__import__('time').time()*1000)}"

        # Wire a real reasoner for the harness path. The adapter only self-builds
        # a bridge with auto_build_agents=False (the "outer layer supplies agents"
        # contract); without this the harness loop has no LLM wired, so every
        # decide() returns no-agent:* and the session ends after one empty step.
        # Supply a bridge that lazily builds the codex->claude reasoner, sharing
        # the single harness sink so its reasoner events land in events.jsonl.
        from agy_orchestrator.computer_use.reasoner import ReasonerBridge

        _shared_sink = _sink_factory(rid)
        reasoner = ReasonerBridge(run_id=rid, audit_sink=_shared_sink, auto_build_agents=True)
        adapter = Adapter(audit_sink_factory=lambda _rid: _shared_sink, reasoner=reasoner)

        req: Dict[str, Any] = {
            "run_id": rid,
            "objective": objective or "computer-use objective from harness",
        }
        if cfg.get("mode"):
            req["mode"] = cfg["mode"]
        if cfg.get("task_priority"):
            req["task_priority"] = cfg["task_priority"]
        if cfg.get("budgets"):
            req["budgets"] = cfg["budgets"]
        # Forward real-gui + browser keys from computer_use_config into req (RunRequest/adapter path).
        if cfg.get("real_gui_policy") is not None:
            req["real_gui_policy"] = cfg["real_gui_policy"]
        if cfg.get("ask_mode") is not None:
            req["ask_mode"] = cfg["ask_mode"]
        if cfg.get("browser_engine") is not None:
            req["browser_engine"] = cfg["browser_engine"]
        if cfg.get("browser_display") is not None:
            req["browser_display"] = cfg["browser_display"]

        # Confirmation tokens are submitted via adapter.submit_confirmation() by caller
        # when high-risk actions are gated; dispatch itself is fire-and-observe for cu.
        #
        # adapter.start() is fully synchronous and runs the entire perceive->reason->
        # act loop inline. run_async() is awaited under an asyncio event loop, so this
        # thread has a *running* loop. The owned BrowserController drives Playwright's
        # SYNC API, which hard-refuses to launch on a thread with a running asyncio loop
        # ("using Playwright Sync API inside the asyncio loop") -> every navigate fails.
        # Offload the whole loop to a worker thread (no asyncio loop there): Playwright
        # sync works, and the reasoner's own asyncio.run (reasoner._call_agent) takes its
        # inline branch since that worker thread has no running loop.
        try:
            try:
                loop = __import__("asyncio").get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                handle = await loop.run_in_executor(None, adapter.start, req)
            else:
                handle = adapter.start(req)
            self.stdout = f"computer-use completed run_id={handle.run_id} status={handle.status}"
            self.returncode = 0 if handle.status in ("completed", "ok") else 1
        except Exception as exc:  # never destabilize harness caller
            self.stderr = f"{type(exc).__name__}: {exc}"
            self.returncode = 1
            self.stdout = "computer-use errored (see stderr)"
        return self.stdout

    # Minimal surface for describe_chain / logging paths
    def __repr__(self) -> str:  # pragma: no cover
        return f"ComputerUseShim(model={self.model!r})"

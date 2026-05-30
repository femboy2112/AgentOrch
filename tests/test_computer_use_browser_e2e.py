"""Autonomous browser/DOM e2e dogfood (env-gated, slow, real browser only).

This test is intentionally the only browser test that may launch a real browser.
It is skipped unless AGY_BROWSER_E2E=1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import pytest

from agy_orchestrator.computer_use.action_executor import ActionExecutor
from agy_orchestrator.computer_use.adapter import ComputerUseWorkerAdapter
from agy_orchestrator.computer_use.audit import AuditEventSink
from agy_orchestrator.computer_use.models import ActionIntent, RunRequest


QUERY = "programmatic browser actuation test"
RANK = 3
RESULT_SELECTOR = "li.b_algo h2 a"


def _extract_destination_token(href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    cand = href.strip()
    for _ in range(3):
        p = urlparse(cand)
        if p.scheme in ("http", "https") and p.netloc:
            host = p.netloc.lower()
            if host:
                return host
        qs = parse_qs(p.query or "")
        next_vals = [v for vals in qs.values() for v in vals if isinstance(v, str) and v.strip()]
        nested = None
        for v in next_vals:
            vv = unquote(v.strip())
            pv = urlparse(vv)
            if pv.scheme in ("http", "https") and pv.netloc:
                nested = vv
                break
        if not nested:
            break
        cand = nested
    return None


def _is_bot_blocked(page: Any) -> bool:
    url = (getattr(page, "url", "") or "").lower()
    if "/sorry/" in url:
        return True
    try:
        body = (page.inner_text("body", timeout=2000) or "").lower()
    except Exception:
        body = ""
    needles = (
        "unusual traffic",
        "are not a robot",
        "are you a robot",
        "verify you are human",
        "captcha",
        "access denied",
    )
    return any(n in body for n in needles)


@dataclass
class _E2EState:
    phase: int = 0
    saw_dom_in_perception: bool = False
    expected_href: Optional[str] = None
    expected_host: Optional[str] = None
    final_url: Optional[str] = None
    browser_pid: Optional[int] = None
    owned_seen: bool = False
    bot_blocked: bool = False
    bot_block_reason: Optional[str] = None
    saw_new_tab: bool = False


class _AutonomousE2EReasoner:
    """Tiny deterministic reasoner that still consumes perception and emits DOM intent."""

    def __init__(self, adapter: ComputerUseWorkerAdapter, state: _E2EState) -> None:
        self.adapter = adapter
        self.state = state

    def engine_status(self) -> Any:
        return SimpleNamespace(claude="ready", codex="ready")

    def _mk_intent(self, *, run_id: str, typ: str, action: Dict[str, Any], rationale: str) -> ActionIntent:
        return ActionIntent(
            intent_id=f"e2e-{self.state.phase}-{typ}",
            snapshot_id=f"e2e-s-{self.state.phase}",
            action=action,
            rationale=rationale,
            risk_level="low",
            requires_confirmation=False,
            confidence=1.0,
        )

    def decide(self, ri: Any, timeout_ms: int = 45000) -> ActionIntent:
        if not isinstance(ri, dict):
            ri = dict(getattr(ri, "__dict__", {}))
        run_id = str(ri.get("run_id") or "e2e")
        snaps = ri.get("snapshots") or {}
        iso = snaps.get("isolated") or {}
        elems = iso.get("elements") if isinstance(iso, dict) else []
        if isinstance(elems, list) and any(isinstance(e, dict) and e.get("source") == "dom" for e in elems):
            self.state.saw_dom_in_perception = True

        sess = self.adapter._controller.get_session(run_id)  # test-only introspection
        if sess is None or getattr(sess, "browser_controller", None) is None:
            raise RuntimeError("e2e reasoner: browser controller session missing")
        bc = sess.browser_controller

        if self.state.phase == 0:
            self.state.phase = 1
            return self._mk_intent(
                run_id=run_id,
                typ="navigate",
                action={
                    "type": "navigate",
                    "display_scope": "isolated",
                    "url": f"https://www.bing.com/search?q={quote_plus(QUERY)}",
                },
                rationale="open bing search results",
            )

        # Phase 1: verify ownership + inspect destination before DOM click.
        if self.state.phase == 1:
            if type(bc).__name__ != "BrowserController":
                raise RuntimeError(f"e2e requires real BrowserController, got {type(bc).__name__}")
            pid = int(getattr(bc, "browser_pid", 0) or 0)
            self.state.browser_pid = pid
            self.state.owned_seen = bool(pid > 0 and sess.supervisor.is_owned(pid))
            context = getattr(bc, "_context", None)
            page = context.pages[-1] if context and getattr(context, "pages", None) else None
            if page is not None:
                if _is_bot_blocked(page):
                    self.state.bot_blocked = True
                    self.state.bot_block_reason = f"bot-block page at {getattr(page, 'url', '<unknown>')}"
                    self.state.phase = 3
                    return self._mk_intent(
                        run_id=run_id,
                        typ="wait",
                        action={"type": "wait", "display_scope": "isolated", "wait_ms": 1},
                        rationale="objective satisfied: bot blocked",
                    )
                try:
                    links = page.locator(RESULT_SELECTOR)
                    if links.count() >= RANK:
                        node = links.nth(RANK - 1)
                        href = node.get_attribute("href")
                        self.state.expected_href = href
                        self.state.expected_host = _extract_destination_token(href)
                        # Force target blank so the run deterministically exercises new-tab follow.
                        node.evaluate("el => el.setAttribute('target', '_blank')")
                except Exception:
                    pass
            self.state.phase = 2
            return self._mk_intent(
                run_id=run_id,
                typ="click",
                action={
                    "type": "click",
                    "display_scope": "isolated",
                    "target": {"kind": "dom", "selector": RESULT_SELECTOR, "index": RANK},
                },
                rationale="open the requested ranked result by DOM index",
            )

        # Phase 2: verify we followed the landed page/new-tab, then finish.
        context = getattr(bc, "_context", None)
        page = context.pages[-1] if context and getattr(context, "pages", None) else None
        if page is not None:
            self.state.final_url = str(getattr(page, "url", "") or "")
            try:
                self.state.saw_new_tab = bool(context and len(getattr(context, "pages", [])) > 1)
            except Exception:
                self.state.saw_new_tab = False
            if _is_bot_blocked(page):
                self.state.bot_blocked = True
                self.state.bot_block_reason = f"bot-block after click at {self.state.final_url}"
        self.state.phase = 3
        return self._mk_intent(
            run_id=run_id,
            typ="wait",
            action={"type": "wait", "display_scope": "isolated", "wait_ms": 1},
            rationale="objective satisfied",
        )


@pytest.mark.slow
@pytest.mark.browser
def test_autonomous_browser_dom_loop_e2e(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    if os.environ.get("AGY_BROWSER_E2E") != "1":
        pytest.skip("AGY_BROWSER_E2E != 1")

    mode_env = (os.environ.get("AGY_BROWSER_E2E_MODE") or "REAL").upper()
    mode = mode_env if mode_env in {"ISOLATED", "REAL"} else "REAL"
    browser_display = os.environ.get("AGY_BROWSER_E2E_DISPLAY")
    if not browser_display:
        browser_display = ":0" if mode == "REAL" else None

    run_calls = []
    orig_run = ActionExecutor._run

    def _spy_run(self: ActionExecutor, cmd: Any, env: Dict[str, str], timeout: float) -> bool:
        run_calls.append([str(x) for x in (cmd or [])])
        return orig_run(self, cmd, env, timeout)

    monkeypatch.setattr(ActionExecutor, "_run", _spy_run)

    sink = AuditEventSink("browser-e2e", events_path=tmp_path / "browser-e2e-events.jsonl")
    adapter = ComputerUseWorkerAdapter(audit_sink_factory=lambda _rid: sink)
    state = _E2EState()
    reasoner = _AutonomousE2EReasoner(adapter, state)
    adapter._reasoner = reasoner  # type: ignore[attr-defined]

    req = RunRequest(
        run_id="browser-e2e",
        objective=f"search '{QUERY}' and open result number {RANK}",
        mode=mode,
        real_gui_policy=os.environ.get("AGY_BROWSER_E2E_POLICY", "children"),
        ask_mode=os.environ.get("AGY_BROWSER_E2E_ASK_MODE", "on"),
        browser_engine="bing",
        browser_display=browser_display,
        budgets={
            "max_steps": 4,
            "max_actions": 4,
            "action_timeout_ms": 30000,
            "reasoning_timeout_ms": 45000,
            "confirmation_wait_timeout_ms": 60000,
            "max_cpu_percent": 300,
            "max_rss_mb": 4096,
            "max_processes": 128,
        },
    )

    handle = adapter.start(req)
    assert getattr(handle, "status", "") in {"completed", "stopped", "errored"}

    # Honest failure contract: if the engine blocks automation, this test must fail (never fake-pass).
    if state.bot_blocked:
        pytest.fail(f"Bing bot-blocked automated browser e2e: {state.bot_block_reason}")

    assert state.saw_dom_in_perception, "DOM perception was not observed in the adapter loop"
    assert state.browser_pid is not None and state.browser_pid > 0, "browser pid was not captured from BrowserController"
    assert state.owned_seen is True, "B1 failed: browser pid was not owned during actuation"

    events = sink.read_all()
    event_types = [str(e.get("event_type", "")) for e in events]
    assert "permission.prompt_shown" not in event_types, "B1 failed: GUI permission prompt was emitted"

    # B2: this DOM-only flow must not perform xdotool/window-coordinate injection.
    assert not run_calls, f"B2 failed: ActionExecutor._run was called unexpectedly: {run_calls}"

    assert state.expected_host, f"Could not derive destination host from ranked result href: {state.expected_href!r}"
    assert state.final_url, "No final landed URL captured after DOM click"
    assert state.expected_host in state.final_url.lower(), (
        f"Final URL {state.final_url!r} did not contain ranked-result destination host {state.expected_host!r}"
    )
    assert state.saw_new_tab, "New-tab follow path was not exercised"

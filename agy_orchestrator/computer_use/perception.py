"""PerceptionPipeline + collectors (Step 8).

FR-05 (AT-SPI graceful degrade), FR-06 (DOM best-effort CDP), FR-17 (multi-scope).
OBSERVE redaction (hardening #4, default-on via utils) before any reasoner payload.
All collectors and tools bound to explicit DISPLAY only; never default to real :0.
Hermetic, stdlib+subprocess, Xvfb-only in tests.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    ElementHandle,
    ElementSource,
    PerceptionSnapshot,
    Scope,
    SnapshotSummary,
    TextBlockSource,
    WorkerEvent,
    WorkerEventType,
)
from .utils import is_redaction_enabled_for_run, redact_secrets

try:
    import gi
    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi  # type: ignore
    _HAS_ATSPI = True
except Exception:
    _HAS_ATSPI = False
    Atspi = None  # type: ignore

try:
    from playwright.sync_api import sync_playwright  # type: ignore
    _HAS_PLAYWRIGHT = True
except Exception:
    _HAS_PLAYWRIGHT = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tid() -> str:
    return f"s-{uuid.uuid4().hex[:10]}"


def _e(display: str, env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    ee = dict(env) if env else os.environ.copy()
    ee["DISPLAY"] = display
    ee.pop("WAYLAND_DISPLAY", None)
    return ee


def _sh(cmd: List[str], env: Dict[str, str], to: float = 3.0) -> str:
    try:
        return subprocess.check_output(cmd, env=env, stderr=subprocess.DEVNULL, timeout=to, text=True)
    except Exception:
        return ""


class ATSPICollector:
    def __init__(self, display: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> None:
        self.display = display
        self.env = env or {}
    def collect(self) -> List[ElementHandle]:
        if not _HAS_ATSPI:
            return []
        return []


class GeometryCollector:
    def __init__(self, display: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> None:
        self.display = display or ":99"
        self.env = _e(self.display, env)
    def collect(self) -> Tuple[List[Dict[str, Any]], List[ElementHandle]]:
        wins: List[Dict[str, Any]] = []
        elems: List[ElementHandle] = []
        # pid enrichment (wmctrl -lp third column after wid+host) for realgui use; absent on isolated Xvfb so callers unchanged
        pid_map: Dict[str, Optional[int]] = {}
        try:
            lp = _sh(["wmctrl", "-lp"], self.env)
            for line in (lp or "").splitlines():
                p = line.split(None, 4)
                if len(p) >= 3:
                    try:
                        pid = int(p[2])
                        if pid > 0:
                            pid_map[p[0]] = pid
                    except Exception:
                        pass
        except Exception:
            pass
        out = _sh(["wmctrl", "-lG"], self.env)
        for line in out.splitlines():
            p = line.split(None, 6)
            if len(p) >= 6:
                try:
                    wid = p[0]
                    x, y, w, h = int(p[2]), int(p[3]), int(p[4]), int(p[5])
                    title = p[6] if len(p) > 6 else ""
                    win_d: Dict[str, Any] = {"window_id": wid, "title": title, "bbox": {"x": x, "y": y, "w": w, "h": h}}
                    if wid in pid_map:
                        win_d["pid"] = pid_map[wid]
                    wins.append(win_d)
                    app_pid = pid_map.get(wid)
                    elems.append(ElementHandle(handle_id=f"geom-{wid}", source=ElementSource.GEOMETRY.value, window_id=wid, name=title or None, app_pid=app_pid, bbox={"x": x, "y": y, "w": w, "h": h}, confidence=0.75, provenance=["GeometryCollector"]))
                except Exception:
                    pass
        if not wins:
            # xwininfo fallback for bare Xvfb (no EWMH wmctrl data)
            tree = _sh(["xwininfo", "-root", "-tree"], self.env, 2.5)
            import re
            for line in tree.splitlines():
                m = re.search(r"(0x[0-9a-f]+).*?\"([^\"]+)\".*?(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", line)
                if m:
                    wid, title, ww, hh, xx, yy = m.groups()
                    bb = {"x": int(xx), "y": int(yy), "w": int(ww), "h": int(hh)}
                    win_d: Dict[str, Any] = {"window_id": wid, "title": title, "bbox": bb}
                    if wid in pid_map:
                        win_d["pid"] = pid_map[wid]
                    wins.append(win_d)
                    app_pid = pid_map.get(wid)
                    elems.append(ElementHandle(handle_id=f"geom-{wid}", source=ElementSource.GEOMETRY.value, window_id=wid, name=title or None, app_pid=app_pid, bbox=bb, confidence=0.6, provenance=["GeometryCollector:xwininfo"]))
        return wins, elems


class OCRCollector:
    def __init__(self, display: Optional[str] = None, env: Optional[Dict[str, str]] = None, lang: str = "eng") -> None:
        self.display = display or ":99"
        self.env = _e(self.display, env)
        self.lang = lang
    def collect(self) -> List[Dict[str, Any]]:
        if not (shutil.which("scrot") and shutil.which("tesseract")):
            return []
        blocks: List[Dict[str, Any]] = []
        with tempfile.TemporaryDirectory() as td:
            img = os.path.join(td, "c.png")
            try:
                subprocess.check_call(["scrot", "-o", img], env=self.env, timeout=3, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                out = _sh(["tesseract", img, "stdout", "-l", self.lang, "--psm", "6"], self.env, 5.0)
                y = 20
                for line in (out or "").splitlines():
                    line = line.strip()
                    if line:
                        blocks.append({"text": line[:120], "bbox": {"x": 10, "y": y, "w": 400, "h": 14}, "source": TextBlockSource.OCR.value})
                        y += 16
            except Exception:
                pass
        return blocks


class BrowserDOMCollector:
    """Best-effort DOM collector (FR-06). Attempts Playwright CDP connect when
    endpoint is known; returns [] on any failure or when unavailable. Never
    required for core perception (geometry + OCR + optional ATSPI suffice).
    """
    def __init__(self, display: Optional[str] = None, env: Optional[Dict[str, str]] = None, cdp_endpoint: Optional[str] = None, live_page: Optional[Any] = None) -> None:
        self.cdp_endpoint = cdp_endpoint
        # Phase 1.x: when the owned BrowserController is driving the run, it hands us its
        # already-open Playwright page. We must read DOM from THAT page — opening a second
        # sync_playwright()/connect_over_cdp in the same thread raises ("Sync API inside the
        # asyncio loop"), which is why the CDP-reconnect path silently returned []. The live
        # page is the authoritative source; CDP reconnect is only the no-controller fallback.
        self.live_page = live_page

    def collect(self) -> List[ElementHandle]:
        # Preferred path: scan the controller's already-open live page directly (no reconnect).
        if self.live_page is not None:
            collect_dom = getattr(self.live_page, "collect_dom_elements", None)
            if callable(collect_dom):
                try:
                    return collect_dom(self._scan_page)
                except Exception:
                    return []
            try:
                return self._scan_page(self.live_page)
            except Exception:
                return []
        if not self.cdp_endpoint or not _HAS_PLAYWRIGHT:
            return []
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(self.cdp_endpoint, timeout=1000)
                contexts = list(browser.contexts)
                if not contexts:
                    return []
                pages = list(contexts[0].pages)
                if not pages:
                    return []
                page = pages[-1]
                return self._scan_page(page)
        except Exception:
            # Any error (no browser, wrong port, playwright transport, timeout, etc.)
            # is degraded silently — perception continues with other signals.
            return []

    def _scan_page(self, page: Any) -> List[ElementHandle]:
        """Extract DOM ElementHandles from a live Playwright page (shared by the live-page
        and CDP-reconnect paths). Bounded + dedup'd; never raises past the caller."""
        # ONE evaluate() round-trip. A per-element Playwright scan (locator.nth +
        # inner_text/evaluate/bounding_box per node) is hundreds of CDP round-trips on a
        # heavy SERP — minutes of wall time, which hung the perceive->reason->act loop.
        # A single in-page JS pass is fast, deterministic, and bounded to 120 elements.
        js = """
        () => {
          const sels = ["#b_results h2 a", "#b_results a", "#search h3 a",
            "#search a h3", "[data-testid*='result'] a", "a[href]", "button",
            "input", "textarea", "[role='link']", "[role='button']", "[role='textbox']"];
          const out = [];
          const seen = new Set();
          for (const sel of sels) {
            let nodes;
            try { nodes = document.querySelectorAll(sel); } catch (e) { continue; }
            for (const n of nodes) {
              if (out.length >= 120) break;
              const r = n.getBoundingClientRect();
              const tag = (n.tagName || "").toLowerCase();
              let text = (n.innerText || n.getAttribute("aria-label") ||
                n.getAttribute("title") || n.getAttribute("placeholder") ||
                n.value || n.getAttribute("href") || "").trim();
              const key = tag + "|" + text.slice(0,120) + "|" +
                Math.round(r.x) + "," + Math.round(r.y);
              if (seen.has(key)) continue;
              seen.add(key);
              out.push({tag: tag, text: text.slice(0,200), sel: sel,
                x: Math.round(r.x), y: Math.round(r.y),
                w: Math.round(r.width), h: Math.round(r.height)});
            }
          }
          return out;
        }
        """
        try:
            rows = page.evaluate(js)
        except Exception:
            return []
        out: List[ElementHandle] = []
        for i, row in enumerate(rows or [], start=1):
            if i > 120:
                break
            try:
                w = int(row.get("w", 0) or 0)
                h = int(row.get("h", 0) or 0)
                bbox_d: Dict[str, int] = {}
                if w > 0 and h > 0:
                    bbox_d = {"x": int(row.get("x", 0) or 0), "y": int(row.get("y", 0) or 0), "w": w, "h": h}
                text = (row.get("text") or "").strip()
                tag = (row.get("tag") or "").strip()
                out.append(
                    ElementHandle(
                        source=ElementSource.DOM.value,
                        handle_id=f"dom-{i}",
                        name=text or None,
                        role=tag or None,
                        bbox=bbox_d,
                        confidence=0.9,
                        provenance=[f"BrowserDOMCollector:{row.get('sel', '')}"],
                    )
                )
            except Exception:
                continue
        return out


class PerceptionPipeline:
    def __init__(
        self,
        audit_sink: Optional[Any] = None,
        redaction_enabled: Optional[bool] = None,
        default_isolated_display: str = ":99",
        default_observe_display: str = ":0",
        **kw: Any,
    ) -> None:
        self.audit_sink = audit_sink
        self.redaction_enabled = is_redaction_enabled_for_run() if redaction_enabled is None else bool(redaction_enabled)
        self.default_isolated_display = default_isolated_display
        self.default_observe_display = default_observe_display

    def snapshot(
        self,
        scope: str,
        *,
        display: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        cdp_endpoint: Optional[str] = None,
        run_id: str = "p8",
        redaction_enabled: Optional[bool] = None,
        **kw: Any,
    ) -> PerceptionSnapshot:
        # Normalize scope; support per-scope overrides for FR-17 multi-scope
        # (e.g. snapshot_set(..., displays={"isolated": ":99", "observe_real": ":0"}, ...))
        if scope not in {s.value for s in Scope}:
            scope = Scope.ISOLATED.value

        # Resolve effective display/env for *this* scope (scalar wins over dict, then default)
        eff_display = display
        eff_env = env
        displays_d = kw.get("displays") or {}
        envs_d = kw.get("envs") or {}
        if isinstance(displays_d, dict):
            eff_display = displays_d.get(scope, eff_display)
        if isinstance(envs_d, dict):
            eff_env = envs_d.get(scope, eff_env)
        if not eff_display:
            eff_display = self.default_isolated_display if scope == Scope.ISOLATED.value else self.default_observe_display
        if eff_env is None:
            eff_env = {}

        use_red = self.redaction_enabled if redaction_enabled is None else bool(redaction_enabled)
        sid = _tid()
        ts = _now()
        m = "ISOLATED" if scope == Scope.ISOLATED.value else "OBSERVE"

        # All collectors receive explicit DISPLAY-bound env (never implicit real :0)
        at = ATSPICollector(eff_display, eff_env).collect()
        gw, ge = GeometryCollector(eff_display, eff_env).collect()
        ob = OCRCollector(eff_display, eff_env).collect()
        # Phase 1.x: prefer the owned BrowserController's live page (passed by the adapter)
        # so DOM perception works in-process; cdp_endpoint is the no-controller fallback.
        de = BrowserDOMCollector(eff_display, eff_env, cdp_endpoint, live_page=kw.get("dom_page")).collect()

        elems: List[ElementHandle] = []
        seen: set[str] = set()
        for el in at + ge + de:
            if el.handle_id not in seen:
                seen.add(el.handle_id)
                elems.append(el)
        raw = list(ob)

        # OBSERVE hardening #4: real-:0-scope text (titles, OCR, AT-SPI names, DOM, terminal)
        # MUST be redacted *before* the snapshot (and thus before any ReasoningInput/envelope).
        # ISOLATED scope is never redacted here (controlled content; redaction would be lossy).
        if scope == Scope.OBSERVE_REAL.value and use_red:
            gw = [
                {**w, "title": redact_secrets(w.get("title", "")) if isinstance(w.get("title"), str) else w.get("title")}
                for w in gw
            ]
            raw = [
                {**b, "text": redact_secrets(b.get("text", "")) if isinstance(b.get("text"), str) else b.get("text")}
                for b in raw
            ]
            elems = [
                ElementHandle(
                    handle_id=el.handle_id,
                    source=el.source,
                    window_id=el.window_id,
                    app_pid=el.app_pid,
                    role=el.role,
                    name=redact_secrets(el.name) if el.name else el.name,
                    bbox=dict(el.bbox),
                    confidence=el.confidence,
                    provenance=list(el.provenance),
                )
                if el.name
                else el
                for el in elems
            ]

        snap = PerceptionSnapshot(
            snapshot_id=sid, run_id=run_id, mode=m, scope=scope, captured_at=ts,
            windows=gw, elements=elems, raw_text_blocks=raw
        )
        if self.audit_sink:
            try:
                self.audit_sink.emit(
                    WorkerEvent(
                        ts=str(time.time()),
                        run_id=run_id,
                        event_type=WorkerEventType.PERCEPTION_SNAPSHOT.value,
                        payload={"scope": scope, "snapshot_id": sid, "windows": len(gw), "elements": len(elems)},
                    )
                )
            except Exception:
                pass
        return snap

    def snapshot_set(self, scopes: List[str], **kw: Any) -> Dict[str, PerceptionSnapshot]:
        out: Dict[str, PerceptionSnapshot] = {}
        for sc in scopes:
            out[sc] = self.snapshot(sc, **kw)
        return out

    def make_summary(self, snap: PerceptionSnapshot) -> SnapshotSummary:
        se: List[Dict[str, Any]] = []
        for el in snap.elements:
            # Serialize element source in the lowercase wire convention the reasoner
            # contract uses (matching DOM-actuation `kind == "dom"`). ElementSource enum
            # values are uppercase ("DOM"/"GEOMETRY"/...); the reasoner-facing snapshot
            # dict expects "dom"/"geometry"/... so the loop can select DOM elements.
            src = el.source.lower() if isinstance(el.source, str) else el.source
            se.append({"handle_id": el.handle_id, "source": src, "role": el.role, "name": el.name, "bbox": dict(el.bbox), "confidence": el.confidence})
        return SnapshotSummary(snapshot_id=snap.snapshot_id, captured_at=snap.captured_at, scope=snap.scope, windows=[dict(w) for w in snap.windows], elements=se, raw_text_blocks=[dict(b) for b in snap.raw_text_blocks])


__all__ = ["ATSPICollector", "GeometryCollector", "OCRCollector", "BrowserDOMCollector", "PerceptionPipeline"]

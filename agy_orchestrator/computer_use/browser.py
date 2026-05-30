"""BrowserController (Phase 1.x Step 3 — autonomous browser/DOM actuation).

A dependency-injected, agent-owned, CDP-controlled Chromium session (one per run,
lazily created on first navigate). Reuses the exact launch mechanism proven in
scripts/agent_browser_demo.py (chromium-1217 preference + glob fallback; the five
launch args + --remote-debugging-port=0; new-tab follow via context.pages[-1]).

- Launch on the run's chosen display (":0" for REAL → visible agent child on
  operator desktop; isolated Xvfb otherwise).
- Registers the browser pid via supervisor.adopt_owned_process(...) so that
  supervisor.is_owned(pid) is True (B1: no GUI prompt under real_full/children;
  B2: controller never injects into foreign windows/pids).
- Exposes .cdp_endpoint (http:// or ws:// form that BrowserDOMCollector can pass
  to connect_over_cdp) for live DOM perception.
- .navigate(url) ensures browser, does goto, returns (landed_url, owned_pid).
- .click_dom(selector, index=1) / .type_dom(...) perform locator-based actuation
  (1-based index) with deterministic new-tab follow, exactly as the prototype.
- .close() calls supervisor.terminate_tree (B3: leak-free; inherited by watchdog).
- Any launch/actuation error is fail-closed (caller maps to FAILED/REJECTED).

FakeBrowserController is the *mandatory* hermetic double (in-memory scripted pages,
deterministic 1-based indexing, new-tab simulation, ZERO playwright/subprocess/psutil/X
imports or side-effects anywhere in its code paths). Mirrors FakeOwnershipResolver /
FakePrompter exactly. **No hermetic test, including release_blocking/browser-marked
suites, ever constructs BrowserController or touches a real browser / :0 display.**

Phase 1.x (design addendum): new module + Fake (additive only; zero behavior change
to ISOLATED/OBSERVE/REAL paths, SafetyKernel, ownership model, or any existing test;
INV F / B4 preserved exactly).

All heavy imports (playwright, psutil) are lazy and confined to the real
BrowserController code paths. The module imports cleanly for Fake-only usage.
"""

from __future__ import annotations

import os
import glob
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .process_supervisor import ProcessSupervisor


def _find_chromium_executable() -> str:
    """Exact logic from scripts/agent_browser_demo.py (verified working 2026-05-29)."""
    pref = os.path.expanduser("~/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome")
    if os.path.exists(pref):
        return pref
    cands = sorted(
        glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome")),
        reverse=True,
    )
    if not cands:
        raise RuntimeError(
            "No cached Playwright chromium found (~/.cache/ms-playwright/chromium-*). "
            "Run: playwright install chromium"
        )
    return cands[0]


class BrowserController:
    """Real, agent-owned CDP Chromium driver (lazily launched on first navigate).

    Dependency-injected: the caller (SessionController / adapter wiring in later steps)
    supplies the supervisor (for adopt + terminate_tree), the display, engine, and
    timeouts. The controller itself never reads os.environ for DISPLAY; it receives
    the authoritative value from RunRequest.browser_display (defaulted per mode).
    """

    def __init__(
        self,
        supervisor: Optional["ProcessSupervisor"] = None,
        display: str = ":99",
        engine: str = "bing",
        cdp_timeout: float = 30000.0,
        playwright_timeout: float = 30000.0,
        **kw: Any,
    ) -> None:
        self._supervisor = supervisor
        self._display = display or ":99"
        self._engine = engine or "bing"
        self._cdp_timeout = cdp_timeout
        self._playwright_timeout = playwright_timeout

        # Lazily populated on first navigate
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._browser_pid: Optional[int] = None
        self._root_id: Optional[str] = None
        self._user_data_dir: Optional[str] = None
        self._cdp_endpoint: Optional[str] = None
        self._closed = False

    @property
    def cdp_endpoint(self) -> Optional[str]:
        """The endpoint (http:// or ws://) that BrowserDOMCollector can pass to connect_over_cdp.
        Populated after successful launch (best-effort; may be None on exotic setups).
        """
        return self._cdp_endpoint

    @property
    def browser_pid(self) -> Optional[int]:
        """The OS pid of the Chromium root process (registered as owned via supervisor)."""
        return self._browser_pid

    def _build_clean_env(self) -> Dict[str, str]:
        """Return env for the browser process.

        - For REAL ":0" (or equivalent): clean copy with ONLY DISPLAY set; zero
          AGY_ISOLATED_X, zero private XAUTHORITY/HOME, zero leakage of operator
          cookie material. This guarantees the launched browser is a normal :0
          client (visible to operator) and is still an *owned child* under the
          supervisor (B1/B2).
        - For isolated Xvfb displays: delegate to get_isolated_env so the browser
          can authenticate to the private Xvfb (required for rendering).
        """
        display = self._display
        is_real_zero = display == ":0"

        if is_real_zero:
            # REAL path: minimal clean env, no isolation markers at all.
            child_env: Dict[str, str] = {}
            for k, v in os.environ.items():
                if k in ("DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "AGY_ISOLATED_X"):
                    continue
                if isinstance(v, str) and (".Xauthority" in v or "/Xauth" in v or "agu-" in v):
                    continue
                child_env[k] = v
            child_env["DISPLAY"] = display
            # Do not set XAUTHORITY → Chrome falls back to ~/.Xauthority (correct for :0)
            child_env.pop("XAUTHORITY", None)
            child_env.pop("WAYLAND_DISPLAY", None)
            child_env.pop("AGY_ISOLATED_X", None)
            return child_env

        # Isolated path: use the hardened isolated env (provides private cookie for the Xvfb).
        # The caller (later wiring) guarantees the Xvfb for this display is already up.
        try:
            from .xauth import get_isolated_env  # local import, only for isolated

            return get_isolated_env(display)
        except Exception:
            # Belt-and-suspenders fallback (still safe; browser will fail to render
            # but never leaks real :0 cookie).
            child_env = {}
            for k, v in os.environ.items():
                if k in ("XAUTHORITY", "WAYLAND_DISPLAY"):
                    continue
                if isinstance(v, str) and (".Xauthority" in v or "/Xauth" in v):
                    continue
                child_env[k] = v
            child_env["DISPLAY"] = display
            return child_env

    def _extract_browser_pid(self, browser: Any, exe: str) -> Optional[int]:
        """Best-effort extraction of the Chromium root pid from playwright internals
        or (fallback) recent psutil scan of matching --remote-debugging-port children.
        Never raises; returns None on total failure (later adopt will be skipped).
        """
        # 1. Playwright internal (fast path when available)
        try:
            impl = getattr(browser, "_impl_obj", None)
            if impl is not None:
                for attr in ("_process", "process", "_browser_process"):
                    cand = getattr(impl, attr, None)
                    if cand is not None:
                        if hasattr(cand, "pid") and isinstance(cand.pid, int) and cand.pid > 0:
                            return cand.pid
                        p = getattr(cand, "process", None)
                        if p is not None and hasattr(p, "pid") and isinstance(p.pid, int) and p.pid > 0:
                            return p.pid
        except Exception:
            pass

        # 2. psutil scan (reliable on Linux; only the chrome we just started will
        #    match the exact exe + the remote-debugging-port arg we passed).
        try:
            import psutil  # type: ignore  # lazy, only real path

            candidates: List[Tuple[float, int]] = []
            for proc in psutil.process_iter(["pid", "cmdline", "create_time", "exe"]):
                try:
                    info = proc.info
                    if not info.get("exe") or exe not in str(info["exe"]):
                        continue
                    cmd = " ".join(info.get("cmdline") or [])
                    if "--remote-debugging-port" not in cmd:
                        continue
                    ct = info.get("create_time") or 0.0
                    candidates.append((float(ct), int(info["pid"])))
                except Exception:
                    continue
            if candidates:
                candidates.sort(reverse=True)  # newest first
                return candidates[0][1]
        except Exception:
            pass
        return None

    def _discover_cdp_endpoint(self, user_data_dir: str) -> Optional[str]:
        """Read Chrome's DevToolsActivePort (written for --remote-debugging-port=0)
        and return an http:// base that connect_over_cdp will resolve via /json/version.
        """
        try:
            p = Path(user_data_dir) / "DevToolsActivePort"
            if p.exists():
                text = p.read_text().strip()
                if text:
                    port = text.splitlines()[0].strip()
                    if port.isdigit():
                        return f"http://127.0.0.1:{port}"
        except Exception:
            pass
        return None

    def _launch(self) -> None:
        """Launch the owned Chromium (idempotent; only first navigate pays the cost).

        Uses launch_persistent_context: modern Playwright rejects ``--user-data-dir``
        as a launch *arg* (it must be the persistent-context's own parameter), and a
        persistent profile dir is exactly what we need for DevToolsActivePort →
        CDP-endpoint discovery with ``--remote-debugging-port=0``. The launched
        Chromium is still an agent-owned child (adopted via the supervisor) on the
        run's display, matching scripts/agent_browser_demo.py's proven mechanism.
        """
        if self._context is not None:
            return
        if self._closed:
            raise RuntimeError("BrowserController already closed")

        exe = _find_chromium_executable()
        child_env = self._build_clean_env()

        # Persistent profile dir (Chromium writes DevToolsActivePort here for port=0).
        self._user_data_dir = tempfile.mkdtemp(prefix="agu-browser-")

        from playwright.sync_api import sync_playwright  # type: ignore  # lazy real-only import

        self._playwright = sync_playwright().start()

        # NOTE: no --user-data-dir here; launch_persistent_context manages the profile.
        args = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-blink-features=AutomationControlled",
            "--remote-debugging-port=0",
        ]

        # Headed on REAL :0 so the operator can watch the agent's browser window.
        # Headless on isolated Xvfb (saves a tiny amount of resources; still renders).
        headless = self._display != ":0"

        # launch_persistent_context returns a BrowserContext directly (no separate
        # Browser object); context options (viewport/locale/user_agent) are passed here.
        context = self._playwright.chromium.launch_persistent_context(
            self._user_data_dir,
            executable_path=exe,
            args=args,
            env=child_env,
            headless=headless,
            slow_mo=0,
            viewport=None if not headless else {"width": 1280, "height": 900},
            locale="en-US",
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        self._context = context
        # The underlying Browser handle (may be None on some persistent setups; kept for close()).
        self._browser = getattr(context, "browser", None)

        # Extract pid (persistent context: psutil scan keyed on exe + remote-debugging-port).
        pid = self._extract_browser_pid(self._browser, exe)
        self._browser_pid = pid if pid else 0

        # Register with supervisor so is_owned(pid) == True and terminate_tree will reap (B1/B3)
        if self._supervisor is not None and self._browser_pid and self._browser_pid > 0:
            try:
                h = self._supervisor.adopt_owned_process(
                    self._browser_pid,
                    display_scope=self._display,
                    argv=["chromium-browser", f"--display={self._display}"],
                    source="browser",
                )
                self._root_id = h.root_id
            except Exception:
                self._root_id = None

        # Persistent context opens with an initial page; reuse it (else create one).
        self._page = context.pages[0] if context.pages else context.new_page()

        # Discover CDP endpoint with short retry (DevToolsActivePort appears shortly after launch).
        # Still best-effort only; BrowserDOMCollector and actuation paths tolerate None (B4 safe).
        for _ in range(12):
            self._cdp_endpoint = self._discover_cdp_endpoint(self._user_data_dir)
            if self._cdp_endpoint:
                break
            time.sleep(0.05)

    def navigate(self, url: str) -> Tuple[str, int]:
        """Ensure owned browser exists (launch on first use), navigate, return landed URL + owned pid.

        Targets the most recent page (context.pages[-1] after new-tab clicks) for consistency
        with click_dom/type_dom and to support autonomous perceive->reason->act flows.
        """
        if not url or not isinstance(url, str):
            raise ValueError("url must be non-empty string")
        self._launch()
        assert self._context is not None and self._page is not None
        try:
            active = self._context.pages[-1] if self._context.pages else self._page
            active.goto(url, wait_until="domcontentloaded", timeout=self._playwright_timeout)
            # Best-effort: give the page a moment for client-side redirects / consent
            try:
                active.wait_for_timeout(800)
            except Exception:
                pass
            landed = active.url or url
            self._page = active  # keep handle in sync for subsequent dom actions / navs
        except Exception as e:
            # Fail-closed for the action layer (caller will turn this into FAILED result)
            raise RuntimeError(f"navigate failed: {type(e).__name__}: {e}") from e

        pid = self._browser_pid or 0
        return landed, pid

    def click_dom(self, selector: Optional[str] = None, index: int = 1) -> Dict[str, Any]:
        """Locator-based click of the Nth (1-based) match. Supports new-tab follow (capture pages[-1])."""
        if not self._context or not self._page or self._closed:
            return {"ok": False, "error_code": "browser_not_open", "index": index}

        sel = selector or "a[href], button, [role='link']"
        idx = max(1, int(index)) - 1  # 0-based for nth()

        try:
            # Always act on the most recent page (handles prior new-tab cases)
            active = self._context.pages[-1] if self._context.pages else self._page
            target = active.locator(sel).nth(idx)
            before_pages = len(self._context.pages)

            target.scroll_into_view_if_needed(timeout=4000)
            target.click(timeout=8000)

            # Give the navigation / new-tab a moment
            try:
                active.wait_for_timeout(1200)
            except Exception:
                pass

            landed_page = self._context.pages[-1] if len(self._context.pages) > before_pages else active
            try:
                landed_page.wait_for_load_state("domcontentloaded", timeout=self._cdp_timeout)
            except Exception:
                pass

            # Update our "current page" handle for subsequent operations
            self._page = landed_page

            return {
                "ok": True,
                "index": index,
                "selector": sel,
                "landed_url": landed_page.url,
                "child_page_url": landed_page.url if len(self._context.pages) > before_pages else None,
                "new_tab": len(self._context.pages) > before_pages,
            }
        except Exception as e:
            return {"ok": False, "error_code": "dom_actuation_failed", "index": index, "error": str(e)}

    def type_dom(
        self, selector: Optional[str], index: int, text: str, press_enter: bool = True
    ) -> Dict[str, Any]:
        """Locator-based type into the Nth match (optionally press Enter)."""
        if not self._context or not self._page or self._closed:
            return {"ok": False, "error_code": "browser_not_open", "index": index}

        sel = selector or "input, textarea, [contenteditable='true']"
        idx = max(1, int(index)) - 1

        try:
            active = self._context.pages[-1] if self._context.pages else self._page
            target = active.locator(sel).nth(idx)

            target.scroll_into_view_if_needed(timeout=4000)
            target.click(timeout=4000)
            target.fill("")  # clear first
            target.type(text, delay=50)
            if press_enter:
                target.press("Enter")
                try:
                    active.wait_for_load_state("domcontentloaded", timeout=8000)
                except Exception:
                    pass

            return {
                "ok": True,
                "index": index,
                "selector": sel,
                "text": text,
                "landed_url": active.url,
            }
        except Exception as e:
            return {"ok": False, "error_code": "dom_actuation_failed", "index": index, "error": str(e)}

    def close(self) -> None:
        """Terminate the owned browser tree (via supervisor) and stop playwright driver. Idempotent."""
        if self._closed:
            return
        self._closed = True

        # B3: always attempt supervisor reaping first (covers the full tree + rlimits)
        if self._root_id and self._supervisor is not None:
            try:
                self._supervisor.terminate_tree(self._root_id)
            except Exception:
                pass

        # Close playwright objects (best-effort). For a persistent context, closing
        # the context is what shuts the browser down; close the Browser handle too if present.
        try:
            if self._context is not None:
                self._context.close()
        except Exception:
            pass
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception:
            pass

        # Clean the temp user-data dir we created for DevToolsActivePort discovery
        if self._user_data_dir:
            try:
                shutil.rmtree(self._user_data_dir, ignore_errors=True)
            except Exception:
                pass

        self._browser = self._context = self._page = self._playwright = None
        self._browser_pid = None
        self._root_id = None
        self._cdp_endpoint = None


class FakeBrowserController:
    """The *mandatory* pure test double for every hermetic browser test (release_blocking + browser group).

    - In-memory only: scripted navigation results, deterministic 1-based DOM indexing,
      synthetic new-tab simulation on click.
    - Never imports playwright, subprocess, psutil, or any X11 module.
    - Zero side effects: no processes, no files, no DISPLAY, no network, no :0.
    - Call recording (self.calls) + pid/display/engine echo for later B1/B2/B3 assertions.
    - Mirrors FakeOwnershipResolver / FakePrompter style exactly (constructor defaults,
      queue not needed here, explicit "fake": True markers, self-test block).
    """

    def __init__(
        self,
        supervisor: Optional[Any] = None,
        display: str = ":99",
        engine: str = "bing",
        cdp_timeout: float = 30000.0,
        playwright_timeout: float = 30000.0,
        **kw: Any,
    ) -> None:
        self._supervisor = supervisor
        self._display = display
        self._engine = engine
        self._cdp_timeout = cdp_timeout
        self._playwright_timeout = playwright_timeout

        self._closed = False
        self._current_url = "about:blank"
        self._owned_pid = 424242  # stable synthetic owned pid for is_owned tests
        self._nav_count = 0
        self.calls: List[Dict[str, Any]] = []

    @property
    def cdp_endpoint(self) -> Optional[str]:
        return "ws://127.0.0.1:0/fake-browser-cdp"

    @property
    def browser_pid(self) -> int:
        return self._owned_pid

    def navigate(self, url: str) -> Tuple[str, int]:
        self.calls.append({"op": "navigate", "url": url, "ts": time.time()})
        self._nav_count += 1
        self._current_url = url or "about:blank"
        # Simple deterministic landing for common test patterns
        landed = self._current_url
        if "example" in self._current_url or "bing.com" in self._current_url:
            landed = "https://example.com/search?q=test"
        return landed, self._owned_pid

    def click_dom(self, selector: Optional[str] = None, index: int = 1) -> Dict[str, Any]:
        self.calls.append({"op": "click_dom", "selector": selector, "index": index})
        if self._closed or self._nav_count == 0:
            return {"ok": False, "error_code": "browser_not_open", "index": index}
        idx = max(1, int(index))
        # Simulate new-tab follow for realism (tests assert child_page_url present)
        new_url = f"https://result{idx}.example.com/"
        self._current_url = new_url
        return {
            "ok": True,
            "index": idx,
            "selector": selector,
            "landed_url": new_url,
            "child_page_url": new_url,
            "new_tab": True,
            "fake": True,
        }

    def type_dom(
        self, selector: Optional[str], index: int, text: str, press_enter: bool = True
    ) -> Dict[str, Any]:
        self.calls.append({"op": "type_dom", "selector": selector, "index": index, "text": text})
        if self._closed or self._nav_count == 0:
            return {"ok": False, "error_code": "browser_not_open", "index": index}
        idx = max(1, int(index))
        landed = self._current_url
        return {
            "ok": True,
            "index": idx,
            "selector": selector,
            "text": text,
            "landed_url": landed,
            "fake": True,
        }

    def close(self) -> None:
        self._closed = True
        self.calls.append({"op": "close"})
        # Fake never calls real supervisor.terminate_tree; tests that care about B3
        # inject a real (or instrumented) supervisor and assert on its registry directly.
        self._supervisor = None


# ------------------------------------------------------------------
# Hermetic self-test (python -m agy_orchestrator.computer_use.browser).
# Exercises ONLY FakeBrowserController. Never constructs the real class,
# never imports playwright, never touches any display or process.
# ------------------------------------------------------------------
if __name__ == "__main__":
    fb = FakeBrowserController(display=":42", engine="bing")
    assert fb.cdp_endpoint is not None and "fake" in fb.cdp_endpoint
    assert fb.browser_pid == 424242

    u, pid = fb.navigate("https://bing.com/search?q=foo")
    assert "bing" in fb.calls[-1]["url"] or "foo" in u
    assert pid == 424242
    assert fb._nav_count == 1

    r1 = fb.click_dom("li.result a", 3)
    assert r1["ok"] is True and r1["index"] == 3 and "result3" in r1["child_page_url"]
    assert r1.get("fake") is True

    r2 = fb.type_dom("#q", 1, "hello world", press_enter=True)
    assert r2["ok"] is True and r2["text"] == "hello world"

    fb.close()
    assert fb._closed is True

    # browser_not_open path
    fb2 = FakeBrowserController()
    r3 = fb2.click_dom("a", 1)
    assert r3["error_code"] == "browser_not_open"

    print("browser.py Fake-only self-test OK (zero playwright / :0 / side effects)")

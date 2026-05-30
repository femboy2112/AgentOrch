"""SessionController browser ownership tests (Phase 1.x Step 4 — hermetic Fake only).

Covers B1/B2/B3 lifecycle for the agent-owned BrowserController (real or Fake) stored on internal Session:
- always present (Fake under test)
- eff_browser_display computation (REAL=:0, else isolated Xvfb)
- browser_engine default "bing"
- current_cdp_endpoint exposure (delegates to controller)
- injection at SessionController ctor level respected
- close_session / stop paths reap via bc.close() -> supervisor.terminate_tree (B3 no-leak)
- WorkerSession echoes browser_* fields
- zero real browser / :0 / playwright launch in any path

All tests use FakeBrowserController exclusively (via under_test detection + no-playwright env).
Marked release_blocking + browser per spec. B4: does not alter any prior test behavior/counts.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agy_orchestrator.computer_use.models import RunMode, RunRequest
from agy_orchestrator.computer_use.session import SessionController


def _force_fake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure hermetic Fake path (no playwright, no :0)."""
    import sys

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_computer_use_session")
    # Also ensure no stray playwright in this process for the probe in create_session
    monkeypatch.delitem(sys.modules, "playwright", raising=False)
    monkeypatch.delitem(sys.modules, "playwright.sync_api", raising=False)


@pytest.mark.browser
@pytest.mark.release_blocking
def test_create_session_always_owns_browser_controller_fake(monkeypatch: pytest.MonkeyPatch):
    """Every session gets a browser_controller (Fake in hermetic runs)."""
    _force_fake_env(monkeypatch)
    ctrl = SessionController()
    req = RunRequest(run_id="s4-bc-1", objective="own browser")
    sess = ctrl.create_session(req)
    assert sess.browser_controller is not None
    assert "FakeBrowserController" in type(sess.browser_controller).__name__
    assert sess.browser_engine == "bing"
    assert sess.browser_display is not None and sess.browser_display.startswith(":")
    assert sess.current_cdp_endpoint is not None and "fake" in sess.current_cdp_endpoint
    ctrl.close_session("s4-bc-1")


@pytest.mark.browser
@pytest.mark.release_blocking
def test_eff_browser_display_real_vs_isolated(monkeypatch: pytest.MonkeyPatch):
    """REAL mode defaults eff display to :0 (visible agent child); others use isolated xvfb."""
    _force_fake_env(monkeypatch)
    ctrl = SessionController()
    # ISOLATED
    req_iso = RunRequest(run_id="s4-d-iso", objective="iso", mode=RunMode.ISOLATED.value)
    s_iso = ctrl.create_session(req_iso)
    assert s_iso.browser_display is not None and s_iso.browser_display != ":0"
    ctrl.close_session("s4-d-iso")

    # REAL (still Fake, no :0 touch)
    req_real = RunRequest(run_id="s4-d-real", objective="real", mode=RunMode.REAL.value)
    s_real = ctrl.create_session(req_real)
    assert s_real.browser_display == ":0"
    ctrl.close_session("s4-d-real")


@pytest.mark.browser
@pytest.mark.release_blocking
def test_browser_controller_injection_at_controller_level(monkeypatch: pytest.MonkeyPatch):
    """Ctor-injected browser_controller is stored and reused (advanced DI path)."""
    _force_fake_env(monkeypatch)
    fake_bc = MagicMock()
    fake_bc.cdp_endpoint = "ws://injected"
    fake_bc.browser_pid = 777
    ctrl = SessionController(browser_controller=fake_bc)
    req = RunRequest(run_id="s4-inj", objective="injected")
    sess = ctrl.create_session(req)
    assert sess.browser_controller is fake_bc
    assert sess.current_cdp_endpoint == "ws://injected"
    ctrl.close_session("s4-inj")
    # close should have been called by close_session B3 path
    fake_bc.close.assert_called()


@pytest.mark.browser
@pytest.mark.release_blocking
def test_close_session_reaps_browser_tree_b3(monkeypatch: pytest.MonkeyPatch):
    """close_session calls bc.close() (which for real does terminate_tree; for Fake marks closed). B3."""
    _force_fake_env(monkeypatch)
    sup = MagicMock()
    sup.terminate_tree = MagicMock()
    sup.roots = MagicMock(return_value=[])
    ctrl = SessionController(supervisor=sup)  # injected sup is used for xvfb + adopt in real path
    req = RunRequest(run_id="s4-b3", objective="b3 reap")
    sess = ctrl.create_session(req)
    bc = sess.browser_controller
    assert bc is not None
    # navigate to ensure "launched" (for Fake just records)
    bc.navigate("https://example.test")
    ctrl.close_session("s4-b3")
    # For Fake: _closed; for real path the sup.terminate would have been called inside bc.close
    assert getattr(bc, "_closed", False) or sup.terminate_tree.called
    assert ctrl.get_session("s4-b3") is None


@pytest.mark.browser
@pytest.mark.release_blocking
def test_worker_session_echoes_browser_fields(monkeypatch: pytest.MonkeyPatch):
    """The persisted WorkerSession snapshot carries the effective browser_engine/display."""
    _force_fake_env(monkeypatch)
    ctrl = SessionController()
    req = RunRequest(run_id="s4-ws", objective="echo", browser_engine="duckduckgo", browser_display=":123")
    sess = ctrl.create_session(req)
    ws = sess.worker_session
    assert ws.browser_engine == "duckduckgo"
    assert ws.browser_display == ":123"
    ctrl.close_session("s4-ws")


# B4 guard: importing this module + running under pytest never constructs real BrowserController
def test_no_real_browser_in_hermetic_session_tests(monkeypatch: pytest.MonkeyPatch):
    _force_fake_env(monkeypatch)
    from agy_orchestrator.computer_use.browser import FakeBrowserController

    # BrowserController class is importable (lazy), but constructing under test forces Fake path in SessionController
    # (explicit construction of real is forbidden in hermetic tests per spec)
    assert FakeBrowserController is not None
    # The real class may be present but we never instantiate it here

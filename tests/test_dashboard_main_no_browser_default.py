"""Regression test: `python -m dashboard` MUST NOT auto-open a browser.

A previous version of `dashboard/__main__.py` defaulted to opening the page
in the system browser unless `--no-browser` was passed. During the W3
master-mode build, a worker smoke-testing the dashboard across 8 phases ×
multiple iterations spawned enough Firefox windows to wedge the operator's
workstation overnight. The fix flipped the default — browser launch is now
opt-in via ``--browser``. This test guards the new default.
"""
from __future__ import annotations

import sys
from unittest import mock

import dashboard.__main__ as dash_main


def _run(args: list[str]):
    """Invoke dashboard.__main__.main() with patched argv + uvicorn + webbrowser.

    We don't want a real uvicorn boot or a real browser launch inside the
    test, so both are stubbed. We assert on whether webbrowser.open was
    called — that's the entire footprint of the bug.
    """
    with (
        mock.patch.object(sys, "argv", ["python -m dashboard", *args]),
        mock.patch("webbrowser.open") as wb_open,
        mock.patch("uvicorn.run") as uv_run,
    ):
        dash_main.main()
        return wb_open, uv_run


def test_default_does_not_open_browser():
    """No flag = no browser. The bug case."""
    wb_open, uv_run = _run([])
    wb_open.assert_not_called()
    uv_run.assert_called_once()


def test_no_browser_flag_does_not_open_browser():
    """Back-compat: the now-deprecated --no-browser must still be accepted
    and must not open a browser (since opt-out can't apply when the default
    is already off, but old scripts still pass it)."""
    wb_open, uv_run = _run(["--no-browser"])
    wb_open.assert_not_called()
    uv_run.assert_called_once()


def test_browser_flag_opens_browser():
    """Opt-in: --browser explicitly asks for the browser; honour it."""
    wb_open, uv_run = _run(["--browser"])
    wb_open.assert_called_once()
    args, _ = wb_open.call_args
    assert args[0].startswith("http://127.0.0.1:")
    uv_run.assert_called_once()


def test_browser_plus_no_browser_does_not_open_browser():
    """Edge case: if both flags are passed (someone migrating scripts), the
    deprecation wins — --no-browser still suppresses the launch. Conservative
    safety bias: when in doubt, don't spawn a browser."""
    wb_open, uv_run = _run(["--browser", "--no-browser"])
    wb_open.assert_not_called()
    uv_run.assert_called_once()

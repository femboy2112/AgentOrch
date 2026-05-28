"""Dashboard launcher.

The dashboard is a dev/automation tool: it boots a uvicorn server on
127.0.0.1 and that's it. It does NOT auto-open a browser by default —
silently spawning a browser window on every invocation is a footgun
(the W3 dispatch reproduced this hilariously: a master-mode worker
smoke-testing the dashboard across 8 steps × multiple iterations spawned
enough Firefox windows to wedge an entire workstation overnight).

Pass ``--browser`` to opt in to opening the page in the default browser.
``--no-browser`` is accepted as a deprecated no-op so existing automation
(including ``python -m harness dashboard --no-browser``) keeps working.
"""
from __future__ import annotations

import argparse
import webbrowser


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m dashboard")
    parser.add_argument("--port", type=int, default=8765)
    # Opt-in: tools that drive the dashboard programmatically (the worker
    # smoke tests, CI, the harness's `dashboard` subcommand) get the safe
    # default of "no browser" automatically.
    parser.add_argument("--browser", action="store_true",
                        help="Open the dashboard in the default browser. "
                             "Off by default — pass this flag to opt in.")
    # Back-compat: --no-browser was the old opt-out when the default opened
    # the browser. Now the default IS no-browser, so this is a no-op; we keep
    # the flag so scripts that pass it don't get an argparse error.
    parser.add_argument("--no-browser", action="store_true",
                        help="(deprecated no-op; not opening a browser is "
                             "now the default)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("uvicorn is required to launch dashboard") from exc

    host = "127.0.0.1"
    if args.browser and not args.no_browser:
        webbrowser.open(f"http://{host}:{args.port}")

    uvicorn.run("dashboard.server:app", host=host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

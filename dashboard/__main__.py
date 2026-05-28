from __future__ import annotations

import argparse
import webbrowser


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m dashboard")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("uvicorn is required to launch dashboard") from exc

    host = "127.0.0.1"
    if not args.no_browser:
        webbrowser.open(f"http://{host}:{args.port}")

    uvicorn.run("dashboard.server:app", host=host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

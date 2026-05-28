# Dashboard Smoke (Phase 3)

Phase 3 smoke for the frontend shell was run against `python -m dashboard --no-browser` on `127.0.0.1:8765`: open `http://127.0.0.1:8765/#/dispatch`, submit a stub instruction in `direct` mode with `codex,agy,grok`, confirm `POST /api/dispatch` returns a `run_id`, and verify the embedded stream (Stay on page) plus `#/live` both show ordered `WorkerEvent` updates (`reasoning`, tool markers, usage, lifecycle/done) with reconnect-safe SSE behavior and watchdog budget preview populated from `/api/dispatch/budget`.

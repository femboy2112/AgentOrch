# AgentOrch Control Dashboard (v1)

Single-user localhost dashboard for dispatch control, live reasoning streams, and run replay.

## Launch

- Direct: `python -m dashboard`
- Harness entrypoint: `python -m harness dashboard`
- Headless/local CI smoke: `python -m dashboard --no-browser`
- Default bind: `127.0.0.1:8765`

## Core Pages

- `#/dispatch`: create dispatches with mode/chain/test settings and budget preview.
- `#/live`: monitor active runs with the hybrid `WorkerEvent` stream view.
- `#/runs`: filter and paginate historical runs.
- `#/runs/<id>`: tabs for `Stream | Prompt | Stdout | Stderr | Diff | Quality`.

## Wireframes

Dispatch:

```text
+-------------------------------------------------------------+
| Dispatch | Live | Runs                         [Light Mode] |
+-------------------------------------------------------------+
| Instruction                                                |
| [ mono textarea ]                                          |
| Mode [adversarial v]  Generator [codex,agy,grok]          |
| Critic [agy,codex,grok]  Test [python -m pytest -q]       |
| [x] web-search [ ] no-fallback [x] Stay on this page      |
| Budget: codex(...) agy(...) grok(...)                     |
| [ DISPATCH ]                                               |
+-------------------------------------------------------------+
| Embedded Stream (optional stay-on-page view)               |
+-------------------------------------------------------------+
```

Run Detail:

```text
+-------------------------------------------------------------+
| Run 20260527-225930-001                                    |
| mode | generator | critic | duration | success | files     |
+-------------------------------------------------------------+
| [Stream] [Prompt] [Stdout] [Stderr] [Diff] [Quality]       |
|                                                             |
| Shared stream renderer replay from runs/<id>/events.jsonl  |
| (reasoning merged, tool/usage markers inline)              |
+-------------------------------------------------------------+
```

## Smoke Notes

- Phase 3 smoke: run `python -m dashboard --no-browser` on `127.0.0.1:8765`, open `http://127.0.0.1:8765/#/dispatch`, submit a stub `direct` dispatch (`codex,agy,grok`), and verify ordered `WorkerEvent` updates in both Dispatch embedded stream and `#/live`.
- Phase 4 smoke: open `#/runs`, apply filters/search, load more, then open `#/runs/<id>` and exercise all tabs (`Stream`, `Prompt`, `Stdout`, `Stderr`, `Diff`, `Quality`), confirming Stream replay from `runs/<id>/events.jsonl`.
- Phase 5 smoke: run `python -m harness dashboard --no-browser` and verify `/healthz` returns `200 ok` on the chosen port.

## v2 Backlog (Explicitly Deferred)

- `/workers` per-provider model/usage cards
- `/calibration` budget table editor
- `/bench` cloud_eval/model_sweep/calibrate launcher
- `/settings` env/default chain controls

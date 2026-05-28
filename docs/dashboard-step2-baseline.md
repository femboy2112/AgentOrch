# Dashboard Step 2 Baseline (Pre-Implementation)

This file captures the repository baseline before dashboard implementation
starts. It is intentionally narrow: dashboard seams, event-model seams, and
protected Claude CLI surfaces from `docs/dashboard-design.md`.

## Dashboard modules (current)

- There is no `dashboard/` package yet.
- Existing operator entrypoints are:
  - `harness/cli.py` (`python -m harness do|runs|show`)
  - `harness/dispatch.py` (synchronous `dispatch(...)`)

## Event model and integration seams (current)

- `agy_orchestrator/core/agent.py` has no `event_callback` hook yet.
- `AgentInstance._stream_communicate(...)` already provides line-by-line
  streaming drainage and watchdog trip markers in stderr.
- `harness/dispatch.py` is the orchestration seam that will wire future
  dashboard event publishing while preserving existing artifacts under `runs/`.

## Claude CLI paths (current baseline)

- Normal Claude worker command is built in
  `agy_orchestrator/core/agents/claude_agent.py`.
- `_build_base_cmd()` currently uses:
  - `claude -p - --output-format json --dangerously-skip-permissions`
- Session parsing (`_extract_session_id`, `_extract_result_text`) currently
  assumes JSON output from the normal path.

## Protected surfaces for dashboard work

These are locked by spec and must not regress:

1. The existing normal Claude CLI path remains `--output-format json`.
2. `--output-format stream-json` is dashboard-only behavior, gated by the new
   dashboard dispatch/event-stream path (not the existing harness CLI path).
3. Default harness dispatch chains continue to exclude Claude:
   - generator: `codex,agy`
   - critic: `agy,codex`

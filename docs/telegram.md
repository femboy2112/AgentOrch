# Telegram build-progress bot

A lightweight, stdlib-only Telegram integration with two halves:

1. **Outbound notifier** — a dispatch streams concise, professional build-progress
   messages to whitelisted chats (`harness/telegram.py`, wired into
   `harness/dispatch.py`).
2. **Command daemon** — a long-poll bot that answers a small command set
   (`harness/telegram_bot.py`, run via `python -m harness.telegram_bot`).

Everything is best-effort and exception-isolated: a Telegram outage can never
block, slow, or fail a dispatch. No third-party dependencies (HTTP via
`urllib.request`).

## Environment variables

| Var | Purpose | Default |
| --- | --- | --- |
| `TELEGRAM_BOT_KEY` | Bot token. **Read from env only — never hardcoded.** | (required) |
| `AGY_TELEGRAM_USERS` | Path to the recipient whitelist JSON (a list of `{"id", "username", "last_chat_id"}`). Lives **outside** the repo. | `/home/leah/tgbot/data/users.json` |
| `AGY_TELEGRAM_VERBOSITY` | Default message verbosity. | `normal` |
| `AGY_TELEGRAM_STATE` | Path to the bot's persisted state file (e.g. default verbosity). Lives **outside** the repo. | `/home/leah/tgbot/data/bot_state.json` |

The whitelist tolerates a missing/garbage file (treated as empty). A chat id is
taken from `last_chat_id` when present, else `id`.

## Enabling notifications on a dispatch

```bash
python -m harness do "INSTRUCTION"                       # auto: on iff key + whitelist present
python -m harness do "INSTRUCTION" --telegram            # force on (warns + stays off if missing)
python -m harness do "INSTRUCTION" --no-telegram         # force off
python -m harness do "INSTRUCTION" --telegram-verbosity verbose
```

Auto-enable fires when `TELEGRAM_BOT_KEY` is set **and** the whitelist is
non-empty, unless `--no-telegram`. `--telegram` forces on; if the key or
whitelist is missing it prints a clear warning and stays off (non-fatal).

## Verbosity levels

Each level is a strict superset of the previous one:

| Level | Adds |
| --- | --- |
| `quiet` | dispatch start + dispatch finish only |
| `normal` | + step started/completed, + failures/stalls (verifier-fail / oom / stalled), + worker spin-ups, + adversarial rounds (draft/verdict/rotation), + plan/reconcile/fallback transitions |
| `verbose` | + ToT branch activity + finer per-iteration detail |
| `debug` | + heartbeats + per-call token usage |

### Rich Stream Output
The output uses a distinct visual language with glyphs for quick scanning:
- **Run Header Card**: At `dispatch_started`, showing mode and run ID.
- **Plan Tree**: At `plan.completed`, rendering the steps structure.
- **Step Cards**: Every step outputs `▸ Step 3/8` and `✅ Step 3/8 done` with exact verifier vs critic outcomes (`✅` vs `☑️`).
- **Adversarial Rounds**: Emits the `✍️ Draft`, `♻️ Verdict`, and `↪️ Rotation` vertical story so it is visible at the `normal` verbosity level.

A polished final-summary card (success/fail, confidence, duration, changed-file
count, token grand total) is always sent at the end. Raw stderr lines are never
forwarded.

## Running the command bot

```bash
python -m harness.telegram_bot
```

It long-polls `getUpdates`, enforces the whitelist by `message.from.id` (silently
ignoring anyone not listed), and shuts down cleanly on `Ctrl-C`. One bad update
or a failed send never crashes the loop.

### Command set

| Command | Action |
| --- | --- |
| `/start` | Greet and confirm this chat will receive build updates |
| `/help` | List commands |
| `/status` | Summarize the most recent run (`runs/<id>/meta.json`), or "in progress" |
| `/summary [latest\|<run_id>]` | Rich recap card from `meta.json` with outcome, mode, duration, confidence, and reconcile findings |
| `/files [latest\|<run_id>]` | Changed-file list with added/modified/deleted glyphs |
| `/why [latest\|<run_id>]` | Explain the verdict: verified/critic/stalled status and any full reconcile findings |
| `/runs [N=5]` | Compact list of the N most recent runs, showing confidence chips and relative age |
| `/verbosity [level]` | Show or set the persisted default verbosity |
| `/track [latest\|<run_id>\|all]` | Follow a live run's progress (default `latest`; `all` aliased as `/showliveall`) |
| `/untrack [<run_id>\|all]` | Stop following one run (bare `/untrack` = all) |

### Live-run tracking (TEMPORARY, cross-process)

The bot runs in a **separate process** from any `harness do`, so it cannot
subscribe to that run's in-memory `EventBus`. Instead `/track` **tails the
persisted on-disk stream** `runs/<id>/events.jsonl`: on every poll iteration the
daemon reads the lines appended since a per-run **byte cursor**, renders each
through the same `render_event` formatter at the chat's current verbosity
(prefixed with a short run tag so concurrent runs don't blur), and advances the
cursor so history is never re-sent.

- **Discovery** is a pure filesystem scan, so the bot can follow a run it never
  started: a run is *live* when it has an `events.jsonl` but no terminal
  `meta.json`. `latest` = the newest live run; `all` tracks every live run at
  once; you can track multiple runs concurrently.
- **Auto-untrack**: when a tracked run finishes (`meta.json` appears) the bot
  sends the polished final-summary card and stops tracking it.
- Tracking is **per-chat** and persisted in the bot state file (outside the
  repo). A missing/rotated/garbage `events.jsonl`, a vanished run dir, or a send
  failure is swallowed+logged per run and never crashes the daemon.

> This command is **explicitly temporary**: once the Phase 3 singleton makes the
> orchestrator resident in memory, any caller can subscribe to a live run
> directly and `/track` becomes obsolete.

## Upcoming & Planned Features (Fast-Follow)

The following UI components and integrations are planned but not yet implemented in the current build:

- **Pinned Live Status Card**: A single, in-place edited status card pinned at the top of the chat, tracking the active worker, real-time elapsed duration, ETA, and progress bar for the active build.
- **Inline Keyboards**: Inline-keyboard buttons and `callback_query` dispatch for interactive bot menus.
- **Notification Management**: `/mute`, `/watch`, quiet-hours, notification digest/batching.
- **Workflow Tools**: `/health`, `/tail`, `/diff`, per-event filters, multi-run labels, `/retry`/`/open-issue` handoff.

## Security notes

- The bot token is read from `TELEGRAM_BOT_KEY` and is never written to disk by
  this code.
- The whitelist and state files live outside the repo; `.gitignore` also blocks
  `bot_state.json`, `*.bot_state.json`, `.telegram_state.json`, and `tgbot/` as
  a backstop.
- All Telegram network I/O is best-effort and exception-isolated.

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
| `normal` | + step started/completed, + failures/stalls (verifier-fail / oom / stalled) |
| `verbose` | + adversarial iteration outcomes, fallback reroutes, plan/reconcile transitions |
| `debug` | + heartbeats + per-call token usage |

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
| `/status` | Summarize the most recent run (`runs/<id>/meta.json`), or "in progress" if it has no meta yet |
| `/runs [N=5]` | Compact list of the N most recent runs |
| `/verbosity [level]` | Show or set the persisted default verbosity |

## Security notes

- The bot token is read from `TELEGRAM_BOT_KEY` and is never written to disk by
  this code.
- The whitelist and state files live outside the repo; `.gitignore` also blocks
  `bot_state.json`, `*.bot_state.json`, `.telegram_state.json`, and `tgbot/` as
  a backstop.
- All Telegram network I/O is best-effort and exception-isolated.

# Telegram bot command reference

User-facing reference for driving AgentOrch builds and notifications from the
Telegram bot (`@AgentOrch_EchoBot`). This is the **command** reference; for the
architecture (EventBus sink, push notifier, embedded poller, singleton lock) see
the design doc at [`docs/telegram.md`](../telegram.md).

The bot is a long-poll daemon (`python -m harness.telegram_bot`) that reads
`getUpdates`, answers a small command set, and can also follow a live build's
on-disk event stream. The same command surface is served *during* a dispatch by
the embedded poller, so commands work whether or not a standalone daemon is
running.

## Auth / whitelist model

Every incoming update — text message, edited message, and inline-button
`callback_query` — is gated by `message.from.id` (resp. `callback_query.from.id`)
against the whitelist file. A sender whose user id is **not** in the whitelist is
**silently dropped**: no reply, and for callbacks not even a button-spinner
acknowledgement. The whitelist is the single trust boundary for everything the
bot can do, including launching builds and the `--test-cmd` shell channel
(below).

State, whitelist, and the shared id-log live **outside the repo**, by default
under `/home/leah/tgbot/data/`:

| File | Purpose | Default path | Override env |
| --- | --- | --- | --- |
| `users.json` | Whitelist (allowed user/chat ids) | `/home/leah/tgbot/data/users.json` | `AGY_TELEGRAM_USERS` |
| `bot_state.json` | Persisted verbosity, per-chat mute/quiet prefs, `/track` cursors | `/home/leah/tgbot/data/bot_state.json` | `AGY_TELEGRAM_STATE` |
| `sent_messages.jsonl` | Cross-process sent/seen message-id log used by `/clear` | next to the state file | (follows `AGY_TELEGRAM_STATE`) |
| `telegram_poller.lock` | `flock` singleton guard so only one `getUpdates` poller runs | next to the state file | (follows `AGY_TELEGRAM_STATE`) |

The bot token is read from the `TELEGRAM_BOT_KEY` environment variable and is
never hardcoded or stored in the repo.

## Command summary

`ACTION` = mutates state, drives/kills a build, or deletes messages.
`READ` = read-only text (or button) reply.

### Registered commands (shown in the Telegram `/` menu)

| Command | Args | Purpose | Kind |
| --- | --- | --- | --- |
| `/build` | `<instruction> [--mode m] [--test-cmd c] [--mission-critical] [--web-search]` | Launch a build and auto-track it | ACTION |
| `/cancel` | `[latest \| <run_id>]` | Abort a running dispatch (SIGTERM) | ACTION |
| `/retry` | `[latest \| <run_id>]` | Re-dispatch a finished run's instruction | ACTION |
| `/run` | `[latest \| <id>] [summary \| why \| files \| diff]` | Recap facets for a run (default `summary`) | READ |
| `/notify` | `[show \| mute … \| watch \| quiet … \| verbosity <lvl>]` | One place for mute/quiet/verbosity controls | ACTION (except `show`) |
| `/status` | — | Most recent run (or "in progress"), with inline buttons | READ |
| `/runs` | `[N]` | Recent runs (default 5, capped at 20) | READ |
| `/track` | `[latest \| <run_id> \| all]` | Follow a live run's stream (default `latest`) | ACTION |
| `/health` | — | Poller status, live-run count, last outcome, recent signals | READ |
| `/tail` | `[N] [run]` | Last N rendered events (default 10, capped at 30) | READ |
| `/clear` | — | Delete this chat's bot-known messages (<48h old) | ACTION |
| `/help` | — | List commands | READ |

### Back-compat aliases (work but not in the menu)

| Alias | Equivalent to | Kind |
| --- | --- | --- |
| `/summary [id]` | `/run [id] summary` | READ |
| `/files [id]` | `/run [id] files` | READ |
| `/why [id]` | `/run [id] why` | READ |
| `/diff [id]` | `/run [id] diff` | READ |
| `/mute [30m\|2h\|on]` | `/notify mute …` | ACTION |
| `/watch` | `/notify watch` | ACTION |
| `/quiet [HH:MM-HH:MM\|off]` | `/notify quiet …` | ACTION |
| `/verbosity [lvl]` | `/notify verbosity …` | ACTION |
| `/untrack [<run_id>\|all]` | (no consolidated form) stop following | ACTION |
| `/showliveall` | `/track all` | ACTION |
| `/start` | greet + confirm this chat receives updates, then `/help` | READ |

`/cmd@BotName` forms are tolerated (the `@BotName` suffix is stripped).

## Action commands

### `/build` — launch a build from the phone

```
/build <instruction> [--mode m] [--test-cmd c] [--mission-critical] [--web-search]
```

Spawns a detached `python -m harness do <instruction> [flags]` in its own
process group (so `/cancel` can kill the whole tree), discovers the new
`runs/<id>/` directory within a bounded ~5s poll, and auto-tracks it — you start
receiving its live stream immediately.

The instruction is everything before the first recognised flag and is passed as a
**single argv element** (never through a shell), so shell metacharacters inside it
are inert. Only the allowlisted flags below are honoured; anything else (or a bad
`--mode` value) is silently dropped rather than passed through.

| Flag | Value | Notes |
| --- | --- | --- |
| `--mode` | `direct`, `adversarial`, `feedback`, `cascade`, `master`, `pat`, `vote`, `auto` | An unknown mode is dropped |
| `--test-cmd` | shell command string | Quality gate; multi-word values are preserved |
| `--mission-critical` | (boolean) | — |
| `--web-search` | (boolean) | Enable codex web search |

> **`--test-cmd` is a deliberate phone → host shell channel.** Its value is, by
> design, run through `/bin/sh -c` by the quality verifier (mirroring the CLI's
> own `--test-cmd` contract). This is intended operator functionality, not an
> injection hole — the only gate on it is the `from.id` whitelist, the same
> boundary that authorises launching builds at all.

```
/build add a --version flag to both CLIs --mode pat --test-cmd "pytest -q"
```

### `/cancel` — abort a running dispatch

```
/cancel [latest | <run_id>]
```

Reads the pid from the target run's `runs/<id>/run.pid` and sends `SIGTERM`
(killpg of the process group first, then a single-process kill as a fallback).
It only ever signals a pid read from a real `run.pid` — never an arbitrary pid.

Safety gates:
- **Terminal-state gate**: a run whose `meta.json` exists has finished; `/cancel`
  refuses it (the on-disk pid may have been OS-recycled). The bare `latest` form
  is already filtered to live runs only.
- **Self-signal gate**: if the recorded pid resolves to the bot's own process or
  process group, `/cancel` refuses rather than risk SIGTERM-ing the daemon.

### `/retry` — re-run a finished run

```
/retry [latest | <run_id>]
```

Reads the target run's `prompt.txt` and re-dispatches that instruction, carrying
forward the original `--mode` (when it was a recognised mode). Like `/build`, the
new run is spawned detached and auto-tracked. Empty/missing prompts are reported,
not silently retried.

### `/clear` — delete this chat's bot messages

```
/clear
```

Telegram has **no wipe-history API**. A bot can only delete *individual* messages
that are **less than 48h old** and whose `message_id` it knows. Because build
chatter is sent by a different process (the dispatch's notifier) than the daemon,
both append every sent/seen `(chat_id, message_id, ts)` to one shared id-log
(`sent_messages.jsonl`). `/clear` reads the in-window ids for the chat,
bulk-deletes them (falling back to per-message deletes so one un-deletable id
can't block the rest), purges the log, and sends an honest confirmation
including how many couldn't be removed.

Limits, stated plainly: only messages the bot **saw while running** and that are
**<48h old** can be cleared. The retention cutoff is 47h (safely under
Telegram's 48h boundary). The confirmation message itself is re-tracked, so the
next `/clear` removes it too.

## Consolidated verbs

Two umbrella commands fold the older standalone names; the aliases still work.

### `/run` — run recap facets

```
/run [latest | <id>] [summary | why | files | diff]
```

The id and the facet may appear in either order; both default sensibly
(`latest` + `summary`).

| Facet | Shows |
| --- | --- |
| `summary` | Success/fail, mode, confidence, duration, file counts, quality (verifier/critic/iters), reconcile verdict, total tokens |
| `why` | Quality note, verified/critic/stalled status, reconcile verdict + findings |
| `files` | List of changed files (➕/➖/📝, capped at 30) |
| `diff` | `--stat`-style header + first ~60 lines of `changed-files.diff` in a `<pre>` block |

### `/notify` — notification controls

```
/notify [show | mute 30m|2h|on | watch | quiet HH:MM-HH:MM|off | verbosity <lvl>]
```

| Subcommand | Effect |
| --- | --- |
| `show` (default) | One-card view of the chat's mute + quiet-hours + verbosity state (READ) |
| `mute 30m` / `2h` / `90s` / `45` / `on` | Silence progress chatter (bare number = minutes; `on` = indefinite). Failures + the final summary still come through |
| `watch` | Clear any mute, resume progress updates |
| `quiet HH:MM-HH:MM` / `off` | Do-not-disturb window; progress is held during it, failures + final summary still pass |
| `verbosity <lvl>` | Show or set the persisted default verbosity (see below) |

`mute`/`watch`/`quiet` are **per-chat**; `verbosity` is the **global** default.

## Verbosity levels

Default is `normal`. The persisted default lives in the bot state file (settable
via `/notify verbosity` or `/verbosity`). The two readers fall back differently
when nothing is persisted: the dispatch-side push notifier falls back to the
`AGY_TELEGRAM_VERBOSITY` environment variable (`resolve_verbosity`), while the
bot daemon's own `/tail` rendering falls back directly to `normal`
(`get_verbosity`) and never consults that env var. The push notifier re-reads the
persisted level per event, so changing it takes effect on a *running* build.
Each level is a superset of the one before it.

| Level | Surfaces |
| --- | --- |
| `quiet` | Dispatch start + the end-of-run summary card only |
| `normal` | + per-step started/completed, failures/stalls, **worker spin-ups**, **adversarial rounds** (draft / verdict / generator rotation), plan / reconcile / fallback transitions |
| `verbose` | + Tree-of-Thought branch activity and finer per-iteration detail |
| `debug` | + heartbeats and per-call token usage |

## Message format — Card + expandable detail

All build messages render as **`parse_mode=HTML`** and follow one layout rule: a
compact, emoji-led **card** carries the at-a-glance state on the surface, and any
long or secondary detail is tucked into a Telegram **`<blockquote expandable>`** —
a block that collapses by default and expands when the operator taps it. This
keeps each message short and scannable while still being more informative on tap.
Three render sites use it:

- **Live status card** (`render_status_card`, the pinned edit-in-place card): the
  surface always shows status (`⚡ Building · <id>`), the goal, the progress bar +
  step reference (`Step 2/4`), the phase · worker · current-step line, and the
  `⏱ elapsed · ETA` line. The recent-activity line and the per-step "steps so far"
  progress are tucked into the expandable. The expandable is omitted entirely when
  there is no secondary detail yet (no empty block is ever sent).
- **Plan-ready event** (`render_event`, plan phase): the `📋 Plan ready · N steps`
  headline stays on the surface; the full numbered step list (tree connectors,
  capped with a `… +K more` tail for very large plans) is tucked into the
  expandable, so a 12-step plan collapses to a one-line headline.
- **End-of-run summary card** (`_summary_card`): the surface keeps the outcome
  headline (`✅ Build verified` / `☑️ Build complete` / `🛑 Build failed`), the
  goal, the `N/N steps · status · duration` line, and the tokens · mode · run
  footer. The changed-files list **and** the `<pre>` per-step recap are tucked
  into the expandable (`<pre>` nests inside `<blockquote expandable>`), so a
  many-file / many-step run stays a tidy card.

Only the allowed Telegram HTML tags are emitted (`<b> <i> <u> <s> <code> <pre>
<a> <blockquote> <blockquote expandable> <tg-spoiler>`); all dynamic text (goal,
titles, file paths, run ids) is HTML-escaped, and blockquotes are never nested.

## Inline callback buttons

`/status` (and the per-run live summary card) attaches an inline keyboard:

```
[ 📂 Files ]  [ ❓ Why ]  [ 📊 Diff ]
```

A mission-critical approve/reject gate card is **defined but not yet wired**: the
keyboard builder (`build_gate_keyboard`), the `approve`/`reject` callback verbs,
and the `_record_gate_decision` handler all exist, but no current dispatch or
notifier path ever attaches this keyboard to a sent message, so these buttons are
**not shown today**:

```
[ ✅ Approve ]  [ ⛔ Reject ]    # seam only — never emitted by any code path
```

Each button carries `callback_data` of the form `verb:<short_run_id>` (the last 6
chars of the run id, kept under Telegram's 64-byte cap). Taps reuse the same
`from.id` whitelist as text commands — a non-whitelisted tap is dropped with no
reply and no spinner clear; a whitelisted tap is always answered so the button
spinner clears even on a no-op.

- `Files` / `Why` / `Diff` route to the same read-only helpers as the `/run`
  facets.
- `Approve` / `Reject` callbacks, *if* ever delivered, write a
  `runs/<id>/gate_decision.json` sidecar (server-stamped timestamp). This is an
  **unwired seam**: the gate keyboard is never emitted, and even if a decision
  were recorded the default dispatch flow does **not** block on or consume that
  sidecar.

## Live-run tracking

The bot is a separate process from a running `harness do`, so it follows a build
by tailing that run's on-disk `runs/<id>/events.jsonl`, advancing a per-run byte
cursor so only new lines are emitted.

```
/track [latest | <run_id> | all]      # default: latest
/untrack [<run_id> | all]             # default: all
```

- `/track latest` follows the newest live run; `/track all` follows every live
  run; `/track <run_id>` follows a specific run (must exist and be live).
- Tracked events are rendered at the chat's current verbosity, prefixed with a
  short run tag so concurrent runs stay distinguishable. Muted / quiet-hours
  chats drop progress lines but still get failures.
- When a tracked run finishes (its `meta.json` appears) a final summary card is
  sent and the run is **auto-untracked**.
- `/untrack` (bare) stops following everything; `/untrack <run_id>` (full id or
  short tag) stops one. `/showliveall` is an alias for `/track all`.

> Live `/track` tailing is explicitly temporary: once the orchestrator becomes a
> resident singleton, callers subscribe to a live run directly and `/track`
> becomes obsolete. See [`docs/telegram.md`](../telegram.md).

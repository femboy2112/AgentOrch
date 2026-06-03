# Telegram Overhaul — Fast-Follow Build Spec

Finishes the two-pillar overhaul (commits d541870, 18b0b6a) by closing the honest
gaps and building the deferred usability features. Same invariants as
`docs/telegram-overhaul-spec.md` §0 (stdlib-only, exception-isolation, off-loop
delivery, HTML-escape, spam-gate, dynamic /verbosity, security/hermetic tests).

## MANDATORY build method (lesson from the no-op disaster)
The first big-bang master build NO-OP'd and false-greened (verifier ran on
unchanged files). Every sub-build here MUST:
1. Ship with a HAND-WRITTEN anti-no-op contract test that FAILS on the current
   code (assert the NEW surface/behavior), placed so `-k telegram` collects it
   (filename must contain "telegram").
2. Build in ADVERSARIAL mode (holistic verifier — a no-op can't pass) OR a
   Workflow with adversarial verify; gate = `timeout 300 .venv/bin/python -m
   pytest -x -q -p no:cacheprovider -k telegram tests/`.
3. `--protect-paths "agy_orchestrator/**,dashboard/**"` (+ already-done files when
   a sub-build shouldn't touch them).
4. SELF-AUDIT after: render the actual output, `grep -n '\\\\n'` for literal
   newline bugs, full suite, protected-paths + secret scan, remove worker
   scratch/cache detritus, THEN commit. Never trust verified=True.

---

## Part A — honest gaps (finish the pillars; smaller)

### G1. Goal in the header + status card
dispatch_started carries no instruction, so the header card omits the goal.
Thread it: `dispatch_async` passes `instruction` (truncated) into
`_build_telegram_notifier` → `TelegramNotifier(instruction=...)`; the notifier
stores it on `BuildState.goal` and renders it as the bold lead of the header card
and the `<i>` title line of the status/terminal card.
- Touches `harness/dispatch.py` (pass instruction) + `harness/telegram.py`.
- Acceptance: a notifier built with `instruction="Add OAuth device flow"` renders
  that text in the header card and the status card; absent instruction → no crash,
  no empty bold line.

### G2. Step-done line carries duration · rounds
`render_event` is stateless so the step-completed line shows only the outcome.
The notifier owns `BuildState` (per-step durations + iterations). Annotate the
step-completed line via the existing `context` channel (or render it from state in
`__call__`): `✅ Step 3/8 done · <i>title</i>  2m14s · 2 rounds · verified`.
- Touches `harness/telegram.py` only.
- Acceptance: after a step with a known start/end ts + 2 adversarial iterations,
  the rendered step-done line contains the duration and the round count.

### G3. Rich multi-line end-of-run summary card (spec §8)
`finished(meta)` builds the rich card from `BuildState` + meta: headline
(✅ verified / ☑️ complete / 🛑 failed), goal, `N/N steps · status · duration`,
a changed-files block (first ~8 + "… +K more"), and a `<pre>` per-step recap
(`i ✅ title  Xm Ys`, columns aligned) from tracked step durations/outcomes, plus
tokens + mode.
- Touches `harness/telegram.py` only.
- Acceptance: a finished() with 3 tracked steps + changed_files renders a
  multi-line card containing the file count, a `<pre>` recap with each step, and
  the verified/approved/failed headline; REAL newlines (no literal `\n`).

---

## Part B — usability features (new capabilities; larger)

### F1. Inline-keyboard buttons + callback_query dispatch (foundation)
Attach `reply_markup` buttons to the end-of-run card and `/status`:
**[📂 Files] [❓ Why] [📊 Diff]**. Tapping fires a `callback_query`.
- `TelegramClient`: `answer_callback_query(id, text="")`, and `reply_markup`
  passthrough on `send_message` (+ already-added `edit_message_text`). All thin
  `_post` wrappers, best-effort.
- `telegram_bot._process_update`: NEW branch for `update.get("callback_query")` —
  whitelist-gate by `callback_query.from.id`, parse `callback_query.data`
  (`"verb:short_run_id"`, ≤64 bytes), route to the existing `summarize_run` /
  `/why` / `/files` helpers, reply, then ALWAYS `answer_callback_query` (clears the
  client spinner).
- Acceptance (hermetic, fake callback dict): a `why:<rid>` callback from a
  whitelisted user → a reply containing the verdict + an answerCallbackQuery call;
  a non-whitelisted callback → ignored, no reply; data >64 bytes / unknown verb →
  safe no-op.

### F2. /mute, /watch, quiet-hours (highest-felt "stop buzzing me")
Per-chat prefs under `state["chats"][str(chat_id)]` = `{mute_until, quiet_window}`.
- `/mute [30m|2h|on]`, `/watch` (unmute), `/quiet HH:MM-HH:MM` (DND), `/quiet off`.
- Enforcement: a cheap `_suppressed(chat, event_kind, state)` consulted in BOTH
  the dispatch-side `TelegramNotifier._broadcast` AND the bot's
  `tail_tracked._send_tail` (reads bot_state.json — same per-event pattern as the
  live /verbosity reader). Policy: while muted/in-quiet-window, drop progress
  chatter but ALWAYS deliver failures + the final summary card.
- Acceptance: a muted chat suppresses a step-started message but NOT a
  build-failed/summary message; `/quiet 23:00-07:00` parses + persists; a window
  parser rejects garbage without raising; the suppression read never raises.

### F3. /health, /tail, /diff (read-only commands)
- `/health` — poller alive (flock test, non-destructive), live-run count, last
  outcome, recent reroute/usage-wall scraped from newest events.jsonl tails.
- `/tail [n=10] [run]` — last N rendered events via `render_event` (bounded read).
- `/diff [run]` — `--stat` summary + first ~60 lines of `changed-files.diff` in
  `<pre>`, hard 4096 cap + escape.
- Acceptance: each returns non-empty content from a fixture run; length-guarded;
  `/health` flock test does not steal the lock.

### Deferred AGAIN (do NOT build here): notification digest/batching (hot-path,
risk), per-event filters, multi-run labels, /retry + /open-issue handoff
(Phase-3 singleton owns auto-dispatch). Note them in docs/telegram.md.

---

## Suggested execution order (each its own contract + adversarial build + audit + commit)
1. **Part A (G1+G2+G3)** — one build; finishes the pillars; lowest risk.
2. **F1** — buttons/callbacks; lands the reusable callback machinery.
3. **F2** — mute/quiet; highest-felt.
4. **F3** — health/tail/diff; read-only.
Push to main after each green sub-build (or batch — operator's call).

## Acceptance (whole)
`pytest -k telegram` green incl. new contracts; full suite green; protected paths
untouched; rendered output visually verified (no literal `\n`); hermetic tests;
no secrets/real-ids/junk; docs/telegram.md updated; live-demo to phone optional.

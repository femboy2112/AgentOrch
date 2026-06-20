# Telegram Bot Overhaul — Build Spec

**Goal.** Turn AgentOrch's Telegram output from terse one-liners into a **professional,
beautiful, coherent build-progress stream** that always conveys **build state** and the
**reference point within the build** (which step of how many, which phase, which adversarial
round, elapsed, ETA, what just happened, what's next) — plus a set of **usability commands**.

This is a render-and-UX upgrade. It must NOT change orchestrator behavior, and Telegram must
remain unable to ever block, slow, or fail a dispatch.

---

## 0. Non-negotiable invariants (preserve exactly)

1. **stdlib only** — HTTP via `urllib` (no third-party deps).
2. **Exception-isolation** — every render/notify/client method catches all exceptions and never
   raises into a dispatch. A Telegram outage is a no-op.
3. **Off-event-loop delivery** — `TelegramNotifier` delivers on its background daemon thread via
   the existing bounded queue (drop-oldest under pressure). New edits/sends use the SAME queue.
4. **HTML parse_mode** — allowed tags only: `b i u s code pre a`. Escape ALL dynamic text via `_e`.
5. **Spam-gate** — `agent_started` renders once per canonical CLI call (dict `detail`); a non-empty
   string `detail` (per-turn adapter noise) stays suppressed.
6. **Dynamic `/verbosity`** — the notifier keeps re-reading the persisted level per event
   (`_effective_verbosity` / `load_persisted_verbosity`). Don't regress this.
7. **Security** — token from `TELEGRAM_BOT_KEY` env, NEVER hardcoded/persisted. Whitelist + state
   live OUTSIDE the repo (`~/tgbot/data/{users,bot_state}.json`, env overridable). NEVER
   write a real chat id into code or tests. All new tests hermetic: fake token, fake ids
   (`FAKE_CHAT_ID = 444555666`), `_post` monkeypatched, state pinned to `tmp_path`.
8. **Pure `render_event`** stays pure (no I/O, no global state). New `render_status_card(state)` is
   also pure. The notifier owns all mutable state and all I/O.
9. **Back-compat** — keep the existing verbosity ladder semantics; existing gating tests must be
   updated intentionally (not broken) when counts change. The discrete stream stays the default;
   the new pinned card is additive.

---

## 1. Event vocabulary (the data you render from)

Every event is normalized by `dashboard/event_bus.py::_normalize_worker_event` with TOP-LEVEL:
`ts` (float epoch), `run_id`, `worker`, `model`, `effort`, `branch`, `kind`, `text`, `data`.

`kind ∈ {lifecycle, reasoning, message, tool_call, tool_result, usage, stderr, watchdog, heartbeat}`.

**lifecycle / `data.event`:**
- `dispatch_started` → `data.detail = {mode, generator_chain, critic_chain}` (NO step total).
- `dispatch_finished` → render returns None; the summary card owns end-of-run.
- `agent_started` → top-level `worker/model/effort`; `data.detail` is `{}` or `{"role": …}`
  (canonical, render it) OR a non-empty string (per-turn noise, suppress).
  Roles: `draft · critic · summary · compact · plan · tot · tot-judge · reconcile`.
- `orchestration_transition` → `data.orchestration = {workflow, phase, action, step_index,
  step_total, step_title, iteration, iteration_total, outcome, verified, approved, to_worker,
  generator_rotations, model, effort, step_titles}` (fields present per phase only).

**Orchestration phases / actions:**
- `step`: `started` (step_index, step_total, step_title) · `completed` (+ outcome, verified, approved).
- `adversarial`: `iteration_started` (iteration, iteration_total, model) · `iteration_completed`
  (+ outcome, verified, approved) · `generator_rotation` (to_worker).
- `plan`: `completed` (step_total, step_titles).
- `reconcile`: `trace_started/trace_completed/ablation_measured/ablation_mismatch`.
- `tot`: `branch_selected` (selected_branch, branch_total, selector, score) — verbose tier.

**Outcomes:** `verified` (programmatic verifier passed) · `approved` (critic approved, no/!verifier)
· `continue` (revised — critic sent back) · `stalled` (max iters) · `verifier_timeout` /
`verifier_resource_exceeded` / `verifier_infra_failed` (INFRA, not a code defect — distinct glyph).

**heartbeat** → `data = {run_id, step, elapsed_s, since_progress_s, free_mem_mb}` (~30s; default on).
This is the off-loop liveness tick that lets a long silent step's elapsed/ETA advance.

**usage** → `data = {usage_kind:"call", input_tokens, output_tokens, cache_read_tokens,
total_tokens, token_source}` (debug tier).

`meta.json` (for summary card + commands): `success, mode, duration_s, changed_files/added/
modified/deleted, quality{confidence, verified, critic_approved, stalled, iterations_used, note},
reconciliation{verdict, disposition, findings[{name, classification, sub_kind}]}, tokens{grand_total,
per_worker}, error, run_outcome`. Each run dir also has `changed-files.diff`, `run.pid`, `events.jsonl`.

---

## 2. Visual language (the grammar)

- **Separator** ` · ` (space middot space), always.
- **One leading glyph** per stream line = the category. Glyph alphabet:
  `🟢` started · `📋` plan · `▸` step started · `✅` verified/approved-verdict/step-ok ·
  `☑️` completed-unverified · `♻️` revised · `✍️` draft · `🤖` spin-up · `⏹` stalled ·
  `⚠️` verifier infra-fail · `❌` step failed · `↪️` reroute/rotation · `🔀` reconcile ·
  `🏁` success card · `🛑` failure card · `⏱` timing (never leading).
- **Subject** = one `<b>…</b>` immediately after the glyph. **Human title** = `<i>…</i>`.
  **Machine token** (run id, model, effort, mode) = `<code>…</code>`.
- **Run tag** — every stream line ends ` · <code>{last6 of run_id}</code>` so interleaved runs
  in one chat stay attributable.
- **Reference point** — every line states position: `Step X/N`, `iter k/N`, or both.
- **Multi-line** is reserved for: the run-header card, the plan tree, the pinned status card, and
  the end-of-run summary card. `<pre>` is reserved for the per-step recap (needs column alignment).
- **✅ vs ☑️ is load-bearing:** ✅ = a programmatic verifier passed; ☑️ = critic-approved only,
  nothing run. The output must never overstate what was checked.

### 2.1 Stream message mockups (literal HTML)

Run header (on `dispatch_started`, enriched once `plan.completed` gives step_total):
```
🟢 <b>Build started</b>

<b>{goal, ≤80 chars}</b>
<i>orchestrating {worker · worker · …}</i>

📋 <b>{N} steps</b> planned · mode <code>{mode}</code>
⏱ started {HH:MM} · run <code>{short}</code>
```
For modes with no planner (direct/feedback) omit the 📋 line.

Plan tree (on `plan.completed`; cap ~12 rows, then `└ … +K more`):
```
📋 <b>Plan ready</b> · <b>{N} steps</b> · <code>{short}</code>
├ 1 · {title}
├ 2 · {title}
└ {N} · {title}
```

Step started / done (done line carries duration · rounds · outcome):
```
▸ <b>Step 3/8</b> · <i>{title}</i> · <code>{short}</code>

✅ <b>Step 3/8 done</b> · <i>{title}</i>
   2m14s · 2 rounds · verified · <code>{short}</code>
```
`☑️` when outcome=approved (no verifier); `❌` when stalled/failed.

Adversarial round lines (a step reads as a clean ✍️→verdict→✍️→✅ vertical story):
```
✍️ <b>Draft</b> · iter 1/5 · <code>{model}</code> · <i>Step 3/8</i>
♻️ <b>Verdict</b> · revised · iter 1/5 · <i>Step 3/8</i>
✅ <b>Verdict</b> · approved · iter 2/5 · <i>Step 3/8</i>
⏹ <b>Verdict</b> · stalled at iter 5/5 · <i>Step 6/8</i>
⚠️ <b>Verdict</b> · 💥 verifier OOM · iter 2/5 · <i>Step 4/8</i>
   infra fault — not a code defect · <code>{short}</code>
```

Worker spin-ups (role → verb; keep concise):
```
🤖 <b>codex</b> spun up · drafting · <code>{model · effort}</code> · <i>Step 3/8 · Adversarial</i>
🤖 <b>codex</b> spun up · summarizing · <i>Step 3/8</i>
🤖 <b>codex</b> spun up · planning · <i>Plan</i>
```

Rotation / reroute / reconcile:
```
↪️ <b>Rotation</b> → <b>agy</b> · iter 3/5 · <i>Step 6/8 · Adversarial</i>
↪️ <b>Reroute</b> · codex → agy · <i>usage wall</i> · <code>{short}</code>
🔀 <b>Reconcile</b> · {action} · <i>integration check</i> · <code>{short}</code>
```

### 2.2 End-of-run summary card (rich; 3 variants)

Success-verified (per-step recap uses `<pre>` for column alignment):
```
🏁 <b>Build verified</b> · <code>{short}</code>

<b>{goal}</b>
✅ {N}/{N} steps · verified · {duration}

<b>Changed</b> · {n} files
<code>{path}
{path}
… +{k} more</code>

<b>Recap</b>
<pre>1 ✅ {title}      0m44s
2 ✅ {title} 2rds 2m14s
…</pre>

{tokens:,} tokens · mode <code>{mode}</code>
```
Success-unverified → headline `☑️ <b>Build complete</b>`, status `approved (no test gate)`.
Failure → `🛑 <b>Build failed</b>`, names where it died (`stalled at step 6/8`), how far it got
(reached k/N, files touched), last terminal event. Infra failure → `⚠️ Build halted · verifier OOM`.

---

## 3. Live pinned status card (the centerpiece — "reference point within the build")

A single message, sent + **pinned** at build start, **edited in place** as state changes. It is the
*dashboard*; the discrete stream is the *journal*. Fixed-shape so the eye lands on the same row:

```
⚡ <b>Building</b> · <code>{short}</code>
<code>{10-cell bar}</code>  <b>Step {i}/{N}</b>
<i>{current step title}</i>

{phase glyph} {Phase} · {state} · <b>iter {k}/{n}</b>
⏱ {elapsed} elapsed · {~Nm left | estimating…}
└ last: {mirror of most recent stream line}
```
On finish it freezes to a terminal card (🏁/🛑 + totals).

### 3.1 BuildState (sink-side state machine, in-memory on the notifier)

Maintain a `BuildState` updated in `TelegramNotifier.__call__` BEFORE render, fully
exception-isolated, keyed off event `ts` (NOT `time.time()`, so replay == live). Track at minimum:
- identity/lifecycle: `run_id, mode, generator_chain, critic_chain, run_start_ts, run_end_ts,
  finished, success, failed_reason`.
- geometry (late-bound — nothing assumed until the planner speaks): `step_total (None until
  plan.completed or first step event), plan_known, is_graph, resume_base_index`.
- position: `cur_step_index, cur_step_title, cur_phase, cur_step_start_ts, cur_iter, cur_iter_total,
  cur_iter_start_ts`.
- worker focus: `active_worker, active_role, active_worker_start_ts`.
- history (for ETA): `completed_steps:set, step_durations:list, _step_started_at:dict`.
- tallies: `n_steps_done, n_verified, n_approved, n_revised, n_rotations, n_infra_fail, n_stalled,
  reconcile_status, compactions`.
- liveness: `last_event_ts, last_progress_ts, last_summary, last_heartbeat_ts, since_progress_s,
  free_mem_mb`.
- render bookkeeping: `pinned_message_id, last_render_ts (monotonic), last_render_signature, dirty`.

**Update rules** (abbreviated; see event vocab §1):
- `dispatch_started` → seed identity, `run_start_ts`.
- `agent_started` (canonical) → set active worker/role + ts; does NOT move geometry.
- `plan.completed` → `step_total`, `plan_known=True` (+ `is_graph` if the plan carries graph/nodes).
- `step.started` → adopt `step_total` if unset; **graph detect**: `idx != (cur or 0)+1` and cur set
  → `is_graph=True`; **resume detect**: first step with `idx>1` → seed `completed_steps={1..idx-1}`
  as done-but-unverified; set cur_* + `_step_started_at[idx]=ts`; reset iter fields.
- `step.completed` → `completed_steps.add(idx)`; append `ts - _step_started_at[idx]` to durations;
  tally outcome (verified/approved/infra/stalled).
- `adversarial.*` → update `cur_iter/total/start`; tally revised/rotation/infra/stalled.
- `reconcile.*` → `reconcile_status`. `tot/fallback` → `last_summary` only.
- `heartbeat` → advance `last_heartbeat_ts/since_progress_s/free_mem_mb`; mark `dirty` (liveness
  tick) but NOT a progress event.
- **compaction guard:** `role=="compact"` spin-ups must NOT reset geometry — only `phase=="step"`
  mutates step position/durations.

### 3.2 Derived metrics
- **Elapsed** = `last_event_ts - run_start_ts` (prefer heartbeat `elapsed_s` when fresher). Per-step
  = `last_event_ts - cur_step_start_ts`.
- **Progress bar** (10 cells) when `step_total` known: filled = `round(done/N*10)`, current cell
  distinct. Unknown → indeterminate marquee (cell at `cur_step_index % 10`), label `Step k (of ?)`
  / `Planning…` — never a fake percent.
- **ETA** from `step_durations` ONLY, honest + bracketed: none until ≥1 step completes
  (`estimating…`); center = median (≥3 samples) else EMA(α=0.5); `R = N - done`; subtract in-flight
  `min(cur_elapsed, d̂)`; present a range `~6–11m (≈8m)` widening with fewer samples; always `~`.
  When `is_graph`, switch to makespan-rate mode (`rate = elapsed/done`, wide band, `(parallel)`).
- **Stall flag** — if `since_progress_s` (or `last_event_ts - last_progress_ts`) exceeds
  `2×heartbeat_interval`, show `⚠ no progress {dur}` so a wedge is visible before the watchdog aborts.
- **Spinner** — indeterminate phases rotate a glyph off `int(last_event_ts)%4` so the card visibly
  breathes on each heartbeat without implying quantitative progress.

### 3.3 Edit / throttle policy
- First meaningful event → `sendMessage` the card, store `pinned_message_id` from `result.message_id`,
  optional `pinChatMessage`.
- Subsequent → `editMessageText` on `pinned_message_id`. Coalesce: `MIN_EDIT_INTERVAL ≈ 1.2s`;
  skip if `signature == last_render_signature` (signature = rounded tuple of render inputs; elapsed/ETA
  rounded ~5s so jitter doesn't trigger edits). Burst → ONE trailing edit. **Force-edit** (still
  ≥1.2s apart) on high-salience transitions: step started/completed, verdict, rotation, failure,
  finished. Heartbeat edits only if `>~5s` since last AND signature changed.
- All edits ride the existing background queue (ordering preserved; never on the event loop).
- `finished()` → final edit to terminal card, optional unpin, then `flush(timeout)`.
- "message is not modified" 400s are swallowed by best-effort `_post` + the signature guard.
- **Opt-in / safe default:** the pinned card is additive. Gate it behind a setting (e.g.
  `state["pinned_live"]`, default ON at `normal`+; at `quiet` the card can be the ONLY message).
  If `editMessageText`/message_id capture fails, degrade silently to the discrete stream.

---

## 4. New `TelegramClient` methods (thin `_post` wrappers, best-effort, return None on failure)
- `edit_message_text(chat_id, message_id, text, *, reply_markup=None)` → `editMessageText`.
- `pin_chat_message(chat_id, message_id)` / `unpin_chat_message(chat_id, message_id)`.
- `send_message(...)` already exists; capture `result["message_id"]` from its return for pin/edit.
- (Defer `answer_callback_query` + `reply_markup` inline keyboards to the follow-on — see §6.)

---

## 5. Usability commands (read-only, reuse `_read_meta` / `render_event` / `_fmt_duration`)

Add to `harness/telegram_bot.py::handle_command` + register in `BOT_COMMANDS` (setMyCommands):
- **`/summary [latest|<run_id>]`** — rich recap card from `meta.json`: outcome, mode, duration,
  confidence, verifier ✅ vs critic ✓/✗, iterations, file counts (+a/~m/−d), token total, and a
  one-line reconcile verdict when present. New `summarize_run(run_id)` beside `summarize_latest`.
- **`/files [latest|<run_id>]`** — changed-file list with glyphs (➕ added / 📝 modified / ➖ deleted);
  when `changed-files.diff` exists, append per-file `+a −d` parsed from the diff. Cap ~30 files +
  `… +K more`; hard length-guard (< 4096) + HTML-escape.
- **`/why [latest|<run_id>]`** — "explain the verdict": `quality.note`, the verified/critic/stalled
  triad, and the full `reconciliation.findings` (name · classification · sub_kind). For a still-live
  run, fall back to the last adversarial `iteration_completed` via `render_event`.
- **Richer `/runs [N]`** — keep ✅/❌/⏳ + mode + duration; ADD confidence chip, file count, and a
  relative age (`12m ago`) via a new `_fmt_age` helper.

All new helpers are pure (take a run dir / meta dict) → unit-test with fixture `meta.json`.

---

## 6. Out of scope for THIS build (explicit — fast-follow)
- Inline-keyboard buttons + `callback_query` dispatch (new update code path; must stay
  whitelist-gated + always `answer_callback_query`).
- `/mute`, `/watch`, quiet-hours, notification digest/batching (touch the dispatch hot path;
  deserve their own focused build — keep mute as the next-up item).
- `/health`, `/tail`, `/diff`, per-event filters, multi-run labels, `/retry`/`/open-issue` handoff.
Mention these in `docs/telegram.md` as planned, but DO NOT build them here.

---

## 7. Files to touch
- `harness/telegram.py` — `BuildState` + update logic on `TelegramNotifier`; `render_status_card`;
  richer `render_event` strings; richer `_summary_card` + per-step recap from tracked state in
  `finished()`; new `TelegramClient` methods (edit/pin); message_id capture; throttle.
- `harness/telegram_bot.py` — `/summary`, `/files`, `/why`, richer `/runs`, `summarize_run`,
  `_fmt_age`, `BOT_COMMANDS` entries, `HELP_TEXT`.
- `docs/telegram.md` — document the new cards, the pinned live card, the new commands, and the
  fast-follow list.
- `tests/` — extend `test_telegram*.py`; add `test_telegram_status_card.py` (BuildState transitions,
  progress/ETA derivation, throttle signature, resume/graph/compaction edge cases) and
  `test_telegram_commands_rich.py` (summary/files/why/runs from fixture meta.json). Update the
  existing gating-count tests intentionally for any newly-rendered messages.

## 8. Acceptance
- `python -m pytest tests/ -k telegram -q` green (incl. new tests); full suite green.
- All new dynamic text HTML-escaped; spam-gate + dynamic-verbosity intact; no real ids/token in
  code or tests; tests hermetic.
- A simulated event stream (fixture) drives: header card → plan tree → step/adversarial stream →
  a pinned card that edits across ≥3 states (incl. an ETA appearing after step 1) → a rich terminal
  card. Demonstrate ✅ vs ☑️ honesty and an infra-fail rendering distinctly.
- Telegram remains best-effort: with `_post` forced to raise, no dispatch path raises or blocks.

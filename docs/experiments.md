# Experiments — what we measured about worker behaviour

Everything in this doc is reproducible from the bench scripts in `scripts/`:
`cloud_eval.py`, `model_sweep.py`, `token_efficiency.py`, `task_synth.py`, and
`calibrate.py`. The findings here directly informed the **adaptive re-routing
v1** feature (`agy_orchestrator/core/calibration.py` + streaming watchdog in
`core/agent.py`) and the default-config recommendations in `CLAUDE.md`.

If you're picking models for a dispatch, jump to **§5 Recommendations**.

---

## 1. Bench design

We needed three orthogonal questions answered:

1. **Quality**: does the worker get the right answer?
2. **Efficiency**: how many output tokens and how much wall-time does it spend
   getting there?
3. **Reliability under structure**: does it hold up on multi-file work, not
   just single-function puzzles?

Each is graded objectively — every task ships a hidden pytest suite that runs
against the worker's code in an isolated tempdir. No LLM-as-judge; no rubric.

The four scripts:

| Script | What it measures | When to run |
|--------|------------------|-------------|
| `cloud_eval.py` | per-task quality on a fixed pool | sanity check a single config |
| `model_sweep.py` | end-to-end wall-clock across workers | "which config is fastest in practice" |
| `token_efficiency.py` | raw token / cost / api-time from CLI JSON | "which config is cheapest at quality" |
| `task_synth.py` | synthesizer over (n_units × difficulty × structure) | feed `calibrate.py` |
| `calibrate.py` | OFAT crossover sweep × configs | "where does the cheap config beat the expensive one" |

All bench scripts run workers in **per-call tempdir sandboxes** so multi-file
prompts can't bleed `# file:` outputs into the repo root. Earlier runs of
`calibrate.py` lacked this isolation and produced contaminated rows — see §4.

## 2. Task tiers

`cloud_eval.py` ships three pools:

- **EASY** — terse algorithmic primitives (e.g. `roman_to_int`,
  `is_balanced`). Every modern worker scores 5/5; useful only for smoke tests.
- **HARD** — LeetCode-medium-class problems. Same story: saturated.
- **BRUTAL** (added this session) — single-function algorithmic tasks with
  subtle edge cases verified against reference solutions: `calc3`
  (truncate-toward-zero division + unary minus), `regex_match` (LeetCode 10
  full regex DP), `min_window` (sliding window), `decode_ways` (0-handling DP),
  `candy` (greedy two-pass).

**Brutal-tier finding**: still saturated. Every config we measured scored
5/5 first try. Single-function tasks — even hard ones with traps — don't
discriminate frontier models. **For real quality separation you need
multi-file repo work**, which the bench can't cleanly grade and is captured
as the gap that motivates the synthesizer (`task_synth.py`).

## 3. The empirical headlines

### 3.1 Token efficiency — the verbosity trap is real

From `scripts/token_efficiency.py` on the brutal tier:

| Config | median out_tok | $ / call | api wall | pass |
|---|---:|---:|---:|---:|
| **codex:gpt-5.4-mini:low** | **184** | n/a | 7.1s | 5/5 |
| claude:sonnet:low | 204 | $0.013 | 5.5s | 5/5 |
| codex:gpt-5.3-codex:low | 203 | n/a | 9.5s | 5/5 |
| codex:gpt-5.5:low | 249 | n/a | 11.4s | 5/5 |
| claude:opus:high | 228 | $0.054 | 5.8s | 5/5 |
| **claude:haiku:low** | **1327** | $0.021 | 15.7s | 5/5 |

Two surprises:

- **`claude:haiku:low` costs MORE than `claude:sonnet:low`** ($0.021 vs $0.013)
  while doing the same work. The cheap-per-token model emits ~7× the tokens —
  enough to flip the cost ranking. On `calc3` haiku spat 4528 tokens vs
  sonnet's ~190. This is the "**verbosity tax**" — weak models compensate
  with output length, paying it back in tokens / wall / dollars. The whole
  `core/calibration.py` watchdog exists to catch this case at dispatch time.
- **`codex:gpt-5.4-mini:minimal` returns empty output (0/5)**. There's a
  floor below which "less effort" means "no effort" — `low` works, `minimal`
  doesn't.

The lesson: **token efficiency is "capable model at LOW effort (terse)",
not "smallest model (verbose)"**.

### 3.2 Latency regime flip — opus:high *can* win, but not where it counts

Clean sequential timing on trivial brutal tasks (no contention,
`model_sweep.py --sequential`):

| Config | avg s | pass |
|---|---:|---:|
| claude:opus:high | **7.5s** | 5/5 |
| codex:gpt-5.4-mini:low | 8.2s | 5/5 |
| codex:gpt-5.5:high | 12.1s | 5/5 |
| claude:haiku:low | 21.4s | 5/5 |
| agy:flash:medium | 30.3s | 5/5 |
| grok:grok-build | 32.8s | 5/5 |
| agy:pro:high | 93.8s | 4/5 |

On trivial tasks, output-token *count* dominates wall-clock and capable
models are TERSE — so `opus:high` is genuinely fastest end-to-end. **This
generalises poorly.** On real multi-step work (operator's other
orchestrator, agi2), `claude:sonnet:medium` is faster than `claude:opus:high`
because opus has higher intrinsic per-token latency AND `high` effort
burns far more reasoning tokens. `calibrate.py`'s OFAT sweep is built
specifically to map where this crossover happens.

### 3.3 Token telemetry coverage

Only `codex` and `claude` expose real `output_tokens` in their CLI JSON.
`grok` and `agy` print text only (text/stopReason/sessionId for grok; raw
text for agy). Token-efficiency questions are therefore answerable for
codex/claude only; for grok/agy fall back to `model_sweep`'s wall-clock.

### 3.4 The grok web_search 6-domain bug (resolved)

Independent of model quality: `grok` deep-research prompts failed with a
server-side 400 because the model dynamically attached an
`allowed_domains` list with >5 entries to its own `web_search` calls.
Triggered by prompt framing like "DEEP research / verify with authoritative
sources". Fix: de-trigger the prompt ("search the web for X"). Full
write-up in [`docs/grok-search-findings.md`](grok-search-findings.md).

## 4. The contaminated multi-file calibrate run

The first `calibrate.py` pass produced rows like:

```
[structure] multi_file     claude:opus:high       0/0
[structure] multi_file     claude:sonnet:medium   3/3
[structure] interdependent claude:opus:high       0/0
[structure] interdependent claude:sonnet:medium   6/6
```

The `0/0` rows aren't quality failures — they're **the workers writing
their `# file: mod_*.py` outputs as actual files in the bench's cwd
(= repo root) instead of returning them as response text**. The grader,
running in an isolated tempdir, found no file blocks in the response and
scored 0/0. `sonnet:low/medium` happened to emit response-text blocks and
scored real. The comparison is therefore **invalid** — you cannot
conclude `sonnet:medium > opus:high` on multi-file work from that run.

**Fix shipped** (commit `6f109dc`): `token_efficiency.run_config` now runs
each worker in a `tempfile.TemporaryDirectory()` cwd. Re-running calibrate
will give clean rows. `.gitignore` covers `/mod_*.py` + `/pipeline.py` as a
defence in depth.

## 5. Recommendations

For day-to-day building via the harness:

- **Default generator (`codex`)** is correct — best Pareto position for
  code quality at low cost.
- **For terse-cheap-correct micro tasks**: `codex:gpt-5.4-mini:low` is the
  token-efficiency winner; `claude:sonnet:low` is a close second with the
  cheapest dollar cost we measured.
- **For real multi-step builds**: prefer `claude:sonnet:medium` over
  `claude:opus:high` (the agi2 observation, indicative pending clean
  calibrate re-run). The `--fallback` chain (default on) handles usage
  walls between providers.
- **Never default to `claude:haiku:low` for code** — costs more than
  `claude:sonnet:low` while emitting an order of magnitude more tokens.
  The streaming watchdog (`core/calibration.py`) is specifically designed
  to catch the case where it slips in via fallback.
- **Don't share a provider+account between the calling agent and the
  orchestrator's workers** — see "Account-sharing rule" in
  [`CLAUDE.md`](../CLAUDE.md).

## 6. Reproducing these numbers

```bash
# Token-efficiency frontier (codex + claude only).
python scripts/token_efficiency.py

# End-to-end speed across all four workers, sequential = no contention.
python scripts/model_sweep.py --sequential

# Calibration sweep (OFAT) — populates /tmp/agentorch_research/calibrate.jsonl
# which the watchdog reads to tighten per-config budgets.
python scripts/calibrate.py

# Re-render an existing run's analysis without spawning workers.
python scripts/calibrate.py --analyze-only /tmp/agentorch_research/calibrate.jsonl
```

All bench scripts stream JSONL to `/tmp/agentorch_research/`. Pass `--repeats N`
to `model_sweep` / `token_efficiency` for medians; pass `--sequential` to
`model_sweep` to remove within-combo concurrency contention.

## 7. Open items

- **Retired (this dispatch):** UsageAwareAllocator removed; --mode auto + live ledger + fallback chain provide real routing.
- **Clean calibrate re-run** (post-cwd-sandbox fix): needed before
  publishing a definitive multi-file / interdependent crossover table.
- **Patch-vs-rewrite economics**: the cheap-first draft-and-fill workflow
  was considered and rejected based on literature + saturation; see
  `memory/rejected-cheap-draft-cascade.md` for the analysis.
- **Adaptive re-routing v1**: shipped opt-in (always-on safe defaults with
  `AGY_WATCHDOG=off` opt-out). Next: pipe `AgentInstance` telemetry
  (`last_wall_ms`, `last_out_bytes`, `_watchdog_reason`) into ledger rows
  at the dispatch boundary so the calibration table self-updates from real
  dispatch traffic, not just from `calibrate.py` runs.

### Watchdog signal/noise fix (post-dashboard dispatch, 2026-05-28)

Observing the dashboard build's repeated stall trips (always agy critic at
180s → fallback to grok → grok recovers) plus the operator's note about
codex's network-retry windows prompted two related changes:

1. **Stderr now counts as progress** in the streaming watchdog. Previously
   `last_progress` was reset only by stdout bytes — but every worker writes
   its visible work to stderr (codex `exec` lines + `apply_patch` traces,
   claude/agy/grok status, network-retry messages) and reserves stdout for
   the final reply. A codex run reasoning silently on stdout while busy on
   stderr was being killed as "stalled". After the fix, any output on either
   stream resets the clock; the byte budget (which catches the haiku
   verbosity tax) still counts stdout only.
2. **Per-worker stall defaults**: `DEFAULT_STALL_SECONDS_BY_WORKER` now
   holds codex=600s, agy=600s, grok=600s, claude=180s. The global default
   moved from 180s → 300s for unknown workers. Codex was the immediate
   motivator: its in-CLI network-retry can leave it silent on both streams
   for several minutes before reconnecting; cheaper to let it ride than to
   roll the fallback chain forward and lose accumulated session context.

Both fixes are conservative: the watchdog still kills genuinely-hung workers
(see `test_watchdog_still_kills_truly_silent_worker`), and the hard
wall-timeout (`AGY_TIMEOUT`, default 2400s) is unchanged.

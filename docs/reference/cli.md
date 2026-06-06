# CLI reference

AgentOrch exposes two command-line entrypoints:

- **`python -m harness …`** — the operator layer. Dispatches coding work to
  worker CLIs, captures every run under `runs/<timestamp>/`, and provides run
  inspection, a broker queue, and a dashboard launcher. This is the primary
  day-to-day interface.
- **`python -m agy_orchestrator …`** — the standalone multi-agent engine. Runs a
  single workflow (`adversarial`, `tot`, `chain`, `master`) directly, without the
  harness's snapshot/capture/role machinery.

Environment variables referenced below (e.g. `AGY_NOTIFY`, `AGY_WATCHDOG_SCALE`,
`AGY_MAX_PARALLEL_NODES`, `TELEGRAM_BOT_KEY`) are documented in
[configuration.md](configuration.md); this page lists only the flags and their
defaults.

**Exit codes:** `python -m harness` exits `0` on success and `1` on failure:
`harness do` returns `1` when the dispatch result is not successful, and its own
validation (e.g. `--fresh`/`--resume`, `--plan`/`--plan-graph`,
`--plan-only`+`--plan`, `--detach`+`--direct`) returns `1`. Argparse-level errors
exit with code `2` — this covers unknown or wrongly-typed arguments and the
argparse mutually-exclusive pairs `--queue`/`--direct` and
`--telegram`/`--no-telegram`.

`python -m agy_orchestrator` always exits `0` on completion: `main()` returns
`None` and is invoked without `SystemExit`, so a workflow that fails or does not
verify still exits `0` (it does not propagate a workflow-failure exit code). It
exits non-zero only on an argparse error or an uncaught exception.

---

## Modes at a glance (`harness do --mode`)

| Mode | What it does | Needs `--test-cmd` |
|---|---|---|
| `direct` | One-shot generation. Fast and cheap. | no |
| `adversarial` | Generate + critic refinement loop. **Default.** | no |
| `feedback` | Generate → run tests → feed errors back → repair loop. | yes |
| `cascade` | Cheap-first escalation across the `--generator` stages; escalate on verifier failure. | yes |
| `master` | Plan → Tree-of-Thought → adversarial, for whole features. Also runs graph DAG plans. | optional |
| `pat` | Plan-after-Trial: a direct attempt first, escalate to `master` only on verifier failure (~40% cost savings on easy tasks). | yes |
| `vote` | K parallel candidates in isolated workspaces; the verifier picks the winner (K = `--branches`). Heterogeneous when the chain spans providers. | yes |
| `auto` | Rule-based router picks the concrete mode from task features (test-cmd presence, prompt scale, ambiguity keywords). | no |

## Workers and default chains

Worker tokens accepted in `--generator` / `--critic` chains (comma-separated, in
fallback priority order):

| Token | Provider family | Default model / effort |
|---|---|---|
| `codex` | openai | `standard` (→ `gpt-5.3-codex-spark`) / `high` |
| `agy` | google | `pro` / `high` |
| `claude` | anthropic | `opus` / `high` |
| `grok` | xai | `grok-build` / `n/a` (rejects effort) |
| `computer-use` | computer-use | `n/a` (non-LLM shim; `direct` generator only) |

Default chains (from `harness/roles.py`):

- **Generator chain:** `codex,agy,grok`
- **Critic chain:** `agy,codex,grok`

`claude` is excluded from the defaults by the account-sharing rule. A lead
generator and lead critic in the same provider family triggers a cross-family
self-verification warning (suppress with `AGY_CRITIC_FAMILY_CHECK=off`).

---

# Part A — `python -m harness`

```
python -m harness [-v|--verbose] [--version] <command> [args]
```

| Global flag | Meaning |
|---|---|
| `-v`, `--verbose` | DEBUG-level logging (default INFO). |
| `--version` | Print the running build (version + git commit) and exit. |

Subcommands: [`do`](#harness-do), [`spec`](#harness-spec), [`runs`](#harness-runs),
[`pr`](#harness-pr), [`merge`](#harness-merge), [`abandon`](#harness-abandon),
[`show`](#harness-show), [`serve`](#harness-serve), [`queue`](#harness-queue),
[`dashboard`](#harness-dashboard).

---

## `harness do`

Dispatch one coding instruction to a worker.

```
python -m harness do "INSTRUCTION" [flags]
```

**Positional:** `instruction` — the instruction for the worker.

### Core workflow flags

| Flag | Type / choices | Default | Meaning |
|---|---|---|---|
| `--mode` | `direct`,`adversarial`,`feedback`,`cascade`,`master`,`pat`,`vote`,`auto` | `adversarial` | Workflow shape (see [modes table](#modes-at-a-glance-harness-do---mode)). |
| `--context` | str | `None` | Extra context appended to the instruction. |
| `--generator` | str (comma chain) | `codex,agy,grok` | Generator fallback chain. |
| `--critic` | str (comma chain) | `agy,codex,grok` | Critic fallback chain. |
| `--fallback` / `--no-fallback` | bool | `True` (on) | Wrap roles in usage-exhaustion fallback. |
| `--cycles` | int | `2` | Times the fallback chain is cycled before giving up. |
| `--max-iterations` | int | `5` | Max generator/critic iterations (adversarial/master). |
| `--branches` | int | `3` | ToT branches (master); candidate count K (vote). |
| `--test-cmd` | str | `None` | Verification command run as a quality gate. |
| `--spec` | path | `None` | Path to an approved FloodSpec design doc (see [`harness spec`](#harness-spec)); injected as authoritative design. In `master` the planner decomposes this instead of re-inventing one. |

### Verifier / candidate flags

| Flag | Type / metavar | Default | Meaning |
|---|---|---|---|
| `--verifier-mem-max` | `SIZE` (e.g. `3G`) | `None` | Run `--test-cmd` in its own memory-capped `systemd-run --user` scope so a heavy gate is OOM-killed in its scope instead of freezing the host. Degrades to uncapped with a warning if systemd is unavailable. Env: `AGY_VERIFIER_MEM_MAX`. |
| `--candidate-setup` | `CMD` | `None` | `vote` mode: shell command run inside each candidate's isolated workspace BEFORE its verifier (e.g. `python -m venv .venv && .venv/bin/pip install -e .`). Makes vote isolation sound on editable-install repos. Bounded by the verifier-concurrency cap. |
| `--baseline-gate` | flag | off | Run the pre-run baseline verifier (full `--test-cmd` on the unchanged tree) for non-vote modes too. Off by default; set it to restore `verifier_delta` telemetry (preserved/regressed/fixed). |

### Checkpoint / resume (master, pat)

| Flag | Default | Meaning |
|---|---|---|
| `--fresh` (alias `--no-resume`) | off | Ignore any salvage checkpoint and start clean. |
| `--resume` | off | Force resume from the salvage checkpoint even if the out-dir diverged from the tree it was saved against. |

`--fresh` and `--resume` are mutually exclusive. Default policy is `auto`:
resume only if the out-dir still matches the checkpointed tree, else start fresh.

### Plan injection / dry-run (master, pat)

| Flag | Type / metavar | Default | Meaning |
|---|---|---|---|
| `--plan-only` (alias `--dry-run`) | flag | off | Run only the planner, emit the step plan (stdout + events + `runs/<id>/plan.json`), and exit BEFORE any worker writes. Only applies to `master`/`pat` (warns otherwise). |
| `--plan` | `FILE` | `None` | Execute the steps in this plan file VERBATIM, skipping the planner. Accepts a flat JSON list of step strings OR a graph `nodes` DAG. A graph DAG with non-linear deps runs the concurrent frontier scheduler. |
| `--plan-graph` | `FILE` | `None` | Like `--plan` but STRICT — the file MUST be a graph `nodes` DAG (errors on a flat plan). `master`-only for v1. |
| `--plan-expect-sha` | `SHA256` | `None` | With `--plan`/`--plan-graph`: REFUSE to run unless the plan file's sha256 matches this hash (a hard pin for unattended dispatch). Without it, provenance is recorded but not gated. |
| `--max-parallel-nodes` | `N` | `None` (unbounded) | `master`: cap concurrent DAG nodes when executing a graph plan. Env `AGY_MAX_PARALLEL_NODES`. `1` serializes a wide layer. Only affects a graph plan with non-linear deps. |
| `--merge-policy` | `disjoint`,`reconcile`,`fail` | `reconcile` | `master` graph mode: how two parallel nodes writing the SAME file are reconciled. `reconcile` auto-applies disjoint writes and sends overlaps to the reconcile station (then re-verifies the merged tree); `disjoint`/`fail` abort on any overlap (`fail` records conflicting paths in `meta.json`). |

`--plan` and `--plan-graph` are mutually exclusive; both are mutually exclusive
with `--plan-only`. Flat plans run on `master` or `pat`; a graph DAG with
non-linear deps requires `--mode master`.

### Path policy

| Flag | Metavar | Default | Meaning |
|---|---|---|---|
| `--protect-paths` | `GLOB[,GLOB...]` | `None` | Fail the run if any worker modifies a path matching these denylist globs (e.g. `docs/core/**,**/*.lock`). `**` spans directories. Violations recorded in `meta.json`. |
| `--allow-paths` | `GLOB[,GLOB...]` | `None` | Allowlist inverse: fail the run if a worker writes any path OUTSIDE these globs. |

### Run watchdog / notifications

| Flag | Type / metavar | Default | Meaning |
|---|---|---|---|
| `--run-stall-abort` | `SEC` (float) | `None` | Whole-run watchdog: abort if NO run-level forward progress (step/branch/plan milestone, worker-call boundary, usage tick) happens within SEC, classify `stalled` in `meta.json`, fire `--notify`. Complements the per-agent stall watchdog. |
| `--notify` | `URL\|CMD` | `None` | Best-effort notification on lifecycle/anomaly events. An `http(s)://` value is POSTed a JSON payload; anything else runs as a shell command (payload on stdin + `AGY_NOTIFY_*` env). Env: `AGY_NOTIFY`. |
| `--notify-cmd` | `CMD` | `None` | Shell-command form of `--notify`; runs on the same events. |
| `--heartbeat-interval` | `SEC` (float) | `None` | Seconds between run-level `heartbeat` events in `events.jsonl`. Default 30 (env `AGY_HEARTBEAT_SECONDS`); `0` disables. |

### Telegram build-progress bot

| Flag | Choices | Default | Meaning |
|---|---|---|---|
| `--telegram` / `--no-telegram` | flag (mutually exclusive) | auto | Force-enable / force-disable Telegram notifications. Auto-on when `TELEGRAM_BOT_KEY` is set AND the whitelist (`AGY_TELEGRAM_USERS`) is non-empty; `--telegram` warns and stays off if either is missing. |
| `--telegram-verbosity` | `quiet`,`normal`,`verbose`,`debug` | `None` | Message verbosity (default env `AGY_TELEGRAM_VERBOSITY` else `normal`). |

### Effort / model overrides

The default tier (codex `gpt-5.3-codex-spark` / `high`) suits routine work; crank
these for a mission-critical, invariant-touching build. Effort `max` maps to codex
`reasoning_effort=xhigh`. Explicit flags override `--effort-profile`.

| Flag | Type / metavar | Default | Meaning |
|---|---|---|---|
| `--gen-effort` | `TIER` (`low\|medium\|high\|max`) | `None` | Generator effort tier; applies to every effort-capable provider in the generator chain (grok no-ops). |
| `--critic-effort` | `TIER` | `None` | Critic effort tier across the critic chain. |
| `--architect-effort` | `TIER` | `None` | Effort tier for the master/pat architect chain (alias of `--gen-effort` for those modes). |
| `--gen-model` | `NAME` | `None` | Model for the generator chain lead (provider-specific). |
| `--critic-model` | `NAME` | `None` | Model for the critic chain lead. |
| `--architect-model` | `NAME` | `None` | Model for the master/pat architect chain lead. |
| `--codex-model` | `NAME` | `None` | Convenience: set the codex model anywhere it appears in any chain (e.g. `gpt-5.5`). Validated against codex's model list. |
| `--effort` | `MAP` | `None` | Per-provider effort map, e.g. `codex=max,agy=high` (or a bare tier applied to all effort-capable providers). `grok=…` is dropped. |
| `--model` | `MAP` | `None` | Per-provider model map, e.g. `codex=gpt-5.5`. |
| `--effort-profile` | `low`,`balanced`,`max` | `None` | One-switch preset: `max` cranks every effort-capable provider to its strongest model + ceiling effort (codex `gpt-5.5`/`xhigh`); `balanced` == defaults; `low` = cheap tier. |
| `--watchdog-scale` | `FLOAT` | `None` | Multiply the streaming-watchdog stall/byte budgets (>1.0) for known-heavy tiers so a long `xhigh` run isn't truncated mid-flight. Note this scales **both** the byte budget and the stall timeout together; use the two flags below to move one without the other. Env `AGY_WATCHDOG_SCALE`. |
| `--watchdog-max-bytes` | `BYTES` (int) | `None` | Absolute verbose **byte** budget (issue #83), applied *after* `--watchdog-scale` and **replacing** the byte budget alone — the stall budget stays calibrated+scaled. Decouples bytes from stall so a read-heavy primary generator (e.g. codex reading a design doc + several modules) isn't SIGKILLed as `runaway:verbose` and silently demoted onto a fallback worker. Must be `>0`. Env `AGY_WATCHDOG_MAX_BYTES`. |
| `--watchdog-stall` | `SEC` (float) | `None` | Absolute per-worker **stall** budget in seconds (issue #83), applied *after* `--watchdog-scale` and **replacing** the stall budget alone — the byte budget stays calibrated+scaled. Decouples stall from bytes so raising one failure mode's tolerance doesn't inflate the other. Must be `>0`. Env `AGY_WATCHDOG_STALL`. |
| `--max-parallel-workers` | `N` | `None` | Cap how many candidates run end-to-end at once in `vote` mode. Env `AGY_MAX_PARALLEL_WORKERS`. |
| `--worker-mem-max` | `SIZE` (e.g. `4G`) | `None` | Per-candidate verifier memory cap for `vote`/`tot`, run in its own systemd scope. Guards against OOM/freeze when `--branches`>1 verify in parallel. |

### Reconciliation station

| Flag | Choices / metavar | Default | Meaning |
|---|---|---|---|
| `--reconcile` | flag | off (on for `--mission-critical`) | After a converged + green build, trace each goal-named mechanism to the live path and flag `exists-but-not-load-bearing` defects (dead/stubbed/bypassed code that passes tests). Env `AGY_RECONCILE=1`. Writes `runs/<id>/reconcile.json`. Verdict is distinct — never folded into the verifier's pass/fail. |
| `--reconcile-disposition` | `warn`,`fail`,`open-task` | `None` (→ `warn`; `fail` under `--mission-critical`) | What a non-reconciled verdict does: `warn` (report loudly, don't fail), `fail` (flip the run to failed), `open-task` (warn + recommend a follow-up task). |
| `--ablation-cmd` | `'CMD {MECH}'` | `None` | Opt-in programmatic ablation witness: a READ-ONLY shell command the reconcile station runs in a throwaway worktree to MEASURE each mechanism's load-bearing signal. Run twice per mechanism (clean, then with `AGY_ABLATE=<mech>` set); the last number printed (or a `WITNESS:<n>` tag) is the signal. `{MECH}` is replaced with each mechanism name. |

### Other dispatch flags

| Flag | Type / metavar | Default | Meaning |
|---|---|---|---|
| `--web-search` | flag | off | Enable codex web search (`-c tools.web_search=true`) for accuracy. |
| `--mission-critical` | flag | off | Prepend a catastrophic-failure-focused preamble to the critic prompt (adversarial); more exhaustive, severity-prioritized review. Implies `--reconcile` and `fail` disposition unless overridden. |
| `--out-dir` | `PATH` | AgentOrch repo root | Directory the worker writes into (its cwd). Set when invoking AgentOrch from another repo. The snapshot diff and changed-files list follow this path. |

### Git PR mode

Opt-in. A `--git-pr` dispatch runs on an **isolated git worktree + temp branch**
(`agentorch/<run_id>`) instead of writing into your checkout — your own checkout is
never moved. It commits one accepted step at a time, then pushes the branch and
opens a **draft PR** to your current (base) branch, promoting it to ready-for-review
when the run verifies. State is persisted to `runs/<id>/pr_session.json` and
summarised in `meta.json` under `git_pr`. Decide later with
[`pr`](#harness-pr) / [`merge`](#harness-merge) / [`abandon`](#harness-abandon), or
fire a corrective plan with `--continue`. Requires a clean git work tree (refuses a
dirty / detached / non-repo tree); degrades to a local branch (no PR) without a
remote or `gh`. See [`docs/git-pr-mode-design.md`](../git-pr-mode-design.md).

| Flag | Type / metavar | Default | Meaning |
|---|---|---|---|
| `--git-pr` | flag | off | Run on an isolated worktree + temp branch, commit each accepted step, and open a draft PR to the base branch. |
| `--continue` | `RUN_ID` | `None` | Corrective resume: re-attach to a prior `--git-pr` run's temp branch and run THIS instruction on top, updating the **same** branch + PR (no second PR; promoted again when it verifies). Implies `--git-pr`. The corrective run gets its own `run_id` but updates the original run's `pr_session.json`. Repeatable. |

### Computer-use flags

These take effect only when the generator chain leads with `computer-use`.

| Flag | Choices | Default | Meaning |
|---|---|---|---|
| `--computer-use-mode` | `ISOLATED`,`OBSERVE`,`REAL` | `None` (→ ISOLATED) | ISOLATED: private Xvfb, full perceive+act. OBSERVE: real `:0` read-only perception, isolated actions. REAL: real `:0` perception + owned-child real act under SafetyKernel gates. |
| `--real-gui-policy` | `full`,`children` | `None` | REAL mode foreign-target policy: `full` allows prompt-gated foreign act; `children` only owned-child actuation. |
| `--ask-mode` | `on`,`off` | `None` | REAL mode: GUI confirmation prompting for foreign-target actions. |
| `--browser-engine` | `bing`,`duckduckgo`,`google` | `bing` | Browser engine for autonomous navigate/search flows. |
| `--browser-display` | str | `None` | Browser display override. Default: `:0` in REAL mode, isolated Xvfb otherwise. |
| `--computer-use-task-priority` | `normal`,`high` | `None` (→ normal) | `high` routes reasoner claude→codex; `normal` codex→claude. |
| `--computer-use-budgets` | `JSON` | `None` | JSON dict overriding budgets, e.g. `{"max_steps": 50, "max_actions": 30}`. |

### Broker routing (singleton layer)

Default behavior (neither `--queue` nor `--direct`, no broker running) is a local
in-process dispatch. With a broker reachable, the default AUTO path routes to it.

| Flag | Default | Meaning |
|---|---|---|
| `--queue` | off | Submit to a running broker (`harness serve`) instead of running in-process; error if no broker is reachable. The broker drains jobs at a concurrency cap of 2, keeping the two live lines off the same provider account pool. |
| `--direct` | off | Force a local in-process dispatch even if a broker is running. |
| `--detach` | off | With broker routing: submit the job, print its id, and exit (don't wait). Track with `harness queue`. Incompatible with `--direct`. |

`--queue` and `--direct` are mutually exclusive. A live graph DAG plan is not
serializable over IPC, so such a run stays local (errors under `--queue`).

### Examples

```bash
# Default adversarial dispatch
python -m harness do "add a --version flag to the CLI"

# Fast one-shot, no fallback
python -m harness do "rename helper X to Y" --mode direct --no-fallback

# Test-feedback repair loop gated by pytest
python -m harness do "fix the failing parser" --mode feedback --test-cmd "pytest -q"

# Whole-feature build with a quality gate
python -m harness do "build feature Z" --mode master --test-cmd "pytest -q"

# Plan round-trip: emit, review/edit, execute verbatim
python -m harness do "build feature Z" --mode master --plan-only
python -m harness do "build feature Z" --mode master --plan runs/<id>/plan.json

# Concurrent dependency DAG with reconcile merge
python -m harness do "build feature Z" --mode master \
    --plan graph.json --merge-policy reconcile --max-parallel-nodes 3

# Crank every provider to its ceiling for a critical build
python -m harness do "rework the scheduler" --mode master \
    --effort-profile max --mission-critical --test-cmd "pytest -q"

# Write into another repo, protect lockfiles
python -m harness do "add endpoint" --out-dir /path/to/app \
    --protect-paths "**/*.lock,migrations/**"

# K-candidate vote in isolated workspaces
python -m harness do "optimize the hot loop" --mode vote --branches 4 \
    --test-cmd "pytest -q" --candidate-setup "python -m venv .venv && .venv/bin/pip install -e ."

# Git PR mode: build on an isolated branch, open a draft PR, then decide
python -m harness do "build feature Z" --mode master --test-cmd "pytest -q" \
    --git-pr --out-dir /path/to/app
python -m harness pr <run_id>                       # review branch / PR / commits
python -m harness do "also handle empty input" --continue <run_id>   # corrective
python -m harness merge <run_id> --method squash --delete-branch
```

---

## `harness spec`

FloodSpec: turn a short goal + constraints into a complete design doc. Writes
`runs/<id>/spec.md`; feed the result to `harness do --mode master --spec`.

```
python -m harness spec "GOAL" [flags]
```

**Positional:** `goal` — the short goal to design a system for.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `-c`, `--constraint` | str (repeatable) | `[]` | A constraint the design must honor. |
| `--architect` | str (comma chain) | `codex,agy,grok` | Architect (author) chain. |
| `--critic` | str (comma chain) | `agy,codex,grok` | Design-critic chain (cross-provider gives stronger gates). |
| `--fallback` / `--no-fallback` | bool | `True` (on) | Wrap roles in usage-exhaustion fallback. |
| `--cycles` | int | `2` | Times the fallback chain is cycled before giving up. |
| `--max-iterations` | int | `3` | Max architect/critic refinement rounds. |
| `-o`, `--output` | path | `None` | Also write the doc here (e.g. a target repo's `DESIGN.md`). The `runs/<id>/spec.md` artifact is always written. |

---

## `harness runs`

List recent runs.

```
python -m harness runs [--limit N]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--limit` | int | `20` | Number of recent runs to list. |

---

## `harness show`

Print a run's `meta.json` and `changed-files.diff`.

```
python -m harness show <run_id>
```

**Positional:** `run_id` — the run directory name under `runs/`.

---

## `harness pr`

Show a [`--git-pr`](#git-pr-mode) run's session: status, base/temp branch, PR url
(draft|ready), and the per-step commit list. When the run is `awaiting_decision`,
prints the merge / corrective / abandon next-step commands.

```
python -m harness pr <run_id>
```

**Positional:** `run_id` — the original `--git-pr` run's id.

---

## `harness merge`

Merge a [`--git-pr`](#git-pr-mode) run's PR (`gh pr merge`) and mark the session
`merged`. Errors if the run has no open PR.

```
python -m harness merge <run_id> [--method squash|merge|rebase] [--delete-branch]
```

| Flag | Choices | Default | Meaning |
|---|---|---|---|
| `--method` | `squash`,`merge`,`rebase` | `squash` | Merge method passed to `gh pr merge`. |
| `--delete-branch` | flag | off | Delete the temp branch after merging. |

---

## `harness abandon`

Close a [`--git-pr`](#git-pr-mode) run's PR (if any, via `gh pr close`) and mark the
session `abandoned`. A `gh` failure is reported but the session is still marked.

```
python -m harness abandon <run_id> [--delete-branch]
```

| Flag | Default | Meaning |
|---|---|---|
| `--delete-branch` | off | Also delete the temp branch. |

---

## `harness serve`

Run the singleton broker in the foreground: bind the IPC socket, honor the
singleton guard (refuse a rival), and drain the persistent queue with a
concurrency cap. Daemonize via `nohup`/`systemd` as desired.

```
python -m harness serve [--cap N]
```

| Flag | Type / metavar | Default | Meaning |
|---|---|---|---|
| `--cap` | `N` | `None` (→ 2) | Max concurrent orchestration lines drained at once. Two lines are kept off the same provider account pool. |

---

## `harness queue`

List the broker's build queue as a table (id, status, mode, instruction head).
Prints a notice if no broker is running.

```
python -m harness queue
```

(No flags.)

---

## `harness dashboard`

Launch the AgentOrch control dashboard (execs `python -m dashboard`).

```
python -m harness dashboard [--port PORT] [--browser]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--port` | int | `8765` | Dashboard port. |
| `--browser` | flag | off | Open the dashboard in the default browser. Off by default. |
| `--no-browser` | flag | — | Deprecated no-op (not opening a browser is now the default). |

---

# Part B — `python -m agy_orchestrator`

The standalone engine. One subcommand per workflow.

```
python -m agy_orchestrator [global flags] <command> --prompt "…" [flags]
```

### Global flags

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--version` | — | — | Print the running build (version + git commit) and exit. |
| `--claude-plan` | str | `free` | Claude subscription plan (e.g. `20x max`); feeds baseline-effort selection. |
| `--codex-plan` | str | `free` | Codex subscription plan (e.g. `$100`). |
| `--agy-plan` | str | `free` | Agy subscription plan. |

Subcommands: `adversarial`, `tot`, `master`, `chain`. Each requires `--prompt`.
`--agent` choices are `agy`, `claude`, `codex`, `grok` (default `agy`, except
`chain` which uses `--agents` defaulting to `codex`). `--model` defaults to
`standard` everywhere.

### `adversarial`

Generator/critic refinement loop.

| Flag | Type / choices | Default | Meaning |
|---|---|---|---|
| `--prompt` | str (required) | — | The goal prompt. |
| `--agent` | `agy`,`claude`,`codex`,`grok` | `agy` | Agent type to use. |
| `--model` | str | `standard` | Base model. |
| `--test-cmd` | str | `None` | Optional programmatic test command (builds a `QualityVerifier`). |
| `--max-iterations` | int | `5` | Max loops. |

### `tot`

Tree-of-Thought: parallel branch generation + scored selection.

| Flag | Type / choices | Default | Meaning |
|---|---|---|---|
| `--prompt` | str (required) | — | The goal prompt. |
| `--agent` | `agy`,`claude`,`codex`,`grok` | `agy` | Agent type to use. |
| `--model` | str | `standard` | Base model. |
| `--branches` | int | `3` | Number of ToT branches. |

### `master`

Plan → ToT → adversarial, with checkpointing and context compaction.

| Flag | Type / choices | Default | Meaning |
|---|---|---|---|
| `--prompt` | str (required) | — | The goal prompt. |
| `--agent` | `agy`,`claude`,`codex`,`grok` | `agy` | Agent type to use. |
| `--model` | str | `standard` | Base model. |
| `--test-cmd` | str | `None` | Optional programmatic test command. |
| `--branches` | int | `3` | ToT branches per task. |
| `--max-iterations` | int | `5` | Max loops for adversarial refinement. |
| `--fallback` | flag (`store_true`) | off | On a provider usage/quota wall, fall back `codex→agy→claude→codex` (chain leads with `--agent`, deduped, cycled). |
| `--checkpoint` | str | `None` | Path to a checkpoint file; resume in place from the last completed step if it exists. |
| `--compaction-interval` | int | `6` | Compact context + reset the chained session every N steps (`0` disables). |
| `--max-context-chars` | int | `12000` | Also compact whenever accumulated project context exceeds this many chars (`0` disables). |

> Note: in the standalone `master`, `--fallback` is an opt-in `store_true` flag
> (default off). This differs from `chain` and from `harness do`, where fallback
> is on by default.

### `chain`

Linear pipeline: feed the prompt through agents in sequence.

| Flag | Type / choices | Default | Meaning |
|---|---|---|---|
| `--prompt` | str (required) | — | The goal prompt. |
| `--agents` | `agy`,`claude`,`codex`,`grok` (1+, `nargs="+"`) | `["codex"]` | Agents to chain. Every run leads with codex so a refreshed codex quota is auto-detected next run. |
| `--model` | str | `standard` | Base model. |
| `--fallback` / `--no-fallback` | bool | `True` (on) | Wrap each stage in usage-exhaustion fallback (`primary→agy→claude→codex`, cycling). |

### Examples

```bash
python -m agy_orchestrator adversarial --prompt "write a CSV parser" --agent codex
python -m agy_orchestrator tot --prompt "design a cache layer" --branches 4
python -m agy_orchestrator master --prompt "build the importer" \
    --agent codex --fallback --test-cmd "pytest -q" --checkpoint /tmp/ckpt.json
python -m agy_orchestrator chain --prompt "refactor module X" --agents codex agy
```

---

## Plan JSON shapes

`--plan-only` emits `runs/<id>/plan.json`. `--plan` / `--plan-graph` accept two
shapes:

**Flat plan** — a JSON array of step strings, executed in order (runs on
`master` or `pat`):

```json
[
  "Add the config schema and loader",
  "Wire the loader into the CLI",
  "Add tests for the loader"
]
```

**Graph DAG** — an object with a `nodes` list; disjoint subtrees run concurrently
in isolated workspaces (runs on `master` only; `--plan-graph` requires this
shape). Each node carries an id, the step text, and its dependencies:

```json
{
  "nodes": [
    {"id": "schema", "task": "Add the config schema", "deps": []},
    {"id": "loader", "task": "Add the loader", "deps": ["schema"]},
    {"id": "docs",   "task": "Document the config", "deps": ["schema"]}
  ]
}
```

Concurrency is bounded by `--max-parallel-nodes`; conflicting writes between
parallel nodes are handled per `--merge-policy`. See
[graph-execution-design.md](../graph-execution-design.md) for the full design.

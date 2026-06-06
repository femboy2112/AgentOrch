# AgentOrch

Dispatch coding work to multiple LLM CLIs, gate it with real verification, and get a reviewable diff-backed run record every time.

## What is this?
AgentOrch is a cloud-only multi-LLM orchestrator you run from your own
workstation. It dispatches implementation work to existing worker CLIs
(`codex`, `claude`, `agy`, `grok`) instead of introducing another hosted agent
service.

The core pattern is simple: send a concrete instruction, choose a workflow,
run workers, and gate outcomes with a programmatic verifier when available.
Depending on mode, this can be a one-shot pass, a generator/critic loop,
a test-feedback repair loop, or a multi-step planner pipeline.

Every dispatch is captured under `runs/<timestamp>/` with prompt, logs,
metadata, and a disk snapshot diff (`changed-files.diff`). You can review
what changed and why before accepting the result.

AgentOrch is complementary to your primary coding environment. You still define
requirements and decide what to merge; AgentOrch handles dispatching,
orchestration, verification loops, and run bookkeeping in a repeatable way.

## Why use it?
- Compose multiple worker CLIs through seven workflow modes (`direct`,
  `adversarial`, `feedback`, `cascade`, `master`, `pat`, `vote`) depending on
  task size, quality bar, and cost constraints. `master` also executes
  **dependency DAGs** — hand it a graph plan and concurrent branches run in
  isolated workspaces with a switchable merge policy
  (`--plan graph.json --merge-policy reconcile`; see
  `docs/graph-execution-design.md`).
- Gate outputs with a programmatic verifier (`--test-cmd`) in the modes that
  require hard pass/fail signals, instead of relying on LLM self-evaluation.
- Capture each run with artifacts (`prompt.txt`, `stdout.log`, `stderr.log`,
  `events.jsonl`, `changed-files.diff`, `meta.json`) so review is reproducible.
- Monitor long-running work from a local dashboard (`python -m harness dashboard`)
  with streaming per-worker events.
- Protect dispatch stability with per-worker watchdog budgets backed by a
  calibration table and an append-only live ledger of verified runs.
- Use provider chains and fallback (`--generator`, `--critic`, `--fallback`) so
  a usage wall on one provider does not automatically terminate the run.
- Target other repositories safely with `--out-dir`, so workers write into the
  intended project while AgentOrch keeps its own orchestration artifacts local.

In practice, this means you can delegate mechanical or high-volume edits,
preserve your own context for review and design decisions, and still keep a
strict acceptance gate based on real commands rather than model confidence.

## Quick start
```bash
# 1) clone
git clone https://github.com/femboy2112/AgentOrch.git
cd AgentOrch

# 2) install (editable)
pip install -e .

# optional dev deps
# pip install -e ".[dev]"

# 3) run a trivial dispatch in this repo
python -m harness do "add a short comment to docs/notes.txt" \
  --mode direct \
  --out-dir .
```

See [Workflows](#workflows) below to pick the right mode for your task.

If you are invoking AgentOrch from another repository, point `--out-dir` at
that repository path so workers write there instead of in AgentOrch.

Typical follow-up commands:

```bash
# list recent dispatches
python -m harness runs

# inspect one run's diff and metadata
python -m harness show <run_id>

# launch the local operator dashboard (127.0.0.1:8765 by default)
python -m harness dashboard
```

Each run directory includes:

- `prompt.txt` — exact worker prompt used for the dispatch.
- `stdout.log` — final workflow output stream.
- `stderr.log` — detailed workflow/runtime logging.
- `events.jsonl` — structured per-worker event stream.
- `changed-files.diff` — snapshot diff of on-disk changes.
- `meta.json` — mode, quality signals, timing, and file-change metadata.

## Workflows
| Mode | What it does | When to use |
|---|---|---|
| `direct` | One generator pass, no critic loop, no verifier gate by default. | Small, precise edits you already scoped. |
| `adversarial` | Generator + critic refinement loop; optional verifier can short-circuit on pass. | You want stronger quality without a mandatory test gate. |
| `feedback` | Generator -> verifier -> repair loop using real test/build/lint errors (**requires `--test-cmd`**). | A programmatic test is the source of truth and you want iterative repair. |
| `cascade` | Cheap-first generator stages; each stage uses test-feedback and escalates only on verifier failure (**requires `--test-cmd`**). | Task difficulty is unknown and you want cost-aware escalation. |
| `master` | Planner + tree-of-thought + adversarial refinement for multi-step builds; optional verifier. | Whole features or long multi-step work with broad context. |
| `pat` | Plan-after-Trial: try direct first, escalate to master only if verifier fails (**requires `--test-cmd`**). | General default when many tasks are easy but you want a robust fallback path. |
| `vote` | K parallel candidates in isolated workspaces; verifier picks and applies a winner (**requires `--test-cmd`**). | You want candidate diversity and verifier-selected best result. |

Verifier requirement summary:

- `--test-cmd` required: `feedback`, `cascade`, `pat`, `vote`
- `--test-cmd` optional: `adversarial`, `master`
- `--test-cmd` not required for baseline behavior: `direct`

## Architecture
```text
agy_orchestrator/            # Multi-agent engine and workflow implementations
  core/                      # Worker adapters, fallback wrappers, usage/cost controls
  execution/                 # Verifier, ledgers, pipelines, and workspace execution primitives
  interaction/               # Decision-handling components for workflow interaction points
  workflows/                 # Mode implementations (adversarial, test-feedback, cascade, master, pat, vote)
  cli.py                     # Standalone orchestrator CLI entrypoint

harness/                     # Operator-facing dispatch/control layer
  cli.py                     # `harness do|runs|show|dashboard` command interface
  dispatch.py                # Prompt build, mode routing, snapshot diffing, run artifact capture
  roles.py                   # Generator/critic chain defaults, worker mapping, watchdog setup
  snapshot.py                # Git-independent before/after filesystem snapshot + diff helpers
```

The dashboard package is intentionally separate from this high-level tree and
is launched through `harness dashboard`; see `docs/dashboard-design.md` for the
v1 UX and API contract.

## Status
AgentOrch is a cloud-only orchestrator extracted from earlier internal lineage
and is in active polish toward public release. It requires Python 3.10+ in
operator environments and is exercised against locally installed `codex`,
`claude`, `agy`, and `grok` CLIs.

## Documentation
Reference (source-checked, in [`docs/reference/`](docs/reference/)):

- [`docs/reference/cli.md`](docs/reference/cli.md) — complete CLI reference for `harness` and `agy_orchestrator` (every subcommand, flag, and mode).
- [`docs/reference/python-api.md`](docs/reference/python-api.md) — programmer's API: agents, workflows, verifier, and fallback chains.
- [`docs/reference/configuration.md`](docs/reference/configuration.md) — the full environment-variable surface.
- [`docs/reference/telegram-commands.md`](docs/reference/telegram-commands.md) — Telegram bot command reference.

Guides and design:

- `llms.txt` — dense, token-efficient digest of the whole surface for LLM coding agents.
- `AGENTS.md` — guidance for LLM agents integrating AgentOrch as a callable tool.
- `CLAUDE.md` — operator/maintainer guide with runtime rules and architecture notes.
- `docs/experiments.md` — empirical findings that informed workflow defaults.
- `docs/dashboard-design.md` — dashboard v1 design and implementation spec.

Additional practical entry points:

- `harness/cli.py` — command surface for `do`, `runs`, `show`, and `dashboard`.
- `harness/dispatch.py` — mode routing, snapshot diff capture, and run metadata.
- `agy_orchestrator/workflows/*.py` — per-mode workflow behavior and docstrings.

## Contributing
Issues and pull requests are welcome.
Run `python -m pytest -q` before submitting changes.
The CLI is the contract; flags and behavior are documented in `--help`.

## License
License: TBD

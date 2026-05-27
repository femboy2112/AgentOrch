# agy_orchestrator

An advanced multi-agent CLI orchestrator. It wraps the `agy`, `claude`, and `codex`
command-line agents behind a common interface and composes them into higher-order
workflows that are **usage-aware**, **gracefully failing**, and **verbose** so a long
autonomous run is fully tappable.

This is the merged, upgraded orchestrator: it takes the best functionality and
bugfixes from two prior lineages and combines them into one self-contained package.

## What's in the merge

Every robustness feature from both source trees, reconciled into one:

- **Cross-provider fallback with cycling** (`core/agents/fallback_agent.py`) — wrap a
  chain of providers; on a usage/quota wall the next provider is tried, and the whole
  chain is *cycled* (default 3×) so a provider whose quota has recovered gets retried.
  Usage-wall markers in stderr are detected for clearer logging.
- **Process cleanup on cancel** (`core/agents/claude_agent.py`) — the live child
  process is tracked and killed in a `finally`, so a cancelled/aborted run never
  leaves an orphaned agent CLI behind.
- **Live stderr streaming** (`core/agent.py`) — set `AGY_STREAM=1` to mirror a child's
  stderr line-by-line as it arrives, making a long build tailable instead of buffered.
- **Effort mapping** (`core/agents/codex_agent.py`) — `--effort max` maps to codex's
  `xhigh` reasoning effort.
- **Checkpoint + resume** (`workflows/master.py`) — the master workflow persists progress
  atomically and resumes in place from the next step after a crash.
- **Session compaction** (`workflows/master.py`) — over a long chained run the context
  is condensed to a bounded digest and the session reset every N steps (or when context
  exceeds a char budget), keeping token cost flat. Falls back to a recent-tail truncation
  if the compaction call itself fails.
- **Usage-aware allocation** (`core/optimizer.py`) — effort/model auto-downgrade when a
  provider's remaining usage runs low; effort baseline derives from the subscription plan.

## Install

```bash
pip install -e .          # or: pip install -e ".[dev]" for the test deps
```

Requires the underlying agent CLIs (`agy`, `claude`, `codex`) on `PATH` for live runs.

## Usage

```bash
# Run as a module (no install needed)
python -m agy_orchestrator <command> ...
# or via the console script after install
agy-orchestrator <command> ...
```

Top-level options (apply to every command) declare your subscription plans, which set
the baseline effort tier:

```
--claude-plan "20x max"  --codex-plan "$100"  --agy-plan free
```

### Commands

| Command       | What it does |
|---------------|--------------|
| `master`      | Full pipeline: plan → per-step Tree-of-Thought exploration → adversarial refinement, with checkpointing + compaction. |
| `chain`       | Linear pipeline; each stage's output is piped as context into the next. Per-stage fallback is **on by default**. |
| `adversarial` | Generator/critic loop until the critic approves (optionally gated by a programmatic test command). |
| `tot`         | Single-layer Tree-of-Thought: N parallel branches, scored by an evaluator, best selected. |

Examples:

```bash
# Plan-and-build a project end to end, resilient to any one provider running out of usage
python -m agy_orchestrator --codex-plan "$100" master \
  --prompt "Build a single-file WebGL black hole demo" \
  --agent codex --fallback \
  --checkpoint .run.ckpt --compaction-interval 6

# Refinement chain: codex -> claude, each stage wrapped in fallback (default)
python -m agy_orchestrator chain --prompt "Optimize this shader" --agents codex claude

# Adversarial loop gated by a build/test command
python -m agy_orchestrator adversarial --prompt "Fix the failing tests" --test-cmd "pytest -q"
```

Key `master` flags: `--branches`, `--max-iterations`, `--test-cmd`, `--fallback`,
`--checkpoint`, `--compaction-interval` (0 disables), `--max-context-chars` (0 disables).

## Tracking a run

Logging is INFO-level by default — planner steps, ToT branch scores, adversarial
iterations, fallback transitions, checkpoint writes, and compaction events all log as
they happen. For live child output, prefix with `AGY_STREAM=1`.

## Workflow harness

`harness/` is the operator layer used to drive workers (codex/agy) via the
orchestrator during design sessions. Every dispatch is captured for review.

```bash
python -m harness do "add a --version flag to the CLI"          # adversarial (default)
python -m harness do "rename helper X to Y" --mode direct       # one-shot
python -m harness do "build feature Z" --mode master --test-cmd "pytest -q"
python -m harness runs                                          # list recent runs
python -m harness show <run_id>                                 # print a run's diff + meta
```

Roles (all overridable via `--generator` / `--critic`, fallback on by default):

- **generator** = `codex` (code writer) → falls back to `agy` on a usage wall.
- **critic** = `agy` at high effort / best model → falls back to `codex` when agy
  usage is exhausted.

Each dispatch writes `runs/<timestamp>/`:

```
runs/<ts>/
  prompt.txt           # exact prompt sent to the worker
  stdout.log           # final workflow output
  stderr.log           # full INFO-level tracking (planner/critic/fallback/...)
  changed-files.diff   # git-independent unified diff of what changed on disk
  meta.json            # mode, roles, success, duration, changed-file lists
```

Workers write **directly into the repo**; the diff is computed by snapshotting
the tree before/after, so it works with or without git. Failures are recorded,
never raised into the operator's shell.

## Tests

```bash
python -m pytest -q
```

## Layout

```
agy_orchestrator/
  cli.py                      # argparse entrypoint + workflow runners
  core/
    agent.py                  # AgentInstance ABC: async exec, retries/backoff, AGY_STREAM
    optimizer.py              # UsageAwareAllocator: effort/model downgrade on low usage
    profile.py                # UserProfile: plan -> baseline effort
    agents/                   # agy / claude / codex adapters + fallback wrapper
  execution/
    pipeline.py               # LinearPipeline, ParallelSwarm
    tdag.py                   # TaskDAG (dependency-aware concurrent execution)
    verifier.py               # QualityVerifier (programmatic test gate)
  interaction/
    decision_engine.py        # auto/human question resolution
  workflows/
    adversarial.py            # generator/critic refinement loop
    tree_of_thought.py        # parallel branch generation + scored selection
    master.py                 # plan -> ToT -> adversarial, checkpoint + compaction
```

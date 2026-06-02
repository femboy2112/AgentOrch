# CLAUDE.md — operating guide for this project

This file is for the Claude session driving this project. Read it first.

## What this repo is

AgentOrch is a workstation for building software by **orchestrating worker CLIs**
(`codex`, `agy`, `claude`, `grok`) rather than writing all code by hand. Two layers:

1. **`agy_orchestrator/`** — the multi-agent engine. Composes cloud worker CLIs
   into workflows: adversarial (generator↔critic), tree-of-thought, linear chains,
   test-feedback (generate→run tests→repair), cascade (cheap-first escalation), and
   a master plan→ToT→adversarial pipeline. Usage-aware, fallback-resilient,
   checkpointed, verbose. Run standalone via `python -m agy_orchestrator`.
2. **`harness/`** — the operator layer **I drive** to dispatch coding work and
   capture every run. This is the primary interface for day-to-day building.

This repo is the cloud-only orchestrator: all local-LLM-runner functionality has
been stripped. The goal now is to upgrade and polish it toward a public release.

## How I work here

We (operator + I) design together; I then dispatch precise instructions to the
workers through the harness and review what they produced. I do **not** hand-write
feature code that a worker should build — I instruct, capture, verify.

```bash
python -m harness do "INSTRUCTION"            # default: adversarial (gen+critic)
python -m harness do "INSTRUCTION" --mode direct     # one-shot, fast/cheap
python -m harness do "INSTRUCTION" --mode feedback --test-cmd "pytest -q"  # repair loop
python -m harness do "INSTRUCTION" --mode master     # whole-feature build
python -m harness do "INSTRUCTION" --mode master --plan-only   # emit runs/<id>/plan.json, write nothing
python -m harness do "INSTRUCTION" --mode master --plan plan.json   # execute an edited plan verbatim (skip planner)
python -m harness do "INSTRUCTION" --mode master --plan graph.json --merge-policy reconcile  # run a dependency DAG (concurrent branches); see docs/graph-execution-design.md §9
python -m harness do "INSTRUCTION" --mode pat --test-cmd "pytest -q"        # try direct, plan on fail
python -m harness do "INSTRUCTION" --web-search      # enable codex web search
python -m harness do "INSTRUCTION" --test-cmd "pytest -q"   # quality gate
python -m harness do "INSTRUCTION" --out-dir /path/to/other/repo  # write THERE, not into AgentOrch
python -m harness runs                        # list recent runs
python -m harness show <run_id>               # print a run's diff + meta
```

**Working directory (`--out-dir`):** by default, workers run with cwd =
AgentOrch's repo root, so file-writes land here. When another agent or
orchestrator invokes AgentOrch as a tool, pass `--out-dir <path>` so the
worker writes into the caller's repo instead, and the dispatch's snapshot
diff covers that directory. The `runs/<id>/` artifacts always stay under
AgentOrch (they're orchestrator-internal logs, not user data).

**Mode by scope:** `direct` for small/precise edits we already designed;
`adversarial` (default) when quality matters and a critic pass helps; `feedback`
when a programmatic test is the oracle; `cascade` to escalate cheap→strong on
verifier failure; `master` for multi-step features; `pat` when most tasks at
this prompt scale are likely solvable in one shot but you want the master
safety net on misses (Plan-after-Trial, arxiv 2605.07248, ~40% cost saver
when Stage 1 carries). Long builds: run in the background and review the
`runs/<ts>/` artifacts when done.

**Roles (overridable with `--generator` / `--critic`):**
- generator = **codex** writes code → falls back to agy on a usage wall.
- critic = **agy** (high effort, best model) reviews → falls back to codex when
  agy usage is exhausted.

Workers: `codex`, `claude`, `agy`, `grok`. Comma-separate to build a fallback
chain (e.g. `--generator codex,agy`).

**Every dispatch is captured** under `runs/<timestamp>/`: `prompt.txt`,
`stdout.log`, `stderr.log` (full INFO tracking), `changed-files.diff`
(git-independent), `meta.json`. Workers write **directly into the repo**.

Caveat: the diff is a before/after snapshot of the whole tree, so don't create
unrelated files **during** a running dispatch or they'll show up in its diff.

## Keep it lean (anti-bureaucracy)

This workflow tends to over-produce documents. Defaults that fight that:

- **Specs are scope-gated and ephemeral.** Write a standing design doc only for
  a genuinely large or ambiguous feature; for everything else the focused
  instruction *is* the spec. When you do write one it lives in `runs/<id>/spec.md`
  — promote it to a tracked `docs/` doc only if it'll be referenced repeatedly,
  and move it to `docs/archive/` once the feature merges (git keeps the history).
- **One doc per feature, no sidecars.** Fold "validation"/review into a short
  Risks / Open-Questions section of the doc, or do it inline — don't spawn a
  separate `*.VALIDATION.md`.
- **Memory is a working set, not a changelog.** Git records what shipped; memory
  records what's true *now* and what's still pending. Prune DONE/SUPERSEDED
  entries on sight.
- **Instructions are focused PR descriptions, not banner walls.** Cap dispatch
  prompts; inject a long spec with `--spec <file>` instead of inlining 300 lines.
  Reserve invariant-lists for genuinely invariant-critical builds.
- **Default to `pat`/`direct`.** Escalate to `master` (+ a spec) only when the
  task truly spans many distinct steps.

## Account-sharing rule (avoid usage-wall cascades)

**Don't run the orchestrator's worker on the same provider+account as the agent
that's calling it, unless you have to.** A claude-coding session driving the
harness to dispatch claude workers shares one Claude API pool: when the workers
exhaust it, the dispatcher exhausts at the same instant and the whole stack
goes down together. Same logic for sibling orchestrators that target a shared
pool — coordinate model choice across them.

Concretely, when picking generator/critic for a dispatch:
- If the operator's driving agent is `claude`, prefer `codex`/`agy`/`grok` workers.
- If you must reuse the same provider (e.g. only claude is configured), keep
  the worker on a **different model tier** so usage walls are independent — and
  rely on the `--fallback` chain (default on) to roll over when one walls.
- The `_artifact_research_orchestration.md` dossier discusses this under
  "compute-optimal allocation"; the practical version is: shared pool = shared
  failure mode.

## Layout

```
agy_orchestrator/
  cli.py                  # argparse entrypoint + workflow runners
  core/
    agent.py              # AgentInstance ABC: async exec, retries/backoff, AGY_STREAM
    optimizer.py          # UsageAwareAllocator: effort/model downgrade on low usage
    profile.py            # UserProfile: plan -> baseline effort
    agents/               # agy / claude / codex / grok adapters + fallback wrapper
  execution/
    pipeline.py           # LinearPipeline, ParallelSwarm
    tdag.py               # TaskDAG (dependency-aware concurrent execution)
    verifier.py           # QualityVerifier (programmatic test gate)
    ledger.py             # quality-cost ledger (per-run confidence signals)
  interaction/
    decision_engine.py    # auto/human question resolution
  workflows/
    adversarial.py        # generator/critic refinement loop
    tree_of_thought.py    # parallel branch generation + scored/vote selection
    decompose.py          # as-needed recursive decomposition (ADaPT-style)
    generate_and_rank.py  # best-of-K ranking
    test_feedback.py      # generate -> run tests -> feed error back -> repair
    cascade.py            # cheap-first escalation gated by a verifier
    master.py             # plan -> ToT -> adversarial, checkpoint + compaction
harness/
  cli.py / dispatch.py / roles.py / snapshot.py   # operator layer
scripts/
  cloud_eval.py           # cloud-worker code-quality bench (hidden-pytest graded)
```

## Tests

`python -m pytest -q` — orchestrator workflows + harness (snapshot/dispatch/roles).

## Project status

Cloud orchestrator extracted from the prior Agent64 lineage with all local-LLM
runner code removed. Now polishing toward release: tighten the CLIs, docs, and
test coverage; keep every dispatch captured and reviewable.

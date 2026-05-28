# AGENTS.md — for LLM agents encountering this repo

You're an LLM agent (Claude Code, Codex, Antigravity/agy, Grok, or similar)
looking at this repo. There are **two distinct ways** you might be here:

1. You are **integrating AgentOrch into your own workflow** as a tool you can
   call to delegate coding work. → [Read § Integrating from outside](#integrating-from-outside).
2. You were **dispatched as a worker** inside this repo by AgentOrch itself,
   and you need to know the local rules. → [Read § If you're a worker inside this repo](#if-youre-a-worker-inside-this-repo).

If you can call subprocesses, write files into another repo, and the user's
prompt mentions AgentOrch as a tool you should use, it's case 1.

---

# Integrating from outside

AgentOrch is a workstation orchestrator: you instruct it, it dispatches the
real work to one or more CLI workers (`codex`, `claude`, `agy`, `grok`),
gates the result with a programmatic test, and hands you back a structured
result with a diff. It is **complementary** to you, not competitive. You
think and review; AgentOrch executes precise instructions you've already
designed.

## When to use AgentOrch (and when not to)

**Use it for** work that is well-specified and parallelizable, when delegating
saves you from spending your own context window on rote implementation:

- multi-file mechanical refactors (rename a symbol, change a signature
  everywhere, port a pattern across N call sites)
- speculative changes you want graded against a test suite before you commit
  to one (`--mode vote`)
- whole-feature builds with a clear spec and a verifier (`--mode master` or
  `--mode pat`)
- background batches where you want to keep working while the dispatch runs

**Skip it for** work where your own context, taste, or step-by-step
reasoning is the value — small edits you've already designed, exploratory
debugging, anything where you'd lose more by relaying than by doing.

## The shape of one dispatch

```bash
python -m harness do "INSTRUCTION" \
  --mode <direct|adversarial|feedback|cascade|master|pat|vote> \
  --generator codex,agy,grok \
  --critic agy,codex,grok \
  --test-cmd "<your verifier command>" \
  --out-dir <path-to-target-repo>
```

Every dispatch produces a `runs/<timestamp>/` directory you can inspect:
`prompt.txt`, `stdout.log`, `stderr.log`, `changed-files.diff`, `meta.json`,
`events.jsonl`. The `meta.json` carries a `quality` ledger with a
confidence label (`verified` / `approved` / `unverified` / `failed`). Treat
anything below `verified` as a draft for your review.

## Pick the right mode

| Mode | Shape | Verifier required | When |
|---|---|---|---|
| `direct` | one shot, no critic | no | small precise edits you've designed |
| `adversarial` (default) | generator + critic loop | no | quality matters, no test suite handy |
| `feedback` | generator + run-tests-and-repair | **yes** | test is the oracle, simple repair loop |
| `cascade` | cheap → strong escalation on verifier failure | **yes** | unknown task difficulty; minimize cost |
| `master` | plan → tree-of-thought → adversarial per step | no (but advised) | whole features, multi-step, long context |
| `pat` | direct first, escalate to master only on failure | **yes** | most tasks; ~40% cheaper than master when direct carries (arxiv 2605.07248) |
| `vote` | K parallel candidates in isolated workspaces, verifier picks winner | **yes** | high-quality candidate selection; K=`--branches`; heterogeneous when chain has multiple providers |
| `auto` | RoutingPolicy picks the concrete mode | no | When you want explainable routing without hand-picking |

If you don't have a test you can run, use `adversarial`. Otherwise prefer
`pat` for general work, `vote` when you specifically want the strongest
candidate, `master` only when the task spans many distinct steps.

## The two foot-guns to avoid

### 1. Account-sharing rule

**Don't pick worker models that share an account/quota pool with the agent
calling the orchestrator.** When the workers hit a usage wall, the calling
agent walls at the same instant, and the whole stack goes down together.

Concretely: if you are Claude Code, don't dispatch with `--generator claude`
or `--critic claude` unless you have to. Default chains
(`--generator codex,agy,grok / --critic agy,codex,grok`) deliberately exclude
Claude for exactly this reason. The same applies in reverse: if you are
Codex, prefer chains that don't lead with Codex.

If you must reuse the same provider (e.g. it's the only one configured),
keep the worker on a **different model tier** so usage walls are independent,
and rely on the `--fallback` chain to roll over. AgentOrch will also warn
when generator and critic chains share a provider family (the
"cross-family verifier guard").

### 2. Write to the right place — use `--out-dir`

By default the harness runs workers with cwd = AgentOrch's own repo root.
**That's the wrong default if you're dispatching from another repo.** Pass
`--out-dir /path/to/your/repo` and the workers will write there instead of
polluting AgentOrch. The dispatch's snapshot diff scopes to `--out-dir`
correctly; the `runs/<id>/` artifacts always stay under AgentOrch (they're
orchestrator-internal logs).

```bash
# WRONG (from another repo): workers write into AgentOrch
python -m harness do "fix the bug" --test-cmd "pytest -q"

# RIGHT: workers write into your actual project
python -m harness do "fix the bug" \
  --test-cmd "pytest -q" \
  --out-dir "$(pwd)"
```

## Writing a good instruction

The instruction text becomes the worker's primary prompt. Treat it like a
focused PR description, not a vague task title.

**Good shape:**

> Add a `--dry-run` flag to `scripts/sync.py` that prints what would be
> uploaded but does not call S3. Mirror the existing `--verbose` flag's
> parsing and help text. Don't change the default behavior; the flag must
> be opt-in. The verifier (`pytest -q tests/test_sync.py`) covers the
> existing behavior; add `test_dry_run_prints_paths_but_does_not_upload`
> that uses `moto`'s S3 mock and asserts zero `put_object` calls.

**Bad shape:**

> make the sync script better

Specifics worth including:
- exact file paths you expect to be touched
- the test command that should pass (so the verifier can gate it)
- explicit non-goals ("don't refactor adjacent code", "don't bump deps")
- any naming conventions, code style anchors, or existing patterns to mirror

## Reviewing the result

After the dispatch returns, read these in order:

1. `runs/<id>/meta.json` → `quality.confidence`. `verified` = a real test
   passed. `approved` = LLM critic said so (weaker; only as good as the
   critic). Anything else: review carefully.
2. `runs/<id>/changed-files.diff` → the entire disk delta. Read it.
3. `runs/<id>/stdout.log` and `stderr.log` if anything looks surprising.
4. `runs/<id>/events.jsonl` for the per-worker reasoning stream — useful
   when you want to know *why* the worker took an approach you didn't
   expect.

If you don't trust the diff, your options are:
- Re-dispatch with a stricter test command
- Re-dispatch with `--mode vote --branches 5` to grade K alternatives
- Take the diff as a starting point and edit it yourself

## Long dispatches

`master`-mode dispatches can take hours. Patterns that work:

- Fire in the background and review when notified.
- Watch the live dashboard: `python -m harness dashboard` boots a local
  FastAPI on `127.0.0.1:8765` with live SSE streams per worker. **No
  browser opens by default** — pass `--browser` to auto-open.

## Calibration and the live ledger

The streaming watchdog kills workers that emit runaway output or stall.
Per-config budgets come from a `CalibrationTable` that reads
`/tmp/agentorch_research/calibrate.jsonl` (offline benchmark) AND
`/tmp/agentorch_research/live_ledger.jsonl` (every verified dispatch
appends one row). The live ledger is the feedback loop: every successful
dispatch improves the watchdog's accuracy on the next one.

Disable either via `AGY_LIVE_LEDGER=off` or `AGY_WATCHDOG=off`. The
defaults are conservative and won't false-positive on any successful run
we've measured.

## The minimum integration recipe

If you want to add AgentOrch to your toolbox with the smallest possible
code, expose this single function to yourself:

```python
def dispatch(instruction: str, *, test_cmd: str, out_dir: str | None = None,
             mode: str = "pat") -> dict:
    """Run one dispatch and return the meta.json contents."""
    import json, re, subprocess, sys
    from pathlib import Path
    args = [sys.executable, "-m", "harness", "do", instruction,
            "--mode", mode, "--test-cmd", test_cmd]
    if out_dir:
        args.extend(["--out-dir", out_dir])
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    m = re.search(r"dispatch (\d{8}-\d{6}-\d+)", result.stdout)
    if not m:
        raise RuntimeError(f"dispatch failed: {result.stderr[-2000:]}")
    run_id = m.group(1)
    # runs/<id>/meta.json lives under AgentOrch's repo root, NOT out_dir.
    agentorch_root = Path(__file__).resolve().parent  # adjust to where AgentOrch lives
    meta = agentorch_root / "runs" / run_id / "meta.json"
    return json.loads(meta.read_text())
```

Call it with `out_dir=os.getcwd()`. Branch on
`result["quality"]["confidence"]` to decide whether to accept the diff
straight away or hand it to the human.

## Where the source of truth lives

- `CLAUDE.md` — operator/maintainer guide for AgentOrch itself (long form).
- `docs/experiments.md` — empirical findings that motivated the current
  defaults.
- `docs/dashboard-design.md` — v1 dashboard spec.
- `agy_orchestrator/workflows/*.py` — every mode is one short file; read
  the docstring of the mode you're about to use.
- `harness/cli.py` — every flag documented in `--help`.

When in doubt, run `python -m harness do --help` and skim. The CLI is
the contract.

---

# If you're a worker inside this repo

You were dispatched by AgentOrch to make changes inside this repository.

## Hard rules

- Make changes **directly on disk** in the current working directory. Implement
  the instruction; do not merely describe it and do not ask clarifying questions.
- **Never run `sudo`.** Never modify anything outside the folder you were told to
  work in.
- Prefer **stdlib / zero-dependency** solutions unless told otherwise. If a
  dependency is essential, state why.
- Match the existing code style of nearby files. Keep changes minimal and scoped.
- End your reply with a short list of files you created/modified and why.

## Verification

If a test or build command applies, ensure it passes before finishing. The
project tests run with `python -m pytest -q`.

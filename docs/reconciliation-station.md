# Reconciliation / Integration-Skeptic station (issue #43)

A third, orthogonal quality station that holds **both the goal and the running
system** and is mandated to *diff* them. It runs **after** a build converges and
the verifier is GREEN, traces each spec-named mechanism to the live execution
path, and surfaces anything that "exists but isn't load-bearing."

## The failure class: exists-but-not-load-bearing

A component can have real code, correct names, sensible structure, and a **passing
unit test**, yet in the live path / at-scale eval be one of:

| sub_kind | meaning |
|---|---|
| `uncalled` | defined, never invoked on the real path |
| `stub_constant` | hardcoded return (e.g. `surprise() -> 0.0`) shadowing a real impl |
| `untrained` | random-init weights, never in an optimizer / never `backward()` |
| `bypassed_proxy` | learned signal silently replaced by a cruder count/proxy |
| `mocked_none` | `None`/default-off, never enabled at scale |
| `saturated` | near-constant, non-discriminative output for all inputs |

Each passes the test suite. The build is "100% verified" at every step, and the
defect survives repeated increments because nothing ever compares the *accreting
whole* to the *destination*.

## Why the current loop can't catch it

It is a **missing role**, not a tuning/prompt problem. No agent in the loop holds
both sides with a mandate to reconcile them:

| Role | Holds the goal? | Observes the *running* system? | Mandated to diff them? |
|------|:---:|:---:|:---:|
| Planner (`decompose.py`, `master.py`) | yes (spec text) | no | no |
| Implementer / generator | yes (+ local diff) | no (writes code, doesn't run+introspect the whole) | no |
| `QualityVerifier` (`execution/verifier.py`) | no (only `test_commands`) | partial (runs tests) | no |

Two factors make it survive *repeated* runs: (1) the verifier is goal-blind and
strictly downstream of the tests — which were written by the same process that
wrote the dead code, so they encode the same blind spot; (2) plausibility-checking
(what an LLM critic defaults to) is a different cognitive act than execution-tracing
— dead-but-present code passes a plausibility check effortlessly.

`AdversarialReview` does **not** cover this: its `CATASTROPHIC_FOCUS_PREAMBLE` aims
the critic at resource-exhaustion/crash flaws (dead wiring is safe + plausible), and
it reviews the *static artifact* for an `APPROVED` convergence, never running or
tracing the assembled system. `TestFeedbackWorkflow` is, by design, the programmatic
verifier — equally goal-blind.

## The station's mandate

For **each mechanism the goal names as load-bearing**, the agent must TRACE whether
it is on the live path (find the real call site, or declare it uncalled) — *not*
review plausibility. It must produce `file:line` evidence and classify each as:

- `load_bearing` — invoked on the real path AND ablating it would change a live/
  at-scale witness;
- `exists_not_load_bearing` — present + unit-tested but dead, with a required
  `sub_kind` (one of the six above) and `file:line`;
- `absent` — the goal requires it and it does not exist.

### Load-bearing witness (the xfail discriminator)

Per mechanism the agent reports an **ablation witness**: ablate the component →
does a live/at-scale signal move? This is the discriminator the loop lacks:

- **witness == 0** → wiring is **dead** (a bug masquerading as honest-incomplete) →
  surface loudly;
- **witness > 0, metric still low** → **legitimately incomplete** (wired, science/
  training not there yet) → acceptable, not flagged.

Without it, dead wiring hides behind legitimate expected-failures indefinitely.

### Independence constraint

The station is **read-only** w.r.t. the build artifact and the test suite — its
prompt forbids editing code/tests; it may only run / trace / probe. It cannot
satisfy itself by moving the goalposts. Its verdict is a **distinct status**,
reported alongside (NEVER folded into) `VerifierResult.ok`.

### Warn-only default

Default `disposition="warn"`: report loudly as a distinct status + a durable
artifact, but do **not** fail the run — this is the default *even under
`--mission-critical`* (mission-critical only turns the station *on*). `"fail"`
(flip the run to failed when not reconciled) and `"open-task"` (warn + recommend
a follow-up build task; findings recorded, not auto-filed outward) are opt-in.

## Public API (`agy_orchestrator/workflows/reconcile.py`)

- `ReconciliationReview(agent, goal, verifier=None, working_directory=".",
  disposition="warn", diff=None, event_callback=None, max_iterations=1)` —
  `execute()` (async) returns a `ReconciliationResult`. Constructor mirrors
  `AdversarialReview`'s shape; the single LLM call is awaited via `agent.run_async()`.
- `ReconciliationResult` — dataclass with `reconciled: bool` (True iff no
  `exists_not_load_bearing` findings; `absent` does not flip it), `disposition`,
  `findings()` (the dead set), `absent_findings()`, `load_bearing_findings()`,
  `.verdict` (`"reconciled"`/`"not_reconciled"`), `.should_fail_run`, `to_dict()`
  (for the artifact). Deliberately not a `VerifierResult` and not auto-truthy.
- `MechanismFinding` (name, classification, sub_kind, location, witness, evidence)
  and `Witness` (value, description, `.is_dead`).

Output parsing tolerates ```json fences, `<think>` blocks, and surrounding prose;
`reconciled` is **recomputed from the findings** (a model can't bless dead wiring),
and an unparseable reply is treated as *not* reconciled.

## Wiring (implemented in `harness/dispatch.py`)

Rather than wiring into `master.py` alone, the station is invoked once in
`dispatch_async`, **post-convergence**, so it covers *every* LLM mode (`do`,
`master`, `pat`, `adversarial`, …) uniformly — not just master:

1. **Trigger**: after `monitor.run(_run_workflow(...))` returns, when the build
   converged (`output` non-empty) and — if a verifier gated it — the workflow is
   GREEN (`workflow.verified`). Runs inside the try block while the event bus is
   still open, so its trace events stream + persist to `events.jsonl` normally.
   Wrapped best-effort: a station error never fails an otherwise-good build.
2. **Gating** (`reconcile_enabled`): `--reconcile` OR `AGY_RECONCILE=1` OR
   `--mission-critical` (default-on for mission-critical).
3. **Independent reviewer**: `_run_reconciliation` builds the agent from the
   **critic chain** (a cross-provider reviewer, distinct from the generator that
   wrote the code — same independence principle as the adversarial critic), so it
   honours `--critic`/`--critic-effort`/`--critic-model` and the account-sharing
   rule follows from the operator's chain choice.
4. **Goal**: the approved `--spec` text if present, else the bare instruction.
5. **Disposition** (`--reconcile-disposition`, default `warn`): `warn` records +
   logs and **never fails the run — including under `--mission-critical`** (the
   operator's chosen default; a hard gate is explicit opt-in via `fail`). `fail`
   flips `success` when not reconciled. `open-task` warns + logs a follow-up-task
   recommendation but does **not** auto-file an outward GitHub issue (that's an
   outward action left to the operator; the findings live in `reconcile.json`).
6. **Persistence**: `result.to_dict()` → `runs/<id>/reconcile.json`; the verdict
   is also attached to `DispatchResult.reconciliation` (→ `meta.json`) and shown
   in the CLI result summary. It is **never** folded into `success`/the verifier
   under the default disposition.

## Risks / open questions

- The station's accuracy is bounded by the agent's ability to actually run + trace
  the assembled system; for repos with no runnable entry point it degrades to
  static tracing (still useful for `uncalled`/`stub_constant`, weaker for
  `untrained`/`saturated`). The witness is agent-reported, not independently
  recomputed here — a future increment could have the integrator execute a real
  ablation probe and cross-check.
- Per-increment invocation is cheap (scoped to spec-named mechanisms); a periodic
  full-tree sweep (the manual audit, demoted to routine) is a follow-up.

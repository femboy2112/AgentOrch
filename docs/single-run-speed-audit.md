# Single-run speed audit — increase speed without reducing output quality

Branch: `perf/single-run-speed` (isolated worktree). Read-only audit, 2026-06-01.
Method: 5 mapping subagents (claude pool, read-only) → 22 candidate speed-ups →
adversarial skeptic per finding (verify the file:line, is the speed real, does it
reduce quality, is it safe vs the live shared pool) → synthesis. 28 agents, zero
errored. No codex/agy/grok worker was spawned (the live mission-critical agi2
dispatch on that pool was never contended).

## Headline answer to the need-to-know question

**"Is a coding instance getting the entire spec when it only needs the ABI surface
+ goal/constraints?"** — Yes, confirmed with evidence, but the naive fix is NOT
quality-safe:

- `_build_prompt` (`harness/dispatch.py:336`) appends the full spec verbatim.
- master seeds `project_context` with the entire `initial_prompt` (spec included,
  `master.py:449`), re-embeds it into every `step_prompt` (`master.py:469-473`),
  which fans out to 3 ToT branches (`485-488`) + the adversarial generator (`512`)
  + the critic via `initial_prompt` (`adversarial.py:175-177`), re-sent every
  adversarial iteration. `_compact_context` (`master.py:194`) **re-inserts** the
  full `initial_prompt`, so the spec rides along the WHOLE run, not just until first
  compaction.
- **But** FloodSpec puts load-bearing rules in Requirements, Constraints &
  Guardrails, and Data Models — not only Components & Interfaces (`spec.py:50-67`).
  Projecting an implementer/critic to an "interface surface" risks silently dropping
  a normative constraint that lives in another section — exactly a critic-blinding /
  implementer-under-informing regression. Quality-safe ONLY if the projection
  retains all normative sections and strips only true prose (Overview / Alternatives
  / Open-Questions / Assumptions), and the planner always gets the full spec.
- The caching half of the win is **near-dead on the default `codex,agy` pool** —
  only claude/grok adapters have `--resume`; codex/agy re-process the prefix every
  call regardless.

So: the bloat is real, but the biggest *quality-safe* wins are elsewhere.

## Ranked roadmap (quality-safe first)

| # | Win | Impact | Safe? | Effort | Rec |
|---|-----|--------|-------|--------|-----|
| 1 | **Skip pre-run baseline verifier on non-vote modes** (`dispatch.py:1051-1053`) | **high** | ✅ | small | ship |
| 2 | Surface `cache_read_ratio` per-run in meta.json (`dispatch.py:300-318`) | low | ✅ | small | ship |
| 3 | Decouple watchdog 2s poll from completion detection (`agent.py:360-393`) | low | ✅ | small | ship-with-care |
| 4 | Strip `WORKER_PREAMBLE` from the standalone critic view (`adversarial.py:177`) | low | ✅ | small | ship-with-care |
| 5 | Give the ToT judge requirement context (`tree_of_thought.py:13-26,166-170`) | low (quality+) | ✅ | med | ship-with-care |
| 6 | Section-aware context projection for master spec re-inflation | med | ⚠️ | med | investigate |
| 7 | Concurrent independent master steps via TaskDAG | med | ❌ | large | investigate |
| 8 | Warm `--resume` session through standalone adversarial | low | ✅ | med | investigate (claude/grok-led only) |

### Rank 1 — the headline win (high impact, small, quality-safe)
`dispatch.py:1051-1053` unconditionally runs the **full `--test-cmd` suite** on the
unchanged tree before any worker starts. Every consumer of `baseline_result` was
traced: vote mode (`630`, the only real consumer), `_derive_verifier_delta` (`1181`,
telemetry), 3 meta fields (`1206-1208`), an OOM-vs-fail notification label (`1278`).
For adversarial/feedback/pat/cascade/master the in-loop verifier is the real gate;
the upfront baseline only feeds telemetry. Removing it for non-vote modes strips an
entire serial pytest/make-check (verifier default timeout 600s; the code itself
notes `pytest -n auto + mypy + ruff ≈ minutes`) off the critical path with **zero**
effect on produced code or review signal. Keep behind `--baseline-gate` for users
who want the delta telemetry.

### Why ranks 6-7 are gated
- **6 (projection):** real token waste, but needs a section-aware extractor that
  retains all normative FloodSpec sections; no such extractor exists yet, and a
  fail-open "pass full spec when no interface section" does NOT catch the dangerous
  case (interface section present, constraint elsewhere dropped).
- **7 (master concurrency):** the dead `TaskDAG` exists, but master steps share ONE
  mutable `working_directory` (`master.py:73→532`) where workers write files directly
  AND the in-loop verifier runs `--test-cmd` (`adversarial.py:136`). Parallelizing
  races file writes and runs step A's verifier against a tree half-mutated by step B
  — corrupting both artifact and signal. Needs vote-style per-step workspace
  isolation first; uncapped, it also multiplies in-flight ToT swarms onto the shared
  pool. `max_parallel_workers` is wired to vote ONLY (`dispatch.py:637`), not master.

## Benchmark harness (the comprehensive testing workflow)

`scripts/speed_bench.py`, two-tier, structurally unable to touch the live pool:

- **Tier 1 (hermetic, primary):** `MockAgent(AgentInstance)` returns canned output
  after a deterministic, optionally prompt-size-scaled sleep, and emits a synthetic
  usage event computed from the real prompt it was handed. `AGY_BENCH_MOCK=1` makes
  the role/agent factory resolve to `MockAgent` (one factory indirection, not
  scattered edits), so EVERY harness path (`dispatch_async`, prompt construction,
  projection, baseline-verify, checkpoint, meta.json) runs end-to-end with **zero
  network and zero pool contention**. Harness overhead is computed exactly:
  `wall-clock − Σ(mock sleeps)`.
  - The watchdog-tail finding needs a **real** short-lived subprocess (`printf;
    exit 0`) so `_stream_communicate`'s gather+watchdog timing is authentic — a pure
    in-memory mock would falsely report zero tail.
- **Tier 2 (optional, default OFF):** a SINGLE claude/grok worker on a SEPARATE
  account/pool, only to validate cache_read telemetry + warm-session prefix hits on
  a handful of runs. Gated behind an env flag; never the codex/agy pool, never
  concurrent with the mission-critical dispatch.

**Metrics:** total wall-clock; harness overhead (wall-clock − mock model-time);
summed `input_tokens` across all calls (noise-free proxy for prompt-bloat findings);
`cache_read_ratio`; per-phase breakdown (baseline-verify / planner / per-step ToT /
adversarial / summarize / checkpoint); worker-call count; watchdog subprocess
exit→observation delta; isolated baseline-verify duration.

**Fixed corpus** under `scripts/bench_fixtures/`: 2-3 canned instructions + 1
representative FloodSpec doc, so inputs never drift. Run N≥10 per arm, report p50/p95,
vary ONE knob per arm, prefer the noise-free token-count metric where the mechanism
is token reduction.

### Pitfalls
- A constant-latency mock hides the prefill-cost mechanism behind token-reduction
  findings — make sleep optionally scale with `len(prompt)`, keep a constant mode too.
- `cache_read_ratio` is undefined when codex reports `input_tokens=None`
  (`codex_agent.py:106,113`); providers differ on cache-inclusive vs -exclusive
  input — normalize + document or the ratio misleads the A/B it gates.
- Latency jitter (esp. the concurrent live run on this box) swamps the ≤2s watchdog
  tail — pin to a quiet window, prefer token counts where possible.
- The projection bench must A/B the ACTUAL implemented extractor (incl. fail-open),
  not an idealized one, or it overstates the win.

## Open questions
1. Can a section-aware extractor reliably keep all normative FloodSpec sections and
   strip only prose? Without it the projection token win is unrealizable safely.
2. What fraction of a real spec is prose vs normative? If interface+requirements+
   constraints is most of it, the projection win is small.
3. Is vote-style per-step isolation feasible without a costly per-step venv, and how
   often do real plans contain independent step groups (vs linear chains)?
4. Does any default-pool (codex/agy) run benefit from warm sessions, or is it
   claude/grok-only? The cache-telemetry metric (rank 2) answers this empirically.

# Validation addendum — dashboard redesign design

**Validator:** driving agent (Claude), 2026-05-30. **Status: design APPROVED for build**, with
the verified facts + one risk reframing below. This file is authoritative where it
contradicts guesses in `dashboard-redesign-design.md`; the design's architecture and
constraint coverage stand. Every claim here was checked against `main` (HEAD `26cc012`).

## Verified TRUE (safe to build on)

- **SSE transport.** `GET /api/sse/{run_id}` exists, emits `event: worker_event` per event
  and a final `event: done` carrying `meta.json`. Resume works via BOTH `?after_id=` query
  param AND `Last-Event-ID` header. Replays `runs/<id>/events.jsonl` first, then live bus.
  → `dashboard/routers/live.py:148-150`, `:114-145`, `:118-125`. The design's transport
  section is accurate; reuse it unchanged (no new transport — honors the constraint).

- **`out_dir` backend is DONE.** `dispatch()` / `dispatch_async()` already accept
  `out_dir` and fully implement it (worker cwd + snapshot diff scope; artifacts stay under
  AgentOrch). → `harness/dispatch.py:743`, `:360`, `:421-424`, `:490-491`.
  **So absorbing #28 is a thin frontend+router job, NOT backend work:**
  1. add `out_dir: Optional[str] = None` to `DispatchRequest` → `dashboard/routers/dispatch.py:19-29`
  2. pass `out_dir=payload.out_dir` in the `dispatch()` call → `dashboard/routers/dispatch.py:44-58`
  3. add the text input to the Dispatch page form.
  Backend already validates/creates the dir. Do not re-implement it.

- **Additive `data.orchestration` is genuinely safe.** Live events are plain dicts with a
  free-form `data` field; legacy renderers ignore unknown keys. → `dashboard/event_bus.py:52`.

- **CRITICAL CORRECTNESS GUARD — use `kind:"lifecycle"`, never a new top-level kind.**
  `event_bus.py` whitelists `kind` to exactly these 8: {lifecycle, reasoning, message,
  tool_call, tool_result, usage, stderr, watchdog} (`event_bus.py:9`, verified at runtime).
  **Any unrecognized kind is silently downgraded to `"stderr"`** (`dashboard/event_bus.py:50-51`)
  — note this even bites the existing `computer_use_event` payloads, which are NOT whitelisted. The design's choice
  (§5, §10) to carry orchestration data as `data.orchestration` inside a `lifecycle` event
  is therefore REQUIRED, not just preferred. A builder who invents `kind:"orchestration"`
  will see every diagram event vanish into the stderr lane. Call this out in the build prompt.

- **All 8 dispatch modes exist** in the harness (`direct/adversarial/feedback/cascade/
  master/pat/vote/auto` → `harness/dispatch.py:179-340`), but the dashboard API only exposes
  5 (`dispatch.py:21`). The redesign may keep the 5-mode set; if it surfaces `pat/vote/auto`
  it must also widen the `DispatchRequest.mode` Literal. Not a blocker — just don't claim the
  dashboard supports 8 today.

## THE LOAD-BEARING RISK (design understates this — read before building)

The user-friendly mode's entire value is the live plan diagram:
`plan → ToT branch pick → adversarial iter k/N → step N/M → fallback/reroute`.
**NONE of these transitions emit a structured event today. They are all `logger.info()`
only, and the orchestration layer has NO event sink wired into it at all.**

Verified absence of any `event_callback`/`_emit_event` in:
- `agy_orchestrator/workflows/master.py` — plan + step loop: log-only (`:245,:276,:286,:352,:395`)
- `agy_orchestrator/workflows/tree_of_thought.py` — branch selection: log-only (`:124`)
- `agy_orchestrator/workflows/adversarial.py` — iteration k/N: log-only (`:85`)
- `agy_orchestrator/core/agents/fallback_agent.py` — provider rollover: log-only (`:169,:184,:209`)

Today's events come only from the LEAF `AgentInstance` (`agent.py:348/451` →
`agent_started`/`agent_finished`, plus stderr/watchdog). The event callback is attached to
leaf agents in `harness/dispatch.py:476-482` (`EVENT_BUS.publisher_for(...)`). The
**workflow objects never receive a callback**. So out of the box the diagram could only show
anonymous start/finish pairs — it could not distinguish planner vs ToT-branch-2 vs
critic-iteration-3. That is the difference between "a real plan diagram" and "a useless
blinking list," and it is the CENTERPIECE of the build, not a footnote.

### What the build must actually do (the real backend task)

1. **Thread an orchestration event sink from `dispatch_async` into the workflow layer.**
   `dispatch_async` already owns `EVENT_BUS.publisher_for(run_id, ...)`
   (`harness/dispatch.py:502`). Pass a publisher (or a small `emit(transition)` callable)
   into `MasterWorkflow(...)` (`dispatch.py:261-269`), and have Master forward it into the
   `TreeOfThought(...)` and `AdversarialReview(...)` it constructs (`master.py:314,:342`).
   Mirror the existing leaf pattern; do not invent a second transport.

2. **Emit `kind:"lifecycle"` + `data.orchestration: OrchestrationTransition`** (design §6
   shape is good) at exactly these points:
   - master plan parsed → `phase:"plan", action:"completed", step_total:len(tasks)` (`master.py:276`)
   - each step start/end → `phase:"step", step_index:i+1, step_total:N` (`master.py:286,:352`)
   - ToT branch chosen → `phase:"tot", action:"branch_selected", selected_branch:winner` (`tree_of_thought.py:124`)
   - adversarial iter → `phase:"adversarial", iteration:k, iteration_total:max` (`adversarial.py:85`)
   - fallback rollover → `phase:"fallback", action:"reroute", from_worker, to_worker, reason` (`fallback_agent.py:184/209`)

3. **Keep it backward-compatible (design FR-07/FR-11).** Old `events.jsonl` lacks
   `data.orchestration`; the friendly renderer must fall back to an inferred timeline and
   label it `inferred` rather than erroring. The professional mode is unaffected either way.

4. **Add focused tests** for the new emission (one per transition type) and for the
   legacy-replay degraded path. Existing dashboard tests must stay green.

This plumbing is roughly half the backend effort of P2. The build prompt for #50 must name
it as the primary task, or the worker will under-build it and the headline feature ships hollow.

## Dogfood (design §8) — realistic, with one caveat

OBSERVE-mode read + REAL-mode drive of the dashboard is exactly the capability just merged in
P1.x (verified: agent reads 15 DOM elements + drives SPA nav). One caveat to carry forward:
P1.x perceive is verified on a **headed browser on `:0`**; the fully-display-isolated private
Xvfb path hangs `page.evaluate` headless-on-swiftshader. So the dogfood run must target a
headed `:0` session (design A-03 already assumes an accessible `:0` — good). See memory
`bing-e2e-deferred`.

## Bottom line for #50 (build)

Build it. The design is architecturally correct and honors every constraint. Two must-dos
for the build dispatch prompt: (a) make the **workflow→event plumbing** the explicit primary
task with the file:line targets above; (b) hard-require `kind:"lifecycle"` carriage so events
aren't swallowed by the stderr-downgrade. `out_dir` is mostly done; dogfood targets headed `:0`.

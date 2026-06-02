<!--
Provenance: drafted 2026-06-01 via the `design-graph-execution` fan-out workflow
(4 parallel design agents -> synthesis). The isolation/merge sub-designer failed to
return structured output; the synthesis reconstructed §3.5/§4 from the grounding +
vote.py reuse, so M4 (the merge subsystem) is the LEAST independently-validated phase
— give it a dedicated design pass before coding it.
-->

> **STATUS: BUILT (M1–M5 shipped, 2026-06-02).** The operator-confirmed
> `reconcile` default (§4) is the v1 merge policy. Graph execution runs end-to-end
> from `harness do --mode master --plan <graph.json>`. See the
> **[Operator guide](#9-operator-guide-built)** below for the concrete shape,
> flags, and round-trip. The remainder of this doc is the original build plan,
> kept for design rationale; deviations are noted in §9.

# Build Plan: Complex Task-Graph (Dependency DAG) Execution for AgentOrch

## 1. Goal & "what exists today vs the gap"

**Goal.** Make AgentOrch build *and execute* complex task graphs — dependency DAGs with concurrent branches — end-to-end from the operator path (`python -m harness do ... --plan graph.json`), reusing the master per-step quality machinery (ToT + adversarial + verifier + checkpoint) for each node, isolating concurrent nodes the way `vote` already isolates candidates, and merging their file writes safely. A flat plan must stay a byte-for-byte linear chain.

**What exists today (verified by reading):**
- The only task path reachable from `harness do` is `MasterWorkflow.execute` (`master.py:432`). It builds a flat `tasks: List[str]` (from the planner, or from an injected `--plan`) and runs the strictly sequential loop `for i in range(start_index, len(tasks))` at **`master.py:499`**. Each step runs Phase A ToT (`master.py:519-548`), Phase B `AdversarialReview` + verifier (`master.py:550-584`), a rich step summarizer (`master.py:606-631`), and a per-step checkpoint save (`master.py:633`). `project_context` is a single accumulated string threaded step→step (`master.py:487`, appended `master.py:631`).
- `--plan` is loaded by `load_plan_steps(path) -> List[str]` at **`dispatch.py:228-268`**, which accepts a bare list of strings or `{"steps":[...]}`, validates non-empty strings, and returns a flat list. It flows `cli.py:164 → dispatch(plan_steps=) → dispatch_async (dispatch.py:873) → _run_workflow (dispatch.py:540) → MasterWorkflow(plan_steps=) (dispatch.py:695, :783)`. The checkpoint-equality guard at **`master.py:459-460`** compares `list(resumed[0]) != list(self.plan_steps)` to detect an operator edit.
- `agy_orchestrator/execution/tdag.py` (58 lines) has a `TaskDAG` primitive: `add_node(name, instance, deps)` (`tdag.py:19`), `_run_node` concatenates dep outputs as `piped_input` text (`tdag.py:25-41`), `execute` does create-all-tasks-then-await (`tdag.py:43-58`). **It is orphaned** — imported nowhere, no tests, no mode. Its node is a *single* `AgentInstance.run_async()` with **no model for parallel nodes writing files to a shared tree**. We reuse its topology *pattern*, not its node body.
- `vote.py` already solves per-candidate **workspace isolation**: `_ManagedWorkspace` (`vote.py:74`) builds a git-worktree-or-copy candidate, `_apply_workspace` (`vote.py:595-652`) mirrors a winner's files back into the live tree (content-diffed, skips `.git`/`runs`/`__pycache__`, removes deleted files), with a dirty-tree copy-mode fallback (#36) and `candidate_setup` per-candidate venv bootstrap (#34). It caps concurrency with `verifier_concurrency` (`vote.py:139`, default 1) AND a `max_parallel` generation semaphore (`vote.py:170`).
- `dispatch.py` already plumbs `max_parallel_workers` (`dispatch.py:544`) into vote (`dispatch.py:746`) and exposes the #43 reconcile station `ReconciliationReview` (`dispatch.py:37, 834`).

**The gap.** There is no way to express deps, no scheduler that runs ready nodes concurrently, no per-node workspace isolation in the master path, and **no merge subsystem** for two parallel nodes that write overlapping files. That last item — the "net-new merge subsystem" prior work deferred — is the crux and the one decision needing operator sign-off (§4).

---

## 2. Target UX

### Round-trip commands

```bash
# 1. Generate a graph plan (planner emits a DAG; operator reviews/edits).
python -m harness do "Build a REST service with auth, storage, and an API layer" \
    --mode master --plan-only
#   -> writes runs/<id>/plan.json  (graph shape when the build is graph-shaped)

# 2. Edit runs/<id>/plan.json by hand: add deps, split a step into parallel nodes.

# 3. Feed the edited graph back in; master auto-detects the graph shape and runs the DAG.
python -m harness do "Build a REST service with auth, storage, and an API layer" \
    --mode master --plan runs/<id>/plan.json

#   Optional knobs (all additive, all default to safe values):
#     --max-parallel-nodes 3            # cap concurrent DAG nodes (env AGY_MAX_PARALLEL_NODES)
#     --merge-policy reconcile          # disjoint | reconcile | fail  (default: reconcile)
#     --plan-graph runs/<id>/plan.json  # strict alias: errors if the file is NOT a graph
```

`--plan` accepts **either** shape (auto-detected), so the existing round-trip flag "just works" on a graph file. `--plan-graph` is strict sugar that errors if the file lacks `"nodes"`. A flat `plan.json` is unchanged in every respect.

### Sample graph.json (v2 shape)

```json
{
  "version": 2,
  "instruction": "Build a REST service with auth, storage, and an API layer",
  "nodes": [
    {"id": "schema",  "task": "Define the DB schema + migrations module.",        "deps": []},
    {"id": "auth",    "task": "Implement the auth module (login, tokens).",        "deps": ["schema"], "group": "services"},
    {"id": "storage", "task": "Implement the storage/persistence layer.",          "deps": ["schema"], "group": "services"},
    {"id": "api",     "task": "Wire the HTTP API over auth + storage.",            "deps": ["auth", "storage"]},
    {"id": "tests",   "task": "Add hermetic integration tests for the API.",        "deps": ["api"]}
  ]
}
```

Topology: `schema → {auth, storage}` run **concurrently** (the parallel group), both join into `api`, then `tests`. The `group` field is an advisory scheduler hint only — **deps are authoritative**.

A flat plan stays exactly:
```json
{"instruction": "...", "n_steps": 3, "steps": ["Step 1 ...", "Step 2 ...", "Step 3 ..."]}
```

---

## 3. Architecture

### 3.1 Chosen approach: **extend `MasterWorkflow`, do NOT add `--mode graph`**

Sub-designs B and C independently reached the same conclusion; we adopt it. The per-node quality machinery the operator wants (ToT + Adversarial + verifier + summarizer + checkpoint) is ~130 lines of tightly-coupled logic *inside* the `master.py:499` loop body. A standalone `GraphWorkflow`/`--mode graph` would have to duplicate or fragilely import all of it, and would fork the entire effort-override / checkpoint / reconcile / notify / protect-paths surface (`dispatch.py:677-704`). Instead:

1. **Refactor the loop body into a coroutine.** Extract `master.py:500-633` (everything that executes one task → produces a `step_summary` + verified/approved signals) into:
   ```python
   async def _execute_step(self, *, step_index, step_total, task, project_context,
                            workflow_session_id, working_directory) -> StepResult
   ```
   returning `StepResult(final_output, step_summary, verified, approved, stalled, iterations_used, session_id)`. The existing `for i in range(...)` loop becomes a thin caller — **byte-for-byte identical** linear behavior (a golden-trace test guards this; see §7). The `working_directory` param lets a DAG node point `_execute_step` at an isolated workspace without `_execute_step` knowing anything about isolation.

2. **Generalize the plan to nodes internally.** Parse `plan_steps` / checkpoint tasks into `List[PlanNode]{id, task, deps}`. A bare string list (or `{"steps":[...]}`) lifts to N nodes with **implicit linear deps** (`deps=[prev_id]`), giving identical execution order to today. A graph plan (`{"nodes":[...]}`) builds the real DAG.

3. **Auto-route in `execute()`.** If any node has a non-linear dep set → call `_execute_graph`; otherwise keep the existing linear loop (the cheapest, proven, default path stays default and untouched). `--mode graph` is *not* added; the plan payload determines the shape. (An optional `--mode graph` alias could be added later as pure cosmetic sugar that just asserts a graph plan was supplied — out of scope for v1.)

### 3.2 In-memory plan model — `agy_orchestrator/execution/graph_plan.py` (NEW, pure, no I/O, no agents)

```python
class PlanCycleError(ValueError): ...

@dataclass
class PlanNode:    id: str; task: str; deps: List[str]; group: Optional[str] = None

@dataclass
class GraphPlan:
    nodes: List[PlanNode]
    def node_ids(self) -> List[str]: ...
    def topo_order(self) -> List[str]:          # stable Kahn, ties broken by input order; raises PlanCycleError
    def as_steps(self) -> List[str]:            # [node.task for id in topo_order] — the backward-compat linearization
    def parallel_groups(self) -> List[List[str]]:  # Kahn levels — the concurrency unit
    def ancestors(self, node_id) -> List[str]:  # transitive dep closure (for context composition)

@dataclass
class ChainPlan:
    steps: List[str]
    def as_steps(self) -> List[str]: return self.steps
    def as_graph(self) -> GraphPlan:            # lift flat -> implicit linear deps (ids s1..sN)

Plan = Union[ChainPlan, GraphPlan]
def to_json(plan: Plan) -> dict                 # GraphPlan -> v2 object; ChainPlan -> {"steps":[...]}
def validate_graph(raw: list) -> List[PlanNode] # dup-id / dangling-dep / cycle / empty / non-string-task
def topological_layers(nodes) -> List[List[str]]
```

**Determinism is load-bearing:** `topo_order` must be a *stable* Kahn sort (ties broken by input order) so a graph linearizes identically across runs — otherwise the checkpoint-equality guard at `master.py:459` flaps and resume breaks.

### 3.3 Loader — `load_plan(path) -> Plan` (`dispatch.py`, supersedes `load_plan_steps`)

Detection rule, in order: (a) bare JSON list → `ChainPlan`; (b) object with `"nodes"` → `GraphPlan` (validated immediately); (c) object with `"steps"` and **no** `"nodes"` → `ChainPlan`; (d) neither, or **both** → `ValueError` (hard error — never silently pick a shape). Validation (dup ids, dangling deps, cycles, empty, non-string task) runs at **load time** (CLI path) so the operator sees errors before any worker writes — matching today's fail-fast. `load_plan_steps` stays as a thin shim `return load_plan(path).as_steps()` so `cli.py:164` and `test_plan_injection.py` are untouched.

### 3.4 Scheduler — `_execute_graph` (in `master.py`)

Frontier-based (not barrier-per-level), so a fast branch's successors start while a slow sibling is still running:

```
done: Dict[id, StepResult] = {seed from start_completed_ids on resume}
running: Dict[id, asyncio.Task] = {}
while not all nodes in done:
    ready = {n : n not started, all n.deps in done}
    for n in ready:
        async with self._node_sem:                       # Semaphore(max_parallel_nodes)
            running[n.id] = create_task(self._run_dag_node(n, done, base_work_dir))
    completed, _ = await asyncio.wait(running.values(), return_when=FIRST_COMPLETED)
    move completed -> done; checkpoint(done); re-loop to release newly-unblocked nodes
```

`_run_dag_node(node, done, base_work_dir)` is the **isolation + merge wrapper** (§3.5):
1. Compose the node's context: `_compose_context(node, done) = header(goal) + "".join(done[a].step_summary for a in GraphPlan.ancestors(node))`. A join node sees the **union** of both branches; two parallel branches see **only** their own ancestors (the key correctness property). For a linear plan this reproduces today's cumulative string exactly (each node's ancestor set is all prior steps).
2. Open an isolated workspace (`_ManagedWorkspace`), call `_execute_step(... working_directory=ws.path)`.
3. On success, hand the workspace to the merge subsystem (§3.5) to fold writes into the live tree under the active `--merge-policy`.
4. Return `StepResult`.

**Two concurrency caps** (the exact layering vote uses): `self._node_sem = asyncio.Semaphore(max_parallel_nodes)` bounds whole nodes (gen+ToT+verify), AND the `verifier_concurrency` semaphore (default 1) threaded into each node's `AdversarialReview.verifier` serializes the local `make check` spike so K parallel nodes never run K full verifiers at once and OOM the box.

**Sessions:** DAG nodes run **sessionless** (`workflow_session_id=None` into `_execute_step`) — concurrent `--fork-session` on one parent races (already documented at `master.py:527-528`, and is why ToT branches are already sessionless). Cross-step session reuse stays a linear-mode-only fast path. Accepted token cost; surfaced in `meta.json`.

### 3.5 Isolation + Merge subsystem — `agy_orchestrator/workflows/graph_merge.py` (NEW)

**Lives** in its own module so it is unit-testable without agents. **Isolation reuses `vote.py` verbatim:** `_ManagedWorkspace` (worktree-or-copy + dirty-tree copy fallback from #36) builds each node's sandbox; `_apply_workspace` (content-diffed mirror, `vote.py:595`) is the disjoint-write fast path. We lift the relevant helpers into a shared spot (or import them from `vote.py`) so both modes share one isolation implementation.

**Interface:**
```python
@dataclass
class MergeOutcome:
    node_id: str; layer: int; overlapping_paths: List[str]
    policy: str; conflict: bool; resolution: str

async def merge_node(*, node_id: str, layer: int, src_ws: Path, live_tree: Path,
                     base_snapshot: dict[str, bytes],  # path -> bytes at node start
                     policy: str, reconcile_station, sem: asyncio.Semaphore) -> MergeOutcome
```

Merges happen **serially per completed node** (guarded by a merge semaphore so two nodes never write the live tree at once). For each file the node changed, compare against `base_snapshot` (the live tree's bytes when the node started) to detect whether another *already-merged sibling* touched the same path → that is an **overlap**. Disjoint changes are applied via the `_apply_workspace` content-diff. Overlaps are resolved per `--merge-policy` (§4). All outcomes are recorded in `merge_outcomes` (surfaced in `meta.json`), even when warn-only.

---

## 4. THE MERGE-POLICY DECISION  ⚠️ NEEDS OPERATOR SIGN-OFF BEFORE CODING

When two parallel nodes write the **same file**, how do we reconcile? This is the one genuinely net-new subsystem and the one product fork.

| Policy | Behavior on overlap | Pros | Cons | Risk |
|---|---|---|---|---|
| **`disjoint`** | Auto-apply only when every node's writes are disjoint; **any** overlap aborts the run | Simplest, zero ambiguity, fast, no extra worker call | Brittle — many real DAGs touch a shared `__init__.py` / registry / config; aborts surprise the operator | Low data-loss; high abort rate |
| **`reconcile`** *(recommended v1)* | Disjoint writes auto-apply; overlaps invoke the #43 reconcile/critic station (`ReconciliationReview`, `dispatch.py:834`) to produce a merged file, recorded as a `MergeOutcome` | Safe superset: degrades to a plain apply when disjoint, only spends a worker call on true conflicts; reuses an existing cross-provider reviewer | Costs a critic call per conflicting file; merge quality bounded by the reviewer; non-deterministic | Medium (mitigated: conflict recorded; verifier re-runs on the joined tree) |
| **`fail`** | Any overlap aborts and records the conflicting paths in `meta.json` | Strictest; never silently merges; good for mission-critical | Same brittleness as `disjoint` but with a richer report | Low data-loss; high abort rate |

**RECOMMENDED v1: `reconcile`** — the safe superset. It behaves like `disjoint` when writes don't overlap (the common case) and only escalates to the reconcile station on a true collision, recording the resolution in `meta.json`. After any overlap merge, the **join node's verifier re-runs on the merged tree**, so a bad merge is caught by the existing quality gate rather than shipped blind. `--merge-policy fail` is available for operators who want a hard stop; `disjoint` for provably-independent subtrees that want to skip the reconcile cost.

> **Operator: please confirm `reconcile` as the v1 default before M4 is built.** This is the only decision that changes the externally-visible safety contract. Everything else in this plan is mechanical.

---

## 5. Phased milestones

Each phase is independently testable and ships value. **Real parallelism is introduced only in M3** — M1/M2 validate topology and node-reuse *serially* first, so a regression in the default path is caught before any concurrency exists.

### M1 — Schema + backward-compatible loader (NO executor change)
- **Add** `agy_orchestrator/execution/graph_plan.py`: `PlanNode`, `GraphPlan`, `ChainPlan`, `Plan`, `PlanCycleError`, `to_json`, `validate_graph`, `topological_layers`, stable `topo_order`, `as_steps`, `parallel_groups`, `ancestors`, `ChainPlan.as_graph`.
- **Add** `load_plan(path) -> Plan` in `dispatch.py` (~line 228) with flat-vs-graph auto-detect + all validation; keep `load_plan_steps` as a `load_plan(p).as_steps()` shim.
- **Change** the checkpoint-equality guard at `master.py:459-460` to compare `plan.as_steps()` (normalized linearization) instead of raw lists, so an injected graph whose topo-order matches the checkpoint resumes.
- **Ships:** a graph plan can be authored, loaded, validated, and round-tripped; if executed it linearizes via `as_steps()` and runs on the **existing** master loop — zero behavior change for flat plans, and a graph "works" (serially) before any concurrency exists.
- **Unblocks:** everything downstream consumes the `Plan` object.

### M2 — Sequential DAG executor reusing master step machinery (still NO concurrency)
- **Refactor** `master.py:500-633` into `async def _execute_step(...) -> StepResult` (golden-trace test guards byte-identical linear behavior).
- **Add** the internal `PlanNode` lift + `_execute_graph` that walks nodes in **topological order one at a time** (a degenerate scheduler: `max_parallel_nodes` effectively 1), calling `_execute_step` per node with `_compose_context(node, done)` (ancestor-closure summaries).
- **Add** `execute()` auto-route: non-linear deps → `_execute_graph`, else existing loop.
- **Ships:** a real DAG runs correctly *serially* — proves context composition (ancestor-only / join-union), topological ordering, and machinery reuse, with no concurrency or isolation complexity yet. This is the safest place to validate correctness.
- **Unblocks:** M3 swaps the serial walk for a concurrent frontier scheduler.

### M3 — Isolation + concurrent scheduling (**introduces real parallelism**)
- **Add** `self._node_sem = asyncio.Semaphore(max_parallel_nodes)` and the frontier scheduler (`create_task` + `asyncio.wait(FIRST_COMPLETED)`) into `_execute_graph`.
- **Add** `_run_dag_node`: open a `_ManagedWorkspace` (reused from `vote.py`) per node, run `_execute_step(working_directory=ws.path)`, thread `verifier_concurrency` (default 1) into each node's verifier.
- **Add** constructor params to `MasterWorkflow.__init__` (~`master.py:38-56`): `max_parallel_workers: Optional[int]`, `verifier_concurrency: int = 1`; forward them from `dispatch.py:685/773` (the `max_parallel_workers` param already exists at `dispatch.py:544`).
- **Add** CLI flags in `cli.py` `do` subparser: `--max-parallel-nodes N` (env `AGY_MAX_PARALLEL_NODES`), thread through `cli.py:235 → dispatch → dispatch_async → _run_workflow`.
- **Merge in this phase is still trivial**: nodes run in isolated workspaces but each node's writes are applied via `_apply_workspace` **without** overlap detection — diamond DAGs with disjoint writes work; overlapping writes are last-writer-wins (a documented temporary limitation until M4).
- **Ships:** true wall-clock parallelism for disjoint-subtree DAGs (the common, safe case).
- **Unblocks:** M4 makes overlaps safe.

### M4 — Merge subsystem (the net-new piece; gated on §4 sign-off)
- **Add** `agy_orchestrator/workflows/graph_merge.py`: `MergeOutcome`, `merge_node(...)`, overlap detection via per-node `base_snapshot`, a merge semaphore (serial application), and the three policies (`disjoint` / `reconcile` / `fail`). `reconcile` invokes `ReconciliationReview` (`dispatch.py:834` / `workflows/reconcile.py`); the join node's verifier re-runs on the merged tree.
- **Add** CLI flag `--merge-policy {disjoint,reconcile,fail}` (default `reconcile`), threaded through the dispatch layers.
- **Wire** `merge_outcomes` collection in `_run_dag_node` / `_execute_graph`.
- **Ships:** overlapping parallel writes are reconciled (or fail-fast) instead of silently clobbering.
- **Unblocks:** M5 surfaces the results.

### M5 — Round-trip emit + meta + checkpoint/resume + docs
- **Extend** `_save_checkpoint`/`_load_checkpoint` (`master.py:355-384`, `:283`+) with optional `graph: [{id,task,deps}]` + `done: {id: {step_summary, verified, ...}}`; keep `completed:int` for linear back-compat. On resume, seed `start_completed_ids = set(done)` into `_execute_graph` (per-node frontier resume). The #37 `base_fingerprint` gate is unchanged and still guards the whole out-dir.
- **Extend** `_emit_plan` + the plan.json emit block (`dispatch.py:1448-1456`) to write the v2 graph shape when the workflow produced a `GraphPlan` (guarded by `getattr` so flat emit is byte-identical).
- **Add** a `graph` block to `DispatchResult` and the meta writer (`dispatch.py:1448-1521`): `{"nodes":[...statuses...], "layers":[[...]], "merges":[...], "merge_policy":"..."}`, populated by `getattr` on the workflow (`node_status`, `node_results`, `merge_outcomes`). Bound the per-node detail to a summary (CLAUDE.md discourages sidecar files; keep it in `meta.json`).
- **Add** `--plan-graph FILE` strict alias + the `--plan-only` graph round-trip hint in `cli.py`.
- **Docs:** one focused `docs/` doc (this spec, promoted) describing the graph shape, flags, and merge policy.
- **Ships:** full operator round-trip, observability, and crash-resume for partial DAGs.

---

## 6. Backward-compatibility guarantees, risks, mitigations

**Guarantees (the hard requirement):**
- A flat plan (bare list / `{"steps":[...]}`) → `ChainPlan.as_steps()` returns strings verbatim → master runs the **existing** `for i in range(len(tasks))` loop (`master.py:499`) unchanged.
- `load_plan_steps` keeps its public signature/return (`List[str]`) via the shim — `cli.py:164` and `test_plan_injection.py` untouched.
- Auto-route gates strictly on "any non-linear dep present"; a flat plan never enters `_execute_graph`.
- Plan.json emit for a flat plan is byte-identical (graph emit is `getattr`-guarded).

| Risk | Mitigation |
|---|---|
| Refactoring the 130-line loop body (`master.py:500-633`) into `_execute_step` drifts the default linear path | Golden-trace test asserts the refactored linear loop is byte-identical **before** any DAG logic lands (M2 gate) |
| Non-deterministic `topo_order` flaps the checkpoint guard (`master.py:459`) | Stable Kahn sort, ties broken by input order; unit-tested for determinism |
| Object with both `"steps"` and `"nodes"` silently mis-routes | `load_plan` raises `ValueError` (hard error), never picks a precedence |
| Parallel nodes contaminate each other's context | `_compose_context` uses ancestor-closure only; join node gets the union; unit-tested (B sees neither C, D sees both) |
| Parallel writes clobber the shared tree | Nodes **always** run in isolated `_ManagedWorkspace`; merge is serial per node; M4 adds overlap detection. Forbid live-tree node writes (the tdag failure mode) |
| Checkpoint schema change breaks old unpack | `graph`/`done` keys optional; old linear checkpoint (no `graph`) loads via the `completed:int` path; forward graph checkpoint read by old code is ignored (safe key mismatch) |
| Verifier OOM under K parallel nodes | `verifier_concurrency` semaphore (default 1) threaded into each node's verifier (exact vote layering) |
| Node failure orphans siblings | Frontier scheduler propagates exceptions; **fail-fast** for mission-critical (cancel in-flight, checkpoint the done frontier, surface the failing node); continue-independent-branches otherwise (operator flag `--graph-on-node-fail abort|continue`, default abort) |
| Deep join balloons composed context | Per-node compaction (`_compact_context`, `master.py:638`) on the composed ancestor string; cap at `max_context_chars` |
| Token cost rises (sessionless parallel nodes) | Accepted price of parallelism; surfaced in `meta.json` |

---

## 7. Hermetic test plan

All tests use the `AGY_BENCH_MOCK` MockAgent / `_StubAgent` + monkeypatch patterns already in `tests/test_plan_injection.py` and `tests/test_master_verified_propagation.py` — **no live workers**.

**`tests/test_graph_plan.py`** (pure, no agents):
- bare list → `ChainPlan`; `{"steps":[...]}` → `ChainPlan` and round-trips via `to_json`.
- `{"nodes":[...]}` → `GraphPlan`; `topo_order`/`as_steps` deterministic and stable.
- `parallel_groups` / `topological_layers`: diamond `A→{B,C}→D` gives `[[A],[B,C],[D]]`; linear chain gives single-node layers.
- cycle / dangling-dep / dup-id / empty-nodes / non-string-task / self-dep / both-`steps`-and-`nodes` each raise `ValueError` with a specific message.
- `load_plan_steps` shim returns the legacy `List[str]` for both legacy shapes.
- emit→load round-trip identity for `ChainPlan` and `GraphPlan`.
- `ChainPlan.as_graph()` lifts to implicit linear deps (`s1..sN`).

**`tests/test_graph_dispatch.py`** (stub agents):
- **Backward-compat (M1/M2 gate):** a flat plan runs the linear `MasterWorkflow` loop unchanged — planner-boom patch never fires, steps execute in order, step context matches the legacy cumulative string.
- `load_plan` auto-detects graph vs flat.
- **Concurrency (M3):** a diamond `A→{B,C}→D` runs B and C concurrently — stub agent records start/stop (or an `asyncio.Event`) to prove overlap; topological order respected (a dep node never starts before its parent completes).
- **Context:** D's composed context contains BOTH B and C summaries; B's context contains NEITHER C's.
- **Parallel cap:** with `--max-parallel-nodes 1`, a 2-wide layer serializes (no overlap observed).
- **Merge (M4):** disjoint writes apply both nodes' files; overlapping writes under `--merge-policy fail` abort with the conflict recorded in `meta`; under `reconcile` invoke the (stubbed) reconcile station and re-verify.
- **Meta:** `meta.json` gets the `graph` block (`node_status` all `passed`, `layers`, `merges`, `merge_policy`).
- **Round-trip:** `--plan-only` emits a graph `plan.json` that re-loads via `load_plan`.
- **Checkpoint/resume (M5):** kill after B done; resume runs only C, D from the frontier; a base_fingerprint divergence discards the stale graph checkpoint.

**`tests/test_graph_merge.py`** (pure / filesystem, no agents):
- `_apply_workspace`-style disjoint merge copies both nodes' files.
- overlap detection via `base_snapshot` correctly flags a shared path.
- `disjoint` aborts on overlap; `fail` records conflicting paths; `reconcile` calls the (stubbed) reconciler and records a `MergeOutcome`.

**Golden-trace test (M2):** assert the refactored linear loop (`_execute_step` caller) produces an identical sequence of agent calls + context strings to the pre-refactor loop, for a 3-step flat plan.

---

## 8. Open questions for the operator

1. **Merge policy default (§4) — the one sign-off needed before M4.** Recommended `reconcile` (safe superset, reuses #43 reconcile station, re-verifies the merged tree). Confirm, or pick `fail` (stricter) / `disjoint` (fastest).
2. **Node-failure policy:** fail-fast (cancel siblings, checkpoint frontier, abort) vs. continue-independent-branches. Recommended `--graph-on-node-fail` flag defaulting to `abort` (mission-critical-safe). OK?
3. **Verifier scope:** verify *every* node, or only join/leaf nodes? Recommended every node (each output verified before successors build on it), bounded by `verifier_concurrency=1`. Acceptable cost?
4. **`pat` graph support:** restrict graphs to `--mode master` for v1 (pat's Stage-1 is a single direct attempt; a graph only makes sense in the escalation path)? Recommended yes — error for `pat` until a use case appears.
5. **`tdag.py` disposition:** delete it (superseded), or keep only its create-task/dep-wait topology helpers with a "superseded by master graph mode" comment? Recommended: leave the file, do not import, add a test asserting it stays orphaned to avoid dead-code ambiguity.
6. **`--plan-graph` flag:** real strict alias, or just document that `--plan` auto-accepts graphs? Recommended keep the strict alias (clear operator intent + a useful error when a flat file is passed where a graph was expected).

---

## 9. Operator guide (BUILT)

Graph execution is an **extension of `--mode master`** — there is no `--mode graph`.
The plan *payload* decides the shape: a flat plan runs the existing linear loop
(byte-identically); a `{"nodes":[...]}` DAG with a real branch/join runs the
concurrent frontier scheduler. Graphs are **master-only** in v1 (`pat` errors).

### The graph plan shape (v2)

```json
{
  "version": 2,
  "instruction": "Build a REST service with auth, storage, and an API layer",
  "nodes": [
    {"id": "schema",  "task": "Define the DB schema + migrations.", "deps": []},
    {"id": "auth",    "task": "Implement auth (login, tokens).",     "deps": ["schema"]},
    {"id": "storage", "task": "Implement the storage layer.",        "deps": ["schema"]},
    {"id": "api",     "task": "Wire the HTTP API over auth+storage.", "deps": ["auth", "storage"]},
    {"id": "tests",   "task": "Add integration tests for the API.",   "deps": ["api"]}
  ]
}
```

`schema → {auth, storage}` run **concurrently** (the parallel layer), both join
into `api`, then `tests`. **Deps are authoritative** (an optional `group` field is
an advisory hint only). Each node reuses the full master per-step machinery (ToT +
adversarial + verifier + summarizer) inside its OWN isolated workspace; a node's
context is its **ancestor closure only** — a join node sees the union of its
branches, two parallel branches never contaminate each other. A flat plan, or a
graph whose deps form a pure linear chain, runs the unchanged linear loop.

### Round-trip

```bash
# 1. Emit a plan (graph-shaped when you hand one back; the planner emits flat).
python -m harness do "Build a REST service ..." --mode master --plan-only
#    -> runs/<id>/plan.json   (graph emit when a --plan graph is echoed through)

# 2. Edit runs/<id>/plan.json — add deps, split a step into parallel nodes.

# 3. Feed it back; the DAG runs (auto-detected from the 'nodes' shape).
python -m harness do "Build a REST service ..." --mode master --plan runs/<id>/plan.json
```

### Flags

| Flag | Effect |
|---|---|
| `--plan FILE` | Execute the plan verbatim (skip the planner). Accepts **either** a flat `steps` plan or a graph `nodes` DAG (auto-detected). |
| `--plan-graph FILE` | **Strict** alias of `--plan`: errors if the file is *not* a graph DAG (use when you mean a graph and want a clear error on a flat file). Master-only. |
| `--max-parallel-nodes N` | Cap how many DAG nodes run at once (env `AGY_MAX_PARALLEL_NODES`; default unbounded). `1` serializes a wide layer. Only affects a non-linear graph. |
| `--merge-policy disjoint\|reconcile\|fail` | How two parallel nodes that write the **same** file are reconciled. Default **`reconcile`**. |

A shared verifier semaphore (`verifier_concurrency`, default 1) serializes the
local `make check` spike so K parallel nodes never run K full verifiers at once.

### Merge policy (when two parallel nodes write the same file)

* **`reconcile`** *(default)* — the safe superset. Disjoint writes auto-apply; an
  overlap is sent to the reconcile station (a critic-chain reviewer that produces
  a merged file), and the **join node's verifier re-runs on the merged tree** so a
  bad merge is caught by the existing gate.
* **`disjoint`** — apply only when every node's writes are disjoint; **any** overlap
  aborts the run (fast, for provably-independent subtrees).
* **`fail`** — any overlap aborts and records the conflicting paths in `meta.json`
  (strictest; good for mission-critical).

### Observability + crash-resume

Every graph run writes a `graph` block to `runs/<id>/meta.json`:

```json
"graph": {
  "nodes": {"schema": "passed", "auth": "passed", "...": "..."},
  "layers": [["schema"], ["auth", "storage"], ["api"], ["tests"]],
  "merges": [{"node_id": "...", "overlapping_paths": [], "resolution": "disjoint", "...": "..."}],
  "merge_policy": "reconcile"
}
```

The salvage checkpoint (issue #31) records the node DAG + the per-node **done
frontier** (id → summary + signals), so a crashed DAG resumes the **exact**
unfinished nodes — a diamond that died after `auth` finished re-runs only
`storage`, `api`, `tests`, not a "first-K-topo" guess. The #37 `base_fingerprint`
gate is unchanged: if the out-dir diverged from the tree the checkpoint was saved
against, the stale graph checkpoint is **discarded** and the run starts fresh.

### Deviations from the plan

* `tdag.py` is left **orphaned** (not deleted): no production module imports it;
  a test (`test_tdag_stays_orphaned`) guards against a stray import resurrecting
  the dead code (§8 Q5).
* The merge subsystem lives in `agy_orchestrator/workflows/graph_merge.py`;
  isolation reuses `_ManagedWorkspace` / `_snapshot_tree` from
  `agy_orchestrator/execution/workspace.py` (lifted out of `vote.py`).
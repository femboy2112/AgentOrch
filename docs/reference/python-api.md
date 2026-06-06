# Python API reference

This is the programmer's reference for importing **AgentOrch** as a library and
driving the orchestrator from Python. It documents the public classes, their
constructor signatures, the `async execute(...)` contracts, and the result
attributes each workflow sets.

Everything here lives under the `agy_orchestrator` package. The day-to-day
operator entrypoints (`python -m harness do`, `python -m agy_orchestrator`) wrap
these same classes; this document is for callers who want to compose them
directly.

Conventions:

- Every workflow and agent is **async**. Drive it from an event loop
  (`asyncio.run(...)`) or `await` it from an existing coroutine.
- Agents write artifacts **directly into the working directory** (`cwd` /
  `working_directory`). There is no return-value-only mode.
- Most tunables are also exposed as environment variables; the env var is named
  in the relevant table.

---

## Table of contents

- [core/agent.py — `AgentInstance` ABC and failure classifiers](#coreagentpy)
- [core/agents — the CLI adapters and `make_fallback_agent`](#coreagents)
- [core/profile.py — `UserProfile`](#coreprofilepy)
- [execution/verifier.py — `QualityVerifier` + `VerifierResult`](#executionverifierpy)
- [execution/pipeline.py — `LinearPipeline`, `ParallelSwarm`](#executionpipelinepy)
- [execution/ledger.py — `build_ledger`](#executionledgerpy)
- [execution/graph_plan.py — `PlanNode`, `GraphPlan`](#executiongraph_planpy)
- [workflows — `AdversarialReview`, `TreeOfThought`, `TestFeedbackWorkflow`, `CascadeWorkflow`, `GenerateAndRankWorkflow`, `AdaptiveDecomposer`, `MasterWorkflow`, `ReconciliationReview`](#workflows)
- [workflows/graph_merge.py — merge subsystem](#workflowsgraph_mergepy)
- [interaction/decision_engine.py — `DecisionEngine`](#interactiondecision_enginepy)

---

## core/agent.py

`from agy_orchestrator.core.agent import AgentInstance`

### `AgentInstance` (abstract base class)

A single execution of one AI-agent CLI. Subclasses (`CodexAgent`, `ClaudeAgent`,
`AgyAgent`, `GrokAgent`) implement only how the prompt is delivered and how raw
stdout is interpreted; the hardened run loop (retry / timeout / watchdog / kill /
cancel) lives here.

#### Constructor

```python
AgentInstance(
    prompt: str,
    model: Optional[str] = None,
    additional_flags: Optional[Dict[str, str]] = None,
    **kwargs,
)
```

`additional_flags` is rendered as `--<key> <value>` onto the CLI command line.
Any extra `**kwargs` are set as attributes on the instance (this is how
subclasses receive `effort`, `session_id`, etc. — see each adapter).

#### Run contract

```python
async def run_async(self, piped_input: Optional[str] = None) -> str
def run(self, piped_input: Optional[str] = None) -> str   # sync wrapper: asyncio.run(run_async(...))
```

- Builds the command via `build_command`, runs the CLI subprocess, and returns
  the **final post-processed stdout string** (the model's answer text).
- `piped_input`, if given, is appended to the prompt as a context block
  (`AgyAgent` labels it `[Piped Context from previous step]`; the codex/claude/grok
  adapters label it `[Context]`).
- Retries only on **transient** errors with jittered backoff
  (`min(8, 2**attempt) * (0.5 + random())`), up to `self.max_retries` (default
  3). **Fails fast — no retry —** on a timeout, a usage/quota wall, a context
  overflow, or a wedged session, raising `RuntimeError` so a fallback layer can
  roll over immediately.
- The child is always killed (its whole process group) on timeout, exception, or
  cancellation; no orphaned worker leaks.
- Streaming mode (line-by-line drain + watchdog) is engaged automatically when
  `AGY_STREAM` is set, any watchdog budget is armed, or `event_callback` is set.

#### Key public attributes

Populated across a call (read after `run_async`):

| Attribute | Type | Meaning |
|-----------|------|---------|
| `stdout` | `str` | Final post-processed result (same value `run_async` returns). |
| `stderr` | `str` | Filtered stderr; carries `[watchdog:<reason>]` / synthetic failure markers. |
| `returncode` | `Optional[int]` | Child exit code of the last attempt. |
| `last_wall_ms` | `Optional[float]` | Wall-clock of the last successful call, ms. |
| `last_out_bytes` | `Optional[int]` | Bytes emitted on stdout (streaming mode only). |
| `last_usage` | `Optional[Dict]` | Token usage row (`input_tokens`, `output_tokens`, `cache_read_tokens`, `total_tokens`, `token_source`, ...). |
| `event_callback` | `Optional[Callable[[dict], None]]` | Best-effort observability sink; failures never affect execution. |
| `cwd` | `Optional[str]` | Working directory for the child. `None` = inherit parent cwd. |
| `extra_env` | `Dict[str, str]` | Extra env vars injected into the child (always win over inherited/bounded vars). |

#### Timeout / watchdog budget attributes

All are read from env in `__init__` and may be overridden per-instance before
`run_async`. `0`/unset disables that signal.

| Attribute | Env var | Default | Meaning |
|-----------|---------|---------|---------|
| `timeout` | `AGY_TIMEOUT` | `2400` | Wall-clock ceiling per call; in streaming mode reinterpreted as the **idle** ceiling (extends while output flows). |
| `absolute_timeout` | `AGY_ABSOLUTE_TIMEOUT` | `0` → `4.0 × timeout` | Hard cap on liveness-based extension. |
| `worker_cmd_timeout` | `AGY_WORKER_CMD_TIMEOUT` | `0` | Hard per-call kill that fires even while output streams (tightest of it / `absolute_timeout` wins). |
| `max_output_bytes` | `AGY_MAX_OUTPUT_BYTES` | `0` | Kill on stdout byte budget (runaway-verbose). |
| `stall_seconds` | `AGY_STALL_SECONDS` | `0` | Kill if no real progress for this long. |
| `transport_max_errors` | `AGY_TRANSPORT_MAX_ERRORS` | `25` | Cumulative transport-error count tripping a degraded spell. |
| `transport_max_seconds` | `AGY_TRANSPORT_MAX_SECONDS` | `300` | Max wall-clock a degraded transport spell may persist. |
| `transport_recovery_window` | `AGY_TRANSPORT_RECOVERY_WINDOW` | `60` | Seconds of clean output that clears a spell. |
| `transport_decay_per_window` | `AGY_TRANSPORT_DECAY_PER_WINDOW` | `1.0` | Cumulative-weight decay rate during sustained clean output. |
| `silence_max_seconds` | `AGY_SILENCE_MAX_SECONDS` | `600` | Trip `stalled` when a call emits *nothing at all* for this long (independent of `stall_seconds`). |
| `max_retries` | — | `3` | Transient-error retry budget. |

After a watchdog trip, `self._watchdog_reason` is one of the module constants
`WATCHDOG_VERBOSE` (`"verbose"`), `WATCHDOG_STALLED` (`"stalled"`),
`WATCHDOG_TRANSPORT_STALL` (`"transport_stall"`), and the marker
`[watchdog:<reason>]` is prefixed onto `self.stderr`.

#### Abstract hooks subclasses must / may implement

```python
@classmethod
async def get_available_models(cls) -> List[str]        # abstract
@classmethod
async def get_model_usage(cls, model: str) -> float     # abstract; 0.0–100.0
def build_command(self, piped_input=None) -> List[str]  # abstract — the CLI argv
```

Optional template-method hooks (default no-op / identity):

- `filter_stderr(stderr) -> str` — drop benign network noise.
- `_stdin_bytes(piped_input) -> Optional[bytes]` — deliver the prompt via stdin
  (dodges `ARG_MAX`).
- `_postprocess(raw_stdout) -> str` — unwrap a JSON envelope / capture a session id.
- `_augment_failure_stderr(raw_stdout, stderr) -> str` — fold out-of-band failure
  telemetry into stderr before classification (codex overrides this).
- `_extract_usage(raw_stdout, raw_stderr) -> dict` — token counts.

### Module-level failure classifiers

All take an `stderr` string and return `bool`. The usage-wall /
context-overflow / model-unavailable / transport classifiers guard against each
other so they don't overlap; `is_wedged_session` is independent and may co-occur
(the fallback layer prioritizes it explicitly).

| Function | True when |
|----------|-----------|
| `is_usage_wall(stderr)` | A quota / rate-limit wall (`usage limit`, `rate limit`, `429`, `quota exceeded`, ...). |
| `is_context_overflow(stderr)` | The prompt overflowed the model context window (`context_length_exceeded`, `context window`, ...). |
| `is_model_unavailable(stderr)` | The requested model is inaccessible to this account (`model is not supported`, `model_not_found`, ...). |
| `is_transport_error(stderr)` | A transient transport/network blip (`connection reset`, `websocket`, `502/503`, ...). |
| `is_wedged_session(stderr)` | An unrecoverably wedged agent session (codex `stdin is closed for this session`). |

### `apply_worker_resource_bounds(env: Dict[str, str]) -> Dict[str, str]`

Injects conservative resource bounds into a worker child env: pins
`{OPENBLAS,OMP,MKL,NUMEXPR}_NUM_THREADS=1` (only when absent), and appends
`-n <K>` (`AGY_WORKER_PYTEST_XDIST`, default 2; `K<=0` → `-p no:xdist`) plus an
optional `-m <expr>` (`AGY_WORKER_PYTEST_MARKERS`) to `PYTEST_ADDOPTS`. Returns
`env` unchanged when `AGY_WORKER_RESOURCE_BOUND` is `0`/`no`/`false`/`off`.

---

## core/agents

`from agy_orchestrator.core.agents.codex_agent import CodexAgent`
(and `claude_agent.ClaudeAgent`, `grok_agent.GrokAgent`, `agy_agent.AgyAgent`)

All four are `AgentInstance` subclasses and share the run contract above. Their
distinguishing constructor parameters:

### `CodexAgent`

```python
CodexAgent(prompt, model=None, additional_flags=None, **kwargs)
```

- `model`: one of `CodexAgent.AVAILABLE_MODELS`
  (`gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex`, `gpt-5.3-codex-spark`,
  `gpt-5.2`). The sentinel `"standard"` maps to `gpt-5.3-codex-spark` (the only
  model a ChatGPT-account codex accepts).
- `effort` (kwarg): mapped to `model_reasoning_effort`; `"max"` → `xhigh`.
- `config_overrides` (kwarg): list of raw `-c key=value` strings (e.g.
  `["tools.web_search=true"]`).
- `usage_wall_resets_at` / `usage_wall_limit_name` (attrs): **set by the agent**
  when it detects an out-of-band quota wall in the rollout JSONL — the unix-epoch
  reset time and the limit window name, read by `make_fallback_agent`'s rate-limit
  cache. Detection threshold: `AGY_CODEX_USAGE_WALL_PERCENT` (default `99`).

### `ClaudeAgent`

```python
ClaudeAgent(*args, session_id: Optional[str] = None,
            fork_session: bool = False, **kwargs)
```

- `session_id=None` → fresh session; the established id is captured to
  `self.session_id` for warm-cache reuse on the next call.
- `session_id=<uuid>` → resumes that session (`--resume`).
- `fork_session=True` → resumes `session_id` but forks into a new session (used
  for ToT branches so they don't pollute the main thread).
- `effort` (kwarg) → `--effort`.
- `model`: `opus` / `sonnet` / `haiku` (bare names resolve to the CLI's current
  dated default), or a full `claude-opus-4-N` id. `"standard"` → `sonnet`.
- `dashboard_stream_json` (attr, default `False`) switches output to
  `stream-json` for the dashboard adapter.

### `GrokAgent`

```python
GrokAgent(*args, session_id: Optional[str] = None,
          web_search: bool = True, **kwargs)
```

- `session_id` → `--resume` (captured from the JSON `sessionId` on success).
- `web_search=False` → `--disable-web-search`.
- `model`: defaults to `grok-build` (the only model `grok models` surfaces today).
  Effort is **not** sent for `grok-build` (it 400s).

### `AgyAgent`

```python
AgyAgent(prompt, model=None, effort=None,
         input_files=None, output_files=None, additional_flags=None)
```

- `model` + `effort` are resolved to an exact picker display name (e.g.
  `"Gemini 3.1 Pro (High)"`) and written to agy's global `settings.json` before
  the run, then restored after. Model-pinned runs are serialized cross-process
  with a file lock (agy's model selection is global). An unrecognized model
  leaves agy on its current default with no settings touch.
- `input_files` / `output_files` are injected as read/create hints in the prompt.

### `make_fallback_agent(...)`

`from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent`

```python
make_fallback_agent(
    chain: List[Type[AgentInstance]],
    cycles: int = 3,
    configs: Optional[Dict[Type[AgentInstance], Dict[str, object]]] = None,
    watchdog_rules: Optional[Dict[str, List[Type[AgentInstance]]]] = None,
    post_construct_hook: Optional[Callable[[AgentInstance, Type[AgentInstance]], None]] = None,
    model_fallbacks: Optional[Dict[Type[AgentInstance], List[str]]] = None,
) -> Type[AgentInstance]
```

Returns a **new `AgentInstance` subclass** that, per call, tries `chain` in
order; on a produce-failure it advances to the next provider, repeating the whole
chain `cycles` times.

| Parameter | Meaning |
|-----------|---------|
| `chain` | Ordered provider classes, e.g. `[CodexAgent, AgyAgent, ClaudeAgent]`. Must be non-empty. |
| `cycles` | Times the whole chain repeats before giving up (`>= 1`). A walled provider's quota may recover before the cycle returns to it. |
| `configs` | Per-class overrides `{Cls: {"model": ..., "effort": ..., <extras>}}`. Essential when chaining providers with disjoint model namespaces. |
| `watchdog_rules` | `{"verbose"/"stalled": [Cls, ...]}` — re-route targets spliced to the front when a sub trips that watchdog reason (applies once per trip; bounded so it can't ping-pong forever). |
| `post_construct_hook` | `hook(sub, cls)` run after each sub is built (e.g. arm calibration watchdog budgets). |
| `model_fallbacks` | `{Cls: ["gpt-5.5", "gpt-5.4", ...]}` — alternate models to try on the **same** provider on a usage wall / inaccessible model, before advancing the chain. Each tried at most once per call. |

The instance exposes, after a call: `last_provider` (the class name that produced
the accepted output), and reuses warm `session_id` only for the class that
created it.

Construct an instance like any agent: `Agent = make_fallback_agent([...]);
agent = Agent(prompt="...")`.

#### Process-level caches (module functions)

These caches are shared across every fallback instance in the process:

- `record_inaccessible_model(provider: str, model)` /
  `is_model_known_inaccessible(provider, model) -> bool` /
  `reset_inaccessible_models()` — **permanent** per-process pruning of a model
  the account cannot use. `provider` is the class name (`"CodexAgent"`).
- `record_rate_limited(provider, model, resets_at)` /
  `is_rate_limited(provider, model, now=None) -> bool` /
  `clear_rate_limited()` — **time-bounded** cache (issue #82). A `(provider,
  model)` is skipped until `resets_at` (unix epoch) passes, then auto-expires on
  the next lookup. `record_rate_limited` ignores `None`/`NaN`/`inf`/`<=0`.

---

## core/profile.py

`from agy_orchestrator.core.profile import UserProfile`

```python
UserProfile(claude_plan: str = "free",
            codex_plan: str = "free",
            agy_plan: str = "free")

def get_baseline_effort(self, agent_name: str) -> str
```

`get_baseline_effort("claude" | "codex" | "agy")` reads the matching
`<name>_plan` string and maps it to a baseline effort:

| Plan string contains | Returned effort |
|----------------------|-----------------|
| `max` / `$100` / `20x` | `"max"` |
| `pro` / `$50` | `"high"` |
| `plus` / `$20` | `"medium"` |
| (anything else) | `"low"` |

---

## execution/verifier.py

`from agy_orchestrator.execution.verifier import QualityVerifier, VerifierResult`

### `QualityVerifier`

```python
QualityVerifier(test_commands: List[str],
                timeout: float = None,
                mem_max: Optional[str] = None)

async def verify(self, working_directory: str) -> VerifierResult
```

Runs each command in `test_commands` (shell) in `working_directory`, returning on
the first failure. Empty `test_commands` returns `ok=True`.

| Param | Env var | Default | Meaning |
|-------|---------|---------|---------|
| `test_commands` | — | — | Shell commands to run as the gate (e.g. `["pytest -q"]`). |
| `timeout` | `AGY_TEST_TIMEOUT` | `600` | Per-command wall-clock kill (whole process group). |
| `mem_max` | `AGY_VERIFIER_MEM_MAX` | `None` | Opt-in memory cap; each command runs in a transient `systemd --user --scope` with `MemoryMax`. Degrades to uncapped (with a warning) when `systemd-run` is unavailable. |

Other behaviour: pins BLAS thread pools and clamps any `-n K` xdist count to the
host core count (opt out via `AGY_WORKER_RESOURCE_BOUND=0`); persists per-iteration
logs and emits failing-test nodeids when `run_dir` / `event_callback` /
`step_index` are set on the instance. The most recent result is cached on
`self.last_result`.

### `VerifierResult` (dataclass)

```python
@dataclass
class VerifierResult:
    ok: bool
    message: str = ""
    returncode: int = 0
    stdout_tail: str = ""          # last ~2 KB
    stderr_tail: str = ""          # last ~2 KB
    duration_ms: int = 0
    timeout: bool = False          # killed by its own wall-clock timeout (infra)
    cmd: str = ""
    error_hash: Optional[str] = None        # sha256(stderr_tail)[:16]
    resource_exceeded: bool = False         # OOM-killed in its cgroup scope (infra)
    infra_suspected: bool = False           # heuristic: broken host env, not bad code
```

Back-compat protocols:

- `__bool__` → `self.ok`, so `if result:` works.
- `__iter__` yields `(ok, message)`, so `success, error = await verifier.verify(...)`
  unpacks.

`timeout` / `resource_exceeded` / `infra_suspected` let callers distinguish an
**infra failure** (don't regenerate) from a genuine code defect.

---

## execution/pipeline.py

`from agy_orchestrator.execution.pipeline import LinearPipeline, ParallelSwarm`

### `LinearPipeline`

```python
LinearPipeline(instances: List[AgentInstance])
async def execute(self, initial_input: Optional[str] = None) -> str
```

Runs `instances` in sequence, piping each stdout into the next as `piped_input`.
Returns the final stage's output.

### `ParallelSwarm`

```python
ParallelSwarm(instances: List[AgentInstance])
async def execute(self, common_input: Optional[str] = None) -> List[str]
```

Runs all `instances` concurrently with the same `common_input`. Crash-tolerant:
failed branches (usage wall, hang, OOM) are dropped and only successful outputs
are returned; if **all** fail, the first exception is re-raised. Each branch is
bounded by `AGY_SWARM_BRANCH_TIMEOUT` (seconds; `0`/unset = unbounded).

---

## execution/ledger.py

`from agy_orchestrator.execution.ledger import build_ledger`

```python
build_ledger(workflow: Any, *,
             mode: str,
             had_verifier: bool,
             produced_output: bool,
             telemetry: Optional[Dict[str, Any]] = None,
             run_aborted: bool = False) -> Dict[str, Optional[object]]
```

Reads whatever signals `workflow` exposes (`verified`, `approved`, `stalled`,
`iterations_used` via `getattr`, all defaulting gracefully) and derives a coarse
`confidence` label. Returns a dict with keys: `confidence`, `verified`,
`critic_approved`, `stalled`, `iterations_used`, `had_verifier`, `note`, plus any
non-`None` `telemetry` values (`wall_ms`, `out_bytes`, `watchdog_reason`,
`worker`, `model`, `effort`, `baseline_*`, `verifier_delta`).

Confidence ladder:

| `confidence` | When |
|--------------|------|
| `verified` | A programmatic verifier passed (ground truth). |
| `approved` | No verifier, but the LLM critic approved. |
| `unverified` | Output produced, nothing confirmed it. |
| `stalled` | `run_aborted=True` and output was produced (watchdog-aborted run — honest downgrade). |
| `failed` | No output produced. |

---

## execution/graph_plan.py

`from agy_orchestrator.execution.graph_plan import PlanNode, GraphPlan, ChainPlan`

Pure in-memory plan model (no I/O, no agents). Determinism is load-bearing: the
topo sort is a stable Kahn sort (ties broken by input order).

### `PlanNode` (dataclass)

```python
@dataclass
class PlanNode:
    id: str
    task: str
    deps: List[str] = []          # node ids this depends on
    group: Optional[str] = None   # advisory scheduler hint only; deps are authoritative
```

### `GraphPlan` (dataclass)

```python
@dataclass
class GraphPlan:
    nodes: List[PlanNode]
```

| Method | Returns |
|--------|---------|
| `node_ids()` | Ids in input order. |
| `topo_order()` | Stable topological order of ids. Raises `PlanCycleError` on a cycle. |
| `as_steps()` | Tasks (strings) in topological order — the linearization the master loop consumes. |
| `parallel_groups()` | Kahn levels: `List[List[str]]`; each level may run in parallel (a diamond `A→{B,C}→D` yields `[[A],[B,C],[D]]`). |
| `ancestors(node_id)` | Transitive dependency closure of `node_id`, in topological order. Raises `ValueError` on an unknown/dangling id. |

Helpers: `validate_graph(raw: list) -> List[PlanNode]` (fail-fast parse of a raw
`nodes` list), `topological_layers(nodes)`, `to_json(plan)`, and `ChainPlan`
(flat plan with `as_steps()` / `as_graph()`).

---

## workflows

All workflow classes live under `agy_orchestrator.workflows`. Each is constructed
with already-built agent instances and an `async execute(...)`.

### `AdversarialReview`

`from agy_orchestrator.workflows.adversarial import AdversarialReview`

```python
AdversarialReview(
    generator_instance: AgentInstance,
    critic_instance: AgentInstance,
    verifier: Optional[QualityVerifier] = None,
    max_iterations: int = 5,
    diff_only: bool = False,
    working_directory: str = ".",
    critic_preamble: str = "",
    critic_requirement: Optional[str] = None,
    event_callback: Optional[Callable[[dict], None]] = None,
)

async def execute(self, initial_prompt: str) -> str
```

Loop: generator produces → if a `verifier` is set, a **passing** verifier is
ground truth and returns immediately (no critic pass); otherwise the LLM critic
must reply exactly `APPROVED`. Bails early on a repeated-identical critique
(`stalled`) or an infra-class verifier failure.

Result attributes (read after `execute`):

| Attribute | Meaning |
|-----------|---------|
| `verified` | Programmatic verifier passed. |
| `approved` | Critic approved (or verifier passed → also sets this). |
| `stalled` | Bailed on a repeated critique. |
| `iterations_used` | Iterations actually run. |
| `verifier_infra_failed` / `infra_reason` | Verifier hit a timeout / OOM / infra-class error; `infra_reason` is `verifier_timeout` / `verifier_resource_exceeded` / `verifier_infra_suspected`. |
| `generator_rotations` | Times the generator fallback chain was rotated after `AGY_GENERATOR_ROTATE_AFTER` (default 2) consecutive verify-fails. |

### `TreeOfThought`

`from agy_orchestrator.workflows.tree_of_thought import TreeOfThought`

```python
TreeOfThought(
    branch_instances: List[AgentInstance],
    evaluator_instance: AgentInstance,
    selector: str = "judge",
    event_callback: Optional[Callable[[dict], None]] = None,
    requirement: Optional[str] = None,
)

async def execute(self) -> str
```

Generates all branches concurrently (`ParallelSwarm`), then selects:

- `selector="judge"` (default) — clones the evaluator per branch, scores each
  1–10, returns the argmax (stable on ties). Pass `requirement` so the judge
  scores against the goal rather than blind internal consistency.
- `selector="vote"` — majority vote over normalized branch outputs; **no**
  evaluator pass (best-of-N at zero extra model cost).

A single surviving branch is returned without an evaluator pass. Constructor
raises `ValueError` if `branch_instances` is empty.

### `TestFeedbackWorkflow`

`from agy_orchestrator.workflows.test_feedback import TestFeedbackWorkflow`

```python
TestFeedbackWorkflow(
    generator_instance: AgentInstance,
    verifier: QualityVerifier,
    max_iterations: int = 4,
    working_directory: str = ".",
    preserve: bool = True,
)

async def execute(self, initial_prompt: str) -> str
```

Generate → run the real tests → feed the verbatim error + last output back →
repair. No LLM critic. `preserve=True` instructs the model to fix only what
failed. Returns as soon as the verifier passes. `verifier` is required
(`ValueError` otherwise).

Result attributes: `verified` (bool), `iterations_used` (int).

### `CascadeWorkflow`

`from agy_orchestrator.workflows.cascade import CascadeWorkflow`

```python
CascadeWorkflow(
    stages: List[AgentInstance],
    verifier: QualityVerifier,
    max_iterations_per_stage: int = 2,
    working_directory: str = ".",
)

async def execute(self, initial_prompt: str) -> str
```

Cheap-first escalation: each stage runs a `TestFeedbackWorkflow`; the verifier is
the gate between stages. Returns as soon as a stage passes. An infra-class
verdict (OOM / timeout / command-not-found) bails without escalating; an
identical failure across tiers short-circuits. Between stages the shared tree is
reset to a clean baseline (opt out with `AGY_CASCADE_ISOLATE=0`). Requires
non-empty `stages` and a non-`None` `verifier`.

Result attributes:

| Attribute | Meaning |
|-----------|---------|
| `verified` | A stage passed the verifier. |
| `stage_used` | Index of the passing stage (`-1` = none). |
| `iterations_used` | Total repair rounds across stages. |
| `stalled` | Exhausted all stages / short-circuited / infra-bailed without passing. |
| `verifier_infra_failed` / `infra_reason` | An infra-class verdict aborted the cascade. |
| `short_circuited` | Bailed on a repeated-identical failure. |
| `no_work` | `max_iterations_per_stage <= 0` (misconfig — zero generators ran). |

### `GenerateAndRankWorkflow`

`from agy_orchestrator.workflows.generate_and_rank import GenerateAndRankWorkflow`

```python
GenerateAndRankWorkflow(
    generator_instances: List[AgentInstance],
    verifier: Optional[QualityVerifier] = None,
    ranker: Optional[AgentInstance] = None,
    execution_selector: Optional[ExecutionSelector] = None,
    working_directory: str = ".",
)

async def execute(self, prompt: str) -> str
```

Generates K candidates concurrently, then ranks. With K == 1 the verifier is a
true pass/fail gate on the resident artifact; with K > 1 ranking is by
`execution_selector` (per-candidate sandboxes), else the LLM `ranker`, else the
first candidate (the verifier then only sets the `verified` signal — disk state
can't reorder un-isolated candidates).

Result attributes: `verified` (bool), `n_candidates` (int), `n_passed` (int).

### `AdaptiveDecomposer`

`from agy_orchestrator.workflows.decompose import AdaptiveDecomposer`

Model-agnostic ADaPT-style recursion. **Synchronous** and agent-free — you pass
two callables.

```python
AdaptiveDecomposer(max_depth: int = 2, max_subtasks: int = 6)

def run(self, task,
        solve_one: Callable[[Any], Tuple[str, bool]],
        decompose: Callable[[Any], List[Any]]) -> Tuple[str, bool]
```

Try the task whole; on failure decompose (up to `max_subtasks`) and recurse (down
to `max_depth`, clamped below the interpreter recursion limit). Returns
`(output, ok)` where `ok` is the AND of subtask results.

Result attributes: `root` (a `DecompNode`, `.as_dict()` gives the full tree),
`depth_reached` (int), `n_decompositions` (int).

### `MasterWorkflow`

`from agy_orchestrator.workflows.master import MasterWorkflow`

The whole-feature pipeline: plan → ToT → adversarial, with checkpointing,
context compaction, and an optional dependency-DAG walker.

```python
MasterWorkflow(
    model: str,
    effort: str,
    branches: int = 3,
    max_iterations: int = 5,
    verifier: Optional[QualityVerifier] = None,
    agent_class=AgyAgent,
    critic_agent_class=None,
    critic_model: Optional[str] = None,
    critic_effort: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    compaction_interval: int = 6,
    max_context_chars: int = 12000,
    selector: str = "judge",
    working_directory: str = ".",
    recent_steps_verbatim: int = 2,
    event_callback: Optional[Callable[[dict], None]] = None,
    resume_policy: str = "auto",
    plan_only: bool = False,
    plan_steps: Optional[List[str]] = None,
    plan_graph: Optional[GraphPlan] = None,
    max_parallel_workers: Optional[int] = None,
    verifier_concurrency: int = 1,
    merge_policy: str = "reconcile",
    reconcile_station_factory: Optional[Callable[..., "Reconciler"]] = None,
)

async def execute(self, initial_prompt: str) -> str
```

Notable parameters:

| Param | Meaning |
|-------|---------|
| `agent_class` | Generator agent class (default `AgyAgent`). |
| `critic_agent_class` / `critic_model` / `critic_effort` | The **distinct** in-loop adversarial critic (built from the critic chain). When `critic_agent_class` is `None` it falls back to `agent_class` + generator model (back-compat). `critic_effort` defaults to `"high"`. |
| `plan_steps` | Execute these step prompts **verbatim**, skipping the planner (the `--plan` round-trip). |
| `plan_graph` | A `GraphPlan`; a non-linear graph routes to the concurrent DAG walker. |
| `merge_policy` | `"disjoint"` / `"reconcile"` (default) / `"fail"` — how overlapping writes from parallel DAG nodes are resolved. |
| `reconcile_station_factory` | Builds the per-overlap async reconciler (wired from the critic chain) for `reconcile` policy. |
| `checkpoint_path` | Where to persist/resume the run; `resume_policy` is `"auto"` (default) / `"force"` / `"never"` (the harness `--resume` flag passes `"force"` and `--fresh` passes `"never"`). |
| `plan_only` | Emit the plan and write nothing. |
| `compaction_interval` / `max_context_chars` / `recent_steps_verbatim` | Two-tier context compaction tuning. |
| `verifier_concurrency` | Cap on simultaneous verifier runs across concurrent DAG nodes (default 1). |

Result attributes (read after `execute`, consumed by `build_ledger`):
`verified`, `approved`, `stalled`, `iterations_used` (mirrored from the final
accepted step's `AdversarialReview`); `plan_degraded` / `plan_parse_error`
(planner decomposition failed → ran as one mega-step); `checkpoint_write_failed`;
`merge_outcomes` (`List[MergeOutcome]`); `node_status` / `node_results` (graph
observability, empty on the linear path).

### `ReconciliationReview`

`from agy_orchestrator.workflows.reconcile import ReconciliationReview, ReconciliationResult`

Post-verifier "Integration-Skeptic" station: after a build converges and the
verifier is green, it traces each spec-named mechanism to the **live execution
path** and flags `exists-but-not-load-bearing` (dead/stubbed/bypassed) code. Its
verdict is a distinct status, **never** folded into `VerifierResult.ok`.

```python
ReconciliationReview(
    agent: AgentInstance,
    goal: str,
    verifier: Optional[object] = None,
    working_directory: str = ".",
    disposition: str = "warn",          # "warn" (default) | "fail" | "open-task"
    diff: Optional[str] = None,
    event_callback: Optional[Callable[[dict], None]] = None,
    max_iterations: int = 1,
    ablation_cmd: Optional[str] = None,
    ablation_cmd_map: Optional[Dict[str, str]] = None,
    ablation_timeout: Optional[float] = None,   # else AGY_ABLATION_TIMEOUT or 600
    prefer_worktree: bool = True,
)

async def execute(self) -> ReconciliationResult
```

`disposition="warn"` (default) reports loudly but never fails the run; `"fail"`
hard-fails when not reconciled. With `ablation_cmd` set, the station **measures**
each mechanism's load-bearing witness (runs the command clean vs with
`AGY_ABLATE=<mech>` in a throwaway worktree) instead of trusting the model's
self-report. After `execute`, `self.result` and `self.reconciled` are populated.

`ReconciliationResult` (dataclass) fields: `reconciled: bool`, `findings_list:
List[MechanismFinding]`, `disposition: str`, `raw: str`, `parse_error:
Optional[str]`, `starved_reason: Optional[str]`. Helper accessors include
`findings()` (the actionable dead-wired `exists_not_load_bearing` set),
`absent_findings()`, `load_bearing_findings()`, `excused_findings()`, and
`should_fail_run` (True only when `disposition == "fail"` and not reconciled). A starved (killed/exhausted) trace
is forced `reconciled=False` so it can never bless dead wiring.

---

## workflows/graph_merge.py

`from agy_orchestrator.workflows.graph_merge import merge_node, MergeOutcome, MergeConflict, MERGE_POLICIES, DEFAULT_MERGE_POLICY`

Merge subsystem the master DAG walker uses to fold each completed node's writes
into the live tree. Overlap detection compares a file's current bytes against the
node's `base_snapshot`. `MERGE_POLICIES = ("disjoint", "reconcile", "fail")`,
`DEFAULT_MERGE_POLICY = "reconcile"`.

A `Reconciler` is `Callable[[str, Optional[bytes], bytes, bytes],
Awaitable[bytes]]` — given a path and the (base, sibling, node) byte views, it
returns the merged bytes (async so it can route through a worker).

`MergeOutcome` (dataclass): `node_id`, `layer`, `overlapping_paths`, `policy`,
`conflict`, `resolution`, with `to_dict()` for `meta.json`.

---

## interaction/decision_engine.py

`from agy_orchestrator.interaction.decision_engine import DecisionEngine`

```python
DecisionEngine(auto_mode: bool = False, auto_resolver_instance=None)
async def resolve_question(self, question_text: str) -> str
```

When `auto_mode` is set and an `auto_resolver_instance` (an `AgentInstance`) is
supplied, the question is answered automatically via the resolver's
`run_async()`. Otherwise it falls back to a blocking human prompt on stdin.

---

## Putting it together

### `make_fallback_agent` + `QualityVerifier`

```python
import asyncio
from agy_orchestrator.core.agents.codex_agent import CodexAgent
from agy_orchestrator.core.agents.agy_agent import AgyAgent
from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent
from agy_orchestrator.execution.verifier import QualityVerifier

Generator = make_fallback_agent(
    [CodexAgent, AgyAgent],
    cycles=3,
    configs={AgyAgent: {"model": "pro", "effort": "high"}},
    model_fallbacks={CodexAgent: ["gpt-5.5", "gpt-5.4"]},
)

async def main():
    gen = Generator(prompt="Implement foo() in foo.py and a pytest for it.")
    gen.cwd = "/path/to/repo"
    out = await gen.run_async()
    print("produced via", gen.last_provider)

    verifier = QualityVerifier(["pytest -q"], timeout=600)
    result = await verifier.verify(working_directory="/path/to/repo")
    print(result.ok, result.message)

asyncio.run(main())
```

### `AdversarialReview`

```python
import asyncio
from agy_orchestrator.core.agents.codex_agent import CodexAgent
from agy_orchestrator.core.agents.agy_agent import AgyAgent
from agy_orchestrator.execution.verifier import QualityVerifier
from agy_orchestrator.workflows.adversarial import AdversarialReview

async def main():
    review = AdversarialReview(
        generator_instance=CodexAgent(prompt=""),
        critic_instance=AgyAgent(prompt="", model="pro", effort="high"),
        verifier=QualityVerifier(["pytest -q"]),
        max_iterations=5,
        working_directory="/path/to/repo",
    )
    out = await review.execute("Add a retry decorator with exponential backoff.")
    print(out, review.verified, review.approved, review.iterations_used)

asyncio.run(main())
```

### `TreeOfThought`

```python
import asyncio
from agy_orchestrator.core.agents.codex_agent import CodexAgent
from agy_orchestrator.core.agents.agy_agent import AgyAgent
from agy_orchestrator.workflows.tree_of_thought import TreeOfThought

async def main():
    req = "Write a thread-safe LRU cache."
    tot = TreeOfThought(
        branch_instances=[CodexAgent(prompt=req) for _ in range(3)],
        evaluator_instance=AgyAgent(prompt="", model="pro", effort="high"),
        selector="judge",
        requirement=req,
    )
    print(await tot.execute())

asyncio.run(main())
```

### `CascadeWorkflow`

```python
import asyncio
from agy_orchestrator.core.agents.agy_agent import AgyAgent
from agy_orchestrator.core.agents.codex_agent import CodexAgent
from agy_orchestrator.execution.verifier import QualityVerifier
from agy_orchestrator.workflows.cascade import CascadeWorkflow

async def main():
    cascade = CascadeWorkflow(
        stages=[AgyAgent(prompt="", model="flash"), CodexAgent(prompt="", model="gpt-5.5")],
        verifier=QualityVerifier(["pytest -q"]),
        max_iterations_per_stage=2,
        working_directory="/path/to/repo",
    )
    out = await cascade.execute("Fix the failing test in module x.")
    print(out, cascade.verified, "stage", cascade.stage_used)

asyncio.run(main())
```

### `MasterWorkflow`

```python
import asyncio
from agy_orchestrator.core.agents.codex_agent import CodexAgent
from agy_orchestrator.core.agents.agy_agent import AgyAgent
from agy_orchestrator.execution.verifier import QualityVerifier
from agy_orchestrator.workflows.master import MasterWorkflow

async def main():
    master = MasterWorkflow(
        model="gpt-5.5",
        effort="high",
        branches=3,
        max_iterations=5,
        verifier=QualityVerifier(["pytest -q"]),
        agent_class=CodexAgent,
        critic_agent_class=AgyAgent,
        critic_model="pro",
        checkpoint_path="/tmp/run.ckpt.json",
        working_directory="/path/to/repo",
    )
    out = await master.execute("Build a CLI that imports CSV into SQLite with a schema flag.")
    print(out, master.verified, master.iterations_used, master.plan_degraded)

asyncio.run(main())
```

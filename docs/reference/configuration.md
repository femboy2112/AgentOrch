# Configuration reference

Runtime tuning for AgentOrch is done with environment variables. Most behavior is
controlled by CLI flags on `python -m harness do` (see the README); the variables
below cover the lower-level knobs — worker watchdogs, transport retry, resource
caps, the Telegram bot, calibration, and computer-use testing.

Conventions used in this document:

- **Default-when-unset** is the value the code falls back to when the variable is
  absent or empty. Most numeric reads use the `os.environ.get(NAME, "<default>") or 0`
  idiom, so an **empty string is treated as `0`** (often "disabled") unless noted.
- Every row cites the exact `file:line` that reads the variable so the default and
  semantics can be re-verified against source.
- Booleans note the exact truthy/falsy tokens the code accepts; they are not all
  identical.

> Secrets: never commit `TELEGRAM_BOT_KEY` or print its value. The Telegram
> whitelist and state files (`AGY_TELEGRAM_USERS`, `AGY_TELEGRAM_STATE`) live
> **outside the repository** by default (`/home/leah/tgbot/data/…`) and must stay
> outside it — they hold user IDs and bot state, not repo data.

---

## Worker & resource bounds

Pin a worker child process's resource usage so an in-loop `pytest -n auto` (or
similar) can't OOM the host and take the orchestrator down with it.

| Variable | Reads at | Controls | Default | Values |
|---|---|---|---|---|
| `AGY_WORKER_RESOURCE_BOUND` | `agy_orchestrator/core/agent.py:216` | Master switch for pinning the worker child env (thread pools → 1, append pytest `-n K`). | `1` (on) | Off when value (lowercased, stripped) is `0`/`false`/`no`/`off`; any other value enables. |
| `AGY_WORKER_PYTEST_XDIST` | `agy_orchestrator/core/agent.py:255` | `-n K` appended to the worker's `PYTEST_ADDOPTS` (overrides an ini `-n auto`). | `2` | Integer. |
| `AGY_WORKER_PYTEST_MARKERS` | `agy_orchestrator/core/agent.py:259` | Marker expression folded into the worker's `PYTEST_ADDOPTS` (e.g. `not slow`); loses to an explicit `-m` in the command. | unset (no marker injected) | pytest marker expression string. |
| `AGY_WORKER_DRAIN_GRACE` | `agy_orchestrator/core/agent.py:202` | Seconds to keep draining a worker's pipe after the process exits before abandoning it. | `5` | Float seconds. |
| `AGY_WORKER_CMD_TIMEOUT` | `agy_orchestrator/core/agent.py:321` | Per-call wall-clock kill for a worker command, **even with active output** (folded into the tightest-of-set absolute cap). | `0` (disabled) | Float seconds; `0` = off. |

---

## Timeouts & watchdog

The per-call idle watchdog, absolute caps, and the verification/auxiliary command
timeouts.

| Variable | Reads at | Controls | Default | Values |
|---|---|---|---|---|
| `AGY_TIMEOUT` | `agy_orchestrator/core/agent.py:306` | Idle (per-call) timeout for a worker invocation. | `2400` (40 min) | Float seconds. |
| `AGY_ABSOLUTE_TIMEOUT` | `agy_orchestrator/core/agent.py:310` | Hard wall-clock cap for a single worker call regardless of liveness. | `0` (disabled) | Float seconds; `0` = off. |
| `AGY_STALL_SECONDS` | `agy_orchestrator/core/agent.py:341` | Idle-output stall threshold; arming it (`>0`) enables the streaming watchdog's STALLED trip. | `0` (disabled) | Float seconds; `0` = off. |
| `AGY_SILENCE_MAX_SECONDS` | `agy_orchestrator/core/agent.py:405` | Default-on ceiling: trip `WATCHDOG_STALLED` when a call has emitted **nothing at all** for this long, independent of `AGY_STALL_SECONDS`. | `600` | Float seconds; `0` = off. |
| `AGY_TEST_TIMEOUT` | `agy_orchestrator/execution/verifier.py:147` | Timeout for a verifier test command. | `600` | Float seconds. |
| `AGY_ABLATION_TIMEOUT` | `agy_orchestrator/workflows/reconcile.py:675` | Timeout for the reconcile station's ablation command. | `600` | Float seconds. |
| `AGY_SETUP_TIMEOUT` | `agy_orchestrator/workflows/vote.py:146` | Timeout for a per-candidate `--candidate-setup` bootstrap in vote mode. | `1200` | Float seconds. |
| `AGY_PRINT_TIMEOUT` | `agy_orchestrator/core/agents/agy_agent.py:131` | Passed to the `agy` CLI as `--print-timeout`. | `300s` | Duration string the `agy` CLI accepts (e.g. `300s`). |
| `AGY_SWARM_BRANCH_TIMEOUT` | `agy_orchestrator/execution/pipeline.py:28` | Per-branch wall-clock bound for a `ParallelSwarm` branch (backstop for a branch whose own watchdog hung). | `0` (unbounded) | Float seconds; malformed → unbounded. |
| `AGY_WATCHDOG` | `harness/roles.py:197` | Disables the calibration-driven watchdog budget for a dispatch. | enabled | Set to `off` (case-insensitive) to disable; any other value keeps it on. |
| `AGY_WATCHDOG_SCALE` | `harness/roles.py:212` | Multiplier applied to the calibrated `max_bytes` and `stall` budget (loosen/tighten the watchdog). Scales **both** dimensions together; use the two overrides below to move one independently. | `1.0` (no scaling) | Float `>0`; empty/`0`/non-numeric → no scaling. |
| `AGY_WATCHDOG_MAX_BYTES` | `harness/roles.py` (`_arm_watchdog`) | Absolute verbose **byte** budget (issue #83), applied *after* `AGY_WATCHDOG_SCALE` and **replacing** the byte budget alone — the stall budget stays calibrated+scaled. Decouples bytes from stall so a read-heavy primary generator isn't SIGKILLed as `runaway:verbose`. CLI mirror: `--watchdog-max-bytes`. Distinct from the legacy `AGY_MAX_OUTPUT_BYTES` early-return short-circuit. | unset (use calibrated+scaled budget) | Integer bytes `>0`; non-positive/non-numeric → ignored. |
| `AGY_WATCHDOG_STALL` | `harness/roles.py` (`_arm_watchdog`) | Absolute per-worker **stall** budget in seconds (issue #83), applied *after* `AGY_WATCHDOG_SCALE` and **replacing** the stall budget alone — the byte budget stays calibrated+scaled. Decouples stall from bytes. CLI mirror: `--watchdog-stall`. Distinct from the legacy `AGY_STALL_SECONDS` early-return short-circuit. | unset (use calibrated+scaled budget) | Float seconds `>0`; non-positive/non-numeric → ignored. |

---

## Transport & retry

Resilience to provider network resets and usage walls. Per-spell bounds measure
the current contiguous degradation; the cumulative signal survives single recovery
windows and decays during sustained clean output (see the inline design notes in
`agent.py`).

| Variable | Reads at | Controls | Default | Values |
|---|---|---|---|---|
| `AGY_TRANSPORT_RETRIES` | `agy_orchestrator/core/agents/fallback_agent.py:173` | Re-attempts on a transient transport error before cycling the fallback chain. | `2` | Integer, **clamped to 0–2**. |
| `AGY_TRANSPORT_BACKOFF` | `agy_orchestrator/core/agents/fallback_agent.py:182` | Base backoff seconds between transport re-attempts (multiplied by the 1-based attempt → 0.5s, 1.0s). | `0.5` | Float seconds. |
| `AGY_USAGE_WALL_BACKOFF` | `agy_orchestrator/core/agents/fallback_agent.py:199` | Per-attempt wait when a provider returns a usage/quota wall, so quota windows can elapse between cycles. | `5` | Float seconds (clamped `>=0`). |
| `AGY_TRANSPORT_MAX_ERRORS` | `agy_orchestrator/core/agent.py:357` | Max transport errors in a degraded spell before tripping `WATCHDOG_TRANSPORT_STALL`; also reused as the cumulative-signal error bound. | `25` | Integer. |
| `AGY_TRANSPORT_MAX_SECONDS` | `agy_orchestrator/core/agent.py:358` | Max wall-clock a degraded spell may persist; also the cumulative wall-clock bound. | `300` | Float seconds. |
| `AGY_TRANSPORT_RECOVERY_WINDOW` | `agy_orchestrator/core/agent.py:360` | Seconds of clean output that clears a spell (and gates the start of cumulative decay). | `60` | Float seconds. |
| `AGY_TRANSPORT_DECAY_PER_WINDOW` | `agy_orchestrator/core/agent.py:388` | Accumulated cumulative weight removed per recovery window of clean output. | `1.0` | Float; `0` disables decay (pure cumulative, never forgiving — not recommended). |

---

## Output & streaming

| Variable | Reads at | Controls | Default | Values |
|---|---|---|---|---|
| `AGY_MAX_OUTPUT_BYTES` | `agy_orchestrator/core/agent.py:340` | Output-size budget that, when `>0`, arms the streaming watchdog's runaway-output trip. | `0` (disabled) | Integer bytes; `0` = off. |
| `AGY_STREAM` | `agy_orchestrator/core/agent.py:1280` | Forces streaming mode for a call (streaming is also auto-enabled when a watchdog is armed or an event callback is set). | unset | Any non-empty value enables. |
| `AGY_BENCH_MOCK_OUTPUT` | `agy_orchestrator/core/agents/mock_agent.py:76` | Canned stdout returned by `MockAgent` (used with `AGY_BENCH_MOCK`). | unset (built-in mock output) | Arbitrary string. |

---

## Workflow & parallelism

| Variable | Reads at | Controls | Default | Values |
|---|---|---|---|---|
| `AGY_MAX_PARALLEL_NODES` | `agy_orchestrator/workflows/master.py:271`, `harness/cli.py:419` | Caps concurrent graph-plan nodes in `MasterWorkflow` (also read by the harness CLI to set the same bound). | unset → unbounded (degrades to serial where no DAG) | Digits only; non-digit/empty → `None`. |
| `AGY_MAX_PARALLEL_WORKERS` | `agy_orchestrator/workflows/vote.py:162` | Caps the whole candidate concurrency in a `--branches>1` vote swarm (complements `verifier_concurrency`). | unset → unbounded | Digits only; non-digit/empty → `None`. |
| `AGY_CASCADE_ISOLATE` | `agy_orchestrator/workflows/cascade.py:87` | Run each cascade tier in an isolated workspace. | on | Off when value is `0`, empty, or `false`; any other value enables. |
| `AGY_GENERATOR_ROTATE_AFTER` | `agy_orchestrator/workflows/adversarial.py:112` | Verify-failure count after which the adversarial loop rotates to the next generator in the chain. | `2` | Integer (floored at 1). |
| `AGY_PARALLELISM_CHECK` | `harness/roles.py:476` | Pre-dispatch parallelism sanity check. | enabled | Set to `off` (case-insensitive) to disable. |

---

## Telegram

The bot token comes from the environment; the whitelist and persisted state live
**outside the repo**. Never commit the key or the state/users files.

| Variable | Reads at | Controls | Default | Values |
|---|---|---|---|---|
| `TELEGRAM_BOT_KEY` | `harness/telegram.py:121` | Bot API token (secret — do not print or commit). | unset → bot disabled | Telegram bot token string. |
| `AGY_TELEGRAM_USERS` | `harness/telegram.py:291` | Path to the whitelist of allowed user IDs. **Keep outside the repo.** | `/home/leah/tgbot/data/users.json` | Filesystem path. |
| `AGY_TELEGRAM_STATE` | `harness/telegram.py:1526` | Path to persisted bot state (verbosity, etc.). **Keep outside the repo.** | `/home/leah/tgbot/data/bot_state.json` | Filesystem path. |
| `AGY_TELEGRAM_STATE_TTL` | `harness/telegram.py:1017` | Seconds the persisted-state cache is trusted before re-reading from disk. | `1.0` | Float seconds. |
| `AGY_TELEGRAM_VERBOSITY` | `harness/telegram.py:1516` | Default notification verbosity (a CLI/state setting overrides at runtime). | `normal` | One of `quiet`, `normal`, `verbose`, `debug` (normalized). |
| `AGY_TELEGRAM_LIVE_STALE_S` | `harness/telegram_bot.py:215` | Age after which a run's `events.jsonl` is treated as stale (run no longer live). | `1800` (30 min) | Float seconds; `0` = off. |

---

## Verification

| Variable | Reads at | Controls | Default | Values |
|---|---|---|---|---|
| `AGY_VERIFY_FULL_LOG_MAX` | `agy_orchestrator/execution/verifier.py:97` | Max chars of full verifier output persisted per `verify_step<N>_iter<M>.full.log` (head+tail kept when exceeded). | `1048576` (1 MiB) | Integer chars; `0` = unbounded. |
| `AGY_VERIFIER_MEM_MAX` | `agy_orchestrator/execution/verifier.py:155` | When set, each test command runs in a transient `systemd-run --user --scope` with this `MemoryMax`; a spike is OOM-killed in-scope so the orchestrator survives. Requires a usable user systemd scope manager. | unset (uncapped) | systemd memory size, e.g. `3G`. |
| `AGY_VERIFIER_SWAP_MAX` | `agy_orchestrator/execution/verifier.py:158` | `MemorySwapMax` for the verifier scope. | `0` (hard ceiling, no swap — fail fast vs freeze) | systemd memory size. |

---

## Model & codex

| Variable | Reads at | Controls | Default | Values |
|---|---|---|---|---|
| `CODEX_HOME` | `agy_orchestrator/core/agents/codex_agent.py:31` | Home directory the `codex` CLI uses; sessions/rollout JSONL are read from `<home>/sessions/…`. | `~/.codex` | Filesystem path. |
| `AGY_CODEX_USAGE_WALL_PERCENT` | `agy_orchestrator/core/agents/codex_agent.py:22` | Out-of-band usage-wall detection threshold from rollout `used_percent` (a real wall reports `100.0`; threshold kept deliberately high). | `99` | Float percent; `0` disables the detector. |
| `AGY_CODEX_MODEL_FALLBACKS` | `harness/roles.py:136` | Overrides the intra-codex model fallback order (names not in `AVAILABLE_MODELS` are dropped). | unset → use `AVAILABLE_MODELS` order | Comma-separated model names; **empty value disables** intra-codex fallback. |
| `AGY_CRITIC_FAMILY_CHECK` | `harness/roles.py:443` | Cross-family guard warning when generator and critic are the same provider family. | enabled | Set to `off` (case-insensitive) to disable. |

---

## Benchmark & calibration

The live ledger and calibration sweep refine per-config watchdog budgets from real
traffic. Default paths are under `/tmp` and can be redirected.

| Variable | Reads at | Controls | Default | Values |
|---|---|---|---|---|
| `AGY_BENCH_MOCK` | `harness/roles.py:58` | Routes dispatches to `MockAgent` instead of real worker CLIs (hermetic benchmarks/tests). | unset (off) | Any non-empty value enables. |
| `AGY_BENCH_MOCK_SLEEP` | `agy_orchestrator/core/agents/mock_agent.py:70` | Base simulated model-latency (seconds per `MockAgent` call). | `0.30` | Float seconds; malformed → default. |
| `AGY_BENCH_MOCK_SLEEP_PER_KCHAR` | `agy_orchestrator/core/agents/mock_agent.py:71` | Extra simulated latency added per 1000 prompt chars (so mock latency optionally scales with prompt size). | `0.0` | Float seconds; malformed → default. |
| `AGY_BENCH_MOCK_CACHE_RATIO` | `agy_orchestrator/core/agents/mock_agent.py:103` | Synthetic `cache_read` fraction reported in `MockAgent` usage (clamped to `0..0.95`). | `0.0` | Float `0..0.95`; malformed → default. |
| `AGY_CALIBRATION_JSONL` | `agy_orchestrator/core/calibration.py:65` | Path the offline calibration sweep writes/reads. | `/tmp/agentorch_research/calibrate.jsonl` | Filesystem path. |
| `AGY_LIVE_LEDGER_JSONL` | `agy_orchestrator/core/calibration.py:76` | Path the live ledger (appended after each verified dispatch) is written/read. | `/tmp/agentorch_research/live_ledger.jsonl` | Filesystem path. |
| `AGY_LIVE_LEDGER` | `agy_orchestrator/core/calibration.py:166`, `:258` | Toggles writing/reading the live ledger alongside the offline sweep. | enabled | Set to `off` (case-insensitive) to disable. |

---

## Dispatch & reconciliation

| Variable | Reads at | Controls | Default | Values |
|---|---|---|---|---|
| `AGY_RECONCILE` | `harness/dispatch.py:1575` | Enables the reconcile (integration-skeptic) station for a dispatch. | off | Enabled when value (lowercased) is `1`, `true`, or `on`. |
| `AGY_HEARTBEAT_SECONDS` | `harness/dispatch.py:1762` | Interval between run-level heartbeat events. | `30` | Float seconds. |
| `AGY_NOTIFY` | `harness/cli.py:494` | Default notify target when `--notify` is not passed. | unset (no notify) | Notify target string (overridden by `--notify`). |
| `AGY_MISSION_CRITICAL_RUN_STALL` | `harness/dispatch.py:367` | Overrides the mission-critical run-level stall backstop. Only consulted when `--mission-critical` is set and `--run-stall-abort` is unspecified. | `1800.0` (`MISSION_CRITICAL_RUN_STALL_DEFAULT`) | Float seconds; `<=0` disables the backstop; malformed → default. |
| `AGY_BROKER_POOL_WAIT` | `harness/broker.py:60` | Anti-deadlock wait for a worker slot in the broker pool (explicit arg > env > default). | `5.0` | Float seconds (clamped `>=0`); non-numeric → warn + default. |

---

## Snapshots

| Variable | Reads at | Controls | Default | Values |
|---|---|---|---|---|
| `AGY_SNAPSHOT_BLOCK_BYTES` | `harness/snapshot.py:27` | Block size for the snapshot content hash. | `1048576` (1 MiB) | Integer bytes; floored at `4096`; empty → 1 MiB. |

---

## Computer-use / GUI testing

These gate the real-browser/real-GUI paths in the computer-use subsystem. Outside
those paths the code falls back to a fake controller, so set them explicitly when
you want a live browser/GUI.

| Variable | Reads at | Controls | Default | Values |
|---|---|---|---|---|
| `AGY_BROWSER_E2E` | `agy_orchestrator/computer_use/session.py:270` | Enables the real-browser end-to-end path (otherwise a fake browser controller is used). | unset (fake) | Set to `1` to enable. |
| `AGY_REALGUI_TEST` | `agy_orchestrator/computer_use/adapter.py:72`, `session.py:269` | Enables the real-GUI testing path (also counts as a "pytest-like" context in the session guard). | unset (off) | Set to `1` to enable. |

> Other process/system variables the code reads but does not own as tunables:
> `PYTEST_CURRENT_TEST` (test-context detection, `session.py:269`), `PATH`
> (`computer_use/xauth.py:77`, with a safe fallback), and the ambient
> `DISPLAY`/`XAUTHORITY` used by the computer-use action executor. These are not
> AgentOrch configuration knobs and are listed only for completeness.

# Computer-Use Worker — Real-GUI Security Harness (Design Addendum)

**Status:** authoritative build spec for the real-GUI extension of the
computer-use worker. Stacks on `COMPUTER_USE_DESIGN.md` (the base worker, PR #27)
and is built on branch `feat/computer-use-worker`.

**Relation to the base design.** The base worker can only *actuate* an isolated
Xvfb (`display_scope == "isolated"`, enforced at three layers: `ActionSpec.__post_init__`,
`SafetyKernel.validate` FR-04, and the `ReasoningInput` constraint), and can only
*read* the real `:0` in OBSERVE mode (structurally input-incapable). This addendum
adds a **deliberately gated** ability to do real work on the operator's real `:0`,
governed by an ownership-aware **security harness**. Everything the base design
guarantees (XAUTHORITY isolation #1, killable tree #2, hard rlimits #3, OBSERVE
redaction #4) remains in force unchanged; this addendum only adds a fourth,
opt-in actuation path and the harness that polices it.

---

## 1. Goals (operator requirements, verbatim intent)

1. A security harness for the agent doing work **in the real GUI** (`:0`).
2. An **ask-mode**: the agent may pause and ask the operator **directly through a
   GUI pop-up**. The pop-up presents the question, a set of **multiple-choice**
   answers, **and** a free-text box for the operator to type a direct response.
3. **Process-ownership permission model:**
   - If a process is a **child of the computer-use agent** → the agent may
     interact with it **freely, no permission needed**.
   - If a process was **running *before* the agent started** *and/or* is
     **unrelated to the task** → the agent has **no default access**; it must
     **ask for permission** (only possible when ask-mode is enabled; otherwise the
     interaction is denied).

**Operator decisions locked for this build (2026-05-29):**

- **D1 — two selectable real-GUI policies (both modes ship):**
  - `real_full`: ownership-gated full reach. Owned-child windows: free. Foreign
    windows: hard-denied unless granted via the pop-up.
  - `real_children`: the agent may inject input **only** into windows of apps it
    itself launched. Foreign/pre-existing windows are **never** input-injectable;
    ask-mode for them governs only reading/focusing, never injection.
- **D2 — grant scope is chosen by the operator in real time, inside the pop-up.**
  The dialog offers the scope choice per prompt (allow once / allow this process
  for the run / allow this process for a bounded time / deny), plus the free-text
  box. There is no pre-configured blanket grant.
- **D3 — stack on `feat/computer-use-worker`** (no intermediate merge of PR #27).

---

## 2. The harness in one paragraph

At session start the harness records a **baseline**: every PID alive and every X
window present on `:0` at that instant. For the rest of the run, a process is
**OWNED** iff `ProcessSupervisor.is_owned(pid)` is true (a descendant in a process
group we registered) **and** its PID is not in the baseline; otherwise it is
**FOREIGN**. Every real-`:0` action is resolved to the PID that owns its target
window (`_NET_WM_PID`); owned → allowed; foreign → denied by default, and only
escalatable through the GUI pop-up when policy + ask-mode permit. Because the
operator's other terminals, claude/codex/grok/agy sessions, and the orchestrator
are all alive at session start, they are all **baseline-foreign and protected by
default** — this is exactly the original mission-critical "must not interfere with
my other instances" guarantee, now enforced structurally rather than by prose.

---

## 3. New modes, scopes, and policies

### 3.1 RunMode (extend `models.RunMode`)

```
ISOLATED  (unchanged, default)
OBSERVE   (unchanged, read-only real :0)
REAL      (NEW — real-:0 actuation, opt-in only, governed by the harness)
```

`REAL` is never the default. It is selected explicitly per run
(`RunRequest.mode == "REAL"` / `--computer-use-mode real`). Selecting it requires
`capabilities.action_exec` and a reachable real `:0`; otherwise the session
degrades to OBSERVE (fail-safe, audited).

### 3.2 Scope (extend `models.Scope`)

```
isolated      (unchanged)
observe_real  (unchanged)
real_act      (NEW — actuation against the real :0, gate-only constructible)
```

### 3.3 Real-GUI policy (NEW config on the session)

```
real_gui_policy ∈ { "full", "children" }     # D1; required when mode==REAL
ask_mode        ∈ { "on", "off" }            # default "on" when mode==REAL
```

`real_gui_policy` and `ask_mode` live on `WorkerSession` and are surfaced in the
`ReasoningInput.constraints` so the reasoner knows what it may attempt.

---

## 4. Ownership & baseline model

### 4.1 Baseline capture (FR-28)

At `SessionController` start, before any owned process is spawned, capture:

- `baseline_pids: set[int]` — every PID currently alive (psutil snapshot).
- `baseline_windows: dict[window_id, pid]` — every X window on the real `:0` and
  its `_NET_WM_PID` (via the same EWMH/`wmctrl -lp`/`xprop` path OBSERVE already
  uses; PID may be `None` for legacy clients → treated as foreign).

The baseline is immutable for the run and stored on the session + emitted as a
`baseline.captured` event.

### 4.2 Classification (the single source of truth)

A target PID `p` is classified by `OwnershipResolver.classify(p)`:

```
OWNED   ⟺ ProcessSupervisor.is_owned(p) is True  AND  p ∉ baseline_pids
FOREIGN ⟺ otherwise   (pre-existing, non-descendant, unknown, or unresolvable)
```

Rationale for the `p ∉ baseline_pids` conjunct: PID reuse. A baseline PID that
dies and whose number is later reused by one of our children must **not** silently
become "owned"; the resolver re-confirms ownership via live pgid/ancestry, and any
PID number that was in the baseline is treated conservatively. (If our own child
genuinely reuses a retired baseline number, ownership is re-established only by a
fresh `ProcessSupervisor` registration that supersedes the baseline entry; the
resolver prefers the registry over the baseline when the registration timestamp is
after baseline capture.)

"Unrelated to the task": a process the agent spawned but that is **not** in the
registry (e.g. a daemon that re-parented away) is FOREIGN by construction, since
`is_owned` only returns true for registry-tracked pgid/ancestry. No extra
heuristic is needed — non-ownership *is* the "unrelated" signal.

### 4.3 Target → PID resolution (FR-29)

For a `real_act` action, the harness must resolve the target to an owning PID:

- **element target** → the snapshot element already carries `app_pid`
  (`ElementHandle.app_pid`); use it.
- **coordinate target** → find the topmost window whose bbox contains the point
  (window stack from the perception snapshot), then its `_NET_WM_PID`.

If the PID cannot be resolved (no `_NET_WM_PID`, ambiguous stack, stale snapshot)
the action is **DENIED** (`TARGET_UNRESOLVABLE` / fail-closed). The harness never
acts on a target it cannot attribute to a PID.

---

## 5. SafetyKernel real-act gate (decision flow)

`SafetyKernel.validate` gains a `real_act` branch. The base `isolated` and
`observe_real` paths are **unchanged**. For `display_scope == "real_act"`:

```
1. mode != REAL                       → DENY  (REAL_ACT_NOT_PERMITTED)
2. resolve target → pid; unresolved   → DENY  (TARGET_UNRESOLVABLE)
3. classify(pid):
   OWNED                              → ALLOW (no prompt)            [FR-30]
   FOREIGN:
     policy == "children"            → DENY  (FOREIGN_PROCESS)      [FR-31]
     policy == "full":
       check grant cache for pid:
         valid grant present         → ALLOW                        [FR-36]
         else:
           ask_mode == "off"         → DENY  (ASK_MODE_DISABLED)    [FR-32]
           ask_mode == "on"          → PROMPT operator (§6)         [FR-33]
             operator deny / timeout / cancel → DENY (fail-closed)  [FR-35]
             operator allow(scope, text)      → cache + ALLOW       [FR-34/36]
4. all standard checks still apply (budgets, schema, risk/confirmation FR-09,
   redaction context, etc.). real_act high-risk actions still require the
   FR-09 confirmation token in addition to the ownership grant.
```

A `real_act` `ActionSpec` is **only constructible by the kernel after this gate
passes** (HARNESS INVARIANT E). The reasoner cannot fabricate a directly-executable
real-`:0` action: `ActionSpec.__post_init__` is relaxed to permit
`display_scope ∈ {"isolated", "real_act"}`, but a `real_act` spec carries a
kernel-issued `clearance_token` (signed per run+intent+pid+scope); `ActionExecutor`
refuses any `real_act` spec lacking a clearance token that matches the live grant.
Isolated specs are unaffected.

---

## 6. GUI ask path (the pop-up)

### 6.1 Mechanism

`zenity --forms` (zenity is present on the host; `tkinter` is not, so no new dep).
A single dialog combines:

- **Title/text**: the question + context — action type, target app title, PID,
  window id, the agent's rationale, and the run id.
- **`--add-combo "Decision"`** populated with the multiple-choice grant scopes
  (§6.2). This is the "multiple choice answers" requirement.
- **`--add-entry "Your instructions (optional)"`** — free-text the operator can
  type as a direct response. This is the "write text directly in response"
  requirement.

The prompter is a small `GuiPrompter` class with a single method
`prompt(context) -> PromptResult`. It is **dependency-injected** into the
`SafetyKernel`/`SessionController` so tests substitute a `FakePrompter` and **no
real dialog is ever spawned in CI**.

### 6.2 Decision choices (D2 — chosen in real time)

The combo offers exactly:

```
"Deny"                         → DENY
"Allow once"                   → grant scope = ACTION  (this one action only)
"Allow this process (run)"     → grant scope = PROCESS_RUN (pid for rest of run)
"Allow this process (10 min)"  → grant scope = PROCESS_TTL (pid, bounded TTL)
```

The free-text box is independent of the choice: the operator may deny **and** type
a redirect ("don't touch the browser, use the terminal I opened"), or allow and add
a constraint. The text is always captured and routed back (§6.4).

### 6.3 Rendering on the real `:0` (FR-39)

The pop-up is **for the operator**, so it must appear on the real `:0` — the
`GuiPrompter` spawns zenity with the operator's real `DISPLAY`/`XAUTHORITY` (NOT
the isolated cookie). This is a deliberate, audited, narrow exception to XAUTHORITY
isolation #1 and is safe because: (a) zenity is spawned as an **owned child**
(`ProcessSupervisor`, killable, rlimited), (b) it only renders **our own dialog
window** and injects **nothing** into any foreign window, (c) it reads only the
operator's typed response. A test asserts the prompter performs no input-injection
syscalls/xdotool calls against foreign windows.

### 6.4 Free-text routing (FR-37)

The operator's typed text is appended to `ReasoningInput.orchestrator_messages` as
an `operator_note` on the next reasoning step, so the operator can steer the agent
through the same pop-up. It is redaction-passed (#4) before storage only if it
would re-enter a prompt destined for a reasoner — operator-authored text is trusted
but still scrubbed for accidental secret paste, default-on (consistent with #4).

### 6.5 Fail-closed (FR-35)

Prompt timeout (`budgets.confirmation_wait_timeout_ms`), operator closes the
window, zenity non-zero/unavailable, or empty/garbled return → **DENY**. Every
prompt and outcome is audited.

---

## 7. Grant cache

`GrantCache` keyed by `pid`:

- `ACTION` grants are consumed by the next allowed action against that pid (single
  use; the following action on the same pid re-prompts).
- `PROCESS_RUN` grants persist until the run ends.
- `PROCESS_TTL` grants store an expiry; expired → re-prompt (clock is injectable
  for deterministic tests — no wall-clock in test paths).
- A grant is bound to `(run_id, pid)`. A **new** foreign PID always re-prompts even
  if a sibling was granted (FR-36). If the granted PID dies, its grant is dropped.

The cache is in-memory per run (never persisted across runs — every run re-earns
its foreign access).

---

## 8. Defaults & fail-safes

- Mode default stays **ISOLATED**. `REAL` is opt-in.
- `ask_mode` defaults **on** when `mode==REAL`.
- `ask_mode == off` ⇒ all foreign interaction is **hard-denied** (no prompt); the
  agent is effectively children-only regardless of policy.
- `real_gui_policy == children` ⇒ foreign input-injection is impossible even with
  ask-mode on and a grant (HARNESS INVARIANT D).
- If the real `:0` is unreachable or `zenity` is missing while `ask_mode==on` and a
  foreign interaction is needed → **DENY** (fail-closed) + degrade note.

---

## 9. Data-model additions (`models.py`)

- `RunMode.REAL = "REAL"`; `Scope.REAL_ACT = "real_act"`.
- `RealGuiPolicy(str,Enum) = {FULL="full", CHILDREN="children"}`.
- `AskMode(str,Enum) = {ON="on", OFF="off"}`.
- `GrantScope(str,Enum) = {ACTION, PROCESS_RUN, PROCESS_TTL, DENY}`.
- `WorkerSession`: add `real_gui_policy`, `ask_mode`, `baseline_pids`,
  `baseline_windows`.
- `ActionSpec`: add `clearance_token: Optional[str]`; relax `__post_init__` to
  permit `display_scope ∈ {"isolated","real_act"}` (real_act requires a
  clearance_token to be set, else raise — isolated unaffected).
- New `@dataclass PromptContext`, `PromptResult`, `Grant`.
- New `ViolationCode`s: `REAL_ACT_NOT_PERMITTED`, `ASK_MODE_DISABLED`,
  `GRANT_REQUIRED`, `GRANT_EXPIRED`, `CLEARANCE_TOKEN_INVALID`.
- New `WorkerEventType`s: `BASELINE_CAPTURED`, `PERMISSION_PROMPT_SHOWN`,
  `PERMISSION_GRANTED`, `PERMISSION_DENIED`, `FOREIGN_INTERACTION_BLOCKED`,
  `OPERATOR_NOTE_RECEIVED`.

All additions keep `to_dict`/`from_dict` roundtrip fidelity and enum-value tests.

---

## 10. New modules

```
agy_orchestrator/computer_use/
  ownership.py     # OwnershipResolver: baseline capture + classify(pid)
  gui_prompt.py    # GuiPrompter (zenity --forms) + FakePrompter (tests)
  grants.py        # GrantCache (scopes, TTL via injected clock)
```

Wiring: `SafetyKernel` takes an optional `OwnershipResolver`, `GuiPrompter`, and
`GrantCache` (all injectable; sane defaults). `SessionController` captures the
baseline at start and threads policy/ask_mode from `RunRequest`. `ActionExecutor`
verifies the `clearance_token` before any `real_act` actuation.

CLI/adapter: `--computer-use-mode {isolated,observe,real}` (extend existing flag),
`--real-gui-policy {full,children}`, `--ask-mode {on,off}`. `harness/roles.py` and
`harness/cli.py` pass them through.

---

## 11. Invariants (hard, tested, never relaxed)

- **A — baseline protection:** every PID alive at session start is FOREIGN for the
  run's lifetime; the operator's other terminals/claude/codex/grok/agy/orchestrator
  are default-denied. *(the mission-critical guarantee)*
- **B — ownership truth:** only `ProcessSupervisor`-registered descendants
  (post-baseline) are OWNED.
- **C — fail-closed:** ask-mode off, prompt timeout/cancel, missing zenity,
  unresolvable PID, or any ambiguity → DENY.
- **D — children policy never injects foreign:** no grant can enable foreign input
  injection under `real_gui_policy == "children"`.
- **E — gate-only real_act:** an executable `real_act` action exists only after the
  kernel issues a matching clearance token; the reasoner cannot fabricate one.
- **F — base untouched:** ISOLATED/OBSERVE behavior and all base FRs/tests remain
  byte-for-byte intact; default mode stays ISOLATED.

---

## 12. Functional requirements (new)

- **FR-26** REAL mode is opt-in; default remains ISOLATED; unreachable `:0` →
  degrade to OBSERVE.
- **FR-27** `real_act` actions must pass the §5 ownership gate.
- **FR-28** baseline PID + window/`_NET_WM_PID` capture at session start.
- **FR-29** target→PID resolution; unresolved → DENY.
- **FR-30** owned-child target → allow with no prompt.
- **FR-31** foreign target + `children` policy → DENY (no prompt).
- **FR-32** foreign target + `full` + ask-mode off → DENY.
- **FR-33** foreign target + `full` + ask-mode on → GUI prompt; gate on response.
- **FR-34** prompt offers the four real-time grant choices + a free-text entry.
- **FR-35** timeout/cancel/missing-zenity → DENY, audited (fail-closed).
- **FR-36** grant cache honors chosen scope; a new foreign PID always re-prompts.
- **FR-37** operator free-text routes back as an `operator_note` to the reasoner.
- **FR-38** every permission decision audited to `events.jsonl`.
- **FR-39** the pop-up renders on real `:0` but injects nothing into foreign
  windows (own dialog only).
- **FR-40** *(mission-critical regression)* a simulated "operator's other terminal"
  PID placed in the baseline is denied under `full` policy without an explicit
  grant — proving the agent cannot touch pre-existing instances by default.

---

## 13. Testing strategy (release-blocking first)

All tests are hermetic: a `FakePrompter` (no real zenity), an injected clock for
TTL, and `FakeOwnershipResolver`/synthetic baselines. No real `:0` actuation, no
real foreign-window input ever runs in CI.

Release-blocking:

1. **FR-40 mission-critical:** baseline contains a fake "operator terminal" PID →
   `real_full` + ask-on, no grant, prompter set to auto-deny → action DENIED;
   assert the prompter was even consulted (proving default-no-access), and that the
   PID is classified FOREIGN although it would pass a naive same-UID check.
2. **FR-30:** spawn an owned child via `ProcessSupervisor`; its window/pid →
   classified OWNED → `real_act` ALLOWED with **no** prompt (prompter asserted
   never called).
3. **FR-31:** foreign pid + `children` policy → DENY, prompter never called.
4. **FR-32:** foreign pid + `full` + ask-off → DENY, prompter never called.
5. **FR-33 + FR-34:** foreign pid + `full` + ask-on → prompter invoked with a
   context carrying app/pid/rationale; choices include the four scopes + free-text.
6. **FR-36 grant scopes:** `ACTION` grant → next action on same pid re-prompts;
   `PROCESS_RUN` → subsequent same-pid actions allowed, new pid re-prompts;
   `PROCESS_TTL` → allowed before expiry, denied after (injected clock).
7. **FR-35 fail-closed:** prompter raises/timeouts/returns empty → DENY; missing
   zenity simulated → DENY.
8. **FR-37:** operator free-text appears as an `operator_note` in the next
   `ReasoningInput.orchestrator_messages`.
9. **FR-39:** `GuiPrompter` spawns only its own zenity child and performs no
   xdotool/input call against any foreign window (mock the actuation layer; assert
   zero foreign-target calls).
10. **HARNESS INVARIANT E:** a `real_act` `ActionSpec` without a valid
    clearance_token is refused by `ActionExecutor`; the reasoner-produced
    `ActionIntent` cannot bypass the kernel.
11. **HARNESS INVARIANT F / regression:** the entire existing computer-use suite +
    full `pytest -q` remain green (isolated/observe untouched).

Mark new release-blocking tests `@pytest.mark.release_blocking` and a
`@pytest.mark.realgui` group. Real-zenity / real-`:0` integration tests are
`@pytest.mark.slow` and skipped without an explicit display + `AGY_REALGUI_E2E=1`.

---

## 14. Out of scope (this build)

- Wayland actuation (X11 `:0` only; Wayland → degrade/observe).
- Persisting grants across runs.
- A non-zenity GUI backend (zenity-only; abstract behind `GuiPrompter` so a future
  tkinter/Qt backend can drop in).
- Multi-seat / multiple real displays (single real `:0`).

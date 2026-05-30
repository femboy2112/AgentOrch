# Computer-Use Phase 1.x — Autonomous Browser/DOM Actuation (design addendum)

**Status:** design for the build that makes the FULL autonomous
perceive→reason→act loop able to drive a real browser end-to-end — the
capability `scripts/agent_browser_demo.py` proved out of band (verified
2026-05-29: agent-owned Playwright Chromium-1217 on `:0`, Bing search,
paginate, click the Nth result by DOM, follow the new tab).

**Operator bar (2026-05-29):** "I'll merge the real-GUI branch when it actually
works with real-GUI. Keep building until P1.x works." So this is a **merge
prerequisite**, on branch `feat/computer-use-worker`, stacked on the real-GUI
harness (commits `92f6f90`/`e659843`/`a2f1736`).

This builds ON the existing worker; it does not rebuild it. Base ISOLATED/
OBSERVE/REAL behavior, the ownership+baseline model, the SafetyKernel gate, and
all existing FRs/tests stay **byte-for-byte intact** (INVARIANT F).

---

## 1. The three gaps (why the autonomous loop can't do this today)

1. **`navigate` is a stub.** `action_executor.py:279` spawns `true` — there is
   no real navigation. (Verified.)
2. **Agent-launched apps land on the isolated Xvfb, not `:0`.** `launch_app`/
   `navigate` force `display_scope="isolated"` via `ProcessSupervisor.spawn`, so
   a browser the agent opens is invisible to the operator and cannot be the
   target of `:0` work.
3. **No DOM actuation.** `BrowserDOMCollector` (perception.py:162) *connects*
   over CDP to prove the path, then returns `[]`. There is no click-element /
   type-into-DOM / follow-new-tab actuation, and the REAL-mode CLI flags
   (`--computer-use-mode` / `--real-gui-policy` / `--ask-mode`) are only
   partially wired (roles/shim/adapter accept the keys; argparse in
   `harness/cli.py` does not surface them).

The prototype already proves the *mechanism* (owned CDP Chromium, DOM
click-by-index, new-tab follow, Bing as the bot-tolerant engine). Phase 1.x
folds that mechanism into the worker's loop, under the ownership model.

---

## 2. New component: `BrowserController` (`computer_use/browser.py`)

A dependency-injected, **agent-owned**, CDP-controlled browser session. One per
run, lazily created on the first `navigate`.

- **Launch (real, owned):** launch Chromium via Playwright with
  `executable_path` = the cached `chromium-1217/chrome-linux64/chrome`
  (fallback: scan `~/.cache/ms-playwright/chromium-*`), args
  `--no-sandbox --disable-dev-shm-usage --disable-gpu
  --disable-blink-features=AutomationControlled
  --remote-debugging-port=0`, on the run's **chosen display**:
  - REAL mode → operator `:0` (visible; the browser is the agent's own child).
  - ISOLATED mode → the private Xvfb (default for tests/CI).
- **Ownership:** the launched browser process is registered with
  `ProcessSupervisor` exactly like any owned child (`start_new_session`,
  rlimits, killpg tree-reap). Therefore `is_owned(browser_pid)` is True and the
  browser is **never** a foreign target → driving it needs **no** permission
  prompt (this is the "child of the agent" rule from the original spec). The
  controller injects input **only** into its own browser via CDP — never into
  any foreign window.
- **CDP endpoint:** the controller exposes the browser's CDP endpoint so the
  existing `BrowserDOMCollector` perception path can read live DOM (closing gap
  3's perception half), and so DOM actuation rides the same connection.
- **Lifecycle:** owned by `SessionController`; torn down on session close via
  the existing `terminate_tree` (killpg), so no browser leaks. Fail-closed: any
  launch/connect error → the action returns FAILED, loop continues degraded.

`FakeBrowserController` (in-memory, scripted pages/results, zero subprocess/zero
Playwright/zero `:0`) is the mandatory CI double, mirroring FakePrompter/
FakeOwnershipResolver. **No hermetic test ever launches a real browser.**

---

## 3. Action contract (executor changes)

Extend `ActionExecutor` (additive; isolated xdotool + real_act gate paths
unchanged):

- **`navigate {url}`** → `BrowserController.navigate(url)`: ensures the owned
  browser exists (launch on first use), navigates the active page, returns the
  landed URL + the owned browser pid in `spawned_process_ids`. Replaces the
  `true` stub.
- **`click` / `type` with a `dom` target** → new target kind
  `{"kind": "dom", "selector": "<css>", "index": <n>}` (1-based). Routed to
  `BrowserController.click_dom(selector, index)` /
  `type_dom(selector, index, text)`. Deterministic; supports new-tab follow
  (capture `context.pages[-1]`), as the prototype does. Coordinate/`real_act`
  targets keep their existing behavior.
- DOM actions are valid only when a `BrowserController` exists for the run; a
  `dom` target with no browser → REJECTED `browser_not_open` (fail-closed).

The reasoner schema (`reasoner.py`) gains the `dom` target kind + a short note
that `navigate` opens an agent-owned browser, so the model can plan
search→click-Nth autonomously.

---

## 4. Ownership / safety invariants (hard, tested)

- **B1 — Owned browser, no prompt.** The CDP browser is an owned child
  (`is_owned` True, not in baseline) → DOM actuation against it is allowed with
  **no** GUI prompt, under both `real_full` and `real_children` policies. (It is
  the agent's child; the children policy explicitly permits owned children.)
- **B2 — No foreign injection, ever.** The controller drives only its own
  browser over CDP. It issues **zero** xdotool/CDP input against any window it
  did not launch. Foreign-window actuation remains gated by the existing
  SafetyKernel real_act path (unchanged).
- **B3 — Killable + leak-free.** The browser tree is reaped via
  `ProcessSupervisor.terminate_tree` (killpg) on session close and on
  timeout/watchdog (inherits the runaway-codex fix `a2f1736`).
- **B4 — Base untouched (INVARIANT F).** ISOLATED/OBSERVE/REAL gate behavior and
  every existing test stay byte-for-byte identical; all additions are new
  module + additive branches + new tests.

---

## 5. CLI wiring (close gap 3's flag half)

Surface in `harness/cli.py` argparse and thread through `harness/roles.py` →
adapter (keys already accepted):
`--computer-use-mode {isolated,observe,real}`,
`--real-gui-policy {full,children}`, `--ask-mode {on,off}`, plus
`--browser-engine {bing,duckduckgo,google}` (default **bing** — verified
bot-tolerant; Google/DDG block automated browsers) and `--browser-display`
(default: `:0` in REAL, isolated Xvfb otherwise). Defaults preserve current
behavior (mode default stays ISOLATED).

---

## 6. Autonomous dogfood acceptance (the operator's "it works" bar)

A slow, env-gated e2e (`AGY_BROWSER_E2E=1`, `@pytest.mark.slow`) that drives the
**full adapter loop** (perceive→reason→act), not the standalone script:
objective = "search '<query>' and open result number N". The loop must
`navigate` to the engine, read results via DOM perception, and emit a
`click {kind: dom, index: N}` that lands on the Nth result's destination — with
the owned-browser invariants (B1/B2) asserted and an **honest** failure if the
engine bot-blocks (never a faked pass). Mirrors the verified prototype run.

---

## 7. Testing strategy (all hermetic unless env-gated)

Release-blocking, `FakeBrowserController` + Fakes only:
- `navigate` opens/uses the owned browser; returns landed URL + owned pid.
- DOM `click`/`type` route to the controller; new-tab follow returns the child
  page URL; index is 1-based; missing browser → `browser_not_open`.
- B1: owned browser DOM actuation allowed with no prompt under full + children.
- B2: controller performs zero foreign-window injection (assert no xdotool/CDP
  call targets a non-owned pid/window).
- B3: session close / timeout reaps the browser tree (no leak).
- B4: full prior `pytest -q` stays green.
- CLI: the three (+two) flags parse and thread to the adapter RunRequest.
The real-browser autonomous e2e (§6) is the only non-hermetic test, gated off by
default.

---

## 8. Out of scope

Wayland; non-Chromium engines; cross-run browser reuse; defeating Google's
bot-detection (we route to Bing and report honestly). These are explicitly
deferred.

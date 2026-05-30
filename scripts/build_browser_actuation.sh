#!/usr/bin/env bash
# Phase 1.x guarded build: autonomous browser/DOM actuation for the computer-use
# worker. Stacks on feat/computer-use-worker (real-GUI harness already there).
#
# Feeds COMPUTER_USE_BROWSER_DESIGN.md via --spec, runs master mode (plan -> ToT
# -> adversarial), test-gated on the hermetic pytest suite (the real-browser
# autonomous e2e is env-gated OFF by default, so the gate stays deterministic).
#
# Worker chain (2026-05-29): grok-gen / codex-critic lead (proven clean alongside
# the agi2 build's codex-gen/agy-critic happy path). claude is slotted 3rd —
# allowed now as a MID-CHAIN fallback per the claude-worker-headroom note (20x
# Max headroom; only engages if grok AND codex both wall, never sole lead). agy
# last. This honors the account-sharing rule while spreading load off grok.
#
# Builds INTO the AgentOrch repo on the current branch -> default cwd, no --out-dir.
# Carries worker --dangerously flags -> the classifier blocks the driving agent
# from spawning it, so the OPERATOR fires this via `!`.

set -uo pipefail
cd /home/leah/AgentOrch

PY=/home/leah/AgentOrch/.venv/bin/python
SPEC=COMPUTER_USE_BROWSER_DESIGN.md
LAUNCHLOG=/tmp/browser_build_launch.log

# Self-daemonize so the build survives the `!` wrapper exiting.
if [[ "${_BR_DETACHED:-}" != "1" ]]; then
  # Pre-create the log so its existence alone proves the parent reached the fork.
  : >"$LAUNCHLOG"
  echo "launching browser-actuation build (detached); tail -f $LAUNCHLOG" >&2
  _BR_DETACHED=1 setsid nohup bash "$0" "$@" >>"$LAUNCHLOG" 2>&1 < /dev/null &
  disown
  echo "  detached child launched (parent pid $$ exiting); check $LAUNCHLOG" >&2
  exit 0
fi

# --- detached child: heartbeat first, so a failed start is diagnosable ---
echo "[detached] build started, pid=$$ session=$(ps -o sid= -p $$ 2>/dev/null | tr -d ' ')"
echo "[detached] $(date '+%Y-%m-%d %H:%M:%S')  cwd=$(pwd)  py=$PY"
echo "[detached] spec=$SPEC  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
echo "[detached] handing off to: python -m harness do ... --mode master"

"$PY" -m harness do "Extend the EXISTING computer-use worker (already on this branch under agy_orchestrator/computer_use/ and harness/) with AUTONOMOUS BROWSER/DOM ACTUATION, exactly as specified in the injected design addendum (--spec COMPUTER_USE_BROWSER_DESIGN.md). Do NOT rebuild the base worker; build ON TOP of it. The base worker's ISOLATED/OBSERVE/REAL behavior, the ownership+baseline model, the SafetyKernel real_act gate, the GUI ask path, and ALL existing FRs/tests MUST stay byte-for-byte intact (INVARIANT F / B4).

GOAL: make the full perceive->reason->act loop drive a real browser end-to-end. A WORKING prototype already exists at scripts/agent_browser_demo.py (verified 2026-05-29: agent-owned Playwright chromium-1217 on :0, search a query, paginate, click the Nth result by DOM, follow the new tab). Reuse its proven mechanism. Implement these as HARD, TESTED, release-blocking requirements:

(1) BrowserController (new module agy_orchestrator/computer_use/browser.py): a dependency-injected, AGENT-OWNED, CDP-controlled Chromium session, one per run, lazily created on first navigate. Launch via Playwright with executable_path = cached ~/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome (fallback: scan chromium-*), args --no-sandbox --disable-dev-shm-usage --disable-gpu --disable-blink-features=AutomationControlled --remote-debugging-port=0, on the run's chosen display (:0 for REAL, isolated Xvfb otherwise). Register the launched browser with ProcessSupervisor as an owned child (start_new_session, rlimits, killpg tree-reap) so is_owned(browser_pid) is True. Expose the CDP endpoint so the existing BrowserDOMCollector perception path reads live DOM and DOM actuation rides the same connection. Provide FakeBrowserController (in-memory, scripted pages/results, ZERO subprocess/Playwright/:0) as the mandatory CI double. NO hermetic test may launch a real browser.

(2) ActionExecutor (additive; isolated xdotool + real_act gate paths unchanged): wire \`navigate {url}\` to BrowserController.navigate (replace the action_executor.py:279 \`true\` stub) returning landed URL + owned browser pid in spawned_process_ids; add target kind {kind: dom, selector, index (1-based)} routing click/type to BrowserController.click_dom/type_dom with new-tab follow (capture context.pages[-1]); a dom target with no open browser -> REJECTED browser_not_open (fail-closed). Add the dom target kind + a one-line navigate note to reasoner.py so the model can plan search->click-Nth autonomously.

(3) OWNERSHIP INVARIANTS (B1/B2/B3): B1 the owned CDP browser is driven with NO GUI prompt under BOTH real_full and real_children policies (it is the agent's child). B2 the controller injects ONLY into its own browser over CDP — ZERO xdotool/CDP input against any window/pid it did not launch (foreign actuation stays gated by the existing SafetyKernel real_act path, unchanged). B3 the browser tree is reaped via ProcessSupervisor.terminate_tree on session close AND on timeout/watchdog (no leak).

(4) CLI WIRING: surface in harness/cli.py argparse and thread through harness/roles.py -> adapter: --computer-use-mode {isolated,observe,real}, --real-gui-policy {full,children}, --ask-mode {on,off}, --browser-engine {bing,duckduckgo,google} (default bing — Google/DDG bot-block automated browsers; Bing verified working), --browser-display (default :0 in REAL else isolated). Defaults preserve current behavior (mode default ISOLATED).

(5) AUTONOMOUS DOGFOOD (the 'it works' bar): a slow, env-gated e2e (AGY_BROWSER_E2E=1, @pytest.mark.slow) driving the FULL adapter loop (not the standalone script): objective 'search <query> and open result number N' -> loop navigates, reads results via DOM perception, emits click {kind: dom, index: N}, lands on the Nth result's destination; assert B1/B2 hold; HONEST failure (never faked) if the engine bot-blocks.

(6) Provide the full hermetic pytest suite (FakeBrowserController + Fakes only): navigate opens/uses owned browser + returns landed url + owned pid; DOM click/type route to controller + new-tab follow returns child page url + 1-based index + missing-browser -> browser_not_open; B1 owned-browser actuation allowed no-prompt under full+children; B2 zero foreign injection; B3 teardown/timeout reaps the browser tree; B4 whole existing suite stays green; CLI flags parse + thread to RunRequest. Mark new release-blocking tests @pytest.mark.release_blocking and a @pytest.mark.browser group; gate the real-browser e2e behind @pytest.mark.slow + AGY_BROWSER_E2E=1. Do NOT launch a real browser or touch :0 during the build or hermetic tests. Do NOT commit — the operator verifies and commits." \
  --mode master \
  --spec "$SPEC" \
  --generator grok,codex,claude,agy \
  --critic codex,grok,claude,agy \
  --test-cmd "$PY -m pytest -q"

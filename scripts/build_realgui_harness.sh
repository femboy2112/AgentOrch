#!/usr/bin/env bash
# Phase: guarded build of the REAL-GUI SECURITY HARNESS extension to the
# computer-use worker. Stacks on branch feat/computer-use-worker (the base
# worker from PR #27 already lives there).
#
# Feeds COMPUTER_USE_REALGUI_DESIGN.md (the authoritative addendum) via --spec,
# then runs master mode (plan -> ToT -> adversarial), test-gated.
#
# Worker chain rationale (2026-05-29): a concurrent master build into
# /home/leah/agi2 (PID ~125508) is live on gen=codex / critic=agy (happy path).
# To avoid fan-out contention we lead with grok-gen / codex-critic (proven clean
# alongside agi2 in the prior build, cross-family review). The operator's
# freshly pre-warmed agy is slotted as the 3rd fallback on BOTH roles -> it only
# engages if grok AND codex both wall (unlikely), so the pre-warm is a safety net
# without contending with agi2's agy-critic on the happy path. claude is
# dead-last (account-sharing rule: it shares the driving session's pool).
#
# Builds INTO the AgentOrch repo on the current branch -> default cwd, no --out-dir.
#
# Carries worker --dangerously flags -> the classifier blocks the driving agent
# from spawning it, so the OPERATOR fires this via `!`.

set -uo pipefail
cd /home/leah/AgentOrch

PY=/home/leah/AgentOrch/.venv/bin/python
SPEC=COMPUTER_USE_REALGUI_DESIGN.md
LAUNCHLOG=/tmp/realgui_build_launch.log

# Self-daemonize so the build survives the `!` wrapper exiting.
if [[ "${_RG_DETACHED:-}" != "1" ]]; then
  echo "launching real-GUI harness build (detached); tail -f $LAUNCHLOG" >&2
  _RG_DETACHED=1 setsid nohup bash "$0" "$@" >"$LAUNCHLOG" 2>&1 < /dev/null &
  disown
  exit 0
fi

"$PY" -m harness do "Extend the EXISTING computer-use worker (already implemented on this branch under agy_orchestrator/computer_use/ and harness/) with a REAL-GUI SECURITY HARNESS, exactly as specified in the injected design addendum (--spec COMPUTER_USE_REALGUI_DESIGN.md). Do NOT rebuild the base worker; build ON TOP of it, preserving all of its behavior. The base worker already enforces XAUTHORITY isolation, killable tree, hard rlimits, and OBSERVE redaction; keep those intact.

MISSION-CRITICAL host-safety: this adds a deliberately gated ability to actuate the operator's REAL :0 desktop, and it must NEVER let the agent interfere with the operator's other running terminals/claude/codex/grok/agy/orchestrator instances or destabilize the host. The protection mechanism is the ownership+baseline model in the addendum: every PID alive at session start is FOREIGN for the run and default-denied.

Implement per the addendum, with these as HARD, TESTED, release-blocking invariants:
(1) BASELINE PROTECTION (FR-28, FR-40, INVARIANT A): capture all PIDs + windows (_NET_WM_PID) at session start; any pre-existing PID is FOREIGN for the run's lifetime. Add the mission-critical regression test FR-40: a simulated operator 'other terminal' PID placed in the baseline is DENIED under real_full policy with ask-mode on and no grant.
(2) OWNERSHIP GATE (FR-27/29/30/31, INVARIANT B/E): real_act actions resolve target->owning PID; OWNED (ProcessSupervisor.is_owned AND not in baseline) -> allow with no prompt; FOREIGN -> deny by default. real_act ActionSpec is constructible ONLY after the SafetyKernel issues a matching clearance_token; ActionExecutor refuses any real_act spec without a valid token. Unresolvable target -> DENY.
(3) TWO POLICIES (D1, INVARIANT D): real_full (foreign gated by the GUI prompt) and real_children (foreign NEVER input-injectable, even with a grant). Both selectable per run.
(4) GUI ASK PATH (FR-33/34/35/37/39): a zenity --forms pop-up on the real :0 presenting the question + a --add-combo of the four real-time grant scopes (Deny / Allow once / Allow this process (run) / Allow this process (10 min)) + a free-text --add-entry. The prompter MUST be dependency-injected so a FakePrompter is used in CI (NO real zenity, NO real :0, NO real foreign-window input in tests). Operator free-text routes back as an operator_note. Prompt timeout/cancel/missing-zenity -> DENY (fail-closed). The prompter injects NOTHING into foreign windows (own dialog only).
(5) GRANT CACHE (FR-36): scopes ACTION (single-use) / PROCESS_RUN / PROCESS_TTL (injected clock, no wall-clock in tests); a new foreign PID always re-prompts.
(6) DEFAULTS (INVARIANT C/F): mode default stays ISOLATED; REAL is opt-in; ask-mode defaults on in REAL; ask-mode off -> foreign hard-denied; ISOLATED/OBSERVE behavior and all base FRs/tests remain byte-for-byte intact.

Wire CLI/adapter flags (--computer-use-mode {isolated,observe,real}, --real-gui-policy {full,children}, --ask-mode {on,off}) through harness/roles.py + harness/cli.py. Stream all permission decisions (baseline.captured, permission.prompt_shown, permission.granted, permission.denied, foreign_interaction_blocked, operator_note.received) to runs/<id>/events.jsonl.

Provide the full hermetic pytest suite from the addendum's Testing Strategy (release-blocking FRs first: FR-40, FR-30, FR-31, FR-32, FR-33+34, FR-36 grant scopes, FR-35 fail-closed, FR-39, INVARIANT E, plus the INVARIANT F regression that the whole existing suite stays green). Use FakePrompter + injected clock + synthetic baselines; mark new release-blocking tests @pytest.mark.release_blocking and a @pytest.mark.realgui group; gate any real-zenity/:0 e2e behind @pytest.mark.slow + AGY_REALGUI_E2E=1. Do not run any GUI action against the real :0 during the build or tests. Do NOT commit — the operator verifies and commits." \
  --mode master \
  --spec "$SPEC" \
  --generator grok,codex,agy,claude \
  --critic codex,grok,agy,claude \
  --test-cmd "$PY -m pytest -q"

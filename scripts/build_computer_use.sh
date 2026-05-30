#!/usr/bin/env bash
# Phase 3: guarded "mission-critical" build of the computer-use agent worker.
#
# Feeds the FloodSpec-approved design (runs/20260529-030053-838/spec.md) as the
# authoritative design via --spec, then runs master mode (plan -> ToT ->
# adversarial). Worker chain grok->codex->claude (operator choice 2026-05-29) to
# dodge the concurrent agi2 build's codex+agy contention: grok leads (agi2 only
# uses grok as 3rd fallback), agy is dropped, claude is deep-fallback only. The
# critic leads with codex so the happy path is grok-generates / codex-critiques
# (independent cross-family review, not grok grading itself). Test-gated.
#
# Builds INTO the AgentOrch repo (this is an AgentOrch feature) -> default cwd,
# no --out-dir.
#
# Carries worker --dangerously flags -> the classifier blocks the driving agent
# from spawning it, so the OPERATOR fires this via `!`.

set -uo pipefail
cd /home/leah/AgentOrch

PY=/home/leah/AgentOrch/.venv/bin/python
SPEC=runs/20260529-030053-838/spec.md
LAUNCHLOG=/tmp/computeruse_build_launch.log

# Self-daemonize: detach from the controlling shell so the build survives the
# `!` wrapper exiting (the prior `&`-only attempt got SIGHUP'd before it even
# created a run dir). setsid + nohup + redirect; returns immediately.
if [[ "${_CU_DETACHED:-}" != "1" ]]; then
  echo "launching computer-use build (detached); tail -f $LAUNCHLOG" >&2
  _CU_DETACHED=1 setsid nohup bash "$0" "$@" >"$LAUNCHLOG" 2>&1 < /dev/null &
  disown
  exit 0
fi

"$PY" -m harness do "Implement the computer-use agent worker exactly as specified in the injected design (--spec). It integrates as a standard AgentOrch worker. MISSION-CRITICAL host-safety: this must never interfere with the operator's other running terminals/claude/codex/grok/agy/orchestrator instances and must never destabilize the host. Beyond the spec, enforce these four hardening requirements as hard, tested invariants:
(1) XAUTHORITY ISOLATION: the action executor (xdotool) and every spawned GUI app must use a PRIVATE Xauthority cookie scoped to the isolated Xvfb display ONLY; they must NOT inherit or be able to read the user's default ~/.Xauthority. The 'cannot authenticate to real :0' guarantee must hold even though the worker runs as the same UID as the :0 session. Add a test that asserts the executor environment has no path to the real-session X cookie.
(2) KILLABLE TREE: every owned subprocess (Xvfb, launched apps, reasoning CLI) must be started in its own process group/session (start_new_session=True / setsid) so terminate_tree reaps orphaned and daemonized children, not just the direct child. Test that a grandchild process is killed on teardown.
(3) HARD RESOURCE BACKSTOP: in addition to the psutil poll-watchdog, apply OS-level rlimits (RLIMIT_NPROC, RLIMIT_AS) to the owned process tree as a hard cap a fast fork/alloc cannot outrun. Test that an over-fork attempt is capped.
(4) OBSERVE REDACTION (default ON): in OBSERVE mode, all real-:0-scope text (window titles, OCR, AT-SPI text, terminal contents) must pass a secret-redaction pass (token/key/password/secret patterns and env-var-looking KEY=VALUE strings scrubbed) BEFORE it is placed in any prompt sent to the claude/codex reasoning CLI. Provide a per-run opt-out flag, but the default is redact-on. Test that a planted secret never appears in the reasoner prompt payload.
Wire it as a standard worker with is_available() graceful degradation and stream lifecycle events to runs/<id>/events.jsonl. Provide the full pytest suite the spec's Testing Strategy describes (release-blocking FRs first: FR-03, FR-04, FR-09, FR-12, FR-23, FR-24, plus the four hardening tests above). Do not run any GUI action against the real :0 during the build or tests; all actuation tests use the isolated Xvfb display only." \
  --mode master \
  --spec "$SPEC" \
  --generator grok,codex,claude \
  --critic codex,grok,claude \
  --test-cmd "$PY -m pytest -q"

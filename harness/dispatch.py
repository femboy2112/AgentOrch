"""Dispatch core: turn one instruction into a logged, diffed worker run.

A dispatch:
  1. builds a code-writing prompt around the instruction,
  2. snapshots the repo,
  3. runs the chosen workflow (direct / adversarial / master) with the
     fallback-wrapped role agents,
  4. snapshots again and diffs,
  5. writes everything to runs/<timestamp>/ and returns a structured result.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import inspect
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from agy_orchestrator.core.agent import WATCHDOG_MARKER, AgentInstance
from agy_orchestrator.core.calibration import append_live_row
from agy_orchestrator.execution.graph_plan import (
    ChainPlan,
    GraphPlan,
    Plan,
    validate_graph,
)
from agy_orchestrator.execution.ledger import build_ledger
from agy_orchestrator.execution.verifier import QualityVerifier, VerifierResult
from agy_orchestrator.workflows.adversarial import (
    CATASTROPHIC_FOCUS_PREAMBLE,
    AdversarialReview,
)
from agy_orchestrator.workflows.cascade import CascadeWorkflow
from agy_orchestrator.workflows.graph_merge import DEFAULT_MERGE_POLICY
from agy_orchestrator.workflows.master import MasterWorkflow
from agy_orchestrator.workflows.pat import PatWorkflow
from agy_orchestrator.workflows.reconcile import ReconciliationReview
from agy_orchestrator.workflows.test_feedback import TestFeedbackWorkflow
from agy_orchestrator.workflows.vote import VoteWorkflow
from dashboard.event_bus import EventBus
from harness import roles
from harness.run_monitor import Notifier, RunMonitor, RunStalled
from harness.snapshot import diff_snapshots, take_snapshot
from harness.telegram import (
    TelegramClient,
    TelegramNotifier,
    load_persisted_verbosity,
    load_whitelist,
    resolve_verbosity,
    whitelist_chat_ids,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
# Stable, instruction-keyed master/pat checkpoints (issue #31, salvage-on-death).
# A long master run that dies abruptly mid-step (e.g. systemd scope exit-144 from
# a worker's stray pkill) leaves completed-step edits on disk; this directory lets
# a re-dispatch of the SAME instruction resume from the last completed step instead
# of restarting. Under runs/ (gitignored), separate from per-run runs/<id>/ logs so
# the path is stable across run_ids.
CHECKPOINT_DIR = RUNS_DIR / ".checkpoints"

logger = logging.getLogger("harness")
EVENT_BUS = EventBus()


def _master_checkpoint_path(prompt: str) -> str:
    """Deterministic checkpoint file for a master/pat run, keyed by the full prompt.

    Same instruction (same built prompt) -> same path, so an abruptly-killed run
    is resumable simply by re-dispatching it (issue #31, fix option 4).
    MasterWorkflow refuses to resume a checkpoint whose key doesn't match the
    prompt, and refuses once completed==len(tasks) — so a *completed* or a
    *different* instruction starts clean. The checkpoint is removed on successful
    completion (see MasterWorkflow.execute), so this directory holds only the
    in-flight / salvageable runs.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return str(CHECKPOINT_DIR / f"{key}.json")

def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Encoding-safe, atomic artifact write (long-run resilience).

    Two defects this guards against on a long / mission-critical run whose only
    durable record is the per-run artifact set:

    * Surrogate / non-encodable chars in worker output (assembled from
      surrogateescape bytes, or a model emitting a stray codepoint) would make a
      plain ``write_text(..., encoding="utf-8")`` raise ``UnicodeEncodeError``.
      The final artifact-write tail is OUTSIDE the dispatch try/except, so one bad
      byte aborts the whole tail and a SUCCESSFUL multi-hour run ends with NO
      meta.json/diff at all. We open with ``errors="backslashreplace"`` so a bad
      char degrades gracefully (round-trippable escape) and the write always
      completes.
    * A plain ``write_text`` is a single open(truncate)+write; a mid-write kill
      (SIGKILL/OOM/scope exit-144/ENOSPC) leaves a truncated, unparseable file in
      place. We write to a sibling temp file (same dir, so ``os.replace`` is a
      same-filesystem atomic rename) and publish with ``os.replace`` — the reader
      sees the target valid-or-old, never half-written. Mirrors the
      tmp+os.replace pattern master.py already uses for its checkpoint.
    """
    tmp = path.with_name(path.name + ".tmp")
    # errors="backslashreplace": never raise on a non-encodable char; round-trippable.
    with open(tmp, "w", encoding=encoding, errors="backslashreplace") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())  # durably on disk before the atomic publish
    os.replace(tmp, path)  # atomic same-dir rename: a mid-write kill can't corrupt


def _glob_to_regex(glob: str) -> "re.Pattern[str]":
    """Compile a path glob to an anchored regex (issue #38).

    Git-style semantics on POSIX-relative paths: ``*`` matches within a single
    path segment, ``**`` spans directories, ``**/`` matches zero or more leading
    directories (so ``**/*.lock`` matches ``a/b/c.lock`` AND a root ``c.lock``),
    ``?`` matches one non-slash char. Everything else is literal.
    """
    g = glob.strip()
    if g.startswith("./"):
        g = g[2:]
    out: List[str] = []
    i, n = 0, len(g)
    while i < n:
        if g.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif g.startswith("**", i):
            out.append(".*")
            i += 2
        elif g[i] == "*":
            out.append("[^/]*")
            i += 1
        elif g[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(g[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def evaluate_path_policy(
    paths: List[str],
    protect_globs: Optional[List[str]] = None,
    allow_globs: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Return path-policy violations for a change set (issue #38).

    A path violates the policy if it matches any ``protect`` (denylist) glob, or
    if any ``allow`` (allowlist) globs are given and it matches NONE of them.
    Returns ``[{"path", "reason"}, ...]`` (empty == clean). Matching is on
    repo-relative POSIX paths.
    """
    protect_pats = [(_glob_to_regex(g), g) for g in (protect_globs or []) if g.strip()]
    allow_pats = [_glob_to_regex(g) for g in (allow_globs or []) if g.strip()]
    violations: List[Dict[str, str]] = []
    for p in paths:
        # Normalize to POSIX separators so globs match regardless of the host's
        # os.sep (snapshot paths use os.sep; treat '\' as a separator too).
        norm = p.replace(os.sep, "/").replace("\\", "/")
        hit = next((g for pat, g in protect_pats if pat.match(norm)), None)
        if hit is not None:
            violations.append({"path": p, "reason": f"matches protected path glob '{hit}'"})
            continue
        if allow_pats and not any(pat.match(norm) for pat in allow_pats):
            violations.append({"path": p, "reason": "outside every --allow-paths glob"})
    return violations


WORKER_PREAMBLE = (
    "You are a coding worker operating inside an existing project repository at the "
    "current working directory. Implement the instruction below by creating and "
    "modifying files DIRECTLY on disk in this directory. Make the changes yourself — "
    "do not ask questions and do not merely describe what to do. Keep the change "
    "minimal and tightly scoped to the instruction, and match the existing code "
    "style. Do NOT run `sudo`. "
    # Process discipline (issue #31): a worker that runs the project's full,
    # long-running test/build gate ITSELF contends with the harness's own
    # --test-cmd verifier on lock-guarded parallel gates, accumulates "stale"
    # jobs, and then "helpfully" pkills them by name — a pattern that matches
    # and kills the orchestrator process running the worker (scope exit-144).
    # The harness owns verification; the worker must neither re-run the heavy
    # gate nor pkill by name.
    "A separate automated harness runs the project's full test/build gate to verify "
    "your work after your turn — do NOT run the full or long-running test suite or "
    "build gate yourself (e.g. `make check`, the complete CI suite); run only the "
    "narrow, fast checks needed to validate your specific change. NEVER kill a "
    "process you did not directly start, and never use `pkill`, `killall`, or `kill` "
    "by process name or pattern — such a pattern can match and take down the "
    "orchestrator that is running you; if a process you spawned hangs, kill only its "
    "exact PID. "
    "When finished, end your reply with a short list of the "
    "files you created or modified and a one-line reason for each."
)


@dataclass
class DispatchResult:
    run_id: str
    run_dir: str
    mode: str
    generator: str
    critic: Optional[str]
    success: bool
    duration_s: float
    changed_files: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    error: Optional[str] = None
    # Quality-cost ledger (task #9): confidence label + signals for this run.
    quality: Optional[dict] = None
    # Per-dispatch token telemetry rollup from usage events.
    tokens: Optional[dict] = None
    # Path-policy guard (#38): change-set entries that violated --protect-paths /
    # --allow-paths. Non-empty forces success=False so a downstream consumer can
    # gate on meta.json instead of a human reading the diff.
    protect_violations: List[Dict[str, str]] = field(default_factory=list)
    # Run-watchdog outcome (#40): set to "stalled" when --run-stall-abort fired
    # (no run-level forward progress within the window). None for a normal run.
    run_outcome: Optional[str] = None
    # Resolved per-agent (model, effort) after CLI override + profile resolution
    # (#42): ``{"generator": {provider: {model, effort}}, "critic": {...},
    # "watchdog_scale": float, "watchdog_max_bytes": int|None,
    # "watchdog_stall": float|None}``. The last two are the issue #83 absolute,
    # independent budget overrides actually used (None = calibrated+scaled stood).
    # Persisted so "what did this run actually use" is answerable from meta.json,
    # not just the live event stream.
    resolved_config: Optional[Dict[str, Any]] = None
    # Reconciliation / Integration-Skeptic verdict (#43), when --reconcile ran:
    # the distinct goal-vs-runtime status (reconciled bool + per-mechanism
    # findings). NEVER merged into ``success``/the verifier under the default
    # "warn" disposition; only an explicit "fail" disposition flips success.
    reconciliation: Optional[Dict[str, Any]] = None
    # Why the reconcile station did or didn't run (#44). Always set (e.g.
    # "skipped:not_enabled" when --reconcile wasn't passed), so a null
    # ``reconciliation`` in meta.json is never ambiguous: "ran", "skipped:<why>",
    # or "error:<ExcType>".
    reconcile_status: Optional[str] = None
    # Graph-execution summary (docs §5 M4/M5), present only when a graph DAG ran:
    # ``{"merge_policy": "...", "merges": [{node_id, layer, overlapping_paths,
    # conflict, resolution}, ...]}``. getattr-guarded off the workflow so a flat
    # plan / non-master mode leaves this None and meta.json is byte-identical.
    graph: Optional[Dict[str, Any]] = None
    # Final-verifier observability (#55), present only when a --test-cmd verifier
    # ran AND reached a converged result: ``{"verifier_duration_ms",
    # "verifier_timeout_margin", "verifier_timeout_margin_pct",
    # "verifier_margin_low", "verifier_oversubscribed"}``. None (and dropped from
    # meta.json) when no verifier ran, so a non-gated run stays byte-identical.
    verifier: Optional[Dict[str, Any]] = None
    # Plan provenance (#56): the sha256 of the plan that this run actually used,
    # so "what plan did run X execute?" is answerable after the fact and a
    # hand-edit between a ``--plan-only`` emit and a ``--plan`` feed-back is
    # detectable. Two shapes, both getattr/drop-guarded so a planner-driven run
    # (no injected/emitted plan) leaves meta.json byte-identical:
    #   * ``--plan-only`` emit  -> ``{"source": "emitted", "sha256": <hex>}``
    #     (the hash of the freshly written runs/<id>/plan.json)
    #   * ``--plan <file>`` feed -> ``{"source": <path>, "sha256": <hex>,
    #     "matches_emitted": True|False|None}`` where matches_emitted compares the
    #     loaded file's hash against an operator-supplied ``--plan-expect-sha``
    #     pin (None = no pin given, so no comparison was made).
    plan_provenance: Optional[Dict[str, Any]] = None
    # --git-pr summary (docs/git-pr-mode-design.md), present only for a --git-pr
    # run: ``{base_branch, temp_branch, status, pr_url, pr_number, draft, verified,
    # commits, decision, contributing_runs}``. None (and dropped from meta.json) for
    # a normal run, so non-git-pr meta.json stays byte-identical.
    git_pr: Optional[Dict[str, Any]] = None


def _decide_reconcile_status(
    reconcile_enabled: bool,
    plan_only: bool,
    has_output: bool,
    disposition: str,
    verifier_green: bool,
) -> str:
    """Decide whether to run the reconcile station and, if not, why (#44).

    Returns "run" to run the station, else a "skipped:<reason>" string. The hard
    verifier gate is kept ONLY for the "fail" disposition: for warn/open-task the
    station is read-only and we most want the dead-wiring trace exactly when the
    build is shaky, so we run regardless of verifier_green.
    """
    if not reconcile_enabled:
        return "skipped:not_enabled"
    if plan_only:
        return "skipped:plan_only"
    if not has_output:
        return "skipped:no_output"
    if disposition == "fail" and not verifier_green:
        return "skipped:verifier_not_green"
    return "run"


def _resolve_reconcile_disposition(
    reconcile_disposition: Optional[str],
    mission_critical: bool,
) -> str:
    """Resolve the effective reconcile disposition (#71).

    ``reconcile_disposition`` is None when the operator did NOT pass
    --reconcile-disposition (argparse default), else the literal flag value.
    --mission-critical semantically means "this must actually work", so an
    UNSPECIFIED disposition under --mission-critical now defaults to "fail"
    (a confirmed dead-wiring finding blocks the run) instead of "warn".

    Cases:
      1. None + mission_critical=True  -> "fail"
      2. None + mission_critical=False -> "warn" (unchanged)
      3. explicit "warn" + mission_critical=True -> honor "warn", but warn loudly
         that reconcile findings will NOT gate the run.
      4. explicit "fail"/"open-task" (any mission_critical) -> use unchanged.
    """
    if reconcile_disposition is None:
        return "fail" if mission_critical else "warn"
    if reconcile_disposition == "warn" and mission_critical:
        logger.warning(
            "--reconcile-disposition warn under --mission-critical: reconcile "
            "findings (including CONFIRMED dead wiring) will NOT gate the run or "
            "flip the exit code. Pass --reconcile-disposition fail to hard-gate."
        )
    return reconcile_disposition


# Generous run-level stall default armed under --mission-critical (#72, Layer 2).
# This is a COARSE backstop, not a tight bound — a healthy multi-step build can
# legitimately go minutes between run-level forward-progress events (a long
# worker call, a slow verifier). 1800s (30 min) is well above any healthy gap we
# have measured, so default-arming it can never trip a healthy run; it only
# engages when a run is wedged with NO run-level progress for half an hour. The
# fine-grained transport-degradation detection + provider cycling lives at the
# worker layer (Layer 1); this run-level net only ABORTS.
MISSION_CRITICAL_RUN_STALL_DEFAULT = 1800.0


def _resolve_run_stall_abort(
    run_stall_abort: Optional[float],
    mission_critical: bool,
) -> Optional[float]:
    """Resolve the effective run-level stall-abort window (#72, Layer 2).

    ``run_stall_abort`` is None when the operator did NOT pass --run-stall-abort
    (argparse default), else the literal flag value (a float, possibly 0.0 to
    disable). --mission-critical means "this must actually work unattended", so
    an UNSPECIFIED window under --mission-critical now DEFAULT-ARMS a generous
    run-level stall backstop (``MISSION_CRITICAL_RUN_STALL_DEFAULT``, overridable
    via ``AGY_MISSION_CRITICAL_RUN_STALL``; 0 disables) instead of leaving the
    run-level net dark.

    An EXPLICIT --run-stall-abort always WINS — including an explicit 0/off,
    which honors the operator's choice to disable the run-level net. Without
    --mission-critical the value passes through unchanged (None stays None), so
    behaviour is byte-identical to the pre-#72 path.

    Cases:
      1. None + mission_critical=True  -> default (env-overridable; 0 disables -> None)
      2. None + mission_critical=False -> None (unchanged)
      3. explicit value (any mission_critical) -> passthrough verbatim (0 = off)
    """
    if run_stall_abort is not None:
        return run_stall_abort
    if not mission_critical:
        return None
    try:
        default = float(
            os.environ.get(
                "AGY_MISSION_CRITICAL_RUN_STALL",
                str(MISSION_CRITICAL_RUN_STALL_DEFAULT),
            )
            or 0
        )
    except ValueError:
        default = MISSION_CRITICAL_RUN_STALL_DEFAULT
    if default <= 0:
        return None  # env-disabled
    logger.info(
        "--mission-critical with no explicit --run-stall-abort: default-arming the "
        "run-level stall backstop at %gs (no run-level forward progress for that "
        "long aborts the run). Pass --run-stall-abort 0 to disable, or a value to "
        "override. Provider cycling is handled at the worker layer.",
        default,
    )
    return default


def _git_pr_body(session, *, mode: str, diff, quality, run_id: str) -> str:
    """Render the draft-PR body: a compact run summary + commit list. The commit
    list is capped (PR bodies have size limits — the Telegram 4096 lesson) and the
    remainder is noted rather than silently dropped."""
    conf = (quality or {}).get("confidence", "?")
    n_add = len(getattr(diff, "added", []) or [])
    n_mod = len(getattr(diff, "modified", []) or [])
    n_del = len(getattr(diff, "deleted", []) or [])
    lines = [
        f"Automated build by **AgentOrch** (`--git-pr`), run `{run_id}`.",
        "",
        f"- mode: `{mode}`",
        f"- confidence: **{conf}**",
        f"- commits: {len(session.commits)}",
        f"- files: +{n_add} ~{n_mod} -{n_del}",
        "",
        "### Commits",
    ]
    _CAP = 50
    for c in session.commits[:_CAP]:
        sha = str(c.get("sha", ""))[:9]
        step = f"step {c['step']}: " if c.get("step") else ""
        lines.append(f"- `{sha}` {step}{c.get('title', '')} [{c.get('outcome', '')}]")
    if len(session.commits) > _CAP:
        lines.append(f"- … and {len(session.commits) - _CAP} more")
    lines += ["", f"Artifacts: `runs/{run_id}/`", "",
              "🤖 Opened by AgentOrch git-pr mode."]
    return "\n".join(lines)


def _drop_git_pr_worktree(target_repo: Optional[Path], worktree: Optional[Path]) -> None:
    """Tear down the temp worktree (the BRANCH + commits persist in the repo).
    Best-effort — never raises."""
    if target_repo is None or worktree is None:
        return
    from harness import gitpr
    gitpr.remove_worktree(target_repo, worktree)
    try:  # drop the now-empty mkdtemp parent
        os.rmdir(worktree.parent)
    except OSError:
        pass


def _finalize_git_pr(*, session, worktree: Optional[Path], target_repo: Optional[Path],
                     run_dir: Path, run_id: str, instruction: str,
                     final_verified: bool, success: bool, mode: str, diff, quality):
    """Commit any leftover work, push the temp branch, and open a draft PR to the
    base branch (promoted to ready when the run verified). Tears down the worktree
    (branch + PR persist for the operator's later merge/corrective decision).

    Every git/gh failure downgrades the session ``status`` and is logged — it
    never crashes the dispatch. Returns the (mutated) session. Status ladder:
      no_changes | branch_ready (local, no remote/push) | pushed_no_pr (gh absent
      / PR failed) | awaiting_decision (PR open) | error (commit failed).
    """
    from harness import gitpr

    # 1. Final-sweep commit of anything not already captured by per-step commits
    #    (single-step modes, or a non-accepted step's leftover output).
    try:
        sweep_sha = gitpr.commit(
            worktree, f"agentorch {run_id}: {instruction.strip()[:72]}",
        )
        if sweep_sha:
            session.commits.append({
                "sha": sweep_sha,
                "outcome": "verified" if final_verified else "unverified",
                "title": instruction.strip()[:72],
            })
    except gitpr.GitError as exc:
        session.status = "error"
        session.verified = final_verified
        gitpr.save_session(run_dir, session)
        logger.warning("git-pr: final commit failed (%s); RETAINING worktree %s "
                       "for inspection", exc, worktree)
        return session  # leave the worktree so the worker's writes aren't lost

    session.verified = final_verified

    # 2. Nothing committed at all -> the branch has no content beyond base; no PR.
    if not session.commits:
        session.status = "no_changes"
        gitpr.save_session(run_dir, session)
        logger.info("git-pr: no changes committed; branch %s left at base, no PR",
                    session.temp_branch)
        _drop_git_pr_worktree(target_repo, worktree)
        return session

    # 3. Push (needs a remote). Degrade to a local-only branch otherwise.
    pushed = False
    if gitpr.has_remote(target_repo):
        try:
            gitpr.push(target_repo, session.temp_branch)
            pushed = True
        except gitpr.GitError as exc:
            logger.warning("git-pr: push failed (%s); branch %s retained locally",
                           exc, session.temp_branch)
    else:
        logger.info("git-pr: no git remote; branch %s retained locally (no PR). "
                    "Add a remote + `gh pr create` to publish.", session.temp_branch)

    if not pushed:
        session.status = "branch_ready"
        gitpr.save_session(run_dir, session)
        _drop_git_pr_worktree(target_repo, worktree)
        return session

    # 4a. A corrective --continue run reuses the SAME branch, so the push above
    #     already updated the existing PR — don't open a second one; just
    #     (re)promote it when the corrective run verified.
    if session.pr_number is not None or session.pr_url:
        if success and final_verified and session.draft and session.pr_number is not None:
            try:
                gitpr.mark_ready(target_repo, session.pr_number)
                session.draft = False
            except gitpr.GitError as exc:
                logger.warning("git-pr: pr ready failed (%s); left as draft", exc)
        session.status = "awaiting_decision"
        gitpr.save_session(run_dir, session)
        logger.info("git-pr: updated existing PR %s (%s, %s)", session.pr_url,
                    session.temp_branch, "ready" if not session.draft else "draft")
        _drop_git_pr_worktree(target_repo, worktree)
        return session

    # 4b. Open a DRAFT PR when gh is usable; promote to ready when the run verified.
    manual = (f"gh pr create --base {session.base_branch} "
              f"--head {session.temp_branch} --draft")
    if gitpr.gh_available() and gitpr.gh_authed(target_repo):
        try:
            title = f"[agentorch] {instruction.strip()[:64]}"
            body = _git_pr_body(session, mode=mode, diff=diff, quality=quality,
                                run_id=run_id)
            info = gitpr.create_pr(target_repo, base=session.base_branch,
                                   head=session.temp_branch, title=title,
                                   body=body, draft=True)
            session.pr_url = info.url
            session.pr_number = info.number
            session.draft = True
            if success and final_verified and info.number is not None:
                try:
                    gitpr.mark_ready(target_repo, info.number)
                    session.draft = False
                except gitpr.GitError as exc:
                    logger.warning("git-pr: pr ready failed (%s); left as draft", exc)
            session.status = "awaiting_decision"
            logger.info("git-pr: opened %s PR %s (%s -> %s)",
                        "draft" if session.draft else "ready",
                        info.url, session.temp_branch, session.base_branch)
        except gitpr.GitError as exc:
            session.status = "pushed_no_pr"
            logger.warning("git-pr: PR creation failed (%s); branch pushed. "
                           "Open it manually: %s", exc, manual)
    else:
        session.status = "pushed_no_pr"
        logger.info("git-pr: gh CLI unavailable/unauthed; branch %s pushed. "
                    "Open a PR with: %s", session.temp_branch, manual)

    gitpr.save_session(run_dir, session)
    _drop_git_pr_worktree(target_repo, worktree)
    return session


def plan_file_sha256(path: Union[str, Path]) -> str:
    """Return the sha256 (hex) of a plan file's raw bytes (#56).

    Hashes the exact on-disk bytes the operator reviewed/edited — NOT a
    re-serialized parse — so any byte-level hand-edit between a ``--plan-only``
    emit and a ``--plan`` feed-back changes the digest. Raises ``ValueError``
    with an operator-facing message if the file is missing (matching load_plan's
    failure surface).
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"no such plan file: {path}")
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read plan file ({path}): {exc}")
    return hashlib.sha256(raw).hexdigest()


def load_plan(path: Union[str, Path]) -> Plan:
    """Load + validate an operator-supplied plan file (flat OR graph shape).

    Closes the plan round-trip: ``--plan-only`` emits ``runs/<id>/plan.json``,
    the operator reviews/edits it, then feeds it back via ``--plan <file>`` and
    master executes it verbatim (no re-planning).

    Auto-detects the shape, in order (never silently picks a precedence):
      * a bare JSON list of step strings ``["...", ...]``     -> ``ChainPlan``
      * an object with ``"nodes"``                            -> ``GraphPlan``
        (a dependency DAG; validated immediately — dup/dangling/cycle/empty/
        non-string-task all raise here so the operator sees errors before any
        worker writes)
      * an object with ``"steps"`` and **no** ``"nodes"``     -> ``ChainPlan``
      * neither, or BOTH ``"steps"`` and ``"nodes"``          -> ``ValueError``

    Raises ``ValueError`` with an operator-facing message on any malformed input
    (missing file, bad JSON, wrong shape, empty plan, or a non-string/empty step).
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"no such plan file: {path}")
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read plan file ({path}): {exc}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"plan file is not valid JSON ({path}): {exc}")

    if isinstance(data, list):
        return ChainPlan(steps=_clean_chain_steps(data, path))

    if isinstance(data, dict):
        has_nodes = "nodes" in data
        has_steps = "steps" in data
        if has_nodes and has_steps:
            raise ValueError(
                f"plan file has BOTH 'nodes' and 'steps'; supply exactly one "
                f"shape (a graph 'nodes' DAG or a flat 'steps' list) ({path})"
            )
        if has_nodes:
            try:
                nodes = validate_graph(data["nodes"])
            except ValueError as exc:
                # Surface the specific graph-validation reason with the file path.
                raise ValueError(f"invalid graph plan ({path}): {exc}")
            return GraphPlan(nodes=nodes)
        # No "nodes": treat as a flat plan. A missing/empty "steps" raises the
        # legacy "no steps to execute" message (back-compat with the prior
        # load_plan_steps behaviour on a shape-less object).
        return ChainPlan(steps=_clean_chain_steps(data.get("steps"), path))

    raise ValueError(
        f"plan file must be a JSON list of step strings, or an object with a "
        f"'steps' list or a graph 'nodes' list ({path})"
    )


def _clean_chain_steps(steps: Any, path: Union[str, Path]) -> List[str]:
    """Validate a flat step list (shared by the bare-list and 'steps' shapes)."""
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"plan file has no steps to execute ({path})")
    cleaned: List[str] = []
    for idx, step in enumerate(steps, 1):
        if not isinstance(step, str) or not step.strip():
            raise ValueError(
                f"plan step {idx} must be a non-empty string ({path})"
            )
        cleaned.append(step)
    return cleaned


def load_plan_steps(path: Union[str, Path]) -> List[str]:
    """Backward-compat shim: load a plan file and return its flat linearization.

    Keeps the historical ``List[str]`` signature for ``cli.py`` and the existing
    plan-injection tests. A graph plan linearizes via ``as_steps()`` (stable
    topological order), so the existing linear master loop runs it verbatim.
    """
    return load_plan(path).as_steps()


def _as_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        iv = int(value)
    except Exception:
        return None
    return iv if iv >= 0 else None


def _cache_read_ratio(cache_read: Optional[int], input_tokens: Optional[int]) -> Optional[float]:
    """Prefix-cache hit rate = cache_read / (cache_read + fresh input).

    ``input_tokens`` is treated as cache-EXCLUSIVE (the fresh, reprocessed input),
    matching how the adapters split read-from-cache vs newly-processed tokens.
    Returns None when either side is unknown (codex frequently reports only
    total_tokens, leaving input None) so a missing denominator can't masquerade as
    a 0% hit rate.
    """
    if cache_read is None or input_tokens is None:
        return None
    denom = int(cache_read) + int(input_tokens)
    if denom <= 0:
        return None
    return round(int(cache_read) / denom, 4)


def _summarize_token_usage(events_path: Path) -> dict:
    per_worker: Dict[str, Dict[str, Any]] = {}
    total_calls = 0
    for raw in events_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if not isinstance(event, dict) or event.get("kind") != "usage":
            continue
        data = event.get("data")
        if not isinstance(data, dict) or data.get("usage_kind") != "call":
            continue
        worker = str(event.get("worker") or data.get("worker") or "unknown")
        model = str(event.get("model") or data.get("model") or "n/a")
        token_source = str(data.get("token_source") or "unavailable")
        token_source = token_source if token_source in {"cli", "unavailable"} else "unavailable"
        row = per_worker.setdefault(
            worker,
            {
                "calls": 0,
                "cli_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "total_tokens": 0,
                "has_input": False,
                "has_output": False,
                "has_cache": False,
                "has_total": False,
                "models": set(),
            },
        )
        total_calls += 1
        row["calls"] += 1
        if token_source == "cli":
            row["cli_calls"] += 1
        row["models"].add(model)
        input_tokens = _as_int(data.get("input_tokens"))
        output_tokens = _as_int(data.get("output_tokens"))
        cache_read_tokens = _as_int(data.get("cache_read_tokens"))
        total_tokens = _as_int(data.get("total_tokens"))
        if input_tokens is not None:
            row["input_tokens"] += input_tokens
            row["has_input"] = True
        if output_tokens is not None:
            row["output_tokens"] += output_tokens
            row["has_output"] = True
        if cache_read_tokens is not None:
            row["cache_read_tokens"] += cache_read_tokens
            row["has_cache"] = True
        if total_tokens is not None:
            row["total_tokens"] += total_tokens
            row["has_total"] = True

    out_per_worker: Dict[str, Dict[str, Any]] = {}
    total_input = 0
    total_output = 0
    total_cache = 0
    total_total = 0
    has_total_input = False
    has_total_output = False
    has_total_cache = False
    has_total_total = False
    for worker in sorted(per_worker):
        row = per_worker[worker]
        calls = int(row["calls"])
        cli_calls = int(row["cli_calls"])
        token_source = "unavailable"
        if cli_calls == calls and calls > 0:
            token_source = "cli"
        elif 0 < cli_calls < calls:
            token_source = "mixed"
        input_tokens = row["input_tokens"] if row["has_input"] else None
        output_tokens = row["output_tokens"] if row["has_output"] else None
        cache_read_tokens = row["cache_read_tokens"] if row["has_cache"] else None
        total_tokens = row["total_tokens"] if row["has_total"] else None
        if input_tokens is not None:
            total_input += int(input_tokens)
            has_total_input = True
        if output_tokens is not None:
            total_output += int(output_tokens)
            has_total_output = True
        if cache_read_tokens is not None:
            total_cache += int(cache_read_tokens)
            has_total_cache = True
        if total_tokens is not None:
            total_total += int(total_tokens)
            has_total_total = True
        out_per_worker[worker] = {
            "calls": calls,
            "token_source": token_source,
            "models": sorted(row["models"]),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "total_tokens": total_tokens,
            # Prefix-cache hit rate over processed input (cache_read / fresh+cached
            # input). Surfaced per-run so any caching optimization is verifiable
            # from meta.json instead of being invisible. None when the denominator
            # is unknown (e.g. codex reports only total_tokens, input=None).
            "cache_read_ratio": _cache_read_ratio(cache_read_tokens, input_tokens),
        }

    return {
        "total_calls": total_calls,
        "per_worker": out_per_worker,
        "grand_total": {
            "input_tokens": total_input if has_total_input else None,
            "output_tokens": total_output if has_total_output else None,
            "cache_read_tokens": total_cache if has_total_cache else None,
            "total_tokens": total_total if has_total_total else None,
            "cache_read_ratio": _cache_read_ratio(
                total_cache if has_total_cache else None,
                total_input if has_total_input else None,
            ),
        },
    }


def _build_prompt(
    instruction: str, context: Optional[str], spec: Optional[str] = None,
    *, include_preamble: bool = True,
) -> str:
    # WORKER_PREAMBLE is process-discipline for the WORKER (don't sudo/pkill, don't
    # re-run the full gate). The critic only judges output against the requirement,
    # so its "Original Requirement" view drops the preamble (win 4) — same goal +
    # spec + context, ~350-420 fewer tokens reprocessed every critic iteration.
    parts = [WORKER_PREAMBLE] if include_preamble else []
    parts.append("\n## Instruction\n" + instruction.strip())
    if spec:
        # An approved FloodSpec design doc: the authoritative source of truth.
        # Placed before free-form context so it dominates the worker's framing —
        # and, for master mode, so the planner decomposes THIS design rather than
        # re-inventing one from the bare instruction.
        parts.append(
            "\n## Approved design specification (authoritative)\n"
            "Build the system described by this specification. It has been reviewed "
            "and approved; treat it as the source of truth for architecture, "
            "interfaces, data models, and constraints. Do not redesign it — "
            "implement it.\n\n" + spec.strip()
        )
    if context:
        parts.append("\n## Additional context\n" + context.strip())
    return "\n".join(parts)


def _worker_hint(agent: AgentInstance, fallback_chain: Optional[List[str]]) -> str:
    if fallback_chain:
        return fallback_chain[0]
    if hasattr(agent, "_worker_name"):
        return agent._worker_name()
    return "agy"


def _derive_verifier_delta(
    baseline_result: Optional[VerifierResult], final_verified: bool
) -> Optional[str]:
    """Classify quality movement between pre-run baseline and final verifier signal."""
    if baseline_result is None:
        return None
    baseline_ok = baseline_result.ok
    if baseline_ok and final_verified:
        return "preserved"
    if baseline_ok and not final_verified:
        return "regressed"
    if not baseline_ok and final_verified:
        return "fixed"
    return "unchanged"


# A final verifier whose wall-clock left less than this fraction of its timeout
# budget unused is flagged verifier_margin_low (#55a): the suite is close to the
# wall-clock kill, so a slightly slower host / heavier change would TIME OUT (an
# infra failure, not a code defect). 20% per the issue.
_VERIFIER_MARGIN_FLOOR_PCT = 0.20

# A final verifier that ran this many times SLOWER than the same suite's pre-run
# baseline on the unchanged tree is flagged verifier_oversubscribed (#55c): the
# code didn't get ~2x slower, the host did — almost always nested thread-pool /
# xdist oversubscription. Conservative (a real 2x regression is rare) so the flag
# stays meaningful.
_VERIFIER_OVERSUBSCRIBE_FACTOR = 2.0


def _verifier_telemetry(
    final_result: Optional[VerifierResult],
    *,
    timeout_s: Optional[float],
    baseline_result: Optional[VerifierResult],
    run_id: str,
) -> Dict[str, Any]:
    """Final-verifier observability for meta.json + the friendly view (#55).

    Returns {} when no verifier reached a converged result (so a non-gated run's
    telemetry/meta.json is byte-identical). Otherwise records the FINAL verify's
    wall-clock, the remaining timeout margin (and a low-margin warning), and a
    suspected-oversubscription flag derived from the same-suite baseline."""
    if final_result is None:
        return {}
    out: Dict[str, Any] = {
        "verifier_duration_ms": final_result.duration_ms,
        "verifier_timeout_margin": None,
        "verifier_timeout_margin_pct": None,
        "verifier_margin_low": None,
        "verifier_oversubscribed": None,
    }
    # (a) timeout margin = timeout_budget - duration; warn when < 20% of budget.
    if timeout_s and timeout_s > 0:
        timeout_ms = timeout_s * 1000.0
        margin_ms = round(timeout_ms - final_result.duration_ms)
        margin_pct = margin_ms / timeout_ms
        out["verifier_timeout_margin"] = margin_ms
        out["verifier_timeout_margin_pct"] = round(margin_pct, 3)
        out["verifier_margin_low"] = margin_pct < _VERIFIER_MARGIN_FLOOR_PCT
        if out["verifier_margin_low"]:
            logger.warning(
                "Dispatch %s: verifier left only %.0f%% of its %.0fs timeout budget "
                "unused (%dms of %dms) — a slightly slower host would TIME OUT. "
                "Raise AGY_TEST_TIMEOUT or lighten the suite.",
                run_id, margin_pct * 100, timeout_s,
                final_result.duration_ms, round(timeout_ms),
            )
    # (c) oversubscription heuristic: FINAL duration >> baseline on the same suite.
    if (baseline_result is not None
            and not baseline_result.timeout
            and baseline_result.duration_ms > 0
            and not final_result.timeout):
        ratio = final_result.duration_ms / baseline_result.duration_ms
        oversub = ratio >= _VERIFIER_OVERSUBSCRIBE_FACTOR
        out["verifier_oversubscribed"] = oversub
        if oversub:
            logger.warning(
                "Dispatch %s: final verifier ran %.1fx slower than the same-suite "
                "baseline (%dms vs %dms) — suspected oversubscription (nested "
                "thread-pool / xdist). Check -n and BLAS thread pins.",
                run_id, ratio, final_result.duration_ms, baseline_result.duration_ms,
            )
    return out


def _build_role_agent_compat(
    chain: List[str],
    *,
    prompt: str,
    fallback: bool,
    cycles: int,
    codex_config: Optional[List[str]],
    computer_use_config: Optional[Dict[str, Any]],
    post_construct_hook: Optional[roles.RolePostConstructHook],
    overrides: Optional[Dict[str, Dict[str, str]]] = None,
    watchdog_scale: float = 1.0,
    watchdog_max_bytes: Optional[int] = None,
    watchdog_stall: Optional[float] = None,
) -> AgentInstance:
    """Call roles.build_role_agent while staying compatible with patched tests.

    Some unit tests monkeypatch ``roles.build_role_agent`` with a legacy
    signature that predates ``computer_use_config`` / ``overrides`` /
    ``watchdog_scale`` / ``watchdog_max_bytes`` / ``watchdog_stall`` (issue #83).
    Pass each of those kwargs only when the callee can accept it (named param or
    ``**kwargs``).
    """
    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "fallback": fallback,
        "cycles": cycles,
        "codex_config": codex_config,
        "post_construct_hook": post_construct_hook,
    }
    try:
        sig = inspect.signature(roles.build_role_agent)
        params = sig.parameters
        has_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        params, has_var_kw = {}, False

    def _accepts(name: str) -> bool:
        return has_var_kw or name in params

    if _accepts("computer_use_config"):
        kwargs["computer_use_config"] = computer_use_config
    if overrides and _accepts("overrides"):
        kwargs["overrides"] = overrides
    if watchdog_scale != 1.0 and _accepts("watchdog_scale"):
        kwargs["watchdog_scale"] = watchdog_scale
    if watchdog_max_bytes is not None and _accepts("watchdog_max_bytes"):
        kwargs["watchdog_max_bytes"] = watchdog_max_bytes
    if watchdog_stall is not None and _accepts("watchdog_stall"):
        kwargs["watchdog_stall"] = watchdog_stall
    return roles.build_role_agent(chain, **kwargs)


def _build_merge_reconciler_factory(
    *,
    critic_chain: List[str],
    fallback: bool,
    cycles: int,
    codex_config: Optional[List[str]],
    critic_overrides: Optional[Dict[str, Dict[str, str]]],
    watchdog_scale: float,
    watchdog_max_bytes: Optional[int] = None,
    watchdog_stall: Optional[float] = None,
    post_construct_hook: Optional[roles.RolePostConstructHook],
):
    """Build the per-node reconciler factory for the ``reconcile`` merge policy (M4).

    Returns a callable ``factory(node, working_directory) -> Reconciler`` where the
    reconciler is an async ``(rel, base, sibling, node) -> bytes`` that resolves ONE
    overlapping file by running an independent critic-chain reviewer over the three
    byte views (base / already-merged sibling / this node) — the SAME cross-provider
    reviewer principle as the #43 reconcile station, here producing a merged FILE.
    The merged tree is re-verified by the join node's verifier (docs §4), so a bad
    merge is caught by the existing gate. Non-text/binary or worker-failure cases
    fall back to the node's own bytes (last-writer-wins) rather than corrupt a file.
    """
    def factory(*, node, working_directory: str):
        async def _reconcile(
            rel: str,
            base_bytes: Optional[bytes],
            sibling_bytes: bytes,
            node_bytes: bytes,
        ) -> bytes:
            # Binary files can't be three-way text-merged by a reviewer; keep the
            # node's write (the merge is then last-writer-wins, recorded as an
            # overlap in meta) rather than ask the model to invent bytes.
            try:
                base_text = (base_bytes or b"").decode("utf-8")
                sibling_text = sibling_bytes.decode("utf-8")
                node_text = node_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return node_bytes
            merge_prompt = (
                "Two parallel build steps modified the SAME file. Produce a single "
                "MERGED version that preserves BOTH steps' intent and is internally "
                "consistent. Output ONLY the full merged file content (no fences, no "
                f"commentary).\n\n--- FILE: {rel} ---\n"
                f"\n=== COMMON BASE (before either step) ===\n{base_text}\n"
                f"\n=== SIBLING STEP'S VERSION ===\n{sibling_text}\n"
                f"\n=== THIS STEP'S VERSION ===\n{node_text}\n"
            )
            try:
                agent = _build_role_agent_compat(
                    critic_chain,
                    prompt=merge_prompt,
                    fallback=fallback,
                    cycles=cycles,
                    codex_config=codex_config,
                    computer_use_config=None,
                    post_construct_hook=post_construct_hook,
                    overrides=critic_overrides,
                    watchdog_scale=watchdog_scale,
                    watchdog_max_bytes=watchdog_max_bytes,
                    watchdog_stall=watchdog_stall,
                )
                merged = await agent.run_async()
            except Exception as exc:
                logger.warning(
                    "Merge reconciler failed on %s (%s); keeping this node's bytes.",
                    rel, exc,
                )
                return node_bytes
            return (merged or "").encode("utf-8") if merged.strip() else node_bytes

        return _reconcile

    return factory


async def _run_workflow(
    mode: str,
    prompt: str,
    *,
    run_id: str,
    generator_chain: List[str],
    critic_chain: List[str],
    fallback: bool,
    cycles: int,
    max_iterations: int,
    branches: int,
    verifier: Optional[QualityVerifier],
    codex_config: Optional[List[str]],
    computer_use_config: Optional[Dict[str, Any]] = None,
    post_construct_hook: Optional[roles.RolePostConstructHook] = None,
    working_directory: str = ".",
    mission_critical: bool = False,
    baseline_result: Optional[VerifierResult] = None,
    candidate_setup: Optional[str] = None,
    resume_policy: str = "auto",
    plan_only: bool = False,
    plan_steps: Optional[List[str]] = None,
    plan_graph: Optional[GraphPlan] = None,
    max_parallel_nodes: Optional[int] = None,
    verifier_concurrency: int = 1,
    merge_policy: str = DEFAULT_MERGE_POLICY,
    gen_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    critic_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    watchdog_scale: float = 1.0,
    watchdog_max_bytes: Optional[int] = None,
    watchdog_stall: Optional[float] = None,
    max_parallel_workers: Optional[int] = None,
    critic_requirement: Optional[str] = None,
    workflow_sink: Optional[List[Any]] = None,
) -> tuple:
    """Run the workflow; return (output, workflow_or_None) so the caller can read
    the workflow's quality signals for the run ledger.

    ``gen_overrides``/``critic_overrides`` are per-provider model/effort maps
    (issue #42) applied to the generator-role and critic-role chains
    respectively; ``watchdog_scale`` widens the armed budgets for heavy tiers;
    ``max_parallel_workers`` caps concurrent candidate generation in vote.

    ``workflow_sink`` is a mutable single-element holder: the constructed
    workflow is appended to it BEFORE ``execute`` runs so the caller can recover
    it even when ``execute`` raises (e.g. a graph ``--merge-policy fail`` abort).
    Without this, an exception from ``execute`` discards the workflow ref and the
    graph/merge outcomes it accumulated never reach ``meta.json``."""
    def _register(_wf):
        # Surface the in-flight workflow to the caller before execute() runs, so
        # an abort policy's MergeConflict (recorded on the workflow) survives the
        # re-raise and lands in meta.json.
        if workflow_sink is not None:
            workflow_sink.append(_wf)
        return _wf
    # Plan-only / dry-run (#41): for master/pat, run JUST the planner and stop
    # before any worker mutates the out-dir. pat's "plan" is the master planner,
    # so both modes route through a plan-only MasterWorkflow — no verifier needed
    # (the direct Stage-1 attempt, which would write, is skipped).
    if plan_only and mode in ("master", "pat"):
        agent_class, model, effort = roles.build_master_agent_class(
            generator_chain, fallback=fallback, cycles=cycles,
            codex_config=codex_config,
            overrides=gen_overrides,
            watchdog_scale=watchdog_scale,
            watchdog_max_bytes=watchdog_max_bytes,
            watchdog_stall=watchdog_stall,
            post_construct_hook=post_construct_hook,
        )
        wf = MasterWorkflow(
            model=model,
            effort=effort,
            branches=branches,
            max_iterations=max_iterations,
            verifier=verifier,
            agent_class=agent_class,
            working_directory=working_directory,
            plan_only=True,
            plan_steps=plan_steps,
            plan_graph=plan_graph,
            event_callback=EVENT_BUS.publisher_for(
                run_id, worker="orchestrator", model=model, effort=effort, branch=None,
            ),
        )
        return await wf.execute(prompt), wf

    if mode == "direct":
        gen = _build_role_agent_compat(
            generator_chain,
            prompt=prompt,
            fallback=fallback,
            cycles=cycles,
            codex_config=codex_config,
            computer_use_config=computer_use_config,
            post_construct_hook=post_construct_hook,
            overrides=gen_overrides,
            watchdog_scale=watchdog_scale,
            watchdog_max_bytes=watchdog_max_bytes,
            watchdog_stall=watchdog_stall,
        )
        return await gen.run_async(), None

    if mode == "adversarial":
        gen = _build_role_agent_compat(
            generator_chain,
            prompt=prompt,
            fallback=fallback,
            cycles=cycles,
            codex_config=codex_config,
            computer_use_config=computer_use_config,
            post_construct_hook=post_construct_hook,
            overrides=gen_overrides,
            watchdog_scale=watchdog_scale,
            watchdog_max_bytes=watchdog_max_bytes,
            watchdog_stall=watchdog_stall,
        )
        critic = _build_role_agent_compat(
            critic_chain,
            prompt="",
            fallback=fallback,
            cycles=cycles,
            codex_config=codex_config,
            computer_use_config=computer_use_config,
            post_construct_hook=post_construct_hook,
            overrides=critic_overrides,
            watchdog_scale=watchdog_scale,
            watchdog_max_bytes=watchdog_max_bytes,
            watchdog_stall=watchdog_stall,
        )
        model = str(getattr(gen, "model", None) or "n/a")
        effort = str(getattr(gen, "effort", None) or "n/a")
        wf = AdversarialReview(gen, critic, verifier, max_iterations=max_iterations,
                               working_directory=working_directory,
                               critic_preamble=(CATASTROPHIC_FOCUS_PREAMBLE if mission_critical else ""),
                               critic_requirement=critic_requirement,
                               event_callback=EVENT_BUS.publisher_for(
                                   run_id,
                                   worker="orchestrator",
                                   model=model,
                                   effort=effort,
                                   branch=None,
                               ))
        return await wf.execute(prompt), wf

    if mode == "feedback":
        # Generator + programmatic verifier loop, no LLM critic. Requires a
        # --test-cmd (the strong oracle) to gate quality on real test results.
        if verifier is None:
            raise ValueError("feedback mode requires --test-cmd (the verifier oracle)")
        gen = _build_role_agent_compat(
            generator_chain,
            prompt=prompt,
            fallback=fallback,
            cycles=cycles,
            codex_config=codex_config,
            computer_use_config=computer_use_config,
            post_construct_hook=post_construct_hook,
            overrides=gen_overrides,
            watchdog_scale=watchdog_scale,
            watchdog_max_bytes=watchdog_max_bytes,
            watchdog_stall=watchdog_stall,
        )
        wf = TestFeedbackWorkflow(gen, verifier, max_iterations=max_iterations,
                                  working_directory=working_directory)
        return await wf.execute(prompt), wf

    if mode == "cascade":
        # Cheap-first escalation: each generator-chain token is a STAGE (cheap ->
        # strong); run the test-feedback loop per stage, escalate only on verifier
        # failure. Requires --test-cmd (the gate between stages).
        if verifier is None:
            raise ValueError("cascade mode requires --test-cmd (the escalation gate)")
        stages = [
            _build_role_agent_compat(
                [token],
                prompt=prompt,
                fallback=False,
                cycles=cycles,
                codex_config=codex_config,
                computer_use_config=computer_use_config,
                post_construct_hook=post_construct_hook,
                overrides=gen_overrides,
                watchdog_scale=watchdog_scale,
                watchdog_max_bytes=watchdog_max_bytes,
                watchdog_stall=watchdog_stall,
            )
            for token in generator_chain
        ]
        wf = CascadeWorkflow(stages, verifier, max_iterations_per_stage=max_iterations,
                             working_directory=working_directory)
        return await wf.execute(prompt), wf

    if mode == "master":
        agent_class, model, effort = roles.build_master_agent_class(
            generator_chain, fallback=fallback, cycles=cycles,
            codex_config=codex_config,
            overrides=gen_overrides,
            watchdog_scale=watchdog_scale,
            watchdog_max_bytes=watchdog_max_bytes,
            watchdog_stall=watchdog_stall,
            post_construct_hook=post_construct_hook,
        )
        # #70: build a DISTINCT critic agent class from the critic chain so the
        # in-loop Phase-B adversarial reviewer is cross-family (agy-led by default)
        # rather than codex critiquing codex. Same builder the generator uses, but
        # seeded from critic_chain + critic_overrides so --critic / CRITIC_CHAIN and
        # --critic-effort/--codex-model actually take effect in master mode.
        critic_agent_class, critic_model, critic_effort = roles.build_master_agent_class(
            critic_chain, fallback=fallback, cycles=cycles,
            codex_config=codex_config,
            overrides=critic_overrides,
            watchdog_scale=watchdog_scale,
            watchdog_max_bytes=watchdog_max_bytes,
            watchdog_stall=watchdog_stall,
            post_construct_hook=post_construct_hook,
        )
        wf = MasterWorkflow(
            model=model,
            effort=effort,
            critic_agent_class=critic_agent_class,
            critic_model=critic_model,
            critic_effort=critic_effort,
            branches=branches,
            max_iterations=max_iterations,
            verifier=verifier,
            agent_class=agent_class,
            working_directory=working_directory,
            checkpoint_path=_master_checkpoint_path(prompt),
            resume_policy=resume_policy,
            plan_steps=plan_steps,
            plan_graph=plan_graph,
            # Graph-execution concurrency caps (docs §5 M3). Linear runs ignore
            # both; a DAG plan bounds whole-node parallelism by max_parallel_nodes
            # and serializes the verifier spike at verifier_concurrency.
            max_parallel_workers=max_parallel_nodes,
            verifier_concurrency=verifier_concurrency,
            # Overlapping parallel writes (docs §4 / M4): reconcile (default) /
            # disjoint / fail. reconcile wires a critic-chain file reconciler;
            # the join node's verifier re-runs on the merged tree.
            merge_policy=merge_policy,
            reconcile_station_factory=_build_merge_reconciler_factory(
                critic_chain=critic_chain, fallback=fallback, cycles=cycles,
                codex_config=codex_config, critic_overrides=critic_overrides,
                watchdog_scale=watchdog_scale,
                watchdog_max_bytes=watchdog_max_bytes, watchdog_stall=watchdog_stall,
                post_construct_hook=post_construct_hook,
            ),
            event_callback=EVENT_BUS.publisher_for(
                run_id,
                worker="orchestrator",
                model=model,
                effort=effort,
                branch=None,
            ),
        )
        _register(wf)
        return await wf.execute(prompt), wf

    if mode == "vote":
        # K-parallel candidates in isolated workspaces; verifier picks the
        # winner. K = `branches` (reusing the existing CLI knob). Each
        # candidate rotates through the generator_chain so K=3 with the
        # default chain (codex,agy,grok) produces one candidate per
        # provider — the heterogeneity gain (arxiv 2602.03794).
        if verifier is None:
            raise ValueError("vote mode requires --test-cmd (the verifier gate)")
        k = max(1, branches)
        vote_generators: List[AgentInstance] = []
        for i in range(k):
            token = generator_chain[i % len(generator_chain)]
            # Each slot is its own single-worker agent (no fallback chain
            # inside a slot — diversity comes from different slots, not
            # from fallback within one slot).
            slot_agent = _build_role_agent_compat(
                [token],
                prompt=prompt,
                fallback=False,
                cycles=cycles,
                codex_config=codex_config,
                computer_use_config=computer_use_config,
                post_construct_hook=post_construct_hook,
                overrides=gen_overrides,
                watchdog_scale=watchdog_scale,
                watchdog_max_bytes=watchdog_max_bytes,
                watchdog_stall=watchdog_stall,
            )
            vote_generators.append(slot_agent)
        wf = VoteWorkflow(
            generators=vote_generators,
            verifier=verifier,
            working_directory=working_directory,
            # Reuse the baseline gate the harness just ran on the unchanged base
            # tree (#33) — lets the preflight skip a redundant full-suite re-run.
            baseline_ok=(baseline_result.ok if baseline_result is not None else None),
            # Per-candidate env bootstrap (#34): makes vote isolation sound on
            # editable-install repos (each candidate gets its own venv).
            candidate_setup=candidate_setup,
            # Host-safety concurrency cap (#42 item 7): bound how many candidates
            # are in flight end-to-end so --branches>1 can't thrash/freeze a
            # small-RAM box. None = all K concurrent (legacy behaviour).
            max_parallel=max_parallel_workers,
        )
        return await wf.execute(prompt), wf

    if mode == "pat":
        # Plan-after-Trial: direct generator attempt gated by verifier;
        # on failure, escalate to master mode. Verifier is mandatory.
        if verifier is None:
            raise ValueError("pat mode requires --test-cmd (the Stage 1 verifier gate)")
        direct_gen = _build_role_agent_compat(
            generator_chain,
            prompt=prompt,
            fallback=fallback,
            cycles=cycles,
            codex_config=codex_config,
            computer_use_config=computer_use_config,
            post_construct_hook=post_construct_hook,
            overrides=gen_overrides,
            watchdog_scale=watchdog_scale,
            watchdog_max_bytes=watchdog_max_bytes,
            watchdog_stall=watchdog_stall,
        )
        agent_class, model, effort = roles.build_master_agent_class(
            generator_chain, fallback=fallback, cycles=cycles,
            codex_config=codex_config,
            overrides=gen_overrides,
            watchdog_scale=watchdog_scale,
            watchdog_max_bytes=watchdog_max_bytes,
            watchdog_stall=watchdog_stall,
            post_construct_hook=post_construct_hook,
        )
        # #70: distinct critic agent for pat's escalated master (cross-family).
        critic_agent_class, critic_model, critic_effort = roles.build_master_agent_class(
            critic_chain, fallback=fallback, cycles=cycles,
            codex_config=codex_config,
            overrides=critic_overrides,
            watchdog_scale=watchdog_scale,
            watchdog_max_bytes=watchdog_max_bytes,
            watchdog_stall=watchdog_stall,
            post_construct_hook=post_construct_hook,
        )
        master_wf = MasterWorkflow(
            model=model,
            effort=effort,
            critic_agent_class=critic_agent_class,
            critic_model=critic_model,
            critic_effort=critic_effort,
            branches=branches,
            max_iterations=max_iterations,
            verifier=verifier,
            agent_class=agent_class,
            working_directory=working_directory,
            checkpoint_path=_master_checkpoint_path(prompt),
            resume_policy=resume_policy,
            plan_steps=plan_steps,
            plan_graph=plan_graph,
            max_parallel_workers=max_parallel_nodes,
            verifier_concurrency=verifier_concurrency,
            merge_policy=merge_policy,
            reconcile_station_factory=_build_merge_reconciler_factory(
                critic_chain=critic_chain, fallback=fallback, cycles=cycles,
                codex_config=codex_config, critic_overrides=critic_overrides,
                watchdog_scale=watchdog_scale,
                watchdog_max_bytes=watchdog_max_bytes, watchdog_stall=watchdog_stall,
                post_construct_hook=post_construct_hook,
            ),
            event_callback=EVENT_BUS.publisher_for(
                run_id,
                worker="orchestrator",
                model=model,
                effort=effort,
                branch=None,
            ),
        )
        wf = PatWorkflow(
            direct_generator=direct_gen,
            master_workflow=master_wf,
            verifier=verifier,
            working_directory=working_directory,
        )
        _register(wf)
        return await wf.execute(prompt), wf

    raise ValueError(f"unknown mode: {mode}")


# Substring in a recon agent's stderr that the FallbackAgent writes when its whole
# provider chain is spent (see fallback_agent.py's "All N fallback attempts
# exhausted" RuntimeError text — which, when caught + folded into stderr rather than
# re-raised, leaves this marker). A trace that ends here returned a degenerate reply,
# not a real verdict.
_FALLBACK_EXHAUSTED_MARKER = "fallback attempts exhausted"


def _detect_recon_starvation(recon_agent: AgentInstance) -> Optional[str]:
    """Return a starvation slug iff the trace agent was killed/exhausted (#59).

    A recon agent can return WITHOUT raising yet have produced only a degenerate
    reply because its watchdog tripped (``_watchdog_reason``) or its own fallback
    chain was exhausted (a ``[watchdog:...]`` marker / "fallback attempts exhausted"
    in stderr). Such a trace is HOLLOW regardless of how the parse landed: it must
    set ``reconcile_status=ran:critic_starved`` and never contribute reconciled=true.

    Returns the slug (a watchdog reason like "stalled"/"verbose", or
    "fallback_exhausted"), or None when the agent ran to a real completion.
    """
    reason = getattr(recon_agent, "_watchdog_reason", None)
    if reason:
        return str(reason)
    stderr = str(getattr(recon_agent, "stderr", "") or "")
    if WATCHDOG_MARKER in stderr:
        # A FallbackAgent copies the sub's stderr (carrying its watchdog marker) up
        # on a success-shaped return; pull the slug the same way fallback_agent does.
        head = stderr.split(WATCHDOG_MARKER, 1)[1]
        slug = head.split("]", 1)[0].strip()
        return slug or "watchdog"
    if _FALLBACK_EXHAUSTED_MARKER in stderr:
        return "fallback_exhausted"
    return None


async def _run_reconciliation(
    *,
    run_id: str,
    goal: str,
    critic_chain: List[str],
    fallback: bool,
    codex_config: Optional[List[str]],
    critic_overrides: Optional[Dict[str, Dict[str, str]]],
    watchdog_scale: float,
    watchdog_max_bytes: Optional[int] = None,
    watchdog_stall: Optional[float] = None,
    post_construct_hook: Optional[roles.RolePostConstructHook],
    working_directory: str,
    disposition: str,
    ablation_cmd: Optional[str] = None,
    ablation_cmd_map: Optional[Dict[str, str]] = None,
) -> "ReconciliationResult":  # type: ignore[name-defined]
    """Build an independent reviewer and run the reconciliation station (#43).

    The reconciler runs on the CRITIC chain — a cross-provider reviewer distinct
    from the generator that wrote the code — so the goal-vs-runtime trace is not
    self-verification (the same independence principle as the adversarial critic).
    Single trace pass; its verdict is returned for the caller to record alongside
    (never folded into) the verifier's result."""
    recon_agent = _build_role_agent_compat(
        critic_chain,
        prompt="",
        fallback=fallback,
        cycles=1,
        codex_config=codex_config,
        computer_use_config=None,
        post_construct_hook=post_construct_hook,
        overrides=critic_overrides,
        watchdog_scale=watchdog_scale,
        watchdog_max_bytes=watchdog_max_bytes,
        watchdog_stall=watchdog_stall,
    )
    station = ReconciliationReview(
        agent=recon_agent,
        goal=goal,
        working_directory=working_directory,
        disposition=disposition,
        ablation_cmd=ablation_cmd,
        ablation_cmd_map=ablation_cmd_map,
        event_callback=EVENT_BUS.publisher_for(
            run_id,
            worker="reconcile",
            model=str(getattr(recon_agent, "model", "n/a") or "n/a"),
            effort=str(getattr(recon_agent, "effort", "n/a") or "n/a"),
        ),
    )
    result = await station.execute()
    # #59: a recon agent that DIDN'T raise can still have been killed/exhausted
    # (watchdog trip or fallback-chain exhaustion) and so returned a degenerate
    # reply. Inspect its post-run telemetry; a starved trace is HOLLOW — mark it so
    # it sets reconcile_status=ran:critic_starved and can't bless dead wiring.
    starved = _detect_recon_starvation(recon_agent)
    if starved is not None:
        logger.warning(
            "Reconciliation %s: trace agent starved (%s) — verdict is hollow, "
            "not reconciled.",
            run_id, starved,
        )
        result.mark_starved(starved)
    return result


def _build_telegram_notifier(
    *,
    run_id: str,
    mode: str,
    enabled: Optional[bool],
    verbosity: Optional[str],
    instruction: Optional[str] = None,
) -> Optional["TelegramNotifier"]:
    """Construct a TelegramNotifier per the tri-state enable policy, or None.

    enabled: None=auto (ON iff key present AND whitelist non-empty), True=force
    ON (warn + return None if key/whitelist missing — non-fatal), False=off.
    Any error here is swallowed; telegram never blocks or fails a dispatch.
    """
    try:
        if enabled is False:
            return None
        client = TelegramClient()
        # Default-off path (auto, no token): avoid the whitelist file read entirely
        # when there's no token, since telegram can't activate anyway. The forced
        # path (--telegram) still falls through to warn about what's missing.
        if enabled is not True and not client.configured:
            return None
        entries = load_whitelist()
        chat_ids = whitelist_chat_ids(entries)
        if not client.configured or not chat_ids:
            if enabled is True:
                missing = []
                if not client.configured:
                    missing.append("TELEGRAM_BOT_KEY")
                if not chat_ids:
                    missing.append("whitelist")
                logger.warning(
                    "telegram: --telegram requested but %s missing — staying off "
                    "(non-fatal)",
                    " and ".join(missing),
                )
            return None
        # Only an explicit per-dispatch --telegram-verbosity flag PINS the level.
        # Otherwise follow the operator's persisted /verbosity default LIVE, so a
        # mid-build /verbosity change takes effect on this run's push
        # notifications (the prior behavior froze it at construction → the
        # /verbosity command appeared to do nothing). The constructed default
        # still resolves env/normal for the very first events before any state
        # file exists; the reader overrides it whenever a persisted level is set.
        pinned = bool(verbosity)
        dynamic = None if pinned else load_persisted_verbosity
        return TelegramNotifier(
            run_id=run_id,
            mode=mode,
            verbosity=resolve_verbosity(verbosity),
            client=client,
            chat_ids=chat_ids,
            dynamic_verbosity=dynamic,
            instruction=instruction,
        )
    except Exception as exc:  # best-effort: telegram never affects dispatch
        logger.debug("telegram notifier setup failed: %s", exc)
        return None


async def dispatch_async(
    instruction: str,
    *,
    run_id: Optional[str] = None,
    mode: str = "adversarial",
    context: Optional[str] = None,
    generator_chain: Optional[List[str]] = None,
    critic_chain: Optional[List[str]] = None,
    fallback: bool = True,
    cycles: int = 2,
    max_iterations: int = 5,
    branches: int = 3,
    test_cmd: Optional[str] = None,
    verifier_mem_max: Optional[str] = None,
    web_search: bool = False,
    mission_critical: bool = False,
    spec: Optional[str] = None,
    dashboard_stream_json: bool = False,
    out_dir: Optional[Union[str, Path]] = None,
    candidate_setup: Optional[str] = None,
    # --git-pr (docs/git-pr-mode-design.md): run on an isolated git worktree +
    # temp branch (agentorch/<run_id>), commit accepted work to it, and persist a
    # pr_session.json. Off (default) => zero git ops, byte-identical dispatch.
    git_pr: bool = False,
    # --continue <prior_run_id>: corrective resume. Re-attach a worktree to the
    # prior run's temp branch and run this instruction on top, updating the SAME
    # branch + PR + canonical session. Implies git_pr.
    git_pr_continue: Optional[str] = None,
    resume_policy: str = "auto",
    protect_paths: Optional[List[str]] = None,
    allow_paths: Optional[List[str]] = None,
    plan_only: bool = False,
    plan_steps: Optional[List[str]] = None,
    # Plan provenance (#56): when --plan injected a file, ``plan_source`` is its
    # path (recorded + hashed into meta.json) and ``plan_expect_sha`` is an
    # optional hard pin — a mismatch is refused BEFORE any worker runs. Both None
    # on a planner-driven (non-injected) run, leaving meta.json byte-identical.
    plan_source: Optional[Union[str, Path]] = None,
    plan_expect_sha: Optional[str] = None,
    # Graph execution (docs §5 M3): an operator-supplied dependency DAG +
    # concurrency caps for the master graph walker. plan_graph routes to the
    # frontier scheduler when it has non-linear deps; max_parallel_nodes bounds
    # whole-node parallelism; verifier_concurrency serializes the verifier spike.
    plan_graph: Optional[GraphPlan] = None,
    max_parallel_nodes: Optional[int] = None,
    verifier_concurrency: int = 1,
    # Overlapping-parallel-write merge policy for the graph walker (docs §4 / M4):
    # reconcile (default) | disjoint | fail. Only affects a graph plan with a true
    # branch/join; linear runs ignore it.
    merge_policy: str = DEFAULT_MERGE_POLICY,
    # Run-level watchdog / heartbeat / notify (#40)
    run_stall_abort: Optional[float] = None,
    notify: Optional[str] = None,
    notify_cmd: Optional[str] = None,
    heartbeat_interval: Optional[float] = None,
    # Telegram build-progress bot. ``telegram_enabled`` is tri-state:
    #   None  -> auto (ON iff TELEGRAM_BOT_KEY set AND whitelist non-empty),
    #   True  -> force ON (warn + stay off if key/whitelist missing; non-fatal),
    #   False -> force OFF. Verbosity falls back to AGY_TELEGRAM_VERBOSITY/normal.
    telegram_enabled: Optional[bool] = None,
    telegram_verbosity: Optional[str] = None,
    # Per-role / per-provider effort+model overrides (#42)
    gen_effort: Optional[str] = None,
    gen_model: Optional[str] = None,
    critic_effort: Optional[str] = None,
    critic_model: Optional[str] = None,
    architect_effort: Optional[str] = None,
    architect_model: Optional[str] = None,
    codex_model: Optional[str] = None,
    effort_map: Optional[str] = None,
    model_map: Optional[str] = None,
    effort_profile: Optional[str] = None,
    watchdog_scale: Optional[float] = None,
    # Issue #83 — ABSOLUTE, INDEPENDENT watchdog budget overrides that decouple
    # the verbose byte budget from the per-worker stall budget (which
    # --watchdog-scale couples). Each replaces exactly one dimension; env mirrors
    # are AGY_WATCHDOG_MAX_BYTES / AGY_WATCHDOG_STALL (read in roles._arm_watchdog).
    watchdog_max_bytes: Optional[int] = None,
    watchdog_stall: Optional[float] = None,
    max_parallel_workers: Optional[int] = None,
    worker_mem_max: Optional[str] = None,
    # Pre-run baseline verifier gate (single-run speed: skip the full --test-cmd
    # suite on the unchanged tree for non-vote modes, where it only feeds telemetry).
    # True restores the always-run baseline (and its verifier_delta telemetry).
    baseline_gate: bool = False,
    # Reconciliation / Integration-Skeptic station (#43)
    reconcile: bool = False,
    reconcile_disposition: Optional[str] = None,
    # Programmatic ablation-witness hook (#52). OPT-IN: a witness command template
    # (``<cmd> {MECH}``) the reconcile station runs READ-ONLY in a throwaway worktree
    # to MEASURE each mechanism's load-bearing delta instead of trusting the model's
    # self-report. None => byte-identical to the self-report path.
    ablation_cmd: Optional[str] = None,
    ablation_cmd_map: Optional[Dict[str, str]] = None,
    # Step 12: computer-use worker params (forwarded to adapter when generator=computer-use)
    # Step 10: real-gui harness wiring (flags only; absent keeps cu_req construction byte-identical)
    computer_use_mode: Optional[str] = None,
    computer_use_task_priority: Optional[str] = None,
    computer_use_budgets: Optional[Dict[str, Any]] = None,
    # real_gui_policy / ask_mode per instruction: bare names (match backend RunRequest/WorkerSession)
    real_gui_policy: Optional[str] = None,
    ask_mode: Optional[str] = None,
    browser_engine: Optional[str] = None,
    browser_display: Optional[str] = None,
) -> DispatchResult:
    """Execute one instruction and capture the run.

    ``spec`` is an approved FloodSpec design document (the text, not a path);
    when present it is injected as the authoritative design the worker must
    implement — for master mode the planner decomposes it instead of the bare
    instruction.

    ``out_dir`` is the directory the worker subprocess runs in (its cwd) and
    the scope of the before/after snapshot diff. Defaults to AgentOrch's own
    repo root, which is the historical behaviour. Set it when invoking
    AgentOrch from another repo so the worker writes there instead of into
    AgentOrch. The ``runs/<id>/`` artifacts always live under AgentOrch
    (they're orchestrator-internal logs, not user data).
    """
    generator_chain = generator_chain or list(roles.GENERATOR_CHAIN)
    critic_chain = critic_chain or list(roles.CRITIC_CHAIN)
    codex_config = ["tools.web_search=true"] if web_search else None
    cu_config: Dict[str, Any] = {
        "mode": computer_use_mode or "ISOLATED",
        "task_priority": computer_use_task_priority or "normal",
        "budgets": computer_use_budgets,
    }
    if real_gui_policy is not None:
        cu_config["real_gui_policy"] = real_gui_policy
    if ask_mode is not None:
        cu_config["ask_mode"] = ask_mode
    if browser_engine is not None:
        cu_config["browser_engine"] = browser_engine
    if browser_display is not None:
        cu_config["browser_display"] = browser_display
    # Step 12: recognize computer-use as standard worker token (no LLM path)
    _is_cu = bool(generator_chain) and generator_chain[0] == roles.COMPUTER_USE_TOKEN
    if mode == "auto":
        from agy_orchestrator.routing.policy import RoutingPolicy, from_dispatch_args
        task = from_dispatch_args(
            instruction=instruction,
            context=context,
            test_cmd=test_cmd,
            branches=branches,
            generator_chain=generator_chain,
        )
        decision = RoutingPolicy().choose(task)
        logger.info("Auto-routing: mode=%s — %s", decision.mode, decision.reason)
        mode = decision.mode
        if decision.branches is not None:
            branches = decision.branches
        if decision.max_iterations is not None:
            max_iterations = decision.max_iterations

    # Per-role / per-provider effort+model overrides (#42). Resolve the CLI
    # surface into per-role {provider: {model, effort}} maps now that `mode` is
    # final (auto-routing may have changed it, which affects the --architect-*
    # alias). resolve_overrides raises OverrideError (a ValueError) on a bad
    # tier/model; the CLI pre-validates so a typo never reaches here as a crash.
    from harness.effort_overrides import effective_config, resolve_overrides
    resolved = resolve_overrides(
        generator_chain=generator_chain,
        critic_chain=critic_chain,
        mode=mode,
        profile=effort_profile,
        gen_effort=gen_effort, gen_model=gen_model,
        critic_effort=critic_effort, critic_model=critic_model,
        architect_effort=architect_effort, architect_model=architect_model,
        codex_model=codex_model,
        effort_map=effort_map, model_map=model_map,
        watchdog_scale=watchdog_scale,
    )
    gen_overrides = resolved.generator
    critic_overrides = resolved.critic
    eff_watchdog_scale = resolved.watchdog_scale
    resolved_config: Dict[str, Any] = {
        "generator": effective_config(generator_chain, gen_overrides, roles.AGENT_DEFAULTS),
        "critic": effective_config(critic_chain, critic_overrides, roles.AGENT_DEFAULTS),
        "watchdog_scale": eff_watchdog_scale,
        # Issue #83 — absolute, independent overrides actually used by this run
        # (None = no override; the calibrated+scaled budget stood for that dimension).
        "watchdog_max_bytes": watchdog_max_bytes,
        "watchdog_stall": watchdog_stall,
    }
    for _note in resolved.notes:
        logger.warning("override: %s", _note)

    # Reconciliation / Integration-Skeptic station (#43). Opt-in via --reconcile
    # or AGY_RECONCILE=1; default-ON for --mission-critical (the runs where dead
    # wiring is most expensive to discover late). Disposition default depends on
    # mission-critical (#71): when --reconcile-disposition is UNSPECIFIED it is
    # "warn" normally, but "fail" under --mission-critical (a confirmed dead-wiring
    # finding must block a run the operator declared must-actually-work). An
    # explicit --reconcile-disposition is always honored; explicit "warn" under
    # --mission-critical is honored too but logs a loud non-gating warning.
    reconcile_enabled = bool(
        reconcile
        or mission_critical
        or os.environ.get("AGY_RECONCILE", "").lower() in ("1", "true", "on")
    )
    recon_disposition = _resolve_reconcile_disposition(reconcile_disposition, mission_critical)

    # Run-level stall backstop (#72, Layer 2): default-arm a generous run-level
    # stall under --mission-critical when the operator did not pass an explicit
    # --run-stall-abort, so the coarse abort net is engaged even if Layer 1's
    # worker-transport bound is somehow defeated. Explicit values (incl. 0/off)
    # always win; non-mission-critical runs are unchanged.
    run_stall_abort = _resolve_run_stall_abort(run_stall_abort, mission_critical)

    # Where the worker actually writes files. Default = AgentOrch repo root,
    # which preserves the prior behaviour exactly.
    work_dir = Path(out_dir).expanduser().resolve() if out_dir else PROJECT_ROOT
    work_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    events_path.touch()
    # Record the dispatching PID so the run tracker can deregister a stillborn /
    # early-exiting / killed run the instant its process is gone, instead of
    # showing it "in progress" forever (issue #67). Best-effort; never fatal.
    try:
        (run_dir / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        pass

    # --git-pr preflight + worktree setup (docs/git-pr-mode-design.md). Run on an
    # isolated git worktree + temp branch so the operator's checkout never moves
    # and accepted work lands on a reviewable branch (later phases push it as a
    # draft PR). Refuse FAST on an unsafe tree (non-repo / detached / dirty)
    # BEFORE any worktree is created — mirroring the #36 data-loss lesson: never
    # risk clobbering uncommitted work. When off, none of this runs and the
    # dispatch is byte-identical.
    git_pr = git_pr or bool(git_pr_continue)  # --continue implies --git-pr
    git_pr_session = None
    git_pr_target_repo: Optional[Path] = None
    git_pr_worktree: Optional[Path] = None
    # Where the canonical pr_session.json lives. Fresh runs own it under their own
    # run dir; a corrective --continue run keeps updating the ORIGINAL run's session.
    git_pr_session_dir = run_dir
    if git_pr_continue:
        from harness import gitpr
        prior_dir = RUNS_DIR / git_pr_continue
        prior = gitpr.load_session(prior_dir)
        if prior is None:
            raise ValueError(
                f"--continue {git_pr_continue}: no git-pr session found at "
                f"{prior_dir} (was that run dispatched with --git-pr?)"
            )
        target_repo = Path(prior.target_repo)
        if not gitpr.is_git_repo(target_repo):
            raise ValueError(
                f"--continue {git_pr_continue}: recorded target repo "
                f"{target_repo} is not a git repository"
            )
        if not gitpr.branch_exists(target_repo, prior.temp_branch):
            raise ValueError(
                f"--continue {git_pr_continue}: temp branch {prior.temp_branch} "
                f"no longer exists in {target_repo} (was it deleted/merged?)"
            )
        git_pr_worktree = (
            Path(tempfile.mkdtemp(prefix=f"agentorch-gitpr-{run_id}-")) / "worktree"
        )
        # Re-attach a worktree to the EXISTING temp branch — it already holds all
        # prior committed work, so the worker continues from where it left off.
        gitpr.add_worktree(
            target_repo, git_pr_worktree, branch=prior.temp_branch, create=False,
        )
        git_pr_target_repo = target_repo
        git_pr_session = prior
        git_pr_session.status = "running"
        git_pr_session.parent_run_id = git_pr_continue
        if run_id not in git_pr_session.contributing_runs:
            git_pr_session.contributing_runs.append(run_id)
        git_pr_session_dir = prior_dir  # keep updating the original's session
        gitpr.save_session(git_pr_session_dir, git_pr_session)
        work_dir = git_pr_worktree
        logger.info(
            "git-pr: continuing run %s on branch %s (corrective run %s; worktree %s)",
            git_pr_continue, prior.temp_branch, run_id, git_pr_worktree,
        )
    elif git_pr:
        from harness import gitpr
        target_repo = work_dir
        if not gitpr.is_git_repo(target_repo):
            raise ValueError(
                f"--git-pr requires a git repository at {target_repo}; "
                f"initialise one (and check out a base branch) first"
            )
        if gitpr.is_detached_head(target_repo):
            raise ValueError(
                "--git-pr requires a checked-out base branch, but HEAD is "
                "detached at {0}; `git checkout <branch>` first".format(target_repo)
            )
        if gitpr.is_dirty(target_repo):
            raise ValueError(
                f"--git-pr requires a clean working tree at {target_repo} "
                f"(commit or stash your changes first); refusing so uncommitted "
                f"or untracked work can never be clobbered"
            )
        base_branch = gitpr.current_branch(target_repo)
        base_sha = gitpr.head_sha(target_repo)
        temp_branch = gitpr.branch_name_for_run(run_id)
        git_pr_worktree = (
            Path(tempfile.mkdtemp(prefix=f"agentorch-gitpr-{run_id}-")) / "worktree"
        )
        gitpr.add_worktree(
            target_repo, git_pr_worktree, branch=temp_branch, start_point=base_sha,
        )
        git_pr_target_repo = target_repo
        git_pr_session = gitpr.PrSession(
            run_id=run_id,
            base_branch=base_branch or "",
            temp_branch=temp_branch,
            base_sha=base_sha or "",
            work_dir=str(git_pr_worktree),
            target_repo=str(target_repo),
            contributing_runs=[run_id],
        )
        gitpr.save_session(git_pr_session_dir, git_pr_session)
        # Repoint the whole run at the worktree: the snapshot scope, worker cwd,
        # verifier, and run monitor all key off work_dir, so this single
        # reassignment routes the entire dispatch into the isolated branch
        # checkout — the operator's own working tree is never touched.
        work_dir = git_pr_worktree
        logger.info(
            "git-pr: base branch=%s -> temp branch=%s (isolated worktree %s)",
            base_branch, temp_branch, git_pr_worktree,
        )

    # Plan provenance (#56): when --plan injected a file, hash its raw bytes and
    # record where it came from so the executed plan is auditable after the fact.
    # An optional --plan-expect-sha pin is a HARD gate for unattended dispatch:
    # a mismatch is refused here, BEFORE any worker writes. ``matches_emitted``
    # reflects that comparison (None when no pin was supplied).
    plan_provenance: Optional[Dict[str, Any]] = None
    if plan_source is not None:
        loaded_sha = plan_file_sha256(plan_source)
        matches_emitted: Optional[bool] = None
        if plan_expect_sha is not None:
            matches_emitted = loaded_sha == plan_expect_sha.strip().lower()
            if not matches_emitted:
                raise ValueError(
                    f"plan sha256 mismatch: --plan {plan_source} hashes to "
                    f"{loaded_sha} but --plan-expect-sha pinned "
                    f"{plan_expect_sha.strip().lower()}; refusing to run "
                    f"(the plan was edited since it was reviewed)"
                )
        plan_provenance = {
            "source": str(plan_source),
            "sha256": loaded_sha,
            "matches_emitted": matches_emitted,
        }

    # The run monitor (#40) is constructed below but referenced from _sink so it
    # can observe run-level forward progress on every event. Use a 1-slot holder
    # to keep the closure simple without reordering the surrounding setup.
    _monitor_holder: List[RunMonitor] = []

    def _sink(event: dict) -> None:
        if _monitor_holder:
            try:
                _monitor_holder[0].observe(event)
            except Exception:
                pass
        with events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    prompt = _build_prompt(instruction, context, spec)
    # Preamble-stripped view for the adversarial critic's "Original Requirement"
    # (win 4): same goal/spec/context, minus the worker process-discipline boilerplate.
    critic_requirement = _build_prompt(instruction, context, spec, include_preamble=False)
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    gen_desc = roles.describe_chain(
        generator_chain,
        fallback,
        real_gui_policy=real_gui_policy,
        ask_mode=ask_mode,
        browser_engine=browser_engine,
        browser_display=browser_display,
    )
    crit_desc = (
        roles.describe_chain(
            critic_chain,
            fallback,
            real_gui_policy=real_gui_policy,
            ask_mode=ask_mode,
            browser_engine=browser_engine,
            browser_display=browser_display,
        )
        if mode == "adversarial"
        else None
    )

    # Cross-family verifier guard: warn (don't block) when the critic chain
    # leads with the same provider family as the generator. Meaningful for every
    # mode that runs a separate critic chain: adversarial, and — since #70 wired
    # critic_chain into the in-loop reviewer — master/pat as well. (pat escalates
    # to master, which now builds a distinct critic agent from critic_chain.)
    if mode in ("adversarial", "master", "pat"):
        family_warning = roles.check_chains_cross_family(generator_chain, critic_chain)
        if family_warning:
            logger.warning(family_warning)
    if mode in ("vote", "tot"):
        agy_warning = roles.check_agy_parallelism_warning(mode, generator_chain, branches)
        if agy_warning:
            logger.warning(agy_warning)

    def _post_construct_hook(agent: AgentInstance, worker: str, cfg: Dict[str, object]) -> None:
        # Prefer the agent's RESOLVED identity so the per-worker spin-up shows the
        # model the CLI actually runs (codex maps its "standard" alias ->
        # gpt-5.3-codex-spark, "max" effort -> xhigh); fall back to the cfg/raw
        # values for agents that don't alias.
        try:
            resolved_model = agent.effective_model()
        except Exception:
            resolved_model = None
        model = str(resolved_model or cfg.get("model") or getattr(agent, "model", None) or "n/a")
        try:
            resolved_effort = agent.effective_effort()
        except Exception:
            resolved_effort = None
        effort_val = resolved_effort if resolved_effort not in (None, "n/a") else cfg.get("effort")
        effort = str(effort_val if effort_val not in (None, "n/a") else getattr(agent, "effort", None) or "n/a")
        agent.event_callback = EVENT_BUS.publisher_for(
            run_id,
            worker=worker,
            model=model,
            effort=effort,
            branch=getattr(agent, "branch", None),
        )
        # Dashboard-only stream JSON; normal CLI path remains --output-format json.
        if worker == "claude" and hasattr(agent, "dashboard_stream_json"):
            setattr(agent, "dashboard_stream_json", bool(dashboard_stream_json))
        # Pin the worker's cwd to the operator-chosen output directory so it
        # doesn't pollute AgentOrch when invoked from elsewhere. Only set when
        # work_dir is not the repo root; staying ``None`` keeps the historical
        # inherit-parent-cwd behaviour for the default case.
        if work_dir != PROJECT_ROOT:
            agent.cwd = str(work_dir)
        # Forward the full computer-use config for shim-based execution paths.
        # This keeps direct adapter short-circuiting unchanged while ensuring
        # --mode workflows that construct ComputerUseShim still receive all
        # CLI-forwarded RunRequest keys.
        if worker == roles.COMPUTER_USE_TOKEN and hasattr(agent, "computer_use_config"):
            setattr(agent, "computer_use_config", dict(cu_config))
            setattr(agent, "_harness_run_id", run_id)
            setattr(agent, "_harness_events_path", str(events_path))

    EVENT_BUS.add_sink(run_id, _sink)

    # --git-pr per-accepted-step commits (Phase 2): for a LINEAR master/pat run,
    # commit the worktree each time a step finishes verified/approved, so the temp
    # branch grows one reviewable commit per accepted step. Master already emits
    # the step-completed orchestration event through the bus, so this needs no
    # workflow changes — it just listens. Graph (DAG) runs write via post-merge
    # into the live tree at a DIFFERENT boundary (a node's step-completed fires
    # before its writes are merged), so they are deliberately NOT committed
    # per-step here; their merged result is captured by the single finalize commit
    # below. (Per-node graph commits are a documented follow-up — design §8.)
    if git_pr and git_pr_session is not None and plan_graph is None:
        from harness import gitpr as _gitpr_step

        def _git_pr_step_commit(event: dict) -> None:
            try:
                data = event.get("data") or {}
                if data.get("event") != "orchestration_transition":
                    return
                orch = data.get("orchestration") or {}
                if orch.get("phase") != "step" or orch.get("action") != "completed":
                    return
                outcome = orch.get("outcome")
                if outcome not in ("verified", "approved"):
                    return
                idx = orch.get("step_index")
                total = orch.get("step_total")
                title = (orch.get("step_title") or "").strip()
                msg = f"step {idx}/{total}: {title} [{outcome}]".strip()
                sha = _gitpr_step.commit(git_pr_worktree, msg)
                if sha:
                    git_pr_session.commits.append(
                        {"step": idx, "sha": sha, "outcome": outcome, "title": title}
                    )
                    _gitpr_step.save_session(git_pr_session_dir, git_pr_session)
            except Exception as exc:  # a commit hiccup must never break the stream
                logger.debug("git-pr step commit skipped: %s", exc)

        EVENT_BUS.add_sink(run_id, _git_pr_step_commit)

    # Telegram build-progress sink (best-effort; fully exception-isolated). Any
    # telegram error here is swallowed (debug log) and never affects dispatch.
    telegram_notifier = _build_telegram_notifier(
        run_id=run_id,
        mode=mode,
        enabled=telegram_enabled,
        verbosity=telegram_verbosity,
        instruction=instruction,
    )
    telegram_poller = None
    if telegram_notifier is not None:
        EVENT_BUS.add_sink(run_id, telegram_notifier)
        # Serve inbound commands (/status, /track, …) WHILE this build runs, so the
        # bot isn't notify-only when no standalone daemon is up (issue #63). Cross-
        # process singleton-guarded (flock): a no-op if a daemon/sibling already
        # polls. Fully best-effort — never blocks or fails the dispatch.
        try:
            from harness.telegram_bot import EmbeddedCommandPoller

            poller = EmbeddedCommandPoller()
            if poller.start():
                telegram_poller = poller
        except Exception as exc:  # telegram must never affect dispatch
            logger.debug("embedded telegram command poller not started: %s", exc)

    dispatch_pub = EVENT_BUS.publisher_for(
        run_id,
        worker=generator_chain[0],
        model="n/a",
        effort="n/a",
    )
    dispatch_pub({
        "kind": "lifecycle",
        "data": {
            "event": "dispatch_started",
            "detail": {
                "mode": mode,
                "generator_chain": generator_chain,
                "critic_chain": critic_chain,
            },
        },
    })

    # Run-level watchdog + heartbeat + notify (#40). Heartbeats publish through
    # the bus (so they persist to events.jsonl AND reach the dashboard) on a
    # dedicated publisher; the monitor observes every event via _sink to track
    # run-level forward progress (chatter doesn't count) and the current step.
    if heartbeat_interval is None:
        heartbeat_interval = float(os.environ.get("AGY_HEARTBEAT_SECONDS", "30") or 0)
    notify_url: Optional[str] = None
    notify_command: Optional[str] = notify_cmd
    if notify:
        if notify.startswith(("http://", "https://")):
            notify_url = notify
        else:
            notify_command = notify_command or notify
    notifier = Notifier(webhook=notify_url, command=notify_command, run_id=run_id)
    heartbeat_pub = EVENT_BUS.publisher_for(
        run_id, worker="run-monitor", model="n/a", effort="n/a",
    )
    monitor = RunMonitor(
        run_id=run_id,
        emit=heartbeat_pub,
        notifier=notifier,
        work_dir=str(work_dir),
        heartbeat_interval=heartbeat_interval or 0.0,
        run_stall_abort=run_stall_abort,
    )
    _monitor_holder.append(monitor)
    monitor.notify("start", extra={"mode": mode})

    # Route all tracking logs into this run's stderr.log while still showing them.
    file_handler = logging.FileHandler(run_dir / "stderr.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    logger.info("Dispatch %s | mode=%s | generator=%s%s",
                run_id, mode, gen_desc, f" | critic={crit_desc}" if crit_desc else "")

    if test_cmd:
        # Pass mem_max only when a cap is actually requested, so the common path
        # (and test doubles that swap in a narrower QualityVerifier) keep their
        # original constructor signature. ``--verifier-mem-max`` (#39) caps the
        # gate directly; ``--worker-mem-max`` (#42 item 7) caps each PARALLEL
        # candidate's verifier in vote/tot — the per-candidate local spike that
        # can OOM/freeze a small-RAM box when --branches>1. The explicit
        # verifier cap wins if both are given.
        _mem_cap = verifier_mem_max or (worker_mem_max if mode in ("vote", "tot") else None)
        _vkwargs = {"test_commands": [test_cmd]}
        if _mem_cap:
            _vkwargs["mem_max"] = _mem_cap
        verifier = QualityVerifier(**_vkwargs)
        # #60: give the verifier a place to persist a per-iteration failure log
        # and a bus to emit failing-test nodeids on, so a thrashing step is
        # diagnosable post-hoc and visible in events.jsonl. Both opt-in attrs;
        # unset they leave the verifier byte-identical.
        verifier.run_dir = str(run_dir)
        verifier.event_callback = EVENT_BUS.publisher_for(
            run_id, worker="verifier", model="n/a", effort="n/a",
        )
    else:
        verifier = None
    baseline_result: Optional[VerifierResult] = None
    # Single-run speed: the pre-run baseline runs the FULL --test-cmd suite on the
    # unchanged tree before any worker starts. Only vote mode consumes its result
    # for a real decision (its red-on-both / skip-redundant-rerun preflight, line
    # ~630). For every other mode it feeds telemetry ONLY — verifier_delta
    # (_derive_verifier_delta returns None when baseline is None), three meta
    # fields, and an OOM-vs-fail notification label — all of which already handle a
    # None baseline (it is set None on exception today). So skip the baseline for
    # non-vote modes unless the operator opts back in via --baseline-gate.
    _run_baseline = verifier is not None and (mode == "vote" or baseline_gate)
    if _run_baseline:
        try:
            baseline_result = await verifier.verify(working_directory=str(work_dir))
            if baseline_result.ok:
                logger.info("Baseline verifier: ok=True")
            else:
                logger.info(
                    "Baseline verifier: ok=False (returncode=%s, error_hash=%s)",
                    baseline_result.returncode,
                    baseline_result.error_hash,
                )
        except Exception as exc:
            logger.warning(
                "Baseline verifier failed (continuing): %s: %s",
                type(exc).__name__,
                exc,
            )
            baseline_result = None

    before = take_snapshot(work_dir)
    started = time.monotonic()
    success = True
    error: Optional[str] = None
    output = ""
    workflow = None
    # Mutable holder the workflow registers itself into BEFORE execute() runs, so
    # a workflow that raised (e.g. a graph --merge-policy fail/disjoint abort) is
    # still recoverable here — otherwise its merge_outcomes never reach meta.json.
    workflow_sink: List[Any] = []
    run_outcome: Optional[str] = None
    reconciliation_result = None  # #43: set after a converged build if --reconcile
    reconcile_status = None  # #44: why reconcile did/didn't run; persisted to meta.json
    try:
        if _is_cu:
            # Step 12 minimal glue: exercise the real adapter (writes events.jsonl under runs/<id>/,
            # respects --out-dir via harness snapshot scope, supports cu params).
            # Only direct-style runs are exercised here; complex workflows stay LLM-only.
            from agy_orchestrator.computer_use.adapter import ComputerUseWorkerAdapter
            from agy_orchestrator.computer_use.audit import AuditEventSink as CUAuditSink

            def _cu_sink(rid: str):
                # Append to the harness-prepared events.jsonl and fan out to bus for dashboard.
                s = CUAuditSink(run_id=rid, events_path=events_path)
                pub = EVENT_BUS.publisher_for(rid, worker="computer-use", model="n/a", effort="n/a")
                s.add_callback(lambda d: pub({"kind": "computer_use_event", "data": d}) or None)
                return s

            adapter = ComputerUseWorkerAdapter(audit_sink_factory=_cu_sink)
            cu_req = {
                "run_id": run_id,
                "objective": instruction,  # raw objective for the reasoner; prompt wrapper is in artifacts
                **cu_config,
            }
            h = adapter.start(cu_req)
            output = f"computer-use:{h.status} run_id={h.run_id} events={h.events_path or ''}"
            # workflow stays None (no quality ledger from cu yet)
        else:
            # Supervise the workflow with the run-level watchdog + heartbeat
            # (#40). When neither is active monitor.run awaits the coroutine
            # directly, so the default path is byte-for-byte unchanged.
            output, workflow = await monitor.run(_run_workflow(
                mode, prompt,
                run_id=run_id,
                generator_chain=generator_chain, critic_chain=critic_chain,
                fallback=fallback, cycles=cycles, max_iterations=max_iterations,
                branches=branches, verifier=verifier, codex_config=codex_config,
                computer_use_config=cu_config,
                post_construct_hook=_post_construct_hook,
                working_directory=str(work_dir),
                mission_critical=mission_critical,
                baseline_result=baseline_result,
                candidate_setup=candidate_setup,
                resume_policy=resume_policy,
                plan_only=plan_only,
                plan_steps=plan_steps,
                plan_graph=plan_graph,
                max_parallel_nodes=max_parallel_nodes,
                verifier_concurrency=verifier_concurrency,
                merge_policy=merge_policy,
                gen_overrides=gen_overrides,
                critic_overrides=critic_overrides,
                watchdog_scale=eff_watchdog_scale,
                watchdog_max_bytes=watchdog_max_bytes,
                watchdog_stall=watchdog_stall,
                max_parallel_workers=max_parallel_workers,
                critic_requirement=critic_requirement,
                workflow_sink=workflow_sink,
            ))

            # Reconciliation / Integration-Skeptic station (#43): runs AFTER the
            # build converges (and, when a verifier gated it, only when GREEN) to
            # trace each goal-named mechanism to the live execution path. The bus
            # is still open here so its trace events stream + persist normally.
            # Wrapped best-effort: a station failure must NEVER fail a good build.
            verifier_green = verifier is None or bool(getattr(workflow, "verified", False))
            has_output = bool(output and output.strip())
            reconcile_status = _decide_reconcile_status(
                reconcile_enabled, plan_only, has_output,
                recon_disposition, verifier_green,
            )
            if reconcile_status == "run":
                try:
                    reconciliation_result = await _run_reconciliation(
                        run_id=run_id,
                        goal=(spec or instruction),
                        critic_chain=critic_chain,
                        fallback=fallback,
                        codex_config=codex_config,
                        critic_overrides=critic_overrides,
                        watchdog_scale=eff_watchdog_scale,
                        watchdog_max_bytes=watchdog_max_bytes,
                        watchdog_stall=watchdog_stall,
                        post_construct_hook=_post_construct_hook,
                        working_directory=str(work_dir),
                        disposition=recon_disposition,
                        ablation_cmd=ablation_cmd,
                        ablation_cmd_map=ablation_cmd_map,
                    )
                    # #51/#59: reconcile_status reflects the RESULT, not merely a
                    # non-exception return. A hollow trace (no JSON parsed, zero
                    # mechanisms traced, or a starved/exhausted critic chain) records
                    # a ran:* variant that callers + the friendly view surface as a
                    # distinct AMBER state rather than a green "ran".
                    reconcile_status = reconciliation_result.substance_status()
                except Exception as rexc:  # best-effort: never fail a good build
                    reconcile_status = "error:" + type(rexc).__name__
                    logger.warning("Reconciliation station errored (continuing): %s: %s",
                                   type(rexc).__name__, rexc)
            else:
                # Never silently un-reconcile (#44): name the run + the reason. A
                # fail-disposition skip on a non-green verifier is a warning (the
                # build is shaky and we're declining the hard gate); the rest are info.
                if reconcile_status == "skipped:verifier_not_green":
                    logger.warning("Dispatch %s reconcile %s (disposition=fail, verifier not green)",
                                   run_id, reconcile_status)
                else:
                    logger.info("Dispatch %s reconcile %s", run_id, reconcile_status)
    except RunStalled as exc:  # run watchdog aborted: classify, never crash
        success = False
        run_outcome = exc.reason
        error = (
            f"run aborted by watchdog: no run-level forward progress within "
            f"{run_stall_abort:g}s (stuck/stalled)"
        )
        logger.error("Dispatch %s %s", run_id, error)
    except Exception as exc:  # graceful: record, never crash the operator's shell
        success = False
        error = f"{type(exc).__name__}: {exc}"
        logger.error("Dispatch %s failed: %s", run_id, error)
    finally:
        # execute() may have raised before the (output, workflow) unpack (e.g. a
        # graph --merge-policy fail/disjoint abort): recover the in-flight
        # workflow from the sink so its merge_outcomes still reach meta.json.
        if workflow is None and workflow_sink:
            workflow = workflow_sink[-1]
        # Always populate reconcile_status for meta.json (#44), even on paths that
        # never reached the gate (computer-use, watchdog stall, or an exception):
        # if reconcile wasn't enabled say so, otherwise the build didn't reach a
        # reconcilable converged state.
        if reconcile_status is None:
            reconcile_status = ("skipped:not_enabled" if not reconcile_enabled
                                else "skipped:no_output")
        duration = time.monotonic() - started
        dispatch_pub({
            "kind": "lifecycle",
            "data": {
                "event": "dispatch_finished",
                "detail": {"success": success, "duration_s": round(duration, 1)},
            },
        })
        EVENT_BUS.close(run_id)
        EVENT_BUS.sinks.pop(run_id, None)
        root_logger.removeHandler(file_handler)
        file_handler.close()

    lead_token = generator_chain[0]
    lead_cfg = roles.AGENT_DEFAULTS.get(lead_token, {})
    effort_val = lead_cfg.get("effort")
    effort = str(effort_val) if effort_val not in (None, "n/a") else None
    final_verified = bool(getattr(workflow, "verified", False))
    verifier_delta = _derive_verifier_delta(baseline_result, final_verified)

    wall_ms_value: Optional[float] = (duration * 1000.0) if duration else None
    out_bytes_value: Optional[int] = None
    watchdog_reason_value: Optional[str] = None
    if workflow is not None:
        # Prefer per-agent telemetry when the workflow exposes it.
        agent_for_telemetry = (
            getattr(workflow, "direct_generator", None)
            or getattr(workflow, "generator", None)
        )
        if agent_for_telemetry is not None:
            out_bytes_value = getattr(agent_for_telemetry, "last_out_bytes", None)
            agent_wall = getattr(agent_for_telemetry, "last_wall_ms", None)
            if agent_wall is not None:
                wall_ms_value = float(agent_wall)
            watchdog_reason_value = getattr(agent_for_telemetry, "_watchdog_reason", None)

    # #55: surface the FINAL converged verifier's wall-clock + timeout margin (and
    # a suspected-oversubscription flag) for the run ledger / friendly view. The
    # verifier records its last result; absent a verifier (or a run that never
    # reached the gate) every field is None and meta.json's shape is unchanged
    # except for these always-present keys (the issue asks them to be recorded).
    final_vr = getattr(verifier, "last_result", None) if verifier is not None else None
    verifier_telemetry = _verifier_telemetry(
        final_vr,
        timeout_s=getattr(verifier, "timeout", None) if verifier is not None else None,
        baseline_result=baseline_result,
        run_id=run_id,
    )

    telemetry = {
        "wall_ms": wall_ms_value,
        "out_bytes": out_bytes_value,
        "watchdog_reason": watchdog_reason_value,
        "worker": lead_token,
        "model": str(lead_cfg.get("model", "") or ""),
        "effort": effort,
        "baseline_ok": baseline_result.ok if baseline_result is not None else None,
        "baseline_error_hash": baseline_result.error_hash if baseline_result is not None else None,
        "baseline_duration_ms": baseline_result.duration_ms if baseline_result is not None else None,
        "verifier_delta": verifier_delta,
        **verifier_telemetry,
    }

    # Quality-cost ledger (task #9): how much to trust this run. A run aborted by
    # the run-level watchdog (#77, run_outcome == "stalled") must NOT report
    # "verified"/high — its in-loop verification only covers the steps that ran.
    quality = build_ledger(
        workflow, mode=mode, had_verifier=verifier is not None,
        produced_output=bool(output and output.strip()),
        telemetry=telemetry,
        run_aborted=(run_outcome == "stalled"),
    )
    logger.info("Dispatch %s | confidence=%s (%s)", run_id, quality["confidence"], quality["note"])

    after = take_snapshot(work_dir)
    diff = diff_snapshots(before, after)

    # Path-policy guard (#38): fail the run if a worker touched an off-limits path
    # (denylist) or wrote outside the allowed subtree (allowlist). Gate on the
    # change set the snapshot already computed — no new instrumentation.
    protect_violations: List[Dict[str, str]] = []
    if protect_paths or allow_paths:
        protect_violations = evaluate_path_policy(diff.changed, protect_paths, allow_paths)
        if protect_violations:
            success = False
            offenders = ", ".join(v["path"] for v in protect_violations)
            policy_err = f"path-policy violation: {offenders}"
            error = f"{error}; {policy_err}" if error else policy_err
            logger.warning(
                "Dispatch %s VIOLATED path policy (%d file(s)): %s",
                run_id, len(protect_violations),
                "; ".join(f"{v['path']} ({v['reason']})" for v in protect_violations),
            )

    # Reconciliation verdict (#43): a DISTINCT status, never folded into the
    # verifier's ok. Persist the durable artifact and surface findings. Under the
    # default "warn" disposition the run's success is untouched; only an explicit
    # "fail" disposition (operator opt-in) flips it; "open-task" warns + records a
    # follow-up recommendation (we do NOT auto-file an outward GitHub issue here —
    # that's an outward action left to the operator, see reconcile.json).
    reconciliation: Optional[Dict[str, Any]] = None
    if reconciliation_result is not None:
        reconciliation = reconciliation_result.to_dict()
        # Encoding-safe + atomic: a surrogate from a reconcile finding (ensure_ascii
        # =False keeps raw non-ASCII) must not crash the artifact tail. See
        # _atomic_write_text.
        _atomic_write_text(
            run_dir / "reconcile.json",
            json.dumps(reconciliation, indent=2, ensure_ascii=False),
        )
        dead = reconciliation_result.findings()
        if reconciliation_result.should_fail_run:
            success = False
            if not reconciliation_result.is_substantive:
                # #51/#59: a hollow trace (starved critic, unparseable reply, or an
                # empty trace) is not reconciled, so a fail disposition flips the run
                # — but it failed because the gate never RAN, not on a dead-wiring
                # finding. Name the hollow reason instead of "0 findings".
                recon_err = (
                    f"reconciliation did not verify "
                    f"({reconciliation_result.substance_status()}): the trace was "
                    f"hollow, so the run cannot pass the fail-disposition gate"
                )
            else:
                recon_err = (
                    f"reconciliation failed: {len(dead)} exists-but-not-load-bearing "
                    f"finding(s) (" + ", ".join(f"{f.name}:{f.sub_kind}" for f in dead) + ")"
                )
            error = f"{error}; {recon_err}" if error else recon_err
        elif dead:
            logger.warning(
                "Dispatch %s reconciliation=%s: %d dead-wiring finding(s) "
                "[disposition=%s] — see %s/reconcile.json",
                run_id, reconciliation_result.verdict, len(dead),
                reconciliation_result.disposition, run_dir,
            )
            if reconciliation_result.disposition == "open-task":
                logger.warning(
                    "Dispatch %s: follow-up build task recommended for %d finding(s) "
                    "(auto-filing is opt-in/outward; findings recorded in reconcile.json)",
                    run_id, len(dead),
                )

    # --git-pr finalize (Phase 3): now that success is final (path-policy +
    # reconciliation gates have run), commit any sweep, push the temp branch, and
    # open a draft PR (promoted to ready when verified). The worktree is torn down
    # here; the branch + PR persist for the operator's later merge/corrective
    # decision. Best-effort — never crashes the dispatch.
    git_pr_pr_url: Optional[str] = None
    if git_pr and git_pr_session is not None:
        git_pr_session = _finalize_git_pr(
            session=git_pr_session,
            worktree=git_pr_worktree,
            target_repo=git_pr_target_repo,
            run_dir=git_pr_session_dir,
            run_id=run_id,
            instruction=instruction,
            final_verified=final_verified,
            success=success,
            mode=mode,
            diff=diff,
            quality=quality,
        )
        git_pr_pr_url = git_pr_session.pr_url

    # --git-pr meta summary (#git-pr Phase 6): a compact, durable record of the
    # branch/PR/decision for meta.json. None for a normal run (dropped below).
    git_pr_meta: Optional[Dict[str, Any]] = None
    if git_pr and git_pr_session is not None:
        git_pr_meta = {
            "base_branch": git_pr_session.base_branch,
            "temp_branch": git_pr_session.temp_branch,
            "status": git_pr_session.status,
            "pr_url": git_pr_session.pr_url,
            "pr_number": git_pr_session.pr_number,
            "draft": git_pr_session.draft,
            "verified": git_pr_session.verified,
            "commits": len(git_pr_session.commits),
            "decision": git_pr_session.decision,
            "contributing_runs": list(git_pr_session.contributing_runs),
        }

    # Notify on anomalies + finish (#40). The stall ping already fired from the
    # watchdog; here we cover OOM / verifier-fail / a clean finish, now that
    # success is final (the path-policy gate above may have flipped it).
    if run_outcome != "stalled" and not success and verifier is not None:
        baseline_oom = bool(baseline_result is not None and getattr(baseline_result, "resource_exceeded", False))
        notifier.fire(
            "oom" if baseline_oom else "verifier_fail",
            monitor.payload(extra={
                "baseline_ok": baseline_result.ok if baseline_result is not None else None,
                "phase": "baseline" if baseline_oom else "run",
            }),
        )
    finish_extra: Dict[str, Any] = {"success": success, "outcome": run_outcome}
    if git_pr_pr_url:  # keep non-git-pr finish payloads byte-identical
        finish_extra["pr_url"] = git_pr_pr_url
    notifier.fire("finish", monitor.payload(extra=finish_extra))

    # Encoding-safe + atomic (long-run resilience): a surrogate in worker output
    # must not crash the unguarded artifact tail, and a mid-write kill must not
    # leave a truncated file. See _atomic_write_text.
    _atomic_write_text(run_dir / "stdout.log", output)
    _atomic_write_text(
        run_dir / "changed-files.diff",
        diff.unified or "(no file changes detected)\n",
    )

    # Plan-only / dry-run (#41): persist the decomposed plan as a structured
    # artifact so it can be reviewed/edited before a real run. The out-dir is
    # untouched in this mode (the diff is empty by construction).
    #
    # Round-trip emit (docs §5 M5): when the workflow produced a GraphPlan (an
    # injected --plan graph echoed back via --plan-only), emit the v2 ``nodes``
    # shape so it re-loads as a graph; a flat plan stays the byte-identical legacy
    # ``steps`` object. Both are getattr-guarded off the workflow.
    plan_steps = getattr(workflow, "plan_steps", None) if workflow else None
    if plan_only and plan_steps is not None:
        plan_graph_out = getattr(workflow, "plan_graph", None) if workflow else None
        if plan_graph_out is not None:
            from agy_orchestrator.execution.graph_plan import to_json as _plan_to_json
            graph_obj = _plan_to_json(plan_graph_out)
            graph_obj["instruction"] = instruction
            # Encoding-safe + atomic; the bytes are re-hashed below for provenance,
            # which _atomic_write_text preserves verbatim. See _atomic_write_text.
            _atomic_write_text(
                run_dir / "plan.json",
                json.dumps(graph_obj, indent=2, ensure_ascii=False),
            )
        else:
            _atomic_write_text(
                run_dir / "plan.json",
                json.dumps(
                    {"instruction": instruction, "n_steps": len(plan_steps), "steps": plan_steps},
                    indent=2, ensure_ascii=False,
                ),
            )
        # Plan provenance (#56): hash the freshly emitted plan.json so the
        # operator can pin it with --plan-expect-sha when feeding it back, and so
        # "which plan did this --plan-only run emit?" is answerable from meta.json.
        plan_provenance = {
            "source": "emitted",
            "sha256": plan_file_sha256(run_dir / "plan.json"),
        }

    # Graph-execution summary for meta.json (docs §5 M5). Read the graph
    # observability off the master workflow (pat wraps one in .master_workflow).
    # All fields are getattr-guarded so a flat plan / non-graph run leaves
    # graph_meta None and meta.json is byte-identical: a run that never entered the
    # frontier scheduler has an empty node_status AND no merge_outcomes.
    #   * "nodes"        — per-node statuses {id: "passed"/"failed"/"pending"}
    #   * "layers"       — Kahn levels [[id, ...], ...], the concurrency units
    #   * "merges"       — one MergeOutcome per merged node (M4)
    #   * "merge_policy" — the active policy (disjoint/reconcile/fail)
    graph_meta = None
    if workflow is not None:
        _mwf = getattr(workflow, "master_workflow", None) or workflow
        _node_status = getattr(_mwf, "node_status", None)
        _merges = getattr(_mwf, "merge_outcomes", None)
        if _node_status or _merges:
            _layers: List[List[str]] = []
            _plan_graph = getattr(_mwf, "plan_graph", None)
            if _plan_graph is not None:
                try:
                    _layers = _plan_graph.parallel_groups()
                except Exception:
                    _layers = []
            graph_meta = {
                "nodes": dict(_node_status) if _node_status else {},
                "layers": _layers,
                "merges": [m.to_dict() for m in (_merges or [])],
                "merge_policy": getattr(_mwf, "merge_policy", DEFAULT_MERGE_POLICY),
            }

    stage_used = getattr(workflow, "stage_used", None) if workflow else None
    n_candidates = getattr(workflow, "n_candidates", None) if workflow else None
    n_passed = getattr(workflow, "n_passed", None) if workflow else None
    winner_index = getattr(workflow, "winner_index", None) if workflow else None
    if stage_used == -1:
        stage_used = None
    if winner_index == -1:
        winner_index = None

    # TODO: Wire the final VerifierResult up from workflows so this can
    # differentiate timeout vs returncode failures for non-verified runs.
    if quality.get("confidence") == "verified":
        verifier_failure_kind = None
    else:
        verifier_failure_kind = None

    # Close the calibration loop: every verified dispatch contributes one
    # observation to the live ledger that CalibrationTable.load() reads on
    # next process start. Only verified runs count — matching the offline
    # sweep's gate, so a critic-approved-but-tests-failed run doesn't
    # pollute the routing baselines. We use the lead generator worker as
    # the key; multi-stage modes (cascade, pat, master) inherit it.
    if quality.get("confidence") == "verified":
        append_live_row(
            worker=lead_token,
            model=str(lead_cfg.get("model", "") or ""),
            effort=effort,
            ok=True,
            out_bytes=out_bytes_value,
            wall_ms=wall_ms_value,
            mode=mode,
            stage_used=stage_used,
            n_candidates=n_candidates,
            n_passed=n_passed,
            winner_index=winner_index,
            verifier_delta=quality.get("verifier_delta"),
            verifier_failure_kind=verifier_failure_kind,
            diff_files_added=len(diff.added),
            diff_files_modified=len(diff.modified),
            diff_files_deleted=len(diff.deleted),
        )

    tokens = _summarize_token_usage(events_path)
    result = DispatchResult(
        run_id=run_id,
        run_dir=str(run_dir),
        mode=mode,
        generator=gen_desc,
        critic=crit_desc,
        success=success,
        duration_s=round(duration, 1),
        changed_files=diff.changed,
        added=diff.added,
        modified=diff.modified,
        deleted=diff.deleted,
        error=error,
        quality=quality,
        tokens=tokens,
        protect_violations=protect_violations,
        run_outcome=run_outcome,
        resolved_config=resolved_config,
        reconciliation=reconciliation,
        reconcile_status=reconcile_status,
        graph=graph_meta,
        verifier=(verifier_telemetry or None),
        plan_provenance=plan_provenance,
        git_pr=git_pr_meta,
    )
    meta_dict = asdict(result)
    # The graph-execution summary is present ONLY for a graph DAG run; honor the
    # graph_meta writer's contract (a flat / non-graph run's meta.json is
    # byte-identical to the pre-graph-feature shape) by dropping the key when null
    # rather than emitting a spurious "graph": null. (reconciliation / reconcile_status
    # intentionally always appear — see their #44 "always set" contract — so they
    # are NOT dropped here.)
    if meta_dict.get("graph") is None:
        meta_dict.pop("graph", None)
    # Final-verifier observability (#55) follows the same drop-when-absent contract
    # as graph: a run without a verifier (or one that never converged a result)
    # leaves meta.json byte-identical to the pre-#55 shape.
    if meta_dict.get("verifier") is None:
        meta_dict.pop("verifier", None)
    # Plan provenance (#56) follows the same drop-when-absent contract: a
    # planner-driven run (no --plan-only emit, no --plan injection) leaves
    # meta.json byte-identical to the pre-#56 shape.
    if meta_dict.get("plan_provenance") is None:
        meta_dict.pop("plan_provenance", None)
    # --git-pr summary follows the same drop-when-absent contract: a normal run
    # leaves meta.json byte-identical to the pre-git-pr shape.
    if meta_dict.get("git_pr") is None:
        meta_dict.pop("git_pr", None)
    # Atomic + encoding-safe: meta.json is the run's authoritative record (success
    # /diff/tokens/reconcile). A mid-write kill must leave it valid-or-absent, and a
    # surrogate that reached a string field must not abort the write. ensure_ascii is
    # left at its default (True) here, so json.dumps already escapes non-ASCII; the
    # backslashreplace in _atomic_write_text is a belt-and-suspenders backstop.
    _atomic_write_text(run_dir / "meta.json", json.dumps(meta_dict, indent=2))

    # Telegram final-summary card (best-effort; never affects dispatch result).
    if telegram_notifier is not None:
        try:
            telegram_notifier.finished(meta_dict)
        except Exception as exc:
            logger.debug("telegram finish summary failed: %s", exc)
    # Stop the embedded command poller (releases the getUpdates singleton lock so a
    # standalone daemon or the next dispatch can take over). Best-effort (#63).
    if telegram_poller is not None:
        try:
            telegram_poller.stop()
        except Exception as exc:
            logger.debug("telegram command poller stop failed: %s", exc)

    return result


def dispatch(
    instruction: str,
    *,
    run_id: Optional[str] = None,
    mode: str = "adversarial",
    context: Optional[str] = None,
    generator_chain: Optional[List[str]] = None,
    critic_chain: Optional[List[str]] = None,
    fallback: bool = True,
    cycles: int = 2,
    max_iterations: int = 5,
    branches: int = 3,
    test_cmd: Optional[str] = None,
    verifier_mem_max: Optional[str] = None,
    web_search: bool = False,
    mission_critical: bool = False,
    spec: Optional[str] = None,
    dashboard_stream_json: bool = False,
    out_dir: Optional[Union[str, Path]] = None,
    candidate_setup: Optional[str] = None,
    git_pr: bool = False,
    git_pr_continue: Optional[str] = None,
    resume_policy: str = "auto",
    protect_paths: Optional[List[str]] = None,
    allow_paths: Optional[List[str]] = None,
    plan_only: bool = False,
    plan_steps: Optional[List[str]] = None,
    plan_source: Optional[Union[str, Path]] = None,
    plan_expect_sha: Optional[str] = None,
    plan_graph: Optional[GraphPlan] = None,
    max_parallel_nodes: Optional[int] = None,
    verifier_concurrency: int = 1,
    merge_policy: str = DEFAULT_MERGE_POLICY,
    run_stall_abort: Optional[float] = None,
    notify: Optional[str] = None,
    notify_cmd: Optional[str] = None,
    heartbeat_interval: Optional[float] = None,
    telegram_enabled: Optional[bool] = None,
    telegram_verbosity: Optional[str] = None,
    # Per-role / per-provider effort+model overrides (#42)
    gen_effort: Optional[str] = None,
    gen_model: Optional[str] = None,
    critic_effort: Optional[str] = None,
    critic_model: Optional[str] = None,
    architect_effort: Optional[str] = None,
    architect_model: Optional[str] = None,
    codex_model: Optional[str] = None,
    effort_map: Optional[str] = None,
    model_map: Optional[str] = None,
    effort_profile: Optional[str] = None,
    watchdog_scale: Optional[float] = None,
    # Issue #83 — absolute, independent watchdog budget overrides (see dispatch_async)
    watchdog_max_bytes: Optional[int] = None,
    watchdog_stall: Optional[float] = None,
    max_parallel_workers: Optional[int] = None,
    worker_mem_max: Optional[str] = None,
    baseline_gate: bool = False,
    reconcile: bool = False,
    reconcile_disposition: Optional[str] = None,
    # Programmatic ablation-witness hook (#52); see dispatch_async.
    ablation_cmd: Optional[str] = None,
    ablation_cmd_map: Optional[Dict[str, str]] = None,
    # Step 12: forwarded for computer-use adapter (see dispatch_async)
    # Step 10: real-gui harness flags (passed through only when present; non-real paths identical)
    computer_use_mode: Optional[str] = None,
    computer_use_task_priority: Optional[str] = None,
    computer_use_budgets: Optional[Dict[str, Any]] = None,
    real_gui_policy: Optional[str] = None,
    ask_mode: Optional[str] = None,
    browser_engine: Optional[str] = None,
    browser_display: Optional[str] = None,
) -> DispatchResult:
    """Execute one instruction and capture the run. Synchronous entrypoint."""
    return asyncio.run(
        dispatch_async(
            instruction,
            run_id=run_id,
            mode=mode,
            context=context,
            generator_chain=generator_chain,
            critic_chain=critic_chain,
            fallback=fallback,
            cycles=cycles,
            max_iterations=max_iterations,
            branches=branches,
            test_cmd=test_cmd,
            verifier_mem_max=verifier_mem_max,
            web_search=web_search,
            mission_critical=mission_critical,
            spec=spec,
            dashboard_stream_json=dashboard_stream_json,
            out_dir=out_dir,
            candidate_setup=candidate_setup,
            git_pr=git_pr,
            git_pr_continue=git_pr_continue,
            resume_policy=resume_policy,
            protect_paths=protect_paths,
            allow_paths=allow_paths,
            plan_only=plan_only,
            plan_steps=plan_steps,
            plan_source=plan_source,
            plan_expect_sha=plan_expect_sha,
            plan_graph=plan_graph,
            max_parallel_nodes=max_parallel_nodes,
            verifier_concurrency=verifier_concurrency,
            merge_policy=merge_policy,
            run_stall_abort=run_stall_abort,
            notify=notify,
            notify_cmd=notify_cmd,
            heartbeat_interval=heartbeat_interval,
            telegram_enabled=telegram_enabled,
            telegram_verbosity=telegram_verbosity,
            gen_effort=gen_effort,
            gen_model=gen_model,
            critic_effort=critic_effort,
            critic_model=critic_model,
            architect_effort=architect_effort,
            architect_model=architect_model,
            codex_model=codex_model,
            effort_map=effort_map,
            model_map=model_map,
            effort_profile=effort_profile,
            watchdog_scale=watchdog_scale,
            watchdog_max_bytes=watchdog_max_bytes,
            watchdog_stall=watchdog_stall,
            max_parallel_workers=max_parallel_workers,
            worker_mem_max=worker_mem_max,
            baseline_gate=baseline_gate,
            reconcile=reconcile,
            reconcile_disposition=reconcile_disposition,
            ablation_cmd=ablation_cmd,
            ablation_cmd_map=ablation_cmd_map,
            computer_use_mode=computer_use_mode,
            computer_use_task_priority=computer_use_task_priority,
            computer_use_budgets=computer_use_budgets,
            real_gui_policy=real_gui_policy,
            ask_mode=ask_mode,
            browser_engine=browser_engine,
            browser_display=browser_display,
        )
    )

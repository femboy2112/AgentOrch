# Git PR Mode (`--git-pr`) — design

**Status:** building. ✅ Phase 0 (`harness/gitpr.py`), ✅ Phase 1 (preflight +
worktree setup), ✅ Phase 2 (per-accepted-step commits), ✅ Phase 3 (push +
draft-PR create/promote), ✅ Phase 4a (CLI verbs `pr`/`merge`/`abandon`),
✅ Phase 5 (corrective resume `--continue`) landed on `main`. Pending: Phase 4b
(Telegram buttons), Phase 6 (observability). Opt-in. Owner: operator + Claude session.

Phase 5 corrective resume: `harness do "FIX…" --continue <run_id>` re-attaches a
worktree to the prior run's temp branch (which already holds all committed work, so
the worker continues from that state), runs the instruction on top, keeps
committing to the SAME branch → the SAME PR updates on push (no second PR; promoted
to ready when the corrective verifies). The corrective run gets its own run_id but
updates the ORIGINAL canonical `pr_session.json` (`parent_run_id` + appended to
`contributing_runs`). Works for any mode — no master-checkpoint surgery needed,
because a completed run's correction is new work on top, not a half-step resume.

Phase 3 session `status` ladder: `no_changes` (nothing committed) · `branch_ready`
(committed, no remote → local branch only) · `pushed_no_pr` (pushed, gh absent or
PR-create failed → manual `gh pr create` logged) · `awaiting_decision` (PR open;
`draft` flips false when the run verified) · `error` (commit failed → worktree
retained for inspection).
**One line:** a dispatch runs on an isolated temp git branch, commits each accepted
step, opens a **draft PR** to the clean base branch (promoted to ready when
verification passes), and persists a session so the operator can later **merge** or
fire a **corrective plan** that resumes on the same branch/PR.

This is additive: with `--git-pr` off, behavior is byte-identical to today (no git
ops). The existing git-independent snapshot diff (`harness/snapshot.py`) is untouched
and keeps running alongside the git commits.

---

## 1. Operator story

```bash
# clean branch, GitHub remote present
python -m harness do "build the X subsystem" --git-pr --mode master --test-cmd "pytest -q"
#   → temp branch agentorch/<run_id> (in an isolated worktree)
#   → one commit per accepted step
#   → draft PR  agentorch/<run_id> → <base>   (auto-ready if verification passed)
#   → operator's checkout never moved; runs/<id>/pr_session.json written

# later — operator decides (async, desk or phone):
python -m harness pr <run_id>                       # show status / PR url
python -m harness merge <run_id> --method squash    # gh pr merge
python -m harness do "also handle the empty-input case" --continue <run_id>   # corrective
python -m harness abandon <run_id>                  # gh pr close + drop branch
```

Telegram finish card carries the same three actions as inline buttons:
`[✅ Merge] [🔧 Corrective…] [🗑 Abandon]`.

---

## 2. Decisions (operator-confirmed)

| Fork | Choice |
|---|---|
| Merge/corrective decision | **Async** — dispatch ends, persists `pr_session.json`; operator decides later via CLI verb or Telegram tap. |
| Commit granularity | **Per accepted step** (verified/approved). Single-step modes get one final commit. |
| PR on verification fail | **Draft PR always** (commits visible, CI runs, corrective attaches to it); auto-promote to ready when verification passes. |
| Control surface | **CLI + Telegram.** CLI is the foundation; Telegram buttons reuse the existing gate-decision/action seam. |
| Execution isolation | **Dedicated git worktree** for the temp branch (design call, §4). Operator's checkout never moves. |

---

## 3. Where it hooks (from the code map)

- `harness/dispatch.py` — completion block runs *after* the snapshot diff, where
  `success` and `final_verified = bool(getattr(workflow, "verified", False))` are
  already known. Preflight slots in before the workflow runs; PR create/promote in
  the completion block. **No git in dispatch today** — all net-new and gated on the flag.
- `harness/snapshot.py` — unchanged. Git-independent before/after diff still produces
  `changed-files.diff`. Git commits run in parallel; they do not replace it.
- `agy_orchestrator/workflows/master.py` — already emits
  `phase=step, action=completed, outcome=verified|approved` per step (linear *and*
  graph, post-merge/re-verify). That boundary is the per-step commit hook. Already has
  checkpoint/resume keyed on prompt-SHA + a git **base-fingerprint guard (#37)** that
  *already* accepts a benign forward HEAD advance — the backbone for corrective resume.
- `agy_orchestrator/execution/workspace.py` — already shells to git with the right
  patterns (worktree add/remove/prune, `status --porcelain` dirty-detect, copy-mode
  fallback, timeouts, surrogate-safe decode, never-wedge). **Reuse it** for worktree
  isolation; mirror its subprocess discipline in the new gh wrapper.
- Telegram **gate-decision sidecar** (`gate_decision.json`) + `/build` detached-spawn
  action seam already exist but are **currently unconsumed**. The merge/corrective
  decision finally *consumes* this seam.
- **No `gh` usage anywhere** — PR create/ready/merge/close is net-new.

---

## 4. Execution isolation — worktree, not in-place checkout

Rather than checking out the temp branch in the operator's working tree (which forces
a checkout-base/restore-base dance, strands the operator on the temp branch if the
process is killed, and makes two concurrent `--git-pr` runs in one repo fight over
HEAD), the whole dispatch runs inside a **dedicated git worktree**:

1. `git worktree add --detach <tmp> <base_sha>` then create+checkout
   `agentorch/<run_id>` inside it (reuse `_ManagedWorkspace`, `prefer_worktree=True`).
2. Workers' cwd = the worktree path (the existing `--out-dir`/`work_dir` plumbing
   already points worker cwd + the snapshot scope at one directory — point both here).
3. Commits, push, PR all operate on the worktree.
4. On completion/abort: `git worktree remove --force` (best-effort, never raises).

Benefits: operator's checkout never moves (nothing to restore), `--git-pr` runs are
concurrency-safe per repo (one worktree per run-id), and graph mode's own per-node
worktrees still attach to the common `.git` fine.

**Fallback** (documented, not default): `--git-pr-inplace` checks out the temp branch
in `work_dir` with strict save/restore-base in a `try/finally` + signal/atexit hook
(same pattern as the worker-death-cascade reaping). For repos where worktrees are
problematic (submodule edge cases, some CI checkouts).

**Open risk:** worktree-of-the-target while graph mode adds *its* worktrees off the
same `.git` — git supports many worktrees off one common dir, but verify the nesting
and `worktree prune` interplay in an integration test before shipping graph+git-pr.

---

## 5. Phased build plan

### Phase 0 — `harness/gitpr.py` (git + gh wrapper)
Thin, fully-unit-tested wrapper mirroring `workspace.py` discipline (timeouts, capture,
`errors="replace"`, never wedge the dispatch — a git/gh failure downgrades to
snapshot-only with a loud warning, never crashes). Surface:
`is_git_repo`, `is_dirty`, `current_branch`, `head_sha`, `has_remote`, `gh_available`,
`gh_authed`, `add_worktree`, `create_branch`, `stage_and_commit(msg)->sha|None`,
`push`, `create_pr(base,head,title,body,draft=True)->PrInfo`, `mark_ready`,
`merge_pr(method)`, `close_pr`, `pr_status`.
*Staging:* `git add -A` **within the worktree** (honors the target repo's `.gitignore`;
this is the correct "commit what the worker produced this step" semantic). Note: the
project's no-`git add -A` operating rule governs *my* commits to the AgentOrch repo —
the feature's commits to a throwaway temp branch in the user's repo are a separate
matter and documented as such.

### Phase 1 — preflight + branch/worktree setup (dispatch.py)
On `--git-pr`: resolve `work_dir`; **refuse fast** if not a git repo / tree dirty
(require clean — `start with clean branch`; mirrors the #36 data-loss lesson) /
detached HEAD. Record `base_branch` + `base_sha`. Create the worktree + temp branch.
Write `pr_session.json` (status `running`). Probe gh/remote now — if absent, warn that
the PR step will be skipped (commits + branch still happen) and continue.

### Phase 2 — per-accepted-step commits
Add an explicit `step_committed` callback param to `MasterWorkflow` (cleaner +
testable than sniffing the event stream), invoked on step-completed with
outcome ∈ {verified, approved}: `git add -A` + commit, message
`step N/M: <title> · <worker> <model>/<effort> · <outcome>`; append sha to session.
Graph mode commits at the same step-completed boundary — **after** node merge-back +
re-verify, serialized by the existing merge semaphore so commits never race.
Single-step modes (direct/adversarial/feedback/cascade/pat-stage-1/vote-winner) have no
per-step events → dispatch does **one final-sweep commit** at completion of any
remaining changes (labelled `(unverified)` when `success` is false, so corrective
resume always has a base; PR still not promoted in that case).

### Phase 3 — PR create / promote (dispatch completion block)
If commits == 0 → skip PR, note in session. Else: push temp branch; if gh+remote,
`gh pr create --draft` (title from instruction; body = run summary: mode, step
outcomes, confidence ladder, commit list, link to `runs/<id>`). If `success and
final_verified` → `gh pr ready`. Persist session (status `awaiting_decision`,
`decision=null`, pr url/number). Tear down the worktree. Emit finish event carrying
`pr_url` + decision affordances.

### Phase 4 — operator decision surface (async)
**CLI:** `harness pr <id>` (status), `harness merge <id> [--method squash|merge|rebase]
[--delete-branch]`, `harness do "FIX" --continue <id>` (Phase 5), `harness abandon <id>`.
**Telegram:** finish card gains `[✅ Merge] [🔧 Corrective…] [🗑 Abandon]`. A tap writes
`runs/<id>/pr_decision.json {decision, ts, instruction?}` (the gate-decision pattern);
a consumer reads it and invokes the matching CLI verb via the existing detached-spawn
seam. Corrective prompts (Telegram conversation step) for the instruction, then spawns
`harness do --continue`.

### Phase 5 — corrective resume (`--continue <id>`)
Load session (base, temp branch, work_dir, PR#, checkpoint_key, plan, completed
frontier). Re-add the worktree on the temp branch (refuse if it was hand-dirtied;
rebase support later). Master: feed prior plan + completed-step summaries + the
corrective instruction to the planner to produce remainder/adjustment steps, seeded
from the prior `project_context` + done-frontier; the #37 base-fingerprint guard
already treats the temp branch's advanced HEAD as a benign forward. Single-step: run
the corrective instruction fresh on the temp branch. Continue per-step commits on the
**same** branch → the **same** PR updates automatically. Re-evaluate verification →
promote/keep-draft. The corrective run gets its own `run_id` but updates the
**original** `pr_session.json` (linked via `parent_run_id`). Repeatable
(corrective-on-corrective).

### Phase 6 — observability
`meta.json` gains a `git_pr` block (base, temp, pr_url, commits, decision).
`harness runs` shows a PR column. Telegram finish card shows PR url + status.
`pr_session.json` is the single source of truth across corrective runs.

---

## 6. Safety invariants

- Never touch the base branch except via `gh pr merge`. **Never force-push.**
- Operator's checkout never moves (worktree isolation); inplace fallback restores base
  via `try/finally` + signal/atexit even on `kill -9` (best-effort PDEATHSIG pattern).
- **Refuse on a dirty tree** — no silent clobber of uncommitted/untracked work (#36).
- Degrade gracefully with no gh/remote: local branch + commits only; print the manual
  `git push` + PR-create command.
- Every git/gh call is timeout-bounded and never wedges the dispatch; a mid-run git
  failure downgrades to snapshot-only mode with a loud warning, not a crash.
- One `--git-pr` dispatch per `work_dir` at a time is naturally safe via per-run-id
  worktrees; document it anyway.

## 7. Test plan (hermetic — `git init` in tmp_path, **gh stubbed on PATH**, never hit real GitHub)

`gitpr.py` units (init/dirty/branch/commit/push/PR); preflight refusals (dirty /
non-repo / detached); per-step commit hook (fake master run → commits appear);
final-sweep commit for single-step modes; draft PR created → promote-on-verified →
no-PR-without-gh degrade; worktree created + torn down (and on exception/abort);
corrective resume (re-add worktree, append steps, same branch, session updated);
CLI verb parsing + spawn; Telegram decision sidecar write + consume → spawn;
**back-compat: without `--git-pr`, zero git ops and byte-identical meta.json.**

## 8. Open questions / risks to settle during build

1. Worktree nesting with graph mode's per-node worktrees (integration test before ship).
2. Corrective resume when the PR received external review commits / the branch was
   hand-edited — default refuse-on-dirty; rebase-onto support is a follow-up.
3. PR body size caps (mirror the Telegram 4096 tail-cap lesson for long summaries).
4. `--out-dir` + worktree: the worker writes into the worktree, not the operator's
   tree — intended (deliverable is the PR), but document so it isn't surprising.

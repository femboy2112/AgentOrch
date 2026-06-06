"""Git + GitHub (``gh``) plumbing for ``--git-pr`` mode.

A ``--git-pr`` dispatch runs on an isolated temp branch (in a dedicated git
worktree), commits each accepted step, and opens a draft PR to the clean base
branch. This module is the thin, never-wedge subprocess layer underneath that:

* Every call is timeout-bounded. A missing binary, a hang, or an OS error is
  turned into a sentinel return (``rc=127``/``rc=124``) — never an unhandled
  exception that could crash the dispatch.
* Probe-style helpers (``is_git_repo``, ``is_dirty``, ``head_sha`` …) fall
  *open* — they return a safe default on any failure so a git hiccup can't wedge
  a dispatch. Mutating helpers (``add_worktree``, ``commit``, ``push``,
  ``create_pr`` …) raise :class:`GitError` on failure so the caller can decide
  to downgrade the run to snapshot-only mode with a loud warning.
* All subprocess work funnels through :func:`_run` so tests can point ``git`` at
  a throwaway ``git init`` repo and ``gh`` at a stub on ``PATH`` — no network,
  no real GitHub.

The git-repo / dirty-tree probes are reused from
:mod:`agy_orchestrator.execution.workspace` (the vote / graph isolation layer)
so there is exactly one implementation of "is this a clean git work tree".
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Reuse the single canonical "clean git work tree?" probes (issue #36 lineage).
from agy_orchestrator.execution.workspace import (
    has_uncommitted_changes as is_dirty,
    is_git_repo,
)

logger = logging.getLogger(__name__)

# Timeout budgets (seconds). Generous enough for a big repo / slow network,
# tight enough that a wedged git/gh can never hang a dispatch indefinitely.
PROBE_TIMEOUT = 10
COMMIT_TIMEOUT = 60
WORKTREE_TIMEOUT = 60
PUSH_TIMEOUT = 180
GH_TIMEOUT = 120

_MISSING_RC = 127   # binary not found
_TIMEOUT_RC = 124   # timed out (matches GNU `timeout`)


class GitError(RuntimeError):
    """A git/gh mutation failed. Caller decides whether to downgrade the run."""


def branch_name_for_run(run_id: str) -> str:
    """The temp branch a ``--git-pr`` run commits to. Namespaced so it never
    collides with a human branch and is trivially greppable / sweepable."""
    return f"agentorch/{run_id}"


# --------------------------------------------------------------------------- #
# Subprocess core
# --------------------------------------------------------------------------- #
def _run(args: Sequence[str], *, cwd: Optional[Path] = None,
         timeout: int = PROBE_TIMEOUT) -> Tuple[int, str, str]:
    """Run a command; return ``(returncode, stdout, stderr)``. Never raises.

    A missing binary yields ``rc=127``; a timeout yields ``rc=124``; any other
    OS error yields ``rc=1`` with the message on stderr. Output is decoded with
    ``errors="replace"`` so a worker that emitted invalid UTF-8 into a path /
    commit message can't crash the decode."""
    try:
        cp = subprocess.run(
            [str(a) for a in args],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True, text=True, errors="replace", timeout=timeout,
        )
        return cp.returncode, cp.stdout, cp.stderr
    except FileNotFoundError:
        return _MISSING_RC, "", f"{args[0]}: command not found"
    except subprocess.TimeoutExpired:
        return _TIMEOUT_RC, "", f"{args[0]}: timed out after {timeout}s"
    except OSError as exc:  # pragma: no cover - defensive
        return 1, "", str(exc)


def _git(path: Path, *args: str, timeout: int = PROBE_TIMEOUT,
         pre: Sequence[str] = ()) -> Tuple[int, str, str]:
    """``git -C <path> [pre...] <args...>`` (``pre`` for ``-c k=v`` globals)."""
    return _run(["git", "-C", str(path), *pre, *args], timeout=timeout)


def _gh(path: Path, *args: str, timeout: int = GH_TIMEOUT) -> Tuple[int, str, str]:
    """``gh <args...>`` run with cwd=path (gh resolves the repo from cwd)."""
    return _run(["gh", *args], cwd=path, timeout=timeout)


# --------------------------------------------------------------------------- #
# Probes (fall open)
# --------------------------------------------------------------------------- #
def current_branch(path: Path) -> Optional[str]:
    """Current branch name, or ``"HEAD"`` when detached, or ``None`` on error."""
    rc, out, _ = _git(Path(path), "rev-parse", "--abbrev-ref", "HEAD")
    return out.strip() if rc == 0 and out.strip() else None


def is_detached_head(path: Path) -> bool:
    """True iff HEAD is detached (``--git-pr`` refuses this: no base branch)."""
    return current_branch(path) == "HEAD"


def head_sha(path: Path) -> Optional[str]:
    """Full SHA of HEAD, or ``None`` on error / empty repo."""
    rc, out, _ = _git(Path(path), "rev-parse", "HEAD")
    return out.strip() if rc == 0 and out.strip() else None


def has_remote(path: Path, remote: Optional[str] = None) -> bool:
    """True iff the repo has the named remote (any remote when ``remote`` None)."""
    rc, out, _ = _git(Path(path), "remote")
    if rc != 0:
        return False
    remotes = out.split()
    return bool(remotes) if remote is None else (remote in remotes)


def gh_available() -> bool:
    """True iff the ``gh`` CLI is on PATH."""
    return shutil.which("gh") is not None


def gh_authed(path: Optional[Path] = None) -> bool:
    """True iff ``gh auth status`` succeeds (a usable GitHub credential)."""
    if not gh_available():
        return False
    rc, _, _ = _gh(Path(path) if path is not None else Path.cwd(),
                   "auth", "status", timeout=PROBE_TIMEOUT)
    return rc == 0


# --------------------------------------------------------------------------- #
# Worktree lifecycle
# --------------------------------------------------------------------------- #
def add_worktree(base: Path, target: Path, *, branch: str,
                 start_point: Optional[str] = None,
                 timeout: int = WORKTREE_TIMEOUT) -> None:
    """``git worktree add -b <branch> <target> [<start_point>]``.

    Creates ``branch`` at ``start_point`` (default: base HEAD) and checks it out
    in a fresh worktree at ``target``. The operator's own checkout never moves.
    Raises :class:`GitError` on failure."""
    args = ["worktree", "add", "-b", branch, str(target)]
    if start_point:
        args.append(start_point)
    rc, out, err = _git(Path(base), *args, timeout=timeout)
    if rc != 0:
        raise GitError(
            f"git worktree add failed (rc={rc}): {(err or out).strip()}"
        )


def remove_worktree(base: Path, target: Path,
                    timeout: int = WORKTREE_TIMEOUT) -> None:
    """Best-effort worktree teardown + registry prune. Never raises — cleanup
    must not mask the original outcome. Mirrors workspace.py's teardown."""
    try:
        _git(Path(base), "worktree", "remove", "--force", str(target),
             timeout=timeout)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("worktree remove failed: %s", exc)
    finally:
        shutil.rmtree(target, ignore_errors=True)
        try:
            _git(Path(base), "worktree", "prune", timeout=timeout)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("worktree prune failed (non-fatal): %s", exc)


# --------------------------------------------------------------------------- #
# Commit
# --------------------------------------------------------------------------- #
def stage_all(path: Path, timeout: int = COMMIT_TIMEOUT) -> None:
    """``git add -A`` within the worktree.

    Honours the target repo's ``.gitignore`` — the correct "commit exactly what
    the worker produced this step" semantic. (The project's no-``git add -A``
    operating rule governs commits to the AgentOrch repo itself; this stages a
    throwaway temp branch in the *user's* target repo, which is a different
    matter.) Raises :class:`GitError` on failure."""
    rc, _, err = _git(Path(path), "add", "-A", timeout=timeout)
    if rc != 0:
        raise GitError(f"git add -A failed (rc={rc}): {err.strip()}")


def commit(path: Path, message: str, *, author_name: Optional[str] = None,
           author_email: Optional[str] = None,
           timeout: int = COMMIT_TIMEOUT) -> Optional[str]:
    """Stage everything and commit. Returns the new commit SHA, or ``None`` when
    there was nothing to commit (a clean tree — not an error).

    When ``author_name``/``author_email`` are given they are applied via ``-c``
    so the commit is attributed (e.g. to the worker that produced the step)
    without mutating repo config. Raises :class:`GitError` on a real failure."""
    path = Path(path)
    stage_all(path, timeout=timeout)
    # Nothing staged -> nothing to commit. `diff --cached --quiet` is rc 0 when
    # the index matches HEAD (clean), rc 1 when there are staged changes.
    rc, _, _ = _git(path, "diff", "--cached", "--quiet", timeout=timeout)
    if rc == 0:
        return None
    pre: List[str] = []
    if author_name and author_email:
        pre = ["-c", f"user.name={author_name}", "-c", f"user.email={author_email}"]
    rc, out, err = _git(path, "commit", "-m", message, pre=pre, timeout=timeout)
    if rc != 0:
        raise GitError(f"git commit failed (rc={rc}): {(err or out).strip()}")
    return head_sha(path)


def push(path: Path, branch: str, *, remote: str = "origin",
         set_upstream: bool = True, timeout: int = PUSH_TIMEOUT) -> None:
    """``git push [-u] <remote> <branch>``. Raises :class:`GitError`."""
    args = ["push"]
    if set_upstream:
        args.append("-u")
    args += [remote, branch]
    rc, out, err = _git(Path(path), *args, timeout=timeout)
    if rc != 0:
        raise GitError(f"git push failed (rc={rc}): {(err or out).strip()}")


# --------------------------------------------------------------------------- #
# GitHub PR (gh)
# --------------------------------------------------------------------------- #
@dataclass
class PrInfo:
    """Result of opening a PR."""
    url: str
    number: Optional[int]


_PR_URL_RE = re.compile(r"https?://\S+?/pull/(\d+)")


def _parse_pr_ref(text: str) -> PrInfo:
    """Extract the PR URL + number from ``gh pr create`` output (URL on stdout)."""
    url = ""
    number: Optional[int] = None
    for m in _PR_URL_RE.finditer(text or ""):
        url = m.group(0)
        number = int(m.group(1))
    if not url:  # fall back to the last bare URL printed
        for line in reversed((text or "").splitlines()):
            line = line.strip()
            if line.startswith("http"):
                url = line
                break
    return PrInfo(url=url, number=number)


def create_pr(path: Path, *, base: str, head: str, title: str, body: str,
              draft: bool = True, timeout: int = GH_TIMEOUT) -> PrInfo:
    """``gh pr create`` (draft by default). Raises :class:`GitError` on failure."""
    args = ["pr", "create", "--base", base, "--head", head,
            "--title", title, "--body", body]
    if draft:
        args.append("--draft")
    rc, out, err = _gh(Path(path), *args, timeout=timeout)
    if rc != 0:
        raise GitError(f"gh pr create failed (rc={rc}): {(err or out).strip()}")
    info = _parse_pr_ref(out + "\n" + err)
    if not info.url:
        raise GitError(f"gh pr create succeeded but no PR URL parsed: {out!r}")
    return info


def mark_ready(path: Path, pr: object, timeout: int = GH_TIMEOUT) -> None:
    """Promote a draft PR to ready-for-review (``gh pr ready``)."""
    rc, out, err = _gh(Path(path), "pr", "ready", str(pr), timeout=timeout)
    if rc != 0:
        raise GitError(f"gh pr ready failed (rc={rc}): {(err or out).strip()}")


def merge_pr(path: Path, pr: object, *, method: str = "squash",
             delete_branch: bool = False, timeout: int = GH_TIMEOUT) -> None:
    """``gh pr merge`` with the given method (squash/merge/rebase)."""
    if method not in ("squash", "merge", "rebase"):
        raise GitError(f"unknown merge method: {method!r}")
    args = ["pr", "merge", str(pr), f"--{method}"]
    if delete_branch:
        args.append("--delete-branch")
    rc, out, err = _gh(Path(path), *args, timeout=timeout)
    if rc != 0:
        raise GitError(f"gh pr merge failed (rc={rc}): {(err or out).strip()}")


def close_pr(path: Path, pr: object, *, delete_branch: bool = False,
             timeout: int = GH_TIMEOUT) -> None:
    """``gh pr close`` (optionally deleting the branch). Raises on failure."""
    args = ["pr", "close", str(pr)]
    if delete_branch:
        args.append("--delete-branch")
    rc, out, err = _gh(Path(path), *args, timeout=timeout)
    if rc != 0:
        raise GitError(f"gh pr close failed (rc={rc}): {(err or out).strip()}")


def pr_status(path: Path, pr: object, timeout: int = GH_TIMEOUT) -> dict:
    """``gh pr view --json …`` -> dict (state/isDraft/url/number/title). Raises."""
    rc, out, err = _gh(Path(path), "pr", "view", str(pr),
                       "--json", "state,isDraft,url,number,title", timeout=timeout)
    if rc != 0:
        raise GitError(f"gh pr view failed (rc={rc}): {(err or out).strip()}")
    try:
        return json.loads(out)
    except (ValueError, TypeError) as exc:
        raise GitError(f"gh pr view returned non-JSON: {exc}") from exc


# --------------------------------------------------------------------------- #
# Session: the persisted record tying base/temp/PR/checkpoint together
# --------------------------------------------------------------------------- #
@dataclass
class PrSession:
    """The single source of truth for one ``--git-pr`` run, persisted to
    ``runs/<run_id>/pr_session.json``. Corrective runs (``--continue``) reuse the
    ORIGINAL run's session, appending commits and linking via ``parent_run_id``.

    ``status`` ladder:
      running -> awaiting_decision -> merged | abandoned
                                   \\-> error | no_changes (terminal-ish)
    """
    run_id: str
    base_branch: str
    temp_branch: str
    base_sha: str
    work_dir: str
    status: str = "running"
    pr_url: Optional[str] = None
    pr_number: Optional[int] = None
    draft: bool = True
    verified: bool = False
    decision: Optional[str] = None
    parent_run_id: Optional[str] = None
    commits: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PrSession":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def session_path(run_dir: Path) -> Path:
    return Path(run_dir) / "pr_session.json"


def save_session(run_dir: Path, session: PrSession) -> None:
    """Atomically persist the session to ``runs/<id>/pr_session.json``."""
    p = session_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(p)


def load_session(run_dir: Path) -> Optional[PrSession]:
    """Load the session for a run dir, or ``None`` if absent/corrupt."""
    p = session_path(run_dir)
    try:
        return PrSession.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (FileNotFoundError, ValueError, TypeError):
        return None

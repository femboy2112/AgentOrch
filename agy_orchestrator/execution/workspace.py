"""Per-candidate isolated workspaces.

For real K-parallel candidate generation, every candidate needs its own copy
of the operator's target directory so workers' file writes don't trample
each other. K agents writing to one cwd is last-writer-wins; the only honest
way to vote/rank K candidates is to grade them on independent disk state.

Two backends, picked automatically:

* **git worktree** — when the base directory is a git repo. ``git worktree
  add --detach`` creates a lightweight checkout sharing the same git db.
  Constant-time clone regardless of repo size, supports clean diff
  extraction via ``git diff HEAD``. The preferred path.

* **directory copy** — fallback for non-git directories. ``shutil.copytree``
  with an ignore list that skips heavy / regenerable trees (.venv,
  __pycache__, .pytest_cache, .mypy_cache, .ruff_cache, runs, node_modules,
  dist, build). Slower than worktree, but works on any directory and
  excludes only obvious bloat.

The ``candidate_workspace`` async context manager handles creation +
cleanup. Cleanup is bulletproof: a failure inside the with-block, an OS
error during teardown, or a worker that left files in a weird state all
still result in the temp dir being nuked.

``diff_workspace_against_base`` returns a unified diff of what the
candidate changed, so callers can rank candidates by their diff,
apply the winner's changes back to the operator's directory, and discard
the losers' workspaces entirely.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Tuple

logger = logging.getLogger(__name__)

# Directories that are heavy or regenerable; skipped in copy-mode so a
# clone doesn't drag a multi-GB .venv along for each candidate.
COPY_IGNORE_PATTERNS = (
    ".git",            # git's own dir; copy-mode workspaces aren't git repos
    ".venv", "venv",
    "__pycache__", "*.pyc",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "runs",            # harness's per-dispatch artifacts; orchestrator-internal
    "node_modules",
    "dist", "build",
)


def is_git_repo(path: Path) -> bool:
    """True iff ``path`` is the top of a git work tree.

    Falls open on any error — including git not installed — so the caller
    silently degrades to copy-mode rather than blowing up on environments
    without git."""
    if not path.is_dir():
        return False
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


async def _create_git_worktree(base: Path, target: Path) -> None:
    """``git worktree add --detach`` from base's HEAD into target.

    Detached so the candidate's commits — if any — don't move the base's
    branch. Raises if git fails so the caller can fall through to copy mode.
    """
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(base), "worktree", "add", "--detach", str(target),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _out, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"git worktree add failed (rc={proc.returncode}): {err.decode(errors='replace').strip()}"
        )


async def _remove_git_worktree(base: Path, target: Path) -> None:
    """Best-effort worktree teardown. Never raises — cleanup paths must not
    propagate errors that could mask the original failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", str(base), "worktree", "remove", "--force", str(target),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
    except Exception as exc:
        logger.debug("worktree remove failed (will rmtree anyway): %s", exc)


async def _copy_workspace(base: Path, target: Path) -> None:
    """``shutil.copytree`` with heavy-dir filtering. Runs in a thread so it
    doesn't block the event loop on multi-GB repos."""
    def _do_copy() -> None:
        shutil.copytree(
            base, target,
            ignore=shutil.ignore_patterns(*COPY_IGNORE_PATTERNS),
        )
    await asyncio.to_thread(_do_copy)


@asynccontextmanager
async def candidate_workspace(
    base_dir: Path,
    *,
    candidate_id: str = "candidate",
    prefer_worktree: bool = True,
) -> AsyncIterator[Tuple[Path, str]]:
    """Yield ``(workspace_path, backend)`` for an isolated copy of base_dir.

    ``backend`` is ``"worktree"`` or ``"copy"`` so the caller can route the
    diff extraction appropriately. The workspace exists for the duration of
    the with-block and is torn down on exit (success OR exception).

    Concurrent calls produce independent workspaces — each gets its own temp
    root with a unique prefix.
    """
    base = Path(base_dir).resolve()
    tmp_root = Path(tempfile.mkdtemp(prefix=f"agentorch-cand-{candidate_id}-"))
    workspace = tmp_root / "workspace"
    backend = "copy"
    worktree_active = False

    try:
        if prefer_worktree and is_git_repo(base):
            try:
                await _create_git_worktree(base, workspace)
                backend = "worktree"
                worktree_active = True
            except Exception as exc:
                logger.warning(
                    "candidate_workspace: git worktree failed for %s (%s); "
                    "falling back to directory copy", base, exc,
                )
        if not worktree_active:
            await _copy_workspace(base, workspace)
        yield workspace, backend
    finally:
        if worktree_active:
            await _remove_git_worktree(base, workspace)
        # Belt-and-braces: even if worktree remove "succeeded", nuke the
        # tmp root so we never leak. ignore_errors so a half-removed
        # worktree (git dropped some files but not all) can't crash us.
        shutil.rmtree(tmp_root, ignore_errors=True)


def diff_workspace_against_base(
    base: Path,
    workspace: Path,
    *,
    backend: str = "copy",
) -> str:
    """Return a unified diff of changes the candidate made.

    For worktree backend: ``git diff HEAD`` from inside the workspace —
    fast and exact. For copy backend: delegate to the harness's
    git-independent snapshot/diff (slower but covers any tree).

    Returns the empty string when there are no changes."""
    workspace = Path(workspace)
    if backend == "worktree":
        try:
            proc = subprocess.run(
                ["git", "-C", str(workspace), "diff", "HEAD"],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                return proc.stdout
            logger.warning("git diff failed in workspace: %s", proc.stderr.strip())
        except Exception as exc:
            logger.warning("git diff raised in workspace: %s", exc)
    # Copy backend, or git-diff fallthrough: snapshot the two trees and diff.
    # Imported here to avoid a hard dependency cycle: agy_orchestrator is the
    # lower layer, harness is the upper. The import is lazy and only fires
    # when someone actually requests a diff in copy mode.
    from harness.snapshot import diff_snapshots, take_snapshot
    before = take_snapshot(Path(base))
    after = take_snapshot(workspace)
    return diff_snapshots(before, after).unified or ""

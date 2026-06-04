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
from typing import AsyncIterator, Dict, Optional, Tuple

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


def has_uncommitted_changes(path: Path) -> bool:
    """True iff the git work tree has uncommitted tracked changes OR untracked,
    non-ignored files (i.e. ``git status --porcelain`` is non-empty).

    This is the dirty-tree signal that makes a ``git worktree`` workspace
    UNSAFE for vote (issue #36): ``git worktree add --detach`` checks out
    HEAD only, so a candidate worktree silently OMITS the operator's
    uncommitted edits and untracked files — candidates then build on stale
    committed state, and mirroring the (HEAD-based) winner back over the
    operator's tree DESTROYS those uncommitted/untracked changes. Copy-mode
    has neither problem (it snapshots the actual working tree, ignored dirs
    excluded), so the caller routes dirty git repos to copy-mode.

    Ignored files (``.venv``, ``runs`` …) do NOT count as dirty — porcelain
    respects ``.gitignore`` — so a clean repo with a git-ignored venv still
    gets the fast worktree path. Falls open to False (treat as clean) on any
    error so a git hiccup can't wedge vote."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


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


def _prune_git_worktrees(base: Path) -> None:
    """Best-effort reconcile of the base repo's ``.git/worktrees`` registry.

    ``git worktree remove`` is the clean path, but when it is skipped or fails
    (process killed mid-teardown, a locked worktree, or a worktree whose dir we
    rmtree'd directly), the base repo keeps a stale ``.git/worktrees/<name>``
    admin entry. On a long, many-node DAG build these accumulate unboundedly and
    eventually slow / clutter the repo. ``git worktree prune`` drops every admin
    entry whose working tree no longer exists on disk — which, since we always
    rmtree the tmp-root in the cleanup ``finally`` first, is exactly the entry we
    just tore down. Synchronous (not awaited) so it still runs inside a
    cancellation-shielded teardown, and swallows every error (cleanup must never
    raise / mask the original failure or block on a missing git)."""
    try:
        subprocess.run(
            ["git", "-C", str(base), "worktree", "prune"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as exc:  # FileNotFoundError, TimeoutExpired, OSError, …
        logger.debug("worktree prune failed (non-fatal): %s", exc)


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
        use_worktree = prefer_worktree and is_git_repo(base)
        if use_worktree and has_uncommitted_changes(base):
            # Dirty tree: a worktree would check out HEAD only, dropping the
            # operator's uncommitted/untracked work and (on winner-apply)
            # destroying it (issue #36). Copy-mode snapshots the real working
            # tree instead, so candidates build on what the operator actually
            # has and the winner can be applied without clobbering their
            # changes. The fast worktree path is reserved for clean repos.
            logger.info(
                "candidate_workspace: %s has uncommitted/untracked changes; "
                "using copy-mode (not worktree) so candidates reflect the real "
                "working tree and vote can't discard uncommitted work (#36)",
                base,
            )
            use_worktree = False
        if use_worktree:
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
        # Shield the rmtree from a teardown interruption: if the worktree
        # remove await re-raises (a CancelledError because the run was killed
        # mid-cleanup, or any other error), an unguarded ``finally`` would
        # propagate BEFORE the belt-and-braces rmtree, leaking the tmp-root on
        # disk. Nesting the rmtree in its own ``finally`` guarantees it always
        # runs; we re-raise the original interruption afterward so the caller
        # still sees the cancellation.
        try:
            if worktree_active:
                await _remove_git_worktree(base, workspace)
        finally:
            # Belt-and-braces: even if worktree remove "succeeded", nuke the
            # tmp root so we never leak. ignore_errors so a half-removed
            # worktree (git dropped some files but not all) can't crash us.
            shutil.rmtree(tmp_root, ignore_errors=True)
            # Reconcile the base repo's worktree registry: if the per-node
            # ``git worktree remove`` was skipped/failed (killed mid-teardown,
            # locked worktree) we just rmtree'd the working tree out from under a
            # surviving ``.git/worktrees/<name>`` admin entry. Prune drops every
            # such orphaned entry so they don't accumulate across a long DAG run.
            # Synchronous + error-swallowing, so it runs even under cancellation.
            if worktree_active:
                _prune_git_worktrees(base)


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


class _ManagedWorkspace:
    """One isolated workspace's lifecycle, managed manually so its writes can be
    applied back to base_dir BEFORE all workspaces are torn down.

    Shared by ``vote`` (per-candidate isolation) and the master graph walker
    (per-DAG-node isolation) so both modes use ONE isolation implementation
    (docs §3.5): worktree-or-copy with the #36 dirty-tree copy fallback, all
    living in :func:`candidate_workspace`.
    """

    def __init__(self, base_dir: str, candidate_id: str, prefer_worktree: bool = True):
        self._cm = candidate_workspace(
            Path(base_dir),
            candidate_id=candidate_id,
            prefer_worktree=prefer_worktree,
        )
        self.path: Optional[Path] = None
        self.backend: Optional[str] = None

    async def open(self) -> "_ManagedWorkspace":
        self.path, self.backend = await self._cm.__aenter__()
        return self

    async def close(self) -> None:
        try:
            await self._cm.__aexit__(None, None, None)
        except Exception as exc:
            logger.debug("workspace close failed (already torn down?): %s", exc)


def _snapshot_tree(root: Path) -> Dict[str, bytes]:
    """Map every non-ignored regular file under ``root`` to its bytes.

    Skips the heavy/regenerable top-level dirs the clone skips (.git, runs, …).
    Used by the graph walker to capture a DAG node's workspace at open so a
    later apply can diff against it and write back ONLY what the node changed
    (not a whole-tree mirror that would clobber a concurrent sibling's writes).
    """
    snapshot: Dict[str, bytes] = {}
    root = Path(root)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        top = rel.parts[0]
        if top in COPY_IGNORE_PATTERNS or top.startswith(".git"):
            continue
        try:
            snapshot[str(rel)] = path.read_bytes()
        except OSError:
            continue
    return snapshot


def _clear_path_conflicts(dst_path: Path, dst_root: Path) -> None:
    """Remove anything on dst that blocks writing a regular file at ``dst_path``.

    A file<->directory refactor (e.g. ``mod.py`` becomes the package
    ``mod.py/__init__.py``, or vice versa) leaves the destination with a node
    of the WRONG type along the path we're about to create:

    * an intermediate component (``mod.py``) is a regular FILE on dst but must
      become a directory — ``mkdir(parents=True)`` would raise FileExistsError;
    * the final ``dst_path`` is a DIRECTORY on dst but the winner wants a regular
      file there — ``write_bytes``/``copy2`` would raise IsADirectoryError, and
      naively copying a file INTO that dir then unlinking it loses the write.

    Both are resolved by removing the conflicting node (rmtree for dirs, unlink
    for files) so the subsequent ``mkdir(parents=True, exist_ok=True)`` + write
    lands the winner's file at the intended path. Never escapes ``dst_root``.
    """
    # Walk each intermediate parent (excluding dst_root): if it exists as a
    # non-directory, it blocks mkdir — remove it.
    parents = []
    p = dst_path.parent
    while p != dst_root and dst_root in p.parents:
        parents.append(p)
        p = p.parent
    for parent in reversed(parents):  # outermost first
        if parent.exists() and not parent.is_dir():
            try:
                parent.unlink()
            except OSError:
                pass
    # The final target itself: a directory where we want a file must go.
    if dst_path.is_dir() and not dst_path.is_symlink():
        shutil.rmtree(dst_path, ignore_errors=True)


async def apply_node_writes(src: Path, dst: Path, base_snapshot: Dict[str, bytes]) -> None:
    """Apply ONLY the files a DAG node changed (vs its open-time snapshot) to dst.

    Unlike :func:`apply_workspace` (a whole-tree mirror that DELETES dst files
    absent from src — right for vote's single winner), this writes back just the
    node's own diff: files added/modified relative to ``base_snapshot`` are
    copied, files the node deleted are removed. Files the node never touched are
    left alone, so a concurrently-finishing sibling's disjoint writes survive
    (docs §5 M3). Overlap detection (two nodes touching the SAME path) is M4;
    here it's last-writer-wins.

    Runs in a thread (file I/O can block the loop on large writes).
    """
    src = Path(src)
    dst = Path(dst)

    def _do_apply() -> None:
        current = _snapshot_tree(src)
        # Added or modified vs the node's base: copy into the live tree.
        for rel, data in current.items():
            if base_snapshot.get(rel) == data:
                continue  # unchanged by this node — don't touch the live tree
            dst_path = dst / rel
            if dst_path.is_file():
                try:
                    if dst_path.read_bytes() == data:
                        continue
                except OSError:
                    pass
            # A file<->dir refactor can leave a wrong-typed node blocking this
            # write (intermediate file where we need a dir, or a dir where we
            # need a file). Clear it so mkdir + write can't raise.
            _clear_path_conflicts(dst_path, dst)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            dst_path.write_bytes(data)
        # Files the node DELETED (present at open, gone now): remove from dst.
        for rel in base_snapshot:
            if rel in current:
                continue
            dst_path = dst / rel
            if not dst_path.exists():
                continue
            try:
                dst_path.unlink()
            except OSError:
                continue
            parent = dst_path.parent
            while parent != dst:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent

    await asyncio.to_thread(_do_apply)


async def apply_workspace(src: Path, dst: Path) -> None:
    """Mirror a workspace's contents over the operator's actual work_dir.

    Walks src for regular files, copies any whose bytes differ from the
    destination's. Skips the same heavy/regenerable tree list the workspace
    clone uses (.git, __pycache__, runs, …) so an unintended meta-file can't
    sneak into work_dir. Files that no longer exist in src are removed from dst.

    Shared by vote (winner-apply) and the master graph walker (per-node
    disjoint-write apply, docs §5 M3). Runs in a thread because shutil.copy2 +
    large diffs can block the loop noticeably on multi-MB writes.
    """

    def _do_apply() -> None:
        for src_path in src.rglob("*"):
            if not src_path.is_file():
                continue
            rel = src_path.relative_to(src)
            # Skip ignored top-level dirs (the clone wouldn't have included
            # these, but in worktree mode .git lives inside; double-check at
            # the apply boundary).
            top = rel.parts[0]
            if top in COPY_IGNORE_PATTERNS or top.startswith(".git"):
                continue
            dst_path = dst / rel
            # Only copy when content actually differs — avoids touching mtimes
            # on identical files and gives a cleaner diff if the operator
            # inspects work_dir after.
            if dst_path.is_file():
                try:
                    if dst_path.read_bytes() == src_path.read_bytes():
                        continue
                except OSError:
                    pass
            # A file<->dir refactor can leave a wrong-typed node blocking this
            # write (intermediate file where we need a dir, or a dir where we
            # need a file). Clear it so mkdir + copy2 can't raise / misplace.
            _clear_path_conflicts(dst_path, dst)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)

        # Remove files that no longer exist in the source workspace.
        for dst_path in dst.rglob("*"):
            if not dst_path.is_file():
                continue
            rel = dst_path.relative_to(dst)
            top = rel.parts[0]
            if top in COPY_IGNORE_PATTERNS or top.startswith(".git"):
                continue
            if (src / rel).exists():
                continue
            try:
                dst_path.unlink()
            except OSError:
                continue
            parent = dst_path.parent
            while parent != dst:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent

    await asyncio.to_thread(_do_apply)

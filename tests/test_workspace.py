"""Per-candidate workspace isolation.

Unlocks real K-parallel candidate generation: each candidate gets its own
copy of base_dir so writes don't trample each other. Worktree mode for git
repos (cheap), copy mode for plain directories (fallback).
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from agy_orchestrator.execution.workspace import (
    candidate_workspace,
    diff_workspace_against_base,
    is_git_repo,
)


def _run(coro):
    return asyncio.run(coro)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True, capture_output=True, text=True,
    )


def _make_git_repo(root: Path, files: dict[str, str]) -> None:
    """Initialize a git repo at root with the given files in an initial commit."""
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@test")
    _git(root, "config", "user.name", "Test")
    for relpath, content in files.items():
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")


# --- is_git_repo --- #

def test_is_git_repo_detects_real_repo(tmp_path):
    _make_git_repo(tmp_path, {"hello.txt": "hi"})
    assert is_git_repo(tmp_path) is True


def test_is_git_repo_false_for_plain_dir(tmp_path):
    (tmp_path / "hello.txt").write_text("hi")
    assert is_git_repo(tmp_path) is False


def test_is_git_repo_false_for_nonexistent_path(tmp_path):
    assert is_git_repo(tmp_path / "nope") is False


# --- worktree backend --- #

def test_worktree_backend_for_git_repo(tmp_path):
    """Git-backed base -> backend is 'worktree' and the workspace is its
    own checkout."""
    _make_git_repo(tmp_path, {"src/main.py": "print('base')\n"})

    async def _scenario():
        async with candidate_workspace(tmp_path) as (ws, backend):
            assert backend == "worktree"
            assert ws.exists()
            assert (ws / "src" / "main.py").exists()
            # The worktree is a real git checkout pointing at the same db.
            assert (ws / ".git").exists()  # may be a file (gitlink) not a dir
            return ws

    ws_seen = _run(_scenario())
    # After the context manager exits, the workspace must be gone.
    assert not ws_seen.exists()


def test_worktree_isolates_writes_from_base(tmp_path):
    """A write inside the workspace must NOT appear in the base."""
    _make_git_repo(tmp_path, {"src/main.py": "print('base')\n"})

    async def _scenario():
        async with candidate_workspace(tmp_path) as (ws, _backend):
            (ws / "candidate-only.txt").write_text("only in candidate")
            (ws / "src" / "main.py").write_text("print('modified')\n")
            # While inside the with-block, base is untouched.
            assert not (tmp_path / "candidate-only.txt").exists()
            assert (tmp_path / "src" / "main.py").read_text() == "print('base')\n"

    _run(_scenario())
    # And still untouched after teardown.
    assert not (tmp_path / "candidate-only.txt").exists()
    assert (tmp_path / "src" / "main.py").read_text() == "print('base')\n"


def test_worktree_diff_against_base(tmp_path):
    """`diff_workspace_against_base` in worktree mode uses git diff HEAD."""
    _make_git_repo(tmp_path, {"src/main.py": "print('base')\n"})

    diff_captured: list[str] = []

    async def _scenario():
        async with candidate_workspace(tmp_path) as (ws, backend):
            (ws / "src" / "main.py").write_text("print('modified')\n")
            (ws / "newfile.py").write_text("# new\n")
            # New files aren't part of HEAD; git diff HEAD only shows
            # tracked-file changes, so we must `git add` the new file
            # to see it in the diff. Mirrors how a worker would commit
            # mid-run if it wanted to.
            _git(ws, "add", "newfile.py")
            diff_captured.append(
                diff_workspace_against_base(tmp_path, ws, backend=backend)
            )

    _run(_scenario())
    diff = diff_captured[0]
    assert "print('modified')" in diff
    assert "newfile.py" in diff
    assert "print('base')" in diff


def test_no_changes_yields_empty_diff(tmp_path):
    """A workspace where nothing changed produces an empty diff."""
    _make_git_repo(tmp_path, {"hello.txt": "hi"})

    captured: list[str] = []

    async def _scenario():
        async with candidate_workspace(tmp_path) as (ws, backend):
            captured.append(
                diff_workspace_against_base(tmp_path, ws, backend=backend)
            )

    _run(_scenario())
    assert captured[0] == ""


# --- copy backend --- #

def test_copy_backend_for_plain_dir(tmp_path):
    """Non-git base -> backend is 'copy' and the tree is duplicated."""
    base = tmp_path / "src"
    base.mkdir()
    (base / "main.py").write_text("print('base')\n")
    (base / "data" / "x.txt").parent.mkdir()
    (base / "data" / "x.txt").write_text("data\n")

    async def _scenario():
        async with candidate_workspace(base) as (ws, backend):
            assert backend == "copy"
            assert (ws / "main.py").read_text() == "print('base')\n"
            assert (ws / "data" / "x.txt").read_text() == "data\n"
            return ws

    ws_seen = _run(_scenario())
    assert not ws_seen.exists()


def test_copy_backend_skips_heavy_dirs(tmp_path):
    """Copy mode must NOT drag .venv / __pycache__ / runs / .git / etc."""
    base = tmp_path / "src"
    base.mkdir()
    (base / "main.py").write_text("ok\n")
    # Heavy dirs that the ignore list should skip:
    for d in (".venv", "__pycache__", ".pytest_cache", "runs", "node_modules"):
        (base / d).mkdir()
        (base / d / "junk.bin").write_text("x" * 1000)
    # A would-be .git dir (this dir isn't actually a git repo since there's
    # no commit history, so is_git_repo returns False -> copy mode).
    (base / ".git").mkdir()
    (base / ".git" / "junk").write_text("y")

    async def _scenario():
        async with candidate_workspace(base) as (ws, backend):
            assert backend == "copy"
            assert (ws / "main.py").exists()
            for d in (".venv", "__pycache__", ".pytest_cache", "runs", "node_modules", ".git"):
                assert not (ws / d).exists(), f"{d} should have been skipped"

    _run(_scenario())


def test_copy_isolates_writes_from_base(tmp_path):
    """Same isolation guarantee as worktree mode."""
    base = tmp_path / "src"
    base.mkdir()
    (base / "main.py").write_text("print('base')\n")

    async def _scenario():
        async with candidate_workspace(base) as (ws, _backend):
            (ws / "main.py").write_text("print('modified')\n")
            (ws / "new.txt").write_text("new")

    _run(_scenario())
    assert (base / "main.py").read_text() == "print('base')\n"
    assert not (base / "new.txt").exists()


def test_copy_backend_diff(tmp_path):
    """In copy mode, diff falls back to the harness's snapshot-based diff."""
    base = tmp_path / "src"
    base.mkdir()
    (base / "main.py").write_text("print('base')\n")

    captured: list[str] = []

    async def _scenario():
        async with candidate_workspace(base) as (ws, backend):
            (ws / "main.py").write_text("print('modified')\n")
            captured.append(
                diff_workspace_against_base(base, ws, backend=backend)
            )

    _run(_scenario())
    diff = captured[0]
    # Snapshot diff produces unified-diff text — exact format depends on
    # the snapshot module, but at minimum the modified content should
    # show up somewhere.
    assert "modified" in diff or "main.py" in diff


# --- isolation across K parallel candidates --- #

def test_k_parallel_workspaces_are_independent(tmp_path):
    """The whole point: K simultaneous candidates each get their own
    isolated workspace; their writes don't interfere with each other or
    with the base."""
    _make_git_repo(tmp_path, {"shared.py": "VERSION = 0\n"})

    async def _candidate(i: int) -> str:
        async with candidate_workspace(tmp_path, candidate_id=f"c{i}") as (ws, _b):
            # Each candidate writes a different value AND a unique marker.
            (ws / "shared.py").write_text(f"VERSION = {i}\n")
            (ws / f"marker_{i}.txt").write_text(f"candidate {i} was here")
            await asyncio.sleep(0.01)  # encourage interleaving
            # Read back our own write — must reflect what this candidate wrote.
            return (ws / "shared.py").read_text()

    async def _scenario():
        return await asyncio.gather(*[_candidate(i) for i in range(5)])

    results = _run(_scenario())
    # Each candidate must observe its own write, not a peer's.
    assert results == ["VERSION = 0\n", "VERSION = 1\n", "VERSION = 2\n",
                        "VERSION = 3\n", "VERSION = 4\n"]
    # Base is untouched.
    assert (tmp_path / "shared.py").read_text() == "VERSION = 0\n"
    for i in range(5):
        assert not (tmp_path / f"marker_{i}.txt").exists()


# --- cleanup robustness --- #

def test_cleanup_runs_even_on_exception(tmp_path):
    """An exception inside the with-block must still trigger teardown."""
    _make_git_repo(tmp_path, {"x.py": "x"})

    seen_paths: list[Path] = []

    async def _scenario():
        with pytest.raises(RuntimeError, match="boom"):
            async with candidate_workspace(tmp_path) as (ws, _b):
                seen_paths.append(ws)
                assert ws.exists()
                raise RuntimeError("boom")

    _run(_scenario())
    # Workspace was created, then nuked on teardown despite the exception.
    assert seen_paths and not seen_paths[0].exists()


def test_concurrent_workspaces_have_unique_temp_roots(tmp_path):
    """Two simultaneous workspaces from the same base must not collide on
    a shared tempdir prefix."""
    _make_git_repo(tmp_path, {"x.py": "x"})

    async def _open_one(label: str) -> Path:
        async with candidate_workspace(tmp_path, candidate_id=label) as (ws, _b):
            await asyncio.sleep(0.05)
            return ws

    async def _scenario():
        return await asyncio.gather(_open_one("a"), _open_one("b"))

    a, b = _run(_scenario())
    # Different paths (parent tmp roots differ).
    assert a != b
    assert a.parent != b.parent


def test_prefer_worktree_false_forces_copy(tmp_path):
    """Operator escape hatch: even on a git repo, prefer_worktree=False
    must use the copy backend. Useful when worktree paths interact poorly
    with downstream tools."""
    _make_git_repo(tmp_path, {"x.py": "x"})

    async def _scenario():
        async with candidate_workspace(tmp_path, prefer_worktree=False) as (_ws, backend):
            assert backend == "copy"

    _run(_scenario())

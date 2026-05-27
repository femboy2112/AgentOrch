"""Git-independent workspace snapshotting + diffing.

The harness records exactly what a worker changed on disk by snapshotting the
tree before and after a dispatch and diffing the two snapshots. This works
whether or not the directory is a git repo (workers run before `git init` too),
and never depends on staging state.
"""
from __future__ import annotations

import difflib
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# Directories never worth snapshotting (volatile, generated, or our own logs).
IGNORE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", ".venv", "venv", ".idea", ".vscode", "runs", "work",
    ".tlica_state", "dist", "build", ".egg-info",
}
# Files larger than this aren't diffed line-by-line (content not retained).
MAX_DIFF_BYTES = 1_000_000


@dataclass
class FileEntry:
    sha256: str
    size: int
    is_text: bool
    content: str | None  # retained only for text files <= MAX_DIFF_BYTES


@dataclass
class Snapshot:
    root: Path
    files: Dict[str, FileEntry] = field(default_factory=dict)


def _looks_text(blob: bytes) -> bool:
    if b"\x00" in blob:
        return False
    try:
        blob.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def take_snapshot(root: str | Path) -> Snapshot:
    """Walk ``root`` and record a hash (+ text content) for every file."""
    root = Path(root).resolve()
    snap = Snapshot(root=root)
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in place so os.walk skips them.
        dirnames[:] = [
            d for d in dirnames
            if d not in IGNORE_DIRS and not d.endswith(".egg-info")
        ]
        for name in filenames:
            fpath = Path(dirpath) / name
            try:
                blob = fpath.read_bytes()
            except OSError:
                continue
            rel = str(fpath.relative_to(root))
            is_text = _looks_text(blob)
            content = None
            if is_text and len(blob) <= MAX_DIFF_BYTES:
                content = blob.decode("utf-8")
            snap.files[rel] = FileEntry(
                sha256=hashlib.sha256(blob).hexdigest(),
                size=len(blob),
                is_text=is_text,
                content=content,
            )
    return snap


@dataclass
class DiffResult:
    added: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    unified: str = ""

    @property
    def changed(self) -> List[str]:
        return sorted(self.added + self.modified + self.deleted)

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.modified or self.deleted)


def diff_snapshots(before: Snapshot, after: Snapshot) -> DiffResult:
    """Compute added/modified/deleted files and a unified textual diff."""
    res = DiffResult()
    before_files, after_files = before.files, after.files

    for rel in sorted(after_files):
        if rel not in before_files:
            res.added.append(rel)
        elif after_files[rel].sha256 != before_files[rel].sha256:
            res.modified.append(rel)
    for rel in sorted(before_files):
        if rel not in after_files:
            res.deleted.append(rel)

    chunks: List[str] = []
    for rel in res.added:
        chunks.append(_one_diff(rel, None, after_files[rel]))
    for rel in res.modified:
        chunks.append(_one_diff(rel, before_files[rel], after_files[rel]))
    for rel in res.deleted:
        chunks.append(_one_diff(rel, before_files[rel], None))
    res.unified = "\n".join(c for c in chunks if c)
    return res


def _one_diff(rel: str, old: FileEntry | None, new: FileEntry | None) -> str:
    # Binary or oversized file: just note the transition, no line diff.
    old_text = old.content if old else None
    new_text = new.content if new else None
    if (old and old.content is None) or (new and new.content is None):
        if old is None:
            return f"### ADDED (binary) {rel} ({new.size} bytes)"
        if new is None:
            return f"### DELETED (binary) {rel}"
        return f"### MODIFIED (binary) {rel} ({old.size} -> {new.size} bytes)"

    old_lines = (old_text or "").splitlines(keepends=True)
    new_lines = (new_text or "").splitlines(keepends=True)
    label = "ADDED" if old is None else "DELETED" if new is None else "MODIFIED"
    udiff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{rel}", tofile=f"b/{rel}",
        lineterm="",
    )
    body = "\n".join(udiff)
    return f"### {label} {rel}\n{body}"

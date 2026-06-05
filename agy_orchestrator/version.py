"""Single source of truth for the orchestrator's runtime version string.

Both CLIs (`python -m agy_orchestrator --version` and `python -m harness
--version`) print :func:`version_string`. The point is operational: several
orchestrator processes may be alive at once (a live build, a stale one a long
dispatch is still running under, a checkout on another branch), and the bare
package version `0.1.0` can't tell them apart. So the string also carries the
git commit it's running from plus a `+dirty` marker for uncommitted edits — that
is what actually identifies *which* build is alive.

`__version__` is the canonical runtime version; keep it in sync with
`pyproject.toml`'s `version` (packaging reads that, the running process reads
this).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

__version__ = "0.1.0"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_revision() -> str | None:
    """Short SHA of HEAD (+ ``+dirty`` if the tree has uncommitted changes).

    Best-effort: returns ``None`` when git is absent or this isn't a checkout
    (e.g. an installed wheel), so the caller falls back to the bare version.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=2,
        )
        if head.returncode != 0:
            return None
        sha = head.stdout.strip()
        if not sha:
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=2,
        )
        dirty = status.returncode == 0 and bool(status.stdout.strip())
        return f"{sha}+dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return None


def version_string(prog: str = "agy-orchestrator") -> str:
    """Human-readable version line, e.g. ``harness 0.1.0 (9e849f8)``."""
    rev = _git_revision()
    base = f"{prog} {__version__}"
    return f"{base} ({rev})" if rev else base

"""Regression: apply_workspace / apply_node_writes survive a file<->directory
refactor on the same path (e.g. ``mod.py`` becoming the package
``mod.py/__init__.py`` or vice versa).

Before the fix, ``apply_workspace`` raised FileExistsError on
``dst_path.parent.mkdir`` when an intermediate path component existed in dst as
a regular file (file->package refactor), leaving work_dir half-applied and the
exception escaping ``VoteWorkflow.execute()`` (winner already chosen,
verified=True). The inverse (dir->file) silently lost the winner's file: copy2
dropped it INTO the stale directory which the removal loop then deleted.
"""
import asyncio
import tempfile
from pathlib import Path

import pytest

from agy_orchestrator.execution.workspace import apply_node_writes, apply_workspace


def _mk(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def test_apply_workspace_file_to_package_no_crash():
    src, dst = _mk("src-"), _mk("dst-")
    # src: mod.py became a package directory.
    (src / "mod.py").mkdir()
    (src / "mod.py" / "__init__.py").write_text("# package\n")
    # dst: mod.py is still a regular file.
    (dst / "mod.py").write_text("print('old')\n")

    asyncio.run(apply_workspace(src, dst))

    target = dst / "mod.py" / "__init__.py"
    assert target.is_file()
    assert target.read_text() == "# package\n"


def test_apply_workspace_package_to_file_lands_correctly():
    src, dst = _mk("src-"), _mk("dst-")
    # src: package collapsed back to a single module file.
    (src / "mod.py").write_text("print('new single module')\n")
    # dst: mod.py is currently a package directory.
    (dst / "mod.py").mkdir()
    (dst / "mod.py" / "__init__.py").write_text("# old package\n")

    asyncio.run(apply_workspace(src, dst))

    target = dst / "mod.py"
    assert target.is_file(), "winner's file must land at the intended path"
    assert target.read_text() == "print('new single module')\n"


def test_apply_node_writes_file_to_package_no_crash():
    src, dst = _mk("src-"), _mk("dst-")
    (src / "mod.py").mkdir()
    (src / "mod.py" / "__init__.py").write_text("# package\n")
    (dst / "mod.py").write_text("print('old')\n")
    # base_snapshot = the node's open-time view (mod.py was a file).
    base_snapshot = {"mod.py": b"print('old')\n"}

    asyncio.run(apply_node_writes(src, dst, base_snapshot))

    assert (dst / "mod.py" / "__init__.py").read_text() == "# package\n"


def test_apply_workspace_deep_intermediate_file_conflict():
    # An intermediate component two levels up is a file in dst.
    src, dst = _mk("src-"), _mk("dst-")
    (src / "pkg" / "sub").mkdir(parents=True)
    (src / "pkg" / "sub" / "leaf.py").write_text("leaf\n")
    # dst has a regular file where 'pkg' must become a directory.
    (dst / "pkg").write_text("i am a file\n")

    asyncio.run(apply_workspace(src, dst))

    assert (dst / "pkg" / "sub" / "leaf.py").read_text() == "leaf\n"


def test_apply_workspace_plain_files_still_mirror():
    # Sanity: ordinary same-type updates + deletions still behave.
    src, dst = _mk("src-"), _mk("dst-")
    (src / "keep.txt").write_text("new\n")
    (src / "added.txt").write_text("added\n")
    (dst / "keep.txt").write_text("old\n")
    (dst / "gone.txt").write_text("delete me\n")

    asyncio.run(apply_workspace(src, dst))

    assert (dst / "keep.txt").read_text() == "new\n"
    assert (dst / "added.txt").read_text() == "added\n"
    assert not (dst / "gone.txt").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

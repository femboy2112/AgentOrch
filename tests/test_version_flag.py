"""`--version` on both CLIs prints a build identifier and exits 0.

The string must carry enough to tell *which* build is alive at runtime — the
package version AND, when run from a checkout, the git commit (+dirty marker) —
not just the static `0.1.0`.
"""
from __future__ import annotations

import re

import pytest

from agy_orchestrator import version as ver
from agy_orchestrator.cli import main as agy_main
from harness.cli import main as harness_main


def test_version_string_includes_prog_and_version():
    s = ver.version_string("harness")
    assert s.startswith("harness 0.1.0")


def test_version_string_appends_revision_when_available(monkeypatch):
    monkeypatch.setattr(ver, "_git_revision", lambda: "abc1234+dirty")
    assert ver.version_string("agy-orchestrator") == "agy-orchestrator 0.1.0 (abc1234+dirty)"


def test_version_string_bare_when_no_git(monkeypatch):
    monkeypatch.setattr(ver, "_git_revision", lambda: None)
    assert ver.version_string("harness") == "harness 0.1.0"


def test_git_revision_never_raises(monkeypatch):
    # Best-effort: a broken/absent git must degrade to None, never blow up.
    def _boom(*a, **k):
        raise OSError("no git here")

    monkeypatch.setattr(ver.subprocess, "run", _boom)
    assert ver._git_revision() is None


@pytest.mark.parametrize(
    "main, argv0, prog",
    [(harness_main, "harness", "harness"), (agy_main, "agy_orchestrator", "agy-orchestrator")],
)
def test_cli_version_flag_prints_and_exits_zero(main, argv0, prog, capsys, monkeypatch):
    # argparse's version action exits(0) BEFORE the required-subcommand check,
    # so `--version` alone (no subcommand) must succeed, not error. Both CLIs
    # read sys.argv (harness via argv=None default), so patch it uniformly.
    monkeypatch.setattr("sys.argv", [argv0, "--version"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert re.match(rf"^{re.escape(prog)} \d+\.\d+\.\d+", out)

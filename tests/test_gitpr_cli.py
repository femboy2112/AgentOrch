"""Phase 4 — git-pr decision CLI verbs: `harness pr / merge / abandon`.

Hermetic: a pr_session.json is planted under a monkeypatched RUNS_DIR and `gh`
is stubbed on PATH (logs argv as JSON). No network.
"""
from __future__ import annotations

import json
import os

import pytest

from harness import cli
from harness import gitpr


_GH_STUB = """#!/usr/bin/env python3
import os, sys, json
args = sys.argv[1:]
log = os.environ.get("GH_STUB_LOG")
if log:
    with open(log, "a") as f:
        f.write(json.dumps(args) + "\\n")
if args[:2] in (["pr", "merge"], ["pr", "close"], ["pr", "ready"]):
    sys.exit(0)
if args[:2] == ["auth", "status"]:
    sys.exit(0)
sys.stderr.write("gh stub: unhandled %r\\n" % (args,))
sys.exit(2)
"""


@pytest.fixture
def gh_stub(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "gh"
    stub.write_text(_GH_STUB, encoding="utf-8")
    os.chmod(stub, 0o755)
    log = tmp_path / "gh_calls.log"
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("GH_STUB_LOG", str(log))
    return log


def _gh_calls(log):
    if not log.exists():
        return []
    return [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()]


def _plant_session(tmp_path, monkeypatch, run_id, **overrides):
    runs = tmp_path / "runs"
    monkeypatch.setattr(cli, "RUNS_DIR", runs)
    run_dir = runs / run_id
    fields = dict(
        run_id=run_id, base_branch="main", temp_branch=f"agentorch/{run_id}",
        base_sha="abc123", work_dir="/gone", target_repo=str(tmp_path),
        status="awaiting_decision",
        pr_url="https://github.com/acme/repo/pull/9", pr_number=9, draft=True,
    )
    fields.update(overrides)
    sess = gitpr.PrSession(**fields)
    sess.commits.append({"step": 1, "sha": "deadbeef00", "title": "do x",
                         "outcome": "verified"})
    gitpr.save_session(run_dir, sess)
    return run_dir


def test_cmd_pr_shows_session(tmp_path, monkeypatch, capsys):
    _plant_session(tmp_path, monkeypatch, "r1")
    assert cli.main(["pr", "r1"]) == 0
    out = capsys.readouterr().out
    assert "agentorch/r1" in out
    assert "pull/9" in out
    assert "do x" in out
    assert "awaiting_decision" in out


def test_cmd_pr_missing_session(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "runs")
    assert cli.main(["pr", "nope"]) == 1
    assert "no git-pr session" in capsys.readouterr().err


def test_cmd_merge(tmp_path, monkeypatch, gh_stub, capsys):
    run_dir = _plant_session(tmp_path, monkeypatch, "r2")
    assert cli.main(["merge", "r2", "--method", "squash", "--delete-branch"]) == 0
    sess = gitpr.load_session(run_dir)
    assert sess.status == "merged" and sess.decision == "merge"
    m = [c for c in _gh_calls(gh_stub) if c[:2] == ["pr", "merge"]][0]
    assert "9" in m and "--squash" in m and "--delete-branch" in m


def test_cmd_merge_no_pr(tmp_path, monkeypatch, capsys):
    _plant_session(tmp_path, monkeypatch, "r3", pr_url=None, pr_number=None,
                   status="branch_ready")
    assert cli.main(["merge", "r3"]) == 1
    assert "no open PR" in capsys.readouterr().err


def test_cmd_abandon(tmp_path, monkeypatch, gh_stub, capsys):
    run_dir = _plant_session(tmp_path, monkeypatch, "r4")
    assert cli.main(["abandon", "r4", "--delete-branch"]) == 0
    sess = gitpr.load_session(run_dir)
    assert sess.status == "abandoned" and sess.decision == "abandon"
    close = [c for c in _gh_calls(gh_stub) if c[:2] == ["pr", "close"]][0]
    assert "9" in close and "--delete-branch" in close


def test_cmd_abandon_without_pr_still_marks(tmp_path, monkeypatch, capsys):
    run_dir = _plant_session(tmp_path, monkeypatch, "r5", pr_url=None,
                             pr_number=None, status="branch_ready")
    # No PR -> no gh call needed, but the session is still marked abandoned.
    assert cli.main(["abandon", "r5"]) == 0
    assert gitpr.load_session(run_dir).status == "abandoned"

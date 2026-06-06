"""Hermetic tests for harness/gitpr.py — the never-wedge git+gh layer behind
``--git-pr`` mode.

Everything runs against a throwaway ``git init`` repo in ``tmp_path`` and a
stubbed ``gh`` binary prepended to ``PATH`` (it logs its argv and emits a fake
PR URL). No network, no real GitHub, no mutation of the developer's checkout.
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from harness import gitpr


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _g(repo, *args):
    cp = subprocess.run(["git", "-C", str(repo), *args],
                        capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr
    return cp.stdout


@pytest.fixture
def repo(tmp_path):
    base = tmp_path / "repo"
    base.mkdir()
    _g(base, "init", "-q", "-b", "main")
    _g(base, "config", "user.email", "dev@example.com")
    _g(base, "config", "user.name", "Dev")
    (base / "README.md").write_text("hello\n", encoding="utf-8")
    _g(base, "add", "-A")
    _g(base, "commit", "-q", "-m", "init")
    return base


_GH_STUB = """#!/usr/bin/env python3
import os, sys, json
args = sys.argv[1:]
log = os.environ.get("GH_STUB_LOG")
if log:
    with open(log, "a") as f:
        f.write("\\x1f".join(args) + "\\n")
if os.environ.get("GH_STUB_FAIL") == "1":
    sys.stderr.write("gh: simulated failure\\n")
    sys.exit(1)
if args[:2] == ["auth", "status"]:
    sys.exit(0)
if args[:2] == ["pr", "create"]:
    num = os.environ.get("GH_STUB_PR_NUMBER", "42")
    sys.stdout.write("Creating pull request...\\n")
    sys.stdout.write("https://github.com/acme/repo/pull/%s\\n" % num)
    sys.exit(0)
if args[:2] in (["pr", "ready"], ["pr", "merge"], ["pr", "close"]):
    sys.exit(0)
if args[:2] == ["pr", "view"]:
    sys.stdout.write(json.dumps({
        "state": "OPEN", "isDraft": True,
        "url": "https://github.com/acme/repo/pull/42",
        "number": 42, "title": "t"}))
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
    return [line.split("\x1f") for line in
            log.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
def test_is_git_repo_and_dirty(repo, tmp_path):
    assert gitpr.is_git_repo(repo) is True
    assert gitpr.is_git_repo(tmp_path / "nope") is False
    assert gitpr.is_dirty(repo) is False
    (repo / "new.txt").write_text("x", encoding="utf-8")  # untracked -> dirty
    assert gitpr.is_dirty(repo) is True


def test_branch_sha_and_detached(repo):
    assert gitpr.current_branch(repo) == "main"
    assert gitpr.is_detached_head(repo) is False
    sha = gitpr.head_sha(repo)
    assert sha and len(sha) == 40
    _g(repo, "checkout", "-q", "--detach")
    assert gitpr.is_detached_head(repo) is True


def test_head_sha_none_on_non_repo(tmp_path):
    assert gitpr.head_sha(tmp_path / "nope") is None
    assert gitpr.current_branch(tmp_path / "nope") is None


def test_has_remote(repo, tmp_path):
    assert gitpr.has_remote(repo) is False
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
    _g(repo, "remote", "add", "origin", str(bare))
    assert gitpr.has_remote(repo) is True
    assert gitpr.has_remote(repo, "origin") is True
    assert gitpr.has_remote(repo, "upstream") is False


def test_branch_name_for_run():
    assert gitpr.branch_name_for_run("20260606-1") == "agentorch/20260606-1"


# --------------------------------------------------------------------------- #
# Subprocess core never raises
# --------------------------------------------------------------------------- #
def test_run_missing_binary_returns_127():
    rc, out, err = gitpr._run(["definitely-not-a-real-binary-zzz"])
    assert rc == 127
    assert "not found" in err


# --------------------------------------------------------------------------- #
# Worktree lifecycle
# --------------------------------------------------------------------------- #
def test_add_and_remove_worktree(repo, tmp_path):
    wt = tmp_path / "wt"
    branch = gitpr.branch_name_for_run("run1")
    gitpr.add_worktree(repo, wt, branch=branch, start_point=gitpr.head_sha(repo))
    assert (wt / "README.md").exists()
    assert gitpr.current_branch(wt) == branch
    # worktree commits don't move the operator's main branch
    assert gitpr.current_branch(repo) == "main"
    gitpr.remove_worktree(repo, wt)
    assert not wt.exists()


def test_add_worktree_duplicate_branch_raises(repo, tmp_path):
    branch = gitpr.branch_name_for_run("dup")
    gitpr.add_worktree(repo, tmp_path / "wt1", branch=branch)
    with pytest.raises(gitpr.GitError):
        gitpr.add_worktree(repo, tmp_path / "wt2", branch=branch)


# --------------------------------------------------------------------------- #
# Commit
# --------------------------------------------------------------------------- #
def test_commit_returns_sha_then_none_when_clean(repo):
    (repo / "a.py").write_text("print(1)\n", encoding="utf-8")
    sha = gitpr.commit(repo, "add a.py")
    assert sha and len(sha) == 40
    assert gitpr.head_sha(repo) == sha
    # nothing changed -> a second commit is a no-op, returns None (not an error)
    assert gitpr.commit(repo, "noop") is None


def test_commit_stages_untracked_and_deletions(repo):
    (repo / "keep.txt").write_text("k\n", encoding="utf-8")
    (repo / "README.md").unlink()  # deletion must be committed too
    sha = gitpr.commit(repo, "churn")
    assert sha
    files = _g(repo, "ls-files").split()
    assert "keep.txt" in files and "README.md" not in files


def test_commit_author_attribution(repo):
    (repo / "b.py").write_text("x\n", encoding="utf-8")
    gitpr.commit(repo, "by codex", author_name="codex",
                 author_email="codex@agentorch.local")
    assert _g(repo, "log", "-1", "--format=%an").strip() == "codex"
    assert _g(repo, "log", "-1", "--format=%ae").strip() == "codex@agentorch.local"


# --------------------------------------------------------------------------- #
# Push
# --------------------------------------------------------------------------- #
def test_push_to_bare_remote(repo, tmp_path):
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
    _g(repo, "remote", "add", "origin", str(bare))
    gitpr.push(repo, "main")
    # the bare remote now has the main ref
    out = subprocess.run(["git", "-C", str(bare), "branch", "--list", "main"],
                         capture_output=True, text=True).stdout
    assert "main" in out


def test_push_without_remote_raises(repo):
    with pytest.raises(gitpr.GitError):
        gitpr.push(repo, "main")


# --------------------------------------------------------------------------- #
# gh availability + PR lifecycle (stubbed)
# --------------------------------------------------------------------------- #
def test_gh_available_and_authed(repo, gh_stub):
    assert gitpr.gh_available() is True
    assert gitpr.gh_authed(repo) is True


def test_gh_authed_false_on_failure(repo, gh_stub, monkeypatch):
    monkeypatch.setenv("GH_STUB_FAIL", "1")
    assert gitpr.gh_authed(repo) is False


def test_create_pr_parses_url_and_number(repo, gh_stub):
    info = gitpr.create_pr(repo, base="main", head="agentorch/x",
                           title="t", body="b", draft=True)
    assert info.url == "https://github.com/acme/repo/pull/42"
    assert info.number == 42
    calls = _gh_calls(gh_stub)
    create = [c for c in calls if c[:2] == ["pr", "create"]]
    assert create and "--draft" in create[0]
    assert "--base" in create[0] and "main" in create[0]


def test_create_pr_non_draft_omits_flag(repo, gh_stub):
    gitpr.create_pr(repo, base="main", head="h", title="t", body="b", draft=False)
    create = [c for c in _gh_calls(gh_stub) if c[:2] == ["pr", "create"]][0]
    assert "--draft" not in create


def test_create_pr_failure_raises(repo, gh_stub, monkeypatch):
    monkeypatch.setenv("GH_STUB_FAIL", "1")
    with pytest.raises(gitpr.GitError):
        gitpr.create_pr(repo, base="main", head="h", title="t", body="b")


def test_mark_ready_merge_close(repo, gh_stub):
    gitpr.mark_ready(repo, 42)
    gitpr.merge_pr(repo, 42, method="squash", delete_branch=True)
    gitpr.close_pr(repo, 7)
    calls = _gh_calls(gh_stub)
    assert ["pr", "ready", "42"] in calls
    merge = [c for c in calls if c[:2] == ["pr", "merge"]][0]
    assert "--squash" in merge and "--delete-branch" in merge
    assert any(c[:3] == ["pr", "close", "7"] for c in calls)


def test_merge_unknown_method_raises(repo, gh_stub):
    with pytest.raises(gitpr.GitError):
        gitpr.merge_pr(repo, 42, method="octopus")


def test_pr_status_returns_json(repo, gh_stub):
    st = gitpr.pr_status(repo, 42)
    assert st["number"] == 42 and st["state"] == "OPEN"


# --------------------------------------------------------------------------- #
# _parse_pr_ref unit
# --------------------------------------------------------------------------- #
def test_parse_pr_ref():
    info = gitpr._parse_pr_ref("noise\nhttps://github.com/o/r/pull/123\n")
    assert info.number == 123 and info.url.endswith("/pull/123")
    bare = gitpr._parse_pr_ref("https://example.com/whatever\n")
    assert bare.url == "https://example.com/whatever" and bare.number is None
    assert gitpr._parse_pr_ref("").url == ""


# --------------------------------------------------------------------------- #
# Session persistence
# --------------------------------------------------------------------------- #
def test_session_round_trip(tmp_path):
    run_dir = tmp_path / "runs" / "r1"
    sess = gitpr.PrSession(
        run_id="r1", base_branch="main", temp_branch="agentorch/r1",
        base_sha="abc", work_dir=str(tmp_path),
    )
    sess.commits.append({"step": 1, "sha": "deadbeef", "outcome": "verified"})
    gitpr.save_session(run_dir, sess)
    loaded = gitpr.load_session(run_dir)
    assert loaded is not None
    assert loaded.run_id == "r1"
    assert loaded.temp_branch == "agentorch/r1"
    assert loaded.commits[0]["sha"] == "deadbeef"
    assert loaded.status == "running"


def test_session_load_missing_or_corrupt(tmp_path):
    assert gitpr.load_session(tmp_path / "absent") is None
    run_dir = tmp_path / "bad"
    run_dir.mkdir()
    gitpr.session_path(run_dir).write_text("{not json", encoding="utf-8")
    assert gitpr.load_session(run_dir) is None


def test_session_from_dict_ignores_unknown_keys():
    sess = gitpr.PrSession.from_dict({
        "run_id": "r", "base_branch": "main", "temp_branch": "t",
        "base_sha": "s", "work_dir": "/w", "bogus_future_field": 1,
    })
    assert sess.run_id == "r"

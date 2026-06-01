"""Issue #38 — --protect-paths / --allow-paths guardrail on the change set.

The dispatcher already snapshots the out-dir before/after and computes the
added/modified/deleted set. These tests cover the glob matcher and the policy
gate that rides that change set: a worker that touches a denylisted path (or
writes outside an allowlisted subtree) fails the run and is recorded in
meta.json, instead of relying on a human reading the diff.
"""
from __future__ import annotations

from dataclasses import asdict

from harness.dispatch import DispatchResult, _glob_to_regex, evaluate_path_policy


def _matches(glob, path):
    return bool(_glob_to_regex(glob).match(path))


# --- glob semantics --- #

def test_doublestar_spans_directories():
    assert _matches("docs/core/**", "docs/core/x.py")
    assert _matches("docs/core/**", "docs/core/sub/deep/y.py")
    assert not _matches("docs/core/**", "docs/other.py")
    assert not _matches("docs/core/**", "docsX/core/a.py")


def test_leading_doublestar_matches_root_and_nested():
    assert _matches("**/*.lock", "package.lock")
    assert _matches("**/*.lock", "a/b/c.lock")
    assert not _matches("**/*.lock", "a/b/c.txt")


def test_single_star_is_segment_bounded():
    assert _matches("*.py", "main.py")
    assert not _matches("*.py", "pkg/main.py")          # * does not cross /
    assert _matches("migrations/**", "migrations/001_init.sql")


# --- denylist --- #

def test_protect_flags_denylisted_path():
    v = evaluate_path_policy(
        ["src/app.py", "go.sum", "docs/core/spec.md"],
        protect_globs=["go.sum", "docs/core/**"],
    )
    offenders = {x["path"] for x in v}
    assert offenders == {"go.sum", "docs/core/spec.md"}
    assert "src/app.py" not in offenders


def test_no_globs_means_no_violations():
    assert evaluate_path_policy(["anything", "a/b.lock"], None, None) == []


# --- allowlist --- #

def test_allow_flags_paths_outside_subtree():
    v = evaluate_path_policy(
        ["src/a.py", "src/sub/b.py", "README.md", "Makefile"],
        allow_globs=["src/**"],
    )
    offenders = {x["path"] for x in v}
    assert offenders == {"README.md", "Makefile"}
    assert all("outside" in x["reason"] for x in v)


def test_protect_takes_precedence_in_reason():
    # A path both denied and outside the allowlist reports the protect reason.
    v = evaluate_path_policy(
        ["secret.key"],
        protect_globs=["*.key"],
        allow_globs=["src/**"],
    )
    assert len(v) == 1 and "protected" in v[0]["reason"]


def test_backslash_paths_normalized():
    # Snapshot paths use os.sep; matcher normalizes to POSIX before matching.
    v = evaluate_path_policy(["docs\\core\\spec.md"], protect_globs=["docs/core/**"])
    assert len(v) == 1


# --- result plumbing --- #

def test_violations_serialize_in_meta():
    r = DispatchResult(
        run_id="r", run_dir="/x", mode="direct", generator="codex", critic=None,
        success=False, duration_s=1.0,
        protect_violations=[{"path": "go.sum", "reason": "matches protected path glob 'go.sum'"}],
    )
    d = asdict(r)
    assert d["protect_violations"][0]["path"] == "go.sum"
    assert d["success"] is False

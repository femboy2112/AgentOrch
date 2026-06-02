"""Issue #56 — plan provenance: hash the emitted/executed plan + detect drift.

``load_plan`` validates only the SHAPE of an operator-supplied plan; nothing
recorded which plan a run actually executed, nor flagged a hand-edit between a
``--plan-only`` emit and a ``--plan`` feed-back. This adds:

  * ``--plan-only``  -> meta.json plan_provenance {source: "emitted", sha256}
  * ``--plan <file>``-> meta.json plan_provenance {source, sha256, matches_emitted}
  * ``--plan-expect-sha`` -> a hard pin that REFUSES to run on a mismatch.

Default behaviour (no injected/emitted plan, no flag) keeps meta.json
byte-identical (the plan_provenance key is dropped when absent).

All hermetic: ``_run_workflow`` is monkeypatched to a trivial coroutine so no
real worker CLI runs and no network is touched; tmp paths only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.dispatch import plan_file_sha256


# --------------------------------------------------------------------------- #
# plan_file_sha256: hashes the exact on-disk bytes (drift detector primitive)
# --------------------------------------------------------------------------- #
def test_plan_file_sha256_hashes_raw_bytes(tmp_path):
    import hashlib

    p = tmp_path / "plan.json"
    body = b'["Step 1", "Step 2"]'
    p.write_bytes(body)
    assert plan_file_sha256(p) == hashlib.sha256(body).hexdigest()


def test_plan_file_sha256_changes_on_edit(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text('["Step 1", "Step 2"]', encoding="utf-8")
    before = plan_file_sha256(p)
    # A hand-edit (even a single added step) changes the digest -> drift detected.
    p.write_text('["Step 1", "Step 2", "Step 3 (sneaked in)"]', encoding="utf-8")
    after = plan_file_sha256(p)
    assert before != after


def test_plan_file_sha256_missing_file_raises(tmp_path):
    with pytest.raises(ValueError, match="no such plan file"):
        plan_file_sha256(tmp_path / "does-not-exist.json")


# --------------------------------------------------------------------------- #
# --plan-only emit: meta.json records the emitted plan's sha256
# --------------------------------------------------------------------------- #
def test_plan_only_emit_records_sha(tmp_path, monkeypatch):
    from harness import dispatch as dispatch_mod

    class _WF:
        plan_steps = ["Step 1: do A", "Step 2: do B"]
        plan_graph = None

    async def _ok_workflow(*args, **kwargs):
        return ("plan-only: here is the plan", _WF())

    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow)

    result = dispatch_mod.dispatch(
        "build X",
        mode="master",
        out_dir=str(tmp_path),
        plan_only=True,
        heartbeat_interval=0,
    )
    assert result.success is True

    run_dir = Path(result.run_dir)
    plan_json = run_dir / "plan.json"
    assert plan_json.exists()

    prov = result.plan_provenance
    assert prov is not None
    assert prov["source"] == "emitted"
    # The recorded hash is the hash of the actual emitted plan.json bytes.
    assert prov["sha256"] == plan_file_sha256(plan_json)

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["plan_provenance"]["source"] == "emitted"
    assert meta["plan_provenance"]["sha256"] == prov["sha256"]
    # No matches_emitted key on the emit side (nothing to compare against yet).
    assert "matches_emitted" not in meta["plan_provenance"]


# --------------------------------------------------------------------------- #
# --plan <file> feed-back: records source + sha + matches_emitted
# --------------------------------------------------------------------------- #
def _plan_file(tmp_path) -> Path:
    p = tmp_path / "plan.json"
    p.write_text('["Step 1: do A", "Step 2: do B"]', encoding="utf-8")
    return p


def test_plan_inject_records_provenance(tmp_path, monkeypatch):
    from harness import dispatch as dispatch_mod

    async def _ok_workflow(*args, **kwargs):
        return ("done", None)

    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow)

    plan = _plan_file(tmp_path)
    result = dispatch_mod.dispatch(
        "build X",
        mode="master",
        out_dir=str(tmp_path / "work"),
        plan_steps=["Step 1: do A", "Step 2: do B"],
        plan_source=str(plan),
        heartbeat_interval=0,
    )
    assert result.success is True
    prov = result.plan_provenance
    assert prov is not None
    assert prov["source"] == str(plan)
    assert prov["sha256"] == plan_file_sha256(plan)
    # No pin supplied -> matches_emitted is None (no comparison made).
    assert prov["matches_emitted"] is None

    meta = json.loads((Path(result.run_dir) / "meta.json").read_text())
    assert meta["plan_provenance"]["source"] == str(plan)
    assert meta["plan_provenance"]["matches_emitted"] is None


def test_plan_inject_matching_pin_sets_matches_emitted_true(tmp_path, monkeypatch):
    from harness import dispatch as dispatch_mod

    async def _ok_workflow(*args, **kwargs):
        return ("done", None)

    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow)

    plan = _plan_file(tmp_path)
    sha = plan_file_sha256(plan)
    result = dispatch_mod.dispatch(
        "build X",
        mode="master",
        out_dir=str(tmp_path / "work"),
        plan_steps=["Step 1: do A", "Step 2: do B"],
        plan_source=str(plan),
        plan_expect_sha=sha,
        heartbeat_interval=0,
    )
    assert result.success is True
    assert result.plan_provenance["matches_emitted"] is True


def test_plan_inject_mismatched_pin_refuses(tmp_path, monkeypatch):
    from harness import dispatch as dispatch_mod

    async def _ok_workflow(*args, **kwargs):
        return ("done", None)

    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow)

    plan = _plan_file(tmp_path)
    bad = "0" * 64
    # The hard pin refuses BEFORE any worker — raises a clear ValueError.
    with pytest.raises(ValueError, match="plan sha256 mismatch"):
        dispatch_mod.dispatch(
            "build X",
            mode="master",
            out_dir=str(tmp_path / "work"),
            plan_steps=["Step 1: do A", "Step 2: do B"],
            plan_source=str(plan),
            plan_expect_sha=bad,
            heartbeat_interval=0,
        )


# --------------------------------------------------------------------------- #
# Backward-compat: a planner-driven run (no plan injected/emitted) is unchanged
# --------------------------------------------------------------------------- #
def test_no_plan_meta_has_no_provenance_key(tmp_path, monkeypatch):
    from harness import dispatch as dispatch_mod

    async def _ok_workflow(*args, **kwargs):
        return ("done", None)

    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow)

    result = dispatch_mod.dispatch(
        "build X",
        mode="direct",
        out_dir=str(tmp_path),
        heartbeat_interval=0,
    )
    assert result.success is True
    assert result.plan_provenance is None
    meta = json.loads((Path(result.run_dir) / "meta.json").read_text())
    # Drop-when-absent contract: the key is NOT present for a non-plan run.
    assert "plan_provenance" not in meta


# --------------------------------------------------------------------------- #
# CLI guards: --plan-expect-sha mismatch / misuse fail fast with exit 1
# --------------------------------------------------------------------------- #
def test_cli_plan_expect_sha_without_plan_errors(tmp_path, capsys):
    from harness import cli

    rc = cli.main([
        "do", "build X", "--mode", "master",
        "--plan-expect-sha", "0" * 64,
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "--plan-expect-sha requires --plan" in err


def test_cli_plan_expect_sha_mismatch_refuses(tmp_path, capsys, monkeypatch):
    from harness import cli
    from harness import dispatch as dispatch_mod

    # dispatch must never be reached on a mismatch; make it explode if it is.
    def _boom(*a, **k):  # pragma: no cover - asserts non-invocation
        raise AssertionError("dispatch should not run on a sha mismatch")

    monkeypatch.setattr(cli, "dispatch", _boom)

    plan = _plan_file(tmp_path)
    rc = cli.main([
        "do", "build X", "--mode", "master",
        "--plan", str(plan),
        "--plan-expect-sha", "0" * 64,
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "plan sha256 mismatch" in err

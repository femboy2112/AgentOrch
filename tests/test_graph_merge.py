"""Merge subsystem unit tests (docs §5 M4 / §3.5 / §4) — pure + filesystem only.

Covers ``agy_orchestrator/workflows/graph_merge.py`` with NO agents and no
network: a tmp "live tree" directory + a per-node "workspace" directory + a
per-node ``base_snapshot`` (the live tree's bytes when the node opened), exactly
the inputs the master graph walker hands ``merge_node``.

  * disjoint merge copies BOTH nodes' files (no overlap).
  * overlap detection via ``base_snapshot`` flags a path a sibling re-wrote.
  * ``disjoint`` policy ABORTS on overlap (MergeConflict).
  * ``fail`` policy records the conflicting paths (MergeConflict.outcome).
  * ``reconcile`` policy calls a STUBBED reconciler and records a MergeOutcome
    with the merged bytes in the live tree.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agy_orchestrator.execution.workspace import _snapshot_tree
from agy_orchestrator.workflows.graph_merge import (
    DEFAULT_MERGE_POLICY,
    MergeConflict,
    MergeOutcome,
    merge_node,
)


def _write(root: Path, rel: str, data: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data)


@pytest.fixture
def trees(tmp_path):
    """A (live_tree, ws) pair seeded with one common base file.

    ``live_tree`` is the operator's tree; ``ws`` is a node's isolated workspace
    starting as a copy. The returned ``base`` is the snapshot the node took at
    open (== the live tree before any sibling merged)."""
    live = tmp_path / "live"
    ws = tmp_path / "ws"
    live.mkdir()
    ws.mkdir()
    _write(live, "base.txt", "base\n")
    _write(ws, "base.txt", "base\n")
    base = _snapshot_tree(live)
    return live, ws, base


def test_disjoint_write_applied(trees):
    # The node adds a brand-new file the sibling never touched -> disjoint -> copied.
    live, ws, base = trees
    _write(ws, "node_a.txt", "A wrote here\n")

    sem = asyncio.Semaphore(1)
    outcome = asyncio.run(merge_node(
        node_id="A", layer=1, src_ws=ws, live_tree=live,
        base_snapshot=base, policy=DEFAULT_MERGE_POLICY,
        reconcile_station=None, sem=sem,
    ))
    assert outcome.conflict is False
    assert outcome.resolution == "disjoint"
    assert outcome.overlapping_paths == []
    assert (live / "node_a.txt").read_text() == "A wrote here\n"


def test_two_disjoint_nodes_both_applied(trees):
    # Two nodes, distinct files. Merge A then B (serial): the live tree already
    # carries A's file when B merges, but B's path is disjoint so B's lands too —
    # AND A's file is NOT clobbered.
    live, ws, base = trees
    ws_b = ws.parent / "ws_b"
    ws_b.mkdir()
    _write(ws_b, "base.txt", "base\n")
    base_b = _snapshot_tree(live)  # B opened against the same pristine live tree

    _write(ws, "node_a.txt", "A\n")
    _write(ws_b, "node_b.txt", "B\n")
    sem = asyncio.Semaphore(1)

    async def _both():
        oa = await merge_node(node_id="A", layer=1, src_ws=ws, live_tree=live,
                              base_snapshot=base, policy="disjoint",
                              reconcile_station=None, sem=sem)
        ob = await merge_node(node_id="B", layer=1, src_ws=ws_b, live_tree=live,
                              base_snapshot=base_b, policy="disjoint",
                              reconcile_station=None, sem=sem)
        return oa, ob

    oa, ob = asyncio.run(_both())
    assert oa.resolution == "disjoint" and ob.resolution == "disjoint"
    assert (live / "node_a.txt").read_text() == "A\n"
    assert (live / "node_b.txt").read_text() == "B\n"


def test_overlap_detected_against_base_snapshot(trees):
    # A sibling already merged a change to shared.txt (so the live tree no longer
    # matches the node's base); this node ALSO wrote shared.txt -> overlap flagged.
    live, ws, base = trees
    _write(ws, "shared.txt", "node version\n")  # node opened with no shared.txt
    # Simulate a sibling having merged shared.txt into the live tree already.
    _write(live, "shared.txt", "sibling version\n")

    sem = asyncio.Semaphore(1)
    outcome = asyncio.run(merge_node(
        node_id="N", layer=2, src_ws=ws, live_tree=live,
        base_snapshot=base, policy="reconcile",
        reconcile_station=_const_reconciler(b"MERGED\n"), sem=sem,
    ))
    assert "shared.txt" in outcome.overlapping_paths


def test_disjoint_policy_aborts_on_overlap(trees):
    live, ws, base = trees
    _write(ws, "shared.txt", "node\n")
    _write(live, "shared.txt", "sibling\n")  # sibling already there

    sem = asyncio.Semaphore(1)
    with pytest.raises(MergeConflict) as ei:
        asyncio.run(merge_node(
            node_id="N", layer=2, src_ws=ws, live_tree=live,
            base_snapshot=base, policy="disjoint",
            reconcile_station=None, sem=sem,
        ))
    oc = ei.value.outcome
    assert oc.conflict is True
    assert oc.resolution == "aborted:disjoint-policy"
    assert "shared.txt" in oc.overlapping_paths
    # The live tree was NOT overwritten by the aborted node.
    assert (live / "shared.txt").read_text() == "sibling\n"


def test_fail_policy_records_conflicting_paths(trees):
    live, ws, base = trees
    _write(ws, "shared.txt", "node\n")
    _write(ws, "also.txt", "node2\n")
    _write(live, "shared.txt", "sibling\n")
    _write(live, "also.txt", "sibling2\n")

    sem = asyncio.Semaphore(1)
    with pytest.raises(MergeConflict) as ei:
        asyncio.run(merge_node(
            node_id="N", layer=3, src_ws=ws, live_tree=live,
            base_snapshot=base, policy="fail",
            reconcile_station=None, sem=sem,
        ))
    oc = ei.value.outcome
    assert oc.policy == "fail"
    assert oc.resolution == "aborted:fail-policy"
    assert oc.overlapping_paths == ["also.txt", "shared.txt"]  # sorted
    # The outcome serializes for meta.json.
    d = oc.to_dict()
    assert d["conflict"] is True and d["overlapping_paths"] == ["also.txt", "shared.txt"]


def _const_reconciler(merged: bytes):
    async def _r(rel, base_bytes, sibling_bytes, node_bytes):
        return merged
    return _r


def test_reconcile_policy_calls_stub_and_records_outcome(trees):
    # reconcile invokes the stubbed reconciler for the overlapping path, writes the
    # merged bytes into the live tree, and records a MergeOutcome(resolution=
    # "reconciled"). A disjoint write in the SAME node still auto-applies.
    live, ws, base = trees
    _write(ws, "shared.txt", "node version\n")
    _write(ws, "fresh.txt", "node-only\n")          # disjoint
    _write(live, "shared.txt", "sibling version\n")  # sibling already merged

    calls = []

    async def _stub(rel, base_bytes, sibling_bytes, node_bytes):
        calls.append((rel, base_bytes, sibling_bytes, node_bytes))
        return b"RECONCILED CONTENT\n"

    sem = asyncio.Semaphore(1)
    outcome = asyncio.run(merge_node(
        node_id="N", layer=2, src_ws=ws, live_tree=live,
        base_snapshot=base, policy="reconcile",
        reconcile_station=_stub, sem=sem,
    ))
    assert isinstance(outcome, MergeOutcome)
    assert outcome.conflict is False
    assert outcome.resolution == "reconciled"
    assert outcome.overlapping_paths == ["shared.txt"]
    # The reconciler was consulted for the overlapping path exactly once, with the
    # three byte views.
    assert len(calls) == 1
    rel, base_b, sib_b, node_b = calls[0]
    assert rel == "shared.txt"
    assert base_b is None  # the path didn't exist in this node's base snapshot
    assert sib_b == b"sibling version\n"
    assert node_b == b"node version\n"
    # Merged bytes landed; the disjoint file also applied.
    assert (live / "shared.txt").read_text() == "RECONCILED CONTENT\n"
    assert (live / "fresh.txt").read_text() == "node-only\n"


def test_reconcile_without_station_is_hard_conflict(trees):
    # reconcile policy with an overlap but NO reconciler wired must not clobber:
    # it raises a MergeConflict recording the no-reconciler resolution.
    live, ws, base = trees
    _write(ws, "shared.txt", "node\n")
    _write(live, "shared.txt", "sibling\n")

    sem = asyncio.Semaphore(1)
    with pytest.raises(MergeConflict) as ei:
        asyncio.run(merge_node(
            node_id="N", layer=2, src_ws=ws, live_tree=live,
            base_snapshot=base, policy="reconcile",
            reconcile_station=None, sem=sem,
        ))
    assert ei.value.outcome.resolution == "aborted:no-reconciler"
    assert (live / "shared.txt").read_text() == "sibling\n"  # untouched


def test_invalid_policy_raises():
    sem = asyncio.Semaphore(1)
    with pytest.raises(ValueError):
        asyncio.run(merge_node(
            node_id="X", layer=1, src_ws=Path("."), live_tree=Path("."),
            base_snapshot={}, policy="bogus", reconcile_station=None, sem=sem,
        ))

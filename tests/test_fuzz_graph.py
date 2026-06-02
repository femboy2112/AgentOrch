"""Hermetic fuzz / property / edge-case tests for the graph-execution layer.

Targets:
  * agy_orchestrator/execution/graph_plan.py  (pure DAG model + validation)
  * agy_orchestrator/workflows/graph_merge.py  (per-node merge into the live tree)

NO real workers, NO network, NO credentials. The merge tests use only the local
filesystem (tmp_path) and trivial async stubs for the reconcile station. Every
test asserts CORRECT, robust behaviour and must stay green.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from agy_orchestrator.execution.graph_plan import (
    ChainPlan,
    GraphPlan,
    PlanCycleError,
    PlanNode,
    to_json,
    topological_layers,
    validate_graph,
)
from agy_orchestrator.execution.workspace import _snapshot_tree
from agy_orchestrator.workflows.graph_merge import (
    DEFAULT_MERGE_POLICY,
    MERGE_POLICIES,
    MergeConflict,
    merge_node,
)


# --------------------------------------------------------------------------- #
# graph_plan: validate_graph — adversarial / malformed input
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw", [
    [],                                              # empty list
    {},                                              # dict, not a list
    None,                                            # None
    "nodes",                                         # string
    [None],                                          # non-dict node
    [{"task": "t"}],                                 # missing id
    [{"id": "", "task": "t"}],                       # blank id
    [{"id": "   ", "task": "t"}],                    # whitespace id
    [{"id": 5, "task": "t"}],                        # non-string id
    [{"id": "a"}],                                   # missing task
    [{"id": "a", "task": ""}],                       # blank task
    [{"id": "a", "task": "   "}],                    # whitespace task
    [{"id": "a", "task": 7}],                        # non-string task
    [{"id": "a", "task": "t", "deps": "b"}],         # deps not a list
    [{"id": "a", "task": "t", "deps": [5]}],         # non-string dep
    [{"id": "a", "task": "t", "deps": [""]}],        # blank dep
    [{"id": "a", "task": "t", "deps": ["a"]}],       # self-dep
    [{"id": "a", "task": "t", "deps": ["ghost"]}],   # dangling dep
    [{"id": "a", "task": "t", "group": 7}],          # non-string group
    [{"id": "a", "task": "t"}, {"id": "a", "task": "u"}],  # duplicate id
])
def test_validate_graph_rejects_malformed(raw):
    """Every malformed shape raises ValueError (fail-fast) — never crashes with
    a bare TypeError/KeyError/IndexError, never silently accepts."""
    with pytest.raises(ValueError):
        validate_graph(raw)


def test_validate_graph_rejects_cycle():
    raw = [
        {"id": "a", "task": "t", "deps": ["b"]},
        {"id": "b", "task": "t", "deps": ["a"]},
    ]
    with pytest.raises(ValueError):
        validate_graph(raw)


def test_validate_graph_strips_ids_and_deps_consistently():
    """An id and a dep that point to the same node modulo surrounding whitespace
    must resolve to the same stripped id (no false 'dangling dep')."""
    raw = [
        {"id": "a", "task": "t", "deps": ["  b  "]},
        {"id": " b ", "task": "t", "deps": []},
    ]
    nodes = validate_graph(raw)
    assert [n.id for n in nodes] == ["a", "b"]
    assert nodes[0].deps == ["b"]


def test_validate_graph_accepts_control_chars_and_unicode():
    """NUL / control / RTL-override chars are not rejected; they round-trip."""
    raw = [{"id": "a\x00b", "task": "do \x07 ‮ thing", "deps": []}]
    nodes = validate_graph(raw)
    assert nodes[0].id == "a\x00b"
    j = to_json(GraphPlan(nodes))
    s = json.dumps(j)
    back = validate_graph(json.loads(s)["nodes"])
    assert back[0].id == "a\x00b"
    assert back[0].task == nodes[0].task


def test_validate_graph_duplicate_deps_are_tolerated():
    raw = [
        {"id": "a", "task": "t", "deps": ["b", "b"]},
        {"id": "b", "task": "t"},
    ]
    nodes = validate_graph(raw)
    g = GraphPlan(nodes)
    assert g.topo_order() == ["b", "a"]
    assert g.parallel_groups() == [["b"], ["a"]]


# --------------------------------------------------------------------------- #
# graph_plan: topo_order / layers — topology edge cases
# --------------------------------------------------------------------------- #

def test_topo_order_single_node():
    g = GraphPlan([PlanNode(id="a", task="t")])
    assert g.topo_order() == ["a"]
    assert g.parallel_groups() == [["a"]]
    assert g.as_steps() == ["t"]


def test_topo_order_stable_tie_break_by_input_order():
    """Independent (no-dep) nodes keep input order; a re-ordered input yields the
    re-ordered topo so resume-equality is deterministic."""
    g1 = GraphPlan([PlanNode(id=x, task=x) for x in ["a", "b", "c"]])
    assert g1.topo_order() == ["a", "b", "c"]
    g2 = GraphPlan([PlanNode(id=x, task=x) for x in ["c", "a", "b"]])
    assert g2.topo_order() == ["c", "a", "b"]


def test_topo_order_diamond():
    nodes = [
        PlanNode(id="A", task="A"),
        PlanNode(id="B", task="B", deps=["A"]),
        PlanNode(id="C", task="C", deps=["A"]),
        PlanNode(id="D", task="D", deps=["B", "C"]),
    ]
    g = GraphPlan(nodes)
    order = g.topo_order()
    # A first, D last, B and C between in input order.
    assert order[0] == "A" and order[-1] == "D"
    assert order.index("B") < order.index("C")
    assert g.parallel_groups() == [["A"], ["B", "C"], ["D"]]


def test_topo_order_is_deterministic_repeated_calls():
    nodes = [
        PlanNode(id="A", task="A"),
        PlanNode(id="B", task="B", deps=["A"]),
        PlanNode(id="C", task="C", deps=["A"]),
    ]
    g = GraphPlan(nodes)
    assert g.topo_order() == g.topo_order() == g.topo_order()


def test_topological_layers_cycle_raises_plancycleerror():
    nodes = [
        PlanNode(id="a", task="t", deps=["b"]),
        PlanNode(id="b", task="t", deps=["a"]),
    ]
    with pytest.raises(PlanCycleError):
        topological_layers(nodes)


def test_topo_order_long_chain_terminates_and_is_correct():
    n = 500
    nodes = [
        PlanNode(id=f"s{i}", task=f"t{i}", deps=([f"s{i-1}"] if i else []))
        for i in range(n)
    ]
    g = GraphPlan(nodes)
    assert g.topo_order() == [f"s{i}" for i in range(n)]
    # Linear chain => one single-node layer per step.
    assert g.parallel_groups() == [[f"s{i}"] for i in range(n)]


def test_ancestors_transitive_closure_in_topo_order():
    nodes = [
        PlanNode(id="A", task="A"),
        PlanNode(id="B", task="B", deps=["A"]),
        PlanNode(id="C", task="C", deps=["B"]),
        PlanNode(id="D", task="D", deps=["A"]),
        PlanNode(id="E", task="E", deps=["C", "D"]),
    ]
    g = GraphPlan(nodes)
    assert g.ancestors("A") == []
    assert g.ancestors("C") == ["A", "B"]
    anc_e = g.ancestors("E")
    assert set(anc_e) == {"A", "B", "C", "D"}
    # Topologically ordered: A precedes all, B precedes C.
    assert anc_e.index("A") < anc_e.index("B") < anc_e.index("C")


def test_ancestors_unknown_node_raises_valueerror():
    g = GraphPlan([PlanNode(id="a", task="t")])
    with pytest.raises(ValueError):
        g.ancestors("nope")


# --------------------------------------------------------------------------- #
# graph_plan: ChainPlan lift + to_json round-trips
# --------------------------------------------------------------------------- #

def test_chainplan_empty_steps():
    cp = ChainPlan(steps=[])
    assert cp.as_steps() == []
    g = cp.as_graph()
    assert g.nodes == []
    assert g.topo_order() == []
    assert to_json(cp) == {"steps": []}


def test_chainplan_lift_to_linear_dag():
    cp = ChainPlan(steps=["x", "y", "z"])
    g = cp.as_graph()
    assert [n.id for n in g.nodes] == ["s1", "s2", "s3"]
    assert g.nodes[0].deps == []
    assert g.nodes[1].deps == ["s1"]
    assert g.as_steps() == ["x", "y", "z"]  # topo order preserves chain


def test_to_json_round_trips_graph():
    nodes = validate_graph([
        {"id": "a", "task": "ta", "deps": []},
        {"id": "b", "task": "tb", "deps": ["a"], "group": "g1"},
    ])
    plan = GraphPlan(nodes)
    j = to_json(plan)
    assert j["version"] == 2
    reparsed = validate_graph(j["nodes"])
    assert [(n.id, n.task, n.deps, n.group) for n in reparsed] == [
        (n.id, n.task, n.deps, n.group) for n in nodes
    ]


def test_to_json_rejects_non_plan():
    with pytest.raises(TypeError):
        to_json({"not": "a plan"})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# graph_merge: helpers + policy validation
# --------------------------------------------------------------------------- #

def _run(coro):
    return asyncio.run(coro)


def test_default_policy_is_valid():
    assert DEFAULT_MERGE_POLICY in MERGE_POLICIES


def test_merge_invalid_policy_raises_before_any_io(tmp_path):
    live = tmp_path / "live"; src = tmp_path / "src"
    live.mkdir(); src.mkdir()
    (src / "f.txt").write_bytes(b"data")
    sem = asyncio.Semaphore(1)
    with pytest.raises(ValueError):
        _run(merge_node(
            node_id="n", layer=0, src_ws=src, live_tree=live,
            base_snapshot={}, policy="bogus", reconcile_station=None, sem=sem,
        ))
    # Nothing was written to the live tree.
    assert not (live / "f.txt").exists()


# --------------------------------------------------------------------------- #
# graph_merge: disjoint writes
# --------------------------------------------------------------------------- #

def test_merge_disjoint_new_file_applies(tmp_path):
    live = tmp_path / "live"; src = tmp_path / "src"
    live.mkdir(); src.mkdir()
    base = _snapshot_tree(live)  # empty
    (src / "a.txt").write_bytes(b"hello")
    sem = asyncio.Semaphore(1)
    out = _run(merge_node(
        node_id="n", layer=0, src_ws=src, live_tree=live,
        base_snapshot=base, policy="reconcile", reconcile_station=None, sem=sem,
    ))
    assert out.resolution == "disjoint"
    assert out.conflict is False
    assert out.overlapping_paths == []
    assert (live / "a.txt").read_bytes() == b"hello"


def test_merge_disjoint_deletion_applies(tmp_path):
    live = tmp_path / "live"; src = tmp_path / "src"
    live.mkdir(); src.mkdir()
    (live / "gone.txt").write_bytes(b"x")
    base = _snapshot_tree(live)
    # node's workspace does NOT contain gone.txt (it deleted it); live untouched.
    sem = asyncio.Semaphore(1)
    out = _run(merge_node(
        node_id="n", layer=0, src_ws=src, live_tree=live,
        base_snapshot=base, policy="reconcile", reconcile_station=None, sem=sem,
    ))
    assert out.resolution == "disjoint"
    assert not (live / "gone.txt").exists()


def test_merge_unchanged_file_is_not_rewritten(tmp_path):
    """A file the node never changed (src==base) is left alone — not an overlap,
    not a write."""
    live = tmp_path / "live"; src = tmp_path / "src"
    live.mkdir(); src.mkdir()
    (live / "keep.txt").write_bytes(b"same")
    base = _snapshot_tree(live)
    (src / "keep.txt").write_bytes(b"same")  # identical to base
    # Sibling rewrote keep.txt while the node ran (live differs from base).
    (live / "keep.txt").write_bytes(b"sibling-wrote-this")
    sem = asyncio.Semaphore(1)
    out = _run(merge_node(
        node_id="n", layer=0, src_ws=src, live_tree=live,
        base_snapshot=base, policy="fail", reconcile_station=None, sem=sem,
    ))
    # The node didn't change keep.txt, so no conflict despite the sibling write.
    assert out.conflict is False
    assert out.overlapping_paths == []
    assert (live / "keep.txt").read_bytes() == b"sibling-wrote-this"


# --------------------------------------------------------------------------- #
# graph_merge: overlap under abort policies
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("policy", ["disjoint", "fail"])
def test_merge_overlap_aborts_before_writing(tmp_path, policy):
    live = tmp_path / "live"; src = tmp_path / "src"
    live.mkdir(); src.mkdir()
    (live / "shared.txt").write_bytes(b"base")
    base = _snapshot_tree(live)
    (live / "shared.txt").write_bytes(b"sibling")  # sibling already merged
    (src / "shared.txt").write_bytes(b"node")      # node also wrote shared
    (src / "disjoint.txt").write_bytes(b"new")     # plus a disjoint new file
    sem = asyncio.Semaphore(1)
    with pytest.raises(MergeConflict) as ei:
        _run(merge_node(
            node_id="n", layer=2, src_ws=src, live_tree=live,
            base_snapshot=base, policy=policy, reconcile_station=None, sem=sem,
        ))
    out = ei.value.outcome
    assert out.conflict is True
    assert out.overlapping_paths == ["shared.txt"]
    assert out.resolution.startswith("aborted:")
    # Abort is atomic: the disjoint file must NOT have leaked into the live tree,
    # and the sibling's bytes on the shared path are untouched.
    assert not (live / "disjoint.txt").exists()
    assert (live / "shared.txt").read_bytes() == b"sibling"


# --------------------------------------------------------------------------- #
# graph_merge: reconcile policy
# --------------------------------------------------------------------------- #

def test_merge_reconcile_resolves_overlap_and_applies_disjoint(tmp_path):
    live = tmp_path / "live"; src = tmp_path / "src"
    live.mkdir(); src.mkdir()
    (live / "shared.txt").write_bytes(b"base")
    base = _snapshot_tree(live)
    (live / "shared.txt").write_bytes(b"sibling")  # sibling merged
    (src / "shared.txt").write_bytes(b"node")      # node overlaps
    (src / "solo.txt").write_bytes(b"solo")        # node disjoint add
    sem = asyncio.Semaphore(1)
    calls = []

    async def reconciler(rel, base_b, sib_b, node_b):
        calls.append((rel, base_b, sib_b, node_b))
        return b"MERGED:" + (node_b or b"")

    out = _run(merge_node(
        node_id="n", layer=1, src_ws=src, live_tree=live,
        base_snapshot=base, policy="reconcile", reconcile_station=reconciler, sem=sem,
    ))
    assert out.resolution == "reconciled"
    assert out.conflict is False
    assert out.overlapping_paths == ["shared.txt"]
    assert calls == [("shared.txt", b"base", b"sibling", b"node")]
    assert (live / "shared.txt").read_bytes() == b"MERGED:node"
    assert (live / "solo.txt").read_bytes() == b"solo"


def test_merge_reconcile_new_file_overlap_passes_none_base(tmp_path):
    """Two branches create the SAME brand-new file: base bytes are None and the
    reconciler still gets called with both candidate byte views."""
    live = tmp_path / "live"; src = tmp_path / "src"
    live.mkdir(); src.mkdir()
    base = _snapshot_tree(live)  # empty
    (live / "new.txt").write_bytes(b"SIB")
    (src / "new.txt").write_bytes(b"NODE")
    sem = asyncio.Semaphore(1)
    seen = {}

    async def reconciler(rel, base_b, sib_b, node_b):
        seen.update(rel=rel, base=base_b, sib=sib_b, node=node_b)
        return b"RESOLVED"

    out = _run(merge_node(
        node_id="n", layer=1, src_ws=src, live_tree=live,
        base_snapshot=base, policy="reconcile", reconcile_station=reconciler, sem=sem,
    ))
    assert out.resolution == "reconciled"
    assert seen == {"rel": "new.txt", "base": None, "sib": b"SIB", "node": b"NODE"}
    assert (live / "new.txt").read_bytes() == b"RESOLVED"


def test_merge_reconcile_without_station_is_hard_conflict(tmp_path):
    """reconcile policy + an overlap + no station wired => MergeConflict (never a
    silent last-writer-wins clobber)."""
    live = tmp_path / "live"; src = tmp_path / "src"
    live.mkdir(); src.mkdir()
    (live / "shared.txt").write_bytes(b"base")
    base = _snapshot_tree(live)
    (live / "shared.txt").write_bytes(b"sibling")
    (src / "shared.txt").write_bytes(b"node")
    sem = asyncio.Semaphore(1)
    with pytest.raises(MergeConflict) as ei:
        _run(merge_node(
            node_id="n", layer=1, src_ws=src, live_tree=live,
            base_snapshot=base, policy="reconcile", reconcile_station=None, sem=sem,
        ))
    assert ei.value.outcome.resolution == "aborted:no-reconciler"
    # The overlapping path keeps the sibling's bytes; nothing clobbered.
    assert (live / "shared.txt").read_bytes() == b"sibling"


def test_merge_outcome_to_dict_is_json_serializable(tmp_path):
    live = tmp_path / "live"; src = tmp_path / "src"
    live.mkdir(); src.mkdir()
    base = _snapshot_tree(live)
    (src / "a.txt").write_bytes(b"x")
    sem = asyncio.Semaphore(1)
    out = _run(merge_node(
        node_id="n", layer=3, src_ws=src, live_tree=live,
        base_snapshot=base, policy="reconcile", reconcile_station=None, sem=sem,
    ))
    d = out.to_dict()
    s = json.dumps(d)  # must not raise
    assert json.loads(s)["node_id"] == "n"
    assert json.loads(s)["layer"] == 3


def test_merge_serializes_under_shared_semaphore(tmp_path):
    """Two concurrent merge_node calls sharing one Semaphore(1) on the same live
    tree must not interleave (overlap detection depends on a quiescent tree).
    Each writes its own disjoint new file; both survive, no crash."""
    live = tmp_path / "live"
    live.mkdir()
    base = _snapshot_tree(live)
    sem = asyncio.Semaphore(1)

    async def make_node(name):
        src = tmp_path / f"src_{name}"
        src.mkdir()
        (src / f"{name}.txt").write_bytes(name.encode())
        return await merge_node(
            node_id=name, layer=0, src_ws=src, live_tree=live,
            base_snapshot=base, policy="reconcile", reconcile_station=None, sem=sem,
        )

    async def driver():
        return await asyncio.gather(make_node("alpha"), make_node("beta"))

    outs = _run(driver())
    assert all(o.resolution == "disjoint" for o in outs)
    assert (live / "alpha.txt").read_bytes() == b"alpha"
    assert (live / "beta.txt").read_bytes() == b"beta"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))

"""Pure tests for the graph plan model (no agents, no I/O, no network).

Covers the M1 schema + loader: ChainPlan / GraphPlan parsing, stable
topological order, parallel layers, ancestor closure, validation errors, the
load_plan auto-detect + flat-vs-graph routing, the load_plan_steps shim, and
emit->load round-trip identity. See docs/graph-execution-design.md §7.
"""
from __future__ import annotations

import json

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
from harness.dispatch import load_plan, load_plan_steps


# --------------------------------------------------------------------------- #
# ChainPlan
# --------------------------------------------------------------------------- #

def test_chain_plan_as_steps_verbatim():
    cp = ChainPlan(steps=["a", "b", "c"])
    assert cp.as_steps() == ["a", "b", "c"]
    # as_steps returns a fresh list (mutating it can't corrupt the plan).
    cp.as_steps().append("x")
    assert cp.steps == ["a", "b", "c"]


def test_chain_plan_as_graph_lifts_to_linear_deps():
    g = ChainPlan(steps=["one", "two", "three"]).as_graph()
    assert g.node_ids() == ["s1", "s2", "s3"]
    by_id = {n.id: n for n in g.nodes}
    assert by_id["s1"].deps == []
    assert by_id["s2"].deps == ["s1"]
    assert by_id["s3"].deps == ["s2"]
    # The lifted graph linearizes back to the original order.
    assert g.as_steps() == ["one", "two", "three"]


def test_chain_plan_round_trip_via_to_json():
    cp = ChainPlan(steps=["x", "y"])
    assert to_json(cp) == {"steps": ["x", "y"]}


# --------------------------------------------------------------------------- #
# GraphPlan: topo order, layers, ancestors
# --------------------------------------------------------------------------- #

def _diamond() -> GraphPlan:
    # A -> {B, C} -> D
    return GraphPlan(nodes=[
        PlanNode(id="A", task="ta", deps=[]),
        PlanNode(id="B", task="tb", deps=["A"], group="svc"),
        PlanNode(id="C", task="tc", deps=["A"], group="svc"),
        PlanNode(id="D", task="td", deps=["B", "C"]),
    ])


def test_topo_order_stable_ties_by_input_order():
    g = _diamond()
    order = g.topo_order()
    # A first, D last; B before C (input order tie-break).
    assert order[0] == "A"
    assert order[-1] == "D"
    assert order.index("B") < order.index("C")
    # Deterministic across calls.
    assert g.topo_order() == order


def test_as_steps_follows_topo_order():
    g = _diamond()
    assert g.as_steps() == ["ta", "tb", "tc", "td"]


def test_parallel_groups_diamond():
    g = _diamond()
    assert g.parallel_groups() == [["A"], ["B", "C"], ["D"]]


def test_topological_layers_linear_chain():
    nodes = ChainPlan(steps=["1", "2", "3"]).as_graph().nodes
    assert topological_layers(nodes) == [["s1"], ["s2"], ["s3"]]


def test_ancestors_branch_isolation_and_join_union():
    g = _diamond()
    # A parallel branch sees only its own ancestor (not its sibling).
    assert g.ancestors("B") == ["A"]
    assert g.ancestors("C") == ["A"]
    assert "C" not in g.ancestors("B")
    # The join node sees the union of both branches (topologically ordered).
    assert g.ancestors("D") == ["A", "B", "C"]
    # A seed node has no ancestors.
    assert g.ancestors("A") == []


def test_node_ids_input_order():
    assert _diamond().node_ids() == ["A", "B", "C", "D"]


# --------------------------------------------------------------------------- #
# GraphPlan: validation errors (each a specific ValueError message)
# --------------------------------------------------------------------------- #

def test_validate_empty_nodes():
    with pytest.raises(ValueError, match="non-empty list"):
        validate_graph([])


def test_validate_dup_id():
    with pytest.raises(ValueError, match="duplicate id 'A'"):
        validate_graph([
            {"id": "A", "task": "t", "deps": []},
            {"id": "A", "task": "t2", "deps": []},
        ])


def test_validate_dangling_dep():
    with pytest.raises(ValueError, match="unknown node 'ghost'"):
        validate_graph([{"id": "A", "task": "t", "deps": ["ghost"]}])


def test_validate_self_dep():
    with pytest.raises(ValueError, match="depends on itself"):
        validate_graph([{"id": "A", "task": "t", "deps": ["A"]}])


def test_validate_cycle():
    with pytest.raises(PlanCycleError, match="cycle"):
        validate_graph([
            {"id": "A", "task": "t", "deps": ["B"]},
            {"id": "B", "task": "t", "deps": ["A"]},
        ])


def test_validate_non_string_task():
    with pytest.raises(ValueError, match="non-empty string 'task'"):
        validate_graph([{"id": "A", "task": 42, "deps": []}])


def test_validate_blank_task():
    with pytest.raises(ValueError, match="non-empty string 'task'"):
        validate_graph([{"id": "A", "task": "  ", "deps": []}])


def test_validate_missing_id():
    with pytest.raises(ValueError, match="non-empty string 'id'"):
        validate_graph([{"task": "t", "deps": []}])


def test_validate_deps_not_a_list():
    with pytest.raises(ValueError, match="'deps' must be a list"):
        validate_graph([{"id": "A", "task": "t", "deps": "B"}])


def test_topo_order_raises_plan_cycle_error():
    g = GraphPlan(nodes=[
        PlanNode(id="A", task="t", deps=["B"]),
        PlanNode(id="B", task="t", deps=["A"]),
    ])
    with pytest.raises(PlanCycleError):
        g.topo_order()
    with pytest.raises(PlanCycleError):
        g.parallel_groups()


# --------------------------------------------------------------------------- #
# to_json round-trip identity
# --------------------------------------------------------------------------- #

def test_graph_to_json_round_trips_through_validate():
    g = _diamond()
    obj = to_json(g)
    assert obj["version"] == 2
    nodes2 = validate_graph(obj["nodes"])
    g2 = GraphPlan(nodes=nodes2)
    assert g2.as_steps() == g.as_steps()
    assert g2.parallel_groups() == g.parallel_groups()
    # group hint preserved on the round-trip.
    by_id = {n.id: n for n in g2.nodes}
    assert by_id["B"].group == "svc"


def test_to_json_omits_absent_group():
    g = GraphPlan(nodes=[PlanNode(id="A", task="t", deps=[])])
    assert "group" not in to_json(g)["nodes"][0]


# --------------------------------------------------------------------------- #
# load_plan: auto-detect flat vs graph; load_plan_steps shim
# --------------------------------------------------------------------------- #

def test_load_plan_bare_list_is_chain(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(["a", "b"]))
    plan = load_plan(str(p))
    assert isinstance(plan, ChainPlan)
    assert plan.as_steps() == ["a", "b"]


def test_load_plan_steps_object_is_chain(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"instruction": "x", "steps": ["a", "b"]}))
    plan = load_plan(str(p))
    assert isinstance(plan, ChainPlan)
    assert plan.as_steps() == ["a", "b"]


def test_load_plan_nodes_object_is_graph(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"version": 2, "nodes": [
        {"id": "A", "task": "ta", "deps": []},
        {"id": "B", "task": "tb", "deps": ["A"]},
    ]}))
    plan = load_plan(str(p))
    assert isinstance(plan, GraphPlan)
    assert plan.as_steps() == ["ta", "tb"]


def test_load_plan_both_shapes_is_error(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"steps": ["a"], "nodes": [
        {"id": "A", "task": "t", "deps": []},
    ]}))
    with pytest.raises(ValueError, match="BOTH 'nodes' and 'steps'"):
        load_plan(str(p))


def test_load_plan_neither_shape_is_error(tmp_path):
    # A shape-less object (no 'steps', no 'nodes') keeps the legacy "no steps"
    # error so the existing load_plan_steps back-compat contract is unchanged.
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"instruction": "x"}))
    with pytest.raises(ValueError, match="no steps"):
        load_plan(str(p))


def test_load_plan_non_object_non_list_is_error(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(42))
    with pytest.raises(ValueError, match="'steps' list or a graph 'nodes' list"):
        load_plan(str(p))


def test_load_plan_graph_validation_at_load_time(tmp_path):
    # A dangling dep in the graph must fail at load time (fail-fast), with the
    # file path surfaced.
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"nodes": [
        {"id": "A", "task": "t", "deps": ["ghost"]},
    ]}))
    with pytest.raises(ValueError, match="invalid graph plan"):
        load_plan(str(p))


def test_load_plan_steps_shim_flat(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"steps": ["one", "two"]}))
    assert load_plan_steps(str(p)) == ["one", "two"]


def test_load_plan_steps_shim_linearizes_graph(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"nodes": [
        {"id": "A", "task": "ta", "deps": []},
        {"id": "B", "task": "tb", "deps": ["A"]},
        {"id": "C", "task": "tc", "deps": ["A"]},
    ]}))
    # Shim returns the legacy List[str] in stable topo order.
    assert load_plan_steps(str(p)) == ["ta", "tb", "tc"]


def test_load_plan_emit_round_trip_chain(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(to_json(ChainPlan(steps=["a", "b"]))))
    plan = load_plan(str(p))
    assert isinstance(plan, ChainPlan)
    assert plan.as_steps() == ["a", "b"]


def test_load_plan_emit_round_trip_graph(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(to_json(_diamond())))
    plan = load_plan(str(p))
    assert isinstance(plan, GraphPlan)
    assert plan.as_steps() == _diamond().as_steps()

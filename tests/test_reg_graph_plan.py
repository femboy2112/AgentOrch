"""Regression test for execution/graph_plan.py — GraphPlan.ancestors() error contract.

Hermetic: pure data + topology, no workers, no network, no subprocess.

Covers one confirmed defect:
  * graph-plan-1: ``ancestors()`` traversed ``by_id[cur].deps`` for a transitively
    reached dep id without guarding that ``cur`` is a declared node. A dangling/
    unknown dep id (reachable only on a directly-constructed, un-validated
    GraphPlan) leaked a bare ``KeyError('ghost')`` from the dict lookup, whereas
    the sibling ``topo_order()`` raises the typed ``PlanCycleError`` and
    ``ancestors()`` itself raises a clean ``ValueError('unknown node id: ...')``
    when the QUERIED node is unknown. The fix raises the same typed
    ``ValueError`` on a dangling dep encountered during traversal, making the
    error contract consistent. Validated plans (the normal loader path) are
    unaffected.
"""
from __future__ import annotations

import pytest

from agy_orchestrator.execution.graph_plan import (
    GraphPlan,
    PlanCycleError,
    PlanNode,
)


def test_ancestors_dangling_dep_raises_typed_valueerror_not_keyerror():
    """A dangling dep reached during traversal -> typed ValueError, not bare KeyError."""
    g = GraphPlan(nodes=[PlanNode(id="a", task="t", deps=["ghost"])])
    with pytest.raises(ValueError) as exc:
        g.ancestors("a")
    # Typed, message-bearing error naming the offending id.
    assert "ghost" in str(exc.value)
    # Must NOT be a bare KeyError (KeyError is a subclass of LookupError, not
    # ValueError, so pytest.raises(ValueError) above already excludes it — but
    # assert the message shape matches the method's own unknown-node contract).
    assert "unknown node id" in str(exc.value)


def test_ancestors_dangling_dep_is_not_keyerror():
    """Belt-and-suspenders: the raised error is not a KeyError."""
    g = GraphPlan(nodes=[PlanNode(id="a", task="t", deps=["ghost"])])
    with pytest.raises(ValueError):
        g.ancestors("a")
    # A KeyError would NOT be caught by pytest.raises(ValueError); confirm the
    # type directly too.
    try:
        g.ancestors("a")
    except KeyError:  # pragma: no cover - would mean the bug is back
        pytest.fail("ancestors() leaked a bare KeyError on a dangling dep")
    except ValueError:
        pass


def test_ancestors_unknown_queried_node_still_valueerror():
    """The pre-existing unknown-queried-node contract is unchanged."""
    g = GraphPlan(nodes=[PlanNode(id="a", task="t")])
    with pytest.raises(ValueError) as exc:
        g.ancestors("nope")
    assert "unknown node id: nope" in str(exc.value)


def test_topo_order_contract_unchanged_on_same_plan():
    """Sibling method still raises the typed PlanCycleError on an unorderable plan
    (the contrast that makes graph-plan-1 an inconsistency)."""
    g = GraphPlan(nodes=[PlanNode(id="a", task="t", deps=["ghost"])])
    with pytest.raises(PlanCycleError):
        g.topo_order()


def test_ancestors_validated_diamond_closure_unchanged():
    """On a well-formed diamond A->{B,C}->D, ancestors() is unaffected by the fix."""
    nodes = [
        PlanNode(id="A", task="a"),
        PlanNode(id="B", task="b", deps=["A"]),
        PlanNode(id="C", task="c", deps=["A"]),
        PlanNode(id="D", task="d", deps=["B", "C"]),
    ]
    g = GraphPlan(nodes=nodes)
    # D's transitive closure is everything below it, in topological order.
    assert g.ancestors("D") == ["A", "B", "C"]
    assert g.ancestors("B") == ["A"]
    assert g.ancestors("A") == []

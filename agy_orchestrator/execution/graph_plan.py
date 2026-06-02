"""In-memory plan model for linear chains and dependency DAGs.

Pure data + topology: no I/O, no agents, no asyncio. This is the layer the
graph-execution build (docs/graph-execution-design.md §3.2) consumes — a flat
plan lifts to ``ChainPlan`` (and runs the existing linear master loop verbatim),
a ``{"nodes":[...]}`` plan parses into a ``GraphPlan`` with a real DAG.

Determinism is load-bearing: ``GraphPlan.topo_order`` is a *stable* Kahn sort
(ties broken by input order), so a graph linearizes identically across runs and
the master checkpoint-equality guard (master.py) never flaps on resume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


class PlanCycleError(ValueError):
    """Raised when a graph plan's deps form a cycle (no topological order)."""


@dataclass
class PlanNode:
    """One DAG node: a task with explicit dependencies on other node ids.

    ``group`` is an advisory scheduler hint only — **deps are authoritative**
    for ordering and concurrency.
    """

    id: str
    task: str
    deps: List[str] = field(default_factory=list)
    group: Optional[str] = None


@dataclass
class GraphPlan:
    """A dependency DAG of :class:`PlanNode`. Assumes already-validated nodes
    (build via :func:`validate_graph` / the loader, which raise on bad input)."""

    nodes: List[PlanNode]

    def _by_id(self) -> Dict[str, PlanNode]:
        return {n.id: n for n in self.nodes}

    def node_ids(self) -> List[str]:
        """Node ids in input order."""
        return [n.id for n in self.nodes]

    def topo_order(self) -> List[str]:
        """Stable Kahn topological sort; ties broken by input order.

        Raises :class:`PlanCycleError` if the deps contain a cycle (i.e. not
        every node can be emitted)."""
        order = self.node_ids()  # input order = tie-break order
        indegree: Dict[str, int] = {nid: 0 for nid in order}
        deps_of = {n.id: list(n.deps) for n in self.nodes}
        for nid in order:
            for dep in deps_of[nid]:
                # dep -> nid edge contributes one in-edge to nid.
                indegree[nid] += 1
        # Ready = no remaining unmet deps, kept in input order for stability.
        ready = [nid for nid in order if indegree[nid] == 0]
        done: List[str] = []
        done_set: set = set()
        while ready:
            nid = ready.pop(0)
            done.append(nid)
            done_set.add(nid)
            # Releasing nid may unblock its dependents; re-scan in input order so
            # ties are broken deterministically (cheap; plans are small).
            newly: List[str] = []
            for cand in order:
                if cand in done_set or cand in ready:
                    continue
                if all(d in done_set for d in deps_of[cand]):
                    newly.append(cand)
            ready.extend(newly)
        if len(done) != len(order):
            unresolved = [nid for nid in order if nid not in done_set]
            raise PlanCycleError(
                f"graph plan has a dependency cycle (cannot order: {unresolved})"
            )
        return done

    def as_steps(self) -> List[str]:
        """The backward-compat linearization: tasks in topological order."""
        by_id = self._by_id()
        return [by_id[nid].task for nid in self.topo_order()]

    def parallel_groups(self) -> List[List[str]]:
        """Kahn levels — the concurrency unit (each level may run in parallel)."""
        return topological_layers(self.nodes)

    def ancestors(self, node_id: str) -> List[str]:
        """Transitive dependency closure of ``node_id``, in topological order.

        Used for context composition: a node sees only its own ancestors, a join
        node sees the union of both branches."""
        by_id = self._by_id()
        if node_id not in by_id:
            raise ValueError(f"unknown node id: {node_id}")
        seen: set = set()
        stack = list(by_id[node_id].deps)
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(by_id[cur].deps)
        # Return in topological order so callers compose context deterministically.
        return [nid for nid in self.topo_order() if nid in seen]


@dataclass
class ChainPlan:
    """A flat linear plan — the legacy shape. ``as_steps`` returns the strings
    verbatim so the existing master loop runs byte-identically."""

    steps: List[str]

    def as_steps(self) -> List[str]:
        return list(self.steps)

    def as_graph(self) -> GraphPlan:
        """Lift a flat plan to an implicit linear DAG (ids ``s1..sN``)."""
        nodes: List[PlanNode] = []
        for i, task in enumerate(self.steps, 1):
            deps = [f"s{i - 1}"] if i > 1 else []
            nodes.append(PlanNode(id=f"s{i}", task=task, deps=deps))
        return GraphPlan(nodes=nodes)


Plan = Union[ChainPlan, GraphPlan]


def topological_layers(nodes: List[PlanNode]) -> List[List[str]]:
    """Kahn levels: layer k = nodes whose deps are all in layers < k.

    Each layer's ids are in input order. Raises :class:`PlanCycleError` on a
    cycle. A linear chain yields one single-node layer per step; a diamond
    ``A->{B,C}->D`` yields ``[[A], [B, C], [D]]``."""
    order = [n.id for n in nodes]
    deps_of = {n.id: list(n.deps) for n in nodes}
    placed: set = set()
    layers: List[List[str]] = []
    remaining = list(order)
    while remaining:
        layer = [nid for nid in remaining if all(d in placed for d in deps_of[nid])]
        if not layer:
            raise PlanCycleError(
                f"graph plan has a dependency cycle (cannot layer: {remaining})"
            )
        layers.append(layer)
        placed.update(layer)
        remaining = [nid for nid in remaining if nid not in placed]
    return layers


def to_json(plan: Plan) -> dict:
    """Serialize a plan to its on-disk JSON shape.

    ``GraphPlan`` -> the v2 ``{"version": 2, "nodes": [...]}`` object;
    ``ChainPlan`` -> the legacy ``{"steps": [...]}`` object. Round-trips with the
    loader (``load_plan(to_json(plan))`` reproduces the same shape)."""
    if isinstance(plan, GraphPlan):
        nodes = []
        for n in plan.nodes:
            node: dict = {"id": n.id, "task": n.task, "deps": list(n.deps)}
            if n.group is not None:
                node["group"] = n.group
            nodes.append(node)
        return {"version": 2, "nodes": nodes}
    if isinstance(plan, ChainPlan):
        return {"steps": list(plan.steps)}
    raise TypeError(f"not a Plan: {type(plan).__name__}")


def validate_graph(raw: list) -> List[PlanNode]:
    """Validate a raw ``"nodes"`` list and return parsed :class:`PlanNode`\\ s.

    Raises ``ValueError`` with a specific message for: empty list, a non-object
    node, a missing/non-string/blank id, a non-string/blank task, a non-list
    deps, a duplicate id, a self-dep, a dangling dep, or a dependency cycle.
    Validation runs at load time so the operator sees errors before any worker
    writes (fail-fast, matching the flat-plan loader)."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("graph plan 'nodes' must be a non-empty list")
    nodes: List[PlanNode] = []
    seen_ids: set = set()
    for idx, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"graph node {idx} must be an object")
        nid = item.get("id")
        if not isinstance(nid, str) or not nid.strip():
            raise ValueError(f"graph node {idx} must have a non-empty string 'id'")
        nid = nid.strip()
        if nid in seen_ids:
            raise ValueError(f"graph node {idx} has a duplicate id '{nid}'")
        seen_ids.add(nid)
        task = item.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"graph node '{nid}' must have a non-empty string 'task'")
        deps_raw = item.get("deps", [])
        if not isinstance(deps_raw, list):
            raise ValueError(f"graph node '{nid}' 'deps' must be a list")
        deps: List[str] = []
        for dep in deps_raw:
            if not isinstance(dep, str) or not dep.strip():
                raise ValueError(
                    f"graph node '{nid}' has a non-string/empty dep"
                )
            dep = dep.strip()
            if dep == nid:
                raise ValueError(f"graph node '{nid}' depends on itself")
            deps.append(dep)
        group = item.get("group")
        if group is not None and not isinstance(group, str):
            raise ValueError(f"graph node '{nid}' 'group' must be a string")
        nodes.append(PlanNode(id=nid, task=task, deps=deps, group=group))
    # Dangling deps: every referenced dep must be a declared node.
    for n in nodes:
        for dep in n.deps:
            if dep not in seen_ids:
                raise ValueError(
                    f"graph node '{n.id}' depends on unknown node '{dep}'"
                )
    # Cycle check (also catches deps that can't be ordered).
    GraphPlan(nodes=nodes).topo_order()
    return nodes

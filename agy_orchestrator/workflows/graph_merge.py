"""Merge subsystem for the master graph walker (docs §5 M4 / §3.5 / §4).

Two parallel DAG nodes can write the SAME file. M3 left that last-writer-wins
(a documented temporary limitation); this module makes overlaps SAFE — detected
against a per-node ``base_snapshot`` and resolved under a switchable, operator-
confirmed-``reconcile``-by-default policy.

How an overlap is detected
--------------------------
Each node opens an isolated ``_ManagedWorkspace`` and snapshots its files at open
(``base_snapshot`` = the live tree's bytes when THIS node started). Merges run
SERIALLY per completed node (guarded by a merge semaphore so two nodes never
write the live tree at once). When a node's merge runs, the live tree already
carries any earlier-merged sibling's writes. So for each file the node changed
(vs its own ``base_snapshot``), we compare the file's CURRENT bytes in the live
tree against ``base_snapshot``:

  * live == base_snapshot  -> nobody touched this path since the node opened ->
    a DISJOINT write -> apply the node's new content.
  * live != base_snapshot  -> an already-merged sibling wrote this same path
    while the node ran -> an OVERLAP -> resolve under the policy.

The three policies (switchable, DEFAULT ``reconcile``)
-----------------------------------------------------
  * ``disjoint``  — apply only when ALL of the node's writes are disjoint; ANY
    overlap aborts the merge (``MergeConflict``).
  * ``reconcile`` (DEFAULT) — disjoint writes auto-apply; each overlapping path is
    handed to the #43 reconcile station (an async ``reconcile`` callable the
    caller wires from ``ReconciliationReview`` / the critic chain) to produce a
    merged file, recorded in the returned :class:`MergeOutcome`. The JOIN node's
    verifier re-runs on the merged tree (in the master loop), so a bad merge is
    caught by the existing gate.
  * ``fail``      — any overlap aborts and the conflicting paths are recorded
    (``MergeConflict.overlapping_paths``) for ``meta.json``.

Pure + filesystem only — no agents, no asyncio beyond the merge semaphore and
the (caller-supplied) reconciler. Unit-testable without workers.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

from agy_orchestrator.execution.workspace import _snapshot_tree

logger = logging.getLogger(__name__)

# Switchable merge policies; ``reconcile`` is the operator-confirmed default (§4).
MERGE_POLICIES = ("disjoint", "reconcile", "fail")
DEFAULT_MERGE_POLICY = "reconcile"

# A reconciler resolves ONE overlapping file. Given the file's path and the three
# byte views (the live tree's base when the node opened, the sibling's already-
# merged bytes now in the live tree, and the node's own new bytes), it returns the
# merged bytes to write. Async so the caller can route it through a worker
# (``ReconciliationReview`` / the critic chain); tests pass a trivial stub.
Reconciler = Callable[[str, Optional[bytes], bytes, bytes], Awaitable[bytes]]


@dataclass
class MergeOutcome:
    """The result of folding ONE DAG node's writes into the live tree (§3.5).

    ``overlapping_paths`` lists the files a sibling had ALSO written since the
    node opened. ``conflict`` is True only when the policy could not safely
    resolve an overlap (``disjoint``/``fail`` on any overlap). ``resolution`` is
    a short human string for ``meta.json`` ("disjoint", "reconciled",
    "aborted:disjoint-policy", "aborted:fail-policy").
    """

    node_id: str
    layer: int
    overlapping_paths: List[str] = field(default_factory=list)
    policy: str = DEFAULT_MERGE_POLICY
    conflict: bool = False
    resolution: str = ""

    def to_dict(self) -> Dict[str, object]:
        """Serialize for the ``meta.json`` ``graph.merges`` block."""
        return {
            "node_id": self.node_id,
            "layer": self.layer,
            "overlapping_paths": list(self.overlapping_paths),
            "policy": self.policy,
            "conflict": self.conflict,
            "resolution": self.resolution,
        }


class MergeConflict(RuntimeError):
    """Raised when a node's merge overlaps a sibling under an abort policy.

    Carries the offending paths + the recorded :class:`MergeOutcome` so the
    graph walker can surface the conflict (and its node id) in ``meta.json``
    before re-raising to fail the run fast.
    """

    def __init__(self, outcome: MergeOutcome):
        self.outcome = outcome
        paths = ", ".join(outcome.overlapping_paths)
        super().__init__(
            f"node '{outcome.node_id}' overlaps an already-merged sibling on "
            f"[{paths}] under --merge-policy {outcome.policy}"
        )


def _node_changes(src_snapshot: Dict[str, bytes], base_snapshot: Dict[str, bytes]) -> Dict[str, bytes]:
    """Files the node added/modified vs its open-time base (path -> new bytes).

    Deletions are handled separately by the apply path; the overlap question is
    only meaningful for files the node WROTE."""
    changed: Dict[str, bytes] = {}
    for rel, data in src_snapshot.items():
        if base_snapshot.get(rel) != data:
            changed[rel] = data
    return changed


async def merge_node(
    *,
    node_id: str,
    layer: int,
    src_ws: Path,
    live_tree: Path,
    base_snapshot: Dict[str, bytes],
    policy: str,
    reconcile_station: Optional[Reconciler],
    sem: asyncio.Semaphore,
) -> MergeOutcome:
    """Fold ONE completed node's writes into ``live_tree`` under ``policy``.

    Runs under ``sem`` (the merge semaphore) so two concurrently-finishing nodes
    never write the live tree at once — overlap detection relies on the live tree
    being quiescent for the duration of this call.

    For each file the node changed (vs ``base_snapshot``), the live tree's
    current bytes are compared to ``base_snapshot``: equal => disjoint (apply);
    differing => an already-merged sibling touched the path => OVERLAP, resolved
    per ``policy``. Disjoint deletions (files the node removed and the sibling
    didn't re-create) are applied too. Returns a :class:`MergeOutcome`; raises
    :class:`MergeConflict` when an abort policy hits an overlap.
    """
    if policy not in MERGE_POLICIES:
        raise ValueError(f"merge policy must be one of {MERGE_POLICIES} (got {policy!r})")

    src_ws = Path(src_ws)
    live_tree = Path(live_tree)

    async with sem:
        # Snapshot the node's final workspace + the live tree's CURRENT state
        # (which already carries any earlier-merged sibling's writes). Off-thread:
        # the reads can block the loop on large trees.
        src_snapshot = await asyncio.to_thread(_snapshot_tree, src_ws)
        live_snapshot = await asyncio.to_thread(_snapshot_tree, live_tree)

        changed = _node_changes(src_snapshot, base_snapshot)

        overlapping: List[str] = []
        disjoint: Dict[str, bytes] = {}
        for rel, new_bytes in changed.items():
            base_bytes = base_snapshot.get(rel)
            live_bytes = live_snapshot.get(rel)
            # A sibling touched this path iff the live tree no longer matches the
            # base this node opened against. (If the sibling wrote the IDENTICAL
            # bytes the node would too, live == new_bytes; still an overlap by
            # provenance, but harmless — apply is a no-op. We classify on base to
            # match the spec's "did a sibling touch this path".)
            if live_bytes != base_bytes:
                overlapping.append(rel)
            else:
                disjoint[rel] = new_bytes

        # Files the node DELETED (present at open, gone now): apply only when the
        # live tree still matches the node's base for that path (no sibling
        # re-created/changed it) — same disjoint discipline.
        deletions: List[str] = []
        for rel in base_snapshot:
            if rel in src_snapshot:
                continue  # not a deletion
            if live_snapshot.get(rel) == base_snapshot.get(rel):
                deletions.append(rel)
            # else: a sibling changed/removed the path; leave it (already counted
            # only if the node ALSO wrote it, which it didn't here).

        outcome = MergeOutcome(
            node_id=node_id, layer=layer, overlapping_paths=sorted(overlapping),
            policy=policy,
        )

        # Abort policies: any overlap stops the merge BEFORE writing anything.
        if overlapping and policy in ("disjoint", "fail"):
            outcome.conflict = True
            outcome.resolution = (
                "aborted:disjoint-policy" if policy == "disjoint" else "aborted:fail-policy"
            )
            logger.warning(
                "Merge ABORT (node '%s', --merge-policy %s): overlapping write(s) on %s.",
                node_id, policy, outcome.overlapping_paths,
            )
            raise MergeConflict(outcome)

        # Apply disjoint writes + disjoint deletions off-thread.
        await asyncio.to_thread(
            _apply_disjoint, live_tree, disjoint, deletions,
        )

        if not overlapping:
            outcome.resolution = "disjoint"
            return outcome

        # reconcile policy with overlaps: resolve each overlapping path via the
        # station, then write the merged bytes. The join node's verifier re-runs
        # on the merged tree (in the master loop), so a bad merge is gated.
        if reconcile_station is None:
            # Defensive: reconcile policy selected but no station wired. Treat as
            # a hard conflict rather than silently clobbering.
            outcome.conflict = True
            outcome.resolution = "aborted:no-reconciler"
            raise MergeConflict(outcome)

        for rel in outcome.overlapping_paths:
            base_bytes = base_snapshot.get(rel)
            sibling_bytes = live_snapshot.get(rel) or b""
            node_bytes = changed[rel]
            merged = await reconcile_station(rel, base_bytes, sibling_bytes, node_bytes)
            await asyncio.to_thread(_write_file, live_tree, rel, merged)
        outcome.resolution = "reconciled"
        logger.info(
            "Merge RECONCILED (node '%s'): %d overlapping path(s) resolved via the "
            "reconcile station; verifier re-runs on the merged tree.",
            node_id, len(outcome.overlapping_paths),
        )
        return outcome


def _apply_disjoint(
    live_tree: Path, disjoint: Dict[str, bytes], deletions: List[str]
) -> None:
    """Write the node's disjoint adds/mods and remove its disjoint deletions.

    Mirrors ``apply_node_writes``'s disjoint path but scoped to the already-
    classified set so an overlapping file is NEVER touched here (it's reconciled
    or aborted by the caller)."""
    for rel, data in disjoint.items():
        _write_file(live_tree, rel, data)
    for rel in deletions:
        dst_path = live_tree / rel
        if not dst_path.exists():
            continue
        try:
            dst_path.unlink()
        except OSError:
            continue
        parent = dst_path.parent
        while parent != live_tree:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def _write_file(live_tree: Path, rel: str, data: bytes) -> None:
    """Write ``data`` to ``live_tree/rel``, skipping an identical-bytes no-op."""
    dst_path = live_tree / rel
    if dst_path.exists():
        try:
            if dst_path.read_bytes() == data:
                return
        except OSError:
            pass
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_bytes(data)

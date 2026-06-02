import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from agy_orchestrator.core.agents.agy_agent import AgyAgent
from agy_orchestrator.execution.graph_plan import GraphPlan, PlanNode
from agy_orchestrator.execution.verifier import QualityVerifier, VerifierResult
from agy_orchestrator.execution.workspace import (
    _ManagedWorkspace,
    _snapshot_tree,
)
from agy_orchestrator.workflows.adversarial import AdversarialReview
from agy_orchestrator.workflows.graph_merge import (
    DEFAULT_MERGE_POLICY,
    MERGE_POLICIES,
    MergeConflict,
    MergeOutcome,
    Reconciler,
    merge_node,
)
from agy_orchestrator.workflows.tree_of_thought import TreeOfThought

logger = logging.getLogger(__name__)

# Matches "--- Step N Summary ---" so we can split project_context into
# per-step blocks for the two-tier compaction below. Tolerant of leading
# whitespace and trailing punctuation/numbers; the master writer uses an
# exact format but a future variant shouldn't blow this up.
_STEP_HEADER_RE = re.compile(r"^---\s*Step\s+(\d+)\s+Summary\s*---\s*$", re.MULTILINE)

# How many recent step summaries to keep VERBATIM in compacted context.
# 2 chosen empirically: the adversarial critic on step N+1 needs immediate
# visibility into the file paths and symbol names from N and N-1; older
# detail is fine to digest. Bump via MasterWorkflow(recent_steps_verbatim=K).
DEFAULT_RECENT_STEPS_VERBATIM = 2


def _truncate_orch_title(value: object) -> str:
    return str(value)[:120]


@dataclass
class StepResult:
    """The outcome of executing one task — what ``_execute_step`` returns.

    Carries the step's worker output, the compact ``step_summary`` threaded into
    later steps' context, the run-level confidence signals (mirrored from the
    step's ``AdversarialReview``), and the (possibly updated) ``session_id``. The
    DAG executor stores one per completed node; the linear loop unpacks it inline
    so its observable behavior is byte-identical to the pre-refactor loop body.
    """

    final_output: str
    step_summary: str
    verified: bool
    approved: bool
    stalled: bool
    iterations_used: int
    session_id: Optional[str]


class _SemaphoreGatedVerifier:
    """Wraps a :class:`QualityVerifier` so its ``verify`` runs under a shared
    semaphore (docs §5 M3).

    K concurrent DAG nodes each build their own ``AdversarialReview`` around this
    wrapper; the SHARED ``sem`` (capacity ``verifier_concurrency``, default 1)
    means at most N of the K nodes run the local ``make check`` spike at once, so
    a wide layer can't fire K full verifiers in parallel and OOM the box — the
    same layering vote uses. Delegates the verifier's full interface via the one
    call sites use (``verify``); any other attribute proxies to the wrapped
    verifier so ``getattr(verifier, ...)`` downstream still works.
    """

    def __init__(self, verifier: QualityVerifier, sem: "asyncio.Semaphore"):
        self._verifier = verifier
        self._sem = sem

    async def verify(self, working_directory: str) -> VerifierResult:
        async with self._sem:
            return await self._verifier.verify(working_directory=working_directory)

    def __getattr__(self, name):
        # Proxy everything else (e.g. test_commands, timeout) to the real verifier.
        return getattr(self._verifier, name)


def _normalized_linearization(steps: List[str]) -> List[str]:
    """Canonical step-string list used by the checkpoint-equality guard.

    Both the checkpoint's tasks and ``self.plan_steps`` are already flat step
    strings (the loader linearizes a graph plan via ``GraphPlan.as_steps()``
    before it reaches here), so this is the identity on a flat list. Routing the
    comparison through ``ChainPlan`` keeps the intent explicit — "compare the
    normalized linearization, not the raw lists" — so an injected graph whose
    topo-order matches the checkpoint still resumes (docs §5 M1)."""
    from agy_orchestrator.execution.graph_plan import ChainPlan

    return ChainPlan(steps=list(steps)).as_steps()


def _has_nonlinear_deps(graph: GraphPlan) -> bool:
    """True iff a graph plan is NOT a pure linear chain (auto-route gate, §3.1).

    A graph is "linear" when its topological order is a simple chain: the k-th
    node (1-indexed) depends on exactly the (k-1)-th node and nothing else (the
    first depends on nothing). Any branch (a node with >1 dep, a node fanned out
    to >1 dependent, or a dep that skips/reorders the chain) is non-linear and
    routes to ``_execute_graph``. A flat plan lifted via ``ChainPlan.as_graph``
    is linear by construction, so it never enters the graph walker.
    """
    order = graph.topo_order()
    by_id = {n.id: n for n in graph.nodes}
    for idx, nid in enumerate(order):
        # A pure chain: node k depends on exactly node k-1 (first on nothing). A
        # fan-out (one node depended on by two) is caught here too — the second
        # dependent's dep would not be its immediate topo predecessor.
        expected = [order[idx - 1]] if idx > 0 else []
        if list(by_id[nid].deps) != expected:
            return True
    return False


class MasterWorkflow:
    """
    Combines Tree of Thought and Adversarial Review to manage and execute
    large, complex projects accurately.
    """
    def __init__(
        self,
        model: str,
        effort: str,
        branches: int = 3,
        max_iterations: int = 5,
        verifier: Optional[QualityVerifier] = None,
        agent_class=AgyAgent,
        checkpoint_path: Optional[str] = None,
        compaction_interval: int = 6,
        max_context_chars: int = 12000,
        selector: str = "judge",
        working_directory: str = ".",
        recent_steps_verbatim: int = DEFAULT_RECENT_STEPS_VERBATIM,
        event_callback: Optional[Callable[[dict], None]] = None,
        resume_policy: str = "auto",
        plan_only: bool = False,
        plan_steps: Optional[List[str]] = None,
        plan_graph: Optional[GraphPlan] = None,
        max_parallel_workers: Optional[int] = None,
        verifier_concurrency: int = 1,
        merge_policy: str = DEFAULT_MERGE_POLICY,
        reconcile_station_factory: Optional[Callable[..., "Reconciler"]] = None,
    ):
        self.model = model
        self.effort = effort
        self.branches = branches
        self.max_iterations = max_iterations
        self.verifier = verifier
        self.agent_class = agent_class
        # Two-tier compaction: when the running context is digested, the
        # most recent N step summaries are kept VERBATIM and only older
        # steps go through the compactor. Avoids the "I lost the file
        # paths from the step I just did" failure mode that hurt long
        # master runs under the single-digest scheme. Anchor:
        # arxiv 2509.13313 (ReSum) — periodic external summarization with
        # bounded recent context improved long-horizon search by +8.2%.
        self.recent_steps_verbatim = max(0, int(recent_steps_verbatim))
        # Where the verifier should run the test_cmd. Threaded into every
        # AdversarialReview the master spawns so cross-repo dispatches
        # (caller's `out_dir != PROJECT_ROOT`) verify in the right tree.
        self.working_directory = working_directory
        # ToT selection: "judge" (an evaluator scores each branch — right for the
        # diverse code outputs here) or "vote" (free, but rarely clusters for code).
        self.selector = selector
        self.checkpoint_path = checkpoint_path
        # Checkpoint-resume safety (issue #37). The #31 checkpoint is keyed only by
        # the prompt hash, so a re-dispatch resumes "from the last completed step"
        # even when the out-dir was reset/reverted underneath it between runs —
        # silently building later steps on a tree missing the earlier steps' edits.
        # `resume_policy` controls how a matching checkpoint is treated:
        #   "auto"  (default) — resume only if the out-dir still fingerprints to the
        #                       tree the checkpoint was saved against; otherwise
        #                       DISCARD the stale checkpoint and start fresh (safe).
        #   "force" (--resume) — resume even when the base diverged (operator override).
        #   "never" (--fresh)  — ignore any checkpoint; always start clean.
        if resume_policy not in ("auto", "force", "never"):
            raise ValueError(
                f"resume_policy must be 'auto', 'force', or 'never' (got {resume_policy!r})"
            )
        self.resume_policy = resume_policy
        # Plan-only / dry-run (#41): run just the planner, emit the decomposed
        # step plan, and exit BEFORE any worker mutates the out-dir. Lets the
        # operator audit/approve the decomposition for the price of one planner
        # call instead of an execute-then-reset cycle.
        self.plan_only = plan_only
        # Operator-supplied plan (plan injection): when provided, master executes
        # THESE steps verbatim and skips the planner — closing the round-trip with
        # the plan.json that --plan-only emits (generate -> review/edit -> --plan
        # <file> -> execute). The attribute is dual-purpose: plan-only mode also
        # (re)writes it with the freshly emitted plan so the harness can serialize
        # it to plan.json.
        self.plan_steps: Optional[List[str]] = list(plan_steps) if plan_steps else None
        # Operator-supplied dependency DAG (graph execution, docs §3.4). When a
        # graph plan with non-linear deps is present, ``execute`` routes to the
        # serial ``_execute_graph`` walker instead of the flat loop; otherwise the
        # default linear path is untouched. A flat / implicitly-linear graph runs
        # the existing loop verbatim (the auto-route gates strictly on the presence
        # of a non-linear dep). ``plan_steps`` stays the canonical linearization so
        # the planner/checkpoint round-trip and back-compat are unchanged.
        self.plan_graph: Optional[GraphPlan] = plan_graph
        if plan_graph is not None and not self.plan_steps:
            # Keep plan_steps in sync with the graph's stable linearization so the
            # checkpoint-equality guard and plan-only emit have a flat view.
            self.plan_steps = list(plan_graph.as_steps())
        # Concurrency caps for the DAG walker (docs §5 M3). ``max_parallel_workers``
        # bounds how many whole DAG nodes (gen + ToT + verify) run at once via
        # ``self._node_sem``; ``verifier_concurrency`` (default 1) serializes the
        # local ``make check`` spike so K parallel nodes never run K full verifiers
        # at once and OOM the box — the exact layering vote uses. Both default to
        # the env (AGY_MAX_PARALLEL_NODES / AGY_VERIFIER_CONCURRENCY) when unset, so
        # the harness can pass through without re-resolving. Linear runs ignore both.
        if max_parallel_workers is None:
            _env_mpn = os.environ.get("AGY_MAX_PARALLEL_NODES", "")
            max_parallel_workers = int(_env_mpn) if _env_mpn.strip().isdigit() else None
        self.max_parallel_nodes: Optional[int] = (
            max(1, int(max_parallel_workers)) if max_parallel_workers else None
        )
        self.verifier_concurrency = max(1, int(verifier_concurrency))
        # Merge policy for overlapping parallel writes (docs §4 / M4). Default is
        # operator-confirmed ``reconcile`` (the safe superset: disjoint writes
        # auto-apply, overlaps go to the #43 reconcile station). ``disjoint`` /
        # ``fail`` abort on any overlap. ``reconcile_station_factory`` builds the
        # per-overlap reconciler (an async callable resolving one file) — wired by
        # the harness from the critic chain; None means reconcile policy degrades
        # to a hard conflict (no clobber). The DAG walker collects one
        # MergeOutcome per node into ``merge_outcomes`` for meta.json (M5 surface).
        if merge_policy not in MERGE_POLICIES:
            raise ValueError(
                f"merge_policy must be one of {MERGE_POLICIES} (got {merge_policy!r})"
            )
        self.merge_policy = merge_policy
        self.reconcile_station_factory = reconcile_station_factory
        self.merge_outcomes: List[MergeOutcome] = []
        # Graph observability for meta.json (docs §5 M5). ``node_status`` maps each
        # DAG node id -> "passed"/"failed"/"pending" as the frontier scheduler
        # harvests it; ``node_results`` keeps the per-node StepResult so the meta
        # writer can read verified/approved without re-deriving. Both stay empty on
        # the linear path (the harness getattr's them, so a flat run is untouched).
        self.node_status: Dict[str, str] = {}
        self.node_results: Dict[str, StepResult] = {}
        # Semaphores + the merge lock are bound lazily in _execute_graph (they
        # must live on the running loop, like vote's). None until the graph walker
        # arms them. ``_merge_lock`` serializes the per-node write-back so two
        # concurrently-finishing nodes never mutate the live tree at once (M3
        # applies disjoint writes; M4 adds overlap detection under the same lock).
        self._node_sem: Optional[asyncio.Semaphore] = None
        self._verify_sem: Optional[asyncio.Semaphore] = None
        self._merge_lock: Optional[asyncio.Lock] = None
        # Session compaction: over a long chained run the resumed workflow session
        # (full transcript re-sent every step) and the growing project_context are the
        # token-cost drivers. Every ``compaction_interval`` steps, OR whenever
        # project_context exceeds ``max_context_chars``, condense the context and RESET
        # the session so the accumulated transcript is shed. 0/negative disables.
        self.compaction_interval = int(compaction_interval)
        self.max_context_chars = int(max_context_chars)
        self.event_callback = event_callback
        # Run-level confidence signals (#45). Master never set these before, so the
        # harness/ledger read the class default (False) on every master run and
        # mislabeled good builds as unverified. The step loop propagates the FINAL
        # accepted step's AdversarialReview signals up onto these attrs. build_ledger
        # (execution/ledger.py) reads exactly these names via getattr.
        self.verified = False
        self.approved = False
        self.stalled = False
        self.iterations_used = 0

    def _emit_orchestration(self, **fields) -> None:
        cb = self.event_callback
        if cb is None:
            return
        orchestration = {"workflow": "master"}
        for key, value in fields.items():
            if value is not None:
                orchestration[key] = value
        try:
            cb(
                {
                    "kind": "lifecycle",
                    "data": {
                        "event": "orchestration_transition",
                        "orchestration": orchestration,
                    },
                }
            )
        except Exception:
            pass

    def _should_compact(self, steps_since_compaction: int, project_context: str) -> bool:
        if self.compaction_interval and steps_since_compaction >= self.compaction_interval:
            return True
        if self.max_context_chars and len(project_context) >= self.max_context_chars:
            return True
        return False

    def _split_context_into_steps(
        self, project_context: str
    ) -> Tuple[str, List[str]]:
        """Split project_context into (preamble, [per-step block, ...]).

        The preamble is the "Original Goal:" header. Each per-step block is
        the text from one ``--- Step N Summary ---`` line up to (but not
        including) the next one. Order is preserved; if no step markers are
        found the whole string is returned as the preamble with an empty list
        of blocks. The caller can rejoin blocks with ``''.join(...)`` since
        each block already starts with its own ``\n--- Step N ...`` header.
        """
        matches = list(_STEP_HEADER_RE.finditer(project_context))
        if not matches:
            return project_context, []
        # Preamble = everything before the first step header. The header line
        # itself belongs to its step (preserves the leading `\n---` form).
        first_start = matches[0].start()
        # Backwards-include the preceding "\n" so each block reads as a
        # standalone segment when rejoined.
        preamble_end = first_start
        if preamble_end > 0 and project_context[preamble_end - 1] == "\n":
            preamble_end -= 1
        preamble = project_context[:preamble_end]
        blocks: List[str] = []
        for i, m in enumerate(matches):
            start = m.start()
            # Walk back one char if the marker is preceded by "\n" so each
            # block carries its leading newline. This makes joining trivial.
            if start > 0 and project_context[start - 1] == "\n":
                start -= 1
            # The end of block i is "just before block i+1's leading \n",
            # which is matches[i+1].start() - 1. No further walk-back —
            # that boundary newline already lives at start-1 of block i+1.
            if i + 1 < len(matches):
                end = matches[i + 1].start() - 1
            else:
                end = len(project_context)
            blocks.append(project_context[start:end])
        return preamble, blocks

    async def _compact_context(self, initial_prompt: str, project_context: str) -> str:
        """Condense older steps; keep recent N steps verbatim (two-tier).

        For a long master run, the single-digest scheme of the prior version
        had a subtle failure mode: after compaction, the immediately
        preceding step's concrete details (file paths, symbol names) were
        absorbed into a paragraph-prose digest and the next critic
        iteration couldn't tell which exact file the previous step had
        touched. Two-tier compaction sidesteps this — older steps go through
        the digest (cheap context), the LAST ``recent_steps_verbatim`` step
        summaries are kept whole (rich, precise context for the next step).
        Anchor: arxiv 2509.13313.

        Falls open on edge cases — if there aren't enough step markers to
        split, behaves identically to the prior single-digest scheme.
        """
        header = f"Original Goal: {initial_prompt}\n\n=== Accumulated Implementation (compacted) ===\n"
        preamble, blocks = self._split_context_into_steps(project_context)
        keep = self.recent_steps_verbatim

        # No step markers OR fewer steps than we want to keep verbatim:
        # nothing to split on. Compact the whole thing the old way.
        if not blocks or keep == 0 or len(blocks) <= keep:
            to_digest = project_context
            verbatim_suffix = ""
        else:
            older = blocks[:-keep]
            recent = blocks[-keep:]
            to_digest = preamble + "".join(older)
            verbatim_suffix = "".join(recent)

        # Bound the digest input — even with two-tier, an enormous older-tier
        # tail can blow up the compactor prompt. The 24K cap matches the
        # prior behaviour; the recent verbatim block is excluded from this
        # cap because preserving its full content is the whole point.
        compactor = self.agent_class(
            prompt=(
                "Condense the following project progress log into a TIGHT running digest "
                "(<= 1500 words). Preserve the original goal, every file created/modified, key "
                "design decisions, and any names (classes/functions/IDs/APIs) later steps need. "
                "Drop redundancy and full code. Output only the digest.\n\n"
                f"{to_digest[:24000]}"
            ),
            model=self.model,
            effort="low",
        )
        try:
            digest = await compactor.run_async()
            compacted = header + digest.strip() + "\n"
            if verbatim_suffix:
                compacted += "\n=== Recent steps (verbatim) ===" + verbatim_suffix + "\n"
            return compacted
        except Exception as exc:  # robust fallback: keep goal + most-recent tail
            logger.warning("Context compaction failed (%s); truncating to recent tail.", exc)
            tail = project_context[-self.max_context_chars :] if self.max_context_chars else project_context
            return header + tail

    def _checkpoint_key(self, initial_prompt: str) -> str:
        return hashlib.sha256(initial_prompt.encode("utf-8")).hexdigest()

    def _base_fingerprint(self) -> Optional[str]:
        """Fingerprint of the out-dir's working state, for resume safety (#37).

        Combines ``git rev-parse HEAD`` with a hash of ``git status --porcelain``
        so BOTH a commit move (reset to a different ref) AND a working-tree change
        (a ``git reset --hard`` that reverts the completed steps' *uncommitted*
        edits — workers don't commit in master mode) flip the fingerprint. Returns
        None when the out-dir isn't a git work tree or git is unavailable, in which
        case resume can't be verified and falls back to legacy (#31) behavior.
        """
        wd = self.working_directory or "."
        try:
            head = subprocess.run(
                ["git", "-C", str(wd), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            if head.returncode != 0:
                return None
            status = subprocess.run(
                ["git", "-C", str(wd), "status", "--porcelain"],
                capture_output=True, text=True, timeout=10,
            )
            if status.returncode != 0:
                return None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None
        head_ref = head.stdout.strip()
        status_hash = hashlib.sha256(status.stdout.encode("utf-8")).hexdigest()[:16]
        return f"{head_ref}:{status_hash}"

    def _load_checkpoint(
        self, initial_prompt: str
    ) -> Optional[Tuple[List[str], int, str, Optional[str]]]:
        """Return (tasks, completed, project_context, session_id) to resume, or None.

        Resume is gated by ``resume_policy`` and a base-fingerprint check (#37): a
        prompt-key match is no longer sufficient — the out-dir must still be the
        tree the checkpoint was saved against, or the operator must force it.
        """
        path = self.checkpoint_path
        if self.resume_policy == "never":
            if path and os.path.exists(path):
                logger.info(
                    "Master resume disabled (--fresh): ignoring checkpoint %s; starting clean.",
                    path,
                )
            return None
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.warning("Could not read checkpoint %s: %s", path, exc)
            return None
        if data.get("key") != self._checkpoint_key(initial_prompt):
            logger.info("Checkpoint %s is for a different project; starting fresh.", path)
            return None
        tasks = data.get("tasks") or []
        completed = int(data.get("completed", 0))
        if not tasks or completed >= len(tasks):
            logger.info("Checkpoint shows project already complete; nothing to resume.")
            return None
        # #37: verify the resume base still matches the tree we checkpointed against.
        stored_fp = data.get("base_fingerprint")
        current_fp = self._base_fingerprint()
        if stored_fp is None or current_fp is None:
            # Can't verify — a pre-#37 checkpoint or a non-git out-dir. Preserve the
            # #31 salvage behavior (resume) but SAY so, so a stale resume is at least
            # visible in the log instead of silent.
            logger.warning(
                "Master resuming checkpoint at step %d/%d but the base could NOT be "
                "fingerprint-verified (out-dir not a git tree, or a pre-#37 "
                "checkpoint); resuming on trust. Pass --fresh to start clean if the "
                "tree was reset since the checkpoint was written.",
                completed + 1, len(tasks),
            )
        elif stored_fp != current_fp:
            if self.resume_policy == "force":
                logger.warning(
                    "Master out-dir DIVERGED from the checkpoint base (expected %s, "
                    "found %s); --resume forces resume at step %d/%d anyway — later "
                    "steps may build on a tree missing earlier steps' edits.",
                    stored_fp, current_fp, completed + 1, len(tasks),
                )
            else:
                logger.warning(
                    "Master checkpoint base no longer matches the out-dir (expected "
                    "%s, found %s): the tree was reset / hand-edited / moved since "
                    "step %d completed. DISCARDING the stale checkpoint and starting "
                    "FRESH to avoid a silently inconsistent build (#37). Pass --resume "
                    "to force resume onto the current tree.",
                    stored_fp, current_fp, completed,
                )
                return None
        else:
            logger.info(
                "Master checkpoint base verified (%s); resuming at step %d/%d.",
                current_fp, completed + 1, len(tasks),
            )
        return tasks, completed, data.get("project_context", ""), data.get("session_id")

    def _load_graph_done_frontier(
        self, initial_prompt: str
    ) -> Optional[Dict[str, StepResult]]:
        """Read the checkpoint's per-node ``done`` frontier for graph resume (M5).

        Returns ``{node_id: StepResult}`` for the nodes that had completed when the
        checkpoint was written, or None when there is no graph frontier to seed (a
        linear checkpoint, a pre-M5 checkpoint without a ``done`` key, a key
        mismatch, or an unreadable file). Called only AFTER ``_load_checkpoint``
        approved the resume, so the #37 base_fingerprint gate already held — a
        divergent tree returns None there and this is never consulted."""
        path = self.checkpoint_path
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return None
        if data.get("key") != self._checkpoint_key(initial_prompt):
            return None
        done_raw = data.get("done")
        if not isinstance(done_raw, dict) or not done_raw:
            return None
        frontier: Dict[str, StepResult] = {}
        for nid, rec in done_raw.items():
            if not isinstance(rec, dict):
                continue
            frontier[nid] = StepResult(
                final_output="",  # not persisted; only the summary is needed downstream
                step_summary=str(rec.get("step_summary", "")),
                verified=bool(rec.get("verified", False)),
                approved=bool(rec.get("approved", False)),
                stalled=bool(rec.get("stalled", False)),
                iterations_used=int(rec.get("iterations_used", 0) or 0),
                session_id=None,
            )
        return frontier or None

    def _save_checkpoint(
        self,
        initial_prompt: str,
        tasks: List[str],
        completed: int,
        project_context: str,
        session_id: Optional[str],
        graph: Optional[List[dict]] = None,
        done: Optional[Dict[str, dict]] = None,
    ) -> None:
        if not self.checkpoint_path:
            return
        try:
            payload: Dict[str, object] = {
                "key": self._checkpoint_key(initial_prompt),
                "tasks": tasks,
                "completed": completed,
                "project_context": project_context,
                "session_id": session_id,
                # #37: snapshot the out-dir state THIS checkpoint was written
                # against, so a later resume can confirm the tree wasn't reset
                # underneath it. Reflects the post-step edits on disk.
                "base_fingerprint": self._base_fingerprint(),
            }
            # Graph resume (docs §5 M5): a DAG run also records its node DAG +
            # the per-node done frontier, so resume seeds completed nodes by id
            # (not just a "first K" count). Keys are OPTIONAL — a linear
            # checkpoint omits them and old readers ignore them (back-compat).
            if graph is not None:
                payload["graph"] = graph
            if done is not None:
                payload["done"] = done
            tmp = self.checkpoint_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self.checkpoint_path)  # atomic: crash mid-write can't corrupt
        except Exception as exc:
            logger.warning("Could not write checkpoint %s: %s", self.checkpoint_path, exc)

    async def _run_planner(self, initial_prompt: str) -> Tuple[List[str], Optional[str]]:
        """Planner phase: decompose the project into a JSON list of step prompts.

        Returns ``(tasks, session_id)``. Makes exactly one planner worker call and
        writes NOTHING to the out-dir — so it is safe to run in plan-only/dry-run
        mode (#41).
        """
        logger.info("Starting Master Workflow Planning Phase...")
        planner_prompt = (
            f"You are the Lead Architect. Break down the following complex project into a logical sequence of implementation steps.\n"
            f"Output ONLY a valid JSON list of strings, where each string is a detailed prompt for a single step.\n"
            f"Example: [\"Step 1: Setup project structure and core utilities...\", \"Step 2: Implement UI component X...\"]\n\n"
            f"Project Request:\n{initial_prompt}"
        )
        planner = self.agent_class(prompt=planner_prompt, model=self.model, effort="high")
        plan_output = await planner.run_async()

        # Capture session established by planner for reuse across all subsequent calls
        workflow_session_id = getattr(planner, "session_id", None)
        if workflow_session_id:
            logger.info("Workflow session established: %s", workflow_session_id)

        # Extract JSON list from plan_output
        tasks: List[str] = []
        try:
            start = plan_output.find('[')
            end = plan_output.rfind(']') + 1
            if start != -1 and end != 0:
                tasks = json.loads(plan_output[start:end])
            else:
                raise ValueError("No JSON array found.")
        except Exception as e:
            logger.warning(f"Failed to parse Planner output as JSON: {e}. Defaulting to a single step.")
            tasks = [initial_prompt]

        logger.info(f"Project broken down into {len(tasks)} steps.")
        return tasks, workflow_session_id

    def _emit_plan(self, tasks: List[str]) -> None:
        self._emit_orchestration(
            phase="plan",
            action="completed",
            step_total=len(tasks),
            step_titles=[_truncate_orch_title(task) for task in tasks],
        )

    async def execute(self, initial_prompt: str) -> str:
        # Plan-only / dry-run (#41): decompose, emit the plan, and STOP before any
        # worker touches the out-dir. Ignores checkpoints (always re-plans) and
        # writes no checkpoint, so it is fully side-effect-free on disk.
        if self.plan_only:
            if self.plan_steps:
                # An injected plan is emitted as-is (no point re-planning what the
                # operator just handed us); useful to echo/validate a plan file.
                tasks = list(self.plan_steps)
            else:
                tasks, _session_id = await self._run_planner(initial_prompt)
            self._emit_plan(tasks)
            self.plan_steps = list(tasks)
            logger.info(
                "Plan-only (dry-run): emitted %d-step plan; NOT executing — the "
                "out-dir was not modified.", len(tasks),
            )
            lines = [
                f"Plan-only dry-run: {len(tasks)} step(s). No worker wrote to the "
                f"out-dir; pass without --plan-only to execute.",
                "",
            ]
            for idx, task in enumerate(tasks, 1):
                lines.append(f"  {idx}. {task}")
            return "\n".join(lines)

        resumed = self._load_checkpoint(initial_prompt)
        # Per-node graph resume (docs §5 M5): when a DAG run resumes, seed the
        # frontier from the checkpoint's recorded ``done`` node-id map (NOT just a
        # "first K" count), so a diamond resumes exactly the nodes that hadn't
        # finished. ``_load_checkpoint`` already passed the #37 base_fingerprint
        # gate (a divergence returns None and discards the stale checkpoint), so
        # this only reads the done frontier for an approved resume. None on the
        # linear path / a fresh run / a pre-M5 checkpoint (no "done" key).
        start_completed_ids: Optional[Dict[str, StepResult]] = None
        if resumed is not None:
            start_completed_ids = self._load_graph_done_frontier(initial_prompt)
        if (resumed is not None and self.plan_steps
                and _normalized_linearization(resumed[0])
                != _normalized_linearization(self.plan_steps)):
            # An injected plan that differs from the saved checkpoint means the
            # operator edited the plan since the last run — the stale checkpoint
            # must not win. Drop it and run the supplied plan fresh. (A matching
            # checkpoint is still honored, so a crashed injected-plan run resumes.)
            logger.info(
                "Injected plan differs from the saved checkpoint; discarding the "
                "stale checkpoint and running the supplied plan."
            )
            resumed = None
            start_completed_ids = None
        if resumed is not None:
            tasks, start_index, project_context, workflow_session_id = resumed
            logger.info(
                "Resuming Master Workflow from checkpoint at step %d/%d (continuing in place).",
                start_index + 1,
                len(tasks),
            )
        elif self.plan_steps:
            # Plan injection: execute the operator-supplied steps verbatim, skip
            # the planner entirely.
            tasks = list(self.plan_steps)
            workflow_session_id = None
            logger.info(
                "Using operator-supplied plan (%d step(s)); skipping the planner.",
                len(tasks),
            )
            self._emit_plan(tasks)
            project_context = f"Original Goal: {initial_prompt}\n\n=== Accumulated Implementation ===\n"
            start_index = 0
            self._save_checkpoint(initial_prompt, tasks, start_index, project_context, workflow_session_id)
        else:
            tasks, workflow_session_id = await self._run_planner(initial_prompt)
            self._emit_plan(tasks)
            project_context = f"Original Goal: {initial_prompt}\n\n=== Accumulated Implementation ===\n"
            start_index = 0
            self._save_checkpoint(initial_prompt, tasks, start_index, project_context, workflow_session_id)

        # 2. Execution Loop. A graph plan with non-linear deps routes to the
        # serial DAG walker (docs §3.4); the default flat path is a thin caller
        # over _execute_step whose observable behavior is byte-identical to the
        # pre-refactor loop body.
        if self._is_graph_plan():
            return await self._execute_graph(
                initial_prompt, tasks, start_index, project_context,
                workflow_session_id, start_completed_ids,
            )

        steps_since_compaction = 0
        for i in range(start_index, len(tasks)):
            task = tasks[i]
            result = await self._execute_step(
                step_index=i + 1,
                step_total=len(tasks),
                task=task,
                project_context=project_context,
                workflow_session_id=workflow_session_id,
                working_directory=self.working_directory,
            )
            workflow_session_id = result.session_id

            # Propagate this step's verifier/critic signals up to the master object
            # (#45). Tracking the LAST step is correct: the run-level question is
            # "did the final accepted output pass a verifier." These are plain Python
            # attributes, so they SURVIVE the end-of-run context compaction (compaction
            # only resets workflow_session_id, not object attributes). When
            # self.verifier is None, adv.verified stays False — correct, no programmatic
            # verifier means not "verified"; adv.approved may still carry a critic OK.
            self.verified = result.verified
            self.approved = result.approved
            self.stalled = result.stalled
            self.iterations_used = result.iterations_used

            project_context += f"\n--- Step {i+1} Summary ---\n{result.step_summary}\n"
            # Persist progress so a crash resumes from the NEXT step, in place.
            self._save_checkpoint(initial_prompt, tasks, i + 1, project_context, workflow_session_id)

            # Session compaction: shed the accumulated transcript + condense context so a
            # multi-hour chained run does not carry an ever-growing session.
            steps_since_compaction += 1
            if self._should_compact(steps_since_compaction, project_context):
                logger.info(
                    "Compacting context + resetting session after step %d/%d (context was %d chars).",
                    i + 1, len(tasks), len(project_context),
                )
                project_context = await self._compact_context(initial_prompt, project_context)
                workflow_session_id = None  # next step starts a fresh session, shedding the transcript
                steps_since_compaction = 0
                self._save_checkpoint(initial_prompt, tasks, i + 1, project_context, workflow_session_id)

        # Run finished cleanly: drop the salvage checkpoint so a later identical
        # dispatch starts fresh instead of "resuming" a completed project, and so
        # the checkpoint dir holds only in-flight/abruptly-killed runs (issue #31).
        self._remove_checkpoint()
        logger.info("Master Workflow Complete!")
        return project_context

    async def _execute_step(
        self,
        *,
        step_index: int,
        step_total: int,
        task: str,
        project_context: str,
        workflow_session_id: Optional[str],
        working_directory: str,
        verifier=None,
    ) -> StepResult:
        """Execute ONE task (ToT explore -> adversarial refine -> summarize).

        This is the per-step quality machinery the master loop runs for every
        step, lifted verbatim out of the loop body so a DAG node can reuse it.
        ``working_directory`` is threaded into the verifier/adversarial review so
        a DAG node can point this at an isolated workspace WITHOUT _execute_step
        knowing anything about isolation (docs §3.1). The linear loop passes
        ``self.working_directory`` and gets byte-identical behavior.

        ``verifier`` defaults to ``self.verifier`` (the linear path, unchanged);
        a DAG node passes a semaphore-gated wrapper so K parallel nodes serialize
        their verifier spike (docs §5 M3).

        Returns a :class:`StepResult` carrying the worker output, the compact
        step summary, the run-level signals, and the (possibly updated) session
        id (a fresh session minted by the summarizer when none existed).
        """
        i = step_index - 1  # preserve the legacy "Step N" / log numbering exactly
        # Linear path: self.verifier (unchanged). DAG node: a semaphore-gated
        # wrapper so K parallel nodes don't run K verifiers at once (docs §5 M3).
        step_verifier = self.verifier if verifier is None else verifier
        logger.info(f"--- Executing Step {step_index}/{step_total} ---")
        logger.info(f"Task description: {task[:100]}...")
        self._emit_orchestration(
            phase="step",
            action="started",
            step_index=step_index,
            step_total=step_total,
            step_title=_truncate_orch_title(task),
            model=self.model,
            effort=self.effort,
        )

        step_prompt = (
            f"You are implementing Step {i+1} of a larger project.\n\n"
            f"Project Context (What has been built so far):\n{project_context}\n\n"
            f"Current Task to implement NOW:\n{task}"
        )

        # Phase A: Tree of Thought (Exploration). Skip entirely when branches<=1
        # — exploring + judging a single branch is pure overhead; go straight to
        # the adversarial refinement of the step prompt (cheap-mode early-exit).
        if self.branches <= 1:
            logger.info("Phase A: skipped (branches<=1) — refining the step directly.")
            best_tot_output = ""
        else:
            logger.info("Phase A: Tree of Thought Exploration")
            # ToT branches run fresh sessions — concurrent --fork-session on the same parent
            # causes race conditions. They're throwaway explorers anyway.
            tot_branches = [
                self.agent_class(prompt=step_prompt, model=self.model, effort=self.effort)
                for _ in range(self.branches)
            ]
            # The judge clones this evaluator per branch and scores them in
            # parallel with independent (sessionless) instances, so do NOT bind
            # the workflow session here — resuming it would both contaminate
            # per-branch scoring and pollute the main thread.
            tot_evaluator = self.agent_class(prompt="", model=self.model, effort="high")
            tot = TreeOfThought(
                tot_branches,
                tot_evaluator,
                selector=self.selector,
                event_callback=self.event_callback,
                # Ground branch scoring in THIS step's task so the judge picks
                # the branch that best solves the step, not the most internally
                # polished one (win 5).
                requirement=task,
            )
            best_tot_output = await tot.execute()

        # Phase B: Adversarial Review (Refinement) — resume main workflow session
        logger.info("Phase B: Adversarial Review Refinement")
        gen_kwargs = dict(model=self.model, effort=self.effort)
        critic_kwargs = dict(model=self.model, effort="high")
        if workflow_session_id:
            try:
                gen_kwargs["session_id"] = workflow_session_id
                critic_kwargs["session_id"] = workflow_session_id
            except Exception:
                pass
        adv_generator = self.agent_class(prompt=step_prompt, **gen_kwargs)

        if best_tot_output:
            adv_prompt = (
                f"{step_prompt}\n\n"
                f"Please refine, finalize, and perfect the following draft implementation. "
                f"Ensure it meets the highest standards and resolves any bugs:\n"
                f"{best_tot_output}"
            )
        else:
            # No exploration draft (branches<=1): implement the step directly.
            adv_prompt = step_prompt

        adv_critic = self.agent_class(prompt="", **critic_kwargs)

        adv = AdversarialReview(
            generator_instance=adv_generator,
            critic_instance=adv_critic,
            verifier=step_verifier,
            max_iterations=self.max_iterations,
            working_directory=working_directory,
            event_callback=self.event_callback,
        )

        final_step_output = await adv.execute(adv_prompt)

        # Capture the step's verifier/critic signals (#45). The caller decides how
        # to fold them up (the linear loop tracks the LAST step; the graph walker
        # tracks the final topo node). When self.verifier is None, adv.verified
        # stays False — correct, no programmatic verifier means not "verified";
        # adv.approved may still carry a critic OK.
        verified = bool(getattr(adv, "verified", False))
        approved = bool(getattr(adv, "approved", False))
        stalled = bool(getattr(adv, "stalled", False))
        iterations_used = int(getattr(adv, "iterations_used", 0) or 0)

        logger.info(f"Step {step_index} Completed. Summarizing for project context.")
        self._emit_orchestration(
            phase="step",
            action="completed",
            step_index=step_index,
            step_total=step_total,
            step_title=_truncate_orch_title(task),
        )
        # Summarize the step output to keep project_context compact.
        # Passing full HTML/code outputs into every subsequent prompt balloons to 50KB+.
        summarize_kwargs = dict(model=self.model, effort="low")
        if workflow_session_id:
            try:
                summarize_kwargs["session_id"] = workflow_session_id
            except Exception:
                pass
        summarizer = self.agent_class(
            prompt=(
                f"In 3-5 bullet points, summarize what was just implemented in Step {i+1}.\n"
                f"Focus on: what files were created/modified, key design decisions, and any\n"
                f"CSS classes, JS functions, or IDs that other steps should know about.\n"
                f"Be concise. Do NOT reproduce the full code.\n\nStep output:\n{final_step_output[:8000]}"
            ),
            **summarize_kwargs
        )
        try:
            step_summary = await summarizer.run_async()
            # Update workflow session from summarizer if we don't have one yet
            if not workflow_session_id:
                workflow_session_id = getattr(summarizer, "session_id", None)
        except Exception as e:
            logger.warning(f"Summarizer failed ({e}), falling back to task description.")
            step_summary = f"Completed: {task[:300]}"

        return StepResult(
            final_output=final_step_output,
            step_summary=step_summary,
            verified=verified,
            approved=approved,
            stalled=stalled,
            iterations_used=iterations_used,
            session_id=workflow_session_id,
        )

    def _is_graph_plan(self) -> bool:
        """True iff an operator-supplied graph has at least one non-linear dep.

        Auto-route gate (docs §3.1): a flat plan, OR a graph whose deps form a
        pure linear chain (``s_k`` depends only on ``s_{k-1}``), runs the existing
        flat loop verbatim. Only a genuine branch/join (a node with >1 dep, a node
        depended on by >1 node, or a seed-after-first) enters ``_execute_graph``.
        """
        g = self.plan_graph
        if g is None:
            return False
        return _has_nonlinear_deps(g)

    def _compose_context(
        self,
        node: PlanNode,
        graph: GraphPlan,
        done: Dict[str, StepResult],
        initial_prompt: str,
    ) -> str:
        """Build a node's project_context from its ANCESTOR closure only.

        ``header(goal) + "".join(done[a].step_summary for a in ancestors(node))``
        in topological order (docs §3.4). A join node sees the UNION of both
        branches' summaries; two parallel branches each see ONLY their own
        ancestors (the key correctness property). For a linear plan every node's
        ancestor set is "all prior steps", so this reproduces today's cumulative
        ``\\n--- Step N Summary ---`` string exactly.
        """
        context = f"Original Goal: {initial_prompt}\n\n=== Accumulated Implementation ===\n"
        order = graph.topo_order()
        # Step numbers mirror the linear loop's "Step N" so a linear graph's
        # composed context is byte-identical to the flat path's.
        step_no = {nid: idx + 1 for idx, nid in enumerate(order)}
        for anc in graph.ancestors(node.id):
            context += f"\n--- Step {step_no[anc]} Summary ---\n{done[anc].step_summary}\n"
        return context

    async def _execute_graph(
        self,
        initial_prompt: str,
        tasks: List[str],
        start_index: int,
        project_context: str,
        workflow_session_id: Optional[str],
        start_completed_ids: Optional[Dict[str, "StepResult"]] = None,
    ) -> str:
        """Run the DAG with a FRONTIER scheduler — true wall-clock parallelism (M3).

        Not a barrier-per-level walk: a ready node is scheduled as soon as ALL its
        deps are done, so a fast branch's successors start while a slow sibling is
        still running. Concurrency is bounded two ways (the exact layering vote
        uses):
          * ``self._node_sem`` (capacity ``max_parallel_nodes``) bounds whole
            nodes (gen + ToT + verify); ``--max-parallel-nodes 1`` serializes a
            wide layer.
          * a SHARED verifier semaphore (``verifier_concurrency``, default 1),
            threaded into each node's verifier via :class:`_SemaphoreGatedVerifier`,
            serializes the local ``make check`` spike so K parallel nodes never run
            K full verifiers at once and OOM the box.

        Each node runs in its OWN isolated ``_ManagedWorkspace`` (reused from vote)
        and its writes are mirrored back via :func:`apply_workspace` (M3 has no
        overlap detection — diamond DAGs with DISJOINT writes work; overlapping
        writes are last-writer-wins, a documented temporary limitation until M4).
        DAG nodes run sessionless (concurrent --fork-session on one parent races;
        ToT branches are already sessionless for the same reason).
        """
        graph = self.plan_graph
        assert graph is not None, "_execute_graph requires a plan_graph (gated by _is_graph_plan)"
        order = graph.topo_order()
        by_id = {n.id: n for n in graph.nodes}
        step_no = {nid: idx + 1 for idx, nid in enumerate(order)}
        step_total = len(order)

        # Resume seeding (docs §5 M5): prefer the per-node ``done`` frontier the
        # checkpoint recorded (``start_completed_ids``) — it carries the EXACT set
        # of finished node ids (+ their summaries), so a diamond that crashed after
        # B done resumes C and D from the frontier, not a "first K topo" guess. Fall
        # back to the count-based seed (M3) for a pre-M5 checkpoint that only has a
        # ``completed`` int. Either way the seeded nodes are marked done so the
        # scheduler skips them.
        done: Dict[str, StepResult] = {}
        if start_completed_ids:
            for nid, res in start_completed_ids.items():
                if nid in by_id:  # ignore ids no longer in the (possibly edited) plan
                    done[nid] = res
        else:
            for nid in order[:start_index]:
                done[nid] = StepResult(
                    final_output="", step_summary=f"Completed: {by_id[nid].task[:300]}",
                    verified=False, approved=False, stalled=False,
                    iterations_used=0, session_id=None,
                )
        # Graph observability (docs §5 M5): node_status drives the meta.json graph
        # block. Seed completed nodes as "passed" and the rest "pending"; the
        # harvest flips each to "passed"/"failed" as it finishes.
        self.node_status = {nid: ("passed" if nid in done else "pending") for nid in order}
        self.node_results = dict(done)

        # Bind the concurrency gates to the running loop (vote's pattern). When no
        # cap is set the node semaphore is effectively unbounded (all-ready run at
        # once); the verifier semaphore always exists (default 1).
        self._node_sem = (
            asyncio.Semaphore(self.max_parallel_nodes)
            if self.max_parallel_nodes else None
        )
        self._verify_sem = asyncio.Semaphore(self.verifier_concurrency)
        self._merge_lock = asyncio.Lock()

        base_work_dir = self.working_directory
        running: Dict[str, asyncio.Task] = {}

        def _schedule_ready() -> None:
            for nid in order:
                if nid in done or nid in running:
                    continue
                node = by_id[nid]
                if all(d in done for d in node.deps):
                    running[nid] = asyncio.create_task(
                        self._run_dag_node(
                            node, step_no[nid], step_total, done,
                            base_work_dir, initial_prompt,
                        )
                    )

        try:
            _schedule_ready()
            while running:
                tasks_now = list(running.values())
                completed_tasks, _pending = await asyncio.wait(
                    tasks_now, return_when=asyncio.FIRST_COMPLETED
                )
                # Map finished tasks back to their node ids and harvest results.
                finished_ids = [nid for nid, t in running.items() if t in completed_tasks]
                for nid in finished_ids:
                    task = running.pop(nid)
                    # Re-raise a node failure AFTER cancelling siblings (fail-fast):
                    # an unhandled node exception aborts the run; in-flight siblings
                    # are cancelled so the loop doesn't orphan tasks.
                    result = task.result()
                    done[nid] = result
                    # Observability (M5): mark the node passed/failed for meta.json.
                    self.node_status[nid] = "passed" if result.verified or result.approved else "failed"
                    self.node_results[nid] = result
                    # Run-level signals track the final topo node (the leaf the
                    # build converges on), mirroring the linear loop's "last step
                    # wins" (#45). Re-assign in topo order each harvest so the last
                    # finished leaf in topo order wins regardless of completion race.
                    last_done = max(
                        (n for n in order if n in done), key=lambda n: order.index(n)
                    )
                    self.verified = done[last_done].verified
                    self.approved = done[last_done].approved
                    self.stalled = done[last_done].stalled
                    self.iterations_used = done[last_done].iterations_used
                # Checkpoint the done frontier so a crash resumes the remaining
                # nodes (docs §5 M5): persist the node DAG + the per-node ``done``
                # map (id -> summary/signals) so resume seeds the exact frontier,
                # not a "first K topo" guess. ``completed`` stays for linear
                # back-compat; the #37 base_fingerprint gate is unchanged.
                completed = sum(1 for n in order if n in done)
                rebuilt = self._rebuild_linear_context(initial_prompt, order, by_id, done)
                self._save_checkpoint(
                    initial_prompt, tasks, completed, rebuilt, None,
                    graph=[{"id": n.id, "task": n.task, "deps": list(n.deps)} for n in graph.nodes],
                    done={n: self._step_result_to_checkpoint(done[n]) for n in done},
                )
                # Release nodes the just-finished frontier unblocked.
                _schedule_ready()
        except BaseException:
            # Fail-fast: cancel any in-flight siblings so a failing/aborted node
            # doesn't leave orphaned tasks (and their open workspaces) running.
            for t in running.values():
                t.cancel()
            if running:
                await asyncio.gather(*running.values(), return_exceptions=True)
            raise

        self._remove_checkpoint()
        logger.info("Master Workflow Complete!")
        return self._rebuild_linear_context(initial_prompt, order, by_id, done)

    async def _run_dag_node(
        self,
        node: PlanNode,
        step_index: int,
        step_total: int,
        done: Dict[str, StepResult],
        base_work_dir: str,
        initial_prompt: str,
    ) -> StepResult:
        """Run ONE DAG node in an isolated workspace, then merge its writes back.

        Bounded by ``self._node_sem`` (whole-node concurrency). Opens a
        ``_ManagedWorkspace`` (vote's worktree-or-copy + #36 dirty-tree copy
        fallback), composes the node's context from its ANCESTOR closure (a join
        node sees the union of both branches; a parallel branch sees only its
        own), runs ``_execute_step`` against the workspace with a semaphore-gated
        verifier, then merges ONLY the files THIS node changed into the live tree.

        The write-back is the node's diff against its OWN base snapshot (taken at
        open), not a whole-tree mirror — so a concurrently-finishing sibling's
        disjoint writes are NOT clobbered (a plain ``apply_workspace`` mirror
        would delete files the sibling added, since they're absent from this
        node's HEAD-based workspace). Applies are serialized by ``_merge_lock``.
        M3 has NO overlap detection: two nodes writing the SAME path are
        last-writer-wins (a documented temporary limitation until M4).
        """
        async def _run() -> StepResult:
            # Context must be composed AFTER the node's deps are in ``done`` —
            # guaranteed because the scheduler only creates this task once all
            # deps completed. Snapshot it now so a concurrently-finishing,
            # unrelated node can't mutate what this node reads (it won't: a node
            # only reads its ancestors, which are already frozen in ``done``).
            graph = self.plan_graph
            assert graph is not None
            node_context = self._compose_context(node, graph, done, initial_prompt)
            ws = _ManagedWorkspace(
                base_work_dir, candidate_id=f"node-{node.id}", prefer_worktree=True
            )
            await ws.open()
            try:
                # Snapshot the workspace's files at open so we can later apply
                # ONLY what this node changed (not a whole-tree mirror that would
                # clobber a sibling's disjoint writes).
                base_snapshot = await asyncio.to_thread(_snapshot_tree, ws.path)
                verifier = (
                    _SemaphoreGatedVerifier(self.verifier, self._verify_sem)
                    if self.verifier is not None else None
                )
                result = await self._execute_step(
                    step_index=step_index,
                    step_total=step_total,
                    task=node.task,
                    project_context=node_context,
                    workflow_session_id=None,  # DAG nodes are sessionless
                    working_directory=str(ws.path),
                    verifier=verifier,
                )
                # Merge the node's writes serially so two concurrently-finishing
                # nodes never touch the live tree at once. merge_node detects an
                # overlap (a sibling that touched the same path since this node
                # opened) against base_snapshot and resolves it under
                # self.merge_policy: disjoint writes auto-apply; overlaps go to the
                # reconcile station (reconcile) or abort (disjoint/fail). The merge
                # lock IS the serializing semaphore (merge_node acquires it).
                assert self._merge_lock is not None
                try:
                    outcome = await merge_node(
                        node_id=node.id,
                        layer=step_index,
                        src_ws=ws.path,
                        live_tree=Path(base_work_dir),
                        base_snapshot=base_snapshot,
                        policy=self.merge_policy,
                        reconcile_station=self._build_reconciler(node, base_work_dir),
                        sem=self._merge_lock,
                    )
                except MergeConflict as mc:
                    # Abort policy (disjoint/fail) hit an overlap: record the
                    # conflicting-path outcome so it lands in meta.json, then
                    # re-raise to fail the run fast (the frontier scheduler cancels
                    # in-flight siblings — docs §4).
                    self.merge_outcomes.append(mc.outcome)
                    raise
                self.merge_outcomes.append(outcome)
                # A reconciled overlap rewrote files in the live tree: re-run THIS
                # node's verifier on the merged tree so a bad merge is caught by the
                # existing gate (docs §4) instead of shipped blind. Best-effort —
                # a verifier that flips red here surfaces via the run-level signals
                # the harvest re-reads; we don't fail the merge itself (the
                # reconcile policy is a safe superset, not a hard gate).
                if outcome.resolution == "reconciled" and verifier is not None:
                    try:
                        reverify = await verifier.verify(working_directory=str(base_work_dir))
                        result.verified = bool(getattr(reverify, "ok", result.verified))
                    except Exception as exc:
                        logger.warning(
                            "Post-merge re-verify failed for node '%s' (%s); "
                            "keeping the pre-merge verdict.", node.id, exc,
                        )
                return result
            finally:
                await ws.close()

        if self._node_sem is None:
            return await _run()
        async with self._node_sem:
            return await _run()

    def _build_reconciler(
        self, node: PlanNode, base_work_dir: str
    ) -> Optional[Reconciler]:
        """Build the per-overlap reconciler for this node, or None.

        Only the ``reconcile`` policy needs one; ``disjoint``/``fail`` abort on an
        overlap before any reconciler is consulted, so we skip the build. The
        ``reconcile_station_factory`` (wired by the harness from the critic chain)
        is called with the node + working directory so the resolver runs an
        independent cross-provider reviewer; when no factory is wired, reconcile
        policy degrades to a hard conflict (merge_node raises) rather than a silent
        clobber. Tests pass a trivial stub factory.
        """
        if self.merge_policy != "reconcile":
            return None
        if self.reconcile_station_factory is None:
            return None
        return self.reconcile_station_factory(
            node=node, working_directory=base_work_dir
        )

    def _rebuild_linear_context(
        self,
        initial_prompt: str,
        order: List[str],
        by_id: Dict[str, PlanNode],
        done: Dict[str, StepResult],
    ) -> str:
        """The cumulative project_context a linear run would have produced.

        Returned as the DAG run's final string and stored in the checkpoint so a
        resumed run reads the same shape the linear path writes."""
        context = f"Original Goal: {initial_prompt}\n\n=== Accumulated Implementation ===\n"
        for idx, nid in enumerate(order, 1):
            if nid in done:
                context += f"\n--- Step {idx} Summary ---\n{done[nid].step_summary}\n"
        return context

    @staticmethod
    def _step_result_to_checkpoint(result: "StepResult") -> dict:
        """The persisted view of a completed node (docs §5 M5).

        Only the fields a resume needs: the ``step_summary`` (so descendants'
        context composition is reproducible) + the confidence signals. The full
        ``final_output`` is intentionally NOT persisted — it can be huge and a
        resumed node never re-reads it (the summary is what threads forward)."""
        return {
            "step_summary": result.step_summary,
            "verified": result.verified,
            "approved": result.approved,
            "stalled": result.stalled,
            "iterations_used": result.iterations_used,
        }

    def _remove_checkpoint(self) -> None:
        path = self.checkpoint_path
        if not path:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Could not remove checkpoint %s: %s", path, exc)

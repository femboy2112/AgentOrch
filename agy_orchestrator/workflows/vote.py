"""VoteWorkflow — K-parallel candidate generation with verifier-gated selection.

Each candidate runs in its OWN per-candidate workspace (see
``execution/workspace.py``) so K workers writing in parallel can't trample
each other. After every candidate completes, the verifier grades each
workspace independently. The first passing candidate wins; its workspace
is applied back to the operator's actual work_dir; losers are discarded.

What this gets us:

* Real diversity gain. The chain-rotation rule (codex → agy → grok →
  codex → ...) ensures K=3 with our default chain produces ONE candidate
  per provider. The multi-agent diversity literature (arxiv 2602.03794)
  finds that 2 heterogeneous agents match 16 identical ones; the
  homogeneous tail saturates fast.
* Honest grading. Each candidate is verified in isolation, so the
  pass/fail signal is sound — no contamination from a peer's writes.
* Sound application. The winner's workspace is mirrored back to
  work_dir; losers vanish with their tempdirs.

What this is NOT:

* This is not adversarial (no critic loop). It's pure parallel-sample +
  pick. The "Debate or Vote" paper (arxiv 2508.17536, NeurIPS Spotlight)
  finds that voting alone captures most of multi-agent debate's gains
  for reasoning tasks while skipping the debate cost.
* This is not deterministic majority voting on identical outputs — code
  candidates are rarely identical text. The "majority" here is "majority
  of candidates that pass the verifier", with first-passer tiebreak.

The verifier is mandatory. Without it, K-vote degenerates to "pick the
first candidate" which is what direct mode already does, cheaper.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import QualityVerifier
from agy_orchestrator.execution.workspace import (
    COPY_IGNORE_PATTERNS,
    candidate_workspace,
)

logger = logging.getLogger(__name__)


class _ManagedWorkspace:
    """One candidate's workspace lifecycle, managed manually so the winner
    can be applied to base_dir BEFORE all workspaces are torn down."""

    def __init__(self, base_dir: str, candidate_id: str, prefer_worktree: bool = True):
        self._cm = candidate_workspace(
            Path(base_dir),
            candidate_id=candidate_id,
            prefer_worktree=prefer_worktree,
        )
        self.path: Optional[Path] = None
        self.backend: Optional[str] = None

    async def open(self) -> "_ManagedWorkspace":
        self.path, self.backend = await self._cm.__aenter__()
        return self

    async def close(self) -> None:
        try:
            await self._cm.__aexit__(None, None, None)
        except Exception as exc:
            logger.debug("workspace close failed (already torn down?): %s", exc)


class VoteWorkflow:
    """K candidates in isolated workspaces; verifier picks the winner.

    Ledger signals:
      * ``self.verified``      — winner passed the verifier.
      * ``self.n_candidates``  — K.
      * ``self.n_passed``      — how many candidates passed.
      * ``self.winner_index``  — index of the chosen candidate (-1 if none).
      * ``self.iterations_used``  — number of candidates that actually ran.
    """

    def __init__(
        self,
        generators: List[AgentInstance],
        verifier: QualityVerifier,
        working_directory: str = ".",
        prefer_worktree: bool = True,
    ):
        if not generators:
            raise ValueError("VoteWorkflow needs at least one generator")
        if verifier is None:
            raise ValueError(
                "VoteWorkflow requires a verifier (it's the gate for K candidates). "
                "Without it, vote degenerates to 'pick the first candidate', which "
                "direct mode does cheaper."
            )
        self.generators = list(generators)
        self.verifier = verifier
        self.working_directory = working_directory
        self.prefer_worktree = prefer_worktree
        # Ledger signals.
        self.verified = False
        self.n_candidates = len(self.generators)
        self.n_passed = 0
        self.winner_index = -1
        self.iterations_used = 0
        # Used by the run ledger / future analyzers.
        self.candidate_outcomes: List[Tuple[bool, Optional[str]]] = []

    async def execute(self, prompt: str) -> str:
        # Seed every generator with the same prompt; differences come from
        # model temperature + provider variation, not prompt drift.
        for gen in self.generators:
            gen.prompt = prompt

        workspaces = [
            _ManagedWorkspace(self.working_directory, candidate_id=f"vote-{i}",
                              prefer_worktree=self.prefer_worktree)
            for i in range(len(self.generators))
        ]
        # Open all K workspaces concurrently. Failures here propagate
        # cleanly — the finally block tears down whatever made it.
        try:
            await asyncio.gather(*[ws.open() for ws in workspaces])
        except Exception:
            await asyncio.gather(*[ws.close() for ws in workspaces], return_exceptions=True)
            raise

        try:
            results = await asyncio.gather(
                *[
                    self._run_one(i, gen, ws)
                    for i, (gen, ws) in enumerate(zip(self.generators, workspaces))
                ],
                return_exceptions=False,
            )

            # results[i] = (output_or_None, passed_bool, error_msg_or_None)
            self.iterations_used = sum(1 for out, _ok, _err in results if out is not None)
            self.candidate_outcomes = [(ok, err) for _out, ok, err in results]
            self.n_passed = sum(1 for _out, ok, _err in results if ok)

            # Pick winner: first passing candidate (lowest index = stable
            # tiebreak when multiple pass). The chain rotation gave each
            # index a different worker, so first-passer naturally favours
            # the operator's preferred lead provider.
            for i, (out, ok, _err) in enumerate(results):
                if ok and out is not None:
                    self.winner_index = i
                    self.verified = True
                    logger.info(
                        "Vote: %d/%d candidates passed; applying candidate %d "
                        "(%s) to %s",
                        self.n_passed, len(results), i,
                        type(self.generators[i]).__name__,
                        self.working_directory,
                    )
                    await self._apply_workspace(
                        workspaces[i].path, Path(self.working_directory),
                    )
                    return out

            # 0/K passed — return the best-effort first non-None output
            # for the operator to inspect, but DO NOT apply anything to
            # work_dir. The whole point of vote is verifier-gated.
            logger.warning(
                "Vote: 0/%d candidates passed verifier; nothing applied.",
                len(results),
            )
            for out, _ok, _err in results:
                if out is not None:
                    return out
            return ""
        finally:
            # Close in parallel; even an exception inside a close path
            # mustn't block the others.
            await asyncio.gather(
                *[ws.close() for ws in workspaces], return_exceptions=True,
            )

    async def _run_one(
        self, idx: int, gen: AgentInstance, ws: _ManagedWorkspace,
    ) -> Tuple[Optional[str], bool, Optional[str]]:
        """Run one candidate inside its workspace, then grade it.

        Returns (output, passed, error_msg). On generator exception,
        returns (None, False, str(exc)) so the gather doesn't blow up
        and the other K-1 candidates can still complete."""
        gen.cwd = str(ws.path)
        try:
            output = await gen.run_async()
        except Exception as exc:
            logger.warning(
                "Vote candidate %d (%s) raised: %s",
                idx, type(gen).__name__, exc,
            )
            return None, False, f"{type(exc).__name__}: {exc}"
        try:
            ok, err = await self.verifier.verify(working_directory=str(ws.path))
        except Exception as exc:
            logger.warning(
                "Vote candidate %d (%s) verifier raised: %s",
                idx, type(gen).__name__, exc,
            )
            return output, False, f"verifier-raised: {exc}"
        if not ok:
            logger.info(
                "Vote candidate %d (%s) failed verifier: %s",
                idx, type(gen).__name__, (err or "")[:160],
            )
        return output, bool(ok), err

    async def _apply_workspace(self, src: Path, dst: Path) -> None:
        """Mirror the winner's contents over the operator's actual work_dir.

        Walks src for regular files, copies any whose bytes differ from the
        destination's. Skips the same heavy/regenerable tree list the
        workspace clone uses (.git, __pycache__, runs, …) so an unintended
        meta-file can't sneak into work_dir.

        Runs in a thread because shutil.copy2 + large diffs can block the
        loop noticeably on multi-MB writes."""

        def _do_apply() -> None:
            for src_path in src.rglob("*"):
                if not src_path.is_file():
                    continue
                rel = src_path.relative_to(src)
                # Skip ignored top-level dirs (the clone wouldn't have
                # included these, but in worktree mode .git lives inside;
                # double-check at the apply boundary).
                top = rel.parts[0]
                if top in COPY_IGNORE_PATTERNS or top.startswith(".git"):
                    continue
                dst_path = dst / rel
                # Only copy when content actually differs — avoids touching
                # mtimes on identical files and gives a cleaner diff if the
                # operator inspects work_dir after.
                if dst_path.exists():
                    try:
                        if dst_path.read_bytes() == src_path.read_bytes():
                            continue
                    except OSError:
                        pass
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)

            # Remove files that no longer exist in the winner workspace.
            for dst_path in dst.rglob("*"):
                if not dst_path.is_file():
                    continue
                rel = dst_path.relative_to(dst)
                top = rel.parts[0]
                if top in COPY_IGNORE_PATTERNS or top.startswith(".git"):
                    continue
                if (src / rel).exists():
                    continue
                try:
                    dst_path.unlink()
                except OSError:
                    continue
                parent = dst_path.parent
                while parent != dst:
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                    parent = parent.parent

        await asyncio.to_thread(_do_apply)

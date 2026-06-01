"""VoteWorkflow — K-parallel candidate generation with verifier-gated selection.

Each candidate runs in its OWN per-candidate workspace (see
``execution/workspace.py``) so K workers writing in parallel can't trample
each other. After every candidate completes, the verifier grades each
workspace independently. Passing candidates are ranked by deterministic,
multi-criteria score; the top-ranked candidate wins and its workspace is
applied back to the operator's actual work_dir; losers are discarded.

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
  of candidates that pass the verifier", then deterministic tie-breaking
  by ranking metrics.

The verifier is mandatory. Without it, K-vote degenerates to "pick the
first candidate" which is what direct mode already does, cheaper.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import QualityVerifier
from agy_orchestrator.execution.workspace import (
    COPY_IGNORE_PATTERNS,
    candidate_workspace,
    diff_workspace_against_base,
)

logger = logging.getLogger(__name__)

VOTE_RANKING_SPEC = (
    "files_changed: asc",
    "has_test_changes: prefer true",
    "diff_size: asc",
    "index: asc",
)


@dataclass
class CandidateScore:
    index: int
    output: str
    workspace_path: Path
    files_changed: int
    diff_size: int
    has_test_changes: bool


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
        verifier_concurrency: int = 1,
        preflight: bool = True,
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
        # Host safety (issue #32 defect 2): K candidates generate concurrently
        # (remote-API-bound, cheap locally) but their verifiers are the local
        # spike — a full `make check` × K can OOM/freeze a small-RAM box. Cap
        # how many verifier runs execute at once; default 1 keeps verifiers
        # serial (like master mode) while generation stays parallel.
        self.verifier_concurrency = max(1, int(verifier_concurrency))
        # Fail-fast preflight (issue #32 defect 1): refuse before fanning out
        # K candidates if an isolated workspace can't even run the verifier.
        self.preflight = preflight
        self._verify_sem: Optional[asyncio.Semaphore] = None
        # Ledger signals.
        self.verified = False
        self.n_candidates = len(self.generators)
        self.n_passed = 0
        self.winner_index = -1
        self.iterations_used = 0
        self.ranking_metric: Optional[dict] = None
        # Used by the run ledger / future analyzers.
        self.candidate_outcomes: List[Tuple[bool, Optional[str]]] = []

    async def execute(self, prompt: str) -> str:
        # Bind the verifier concurrency gate to the running loop.
        self._verify_sem = asyncio.Semaphore(self.verifier_concurrency)

        # Fail fast if the isolated-workspace environment can't honestly run
        # the verifier (e.g. git-ignored .venv + editable install) — refusing
        # here is far cheaper than burning K generations + K verifiers for 0/K.
        if self.preflight:
            reason = await self._preflight_environment_check()
            if reason:
                self.verified = False
                self.n_passed = 0
                self.winner_index = -1
                raise RuntimeError(reason)

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

            passers: List[CandidateScore] = []
            for i, (out, ok, _err) in enumerate(results):
                if not ok or out is None:
                    continue
                score = await self._score_passer(i, out, workspaces[i])
                passers.append(score)

            if passers:
                winner = self._rank_passers(passers)[0]
                self.winner_index = winner.index
                self.verified = True
                self.ranking_metric = {
                    "files_changed": winner.files_changed,
                    "diff_size": winner.diff_size,
                    "has_test_changes": winner.has_test_changes,
                }
                logger.info(
                    "Vote: %d/%d passed; ranked candidate %d as winner "
                    "(files_changed=%d, diff_size=%d, has_test_changes=%s)",
                    self.n_passed,
                    len(results),
                    winner.index,
                    winner.files_changed,
                    winner.diff_size,
                    winner.has_test_changes,
                )
                await self._apply_workspace(
                    winner.workspace_path, Path(self.working_directory),
                )
                return winner.output

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

    async def _preflight_environment_check(self) -> Optional[str]:
        """Return a human-readable refusal reason if an isolated candidate
        workspace cannot honestly run the verifier, else None.

        Issue #32 defect 1: a candidate workspace is a `git worktree`
        (tracked files only) or a copy that deliberately skips `.venv`
        (workspace.py COPY_IGNORE_PATTERNS). A target whose verifier resolves
        its tools out of a git-ignored venv (`make check` -> `.venv/bin/pytest`,
        the common editable-install layout) therefore fails in EVERY candidate
        for environmental — not code — reasons, so vote silently returns 0/K
        after a long, expensive run.

        Detect this generally (no venv-specific hardcoding) by comparing the
        verifier outcome on a PRISTINE isolated workspace vs the real base tree:

          * pristine workspace passes      -> environment is fine; proceed.
          * pristine fails AND base fails  -> genuine red baseline; vote is
                                              legitimately being used to fix
                                              it, so proceed (don't refuse).
          * pristine fails BUT base passes -> isolation stripped something the
                                              gate needs; every candidate would
                                              fail environmentally. Refuse.

        Cost: one verifier run on a pristine workspace always; a second on the
        base tree only when the first fails. Both serial — far cheaper than the
        K generations + K verifiers a doomed run would burn.
        """
        try:
            async with candidate_workspace(
                Path(self.working_directory),
                candidate_id="vote-preflight",
                prefer_worktree=self.prefer_worktree,
            ) as (ws_path, _backend):
                pristine = await self.verifier.verify(working_directory=str(ws_path))
        except Exception as exc:
            # Couldn't even build/probe the workspace — let the real run
            # surface the error rather than guessing at a refusal here.
            logger.warning("Vote preflight probe failed to run (%s); skipping", exc)
            return None

        if pristine.ok:
            return None

        # Pristine workspace failed. Distinguish a broken isolated environment
        # from a genuinely red baseline by checking the real tree.
        base = await self.verifier.verify(working_directory=self.working_directory)
        if not base.ok:
            logger.info(
                "Vote preflight: verifier red on both the pristine workspace and "
                "the base tree — treating as a genuine red baseline; proceeding."
            )
            return None

        msg = (pristine.message or "").strip()
        return (
            "vote preflight refused: an isolated candidate workspace fails the "
            "verifier that the real tree passes, so the isolation strips "
            "something the gate needs. The common cause is a git-ignored .venv "
            "with an editable install (`pip install -e .`): a git worktree (or a "
            "copy, which skips .venv) checks out tracked files only, so "
            "`.venv/bin/pytest` is absent and EVERY candidate would fail for "
            "environmental — not code — reasons (0/K). Refusing now instead of "
            "burning a long run. Fix: use `--mode master` (it operates on the "
            "real tree with its real venv + editable install and verifies "
            "serially), or give the target a self-contained verifier (e.g. "
            "`python -m pytest`, no editable install). "
            f"Pristine-workspace verifier said: {msg[:300]}"
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
            # Throttle concurrent verifier runs — the local resource spike.
            assert self._verify_sem is not None  # set in execute()
            async with self._verify_sem:
                result = await self.verifier.verify(working_directory=str(ws.path))
        except Exception as exc:
            logger.warning(
                "Vote candidate %d (%s) verifier raised: %s",
                idx, type(gen).__name__, exc,
            )
            return output, False, f"verifier-raised: {exc}"
        if not result.ok:
            logger.info(
                "Vote candidate %d (%s) failed verifier: %s",
                idx, type(gen).__name__, (result.message or "")[:160],
            )
        return output, bool(result.ok), result.message

    async def _score_passer(
        self, idx: int, output: str, workspace: _ManagedWorkspace,
    ) -> CandidateScore:
        diff_text = await asyncio.to_thread(
            diff_workspace_against_base,
            Path(self.working_directory),
            workspace.path,
            backend=workspace.backend,
        )
        diff_size = len(diff_text)
        files_changed = 0
        has_test_changes = False
        for line in diff_text.splitlines():
            if not line.startswith("diff --git "):
                continue
            files_changed += 1
            parts = line.split()
            if len(parts) < 4:
                continue
            a_path = parts[2]
            b_path = parts[3]
            if a_path.startswith("a/"):
                a_path = a_path[2:]
            if b_path.startswith("b/"):
                b_path = b_path[2:]
            if "test" in a_path.lower() or "test" in b_path.lower():
                has_test_changes = True
        return CandidateScore(
            index=idx,
            output=output,
            workspace_path=workspace.path,
            files_changed=files_changed,
            diff_size=diff_size,
            has_test_changes=has_test_changes,
        )

    def _rank_passers(self, passers: List[CandidateScore]) -> List[CandidateScore]:
        return sorted(
            passers,
            key=lambda s: (
                s.files_changed,
                0 if s.has_test_changes else 1,
                s.diff_size,
                s.index,
            ),
        )

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

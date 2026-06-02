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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import QualityVerifier
from agy_orchestrator.execution.workspace import (
    _ManagedWorkspace,
    apply_workspace,
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


# ``_ManagedWorkspace`` (worktree-or-copy isolation, #36 dirty-tree fallback)
# now lives in execution/workspace.py so vote AND the master graph walker share
# ONE isolation implementation (docs §3.5). Re-exported here for back-compat with
# any caller that imports it from this module.

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
        baseline_ok: Optional[bool] = None,
        candidate_setup: Optional[str] = None,
        setup_timeout: Optional[float] = None,
        max_parallel: Optional[int] = None,
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
        # Reusable baseline verifier outcome on the UNCHANGED base tree, if the
        # caller already computed it (the harness runs a baseline gate seconds
        # before us). Lets the preflight skip a redundant full-suite re-run of
        # the base (#33). None = unknown; the preflight probes the base if it
        # needs to disambiguate a non-127 failure.
        self.baseline_ok = baseline_ok
        # Per-candidate environment bootstrap (issue #34): an opt-in shell
        # command run inside each candidate's workspace BEFORE its verifier.
        # On an editable-install repo this is what makes vote isolation sound —
        # e.g. `python -m venv .venv && .venv/bin/pip install -e .` gives each
        # candidate its OWN venv whose editable install resolves `import <pkg>`
        # to THAT candidate's source, not the base tree's (the false-green trap
        # #32 documented). None = no bootstrap (legacy behaviour: refuse on the
        # venv trap). Bounded by the same verifier semaphore (it's a heavy local
        # spike — a venv + pip install per candidate).
        self.candidate_setup = candidate_setup
        if setup_timeout is None:
            setup_timeout = float(os.environ.get("AGY_SETUP_TIMEOUT", "1200") or 0)
        self.setup_timeout = setup_timeout
        # Host-safety cap (#42 item 7): bound how many candidates run end-to-end
        # at once (generation + verification). None = all K concurrent (legacy).
        # Complements verifier_concurrency (which caps only the verifier step):
        # this caps the whole candidate so a --branches>1 swarm can't thrash a
        # small-RAM box. Resolves from the arg or AGY_MAX_PARALLEL_WORKERS.
        if max_parallel is None:
            _env_mp = os.environ.get("AGY_MAX_PARALLEL_WORKERS", "")
            max_parallel = int(_env_mp) if _env_mp.strip().isdigit() else None
        self.max_parallel = max(1, int(max_parallel)) if max_parallel else None
        self._verify_sem: Optional[asyncio.Semaphore] = None
        self._gen_sem: Optional[asyncio.Semaphore] = None
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
        # End-to-end candidate gate (#42 item 7): caps how many candidates are
        # in flight at once when --max-parallel-workers is set.
        self._gen_sem = (
            asyncio.Semaphore(self.max_parallel) if self.max_parallel else None
        )

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
                    self._run_one_gated(i, gen, ws)
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

    async def _run_candidate_setup(self, ws_path: str) -> Optional[str]:
        """Run the per-candidate environment bootstrap (#34) inside ws_path.

        Returns None on success (or when no setup is configured), else a
        human-readable error string so the candidate is failed cleanly. This is
        what makes vote isolation sound on editable-install repos: bootstrapping
        a per-candidate venv + editable install means the verifier imports THAT
        candidate's source, not the shared base tree's."""
        cmd = self.candidate_setup
        if not cmd:
            return None
        logger.info("Vote candidate setup: %s in %s", cmd, ws_path)
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=ws_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            comm = proc.communicate()
            if self.setup_timeout and self.setup_timeout > 0:
                _out, err = await asyncio.wait_for(comm, self.setup_timeout)
            else:
                _out, err = await comm
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), 5)
            except Exception:
                pass
            return f"candidate setup timed out after {self.setup_timeout:.0f}s: {cmd}"
        except Exception as exc:
            return f"candidate setup raised: {type(exc).__name__}: {exc}"
        if proc.returncode != 0:
            tail = err.decode(errors="replace")[-500:] if err else ""
            return f"candidate setup failed (rc={proc.returncode}): {cmd}\n{tail}".strip()
        return None

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

        Detect this generally (no venv-specific hardcoding) by checking a
        PRISTINE isolated workspace against the real base tree:

          * pristine workspace passes      -> environment is fine; proceed.
          * pristine fails with exit 127   -> the verifier binary is absent from
            ("command not found")             the workspace = environmental by
                                              definition; refuse immediately, no
                                              base run needed (#33).
          * pristine fails (other) AND
            base fails                     -> genuine red baseline; vote is
                                              legitimately being used to fix it,
                                              so proceed (don't refuse).
          * pristine fails (other) BUT
            base passes                    -> isolation stripped something the
                                              gate needs; every candidate would
                                              fail environmentally. Refuse.

        Cost: one verifier run on a pristine workspace always (instant in the
        common exit-127 case). The base half is taken from the caller-supplied
        baseline when available (#33 — the harness ran it seconds ago on the
        same unchanged tree); only a non-127 failure with no reusable baseline
        pays a base re-run. Far cheaper than the K generations + K verifiers a
        doomed run would burn.
        """
        try:
            async with candidate_workspace(
                Path(self.working_directory),
                candidate_id="vote-preflight",
                prefer_worktree=self.prefer_worktree,
            ) as (ws_path, _backend):
                # Mirror the real per-candidate flow: bootstrap the env (#34)
                # before verifying, so a configured setup that fixes the venv
                # trap lets vote PROCEED instead of refusing.
                setup_err = await self._run_candidate_setup(str(ws_path))
                if setup_err is not None:
                    return (
                        "vote preflight refused: the per-candidate setup hook "
                        f"(`{self.candidate_setup}`) failed in an isolated "
                        "workspace, so candidates can't be bootstrapped or graded "
                        "soundly. Fix the candidate-setup command, or use "
                        f"`--mode master`. Setup said: {setup_err[:300]}"
                    )
                pristine = await self.verifier.verify(working_directory=str(ws_path))
        except Exception as exc:
            # Couldn't even build/probe the workspace — let the real run
            # surface the error rather than guessing at a refusal here.
            logger.warning("Vote preflight probe failed to run (%s); skipping", exc)
            return None

        if pristine.ok:
            return None

        # Exit 127 = the verifier command itself couldn't start (missing binary,
        # e.g. the absent `.venv/bin/pytest`). That's environmental by
        # definition, so refuse without re-running the base tree (#33).
        if pristine.returncode != 127:
            # Other non-zero exit: distinguish a broken isolated environment
            # from a genuinely red baseline. Reuse the caller's baseline if we
            # have it (no redundant base run); otherwise probe the base once.
            base_ok = self.baseline_ok
            if base_ok is None:
                base = await self.verifier.verify(
                    working_directory=self.working_directory,
                )
                base_ok = base.ok
            if not base_ok:
                logger.info(
                    "Vote preflight: verifier red on both the pristine workspace "
                    "and the base tree — genuine red baseline; proceeding."
                )
                return None

        return self._preflight_refusal(pristine)

    @staticmethod
    def _preflight_refusal(pristine) -> str:
        """Build the refusal message, branched on the pristine verifier's exit
        code (#35).

        The two failure modes have different root causes and need different
        guidance — conflating them misdiagnoses the now-common post-#34 path:

          * exit 127/126 (command not found / not executable) -> the verifier
            BINARY never started: environmental by definition (the absent
            `.venv/bin/pytest` of a stripped editable-install workspace).
          * any other non-zero -> the verifier RAN but the suite FAILED: the
            isolated/bootstrapped env didn't reproduce the working tree's
            passing state (most often an undeclared/transitive dependency the
            `--candidate-setup` didn't install, or a suite that depends on
            untracked files / CWD / external state).

        Either way include the verifier's captured output tail so the failing
        tests are visible without a manual repro (#35 defect 2).
        """
        msg = (pristine.message or "").strip()
        rc = getattr(pristine, "returncode", None)
        output = VoteWorkflow._verifier_output_tail(pristine)
        diag = f"Pristine-workspace verifier said: {msg[:300]}"
        if output:
            diag += f"\n--- pristine verifier output (tail) ---\n{output}"

        # "command not found" can surface as 127, "not executable" as 126; some
        # shells/tools also spell it in the message — treat all as env-absent.
        msg_lower = msg.lower()
        env_absent = rc in (126, 127) or "command not found" in msg_lower

        if env_absent:
            return (
                "vote preflight refused: an isolated candidate workspace can't "
                "even START the verifier (exit "
                f"{rc if rc is not None else '127'} = command not found), but the "
                "real tree passes — so the isolation strips a tool the gate "
                "needs. The common cause is a git-ignored .venv with an editable "
                "install (`pip install -e .`): a git worktree (or a copy, which "
                "skips .venv) checks out tracked files only, so `.venv/bin/pytest` "
                "is absent and EVERY candidate would fail for environmental — not "
                "code — reasons (0/K). Refusing now instead of burning a long "
                "run. Fix: configure `--candidate-setup` to rebuild the env in "
                "each workspace (e.g. `python -m venv .venv && .venv/bin/pip "
                "install -e .`), use `--mode master` (it operates on the real "
                "tree with its real venv + editable install and verifies "
                "serially), or give the target a self-contained verifier (e.g. "
                "`python -m pytest`, no editable install). "
                f"{diag}"
            )

        return (
            "vote preflight refused: an isolated candidate workspace BUILT and "
            f"the verifier RAN, but it FAILED (exit {rc}), while the real tree "
            "passes — so the bootstrapped candidate environment did not reproduce "
            "the working tree's passing state. This is NOT the missing-.venv case "
            "(the verifier started and ran tests); the most likely cause is an "
            "undeclared or transitive dependency present in the real environment "
            "but not installed by your `--candidate-setup` (compare with a "
            "`pip freeze` diff between the real venv and a freshly bootstrapped "
            "one), or a suite that depends on untracked files / CWD / external "
            "state. Refusing now instead of returning 0/K after a long run. Fix: "
            "make `--candidate-setup` reproduce the real env (declare the missing "
            "dep in your project's extras, or install it in the setup command), "
            "or use `--mode master` (it verifies the real tree in place). "
            f"{diag}"
        )

    @staticmethod
    def _verifier_output_tail(result, limit: int = 1200) -> str:
        """Last `limit` chars of the verifier's captured stdout/stderr — the
        actual failing test names — so the refusal is diagnosable without a
        manual repro (#35). Returns '' when nothing was captured."""
        parts = []
        for label, attr in (("stdout", "stdout_tail"), ("stderr", "stderr_tail")):
            text = (getattr(result, attr, "") or "").strip()
            if text:
                parts.append(f"[{label}]\n{text}")
        if not parts:
            return ""
        combined = "\n".join(parts)
        return combined[-limit:]

    async def _run_one_gated(
        self, idx: int, gen: AgentInstance, ws: _ManagedWorkspace,
    ) -> Tuple[Optional[str], bool, Optional[str]]:
        """Run one candidate, bounded by the end-to-end parallelism gate (#42).

        With ``--max-parallel-workers`` set, at most N candidates execute
        concurrently; without it the gate is absent and all K run at once
        (byte-identical to the legacy path)."""
        if self._gen_sem is None:
            return await self._run_one(idx, gen, ws)
        async with self._gen_sem:
            return await self._run_one(idx, gen, ws)

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
        # Setup + verify are the per-candidate LOCAL spike (a venv+pip install
        # then a full suite) — gate both under one semaphore so K candidates
        # don't fire K of them at once on a small host.
        assert self._verify_sem is not None  # set in execute()
        async with self._verify_sem:
            setup_err = await self._run_candidate_setup(str(ws.path))
            if setup_err is not None:
                logger.info(
                    "Vote candidate %d (%s) setup failed: %s",
                    idx, type(gen).__name__, setup_err[:160],
                )
                return output, False, setup_err
            try:
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

        Thin wrapper over the shared :func:`apply_workspace` so vote and the
        master graph walker apply isolated writes through ONE implementation
        (docs §3.5)."""
        await apply_workspace(src, dst)

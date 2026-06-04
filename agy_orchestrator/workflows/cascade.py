"""CascadeWorkflow — cheap-first model cascade with as-needed escalation.

The research fan-out's recommended routing for our one-model-resident, RAM-tight
box: try the CHEAP/fast model first, and escalate to a stronger (slower) model
ONLY when the strong verifier (tests/lint/build) still fails. Most steps are
within the small model's competence, so we pay the big model's cost only on the
hard ones — far cheaper than always running the big model, and higher quality
than always running the small one. (Model cascades / FrugalGPT; ADaPT's
escalate-on-failure, arXiv:2311.05772.)

Each stage runs the proven TestFeedbackWorkflow (generate -> run tests -> feed
the error back -> repair). The verifier is the gate between stages: a stage
"succeeds" only if it makes the verifier pass. Returns as soon as a stage passes.

Long-run resilience (fuzz-longrun cascade_pat_pipeline, mirrors #50/#58):
  * An INFRA-class verifier verdict (host OOM / timeout / command-not-found) is
    NOT a code defect the strong model can fix, so it must NOT trigger
    cheap->strong escalation — we bail and flag it as infra (as adversarial.py
    already does for #50/#58).
  * A persistently-IDENTICAL failure across tiers carries no new signal, so we
    short-circuit instead of burning every remaining tier on it.
  * Stages must not poison each other: a failed cheap stage's partial writes are
    reset before the next (stronger) tier starts, so escalation is a clean retry.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.execution.verifier import QualityVerifier, VerifierResult
from agy_orchestrator.workflows.test_feedback import TestFeedbackWorkflow

logger = logging.getLogger(__name__)

# Top-level trees that are heavy/regenerable; never snapshot or reset them when
# isolating stages (mirrors execution/workspace.COPY_IGNORE_PATTERNS so the
# inter-stage reset stays cheap and doesn't churn a multi-GB .venv).
_RESET_IGNORE = (
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "runs", "node_modules", "dist", "build",
)


class CascadeWorkflow:
    def __init__(
        self,
        stages: List[AgentInstance],
        verifier: QualityVerifier,
        max_iterations_per_stage: int = 2,
        working_directory: str = ".",
    ):
        if not stages:
            raise ValueError("CascadeWorkflow needs at least one generator stage")
        if verifier is None:
            raise ValueError("CascadeWorkflow requires a QualityVerifier (the escalation gate)")
        self.stages = stages
        self.verifier = verifier
        self.max_iterations_per_stage = max_iterations_per_stage
        self.working_directory = working_directory
        # Signals for the ledger.
        self.verified = False
        self.stage_used = -1          # index of the stage that passed (-1 = none)
        self.iterations_used = 0      # total repair rounds across stages
        self.stalled = False
        # Long-run resilience signals (read defensively by the ledger via getattr,
        # so adding them is back-compatible).
        self.verifier_infra_failed = False   # an infra-class verdict aborted the cascade
        self.infra_reason: Optional[str] = None
        self.short_circuited = False         # bailed on a repeated-identical failure
        self.no_work = False                 # config ran zero generators (misconfig)

    # --------------------------------------------------------------------- #
    # Inter-stage isolation helpers
    #
    # Cascade stages share one on-disk working_directory. Without a reset a failed
    # cheap stage's broken partial edit becomes the baseline the strong stage
    # starts from. We capture the tree once at entry and restore it before each
    # escalation so the next tier retries from a CLEAN base (graph mode gets this
    # for free via isolated /tmp workspaces; the flat cascade did not). Opt-out:
    # AGY_CASCADE_ISOLATE=0 keeps the legacy shared-dir behaviour.
    # --------------------------------------------------------------------- #
    def _isolation_enabled(self) -> bool:
        return os.environ.get("AGY_CASCADE_ISOLATE", "1") not in ("0", "", "false")

    def _snapshot_tree(self) -> Optional[Dict[str, bytes]]:
        root = Path(self.working_directory)
        if not root.is_dir():
            return None
        snap: Dict[str, bytes] = {}
        try:
            for path in root.rglob("*"):
                if not path.is_file() or path.is_symlink():
                    continue
                rel = path.relative_to(root)
                if rel.parts and (rel.parts[0] in _RESET_IGNORE
                                  or rel.parts[0].startswith(".git")):
                    continue
                try:
                    snap[str(rel)] = path.read_bytes()
                except OSError:
                    continue
        except OSError:
            return None
        return snap

    def _restore_tree(self, snap: Optional[Dict[str, bytes]]) -> None:
        """Reset the tracked files to their snapshot bytes and drop files the
        failed stage newly created. Best-effort: a reset failure must never crash
        the cascade (we just fall back to the legacy shared-dir behaviour)."""
        if snap is None:
            return
        root = Path(self.working_directory)
        try:
            current = self._snapshot_tree() or {}
            # Remove files the failed stage added (not present in the snapshot).
            for rel in current:
                if rel not in snap:
                    try:
                        (root / rel).unlink()
                    except OSError:
                        pass
            # Restore changed/removed files to their original bytes.
            for rel, data in snap.items():
                target = root / rel
                if current.get(rel) == data:
                    continue
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                except OSError:
                    pass
        except OSError:
            pass

    # --------------------------------------------------------------------- #
    # Verifier-verdict inspection
    #
    # TestFeedbackWorkflow gates on result.ok only and does not surface the final
    # VerifierResult, so after a stage fails we take ONE probing verdict to read
    # the structured infra flags (#50 timeout/.resource_exceeded, #58
    # .infra_suspected) and the error_hash used for same-error short-circuiting.
    # On the fake verifier this repeats the last result; on a real one it's a
    # single extra (cheap, already-paid-for) read that only happens off the
    # failure path.
    # --------------------------------------------------------------------- #
    async def _probe_verdict(self) -> Optional[VerifierResult]:
        try:
            return await self.verifier.verify(working_directory=self.working_directory)
        except Exception:  # noqa: BLE001 — a probe must never crash the cascade
            return None

    @staticmethod
    def _is_infra(result: Optional[VerifierResult]) -> Optional[str]:
        if result is None:
            return None
        if getattr(result, "timeout", False):
            return "verifier_timeout"
        if getattr(result, "resource_exceeded", False):
            return "verifier_resource_exceeded"
        if getattr(result, "infra_suspected", False):
            return "verifier_infra_suspected"
        return None

    @staticmethod
    def _signature(result: Optional[VerifierResult]) -> Optional[str]:
        """A stable identity for a failure so consecutive-identical failures can
        be detected. Prefer the verifier's error_hash; fall back to the message."""
        if result is None:
            return None
        h = getattr(result, "error_hash", None)
        if h:
            return str(h)
        msg = getattr(result, "message", "") or ""
        if not msg:
            return None
        return hashlib.sha256(msg.encode("utf-8", "replace")).hexdigest()

    async def execute(self, initial_prompt: str) -> str:
        last_output = ""

        # A non-positive iteration budget runs ZERO generators per stage — the
        # cascade is a silent no-op that would otherwise present identically to a
        # genuine all-stage failure. Surface it as a distinct misconfiguration
        # signal so the ledger/operator can branch on it rather than reading it as
        # a normal converged-but-failed stall.
        if self.max_iterations_per_stage <= 0:
            logger.error(
                "CascadeWorkflow max_iterations_per_stage=%d runs zero repair rounds; "
                "no generator will execute. Refusing to present this as a normal stall.",
                self.max_iterations_per_stage,
            )
            self.no_work = True
            self.stalled = False
            self.stage_used = -1
            return last_output

        baseline = self._snapshot_tree() if self._isolation_enabled() else None
        prev_signature: Optional[str] = None

        for i, gen in enumerate(self.stages):
            logger.info("Cascade stage %d/%d (%s)", i + 1, len(self.stages), type(gen).__name__)
            wf = TestFeedbackWorkflow(
                gen, self.verifier,
                max_iterations=self.max_iterations_per_stage,
                working_directory=self.working_directory,
            )
            last_output = await wf.execute(initial_prompt)
            self.iterations_used += wf.iterations_used
            if wf.verified:
                logger.info("Cascade passed at stage %d after %d total repair rounds.",
                            i + 1, self.iterations_used)
                self.verified = True
                self.stage_used = i
                return last_output

            # The stage failed the gate. Inspect the structured verdict BEFORE
            # deciding to spend the next (stronger, costlier) tier.
            verdict = await self._probe_verdict()

            # #50/#58: an infra-class verdict (host OOM / timeout / command-not-
            # found) is not a code defect — regenerating on a stronger model cannot
            # fix the host. Bail and flag it instead of escalating.
            infra_reason = self._is_infra(verdict)
            if infra_reason is not None:
                logger.warning(
                    "Cascade stage %d hit an infra-class verifier verdict (%s); "
                    "NOT escalating to the strong stage — a host flap is not a defect "
                    "the next model can repair.", i + 1, infra_reason)
                self.verifier_infra_failed = True
                self.infra_reason = infra_reason
                self.stalled = True
                return last_output

            # MED: same-error early-exit. If this tier failed with the IDENTICAL
            # signature as the previous tier (no new signal — host broken or a flaky
            # test no model can repair), escalating again is pure waste; short-circuit.
            signature = self._signature(verdict)
            if (signature is not None and signature == prev_signature):
                logger.warning(
                    "Cascade stage %d reproduced the identical failure (%s) with no new "
                    "signal; short-circuiting the remaining tiers.", i + 1, signature[:12])
                self.short_circuited = True
                self.stalled = True
                return last_output
            prev_signature = signature

            logger.info("Stage %d did not pass the verifier; escalating.", i + 1)
            # Reset the shared tree so the next tier starts from a clean base, not
            # on top of this failed stage's broken partial writes.
            if self._isolation_enabled():
                self._restore_tree(baseline)

        logger.warning("Cascade exhausted all %d stages without passing.", len(self.stages))
        self.stalled = True
        return last_output

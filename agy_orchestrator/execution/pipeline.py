import asyncio
import logging
import os
from typing import List, Optional

from agy_orchestrator.core.agent import AgentInstance

logger = logging.getLogger(__name__)


def _swarm_branch_timeout() -> float:
    """Per-branch wall-clock deadline (seconds) for ParallelSwarm.execute.

    A wedged worker that never returns is NOT an exception, so
    ``asyncio.gather(return_exceptions=True)`` cannot reap it — one hang would
    block the whole swarm forever on a long ToT / GenerateAndRank build. We bound
    each branch with ``asyncio.wait_for`` so a single hang fails its own slot and
    the survivors still come through.

    Tunable via ``AGY_SWARM_BRANCH_TIMEOUT`` (seconds). Per this repo's env-knob
    convention, ``0``/unset preserves the pre-fix unbounded behaviour so existing
    callers (ToT / GenerateAndRank) — whose branches already self-bound via the
    worker watchdog ``AGY_TIMEOUT`` — are byte-identical unaffected. Set a positive
    value on a long / mission-critical run to backstop a branch whose own watchdog
    has ALSO failed (a pure hang that ``gather`` cannot reap). A malformed value
    disables the bound rather than bricking the swarm.
    """
    raw = os.environ.get("AGY_SWARM_BRANCH_TIMEOUT", "0")
    try:
        return float(raw or 0)
    except (TypeError, ValueError):
        logger.warning(
            "AGY_SWARM_BRANCH_TIMEOUT=%r is not a number; leaving swarm unbounded",
            raw,
        )
        return 0.0


async def _run_branch(coro, timeout: float):
    """Await one swarm branch, bounding it at ``timeout`` seconds (0 => unbounded).

    A timed-out branch resolves to an ``asyncio.TimeoutError`` (reaped by the
    gather's ``return_exceptions=True`` like any other branch failure) instead of
    wedging the swarm; the underlying task is cancelled so it stops consuming a
    worker slot.
    """
    if not timeout or timeout <= 0:
        return await coro
    return await asyncio.wait_for(coro, timeout=timeout)

class LinearPipeline:
    """
    Executes a sequence of AgentInstances, piping the stdout of one
    into the stdin (as additional prompt context) of the next.
    """
    def __init__(self, instances: List[AgentInstance]):
        self.instances = instances

    async def execute(self, initial_input: Optional[str] = None) -> str:
        current_input = initial_input
        for idx, instance in enumerate(self.instances):
            current_input = await instance.run_async(piped_input=current_input)
        return current_input

class ParallelSwarm:
    """
    Executes multiple AgentInstances concurrently. Useful for exploring
    multiple paths (like in Tree of Thought) or handling chunked tasks.
    """
    def __init__(self, instances: List[AgentInstance]):
        self.instances = instances

    async def execute(self, common_input: Optional[str] = None) -> List[str]:
        # Per-branch deadline: a wedged worker that never returns is not an
        # exception, so gather() alone cannot reap it — wrap each branch in
        # wait_for so a single hang fails its own slot, not the whole swarm.
        branch_timeout = _swarm_branch_timeout()
        tasks = [
            asyncio.create_task(
                _run_branch(instance.run_async(piped_input=common_input), branch_timeout)
            )
            for instance in self.instances
        ]
        # Crash-tolerant: one branch failing (a usage wall, a hung worker, an OOM)
        # must NOT abort the whole swarm — ToT/Master rely on the survivors. Gather
        # exceptions and return only the successful branch outputs.
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs: List[str] = []
        for inst, res in zip(self.instances, results):
            if isinstance(res, Exception):
                logger.warning("Swarm branch failed (%s): %s",
                               type(inst).__name__, res)
            else:
                outputs.append(res)
        if not outputs and results:
            # All branches failed — re-raise the first so the caller sees a cause.
            raise results[0]
        return outputs

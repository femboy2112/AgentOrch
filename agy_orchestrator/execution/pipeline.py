import asyncio
import logging
from typing import List, Optional

from agy_orchestrator.core.agent import AgentInstance

logger = logging.getLogger(__name__)

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
        tasks = [
            asyncio.create_task(instance.run_async(piped_input=common_input))
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

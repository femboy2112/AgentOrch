import asyncio
import logging
from typing import Dict, List, Optional

from agy_orchestrator.core.agent import AgentInstance

logger = logging.getLogger(__name__)

class TaskDAG:
    """
    Executes a Directed Acyclic Graph of AgentInstances.
    Nodes run concurrently, but wait for their specified dependencies to complete.
    """
    def __init__(self):
        self.nodes: Dict[str, AgentInstance] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.tasks: Dict[str, asyncio.Task] = {}

    def add_node(self, name: str, instance: AgentInstance, deps: Optional[List[str]] = None):
        if name in self.nodes:
            raise ValueError(f"Node {name} already exists.")
        self.nodes[name] = instance
        self.dependencies[name] = deps or []

    async def _run_node(self, name: str) -> str:
        deps = self.dependencies.get(name, [])
        dep_results = []

        # Wait for dependencies
        for dep in deps:
            if dep not in self.tasks:
                raise ValueError(f"Dependency {dep} for node {name} is missing.")
            res = await self.tasks[dep]
            dep_results.append(f"--- Input from dependency: {dep} ---\n{res}")

        combined_input = "\n\n".join(dep_results) if dep_results else None

        logger.info(f"DAG Node '{name}' started.")
        result = await self.nodes[name].run_async(piped_input=combined_input)
        logger.info(f"DAG Node '{name}' completed.")
        return result

    async def execute(self) -> Dict[str, str]:
        """
        Executes the entire DAG and returns a dictionary of node outputs.
        """
        self.tasks.clear()

        # Create all tasks
        for name in self.nodes:
            self.tasks[name] = asyncio.create_task(self._run_node(name))

        # Await all tasks
        results = {}
        for name, task in self.tasks.items():
            results[name] = await task

        return results

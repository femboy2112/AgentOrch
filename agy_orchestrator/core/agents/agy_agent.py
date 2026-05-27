from typing import Optional, Dict, List
import asyncio
from agy_orchestrator.core.agent import AgentInstance

class AgyAgent(AgentInstance):
    def __init__(
        self,
        prompt: str,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        input_files: Optional[List[str]] = None,
        output_files: Optional[List[str]] = None,
        additional_flags: Optional[Dict[str, str]] = None
    ):
        super().__init__(prompt, model, additional_flags)
        self.effort = effort
        self.input_files = input_files or []
        self.output_files = output_files or []

    @classmethod
    async def get_available_models(cls) -> List[str]:
        return ["standard", "pro", "flash"]

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        return 100.0

    def build_command(self, piped_input: Optional[str] = None) -> List[str]:
        injected_prompt = f"System constraints: \n"
        if self.model:
            injected_prompt += f"- Use model architecture equivalent to: {self.model}\n"
        if self.effort:
            injected_prompt += f"- Effort level: {self.effort}\n"
        if self.input_files:
            injected_prompt += f"- Read these files: {', '.join(self.input_files)}\n"
        if self.output_files:
            injected_prompt += f"- Ensure these files are created: {', '.join(self.output_files)}\n"
        
        injected_prompt += f"- EXCELLENCE: Produce work that meets a high, domain-appropriate quality bar for the task — correct, robust, and well-crafted.\n"
        injected_prompt += f"- PERFORMANCE: Where performance matters, write efficient code and avoid needless overhead.\n"
        injected_prompt += f"- CORRECTNESS: Double-check that identifiers, signatures, and interfaces match exactly across files and components.\n"
        injected_prompt += f"- NO SUDO: Do NOT use `sudo` under any circumstances.\n"
        
        injected_prompt += f"\n{self.prompt}"

        if piped_input:
            injected_prompt += f"\n\n[Piped Context from previous step]:\n{piped_input}"

        cmd = ["agy", "--print", injected_prompt, "--dangerously-skip-permissions"]
        for k, v in self.additional_flags.items():
            cmd.extend([f"--{k}", str(v)])
        return cmd

from typing import Optional, List
from agy_orchestrator.core.agent import AgentInstance

class CodexAgent(AgentInstance):
    @classmethod
    async def get_available_models(cls) -> List[str]:
        return ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-5.3-codex-spark", "gpt-5.2"]

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        # Codex exposes no machine-readable remaining-usage value, so report
        # "full" and let real quota exhaustion be handled by the fallback layer
        # (a usage wall surfaces as a non-zero exit -> fail over). Spawning
        # `codex --usage` only to discard its output was pure overhead.
        return 100.0

    def filter_stderr(self, stderr: str) -> str:
        lines = stderr.splitlines()
        filtered = [l for l in lines if "network" not in l.lower() and "timeout" not in l.lower()]
        return "\n".join(filtered)

    def build_command(self, piped_input: Optional[str] = None) -> List[str]:
        full_prompt = self.prompt
        if piped_input:
            full_prompt += f"\n\n[Context]:\n{piped_input}"
            
        cmd = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
        ]
        
        if self.model and self.model != "standard":
            cmd.extend(["--model", self.model])
        elif self.model == "standard":
            cmd.extend(["--model", "gpt-5.3-codex"])
            
        if hasattr(self, "effort") and self.effort:
            effort = "xhigh" if self.effort == "max" else self.effort
            cmd.extend(["-c", f"model_reasoning_effort=\"{effort}\""])

        # Arbitrary `-c key=value` config overrides (e.g. "tools.web_search=true").
        for entry in getattr(self, "config_overrides", None) or []:
            cmd.extend(["-c", entry])

        for k, v in self.additional_flags.items():
            cmd.extend([f"--{k}", str(v)])
            
        cmd.append(full_prompt)
        return cmd

class UserProfile:
    """
    Stores user subscription details and resolves baseline effort/model configurations.
    """
    def __init__(self, claude_plan: str = "free", codex_plan: str = "free", agy_plan: str = "free"):
        self.claude_plan = claude_plan
        self.codex_plan = codex_plan
        self.agy_plan = agy_plan
        
    def get_baseline_effort(self, agent_name: str) -> str:
        """Returns the default starting effort level based on the plan tier."""
        plan = getattr(self, f"{agent_name}_plan", "free").lower()
        if "max" in plan or "$100" in plan or "20x" in plan:
            return "max"
        elif "pro" in plan or "$50" in plan:
            return "high"
        elif "plus" in plan or "$20" in plan:
            return "medium"
        return "low"

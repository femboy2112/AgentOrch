import logging
from typing import Any, Dict, Type

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.core.profile import UserProfile

logger = logging.getLogger(__name__)

class UsageAwareAllocator:
    """
    Manages and scales execution effort and model configuration
    based on success/failure and remaining usage capacity.
    """

    EFFORT_LEVELS = ["low", "medium", "high", "max"]

    def __init__(self, profile: UserProfile, agent_class: Type[AgentInstance], agent_name: str, initial_model: str):
        self.profile = profile
        self.agent_class = agent_class
        self.agent_name = agent_name
        self.base_model = initial_model

        baseline_effort = self.profile.get_baseline_effort(agent_name)
        self.current_effort_index = self.EFFORT_LEVELS.index(baseline_effort)

    async def get_current_config(self) -> Dict[str, Any]:
        """Returns the active configuration dictionary, compensating for usage limits."""
        available_models = await self.agent_class.get_available_models()

        current_model = self.base_model
        current_effort = self.EFFORT_LEVELS[self.current_effort_index]

        usage = await self.agent_class.get_model_usage(current_model)

        if usage < 10.0:
            logger.warning(f"Usage for {current_model} is critically low ({usage}%). Triggering compensator...")

            # Fallback 1: Lower effort
            if self.current_effort_index > 0:
                self.current_effort_index -= 1
                current_effort = self.EFFORT_LEVELS[self.current_effort_index]
                logger.info(f"Fallback 1: Lowered effort to {current_effort}")
            else:
                # Fallback 2: Lower model
                if current_model in available_models:
                    idx = available_models.index(current_model)
                    if idx > 0:
                        current_model = available_models[idx - 1]
                        logger.info(f"Fallback 2: Lowered model to {current_model}")
                    else:
                        logger.warning("Fallback 3: Would cross-chain to different provider, but logic resides in orchestration router.")
                else:
                    logger.warning("Fallback 3: Model not in list. Cannot downgrade model natively.")

        return {
            "effort": current_effort,
            "model": current_model
        }

    def escalate(self) -> bool:
        """Escalates to a higher effort tier."""
        if self.current_effort_index < len(self.EFFORT_LEVELS) - 1:
            self.current_effort_index += 1
            logger.info(f"Escalated effort to: {self.EFFORT_LEVELS[self.current_effort_index]}")
            return True
        logger.warning("Already at max effort, cannot escalate further.")
        return False

    def reset(self):
        """Resets to the baseline effort defined by the profile."""
        baseline = self.profile.get_baseline_effort(self.agent_name)
        self.current_effort_index = self.EFFORT_LEVELS.index(baseline)
        logger.info("Effort allocator reset.")

import sys
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    Handles interaction events, routing questions either to an automated
    LLM resolver or to the human user via standard input.
    """
    def __init__(self, auto_mode: bool = False, auto_resolver_instance=None):
        self.auto_mode = auto_mode
        self.auto_resolver = auto_resolver_instance
        
    async def resolve_question(self, question_text: str) -> str:
        """
        Resolves a question blocking the execution pipeline.
        """
        if self.auto_mode and self.auto_resolver:
            logger.info("Auto-Decision mode enabled. Resolving question automatically.")
            self.auto_resolver.prompt = (
                f"Please provide a concise and correct answer to the following system/user question "
                f"so execution can proceed automatically:\n\n{question_text}"
            )
            answer = await self.auto_resolver.run_async()
            logger.info(f"Auto-resolved answer: {answer}")
            return answer
            
        # Human-in-the-loop fallback
        logger.info("Human-in-the-loop mode: Requesting user input.")
        print(f"\n[SYSTEM QUESTION] {question_text}")
        print("Please enter your response: ", end="", flush=True)
        
        # Async-safe stdin read
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, sys.stdin.readline)
        return answer.strip()

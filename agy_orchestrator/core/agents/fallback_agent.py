"""Cross-provider fallback agent.

Wraps an ordered chain of agent classes (e.g. codex -> agy -> claude). Each call
tries the primary provider; if it fails after its own internal retries (which is
what "runs out of usage"/quota looks like — the CLI exits non-zero and the base
run_async raises RuntimeError), the next provider in the chain is tried. This makes
a long autonomous build resilient to one provider exhausting its usage mid-run.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

from agy_orchestrator.core.agent import AgentInstance, USAGE_MARKERS
from agy_orchestrator.core.agents.claude_agent import ClaudeAgent
from agy_orchestrator.core.agents.grok_agent import GrokAgent

logger = logging.getLogger(__name__)

# Agent classes that support warm-session resume across calls.
_SESSION_CAPABLE = (ClaudeAgent, GrokAgent)


def make_fallback_agent(
    chain: List[Type[AgentInstance]],
    cycles: int = 3,
    configs: Optional[Dict[Type[AgentInstance], Dict[str, object]]] = None,
) -> Type[AgentInstance]:
    """Return an AgentInstance subclass that tries ``chain`` in order per call.

    ``cycles`` is how many times the whole chain is repeated before giving up.
    With ``chain=[codex, agy, claude]`` and ``cycles>=2`` this realizes the
    operator's requested behavior: codex primary, then on usage exhaustion fall
    back agy -> claude -> codex -> repeat (a provider that walled may have had
    its usage recover by the time the cycle comes back around to it).

    ``configs`` optionally maps an agent class to per-provider overrides
    (``{"model": ..., "effort": ...}``). This is essential when chaining
    providers that do not share a model namespace — e.g. an ``agy(model="pro")``
    reviewer falling back to ``codex``, which must NOT receive "pro" as its
    model. Keys absent from ``configs`` fall back to ``self.model`` /
    ``self.effort`` (the prior behavior, so existing callers are unaffected).
    """

    if not chain:
        raise ValueError("fallback chain must be non-empty")
    if cycles < 1:
        raise ValueError("cycles must be >= 1")

    class FallbackAgent(AgentInstance):
        _chain: List[Type[AgentInstance]] = list(chain)
        _cycles: int = cycles
        _configs: Dict[Type[AgentInstance], Dict[str, object]] = dict(configs or {})

        @classmethod
        async def get_available_models(cls) -> List[str]:
            return await cls._chain[0].get_available_models()

        @classmethod
        async def get_model_usage(cls, model: str) -> float:
            return await cls._chain[0].get_model_usage(model)

        def build_command(self, piped_input: Optional[str] = None) -> List[str]:
            # Satisfies the ABC; never used because run_async is overridden.
            return []

        def _make_sub(self, agent_cls: Type[AgentInstance]) -> AgentInstance:
            cfg = self._configs.get(agent_cls, {})
            model = cfg.get("model", self.model)
            effort = cfg.get("effort", getattr(self, "effort", None))
            kwargs: dict[str, object] = {"prompt": self.prompt, "model": model}
            if effort:
                kwargs["effort"] = effort
            # Pass through any other per-provider config (e.g. codex
            # config_overrides) untouched, so each provider gets exactly the
            # extras it understands and no more.
            for key, val in cfg.items():
                if key not in ("model", "effort"):
                    kwargs[key] = val
            # Reuse a warm session ONLY for the same provider class that created
            # it — a claude session id is meaningless to grok (and vice-versa), so
            # feeding one to the other forces a spurious "no such session" failure.
            if issubclass(agent_cls, _SESSION_CAPABLE) and agent_cls is getattr(self, "_session_owner", None):
                sid = getattr(self, "session_id", None)
                if sid:
                    kwargs["session_id"] = sid
                if agent_cls is ClaudeAgent and getattr(self, "fork_session", False):
                    kwargs["fork_session"] = True
            return agent_cls(**kwargs)  # type: ignore[arg-type]

        async def run_async(self, piped_input: Optional[str] = None) -> str:
            last_error: Optional[Exception] = None
            # Repeat the chain ``_cycles`` times: codex -> agy -> claude -> (repeat).
            sequence = list(self._chain) * self._cycles
            total = len(sequence)
            for index, agent_cls in enumerate(sequence):
                label = agent_cls.__name__
                try:
                    sub = self._make_sub(agent_cls)
                except Exception as exc:  # construction failure -> skip provider
                    logger.warning("[Fallback] could not build %s: %s", label, exc)
                    last_error = exc
                    continue

                logger.info("[Fallback] provider %d/%d: %s", index + 1, total, label)
                try:
                    result = await sub.run_async(piped_input)
                except Exception as exc:
                    stderr = (getattr(sub, "stderr", "") or "").lower()
                    looked_like_usage = any(marker in stderr for marker in USAGE_MARKERS)
                    logger.warning(
                        "[Fallback] %s failed%s: %s",
                        label,
                        " (usage/quota wall)" if looked_like_usage else "",
                        exc,
                    )
                    last_error = exc
                    continue

                # Success: propagate session id (for warm-cache reuse) and output.
                # Tag the owner class so the session is only ever resumed by the
                # same provider that produced it (see _make_sub).
                sid = getattr(sub, "session_id", None)
                if sid:
                    self.session_id = sid
                    self._session_owner = agent_cls
                self.stdout = sub.stdout
                self.stderr = sub.stderr
                self.returncode = sub.returncode
                if index > 0:
                    logger.info("[Fallback] recovered via %s", label)
                return result

            raise RuntimeError(
                f"All {total} fallback attempts exhausted "
                f"({len(self._chain)} providers x {self._cycles} cycles); last error: {last_error}"
            )

    FallbackAgent.__name__ = "FallbackAgent[" + "->".join(c.__name__ for c in chain) + "]"
    return FallbackAgent

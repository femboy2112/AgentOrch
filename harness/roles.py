"""Role definitions: how the harness maps abstract roles to concrete agents.

Per the operating agreement:
  - codex writes code (the generator / primary worker).
  - agy, at high effort on its best model, is the reviewer/critic — but agy
    usage gets exhausted, so any agy role is ALWAYS fallback-wrapped to codex.
  - both roles are fallback-wrapped so a single provider's usage wall never
    stalls a working session.

Everything here is overridable per dispatch from the CLI; these are defaults.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.core.agents.agy_agent import AgyAgent
from agy_orchestrator.core.agents.claude_agent import ClaudeAgent
from agy_orchestrator.core.agents.codex_agent import CodexAgent
from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent
from agy_orchestrator.core.agents.grok_agent import GrokAgent

AGENT_CLASSES: Dict[str, Type[AgentInstance]] = {
    "codex": CodexAgent,
    "agy": AgyAgent,
    "claude": ClaudeAgent,
    "grok": GrokAgent,
}

# Per-provider model/effort. "best model" per provider, high effort.
AGENT_DEFAULTS: Dict[str, Dict[str, object]] = {
    "codex": {"model": "standard", "effort": "high"},   # -> gpt-5.3-codex
    "agy": {"model": "pro", "effort": "high"},           # agy's best tier
    "claude": {"model": "opus", "effort": "high"},
    # xAI Grok agentic CLI. Only `grok-build` exists today and it REJECTS the
    # reasoning-effort param, so effort is "n/a" and GrokAgent never sends it.
    "grok": {"model": "grok-build", "effort": "n/a"},
}

# Ordered fallback chains by role. Generator leads with codex (the code writer);
# critic leads with agy (premium review) then drops to codex when agy is walled.
GENERATOR_CHAIN: List[str] = ["codex", "agy"]
CRITIC_CHAIN: List[str] = ["agy", "codex"]


def _cfg_for_token(
    token: str, codex_config: Optional[List[str]] = None
) -> Tuple[str, Dict[str, object]]:
    """Resolve a chain token to ``(agent_name, config)``.

    A token is a plain agent name (``codex``/``agy``/``claude``/``grok``). Codex
    config overrides (e.g. ``tools.web_search=true``) are threaded onto codex."""
    name = token
    cfg = dict(AGENT_DEFAULTS[name])
    if name == "codex" and codex_config:
        cfg["config_overrides"] = list(codex_config)
    return name, cfg


def _configs_for(
    chain: List[str], codex_config: Optional[List[str]] = None
) -> Dict[Type[AgentInstance], Dict[str, object]]:
    out: Dict[Type[AgentInstance], Dict[str, object]] = {}
    for token in chain:
        name, cfg = _cfg_for_token(token, codex_config)
        out[AGENT_CLASSES[name]] = cfg
    return out


def build_role_agent(
    chain: List[str],
    *,
    prompt: str = "",
    fallback: bool = True,
    cycles: int = 2,
    codex_config: Optional[List[str]] = None,
) -> AgentInstance:
    """Instantiate a single agent for a role.

    With ``fallback`` (default) the whole chain is wrapped so usage exhaustion
    rolls to the next provider — with per-provider models so codex never
    receives agy's model name. Without it, only the lead provider is used.

    ``codex_config`` is a list of ``key=value`` codex config overrides (e.g.
    ``["tools.web_search=true"]``) applied only to codex providers in the chain.
    """
    if not chain:
        raise ValueError("role chain must be non-empty")

    lead_name, lead_cfg = _cfg_for_token(chain[0], codex_config)

    if not fallback or len(chain) == 1:
        cls = AGENT_CLASSES[lead_name]
        return cls(prompt=prompt, **lead_cfg)

    classes = [AGENT_CLASSES[name] for name in chain]
    fb_cls = make_fallback_agent(
        classes, cycles=cycles, configs=_configs_for(chain, codex_config)
    )
    # Lead config seeds self.model/self.effort; per-provider configs override.
    return fb_cls(prompt=prompt, model=lead_cfg["model"], effort=lead_cfg["effort"])


def build_master_agent_class(
    chain: List[str], *, fallback: bool = True, cycles: int = 2,
    codex_config: Optional[List[str]] = None,
):
    """Return ``(agent_class, model, effort)`` for MasterWorkflow.

    Master uses a single ``agent_class`` for every internal call, so we hand it
    the fallback wrapper (carrying per-provider configs) plus the lead provider's
    model/effort as the seed values.
    """
    if not chain:
        raise ValueError("role chain must be non-empty")
    lead_name, lead_cfg = _cfg_for_token(chain[0], codex_config)
    if not fallback or len(chain) == 1:
        return AGENT_CLASSES[lead_name], lead_cfg["model"], lead_cfg["effort"]
    classes = [AGENT_CLASSES[name] for name in chain]
    fb_cls = make_fallback_agent(
        classes, cycles=cycles, configs=_configs_for(chain, codex_config)
    )
    return fb_cls, lead_cfg["model"], lead_cfg["effort"]


def describe_chain(chain: List[str], fallback: bool) -> str:
    if not fallback or len(chain) == 1:
        name, cfg = _cfg_for_token(chain[0])
        return f"{name}:{cfg['model']}:{cfg['effort']}"
    parts = []
    for token in chain:
        name, cfg = _cfg_for_token(token)
        parts.append(f"{name}({cfg['model']}/{cfg['effort']})")
    return "->".join(parts) + " (cycled)"

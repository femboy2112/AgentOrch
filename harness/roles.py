"""Role definitions: how the harness maps abstract roles to concrete agents.

Per the operating agreement:
  - codex writes code (the generator / primary worker).
  - agy, at high effort on its best model, is the reviewer/critic — but agy
    usage gets exhausted, so any agy role is ALWAYS fallback-wrapped to codex.
  - both roles are fallback-wrapped so a single provider's usage wall never
    stalls a working session.

Every constructed agent also gets its streaming watchdog auto-armed from a
process-wide CalibrationTable (loaded once, lazily). With no calibration data
the budgets are CONSERVATIVE defaults that won't false-positive on any
successful run we've measured. Opt-out via ``AGY_WATCHDOG=off``.

Everything here is overridable per dispatch from the CLI; these are defaults.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.core.agents.agy_agent import AgyAgent
from agy_orchestrator.core.agents.claude_agent import ClaudeAgent
from agy_orchestrator.core.agents.codex_agent import CodexAgent
from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent
from agy_orchestrator.core.agents.grok_agent import GrokAgent
from agy_orchestrator.core.calibration import CalibrationTable

# Step 12: computer-use as selectable standard worker token (resolves to shim)
from harness.computer_use_role import (
    ComputerUseShim,  # thin adapter delegate; no LLM impact (Step 10: forwards real_gui/ask via its config)
)

logger = logging.getLogger(__name__)

AGENT_CLASSES: Dict[str, Type[AgentInstance]] = {
    "codex": CodexAgent,
    "agy": AgyAgent,
    "claude": ClaudeAgent,
    "grok": GrokAgent,
    # Step 12: 'computer-use' resolves via special-case below (shim, not AgentInstance subclass)
}


def _class_for(name: str) -> Type[AgentInstance]:
    """Resolve a worker token to its agent class.

    Single factory indirection for the speed benchmark: with ``AGY_BENCH_MOCK=1``
    every real worker class is swapped for a hermetic ``MockAgent`` bound to that
    worker name, so a dispatch runs end-to-end (prompt construction, baseline
    verify, streaming/watchdog, usage events, meta.json) with ZERO network and ZERO
    contention on the codex/agy/grok pool. The mock class is stable per name, so it
    keys ``configs_by_cls``/``name_by_cls`` consistently with instantiation. No
    effect on a normal run (env unset → the real class).
    """
    if os.environ.get("AGY_BENCH_MOCK"):
        from agy_orchestrator.core.agents.mock_agent import mock_agent_class_for
        return mock_agent_class_for(name)
    return AGENT_CLASSES[name]

# computer-use is a standard selectable worker token (Step 12 wiring).
# It does not resolve to an LLM AgentInstance; dispatch short-circuits to
# ComputerUseWorkerAdapter for --generator computer-use (direct mode).
COMPUTER_USE_TOKEN = "computer-use"

# Provider family for each worker — used by the cross-family verifier guard
# (see check_chains_cross_family below). The literature on LLM-as-judge
# ("When Does Verification Pay Off?", arxiv 2512.02304) finds that
# self-verification — critic from the same model family as the generator —
# substantially underperforms cross-family verification: the critic
# inherits the same blind spots as the generator, so failure modes go
# unnoticed. We surface this as a warning, not a block — the operator may
# override deliberately (e.g. only one provider is configured).
WORKER_FAMILY: Dict[str, str] = {
    "codex": "openai",
    "claude": "anthropic",
    "agy": "google",
    "grok": "xai",
    "computer-use": "computer-use",
}

# Per-provider model/effort. "best model" per provider, high effort.
AGENT_DEFAULTS: Dict[str, Dict[str, object]] = {
    "codex": {"model": "standard", "effort": "high"},   # -> gpt-5.3-codex
    "agy": {"model": "pro", "effort": "high"},           # agy's best tier
    "claude": {"model": "opus", "effort": "high"},
    # xAI Grok agentic CLI. Only `grok-build` exists today and it REJECTS the
    # reasoning-effort param, so effort is "n/a" and GrokAgent never sends it.
    "grok": {"model": "grok-build", "effort": "n/a"},
    # Step 12: computer-use token (non-LLM); used by _cfg_for_token + describe_chain
    # Step 10: real_gui_policy/ask_mode travel via shim.computer_use_config (not this defaults map)
    "computer-use": {"model": "computer-use", "effort": "n/a"},
}

# Ordered fallback chains by role. Generator leads with codex (the code writer);
# critic leads with agy (premium review) then drops to codex when agy is walled.
# Default excludes claude (account-sharing rule per CLAUDE.md §"Account-sharing rule"
# and dashboard-design.md §8): use codex/agy/grok cycled for generator/critic.
GENERATOR_CHAIN: List[str] = ["codex", "agy", "grok"]
CRITIC_CHAIN: List[str] = ["agy", "codex", "grok"]


# --- watchdog arming ------------------------------------------------------- #
# A single CalibrationTable is loaded on first need and reused for the process.
# Re-loading per dispatch would re-read the JSONL on every call for no benefit;
# tests can reset via _reset_calibration() if they need to.

_CALIBRATION: Optional[CalibrationTable] = None
RolePostConstructHook = Callable[[AgentInstance, str, Dict[str, object]], None]


def _get_calibration() -> CalibrationTable:
    global _CALIBRATION
    if _CALIBRATION is None:
        _CALIBRATION = CalibrationTable.load()
    return _CALIBRATION


def _reset_calibration() -> None:
    """Force a re-load on next access. Test hook."""
    global _CALIBRATION
    _CALIBRATION = None


def _arm_watchdog(
    agent: AgentInstance, worker: str, cfg: Dict[str, object], *, scale: float = 1.0,
) -> None:
    """Arm the streaming watchdog from the calibration table.

    Skipped entirely when ``AGY_WATCHDOG=off`` (so an operator who wants the
    legacy fail-fast-only path can disable the safety net). Also skipped when
    the agent already has explicit budgets set (env-var override case).

    ``scale`` (>= 1.0 in practice) multiplies BOTH the byte and stall budgets.
    A higher-effort / stronger-model run (e.g. codex ``gpt-5.5`` / ``xhigh`` from
    issue #42) is longer and more verbose than the calibrated ``high`` baseline,
    so when no calibration row exists for the exact ``(worker, model, effort)``
    we (a) already fall back to the CONSERVATIVE per-worker defaults — never the
    tighter calibrated budget — and (b) let the operator widen them further via
    ``--watchdog-scale`` so a known-heavy run can't be truncated mid-flight.
    Scale resolves from the explicit arg or the ``AGY_WATCHDOG_SCALE`` env var."""
    if os.environ.get("AGY_WATCHDOG", "").lower() == "off":
        return
    if agent.max_output_bytes > 0 or agent.stall_seconds > 0:
        return  # explicit env-var override; respect it
    model = str(cfg.get("model", "") or "")
    effort_val = cfg.get("effort")
    effort = str(effort_val) if effort_val not in (None, "n/a") else None
    cal = _get_calibration()
    max_bytes, stall = cal.budget_for(worker, model, effort)
    if not cal.has_data_for(worker, model, effort):
        logger.debug(
            "watchdog: no calibration for (%s, %s, %s) — using conservative "
            "budget (max_bytes=%d, stall=%.0fs)", worker, model, effort, max_bytes, stall,
        )
    try:
        env_scale = float(os.environ.get("AGY_WATCHDOG_SCALE", "") or 0) or 0.0
    except ValueError:
        env_scale = 0.0
    eff_scale = scale if scale and scale > 0 else 1.0
    if env_scale > 0:
        eff_scale = env_scale
    if eff_scale != 1.0:
        max_bytes = int(max_bytes * eff_scale)
        stall = stall * eff_scale
        logger.info(
            "watchdog: scaled budgets x%.2f for (%s, %s, %s) -> max_bytes=%d, stall=%.0fs",
            eff_scale, worker, model, effort, max_bytes, stall,
        )
    agent.max_output_bytes = max_bytes
    agent.stall_seconds = stall


def _cfg_for_token(
    token: str,
    codex_config: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[str, Dict[str, object]]:
    """Resolve a chain token to ``(agent_name, config)``.

    A token is a plain agent name (``codex``/``agy``/``claude``/``grok``) or
    the special ``computer-use`` worker (Step 12). The latter returns a stub
    config; dispatch routes it to the adapter instead of LLM path.

    ``overrides`` (issue #42) is a per-provider ``{provider: {model, effort}}``
    map merged over ``AGENT_DEFAULTS`` for THIS token — applied uniformly to the
    lead and every fallback sub so e.g. codex runs at the requested tier wherever
    it appears in the chain."""
    name = token
    if name == COMPUTER_USE_TOKEN:
        cfg = {"model": "n/a", "effort": "n/a"}
    else:
        cfg = dict(AGENT_DEFAULTS[name])
        if overrides and name in overrides:
            for k, v in overrides[name].items():
                if v is not None:
                    cfg[k] = v
    if name == "codex" and codex_config:
        cfg["config_overrides"] = list(codex_config)
    return name, cfg


def _configs_for(
    chain: List[str],
    codex_config: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[Type[AgentInstance], Dict[str, object]]:
    out: Dict[Type[AgentInstance], Dict[str, object]] = {}
    for token in chain:
        name, cfg = _cfg_for_token(token, codex_config, overrides)
        if name == COMPUTER_USE_TOKEN:
            continue  # Step 12: cu never participates in LLM fallback dicts
        out[_class_for(name)] = cfg
    return out


def build_role_agent(
    chain: List[str],
    *,
    prompt: str = "",
    model: Optional[str] = None,
    effort: Optional[str] = None,
    fallback: bool = True,
    cycles: int = 2,
    codex_config: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Dict[str, str]]] = None,
    watchdog_scale: float = 1.0,
    computer_use_config: Optional[Dict[str, Any]] = None,
    post_construct_hook: Optional[RolePostConstructHook] = None,
) -> AgentInstance:
    """Instantiate a single agent for a role.

    With ``fallback`` (default) the whole chain is wrapped so usage exhaustion
    rolls to the next provider — with per-provider models so codex never
    receives agy's model name. Without it, only the lead provider is used.

    ``codex_config`` is a list of ``key=value`` codex config overrides (e.g.
    ``["tools.web_search=true"]``) applied only to codex providers in the chain.

    ``overrides`` (issue #42) is a per-provider ``{provider: {model, effort}}``
    map merged over AGENT_DEFAULTS for every provider in the chain; the per-call
    ``model=``/``effort=`` scalars still apply to the lead as a final layer.
    ``watchdog_scale`` widens the armed stall/byte budgets for known-heavy tiers.
    """
    if not chain:
        raise ValueError("role chain must be non-empty")

    lead_name, lead_cfg = _cfg_for_token(chain[0], codex_config, overrides)
    if model is not None:
        lead_cfg["model"] = model
    if effort is not None:
        lead_cfg["effort"] = effort

    if lead_name == COMPUTER_USE_TOKEN:
        # Step 12: return shim (no LLM class, no watchdog, hook still runs for cwd/run linkage)
        agent = ComputerUseShim(prompt=prompt, computer_use_config=computer_use_config, **lead_cfg)
        if post_construct_hook is not None:
            try:
                post_construct_hook(agent, lead_name, lead_cfg)
            except Exception:
                pass
        return agent  # type: ignore[return-value]

    if not fallback or len(chain) == 1:
        cls = _class_for(lead_name)
        agent = cls(prompt=prompt, **lead_cfg)
        _arm_watchdog(agent, lead_name, lead_cfg, scale=watchdog_scale)
        if post_construct_hook is not None:
            try:
                post_construct_hook(agent, lead_name, lead_cfg)
            except Exception:
                pass
        return agent

    classes = [_class_for(name) for name in chain]
    configs_by_cls = _configs_for(chain, codex_config, overrides)
    # name_by_cls maps each agent class back to its worker token so the hook can
    # look up the right (worker, model, effort) budget for whichever sub the
    # FallbackAgent instantiates this attempt.
    name_by_cls: Dict[Type[AgentInstance], str] = {
        _class_for(name): name for name in chain
    }

    def arm_hook(sub: AgentInstance, agent_cls: Type[AgentInstance]) -> None:
        worker = name_by_cls.get(agent_cls)
        if worker is None:
            return
        cfg = configs_by_cls.get(agent_cls, {})
        _arm_watchdog(sub, worker, cfg, scale=watchdog_scale)
        if post_construct_hook is not None:
            try:
                post_construct_hook(sub, worker, cfg)
            except Exception:
                pass

    fb_cls = make_fallback_agent(
        classes, cycles=cycles, configs=configs_by_cls,
        post_construct_hook=arm_hook,
    )
    # Lead config seeds self.model/self.effort; per-provider configs override.
    return fb_cls(prompt=prompt, model=lead_cfg["model"], effort=lead_cfg["effort"])


def build_master_agent_class(
    chain: List[str], *, fallback: bool = True, cycles: int = 2,
    codex_config: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Dict[str, str]]] = None,
    watchdog_scale: float = 1.0,
    post_construct_hook: Optional[RolePostConstructHook] = None,
):
    """Return ``(agent_class, model, effort)`` for MasterWorkflow.

    Master uses a single ``agent_class`` for every internal call, so we hand it
    the fallback wrapper (carrying per-provider configs + the watchdog arm hook)
    plus the lead provider's model/effort as the seed values. Single-class master
    runs get armed when the workflow itself instantiates the agent; FallbackAgent
    arms via post_construct_hook on each _make_sub.

    ``overrides``/``watchdog_scale`` (issue #42) flow through identically to
    ``build_role_agent`` so a master/pat build honours --gen-effort/--codex-model
    /--effort-profile on the architect chain.
    """
    if not chain:
        raise ValueError("role chain must be non-empty")
    lead_name, lead_cfg = _cfg_for_token(chain[0], codex_config, overrides)
    if lead_name == COMPUTER_USE_TOKEN:
        # cu not supported in master workflows; return a stub class for graceful fail
        return ComputerUseShim, lead_cfg["model"], lead_cfg["effort"]
    if not fallback or len(chain) == 1:
        cls = _class_for(lead_name)
        if post_construct_hook is None:
            return cls, lead_cfg["model"], lead_cfg["effort"]

        class HookedLead(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                try:
                    post_construct_hook(self, lead_name, lead_cfg)
                except Exception:
                    pass

        HookedLead.__name__ = f"Hooked{cls.__name__}"
        return HookedLead, lead_cfg["model"], lead_cfg["effort"]
    classes = [_class_for(name) for name in chain]
    configs_by_cls = _configs_for(chain, codex_config, overrides)
    name_by_cls: Dict[Type[AgentInstance], str] = {
        _class_for(name): name for name in chain
    }

    def arm_hook(sub: AgentInstance, agent_cls: Type[AgentInstance]) -> None:
        worker = name_by_cls.get(agent_cls)
        if worker is None:
            return
        cfg = configs_by_cls.get(agent_cls, {})
        _arm_watchdog(sub, worker, cfg, scale=watchdog_scale)
        if post_construct_hook is not None:
            try:
                post_construct_hook(sub, worker, cfg)
            except Exception:
                pass

    fb_cls = make_fallback_agent(
        classes, cycles=cycles, configs=configs_by_cls,
        post_construct_hook=arm_hook,
    )
    return fb_cls, lead_cfg["model"], lead_cfg["effort"]


def check_chains_cross_family(
    generator_chain: List[str], critic_chain: List[str]
) -> Optional[str]:
    """Return a human-readable warning if the lead generator and lead critic
    are from the same provider family, else None.

    Same-family critic = self-verification. The critic inherits the same
    biases and blind spots as the generator and tends to under-flag the
    failure modes most worth catching. Cross-family critic — drawn from a
    different model family — catches substantially more (arxiv 2512.02304).

    Falls open on unknown workers (returns None) — better to stay silent
    than to false-positive on a worker we haven't mapped.

    Disable the check entirely via env var ``AGY_CRITIC_FAMILY_CHECK=off``
    for operators who deliberately want self-critique (e.g. only one
    provider configured)."""
    if os.environ.get("AGY_CRITIC_FAMILY_CHECK", "").lower() == "off":
        return None
    if not generator_chain or not critic_chain:
        return None
    gen_lead = generator_chain[0]
    crit_lead = critic_chain[0]
    gen_family = WORKER_FAMILY.get(gen_lead)
    crit_family = WORKER_FAMILY.get(crit_lead)
    if gen_family is None or crit_family is None:
        return None
    if gen_family != crit_family:
        return None
    return (
        f"cross-family check: generator={gen_lead!r} and critic={crit_lead!r} "
        f"are both in the {gen_family!r} family — self-verification "
        f"substantially underperforms cross-family critique (arxiv 2512.02304). "
        f"Consider --critic <different-provider> for stronger gates. "
        f"Suppress with AGY_CRITIC_FAMILY_CHECK=off."
    )


def check_agy_parallelism_warning(
    mode: str, generator_chain: List[str], branches: int
) -> Optional[str]:
    """Warn when a high-parallel mode would spawn ≥2 agy candidates.

    Agy's CLI pins ~/.gemini/settings.json via a cross-process file lock to
    keep its single-call model selection safe. Two agy candidates running in
    parallel serialize on that lock — the parallelism gain of vote/tot is
    lost. Cross-provider chains (codex,agy,grok) are unaffected because
    only ONE candidate per dispatch lands on agy.

    Disable via env var AGY_PARALLELISM_CHECK=off."""
    if os.environ.get("AGY_PARALLELISM_CHECK", "").lower() == "off":
        return None
    if mode not in ("vote", "tot"):
        return None
    # vote rotates through generator_chain. tot uses the master agent_class
    # (the lead provider). For both, count how many of the first 'branches'
    # slots would land on agy.
    if not generator_chain:
        return None
    k = max(1, branches)
    agy_slots = sum(
        1 for i in range(k) if generator_chain[i % len(generator_chain)] == "agy"
    )
    if agy_slots < 2:
        return None
    return (
        f"agy-parallelism check: mode={mode!r} with branches={k} would spawn "
        f"{agy_slots} agy candidates in parallel. agy serializes on a cross-"
        f"process settings.json lock, so the parallelism gain is lost. Consider "
        f"a more heterogeneous --generator chain. Suppress with "
        f"AGY_PARALLELISM_CHECK=off."
    )


def describe_chain(chain: List[str], fallback: bool, **kwargs: Any) -> str:
    """Describe for meta/logs; **kwargs accepts real_gui_policy/ask_mode/browser_engine/browser_display (and computer_use_* variants) passed from dispatch for full wiring to shim/RunRequest paths (per Step 10+8); values deliberately ignored so all computer-use:isolated strings and outputs remain byte-identical (INVARIANT F, no behavior/output change for any run)."""
    if not fallback or len(chain) == 1:
        name, cfg = _cfg_for_token(chain[0])
        if name == COMPUTER_USE_TOKEN:
            # Step 10: real-gui flags do not alter this string for non-real runs (INVARIANT F / byte-identical); default ISOLATED always here
            return "computer-use:isolated:n/a"
        return f"{name}:{cfg['model']}:{cfg['effort']}"
    parts = []
    for token in chain:
        name, cfg = _cfg_for_token(token)
        if name == COMPUTER_USE_TOKEN:
            # keep "iso" output exactly for all non-real (and current) paths; real_gui only affects req/events
            parts.append("computer-use(iso)")
        else:
            parts.append(f"{name}({cfg['model']}/{cfg['effort']})")
    return "->".join(parts) + " (cycled)"

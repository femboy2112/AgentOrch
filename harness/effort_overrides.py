"""Per-role / per-provider effort + model overrides for ``harness do`` (issue #42).

The harness bakes a sensible default tier into ``roles.AGENT_DEFAULTS`` (codex on
``gpt-5.3-codex`` / ``high``), but the *right* effort for a build is a
**per-dispatch** decision: trivial edits want the cheap default; a
mission-critical, invariant-touching, single-shot build wants every provider at
its ceiling (codex ``gpt-5.5`` / ``xhigh``). The capability already exists one
layer down — ``build_role_agent`` takes ``model=``/``effort=`` and ``CodexAgent``
maps ``effort="max"`` → ``model_reasoning_effort="xhigh"`` — it was simply never
plumbed to the CLI.

This module is the resolution layer: it turns the CLI surface

  * per-role scalars   ``--gen-effort`` / ``--critic-effort`` / ``--architect-effort``
                       ``--gen-model``  / ``--critic-model``
  * per-provider maps  ``--effort "codex=max,agy=high"`` / ``--model "codex=gpt-5.5"``
  * the codex shortcut ``--codex-model gpt-5.5``
  * the one-switch     ``--effort-profile low|balanced|max``

into a per-role ``{provider: {"model": .., "effort": ..}}`` override map that
``roles.build_role_agent`` / ``build_master_agent_class`` merge over
``AGENT_DEFAULTS`` for **every** provider in the chain (lead *and* fallback), so
``codex`` runs at the requested tier whether it leads the generator chain or sits
mid-critic-chain.

Design rules (the "get it right" bits):
  * **grok no-ops effort.** Grok's CLI rejects the reasoning-effort param, so an
    effort override targeting grok is dropped with a note — never an error, never
    a broken fallback chain.
  * **effort scalars fan out to all effort-capable providers** in the role chain
    (so ``--gen-effort max`` lifts codex *and* agy); **model scalars target the
    chain lead only** (a model name is provider-specific — you can't put
    ``gpt-5.5`` on agy), while ``--codex-model`` targets codex anywhere.
  * **explicit flags beat the profile.** The profile is the base; any explicit
    flag layered on top wins, with a note.
  * **validation up front.** Unknown effort tier or unknown codex model fails
    fast with a clear, enumerated error (raise :class:`OverrideError`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agy_orchestrator.core.agents.codex_agent import CodexAgent

# CLI effort tiers. ``max`` is codex's ``xhigh`` (a tier ABOVE the default
# ``high``); the codex agent performs that aliasing at command-build time.
EFFORT_TIERS = ("low", "medium", "high", "max")

# Providers whose CLI accepts a reasoning-effort knob. grok is excluded: its
# agentic CLI rejects the param, so we never send it one.
EFFORT_CAPABLE = frozenset({"codex", "agy", "claude"})

# Named one-switch presets. ``balanced`` == the baked-in AGENT_DEFAULTS (a no-op
# you can name explicitly); ``max`` cranks every effort-capable provider to its
# strongest model + ceiling effort; ``low`` drops to the cheap tier. grok is
# deliberately absent from every preset (effort n/a, single model).
EFFORT_PROFILES: Dict[str, Dict[str, Dict[str, str]]] = {
    "low": {
        "codex": {"model": "gpt-5.4-mini", "effort": "low"},
        "agy": {"effort": "low"},
        "claude": {"effort": "low"},
    },
    "balanced": {
        "codex": {"model": "standard", "effort": "high"},
        "agy": {"model": "pro", "effort": "high"},
        "claude": {"model": "opus", "effort": "high"},
    },
    "max": {
        "codex": {"model": "gpt-5.5", "effort": "max"},
        "agy": {"model": "pro", "effort": "high"},
        "claude": {"model": "opus", "effort": "high"},
    },
}


class OverrideError(ValueError):
    """Raised on an invalid effort tier / unknown model / malformed map.

    Carries a human-readable, enumerated message the CLI prints verbatim before
    exiting non-zero (so the operator never gets a stack trace for a typo)."""


def _codex_models() -> List[str]:
    """Codex's known model names (single source of truth: the agent itself)."""
    return list(CodexAgent.AVAILABLE_MODELS)


def validate_effort(value: str, *, where: str) -> str:
    if value not in EFFORT_TIERS:
        raise OverrideError(
            f"invalid effort {value!r} for {where}; valid tiers: "
            f"{'|'.join(EFFORT_TIERS)} (codex maps 'max' -> reasoning_effort=xhigh)"
        )
    return value


def validate_model(provider: str, value: str, *, where: str) -> str:
    # Only codex exposes a machine-checkable model list; validate it strictly so
    # a typo'd --codex-model fails fast. Other providers pass through (we don't
    # hold an authoritative list for them) — the provider CLI rejects a bad name.
    if provider == "codex" and value not in _codex_models():
        raise OverrideError(
            f"unknown codex model {value!r} for {where}; available: "
            f"{', '.join(_codex_models())}"
        )
    return value


def parse_kv_map(spec: Optional[str], *, flag: str) -> Dict[str, str]:
    """Parse ``"codex=max,agy=high"`` -> ``{'codex':'max','agy':'high'}``.

    A bare scalar with no ``=`` (e.g. ``"max"``) -> ``{'*': 'max'}`` meaning
    "apply to every applicable provider". Empty/None -> ``{}``. Malformed
    entries raise :class:`OverrideError`."""
    if not spec:
        return {}
    spec = spec.strip()
    if not spec:
        return {}
    if "=" not in spec:
        # scalar form
        if "," in spec:
            raise OverrideError(
                f"{flag}: a comma-separated value must use provider=value pairs "
                f"(e.g. 'codex=max,agy=high'), got {spec!r}"
            )
        return {"*": spec}
    out: Dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise OverrideError(
                f"{flag}: malformed entry {part!r} (expected provider=value)"
            )
        key, val = part.split("=", 1)
        key, val = key.strip(), val.strip()
        if not key or not val:
            raise OverrideError(
                f"{flag}: malformed entry {part!r} (expected provider=value)"
            )
        out[key] = val
    return out


@dataclass
class ResolvedOverrides:
    """Resolved per-role override maps + advisory notes + watchdog scale.

    ``generator``/``critic`` are ``{provider: {"model": .., "effort": ..}}`` maps
    (only the keys actually overridden are present) that roles.py merges over
    AGENT_DEFAULTS. ``notes`` are human-readable advisories (grok no-op, profile
    overridden, unvalidated model) the CLI surfaces. ``watchdog_scale`` multiplies
    the armed stall/byte budgets for known-heavy tiers."""

    generator: Dict[str, Dict[str, str]] = field(default_factory=dict)
    critic: Dict[str, Dict[str, str]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    watchdog_scale: float = 1.0

    def for_role(self, role: str) -> Dict[str, Dict[str, str]]:
        return self.critic if role == "critic" else self.generator

    def is_empty(self) -> bool:
        return not self.generator and not self.critic and self.watchdog_scale == 1.0


def _apply_effort_scalar(ov: Dict[str, Dict[str, str]], chain: List[str],
                         value: str, notes: List[str], *, where: str) -> None:
    """Apply one effort tier to every effort-capable provider in ``chain``."""
    validate_effort(value, where=where)
    for p in chain:
        if p in EFFORT_CAPABLE:
            ov.setdefault(p, {})["effort"] = value
    if "grok" in chain:
        notes.append(f"{where}: grok ignores reasoning-effort — effort no-op for grok")


def _apply_model_to_lead(ov: Dict[str, Dict[str, str]], chain: List[str],
                         value: str, notes: List[str], *, where: str) -> None:
    """Apply one model name to the chain lead (model is provider-specific)."""
    if not chain:
        return
    lead = chain[0]
    validate_model(lead, value, where=where)
    ov.setdefault(lead, {})["model"] = value
    if lead != "codex":
        notes.append(f"{where}: model {value!r} for {lead} is not validated "
                     f"(only codex has a checkable model list)")


def _apply_map(ov: Dict[str, Dict[str, str]], chain: List[str], kv: Dict[str, str],
               field_name: str, notes: List[str], *, flag: str) -> None:
    """Apply a parsed provider=value map (or scalar ``*``) for one field."""
    if not kv:
        return
    chain_set = set(chain)
    if "*" in kv:
        value = kv["*"]
        for p in chain:
            if field_name == "effort":
                if p in EFFORT_CAPABLE:
                    validate_effort(value, where=flag)
                    ov.setdefault(p, {})["effort"] = value
            else:  # model scalar -> lead only (provider-specific)
                pass
        if field_name == "model":
            _apply_model_to_lead(ov, chain, value, notes, where=flag)
        elif field_name == "effort" and "grok" in chain:
            notes.append(f"{flag}: grok ignores reasoning-effort — no-op for grok")
    for provider, value in kv.items():
        if provider == "*":
            continue
        if provider == "grok" and field_name == "effort":
            notes.append(f"{flag}: grok ignores reasoning-effort — dropping grok={value!r}")
            continue
        if field_name == "effort":
            validate_effort(value, where=flag)
        else:
            validate_model(provider, value, where=flag)
        if provider not in chain_set:
            notes.append(f"{flag}: {provider} is not in this chain — override recorded "
                         f"but only applies if {provider} is reached")
        ov.setdefault(provider, {})[field_name] = value


def resolve_overrides(
    *,
    generator_chain: List[str],
    critic_chain: List[str],
    mode: str = "",
    profile: Optional[str] = None,
    gen_effort: Optional[str] = None,
    gen_model: Optional[str] = None,
    critic_effort: Optional[str] = None,
    critic_model: Optional[str] = None,
    architect_effort: Optional[str] = None,
    architect_model: Optional[str] = None,
    codex_model: Optional[str] = None,
    effort_map: Optional[str] = None,
    model_map: Optional[str] = None,
    watchdog_scale: Optional[float] = None,
) -> ResolvedOverrides:
    """Resolve all override flags into per-role provider maps.

    Layering (later wins): profile (base) < per-provider/effort maps < per-role
    scalars < --codex-model. ``--architect-*`` is an alias for ``--gen-*`` that
    applies to the architect role in master/pat (where the generator chain *is*
    the architect)."""
    notes: List[str] = []

    if profile is not None and profile not in EFFORT_PROFILES:
        raise OverrideError(
            f"unknown --effort-profile {profile!r}; valid: {', '.join(EFFORT_PROFILES)}"
        )

    # Architect flags fold into the generator role for master/pat builds (the
    # generator chain is what the master planner/architect runs on).
    eff_gen = gen_effort
    mod_gen = gen_model
    if mode in ("master", "pat"):
        if architect_effort is not None:
            eff_gen = architect_effort
        if architect_model is not None:
            mod_gen = architect_model
    elif architect_effort is not None or architect_model is not None:
        notes.append("--architect-effort/--architect-model only apply to "
                     "--mode master/pat; ignored for this mode")

    effort_kv = parse_kv_map(effort_map, flag="--effort")
    model_kv = parse_kv_map(model_map, flag="--model")

    explicit_effort = any(v is not None for v in (gen_effort, critic_effort,
                                                  architect_effort)) or bool(effort_kv)
    explicit_model = any(v is not None for v in (gen_model, critic_model, codex_model,
                                                 architect_model)) or bool(model_kv)
    if profile is not None and (explicit_effort or explicit_model):
        notes.append(f"--effort-profile {profile!r} is the base; explicit "
                     f"--gen/--critic/--codex/--effort/--model flags override it")

    def _build(chain: List[str], role: str, role_effort: Optional[str],
               role_model: Optional[str]) -> Dict[str, Dict[str, str]]:
        ov: Dict[str, Dict[str, str]] = {}
        # 1. profile (base)
        if profile is not None:
            for p in chain:
                preset = EFFORT_PROFILES[profile].get(p)
                if preset:
                    ov.setdefault(p, {}).update(preset)
        # 2. per-provider maps (role-agnostic)
        _apply_map(ov, chain, effort_kv, "effort", notes, flag="--effort")
        _apply_map(ov, chain, model_kv, "model", notes, flag="--model")
        # 3. per-role scalars
        if role_effort is not None:
            _apply_effort_scalar(ov, chain, role_effort, notes,
                                 where=f"--{role}-effort")
        if role_model is not None:
            _apply_model_to_lead(ov, chain, role_model, notes,
                                 where=f"--{role}-model")
        # 4. --codex-model (most specific convenience; both roles)
        if codex_model is not None and "codex" in chain:
            validate_model("codex", codex_model, where="--codex-model")
            ov.setdefault("codex", {})["model"] = codex_model
        return ov

    generator = _build(generator_chain, "gen", eff_gen, mod_gen)
    critic = _build(critic_chain, "critic", critic_effort, critic_model)

    scale = 1.0
    if watchdog_scale is not None:
        if watchdog_scale <= 0:
            raise OverrideError(f"--watchdog-scale must be > 0, got {watchdog_scale}")
        scale = float(watchdog_scale)

    # De-duplicate notes (the same advisory can arise from both roles) while
    # preserving order.
    seen: set = set()
    deduped = [n for n in notes if not (n in seen or seen.add(n))]
    return ResolvedOverrides(generator=generator, critic=critic,
                             notes=deduped, watchdog_scale=scale)


def effective_config(chain: List[str], overrides: Dict[str, Dict[str, str]],
                     agent_defaults: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, str]]:
    """Compute the effective ``{provider: {model, effort}}`` for a chain.

    Merges the resolved overrides over AGENT_DEFAULTS for each provider actually
    in the chain — exactly what gets dispatched. Persisted into ``meta.json`` so
    "what did this run use" is answerable from the captured artifacts (#42 item 5)."""
    out: Dict[str, Dict[str, str]] = {}
    for p in chain:
        base = agent_defaults.get(p, {})
        merged = {
            "model": str(base.get("model", "") or ""),
            "effort": str(base.get("effort", "") or ""),
        }
        if p in overrides:
            for k, v in overrides[p].items():
                if v is not None:
                    merged[k] = str(v)
        out[p] = merged
    return out

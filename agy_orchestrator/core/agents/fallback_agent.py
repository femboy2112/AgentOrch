"""Cross-provider fallback agent.

Wraps an ordered chain of agent classes (e.g. codex -> agy -> claude). Each call
tries the primary provider; if it fails after its own internal retries (which is
what "runs out of usage"/quota looks like — the CLI exits non-zero and the base
run_async raises RuntimeError), the next provider in the chain is tried. This makes
a long autonomous build resilient to one provider exhausting its usage mid-run.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from typing import Callable, Dict, List, Optional, Type

from agy_orchestrator.core.agent import (
    WATCHDOG_MARKER,
    WATCHDOG_TRANSPORT_STALL,
    WATCHDOG_VERBOSE,
    AgentInstance,
    is_context_overflow,
    is_model_unavailable,
    is_transport_error,
    is_usage_wall,
    is_wedged_session,
)
from agy_orchestrator.core.agents.claude_agent import ClaudeAgent
from agy_orchestrator.core.agents.grok_agent import GrokAgent

logger = logging.getLogger(__name__)

# A post-construct hook receives each newly-built sub-agent (and its class) so
# the caller can configure it before run_async — e.g. arm the streaming watchdog
# with per-provider budgets from a CalibrationTable. Kept generic so FallbackAgent
# stays unaware of calibration; the harness owns the policy.
PostConstructHook = Callable[[AgentInstance, Type[AgentInstance]], None]

# Agent classes that support warm-session resume across calls.
_SESSION_CAPABLE = (ClaudeAgent, GrokAgent)

# Process-level dynamic model-availability cache. When a provider's model fails
# with a model-unavailable signature (e.g. codex reports "model is not supported
# when using a ChatGPT account" — invisible on stderr, folded in by CodexAgent),
# we record (provider_class_name, model) here so NO later step in this process ever
# offers that model again. This is the "dynamic detection" the operator asked for:
# the valid fallback set self-discovers per account instead of being hardcoded — a
# model the account can't use is tried at most once, then pruned for the run.
_INACCESSIBLE_MODELS: set = set()


def record_inaccessible_model(provider: str, model: Optional[str]) -> None:
    if model:
        _INACCESSIBLE_MODELS.add((provider, str(model)))


def is_model_known_inaccessible(provider: str, model: Optional[str]) -> bool:
    return bool(model) and (provider, str(model)) in _INACCESSIBLE_MODELS


def reset_inaccessible_models() -> None:
    """Clear the dynamic model-availability cache (test/diagnostic hook)."""
    _INACCESSIBLE_MODELS.clear()


# Process-level TIME-BOUNDED rate-limit cache (issue #82). When a provider's model
# returns a usage/quota wall that ALSO reports a reset time (codex stashes the
# rollout-JSONL reset epoch on the sub as ``usage_wall_resets_at``; folded in by
# CodexAgent._augment_failure_stderr), we record (provider_class_name, model) ->
# resets_at here. The NEXT call skips that exact (provider, model) until ``resets_at``
# passes, instead of re-probing + re-walling it on every call for the whole window.
# Unlike _INACCESSIBLE_MODELS (permanent for the process), this AUTO-EXPIRES: once
# wall-clock reaches resets_at the entry is popped on lookup and the model re-enabled.
# A wall with NO resets_at is NOT cached -> it is still re-probed exactly as before.
_RATE_LIMITED_UNTIL: Dict[tuple, float] = {}


def _now() -> float:
    # SEAM so tests can monkeypatch fallback_agent._now to drive time deterministically.
    return time.time()


def record_rate_limited(provider: str, model: Optional[str], resets_at) -> None:
    """Cache a (provider, model) as rate-limited until ``resets_at`` (unix epoch).

    Stores ONLY if resets_at is a finite number > 0 (math.isfinite). Silently
    ignores None / NaN / +-inf / <= 0 / non-numeric. No-op if model is falsy.
    """
    if not model:
        return
    try:
        ra = float(resets_at)
    except (TypeError, ValueError):
        return
    if not math.isfinite(ra) or ra <= 0:
        return
    _RATE_LIMITED_UNTIL[(provider, str(model))] = ra


def is_rate_limited(provider: str, model: Optional[str], now: Optional[float] = None) -> bool:
    """True if (provider, model) is cached rate-limited and the window hasn't elapsed.

    Auto-expires: once ``now`` reaches the cached resets_at, the entry is popped
    (model re-enabled) and False is returned. A falsy model is never rate-limited.
    """
    if not model:
        return False
    key = (provider, str(model))
    resets_at = _RATE_LIMITED_UNTIL.get(key)
    if resets_at is None:
        return False
    now = now if now is not None else _now()
    if now < resets_at:
        return True
    # Window elapsed -> auto re-enable.
    _RATE_LIMITED_UNTIL.pop(key, None)
    return False


def clear_rate_limited() -> None:
    """Clear the time-bounded rate-limit cache (test/diagnostic hook)."""
    _RATE_LIMITED_UNTIL.clear()


def _watchdog_reason_in(stderr: str) -> Optional[str]:
    """Extract a watchdog trip reason from a failed sub's stderr, or None."""
    if not stderr or WATCHDOG_MARKER not in stderr:
        return None
    # Markers look like '[watchdog:verbose]' or '[watchdog:stalled]'. Pull the slug.
    head = stderr.split(WATCHDOG_MARKER, 1)[1]
    return head.split("]", 1)[0].strip() or None


def _reason_category(*, looked_like_usage: bool, watchdog_reason: Optional[str],
                     context_overflow: bool = False, transport: bool = False,
                     wedged: bool = False) -> str:
    # Context overflow is checked FIRST and is distinct from a quota wall: the
    # same provider can serve the next (smaller) step, so telemetry must not fold
    # it into "usage" (issue #47).
    if context_overflow:
        return "context_overflow"
    # A wedged session (codex stdin-closed hang, #62) is an AGENT-side defect, not
    # a quota wall and not a code failure. Classify it distinctly so it never reads
    # as provider exhaustion (and so the adversarial loop doesn't misattribute a
    # stall-killed generator to a verification failure). Checked before the stall
    # bucket because a wedged session ends in a -9 stall-kill that would otherwise
    # fold it into the generic "stalled" reason.
    if wedged:
        return "wedged"
    if looked_like_usage:
        return "usage"
    # A transport/network blip (websocket reset, ECONNRESET, transient 502/503)
    # failed the CONNECTION, not the provider — keep it separable from a real quota
    # wall so flakiness telemetry doesn't read as provider exhaustion (issue #57).
    # The watchdog's transport-stall trip funnels here too.
    if transport or watchdog_reason == WATCHDOG_TRANSPORT_STALL:
        return "transport"
    if watchdog_reason == "verbose":
        return "verbose"
    if watchdog_reason == "stalled":
        return "stalled"
    return "error"


# Bounded same-provider backoff-retries for a transport-classified failure before
# advancing to the next provider (issue #57). The connection blipped, not the
# provider, so a short retry on the SAME provider is the right first move. Kept
# small (1-2) and overridable via env so a flapping link can't burn the cycle
# budget. The backoff is brief because these are network resets, not quota waits.
def _transport_retries() -> int:
    try:
        n = int(os.environ.get("AGY_TRANSPORT_RETRIES", "2") or "2")
    except ValueError:
        n = 2
    return max(0, min(n, 2))


def _transport_backoff_seconds(attempt: int) -> float:
    # attempt is 1-based; 0.5s, 1.0s. Overridable base for tests/tuning.
    try:
        base = float(os.environ.get("AGY_TRANSPORT_BACKOFF", "0.5") or "0.5")
    except ValueError:
        base = 0.5
    return base * attempt


# Backoff applied between re-attempts when a provider returns a usage/quota wall
# (issue: usage-wall cascade). The cycles>1 design exists precisely so a walled
# provider's quota can RECOVER before the cycle comes back around to it — but
# that premise is only meaningful if the loop actually waits between attempts
# instead of spinning the whole cycles*providers budget in the seconds it takes
# the CLIs to emit their quota errors. The wait is per-attempt (so the elapsed
# time accumulates toward a real rate-limit window) and overridable; tests pin
# it to 0 to stay sub-second. Default kept modest so a non-pathological run with
# a single transient wall pays only a small price.
def _usage_wall_backoff_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("AGY_USAGE_WALL_BACKOFF", "5") or "5"))
    except ValueError:
        return 5.0


def make_fallback_agent(
    chain: List[Type[AgentInstance]],
    cycles: int = 3,
    configs: Optional[Dict[Type[AgentInstance], Dict[str, object]]] = None,
    watchdog_rules: Optional[Dict[str, List[Type[AgentInstance]]]] = None,
    post_construct_hook: Optional[PostConstructHook] = None,
    model_fallbacks: Optional[Dict[Type[AgentInstance], List[str]]] = None,
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

    ``watchdog_rules`` optionally maps a watchdog trip reason
    (``"verbose"`` / ``"stalled"``) to an ordered list of agent classes to try
    NEXT when a sub fails with that reason. Use it to route a rambling cheap
    model to a terse one, or an empty/stalled run to a stronger model:

        watchdog_rules={
            "verbose": [CodexAgent],    # haiku rambled -> hand to a terse coder
            "stalled": [ClaudeAgent],   # cheap froze -> escalate
        }

    Rules apply ONCE per trip and re-enter the normal sequence afterwards, so a
    bad chain order can't get stuck in an infinite rules-table ping-pong.

    ``model_fallbacks`` optionally maps an agent class to an ordered list of
    ALTERNATE model names to try on the SAME provider when it hits a usage/quota
    wall, BEFORE advancing to the next provider in the chain. Codex's spark window
    can be exhausted while the account still has headroom on gpt-5.5/gpt-5.4 — those
    are separate rate-limit windows — so a per-model wall should retry the provider
    on its other models first (operator request). Each alternate is tried at most
    once per ``run_async``; once they're exhausted the chain advances normally. Only
    a usage wall triggers this (a transport blip / generic error does not) so it
    composes with the bounded transport-retry path without double-counting.
    """

    if not chain:
        raise ValueError("fallback chain must be non-empty")
    if cycles < 1:
        raise ValueError("cycles must be >= 1")

    class FallbackAgent(AgentInstance):
        _chain: List[Type[AgentInstance]] = list(chain)
        _cycles: int = cycles
        _configs: Dict[Type[AgentInstance], Dict[str, object]] = dict(configs or {})
        _watchdog_rules: Dict[str, List[Type[AgentInstance]]] = {
            reason: list(targets) for reason, targets in (watchdog_rules or {}).items()
        }
        _model_fallbacks: Dict[Type[AgentInstance], List[str]] = {
            cls_: list(models) for cls_, models in (model_fallbacks or {}).items()
        }
        _post_construct_hook: Optional[PostConstructHook] = post_construct_hook

        @classmethod
        def resolve_effective_model(cls, model: Optional[str]) -> Optional[str]:
            # Delegate alias resolution to the chain LEAD (the provider whose
            # model/effort the wrapper carries as its seed) so a codex-led
            # fallback wrapper surfaces gpt-5.3-codex-spark, not "standard".
            lead = cls._chain[0] if cls._chain else None
            if lead is not None:
                try:
                    return lead.resolve_effective_model(model)
                except Exception:
                    pass
            return model

        @classmethod
        def resolve_effective_effort(cls, effort: Optional[str]) -> Optional[str]:
            lead = cls._chain[0] if cls._chain else None
            if lead is not None:
                try:
                    return lead.resolve_effective_effort(effort)
                except Exception:
                    pass
            return effort

        @classmethod
        async def get_available_models(cls) -> List[str]:
            return await cls._chain[0].get_available_models()

        @classmethod
        async def get_model_usage(cls, model: str) -> float:
            return await cls._chain[0].get_model_usage(model)

        def build_command(self, piped_input: Optional[str] = None) -> List[str]:
            # Satisfies the ABC; never used because run_async is overridden.
            return []

        def _make_sub(self, agent_cls: Type[AgentInstance],
                      model_override: Optional[str] = None) -> AgentInstance:
            cfg = self._configs.get(agent_cls, {})
            # A usage-wall model-fallback (#78 / operator request) re-runs the SAME
            # provider on a different model; the override wins over both the per-
            # provider config model and the wrapper's self.model, while the rest of
            # the provider's config (effort, codex config_overrides) still applies.
            model = model_override or cfg.get("model", self.model)
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
            sub = agent_cls(**kwargs)  # type: ignore[arg-type]
            # Belt-and-suspenders (issue #75): propagate the wrapper's watchdog
            # budgets to the sub when the wrapper has them set higher than the
            # sub's own value, so an explicitly-configured wrapper's protection
            # reaches the actual worker even if the post-construct arm hook is
            # absent or raises. The hook (below) may still override afterward. We
            # deliberately do NOT propagate timeout/absolute_timeout — those are
            # intentionally per-call.
            for _budget in ("stall_seconds", "max_output_bytes", "silence_max_seconds"):
                try:
                    wrapper_val = getattr(self, _budget, None)
                    sub_val = getattr(sub, _budget, None)
                    if (wrapper_val is not None and sub_val is not None
                            and wrapper_val > sub_val):
                        setattr(sub, _budget, wrapper_val)
                except Exception:  # never let budget propagation block a real call
                    pass
            cb_factory = getattr(self, "event_callback_factory", None)
            if callable(cb_factory):
                try:
                    sub.event_callback = cb_factory(agent_cls, sub)
                except Exception as exc:
                    logger.warning("[Fallback] event_callback_factory raised: %s", exc)
                    sub.event_callback = self.event_callback
            else:
                sub.event_callback = self.event_callback
            if hasattr(sub, "dashboard_stream_json") and hasattr(self, "dashboard_stream_json"):
                setattr(sub, "dashboard_stream_json", bool(getattr(self, "dashboard_stream_json")))
            # Optional caller-supplied hook (e.g. harness watchdog arming) runs
            # AFTER construction so it sees the final sub with its config applied.
            hook = type(self)._post_construct_hook
            if hook is not None:
                try:
                    hook(sub, agent_cls)
                except Exception as exc:  # never let a bad hook block a real call
                    logger.warning("[Fallback] post_construct_hook raised: %s", exc)
            return sub

        def _emit_orchestration(self, **fields) -> None:
            cb = self.event_callback
            if cb is None:
                return
            orchestration = {"workflow": "master"}
            for key, value in fields.items():
                if value is not None:
                    orchestration[key] = value
            try:
                cb(
                    {
                        "kind": "lifecycle",
                        "data": {
                            "event": "orchestration_transition",
                            "orchestration": orchestration,
                        },
                    }
                )
            except Exception:
                pass

        async def run_async(self, piped_input: Optional[str] = None) -> str:
            last_error: Optional[Exception] = None
            # Repeat the chain ``_cycles`` times: codex -> agy -> claude -> (repeat).
            # ``pending`` is a mutable view we can splice rule-based targets into the
            # front of when a watchdog trips, so the next attempt picks them up.
            # ``rotate_offset`` (set by a caller, e.g. the adversarial loop after K
            # consecutive VERIFY-failures, #65) rotates which provider LEADS so the
            # next generator can attempt a step the previous one couldn't pass —
            # the fallback chain normally only advances on a produce-FAILURE, never
            # on a produced-but-failed-verification output.
            base = list(self._chain)
            off = int(getattr(self, "rotate_offset", 0) or 0)
            if base and off:
                off %= len(base)
                base = base[off:] + base[:off]
            # Each pending entry is (agent_cls, model_override): None means "use the
            # provider's configured model". A usage-wall model-fallback splices the
            # same class with a different model override (see below).
            pending: List[tuple] = [(c, None) for c in base] * self._cycles
            total = len(pending)
            # Per-provider set of alternate models already tried this run, so each
            # configured model_fallback is attempted at most once (the provider's
            # initial configured model counts as already-used via the empty-seed).
            model_fallback_tried: Dict[Type[AgentInstance], set] = {}
            attempts = 0
            # Whether ANY sub has actually been invoked this call. Guards the #82
            # rate-limit skip from exhausting to zero attempts: if every candidate is
            # cache-walled and nothing has run, we attempt anyway instead of skipping.
            ran_any = False
            # Per-provider count of bounded same-provider transport retries already
            # spent, so a flapping link can't loop a single provider forever (#57).
            # This budget is SCOPED PER CYCLE (reset whenever a fresh cycle of the
            # base chain begins) so a link that flaps once per cycle still earns its
            # bounded retries in every later cycle instead of spending the whole
            # budget in cycle 1 and getting none afterward. ``base_len`` lets us
            # detect cycle boundaries by counting how many base-chain slots the
            # *normal sequence* has consumed (re-spliced same-provider retries and
            # rule re-routes don't advance the cycle).
            base_len = len(base) if base else 1
            base_consumed = 0  # normal-sequence chain slots drained so far
            transport_retries_used: Dict[Type[AgentInstance], int] = {}
            max_transport_retries = _transport_retries()
            usage_wall_backoff = _usage_wall_backoff_seconds()
            # Global budget on watchdog-rule re-route splices. Without a cap a self-
            # or mutually-referencing rules table (e.g. A trips 'verbose'->[B],
            # B trips 'stalled'->[A]) re-splices a fresh target every failure, so
            # ``pending`` never drains and run_async() spins forever instead of
            # reaching the exhaustion RuntimeError below (the docstring's "can't get
            # stuck in an infinite rules-table ping-pong" invariant). The budget is
            # generous enough that every legitimate per-trip re-route in a normal
            # chain still fires (one rule fan-out per normal attempt, across every
            # cycle), but bounded so a pathological rules table always terminates.
            reroute_budget = total + sum(
                len(targets) for targets in self._watchdog_rules.values()
            ) * self._cycles
            # Count of items spliced to the FRONT of pending that are NOT normal-
            # sequence advances (same-provider transport retries + rule re-routes).
            # Popping one of these does not advance the cycle, so the per-cycle
            # transport budget must not reset on it.
            respliced = 0
            while pending:
                agent_cls, model_override = pending.pop(0)
                # A normal-sequence advance (not a re-spliced retry/reroute) drains
                # one base-chain slot; crossing a base_len boundary starts a fresh
                # cycle, at which point the per-provider transport-retry budget is
                # reset so a once-per-cycle flap earns bounded retries again.
                if respliced > 0:
                    respliced -= 1
                else:
                    if base_consumed and base_consumed % base_len == 0:
                        transport_retries_used.clear()
                    base_consumed += 1
                # Issue #82: skip a (provider, model) we already know is rate-limited
                # until its cached reset window passes — don't re-probe + re-wall it on
                # every call. Resolve the model exactly like the usage-wall path below
                # (model_override or the provider's configured model). SAFETY: never
                # skip down to ZERO attempts — if every remaining candidate is cached-
                # walled AND nothing has actually run this call, attempt anyway (a run
                # that probes nothing is a worse regression than re-probing).
                skip_provider_key = agent_cls.__name__
                skip_resolved_model = model_override or self._configs.get(
                    agent_cls, {}).get("model", self.model)
                if is_rate_limited(skip_provider_key, skip_resolved_model):
                    # Before abandoning the WHOLE provider, preserve #78's "try the
                    # provider's OTHER models before switching providers": if a
                    # same-provider alternate model is configured and is itself
                    # neither cache-walled nor inaccessible nor already tried this
                    # call, splice it to the front PROACTIVELY (without re-probing
                    # the walled lead). Otherwise the cross-call #82 skip would jump
                    # straight to the next provider even though e.g. codex@gpt-5.5
                    # has headroom while only @spark is walled.
                    alt_models = self._model_fallbacks.get(agent_cls) or []
                    tried = model_fallback_tried.setdefault(agent_cls, set())
                    if skip_resolved_model:
                        tried.add(skip_resolved_model)
                    next_model = next(
                        (m for m in alt_models
                         if m not in tried
                         and not is_rate_limited(skip_provider_key, m)
                         and not is_model_known_inaccessible(skip_provider_key, m)),
                        None,
                    )
                    if next_model is not None:
                        tried.add(next_model)
                        logger.warning(
                            "[Fallback] %s lead model rate-limited (cached) — trying "
                            "same-provider model '%s' before switching providers",
                            skip_provider_key, next_model,
                        )
                        self._emit_orchestration(
                            phase="fallback",
                            action="model_fallback",
                            from_worker=skip_provider_key,
                            to_worker=f"{skip_provider_key}({next_model})",
                            reason="usage",
                            reason_category="usage",
                            attempt=attempts,
                            attempt_total=len(self._chain),
                        )
                        pending.insert(0, (agent_cls, next_model))
                        respliced += 1
                        continue
                    # No viable same-provider alternate — skip the whole provider.
                    # SAFETY: never skip down to ZERO attempts; if every remaining
                    # candidate is cache-walled AND nothing has run this call, attempt
                    # anyway (a run that probes nothing is a worse regression).
                    any_viable_left = any(
                        not is_rate_limited(
                            c.__name__,
                            (m or self._configs.get(c, {}).get("model", self.model)),
                        )
                        for (c, m) in pending
                    )
                    if any_viable_left or ran_any:
                        skip_label = skip_provider_key
                        if model_override:
                            skip_label = f"{skip_label}({model_override})"
                        logger.warning(
                            "[Fallback] %s rate-limited until reset — skipping "
                            "(cached resets_at)", skip_label,
                        )
                        self._emit_orchestration(
                            phase="fallback",
                            action="rate_limit_skip",
                            from_worker=skip_label,
                            reason="usage",
                            reason_category="usage",
                            attempt=attempts,
                            attempt_total=len(self._chain),
                        )
                        continue
                    # else: nothing viable left AND nothing ran -> fall through and
                    # attempt this one anyway rather than exhaust to zero attempts.
                attempts += 1
                label = agent_cls.__name__
                if model_override:
                    label = f"{label}({model_override})"
                try:
                    sub = self._make_sub(agent_cls, model_override)
                except Exception as exc:  # construction failure -> skip provider
                    logger.warning("[Fallback] could not build %s: %s", label, exc)
                    last_error = exc
                    continue

                logger.info("[Fallback] provider %d/%d: %s", attempts, total, label)
                try:
                    ran_any = True
                    result = await sub.run_async(piped_input)
                except Exception as exc:
                    stderr = getattr(sub, "stderr", "") or ""
                    # is_usage_wall already excludes context overflow, so the two
                    # flags are mutually exclusive and overflow never reads as usage.
                    looked_like_overflow = is_context_overflow(stderr)
                    looked_like_usage = is_usage_wall(stderr)
                    looked_like_wedged = is_wedged_session(stderr)
                    # The requested model is inaccessible to this account (codex
                    # reports it only in its --json stdout, folded into stderr by
                    # CodexAgent). Prune it dynamically and try a sibling model.
                    looked_like_model_unavailable = is_model_unavailable(stderr)
                    reason = _watchdog_reason_in(stderr)
                    # A transport blip (plain network reset, or the watchdog's
                    # transport-stall trip) failed the CONNECTION, not the provider.
                    # Prefer a BOUNDED same-provider backoff-retry before advancing
                    # to the next provider (issue #57): re-splice this same provider
                    # to the front of pending, sleep a short backoff, and retry.
                    looked_like_transport = (
                        reason == WATCHDOG_TRANSPORT_STALL or is_transport_error(stderr)
                    )
                    # A watchdog TRANSPORT-STALL means the provider emitted noise but
                    # made NO forward progress for the entire stall window
                    # (AGY_TRANSPORT_MAX_SECONDS, ~300s) — it is wedged at the
                    # transport layer, not a momentary blip. Retrying it on the SAME
                    # provider re-incurs the full stall window every time with no
                    # aggregate wall-clock cap (up to ~3x300s = 15min stuck on one
                    # dead link). So treat a transport-stall like a wedged session:
                    # do NOT take the bounded same-provider retry path; cycle straight
                    # to the next provider. A plain transient transport error (a real
                    # connection reset that fails FAST) still earns its bounded
                    # same-provider backoff-retries (#57) below.
                    looked_like_transport_stall = (reason == WATCHDOG_TRANSPORT_STALL)
                    # A wedged session is futile to retry on the same provider (#62):
                    # never take the same-provider transport-retry path for it, even
                    # if earlier transport noise also landed in the accumulated stderr.
                    # A transport-STALL is excluded for the same "futile, and each
                    # retry burns a full stall window" reason (cap the cumulative
                    # same-provider transport time by never stacking stall windows).
                    if (looked_like_transport and not looked_like_wedged
                            and not looked_like_transport_stall
                            and transport_retries_used.get(agent_cls, 0) < max_transport_retries):
                        n = transport_retries_used.get(agent_cls, 0) + 1
                        transport_retries_used[agent_cls] = n
                        backoff = _transport_backoff_seconds(n)
                        logger.warning(
                            "[Fallback] %s transport blip (%s) — same-provider retry "
                            "%d/%d after %.1fs backoff",
                            label, reason or "network", n, max_transport_retries, backoff,
                        )
                        self._emit_orchestration(
                            phase="fallback",
                            action="transport_retry",
                            from_worker=label,
                            to_worker=label,
                            reason=reason or "transport",
                            reason_category="transport",
                            attempt=attempts,
                            attempt_total=len(self._chain),
                        )
                        # Retry the SAME provider (same model) next; it does not
                        # consume a chain slot conceptually, but counts toward total
                        # attempts above. Mark it as re-spliced so the next pop doesn't
                        # advance the cycle (and thus doesn't reset the per-cycle
                        # transport budget).
                        pending.insert(0, (agent_cls, model_override))
                        respliced += 1
                        if backoff > 0:
                            await asyncio.sleep(backoff)
                        last_error = exc
                        continue
                    # This provider's CURRENT model can't serve right now — either a
                    # usage/quota wall (spark window exhausted) OR the model is
                    # inaccessible to this account. Before switching providers, try
                    # the provider's OTHER models (operator request / #78): codex's
                    # spark window can be exhausted while gpt-5.5/gpt-5.4 still have
                    # headroom (separate windows), and an inaccessible model has a
                    # sibling that works. No backoff here: a different model is a
                    # different window. DYNAMIC DETECTION: a model that fails
                    # inaccessible is recorded process-wide and never offered again,
                    # so the valid set self-discovers per account instead of being
                    # hardcoded. Each alternate is used at most once per run.
                    provider_key = agent_cls.__name__
                    current_model = model_override or self._configs.get(
                        agent_cls, {}).get("model", self.model)
                    # Issue #82: a usage wall that reported a reset time gets cached so
                    # the NEXT call skips THIS specific (provider, model) until the
                    # window clears, instead of re-probing + re-walling it every call.
                    # Recorded REGARDLESS of whether a sibling model exists — it caches
                    # the exact walled (provider, model). A wall with no/malformed
                    # resets_at is silently not cached -> still re-probed as before.
                    if looked_like_usage:
                        record_rate_limited(
                            provider_key, current_model,
                            getattr(sub, "usage_wall_resets_at", None),
                        )
                    if looked_like_model_unavailable:
                        record_inaccessible_model(provider_key, current_model)
                        logger.warning(
                            "[Fallback] %s model unavailable on this account — pruned "
                            "'%s' for the run", label, current_model,
                        )
                    alt_models = self._model_fallbacks.get(agent_cls) or []
                    if (looked_like_usage or looked_like_model_unavailable) and alt_models:
                        tried = model_fallback_tried.setdefault(agent_cls, set())
                        # Never re-pick the model that just failed.
                        if current_model:
                            tried.add(current_model)
                        next_model = next(
                            (m for m in alt_models
                             if m not in tried
                             and not is_model_known_inaccessible(provider_key, m)),
                            None,
                        )
                        if next_model is not None:
                            tried.add(next_model)
                            why = "usage" if looked_like_usage else "model_unavailable"
                            logger.warning(
                                "[Fallback] %s %s — trying same-provider model '%s' "
                                "before switching providers", label, why, next_model,
                            )
                            self._emit_orchestration(
                                phase="fallback",
                                action="model_fallback",
                                from_worker=label,
                                to_worker=f"{agent_cls.__name__}({next_model})",
                                reason=why,
                                reason_category="usage" if looked_like_usage else "error",
                                attempt=attempts,
                                attempt_total=len(self._chain),
                            )
                            # Re-splice the SAME provider with the new model to the
                            # front; mark as re-spliced so it doesn't advance the cycle.
                            pending.insert(0, (agent_cls, next_model))
                            respliced += 1
                            last_error = exc
                            continue
                    # Issue #83 — LOUD, DISTINCT signal when the operator's chosen
                    # PRIMARY/lead generator (attempts == 1) is killed by the verbose
                    # BYTE-budget watchdog (NOT a hang or usage wall) and the chain is
                    # about to advance off it. The generic per-reroute line below is too
                    # easy to miss; this names the primary-override + the byte-budget
                    # remedy so a read-heavy generator silently demoted to a fallback is
                    # visible. Scoped to lead+verbose: non-lead or non-verbose unchanged.
                    if attempts == 1 and reason == WATCHDOG_VERBOSE:
                        logger.warning(
                            "[Fallback] PRIMARY generator %s was SIGKILLed by the verbose "
                            "BYTE-budget watchdog (watchdog:verbose) and the chain is being "
                            "rerouted to a FALLBACK worker — your chosen primary generator "
                            "selection was OVERRIDDEN by a byte-budget kill, not a hang or "
                            "usage wall. If %s was legitimately reading a design doc / "
                            "multiple modules, raise the byte budget with "
                            "--watchdog-max-bytes (or env AGY_WATCHDOG_MAX_BYTES) to keep it "
                            "on your primary (issue #83).",
                            label, label,
                        )
                    if reason and reason in self._watchdog_rules and reroute_budget > 0:
                        targets = self._watchdog_rules[reason]
                        # Splice rule-based re-route targets to the FRONT of pending,
                        # then continue. The targets pre-empt the normal sequence once;
                        # the normal sequence still runs afterwards if they too fail.
                        # Spend from the global re-route budget so a self-/mutually-
                        # referencing rules table can't re-splice forever (it drains
                        # to the exhaustion RuntimeError instead of hanging).
                        reroute_budget -= len(targets)
                        pending[0:0] = [(t, None) for t in targets]
                        logger.warning(
                            "[Fallback] %s tripped watchdog:%s — re-routing to %s",
                            label, reason, [c.__name__ for c in targets],
                        )
                        to_worker = targets[0].__name__ if targets else None
                        self._emit_orchestration(
                            phase="fallback",
                            action="reroute",
                            from_worker=label,
                            to_worker=to_worker,
                            reason=reason,
                            reason_category=_reason_category(
                                looked_like_usage=looked_like_usage,
                                watchdog_reason=reason,
                                context_overflow=looked_like_overflow,
                                transport=looked_like_transport,
                                wedged=looked_like_wedged,
                            ),
                            attempt=attempts,
                            attempt_total=len(self._chain),
                        )
                    else:
                        logger.warning(
                            "[Fallback] %s failed%s%s%s: %s",
                            label,
                            " (context overflow)" if looked_like_overflow else "",
                            " (usage/quota wall)" if looked_like_usage else "",
                            f" (watchdog:{reason})" if reason else "",
                            exc,
                        )
                        to_worker = pending[0][0].__name__ if pending else None
                        self._emit_orchestration(
                            phase="fallback",
                            action="reroute",
                            from_worker=label,
                            to_worker=to_worker,
                            reason=reason or ("transport" if looked_like_transport else "error"),
                            reason_category=_reason_category(
                                looked_like_usage=looked_like_usage,
                                watchdog_reason=reason,
                                context_overflow=looked_like_overflow,
                                transport=looked_like_transport,
                                wedged=looked_like_wedged,
                            ),
                            attempt=attempts,
                            attempt_total=len(self._chain),
                        )
                    # A usage/quota wall means this provider is rate-limited NOW. The
                    # cycles>1 design exists so its quota can RECOVER before the cycle
                    # returns to it — but that premise only holds if the loop actually
                    # WAITS between attempts instead of spinning the whole
                    # cycles*providers budget in the seconds it takes the CLIs to emit
                    # their quota errors. Back off before the next attempt (when one
                    # remains) so a real rate-limit window has a chance to clear
                    # (usage-wall cascade). Opt-out with AGY_USAGE_WALL_BACKOFF=0.
                    if looked_like_usage and usage_wall_backoff > 0 and pending:
                        await asyncio.sleep(usage_wall_backoff)
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
                # Record which provider actually produced the accepted output, so a
                # caller (e.g. the adversarial loop) can tell whether a rotation
                # actually changed the generator (#65).
                self.last_provider = agent_cls.__name__
                if attempts > 1:
                    logger.info("[Fallback] recovered via %s", label)
                return result

            raise RuntimeError(
                f"All {attempts} fallback attempts exhausted "
                f"({len(self._chain)} providers x {self._cycles} cycles, "
                f"plus rule-based re-routes); last error: {last_error}"
            )

    FallbackAgent.__name__ = "FallbackAgent[" + "->".join(c.__name__ for c in chain) + "]"
    return FallbackAgent

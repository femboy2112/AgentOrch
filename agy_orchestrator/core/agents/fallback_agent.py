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
import os
from typing import Callable, Dict, List, Optional, Type

from agy_orchestrator.core.agent import (
    WATCHDOG_MARKER,
    WATCHDOG_TRANSPORT_STALL,
    AgentInstance,
    is_context_overflow,
    is_transport_error,
    is_usage_wall,
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


def _watchdog_reason_in(stderr: str) -> Optional[str]:
    """Extract a watchdog trip reason from a failed sub's stderr, or None."""
    if not stderr or WATCHDOG_MARKER not in stderr:
        return None
    # Markers look like '[watchdog:verbose]' or '[watchdog:stalled]'. Pull the slug.
    head = stderr.split(WATCHDOG_MARKER, 1)[1]
    return head.split("]", 1)[0].strip() or None


def _reason_category(*, looked_like_usage: bool, watchdog_reason: Optional[str],
                     context_overflow: bool = False, transport: bool = False) -> str:
    # Context overflow is checked FIRST and is distinct from a quota wall: the
    # same provider can serve the next (smaller) step, so telemetry must not fold
    # it into "usage" (issue #47).
    if context_overflow:
        return "context_overflow"
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


def make_fallback_agent(
    chain: List[Type[AgentInstance]],
    cycles: int = 3,
    configs: Optional[Dict[Type[AgentInstance], Dict[str, object]]] = None,
    watchdog_rules: Optional[Dict[str, List[Type[AgentInstance]]]] = None,
    post_construct_hook: Optional[PostConstructHook] = None,
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
        _post_construct_hook: Optional[PostConstructHook] = post_construct_hook

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
            sub = agent_cls(**kwargs)  # type: ignore[arg-type]
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
            pending: List[Type[AgentInstance]] = list(self._chain) * self._cycles
            total = len(pending)
            attempts = 0
            # Per-provider count of bounded same-provider transport retries already
            # spent, so a flapping link can't loop a single provider forever (#57).
            transport_retries_used: Dict[Type[AgentInstance], int] = {}
            max_transport_retries = _transport_retries()
            while pending:
                agent_cls = pending.pop(0)
                attempts += 1
                label = agent_cls.__name__
                try:
                    sub = self._make_sub(agent_cls)
                except Exception as exc:  # construction failure -> skip provider
                    logger.warning("[Fallback] could not build %s: %s", label, exc)
                    last_error = exc
                    continue

                logger.info("[Fallback] provider %d/%d: %s", attempts, total, label)
                try:
                    result = await sub.run_async(piped_input)
                except Exception as exc:
                    stderr = getattr(sub, "stderr", "") or ""
                    # is_usage_wall already excludes context overflow, so the two
                    # flags are mutually exclusive and overflow never reads as usage.
                    looked_like_overflow = is_context_overflow(stderr)
                    looked_like_usage = is_usage_wall(stderr)
                    reason = _watchdog_reason_in(stderr)
                    # A transport blip (plain network reset, or the watchdog's
                    # transport-stall trip) failed the CONNECTION, not the provider.
                    # Prefer a BOUNDED same-provider backoff-retry before advancing
                    # to the next provider (issue #57): re-splice this same provider
                    # to the front of pending, sleep a short backoff, and retry.
                    looked_like_transport = (
                        reason == WATCHDOG_TRANSPORT_STALL or is_transport_error(stderr)
                    )
                    if (looked_like_transport
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
                        # Retry the SAME provider next; it does not consume a chain
                        # slot conceptually, but counts toward total attempts above.
                        pending.insert(0, agent_cls)
                        if backoff > 0:
                            await asyncio.sleep(backoff)
                        last_error = exc
                        continue
                    if reason and reason in self._watchdog_rules:
                        targets = self._watchdog_rules[reason]
                        # Splice rule-based re-route targets to the FRONT of pending,
                        # then continue. The targets pre-empt the normal sequence once;
                        # the normal sequence still runs afterwards if they too fail.
                        pending[0:0] = list(targets)
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
                        to_worker = pending[0].__name__ if pending else None
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
                            ),
                            attempt=attempts,
                            attempt_total=len(self._chain),
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

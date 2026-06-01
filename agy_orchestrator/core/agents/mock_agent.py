"""Hermetic mock worker for the single-run speed benchmark (scripts/speed_bench.py).

Activated by ``AGY_BENCH_MOCK=1`` via the single factory indirection in
``harness/roles._class_for``: every real worker class (codex/agy/claude/grok) is
swapped for a ``MockAgent`` bound to that worker name. A dispatch then runs
end-to-end — prompt construction, the pre-run baseline verifier, the streaming
drain + watchdog, usage-event emission, meta.json — with ZERO network and ZERO
contention on the real codex/agy/grok pool, so a benchmark can never touch the
operator's live runs.

Design choices that keep the measurement faithful:

* ``build_command`` spawns a REAL short-lived subprocess (``sh -c 'sleep T;
  printf ...'``) rather than returning in-process. This exercises the genuine
  ``_stream_communicate`` gather + watchdog timing, so the watchdog stream-tail
  finding (agent.py:366) is measurable — a pure in-memory mock would falsely
  report a zero tail.
* ``_extract_usage`` synthesises token counts from the ACTUAL prompt the agent
  was handed, so summed ``input_tokens`` across a run is a noise-free proxy for
  prompt bloat (the spec-reinflation / context-projection findings) independent
  of latency jitter.
* The per-call latency is deterministic and optionally scales with prompt size,
  so harness overhead is recoverable as ``wall_clock - sum(mock sleeps)``.

Configuration (env, read per call so a bench can vary it per arm):

* ``AGY_BENCH_MOCK_SLEEP``           base model-latency seconds/call (default 0.30)
* ``AGY_BENCH_MOCK_SLEEP_PER_KCHAR`` extra seconds per 1000 prompt chars (default 0)
* ``AGY_BENCH_MOCK_CACHE_RATIO``     synthetic cache_read fraction 0..0.95 (default 0)
* ``AGY_BENCH_MOCK_OUTPUT``          override canned worker reply (default: a
                                     file-list reply whose last line is APPROVED,
                                     so adversarial converges on iteration 1)
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from agy_orchestrator.core.agent import AgentInstance


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


class MockAgent(AgentInstance):
    """A worker stand-in that sleeps then prints a canned reply via a real child."""

    # Overridden by the per-worker subclasses minted in ``mock_agent_class_for`` so
    # usage events / ledger attribution carry the simulated provider name.
    _MOCK_WORKER: str = "mock"

    @classmethod
    async def get_available_models(cls) -> List[str]:
        return ["mock-model"]

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        # Always plenty of headroom so the UsageAwareAllocator never downgrades —
        # the bench controls effort/model explicitly.
        return 100.0

    def _worker_name(self) -> str:
        return getattr(self, "_MOCK_WORKER", "mock") or "mock"

    def _mock_sleep_seconds(self) -> float:
        base = _envf("AGY_BENCH_MOCK_SLEEP", 0.30)
        per_kchar = _envf("AGY_BENCH_MOCK_SLEEP_PER_KCHAR", 0.0)
        chars = len(self.prompt or "")
        return max(0.0, base + per_kchar * (chars / 1000.0))

    def _mock_output(self) -> str:
        custom = os.environ.get("AGY_BENCH_MOCK_OUTPUT")
        if custom:
            return custom
        prompt = self.prompt or ""
        # Master's planner asks for a JSON list of step prompts — give it one so
        # master mode is also runnable under the bench.
        if "JSON list" in prompt or "valid JSON list of strings" in prompt:
            return '["Implement the core module.", "Add a focused test for it."]'
        # Generic worker reply: non-empty, ends in a file-list line per
        # WORKER_PREAMBLE, last line is exactly APPROVED so a critic instance
        # converges the adversarial loop on the first iteration.
        return (
            "Implemented the requested change (mock mode).\n"
            "- mock_change.py: applied the canned benchmark edit\n"
            "APPROVED"
        )

    def build_command(self, piped_input: Optional[str] = None) -> List[str]:
        sleep_s = self._mock_sleep_seconds()
        # Deliver the canned reply through the child env to avoid any shell
        # quoting hazard on multi-line output.
        self.extra_env = {**(self.extra_env or {}), "AGY_MOCK_OUT": self._mock_output()}
        return ["sh", "-c", f'sleep {sleep_s:.3f}; printf "%s" "$AGY_MOCK_OUT"']

    def _extract_usage(self, raw_stdout: str, raw_stderr: str) -> Dict[str, object]:
        chars = len(self.prompt or "")
        raw_in = max(1, chars // 4)  # ~4 chars/token
        ratio = min(max(_envf("AGY_BENCH_MOCK_CACHE_RATIO", 0.0), 0.0), 0.95)
        cache_read = int(raw_in * ratio)
        input_tokens = max(0, raw_in - cache_read)  # cache-EXCLUSIVE input
        output_tokens = max(1, len(raw_stdout) // 4)
        return {
            "token_source": "cli",
            "input_tokens": input_tokens,
            "cache_read_tokens": cache_read,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + cache_read + output_tokens,
        }


_CLASS_CACHE: Dict[str, type] = {}


def mock_agent_class_for(name: str) -> type:
    """Return a STABLE ``MockAgent`` subclass bound to worker ``name``.

    Stability matters: ``roles.build_role_agent`` keys ``configs_by_cls`` and
    ``name_by_cls`` by the class object, so the class used to instantiate must be
    identical to the one used as the map key.
    """
    key = name or "mock"
    cls = _CLASS_CACHE.get(key)
    if cls is None:
        cls = type(f"Mock_{key}_Agent", (MockAgent,), {"_MOCK_WORKER": key})
        _CLASS_CACHE[key] = cls
    return cls

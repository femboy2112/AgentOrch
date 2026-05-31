"""Regression for #30: the streaming drain must not crash on a worker line
longer than the 64 KiB asyncio StreamReader default. The old _drain used
reader.readline(), which raised LimitOverrunError ("Separator is not found, and
chunk exceed the limit") on any >64 KiB newline-less line (e.g. claude emitting
one long JSON payload) — failing the attempt deterministically on every retry.

The chunked-read drain is immune to line length while still preserving the
one-JSON-object-per-line framing the stream-json adapters require.
"""
from __future__ import annotations

import asyncio
import json
import sys

from agy_orchestrator.core.agent import AgentInstance


class _BaseFake(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0


class _BigLineAgent(_BaseFake):
    def build_command(self, piped_input=None):
        # 200 KiB on stdout as a single line with NO newline — well past the
        # 64 KiB StreamReader default that used to overflow readline().
        return [sys.executable, "-c", "import sys; sys.stdout.write('x' * 200000)"]


def test_drain_handles_oversized_single_line():
    agent = _BigLineAgent(prompt="x")
    agent.event_callback = lambda e: None   # force stream_mode (the drain path)
    agent.max_output_bytes = 0
    agent.stall_seconds = 0
    agent.timeout = 30
    agent.max_retries = 1

    out = asyncio.run(agent.run_async())    # must NOT raise

    assert agent.returncode == 0
    assert agent._watchdog_reason is None
    # Full output reconstructed byte-exact (no truncation, no loss).
    assert len(out) == 200000
    assert set(out) == {"x"}


class _FramedAgent(_BaseFake):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.captured: list[str] = []

    def build_command(self, piped_input=None):
        # Two newline-terminated JSON objects; the first is padded to ~70 KiB so
        # the pair spans a 64 KiB read boundary. Framing must survive chunking.
        return [
            sys.executable,
            "-c",
            "import sys; "
            "big='{\"a\":\"' + 'p' * 70000 + '\"}'; "
            "sys.stdout.write(big + '\\n' + '{\"b\":1}' + '\\n')",
        ]

    def _events_from_stdout_line(self, text):
        if text.strip():
            self.captured.append(text)
        return []


def test_drain_preserves_jsonl_framing_across_chunk_boundary():
    agent = _FramedAgent(prompt="x")
    agent.event_callback = lambda e: None   # force stream_mode
    agent.max_output_bytes = 0
    agent.stall_seconds = 0
    agent.timeout = 30
    agent.max_retries = 1

    asyncio.run(agent.run_async())

    assert agent.returncode == 0
    # Exactly two complete lines, each one intact JSON object — not split or
    # merged by the 64 KiB chunk boundary.
    assert len(agent.captured) == 2
    assert json.loads(agent.captured[0]) == {"a": "p" * 70000}
    assert json.loads(agent.captured[1]) == {"b": 1}

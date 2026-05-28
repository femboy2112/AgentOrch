"""Two-tier compaction in MasterWorkflow.

The old single-digest scheme paragraph-prose'd every step's concrete details
(file paths, symbol names, IDs) into the digest, so the next critic's prompt
lost immediate visibility into the step it was meant to review. Two-tier
fix: digest the OLDER steps, keep the most-recent N step summaries VERBATIM.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.workflows.master import MasterWorkflow


class _StubAgent(AgentInstance):
    """Records the prompt seen on each run; returns a deterministic digest."""

    def __init__(self, *args, canned: str = "DIGEST", **kwargs):
        super().__init__(*args, **kwargs)
        self._canned = canned
        self.prompts_seen: List[str] = []

    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        self.prompts_seen.append(self.prompt)
        return self._canned


# --- splitter unit tests --- #

def _wf(**kwargs) -> MasterWorkflow:
    """MasterWorkflow with stub agent class; we never actually .execute()."""
    defaults = dict(model="m", effort="low", agent_class=_StubAgent)
    defaults.update(kwargs)
    return MasterWorkflow(**defaults)


def _project_context(n_steps: int) -> str:
    """Build a realistic-shaped project_context with n_steps."""
    out = "Original Goal: build a thing\n\n=== Accumulated Implementation ===\n"
    for i in range(1, n_steps + 1):
        out += f"\n--- Step {i} Summary ---\n"
        out += f"- Step {i}: touched file_{i}.py, defined symbol_{i}\n"
        out += f"- key decision: pick_strategy_{i}\n"
    return out


def test_splitter_finds_all_step_blocks():
    wf = _wf()
    preamble, blocks = wf._split_context_into_steps(_project_context(3))
    assert "Original Goal" in preamble
    assert len(blocks) == 3
    # Order preserved; each block carries its own header and content.
    assert "--- Step 1 Summary ---" in blocks[0]
    assert "file_1.py" in blocks[0]
    assert "--- Step 2 Summary ---" in blocks[1]
    assert "file_2.py" in blocks[1]
    assert "--- Step 3 Summary ---" in blocks[2]
    assert "file_3.py" in blocks[2]
    # Crucial: leaking content between blocks would mean a step's details
    # could end up attributed to the wrong step in the verbatim tier.
    assert "file_2" not in blocks[0]
    assert "file_3" not in blocks[1]


def test_splitter_roundtrips():
    """preamble + ''.join(blocks) must reconstruct the original exactly."""
    ctx = _project_context(4)
    wf = _wf()
    preamble, blocks = wf._split_context_into_steps(ctx)
    assert preamble + "".join(blocks) == ctx


def test_splitter_no_steps_yet():
    """Preamble-only string returns preamble + empty list. The compactor
    falls open to the old single-digest path in this case."""
    wf = _wf()
    preamble, blocks = wf._split_context_into_steps("Original Goal: x\n=== Accumulated ===\n")
    assert blocks == []
    assert "Original Goal" in preamble


# --- compaction end-to-end behaviour --- #

def _run(coro):
    return asyncio.run(coro)


def test_compaction_keeps_last_two_verbatim_by_default():
    """The new behaviour: with default recent_steps_verbatim=2 and 5 steps,
    the older 3 go into the digest, the last 2 are appended whole."""
    wf = _wf()
    ctx = _project_context(5)
    out = _run(wf._compact_context("build a thing", ctx))

    # Digest header present.
    assert "compacted" in out
    # Verbatim suffix present with both recent steps intact.
    assert "Recent steps (verbatim)" in out
    assert "--- Step 4 Summary ---" in out
    assert "--- Step 5 Summary ---" in out
    assert "file_4.py" in out
    assert "file_5.py" in out
    # Older steps were digested away — their literal headers should NOT
    # appear in the verbatim tier. The DIGEST text (canned "DIGEST") is
    # what the stub agent returned.
    assert "DIGEST" in out
    # Strict check: the verbatim section comes AFTER the digest.
    digest_pos = out.find("DIGEST")
    verbatim_pos = out.find("Recent steps (verbatim)")
    assert verbatim_pos > digest_pos


def test_compaction_with_fewer_steps_than_keep_does_not_split():
    """If we have N steps and want to keep K verbatim where K >= N,
    splitting would leave nothing for the digest. Fall through to the
    old single-digest scheme — the test asserts no verbatim suffix is
    appended in that case."""
    wf = _wf(recent_steps_verbatim=2)
    ctx = _project_context(2)   # 2 steps, keep last 2 = nothing to digest
    out = _run(wf._compact_context("build", ctx))

    assert "Recent steps (verbatim)" not in out
    # Still produced a digest (stub agent's canned output).
    assert "DIGEST" in out


def test_compaction_recent_steps_verbatim_zero_is_legacy_behaviour():
    """Setting recent_steps_verbatim=0 must reproduce the prior
    single-digest behaviour bit-identically — no verbatim suffix at all."""
    wf = _wf(recent_steps_verbatim=0)
    ctx = _project_context(5)
    out = _run(wf._compact_context("build", ctx))

    assert "Recent steps (verbatim)" not in out
    assert "--- Step 4 Summary ---" not in out
    assert "--- Step 5 Summary ---" not in out
    assert "DIGEST" in out


def test_compactor_only_sees_older_tier_with_keep_2():
    """The digest prompt sent to the compactor agent MUST NOT include the
    recent verbatim steps. Otherwise the digest duplicates them and the
    whole point of saving tokens is lost."""
    # Need the stub agent to capture its prompt — we'll fetch it via the
    # agent_class's last-constructed instance. The compactor inside
    # _compact_context constructs a NEW agent each time, so we patch
    # agent_class to a list-recorder.
    seen_prompts: List[str] = []

    class _RecordingAgent(_StubAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Capture once per construction so we have the digest's prompt
            # available after _compact_context returns.
            seen_prompts.append(self.prompt)

    wf2 = _wf(recent_steps_verbatim=2)
    wf2.agent_class = _RecordingAgent
    ctx = _project_context(5)
    _run(wf2._compact_context("build", ctx))
    assert len(seen_prompts) == 1
    compactor_prompt = seen_prompts[0]
    # The compactor's input should contain steps 1-3 (older) but NOT 4-5
    # (those are kept verbatim and bypassed the compactor entirely).
    assert "file_1.py" in compactor_prompt
    assert "file_2.py" in compactor_prompt
    assert "file_3.py" in compactor_prompt
    assert "file_4.py" not in compactor_prompt
    assert "file_5.py" not in compactor_prompt


def test_falls_back_on_compactor_failure():
    """If the compactor agent fails, _compact_context must not crash the
    workflow — falls back to the tail-truncation safety net."""
    class _BoomAgent(_StubAgent):
        async def run_async(self, piped_input=None):
            raise RuntimeError("compactor died")

    wf = _wf(max_context_chars=200, recent_steps_verbatim=2)
    wf.agent_class = _BoomAgent
    ctx = _project_context(5)
    out = _run(wf._compact_context("build", ctx))
    # Header present, tail content present, no exception.
    assert "Original Goal" in out
    assert "file_5.py" in out   # most-recent step lives in the tail truncation

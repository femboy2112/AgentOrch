"""Issue #90 — `harness spec` incremental persistence + partial-on-interrupt.

A long `--effort-profile max` spec run can hit a wrapping `timeout` (SIGTERM) or
Ctrl-C before the critic loop converges. Before this fix that discarded every
completed iteration — `spec.md`/`meta.json` were only written after
`wf.execute()` returned. Now each architect draft is flushed the moment it is
produced, and an interrupt still finalizes a populated `spec.md` + an honest
`partial=True` meta.json.
"""
from __future__ import annotations

import asyncio
import json
from typing import List, Optional

import pytest

import harness.spec as hs
from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.workflows.spec import SpecWorkflow


class _Stub(AgentInstance):
    """Returns queued replies in order; raises a queued exception if given one."""

    def __init__(self, *, replies: Optional[List[str]] = None, exc: Optional[BaseException] = None,
                 prompt: str = ""):
        super().__init__(prompt=prompt)
        self._replies = list(replies or [])
        self._exc = exc

    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input=None) -> str:
        if self._exc is not None:
            raise self._exc
        return self._replies.pop(0) if self._replies else ""


# --------------------------------------------------------------------------- #
# Workflow-level: draft_callback fires after EVERY architect turn
# --------------------------------------------------------------------------- #
def test_specworkflow_draft_callback_fires_per_turn():
    architect = _Stub(replies=["# D1", "# D2"])
    critic = _Stub(replies=["1. a gap", "APPROVED"])
    drafts: List[tuple] = []

    wf = SpecWorkflow(architect, critic, max_iterations=3,
                      draft_callback=lambda d, i: drafts.append((i, d)))
    doc = asyncio.run(wf.execute("goal", []))

    assert doc == "# D2"
    assert wf.approved is True
    # initial draft emitted at iterations_used=0, revised draft at iterations_used=1
    assert drafts[0] == (0, "# D1")
    assert drafts[1] == (1, "# D2")


def test_specworkflow_draft_callback_error_never_breaks_loop():
    architect = _Stub(replies=["# D1"])
    critic = _Stub(replies=["APPROVED"])

    def boom(_d, _i):
        raise RuntimeError("disk full")

    wf = SpecWorkflow(architect, critic, max_iterations=2, draft_callback=boom)
    doc = asyncio.run(wf.execute("goal", []))  # must not raise
    assert doc == "# D1"


# --------------------------------------------------------------------------- #
# Harness-level: a normal run still works (incremental + finalize)
# --------------------------------------------------------------------------- #
def _patch_build(monkeypatch, *, architect: _Stub, critic: _Stub):
    state = {"n": 0}

    def fake_build(chain, *, prompt="", fallback=True, cycles=2, codex_config=None,
                   post_construct_hook=None, **_kw):
        agent = architect if state["n"] == 0 else critic
        state["n"] += 1
        if post_construct_hook is not None:
            post_construct_hook(agent, chain[0], {"model": "m", "effort": "high"})
        return agent

    monkeypatch.setattr(hs.roles, "build_role_agent", fake_build)


def test_spec_normal_run_writes_spec_and_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "RUNS_DIR", tmp_path)
    _patch_build(monkeypatch, architect=_Stub(replies=["# FINAL DOC\nbody"]),
                 critic=_Stub(replies=["APPROVED"]))

    res = hs.generate_spec("g", run_id="ok", max_iterations=2)

    run_dir = tmp_path / "ok"
    assert (run_dir / "spec.md").read_text().startswith("# FINAL DOC")
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["success"] is True
    assert meta["partial"] is False
    assert res.partial is False


# --------------------------------------------------------------------------- #
# Harness-level: SIGTERM/cancel mid-run leaves a partial spec.md + meta.json
# --------------------------------------------------------------------------- #
def test_spec_interrupt_persists_partial_draft_and_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "RUNS_DIR", tmp_path)
    # architect authors a full draft (flushed by the callback); the critic review
    # is then cancelled as if by an external SIGTERM.
    _patch_build(monkeypatch, architect=_Stub(replies=["# DRAFT ONE\nbody"]),
                 critic=_Stub(exc=asyncio.CancelledError()))

    with pytest.raises(asyncio.CancelledError):
        hs.generate_spec("g", run_id="kill", max_iterations=3,
                         output_path=str(tmp_path / "out" / "DESIGN.md"))

    run_dir = tmp_path / "kill"
    # The most-recent complete draft survives on disk despite the cancellation.
    assert (run_dir / "spec.md").read_text().startswith("# DRAFT ONE")
    # The -o copy was flushed incrementally too.
    assert (tmp_path / "out" / "DESIGN.md").read_text().startswith("# DRAFT ONE")
    # meta.json is an honest partial record.
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["partial"] is True
    assert meta["success"] is False
    assert "interrupted" in (meta["error"] or "")
    assert meta["iterations_used"] == 1  # cancelled during the first critic review
    assert meta["chars"] > 0


def test_spec_interrupt_before_any_draft_writes_placeholder(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "RUNS_DIR", tmp_path)
    # The very first architect turn is cancelled — no draft was ever produced.
    _patch_build(monkeypatch, architect=_Stub(exc=asyncio.CancelledError()),
                 critic=_Stub(replies=["APPROVED"]))

    with pytest.raises(asyncio.CancelledError):
        hs.generate_spec("g", run_id="kill0", max_iterations=3)

    run_dir = tmp_path / "kill0"
    assert (run_dir / "spec.md").read_text() == "(no document produced)\n"
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["partial"] is True and meta["success"] is False
    assert meta["iterations_used"] == 0

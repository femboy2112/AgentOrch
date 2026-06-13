"""Issue #91 — `--plan-only` / the master planner must be provably read-only.

The planner sends codex a prompt that embeds the raw "Implement X" request; codex,
running with `--dangerously-bypass-approvals-and-sandbox` and its built-in
apply_patch/write_file tools, would dutifully start implementing and leak a
partial build into the out-dir during what the operator was told is a no-write
dry-run. The fix runs the planner under codex's `--sandbox read-only` (a hard
guarantee, not a prompt the model can ignore) and propagates that down through the
FallbackAgent wrapper.
"""
from __future__ import annotations

import asyncio

from agy_orchestrator.core.agents.codex_agent import CodexAgent
from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent
from agy_orchestrator.core.agents.grok_agent import GrokAgent
from agy_orchestrator.workflows.master import MasterWorkflow


# --------------------------------------------------------------------------- #
# The hard guarantee: codex's command flips to the read-only sandbox.
# --------------------------------------------------------------------------- #
def test_codex_read_only_uses_sandbox_not_bypass():
    a = CodexAgent(prompt="x", model="gpt-5.5")
    a.read_only = True
    cmd = a.build_command()
    assert "--sandbox" in cmd
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert "--dangerously-bypass-approvals-and-sandbox" not in cmd


def test_codex_default_keeps_write_enabled_bypass():
    # Byte-identical to the prior behavior when read_only is unset (the execute
    # path must still be able to edit accepted steps).
    a = CodexAgent(prompt="x", model="gpt-5.5")
    cmd = a.build_command()
    assert "--dangerously-bypass-approvals-and-sandbox" in cmd
    assert "--sandbox" not in cmd


# --------------------------------------------------------------------------- #
# Propagation: a read-only wrapper hands read-only to whichever provider runs.
# --------------------------------------------------------------------------- #
def test_fallback_propagates_read_only_to_codex_sub():
    fb_cls = make_fallback_agent([CodexAgent, GrokAgent])
    agent = fb_cls(prompt="x", model="gpt-5.5", effort="high")
    agent.read_only = True

    sub = agent._make_sub(CodexAgent)
    assert getattr(sub, "read_only", False) is True
    assert "--sandbox" in sub.build_command()

    # Without read_only the sub keeps the write-enabled bypass.
    agent2 = fb_cls(prompt="x", model="gpt-5.5", effort="high")
    sub2 = agent2._make_sub(CodexAgent)
    assert getattr(sub2, "read_only", False) is False
    assert "--dangerously-bypass-approvals-and-sandbox" in sub2.build_command()


# --------------------------------------------------------------------------- #
# Wiring: the master planner is constructed read-only with a no-write preamble.
# --------------------------------------------------------------------------- #
def _recording_planner(records, plan_json):
    class _Planner:
        def __init__(self, prompt=None, model=None, effort=None, **kwargs):
            records.append({"prompt": prompt, "model": model, "effort": effort,
                            "kwargs": kwargs})
            self.prompt = prompt
            self.session_id = "sess"

        async def run_async(self, piped_input=None):
            return plan_json

    return _Planner


def test_plan_only_planner_is_constructed_read_only(tmp_path):
    records: list = []
    wf = MasterWorkflow(
        model="m", effort="high",
        agent_class=_recording_planner(records, '["Step 1: A", "Step 2: B"]'),
        working_directory=str(tmp_path),
        checkpoint_path=str(tmp_path / "c.json"),
        plan_only=True,
    )
    asyncio.run(wf.execute("Implement the whole world"))

    assert len(records) == 1  # only the planner ran
    rec = records[0]
    assert rec["kwargs"].get("read_only") is True
    assert rec["kwargs"].get("role") == "plan"
    # Defense-in-depth preamble for providers we can't sandbox.
    assert "PLANNING-ONLY" in rec["prompt"]
    assert "do NOT create, write" in rec["prompt"]


def test_planner_read_only_in_normal_master_too(tmp_path):
    """The planner's 'writes NOTHING' contract holds in a normal master run too —
    read-only is not gated on plan_only (a leaking planner would write a partial
    build BEFORE the real steps run). Exercised directly via _run_planner."""
    records: list = []
    wf = MasterWorkflow(
        model="m", effort="high",
        agent_class=_recording_planner(records, '["only step"]'),
        working_directory=str(tmp_path),
        checkpoint_path=str(tmp_path / "c.json"),
        plan_only=False,
    )
    tasks, _sid = asyncio.run(wf._run_planner("build a thing"))
    assert tasks == ["only step"]
    assert records[0]["kwargs"].get("read_only") is True
    assert records[0]["kwargs"].get("role") == "plan"

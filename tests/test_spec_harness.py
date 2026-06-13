from __future__ import annotations

import json

import harness.spec as hs
from agy_orchestrator.core.agent import AgentInstance


class _Stub(AgentInstance):
    def __init__(self, reply: str, prompt: str = ""):
        super().__init__(prompt=prompt)
        self._reply = reply

    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input=None) -> str:
        return self._reply


def _patch_agents(monkeypatch):
    """architect (1st build) -> draft; critic (2nd build) -> APPROVED."""
    replies = ["# DESIGN DOC\nbody", "APPROVED"]
    state = {"n": 0}

    def fake_build(
        chain,
        *,
        prompt="",
        fallback=True,
        cycles=2,
        codex_config=None,
        post_construct_hook=None,
        overrides=None,
        watchdog_scale=1.0,
        watchdog_max_bytes=None,
        **_kw,
    ):
        agent = _Stub(replies[min(state["n"], len(replies) - 1)])
        state["n"] += 1
        if post_construct_hook is not None:
            post_construct_hook(agent, chain[0], {"model": "m", "effort": "high"})
        return agent

    monkeypatch.setattr(hs.roles, "build_role_agent", fake_build)


def test_generate_spec_is_dashboard_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "RUNS_DIR", tmp_path)
    _patch_agents(monkeypatch)

    res = hs.generate_spec(
        "a webhook fan-out service",
        constraints=["survive restarts"],
        run_id="testrun",
    )

    run_dir = tmp_path / "testrun"

    # Live tab keys off events.jsonl — it MUST exist with lifecycle markers.
    events_path = run_dir / "events.jsonl"
    assert events_path.exists()
    lines = [ln for ln in events_path.read_text().splitlines() if ln.strip()]
    blob = "\n".join(lines)
    assert "dispatch_started" in blob
    assert "dispatch_finished" in blob

    # prompt.txt drives the dashboard instruction preview.
    prompt = (run_dir / "prompt.txt").read_text()
    assert prompt.startswith("## Instruction")
    assert "a webhook fan-out service" in prompt

    # meta.json is dashboard-shaped (Runs tab reads mode/generator/critic/...).
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["mode"] == "spec"
    assert meta["generator"]  # architect chain desc, non-empty
    assert meta["critic"]
    assert meta["success"] is True
    assert meta["approved"] is True
    assert isinstance(meta.get("quality"), dict)

    # The deliverable + stdout mirror are present.
    assert (run_dir / "spec.md").read_text().startswith("# DESIGN DOC")
    assert (run_dir / "stdout.log").read_text().startswith("# DESIGN DOC")
    assert res.approved is True
    assert res.spec_path.endswith("spec.md")


def test_generate_spec_output_path_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "RUNS_DIR", tmp_path)
    _patch_agents(monkeypatch)

    dest = tmp_path / "target" / "DESIGN.md"
    hs.generate_spec("goal", run_id="r2", output_path=str(dest))

    assert dest.exists()
    assert dest.read_text().startswith("# DESIGN DOC")

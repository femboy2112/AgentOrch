from __future__ import annotations

from pathlib import Path

from agy_orchestrator.execution.verifier import VerifierResult
from harness import dispatch as dispatch_mod


class _StubWorkflow:
    verified = True
    approved = False
    stalled = False
    iterations_used = 1


class _CapturingVerifier:
    timeout = 5.0

    def __init__(self, test_commands=None, timeout=None):
        self.test_commands = test_commands or []

    async def verify(self, working_directory: str):
        return VerifierResult(
            ok=True,
            message="ok",
            returncode=0,
            cmd="stub",
            duration_ms=0,
        )


def test_dispatch_auto_resolves_to_concrete_mode_before_workflow(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    work_dir = tmp_path / "work"
    captured: dict[str, str] = {}

    monkeypatch.setattr(dispatch_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(dispatch_mod, "QualityVerifier", _CapturingVerifier)
    monkeypatch.setattr(dispatch_mod, "append_live_row", lambda **kwargs: None)

    async def _stub_run_workflow(mode, prompt, **kwargs):
        captured["mode"] = mode
        wd = Path(kwargs["working_directory"])
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "created.txt").write_text("x", encoding="utf-8")
        return "ok", _StubWorkflow()

    monkeypatch.setattr(dispatch_mod, "_run_workflow", _stub_run_workflow)

    result = dispatch_mod.dispatch(
        "small fix",
        mode="auto",
        test_cmd="pytest -q",
        generator_chain=["codex"],
        critic_chain=["agy"],
        out_dir=work_dir,
    )

    concrete_modes = {
        "direct", "adversarial", "feedback", "cascade", "master", "pat", "vote",
    }
    assert captured["mode"] in concrete_modes
    assert captured["mode"] != "auto"
    assert result.mode == captured["mode"]
    assert result.mode in concrete_modes

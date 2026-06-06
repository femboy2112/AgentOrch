"""Codex model/effort are surfaced to Telegram as the RESOLVED identity.

The codex config carries internal aliases ("standard" -> gpt-5.3-codex-spark,
"max" -> reasoning_effort=xhigh) that the command-builder rewrites before exec.
An observer that printed the raw alias showed the operator a model they didn't
pick and couldn't recognise ("codex spun up · standard · high"). These tests pin
that every Telegram surface now shows the model codex ACTUALLY runs:

  * the canonical worker spin-up line (via the agent's effective_* resolvers),
  * the adversarial ✍️ Draft line (now incl. effort), and
  * the live pinned status card (worker + model/effort, not just the role).
"""
from agy_orchestrator.core.agents.agy_agent import AgyAgent
from agy_orchestrator.core.agents.codex_agent import CodexAgent
from agy_orchestrator.core.agents.grok_agent import GrokAgent
from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent
from harness import telegram as tg


# --------------------------------------------------------------------------- #
# Resolver layer
# --------------------------------------------------------------------------- #
def test_codex_effective_model_resolves_standard_alias():
    assert CodexAgent(prompt="", model="standard").effective_model() == "gpt-5.3-codex-spark"
    # An unset model also lands on the ChatGPT-account default the CLI pins.
    assert CodexAgent(prompt="", model=None).effective_model() == "gpt-5.3-codex-spark"


def test_codex_effective_model_passthrough_real_name():
    assert CodexAgent(prompt="", model="gpt-5.5").effective_model() == "gpt-5.5"


def test_codex_effective_effort_resolves_max_to_xhigh():
    assert CodexAgent(prompt="", model="standard", effort="max").effective_effort() == "xhigh"
    assert CodexAgent(prompt="", model="standard", effort="high").effective_effort() == "high"


def test_resolution_mirrors_build_command():
    # The whole point: the resolver must agree with what the CLI is told.
    a = CodexAgent(prompt="x", model="standard", effort="max")
    cmd = a.build_command()
    assert a.effective_model() in cmd  # gpt-5.3-codex-spark passed via --model
    joined = " ".join(cmd)
    assert f'model_reasoning_effort="{a.effective_effort()}"' in joined  # xhigh


def test_non_aliasing_providers_pass_through():
    assert AgyAgent(prompt="", model="pro", effort="high").effective_model() == "pro"
    assert GrokAgent(prompt="", model="grok-build").effective_model() == "grok-build"


def test_fallback_agent_resolves_via_codex_lead():
    fb = make_fallback_agent([CodexAgent, AgyAgent])
    inst = fb(prompt="", model="standard", effort="max")
    assert inst.effective_model() == "gpt-5.3-codex-spark"
    assert inst.effective_effort() == "xhigh"


def test_fallback_agent_non_codex_lead_passthrough():
    fb = make_fallback_agent([AgyAgent, CodexAgent])
    inst = fb(prompt="", model="pro", effort="high")
    assert inst.effective_model() == "pro"
    assert inst.effective_effort() == "high"


# --------------------------------------------------------------------------- #
# Telegram spin-up line (top-level model/effort stamped by the bus)
# --------------------------------------------------------------------------- #
def _spinup(worker, model, effort, detail=None):
    return {
        "ts": 1.0, "run_id": "R", "worker": worker, "model": model,
        "effort": effort, "branch": None, "kind": "lifecycle",
        "data": {"event": "agent_started", "detail": detail if detail is not None else {}},
    }


def test_spinup_renders_resolved_codex_model():
    out = tg.render_event(_spinup("codex", "gpt-5.3-codex-spark", "high"), verbosity="normal")
    assert "gpt-5.3-codex-spark" in out and "high" in out
    assert "standard" not in out


# --------------------------------------------------------------------------- #
# Adversarial ✍️ Draft line now carries effort, not just model
# --------------------------------------------------------------------------- #
def _draft(model, effort):
    return {
        "ts": 1.0, "run_id": "R", "kind": "lifecycle",
        "data": {"event": "orchestration_transition", "orchestration": {
            "workflow": "master", "phase": "adversarial", "action": "iteration_started",
            "iteration": 1, "iteration_total": 5, "model": model, "effort": effort,
        }},
    }


def test_draft_line_includes_model_and_effort():
    out = tg.render_event(_draft("gpt-5.3-codex-spark", "high"), verbosity="normal")
    assert "Draft" in out
    assert "gpt-5.3-codex-spark" in out
    assert "high" in out  # effort was previously dropped


def test_draft_line_model_only_when_no_effort():
    out = tg.render_event(_draft("grok-build", None), verbosity="normal")
    assert "grok-build" in out


# --------------------------------------------------------------------------- #
# Live status card: worker + model/effort, not just the role
# --------------------------------------------------------------------------- #
def _state_after_spinup(worker, model, effort, role="draft"):
    st = tg.BuildState("RUN-123456", "master")
    st.run_start_ts = 0.0
    detail = {"role": role} if role else {}
    st.update(_spinup(worker, model, effort, detail=detail))
    return st


def test_buildstate_tracks_active_worker_model_effort():
    st = _state_after_spinup("codex", "gpt-5.3-codex-spark", "high")
    assert st.active_worker == "codex"
    assert st.active_model == "gpt-5.3-codex-spark"
    assert st.active_effort == "high"


def test_buildstate_ignores_per_turn_string_detail_noise():
    # A per-turn adapter event (str detail) must NOT churn the live identity.
    st = _state_after_spinup("codex", "gpt-5.3-codex-spark", "high")
    st.update(_spinup("codex", "gpt-5.3-codex-spark", "high", detail="turn.started"))
    assert st.active_worker == "codex"  # unchanged, no crash


def test_buildstate_skips_orchestrator_identity():
    # The orchestrator/run-monitor publishers carry n/a — never adopt them as the
    # "running" worker.
    st = tg.BuildState("RUN-123456", "master")
    st.run_start_ts = 0.0
    st.update(_spinup("orchestrator", "n/a", "n/a"))
    assert st.active_worker == ""


def test_status_card_shows_worker_and_resolved_model():
    st = _state_after_spinup("codex", "gpt-5.3-codex-spark", "high")
    card = tg.render_status_card(st)
    assert "codex" in card
    assert "gpt-5.3-codex-spark/high" in card
    assert "standard" not in card


def test_status_card_grok_has_no_trailing_slash():
    # grok has no effort — the model/effort chip must not render a dangling "/".
    st = _state_after_spinup("grok", "grok-build", "n/a", role="")
    card = tg.render_status_card(st)
    assert "grok-build" in card
    assert "grok-build/" not in card


def test_status_card_running_when_no_identity_yet():
    st = tg.BuildState("RUN-123456", "master")
    st.run_start_ts = 0.0
    card = tg.render_status_card(st)
    assert "running" in card

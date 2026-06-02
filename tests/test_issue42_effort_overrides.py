"""Issue #42 — per-role / per-provider effort + model overrides on ``do``.

These exercise the resolution layer (harness/effort_overrides.py), its threading
through roles.py (per-provider configs reach the lead AND fallback subs, watchdog
budgets scale), the vote host-safety concurrency cap, and the dispatch-level
contract that the resolved ``(model, effort)`` lands in meta.json.

All hermetic: no worker CLI is spawned. The dispatch integration test
monkeypatches ``_run_workflow`` to a trivial coroutine — ``resolved_config`` is
computed independently of the workflow, so it still reaches meta.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness import roles
from harness.effort_overrides import (
    EFFORT_PROFILES,
    OverrideError,
    ResolvedOverrides,
    effective_config,
    parse_kv_map,
    resolve_overrides,
)


# --- parsing --------------------------------------------------------------- #

def test_parse_scalar_and_map():
    assert parse_kv_map("max", flag="--effort") == {"*": "max"}
    assert parse_kv_map("codex=max,agy=high", flag="--effort") == {
        "codex": "max", "agy": "high"}
    assert parse_kv_map(None, flag="--effort") == {}
    assert parse_kv_map("  ", flag="--effort") == {}


def test_parse_malformed_raises():
    with pytest.raises(OverrideError):
        parse_kv_map("codex=max,oops", flag="--effort")
    with pytest.raises(OverrideError):
        # a comma'd value with no '=' is ambiguous, not a scalar
        parse_kv_map("max,high", flag="--effort")
    with pytest.raises(OverrideError):
        parse_kv_map("codex=", flag="--model")


# --- the headline acceptance scenario -------------------------------------- #

def test_acceptance_gen_critic_max_codex_gpt55():
    # do "..." --mode master --gen-effort max --critic-effort max --codex-model gpt-5.5
    r = resolve_overrides(
        generator_chain=["codex", "agy", "grok"],
        critic_chain=["agy", "codex", "grok"],
        mode="master",
        gen_effort="max", critic_effort="max", codex_model="gpt-5.5",
    )
    # codex runs gpt-5.5 + max (==xhigh) in BOTH roles, wherever it sits.
    assert r.generator["codex"] == {"effort": "max", "model": "gpt-5.5"}
    assert r.critic["codex"] == {"effort": "max", "model": "gpt-5.5"}
    # grok is left untouched (no effort key) — it no-ops reasoning effort.
    assert "grok" not in r.generator or "effort" not in r.generator.get("grok", {})
    assert any("grok" in n for n in r.notes)


def test_effective_config_for_meta():
    r = resolve_overrides(
        generator_chain=["codex", "agy", "grok"],
        critic_chain=["agy", "codex", "grok"],
        codex_model="gpt-5.5", gen_effort="max",
    )
    eff = effective_config(["codex", "agy", "grok"], r.generator, roles.AGENT_DEFAULTS)
    assert eff["codex"] == {"model": "gpt-5.5", "effort": "max"}
    # grok keeps its defaults (model grok-build, effort n/a).
    assert eff["grok"]["model"] == "grok-build"
    assert eff["grok"]["effort"] == "n/a"


# --- profiles + precedence ------------------------------------------------- #

def test_effort_profile_max():
    r = resolve_overrides(
        generator_chain=["codex", "agy", "grok"],
        critic_chain=["agy", "codex", "grok"],
        profile="max",
    )
    assert r.generator["codex"]["model"] == "gpt-5.5"
    assert r.generator["codex"]["effort"] == "max"
    assert r.generator["agy"]["effort"] == "high"
    assert "grok" not in r.generator  # profiles never touch grok


def test_balanced_profile_is_defaults():
    # 'balanced' mirrors AGENT_DEFAULTS, so the effective config equals defaults.
    r = resolve_overrides(
        generator_chain=["codex", "agy"], critic_chain=["agy", "codex"],
        profile="balanced",
    )
    eff = effective_config(["codex", "agy"], r.generator, roles.AGENT_DEFAULTS)
    assert eff["codex"]["effort"] == "high"
    assert eff["agy"]["model"] == "pro"


def test_explicit_flag_overrides_profile_with_note():
    r = resolve_overrides(
        generator_chain=["codex"], critic_chain=["codex"],
        profile="low", gen_effort="max",
    )
    assert r.generator["codex"]["effort"] == "max"  # explicit wins over profile 'low'
    assert any("profile" in n.lower() for n in r.notes)


def test_unknown_profile_raises():
    with pytest.raises(OverrideError):
        resolve_overrides(generator_chain=["codex"], critic_chain=["codex"],
                          profile="ludicrous")


# --- validation ------------------------------------------------------------ #

def test_invalid_effort_tier_raises():
    with pytest.raises(OverrideError):
        resolve_overrides(generator_chain=["codex"], critic_chain=["codex"],
                          gen_effort="turbo")


def test_unknown_codex_model_passthrough_with_note():
    # Issue #46: an unknown/renamed codex model no longer hard-fails — it warns
    # (advisory note) and passes through to the CLI unchanged. A CLI update or
    # account change can rename a model at any time; rejecting it here used to
    # break dispatch and masquerade as a usage wall.
    r = resolve_overrides(generator_chain=["codex"], critic_chain=["codex"],
                          codex_model="gpt-9-imaginary")
    assert r.generator["codex"]["model"] == "gpt-9-imaginary"
    assert r.critic["codex"]["model"] == "gpt-9-imaginary"
    assert any("gpt-9-imaginary" in n for n in r.notes)


def test_grok_effort_in_map_is_dropped_with_note():
    r = resolve_overrides(
        generator_chain=["grok", "codex"], critic_chain=["codex"],
        effort_map="grok=high,codex=max",
    )
    assert "grok" not in r.generator  # effort dropped for grok
    assert r.generator["codex"]["effort"] == "max"
    assert any("grok" in n for n in r.notes)


def test_watchdog_scale_validation():
    with pytest.raises(OverrideError):
        resolve_overrides(generator_chain=["codex"], critic_chain=["codex"],
                          watchdog_scale=0)
    r = resolve_overrides(generator_chain=["codex"], critic_chain=["codex"],
                          watchdog_scale=2.5)
    assert r.watchdog_scale == 2.5


def test_architect_alias_only_in_master():
    r_master = resolve_overrides(generator_chain=["codex"], critic_chain=["codex"],
                                 mode="master", architect_effort="max")
    assert r_master.generator["codex"]["effort"] == "max"
    r_direct = resolve_overrides(generator_chain=["codex"], critic_chain=["codex"],
                                 mode="direct", architect_effort="max")
    assert "codex" not in r_direct.generator  # ignored outside master/pat
    assert any("architect" in n.lower() for n in r_direct.notes)


# --- threading into roles.py ----------------------------------------------- #

def test_overrides_reach_single_provider_agent(monkeypatch):
    monkeypatch.delenv("AGY_WATCHDOG", raising=False)
    monkeypatch.delenv("AGY_WATCHDOG_SCALE", raising=False)
    agent = roles.build_role_agent(
        ["codex"], prompt="x", fallback=False,
        overrides={"codex": {"model": "gpt-5.5", "effort": "max"}},
    )
    assert agent.model == "gpt-5.5"
    assert agent.effort == "max"


def test_configs_for_merges_override_into_every_provider():
    cfgs = roles._configs_for(
        ["codex", "agy", "grok"], None,
        {"codex": {"model": "gpt-5.5", "effort": "max"}, "agy": {"effort": "high"}},
    )
    by_name = {cls.__name__: cfg for cls, cfg in cfgs.items()}
    assert by_name["CodexAgent"]["model"] == "gpt-5.5"
    assert by_name["CodexAgent"]["effort"] == "max"
    assert by_name["AgyAgent"]["effort"] == "high"
    # grok untouched -> keeps default grok-build / n/a
    assert by_name["GrokAgent"]["model"] == "grok-build"


def test_watchdog_scale_widens_budgets(monkeypatch):
    monkeypatch.delenv("AGY_WATCHDOG", raising=False)
    monkeypatch.delenv("AGY_WATCHDOG_SCALE", raising=False)
    roles._reset_calibration()
    base = roles.build_role_agent(["codex"], prompt="x", fallback=False)
    scaled = roles.build_role_agent(["codex"], prompt="x", fallback=False,
                                    watchdog_scale=3.0)
    assert base.max_output_bytes > 0
    assert scaled.max_output_bytes == base.max_output_bytes * 3
    assert scaled.stall_seconds == pytest.approx(base.stall_seconds * 3)


def test_watchdog_scale_env_var(monkeypatch):
    # Relative check (robust to whatever the live calibration ledger holds for
    # codex): env scale x2 doubles the armed byte budget vs no scale.
    monkeypatch.delenv("AGY_WATCHDOG", raising=False)
    monkeypatch.delenv("AGY_WATCHDOG_SCALE", raising=False)
    roles._reset_calibration()
    base = roles.build_role_agent(["codex"], prompt="x", fallback=False)
    monkeypatch.setenv("AGY_WATCHDOG_SCALE", "2")
    scaled = roles.build_role_agent(["codex"], prompt="x", fallback=False)
    assert scaled.max_output_bytes == base.max_output_bytes * 2


# --- vote host-safety cap (#42 item 7) ------------------------------------- #

def test_vote_max_parallel_param(monkeypatch):
    from agy_orchestrator.workflows.vote import VoteWorkflow

    class _Gen:
        def __init__(self):
            self.prompt = ""

    class _Verifier:
        pass

    monkeypatch.delenv("AGY_MAX_PARALLEL_WORKERS", raising=False)
    wf = VoteWorkflow(generators=[_Gen(), _Gen(), _Gen()], verifier=_Verifier(),
                      max_parallel=2)
    assert wf.max_parallel == 2
    # default (no cap) leaves it None -> all-K concurrent (legacy behaviour).
    wf2 = VoteWorkflow(generators=[_Gen()], verifier=_Verifier())
    assert wf2.max_parallel is None


def test_vote_max_parallel_env(monkeypatch):
    from agy_orchestrator.workflows.vote import VoteWorkflow

    class _Gen:
        def __init__(self):
            self.prompt = ""

    monkeypatch.setenv("AGY_MAX_PARALLEL_WORKERS", "4")
    wf = VoteWorkflow(generators=[_Gen()], verifier=object())
    assert wf.max_parallel == 4


# --- dispatch-level: resolved_config reaches meta.json --------------------- #

def test_dispatch_records_resolved_config(tmp_path, monkeypatch):
    from harness import dispatch as dispatch_mod

    async def _ok_workflow(*args, **kwargs):
        return ("done", None)

    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow)

    result = dispatch_mod.dispatch(
        "build something at the ceiling",
        mode="direct",
        out_dir=str(tmp_path),
        gen_effort="max",
        codex_model="gpt-5.5",
        heartbeat_interval=0,
    )
    assert result.success is True
    rc = result.resolved_config
    assert rc is not None
    assert rc["generator"]["codex"] == {"model": "gpt-5.5", "effort": "max"}

    meta = json.loads((Path(result.run_dir) / "meta.json").read_text())
    assert meta["resolved_config"]["generator"]["codex"]["model"] == "gpt-5.5"


def test_dispatch_invalid_override_is_resolved_error(tmp_path, monkeypatch):
    from harness import dispatch as dispatch_mod

    async def _ok_workflow(*args, **kwargs):
        return ("done", None)

    monkeypatch.setattr(dispatch_mod, "_run_workflow", _ok_workflow)
    # An invalid tier raises OverrideError out of dispatch (the CLI pre-validates
    # for a clean message; programmatic callers get the ValueError subclass).
    with pytest.raises(OverrideError):
        dispatch_mod.dispatch("x", mode="direct", out_dir=str(tmp_path),
                              gen_effort="turbo", heartbeat_interval=0)

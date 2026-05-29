"""SafetyKernel + AppLaunchPolicy tests (Step 5).

Release-blocking FRs first (FR-03/04/09/12/23/24) + FR-07 target resolution with stale rejection,
every reachable ViolationCode, destructive dry-run gate (require_confirmation + CONFIRMATION_REQUIRED),
budget enforcement, foreign-process via is_owned (FR-12), full AppLaunchPolicy schema (pattern + blocked + metachar),
and zero-spawn guarantee on any forbidden launch payload.

All actuation contracts are hermetic (isolated Xvfb only; never touches real :0).
SafetyKernel is verified as the single mandatory pre-action gate.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from agy_orchestrator.computer_use import DEFAULT_APP_LAUNCH_POLICY, SafetyKernel, get_default_app_launch_policy
from agy_orchestrator.computer_use.models import AppLaunchPolicy, ViolationCode
from agy_orchestrator.computer_use.process_supervisor import ProcessSupervisor, SpawnSpec

# Step 5 realgui release-blocking additions (imports only; no existing test bodies touched)
from agy_orchestrator.computer_use.grants import FakeClock, GrantCache
from agy_orchestrator.computer_use.gui_prompt import FakePrompter
from agy_orchestrator.computer_use.models import AskMode, RealGuiPolicy
from agy_orchestrator.computer_use.ownership import FakeOwnershipResolver
from agy_orchestrator.computer_use.safety import FakeOwnershipResolver as _FOR, FakePrompter as _FP  # via safety re-exports (Step 5 "export the new helpers")


# Step 13: release-blocking FR markers applied to the exact FR-03/04/09/12/23/24 tests


@pytest.fixture
def sup() -> ProcessSupervisor:
    s = ProcessSupervisor()
    try:
        yield s
    finally:
        for rid in list(s._registry.keys()):
            try:
                s.terminate_tree(rid)
            except Exception:
                pass


@pytest.fixture
def kernel(sup: ProcessSupervisor) -> SafetyKernel:
    return SafetyKernel(supervisor=sup)


def mk_intent(**kw: Any) -> Dict[str, Any]:
    action: Dict[str, Any] = {"type": kw.pop("type", "wait"), "display_scope": kw.pop("display_scope", "isolated")}
    for f in ("target", "text", "hotkey", "scroll_delta", "drag_to", "wait_ms", "url", "app", "app_args"):
        if f in kw: action[f] = kw.pop(f)
    return {"intent_id": kw.pop("intent_id", "i1"), "snapshot_id": "s1", "action": action,
            "rationale": kw.pop("rationale", "test"), "risk_level": kw.pop("risk_level", "low"),
            "requires_confirmation": False, "confirmation_token": kw.pop("confirmation_token", None)}


def mk_session(**o: Any) -> Dict[str, Any]:
    b = {"max_steps": 200, "max_actions": 200, "action_timeout_ms": 10000, "reasoning_timeout_ms": 45000,
         "confirmation_wait_timeout_ms": 300000, "max_cpu_percent": 200, "max_rss_mb": 2048, "max_processes": 64}
    b.update(o.pop("budgets", {}))
    s = {"run_id": "r1", "budgets": b, "displays": {"isolated_display": ":99"}}
    s.update(o)
    return s


@pytest.mark.not_slow
@pytest.mark.release_blocking
def test_fr04_rejects_non_isolated(kernel: SafetyKernel):
    i = mk_intent(type="click", display_scope="observe_real", target={"kind": "coordinate", "x": 1, "y": 2})
    r = kernel.validate(i, mk_session())
    assert not r.valid and any(v["code"] == ViolationCode.DISPLAY_SCOPE_INVALID.value for v in (r.violations or []))


@pytest.mark.not_slow
@pytest.mark.release_blocking
def test_fr04_missing_scope_rejected(kernel: SafetyKernel):
    i = {"action": {"type": "hotkey", "hotkey": ["CTRL", "L"]}}
    r = kernel.validate(i, mk_session())
    assert not r.valid and any(v["code"] == ViolationCode.DISPLAY_SCOPE_INVALID.value for v in (r.violations or []))


@pytest.mark.not_slow
@pytest.mark.release_blocking
def test_fr23_not_allowlisted_zero_spawn(kernel: SafetyKernel, sup: ProcessSupervisor):
    before = len(sup._registry)
    i = mk_intent(type="launch_app", app="evil", app_args=["-rf", "/"])
    r = kernel.validate(i, mk_session())
    assert not r.valid and any(v["code"] == ViolationCode.LAUNCH_APP_NOT_ALLOWLISTED.value for v in (r.violations or []))
    assert len(sup._registry) == before


@pytest.mark.not_slow
@pytest.mark.release_blocking
def test_fr23_metachar_rejected(kernel: SafetyKernel):
    i = mk_intent(type="launch_app", app="xclock", app_args=["; rm -rf /"])
    r = kernel.validate(i, mk_session())
    assert not r.valid and any(v["code"] == ViolationCode.LAUNCH_APP_ARGS_INVALID.value for v in (r.violations or []))


def test_fr23_allowlisted_passes(kernel: SafetyKernel):
    i = mk_intent(type="launch_app", app="true", app_args=[])
    r = kernel.validate(i, mk_session())
    assert r.valid and r.normalized_action is not None


@pytest.mark.not_slow
@pytest.mark.release_blocking
def test_fr24_spawn_env_override(kernel: SafetyKernel):
    spec = SpawnSpec(argv=["xclock"], display_scope="isolated", env={"DISPLAY": ":0"}, no_shell=True)
    r = kernel.validate_spawn_request(spec, ":99")
    assert not r.valid and any(v["code"] == ViolationCode.SPAWN_ENV_OVERRIDE_FORBIDDEN.value for v in (r.violations or []))


@pytest.mark.not_slow
@pytest.mark.release_blocking
def test_fr12_foreign_pid(kernel: SafetyKernel):
    i = mk_intent(type="click", target={"kind": "element", "handle_id": "h1"})
    snap = {"isolated": {"elements": [{"handle_id": "h1", "app_pid": 1, "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}}]}}
    r = kernel.validate(i, mk_session(), snapshots=snap)
    assert not r.valid and any(v["code"] == ViolationCode.FOREIGN_PROCESS.value for v in (r.violations or []))


@pytest.mark.not_slow
@pytest.mark.release_blocking
def test_fr09_high_risk_needs_token(kernel: SafetyKernel):
    # Use allowlisted app so the only violation is the confirmation gate (FR-09)
    i = mk_intent(type="launch_app", app="true", risk_level="high")
    r = kernel.validate(i, mk_session())
    assert not r.valid and any(v["code"] == ViolationCode.CONFIRMATION_REQUIRED.value for v in (r.violations or []))
    # Must not have leaked a normalized action for a still-gated high-risk intent
    assert r.normalized_action is None


@pytest.mark.not_slow
@pytest.mark.release_blocking
def test_fr09_gate_irreversible(kernel: SafetyKernel):
    g = kernel.require_confirmation(mk_intent(rationale="reboot the host", risk_level="low"))
    assert g.gate == "require_confirmation"


def test_fr09_token_allows(kernel: SafetyKernel):
    i = mk_intent(type="navigate", url="https://ex.com", risk_level="high", confirmation_token="tok-12345678")
    r = kernel.validate(i, mk_session())
    conf = [v for v in (r.violations or []) if v.get("code") == ViolationCode.CONFIRMATION_REQUIRED.value]
    assert not conf


def test_fr07_element_to_coord(kernel: SafetyKernel):
    i = mk_intent(type="click", target={"kind": "element", "handle_id": "h42"})
    snap = {"isolated": {"elements": [{"handle_id": "h42", "bbox": {"x": 100, "y": 200, "w": 80, "h": 30}}]}}
    ct = kernel.resolve_target(i, snap)
    assert ct.x == 140 and ct.y == 215


def test_fr07_stale_unresolvable(kernel: SafetyKernel):
    i = mk_intent(type="click", target={"kind": "element", "handle_id": "ghost"})
    snap = {"isolated": {"elements": []}}
    r = kernel.validate(i, mk_session(), snapshots=snap)
    assert not r.valid and any(v["code"] == ViolationCode.TARGET_UNRESOLVABLE.value for v in (r.violations or []))


def test_budget_exceeded(kernel: SafetyKernel):
    r = kernel.validate(mk_intent(type="wait", wait_ms=1), mk_session(budgets={"max_actions": 0}), current_actions=1)
    assert not r.valid and any(v["code"] == ViolationCode.BUDGET_EXCEEDED.value for v in (r.violations or []))


def test_schema_invalid(kernel: SafetyKernel):
    r = kernel.validate(mk_intent(type="wait"), mk_session())
    assert not r.valid and any(v["code"] == ViolationCode.SCHEMA_INVALID.value for v in (r.violations or []))


def test_target_missing_spatial(kernel: SafetyKernel):
    r = kernel.validate(mk_intent(type="click"), mk_session())
    assert not r.valid and any(v["code"] == ViolationCode.TARGET_MISSING.value for v in (r.violations or []))


def test_default_policy_minimal_and_extension():
    p = get_default_app_launch_policy()
    assert "xclock" in p.allowed_apps or "true" in p.allowed_apps
    k = SafetyKernel(app_policy=AppLaunchPolicy(allowed_apps={"foo": {"exec_path": "foo"}}))
    assert "foo" in k.policy.allowed_apps


def test_forbidden_launch_no_spawn(kernel: SafetyKernel, sup: ProcessSupervisor):
    before = len(sup._registry)
    kernel.validate(mk_intent(type="launch_app", app="evil"), mk_session())
    assert len(sup._registry) == before


def test_all_violation_codes_and_surface(kernel: SafetyKernel):
    assert DEFAULT_APP_LAUNCH_POLICY is not None
    for c in ("display_scope_invalid", "target_missing", "target_unresolvable", "foreign_process",
              "budget_exceeded", "schema_invalid", "confirmation_required",
              "launch_app_not_allowlisted", "launch_app_args_invalid", "spawn_env_override_forbidden"):
        assert any(c == x.value for x in ViolationCode)


# --- AppLaunchPolicy schema enforcement (FR-23) tests exercising allowed_args_pattern + blocked_args ---

def test_launch_app_args_pattern_violation(kernel: SafetyKernel):
    """xclock policy only allows ^-.*$ or empty; plain 'foo' must be rejected with ARGS_INVALID."""
    i = mk_intent(type="launch_app", app="xclock", app_args=["foo"])
    r = kernel.validate(i, mk_session())
    assert not r.valid
    assert any(v["code"] == ViolationCode.LAUNCH_APP_ARGS_INVALID.value for v in (r.violations or []))


def test_launch_app_blocked_arg(kernel: SafetyKernel):
    """test-gui-app explicitly blocks --dangerous in its policy entry."""
    i = mk_intent(type="launch_app", app="test-gui-app", app_args=["--dangerous"])
    r = kernel.validate(i, mk_session())
    assert not r.valid
    assert any(v["code"] == ViolationCode.LAUNCH_APP_ARGS_INVALID.value for v in (r.violations or []))


def test_launch_app_good_schema_passes(kernel: SafetyKernel):
    """--test-mode matches the test-gui-app policy pattern and is not blocked."""
    i = mk_intent(type="launch_app", app="test-gui-app", app_args=["--test-mode"])
    r = kernel.validate(i, mk_session())
    assert r.valid and r.normalized_action is not None
    assert r.normalized_action.app == "test-gui-app"


def test_launch_app_metachar_still_caught_even_on_permissive_policy(kernel: SafetyKernel):
    """Even the wide '.*' policy for 'true' must still reject shell metacharacters (defense in depth)."""
    i = mk_intent(type="launch_app", app="true", app_args=["-c", "echo; rm -rf /"])
    r = kernel.validate(i, mk_session())
    assert not r.valid
    assert any(v["code"] == ViolationCode.LAUNCH_APP_ARGS_INVALID.value for v in (r.violations or []))


# (clean realgui release-blocking tests follow; original base tests above untouched)

@pytest.mark.release_blocking
@pytest.mark.realgui
def test_fr30_owned_child_allow_no_prompt():
    """FR-30: post-baseline owned child (ProcessSupervisor-registered descendant) -> OWNED -> ALLOW with no prompt call."""
    fake_res = FakeOwnershipResolver(synthetic_baseline_pids={1001, 1002}, synthetic_owned={4242})
    fake_prompter = FakePrompter()  # default Deny; must never be called
    kernel = SafetyKernel(
        ownership_resolver=fake_res,
        gui_prompter=fake_prompter,
        grant_cache=None,
    )
    # element target carries app_pid that resolver classifies OWNED
    i = mk_intent(
        type="click",
        display_scope="real_act",
        target={"kind": "element", "app_pid": 4242, "handle_id": "h_owned"},
    )
    sess = mk_session(mode="REAL", real_gui_policy="full", ask_mode="on", run_id="fr30")
    r = kernel._real_act_gate(i, sess, snapshots={})
    assert r.valid is True
    assert r.normalized_action is not None
    assert r.normalized_action.display_scope == "real_act"
    assert r.normalized_action.clearance_token and r.normalized_action.clearance_token.startswith("clr:")
    assert fake_prompter.call_count == 0  # prompter never consulted for owned (FR-30)


@pytest.mark.release_blocking
@pytest.mark.realgui
def test_fr31_children_policy_foreign_deny_no_prompt():
    """FR-31: foreign pid under real_children policy -> hard DENY (FOREIGN_PROCESS), prompter never called."""
    fake_res = FakeOwnershipResolver(synthetic_baseline_pids={100}, synthetic_owned={999})  # 4242 is foreign
    fake_prompter = FakePrompter()
    cache = GrantCache(run_id="fr31")
    kernel = SafetyKernel(ownership_resolver=fake_res, gui_prompter=fake_prompter, grant_cache=cache)
    # coordinate target that _topmost will resolve (via shared helper) to a foreign pid
    snap = {"r": {"windows": [{"pid": 4242, "bbox": {"x": 0, "y": 0, "w": 100, "h": 100}, "z_index": 5, "window_id": "w1"}]}}
    i = mk_intent(type="click", display_scope="real_act", target={"kind": "coordinate", "x": 10, "y": 10})
    sess = mk_session(mode="REAL", real_gui_policy="children", ask_mode="on", run_id="fr31")
    r = kernel._real_act_gate(i, sess, snapshots=snap)
    assert r.valid is False
    codes = [v["code"] for v in (r.violations or [])]
    assert ViolationCode.FOREIGN_PROCESS.value in codes
    assert fake_prompter.call_count == 0  # children policy short-circuits before any prompt (INVARIANT D)


@pytest.mark.release_blocking
@pytest.mark.realgui
def test_fr32_full_ask_off_foreign_deny_no_prompt():
    """FR-32: foreign pid + full policy + ask_mode=off -> hard DENY (ASK_MODE_DISABLED), no prompt."""
    fake_res = FakeOwnershipResolver(synthetic_baseline_pids={100}, synthetic_owned=set())
    fake_prompter = FakePrompter()
    kernel = SafetyKernel(ownership_resolver=fake_res, gui_prompter=fake_prompter, grant_cache=GrantCache(run_id="fr32"))
    snap = {"r": {"windows": [{"pid": 7777, "bbox": {"x": 0, "y": 0, "w": 50, "h": 50}, "z_index": 1}]}}
    i = mk_intent(type="double_click", display_scope="real_act", target={"kind": "coordinate", "x": 5, "y": 5})
    sess = mk_session(mode="REAL", real_gui_policy="full", ask_mode="off", run_id="fr32")
    r = kernel._real_act_gate(i, sess, snapshots=snap)
    assert r.valid is False
    codes = [v["code"] for v in (r.violations or [])]
    assert ViolationCode.ASK_MODE_DISABLED.value in codes
    assert fake_prompter.call_count == 0  # ask=off short-circuits (INVARIANT C)


@pytest.mark.release_blocking
@pytest.mark.realgui
def test_fr40_baseline_foreign_always_denied_skeleton():
    """FR-40 (mission-critical regression): baseline pid is FOREIGN forever; full+ask=on + auto-deny prompter -> DENIED.
    Proves the prompter is consulted (default-deny path) and pid classified FOREIGN even for same-uid pre-existing 'terminal'.
    """
    op_term_pid = 9999  # simulated operator's other terminal / claude / orchestrator pid
    fake_res = FakeOwnershipResolver(synthetic_baseline_pids={op_term_pid, 100}, synthetic_owned={4242})
    fake_prompter = FakePrompter()  # defaults to decision=Deny, granted=False -> exercises fail-closed
    cache = GrantCache(run_id="fr40")
    kernel = SafetyKernel(ownership_resolver=fake_res, gui_prompter=fake_prompter, grant_cache=cache)
    # coordinate target inside a window owned by the baseline pid
    snap = {"r": {"windows": [{"pid": op_term_pid, "bbox": {"x": 0, "y": 0, "w": 800, "h": 600}, "z_index": 10, "window_id": "opwin"}]}}
    i = mk_intent(type="click", display_scope="real_act", target={"kind": "coordinate", "x": 100, "y": 100})
    sess = mk_session(mode="REAL", real_gui_policy="full", ask_mode="on", run_id="fr40-run")
    r = kernel._real_act_gate(i, sess, snapshots=snap)
    assert r.valid is False
    codes = [v.get("code") for v in (r.violations or [])]
    assert any(c in (ViolationCode.GRANT_REQUIRED.value, "grant_required") for c in codes)
    assert fake_prompter.call_count >= 1  # prompter WAS consulted, proving no silent access to baseline pid (FR-40)
    # classification truth: even though 9999 might pass naive uid check, baseline makes it FOREIGN
    assert fake_res.classify(op_term_pid) == "FOREIGN"
    assert fake_res.classify(4242) == "OWNED"

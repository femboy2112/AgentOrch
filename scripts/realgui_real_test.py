#!/usr/bin/env python3
"""REAL (non-mock) smoke test of the computer-use real-GUI security harness.

Run BEFORE verifying the branch. Unlike the hermetic pytest suite (which uses
Fakes), this drives the actual components against your LIVE session:

  A. Capability probe        — what the engine detects on this box (real).
  B. Baseline + ownership    — capture_baseline(:0) over your REAL live PIDs;
                               prove a pre-existing PID is FOREIGN and a
                               freshly-spawned child is OWNED (real psutil/pgid).
  C. SafetyKernel real_act   — FR-40 against a REAL baseline pid (this script's
                               own parent shell) -> DENIED; a real OWNED child
                               -> ALLOWED with a clearance token. (Auto-deny
                               FakePrompter, so NO dialog pops — gate logic only.)
  D. Real zenity dialog      — OPTIONAL, only if RG_SHOW_DIALOG=1: pops the
                               ACTUAL permission pop-up on :0 so you can see it,
                               pick a scope, type text; prints the parsed result.
                               This is the only part that touches :0, and it only
                               ever renders its own dialog (zero foreign input).

Safe by construction: A-C never actuate :0; the only spawned process is a
`sleep` owned child (killed at the end). D only renders zenity's own window.

  .venv/bin/python scripts/realgui_real_test.py                 # A-C (unattended)
  RG_SHOW_DIALOG=1 .venv/bin/python scripts/realgui_real_test.py # + the live dialog
"""
from __future__ import annotations

import os
import sys

# Ensure repo root on path when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agy_orchestrator.computer_use.capability import is_available
from agy_orchestrator.computer_use.grants import GrantCache
from agy_orchestrator.computer_use.gui_prompt import FakePrompter, GuiPrompter
from agy_orchestrator.computer_use.models import PromptContext
from agy_orchestrator.computer_use.ownership import OwnershipResolver
from agy_orchestrator.computer_use.process_supervisor import ProcessSupervisor
from agy_orchestrator.computer_use.safety import SafetyKernel

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{PASS if cond else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def _budgets():
    return {"max_steps": 50, "max_actions": 50, "action_timeout_ms": 10000,
            "reasoning_timeout_ms": 45000, "confirmation_wait_timeout_ms": 300000,
            "max_cpu_percent": 200, "max_rss_mb": 2048, "max_processes": 64}


def _session(**o):
    s = {"run_id": "rg-realtest", "mode": "REAL", "task_priority": "normal",
         "budgets": _budgets(), "displays": {"isolated_display": ":99"},
         "capabilities": {"action_exec": True, "atspi": False, "ocr": False,
                          "geometry": True, "dom": False, "degraded": False, "readiness": "ready"},
         "real_gui_policy": o.pop("real_gui_policy", "full"), "ask_mode": o.pop("ask_mode", "on")}
    s.update(o)
    return s


def _intent(pid):
    return {"intent_id": "i1", "snapshot_id": "s1",
            "action": {"type": "click", "display_scope": "real_act",
                       "target": {"kind": "element", "app_pid": pid}},
            "rationale": "real smoke test", "risk_level": "low",
            "requires_confirmation": False, "confirmation_token": None}


def main() -> int:
    print("\n=== A. Capability probe (real environment) ===")
    rep = is_available()
    print(f"  {rep!r}")

    print("\n=== B. Baseline + ownership over LIVE processes ===")
    sup = ProcessSupervisor()
    res = OwnershipResolver(supervisor=sup)
    info = res.capture_baseline(":0")  # MUST precede any spawn
    nbp = len(info["baseline_pids"])
    nwin = len(info["baseline_windows"])
    print(f"  captured baseline: {nbp} live PIDs frozen FOREIGN, {nwin} windows mapped (display={info['display']})")
    check("baseline captured non-empty", nbp > 0, f"{nbp} PIDs")

    parent = os.getppid()  # the shell that launched this script — alive at capture => FOREIGN
    check("pre-existing parent shell PID is FOREIGN", res.classify(parent) == "FOREIGN",
          f"ppid={parent} -> {res.classify(parent)}")
    check("PID 1 (init) is FOREIGN", res.classify(1) == "FOREIGN")

    child = sup.spawn(["sleep", "30"], display_scope="isolated", no_shell=True)
    try:
        owned = res.classify(child.pid)
        check("freshly-spawned owned child is OWNED", owned == "OWNED",
              f"pid={child.pid} is_owned={sup.is_owned(child.pid)} -> {owned}")

        print("\n=== C. SafetyKernel real_act gate (FR-40 + owned-allow, REAL pids) ===")
        kernel = SafetyKernel(ownership_resolver=res, gui_prompter=FakePrompter(),
                              grant_cache=GrantCache(run_id="rg-realtest"), supervisor=sup)

        # FR-40: a REAL baseline pid (the operator's parent shell) under the most
        # permissive policy (full + ask on) with auto-deny prompter -> DENIED.
        rf = kernel._real_act_gate(_intent(parent), _session(real_gui_policy="full", ask_mode="on"))
        check("FR-40: real baseline pid DENIED under real_full+ask-on", getattr(rf, "valid", None) is False,
              f"valid={getattr(rf, 'valid', '?')}")

        # ask-mode off -> foreign hard-denied with no prompt at all.
        ro = kernel._real_act_gate(_intent(parent), _session(real_gui_policy="full", ask_mode="off"))
        check("foreign DENIED with ask-mode off (no prompt)", getattr(ro, "valid", None) is False)

        # children policy -> foreign never injectable even if ask on.
        rc = kernel._real_act_gate(_intent(parent), _session(real_gui_policy="children", ask_mode="on"))
        check("foreign DENIED under real_children policy", getattr(rc, "valid", None) is False)

        # OWNED child -> ALLOWED, with a clearance token issued (INVARIANT E source).
        ra = kernel._real_act_gate(_intent(child.pid), _session(real_gui_policy="full", ask_mode="on"))
        # The issued clearance_token rides on the normalized ActionSpec the gate
        # builds on ALLOW (_build_real_act_allowed); the executor verifies it (INV E).
        spec = getattr(ra, "normalized_action", None)
        tok = getattr(spec, "clearance_token", None) if spec is not None else None
        check("owned child ALLOWED by gate", getattr(ra, "valid", None) is True,
              f"valid={getattr(ra, 'valid', '?')}")
        check("ALLOW issued a 'clr:' clearance token (INVARIANT E)",
              bool(tok) and str(tok).startswith("clr:"),
              f"token={tok!r}")
    finally:
        try:
            sup.terminate_tree(child.root_id)
        except Exception:
            pass

    if os.environ.get("RG_SHOW_DIALOG") == "1":
        print("\n=== D. REAL zenity permission dialog on :0 (watch your screen!) ===")
        if not os.environ.get("DISPLAY"):
            print("  (no DISPLAY set — skipping; run from a graphical session)")
        else:
            prompter = GuiPrompter(confirmation_timeout_ms=120_000)
            ctx = PromptContext(
                run_id="rg-realtest", pid=parent, action_type="click", policy="full",
                ask_mode="on", target_app_title="(demo: your shell)",
                window_id="n/a", rationale="Real-GUI harness smoke test — pick a scope + type anything.")
            print("  popping zenity --forms ... choose a Decision and optionally type a note.")
            r = prompter.prompt(ctx)
            print(f"  -> decision={r.decision!r} grant_scope={r.grant_scope!r} "
                  f"granted={r.granted} operator_text={r.operator_text!r}")
            check("dialog returned a well-formed PromptResult",
                  r.decision is not None and isinstance(r.granted, bool))
    else:
        print("\n=== D. (skipped: set RG_SHOW_DIALOG=1 to pop the real zenity dialog on :0) ===")

    print()
    if fails:
        print(f"RESULT: {FAIL} — {len(fails)} check(s) failed: {', '.join(fails)}")
        return 1
    print(f"RESULT: {PASS} — all real checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

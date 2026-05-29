# Computer-Use Worker

## Real-GUI Security Harness (REAL mode)
Ownership+baseline (FR-28/40 INV A): every PID + _NET_WM_PID at session start is FOREIGN forever.
Pre-existing terminals/claude/etc default-denied (mission-critical host-safety).
OWNED = post-baseline ProcessSupervisor descendant only (INV B).
Two policies (D1, INV D): real_full (foreign prompt-gated) vs real_children (never foreign-injectable).
ask_mode defaults "on" for REAL (INV C/F); "off" hard-denies foreign (fail-closed).
Integrators: always inject FakePrompter + FakeClock + synthetic baselines in @realgui tests (never real zenity/:0, FR-35/39).

See COMPUTER_USE_DESIGN.md and COMPUTER_USE_REALGUI_DESIGN.md (authoritative specs).
Minimal pointer (Step 14). All hardening invariants + release-blocking FRs enforced.

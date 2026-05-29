"""GuiPrompter + FakePrompter (Step 4 — prompter abstraction for real-GUI ask path).

Exact implementation of COMPUTER_USE_REALGUI_DESIGN.md §6 (GUI ask path) and
the Step-4 contract: Protocol + two implementations, Fake mandatory in CI,
Gui uses zenity --forms exactly as specified, real-operator DISPLAY/XAUTHORITY,
launched with start_new_session + preexec rlimits (killable/owned-child semantics,
supervisor accepted for DI/future wiring; direct Popen required for --forms stdout),
confirmation_wait_timeout_ms enforcement, parse to PromptResult, zero input
injection against any foreign window (own dialog only), side-effect free when
zenity absent.

This is the *only* place the operator GUI prompt (FR-33/34/35/37/39) is rendered.
SafetyKernel and SessionController receive a prompter via dependency injection
(FakePrompter in all hermetic @pytest.mark.realgui + release_blocking tests;
GuiPrompter only in production REAL runs with ask_mode=on + foreign target).

CRITICAL (FR-35/39 + INVARIANT C): FakePrompter is the *only* implementation
exercised by the full release-blocking test suite. No test body, no self-test
block, and no CI path may ever construct GuiPrompter and call .prompt() — doing
so would pop a real zenity on :0 and/or inject against a live desktop. The
module self-test below uses *only* FakePrompter and asserts zero side effects.

GuiPrompter never calls xdotool, xte, or any input-synthesis tool. It only
ever spawns its own zenity dialog (the narrow, audited exception to XAUTHORITY
isolation) and reads back the operator's combo choice + free-text entry. The
text is routed by the caller as an operator_note (FR-37); the grant scope is
fed to GrantCache (FR-36).

Matches the minimal, defensive, long-rationale style of utils.py / grants.py /
ownership.py (from __future__, stdlib only for the real path, broad try/except
for best-effort zenity, explicit positive/int checks, exhaustive docstrings).

Usage (production path, injected by SessionController):
    from agy_orchestrator.computer_use.gui_prompt import GuiPrompter, FakePrompter
    from agy_orchestrator.computer_use.models import PromptContext, PromptResult
    prompter = GuiPrompter(supervisor=proc_sup, confirmation_timeout_ms=budgets["confirmation_wait_timeout_ms"])
    # or in tests:
    fake = FakePrompter().queue(grant_scope="PROCESS_RUN", granted=True, operator_text="ok use it")
    res = fake.prompt(ctx)

The real GuiPrompter launches zenity via hardened direct Popen (stdout capture + preexec
rlimits + new session) using the operator's real DISPLAY/XAUTHORITY so the dialog can
appear on :0. The supervisor is accepted for the DI contract and future wiring; the
zenity child is still killable/rlimited. Zero wall time, zero X11, zero zenity (and
zero foreign input) in any path that supplies a FakePrompter.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Optional, Protocol

from .models import GrantScope, PromptContext, PromptResult
from .process_supervisor import ProcessSupervisor  # for type + future owned-spawn wiring


class Prompter(Protocol):
    """Single-method protocol for the operator-ask gate (FR-33/34).

    Both FakePrompter (CI) and GuiPrompter (prod REAL + ask=on + foreign) satisfy it.
    The SafetyKernel will call prompter.prompt(ctx) only for the real_act foreign+full+ask=on case.
    """

    def prompt(self, ctx: PromptContext) -> PromptResult:
        """Present the question; return operator decision + chosen grant scope + free-text note."""
        ...


class FakePrompter:
    """The *mandatory* test double for every hermetic realgui test (FR-35/39 + INV C).

    - Pre-configure via .queue(...) or constructor defaults (deny/allow with any GrantScope).
    - Supports timeout simulation (raise or scripted DENY) and explicit raise injection for FR-35.
    - Records every call (ctx, count) so tests can assert "prompter was (not) consulted".
    - Never imports subprocess/shutil/zenity paths; pure data structure.
    - A new foreign PID always gets its own prompt call (caller + GrantCache enforce).

    Tests do:
        fake = FakePrompter()
        fake.queue("Deny")
        fake.queue(grant_scope=GrantScope.ACTION.value, granted=True)
        ...
        assert fake.call_count == N
        assert "operator terminal" in str(fake.calls[0].rationale)  # FR-40 style
    """

    def __init__(self) -> None:
        self._queue: list[PromptResult] = []
        self.calls: list[PromptContext] = []
        self.call_count: int = 0
        self.raised: bool = False
        self._raise_next: bool = False
        self._default: PromptResult = PromptResult(decision="Deny", granted=False)
        self._audit_sink: Any = None  # adapter wires for FR-38 prompt_shown emit (hermetic)

    def queue(
        self,
        decision: str = "Deny",
        *,
        grant_scope: Optional[str] = None,
        operator_text: str = "",
        granted: bool = False,
    ) -> "FakePrompter":
        """Append a scripted reply (chainable in test setup)."""
        self._queue.append(
            PromptResult(
                decision=decision,
                grant_scope=grant_scope,
                operator_text=operator_text,
                granted=granted,
            )
        )
        return self

    def set_default(self, **kw: Any) -> None:
        """Change the fallback when queue is empty."""
        d = {**self._default.__dict__, **kw}
        self._default = PromptResult(**d)

    def simulate_raise(self) -> None:
        """Next prompt() will raise (tests FR-35 fail-closed on exception)."""
        self._raise_next = True

    def prompt(self, ctx: PromptContext) -> PromptResult:
        """Record ctx and return next scripted / default result (or raise)."""
        if not isinstance(ctx, PromptContext):
            # defensive: never let bad ctx reach prod logic in tests
            return PromptResult(decision="Deny", granted=False)
        # FR-38: emit prompt_shown from the prompter site (Fake path for hermetic tests; Step 12 exact uniform payload incl. run_id, identical to Gui path)
        try:
            sink = getattr(self, "_audit_sink", None)
            if sink and hasattr(sink, "emit"):
                from .models import WorkerEvent, WorkerEventType
                from datetime import datetime, timezone
                sink.emit(WorkerEvent(ts=datetime.now(timezone.utc).isoformat(), run_id=str(getattr(ctx, "run_id", "r")), event_type=WorkerEventType.PERMISSION_PROMPT_SHOWN.value, payload={"run_id": str(getattr(ctx, "run_id", "r")), "pid": getattr(ctx, "pid", None), "action_type": getattr(ctx, "action_type", None), "policy": getattr(ctx, "policy", None), "ask_mode": getattr(ctx, "ask_mode", None)}))
        except Exception:
            pass
        self.call_count += 1
        self.calls.append(ctx)

        if self._raise_next:
            self._raise_next = False
            self.raised = True
            raise TimeoutError("simulated prompter timeout/cancel/missing-zenity (FR-35)")

        if self._queue:
            return self._queue.pop(0)
        return self._default


class GuiPrompter:
    """Production prompter: zenity --forms on the operator's real :0.

    - Uses the *real* DISPLAY/XAUTHORITY from the current environment (narrow audited
      exception to XAUTHORITY isolation #1 — only for our own dialog, never for any
      action that touches foreign windows).
    - Launched with start_new_session + preexec rlimits (mirrors ProcessSupervisor
      hardening) so the short-lived zenity is a killable owned child even though we
      must capture its --forms stdout (supervisor.spawn DEVNULLs stdio).
    - Exactly four grant scopes in --add-combo per §6.2 + free-text --add-entry.
    - --timeout derived from budgets.confirmation_wait_timeout_ms; python-level
      communicate() timeout as backstop.
    - Any non-zero exit, timeout, empty/garbled output, or missing zenity -> DENY
      (fail-closed, FR-35).
    - Never emits xdotool, xte, ydotool, or any synthetic input. The only X11 client
      launched is zenity itself (its own dialog window).
    - Side-effect free on systems without zenity(1): early return DENY result, zero Popen.
    - The injected supervisor (if any) is stored for the DI contract / future ownership
      registration; FakePrompter is the only implementation used in hermetic tests.

    The caller (SafetyKernel) is responsible for turning the returned grant_scope into
    a cached Grant (via GrantCache) and for emitting the permission.* events (FR-38).
    """

    # Exact four choices the operator sees (D2 + §6.2). Order is significant for UX.
    _CHOICES: str = "Deny|Allow once|Allow this process (run)|Allow this process (10 min)"

    def __init__(
        self,
        supervisor: Optional[ProcessSupervisor] = None,
        confirmation_timeout_ms: int = 300_000,
    ) -> None:
        self.supervisor = supervisor
        self.timeout_ms = max(5_000, int(confirmation_timeout_ms or 300_000))
        self._audit_sink: Any = None  # wired by adapter for FR-38 (prompt_shown only; zero foreign act)

    def _zenity_available(self) -> bool:
        return shutil.which("zenity") is not None

    def prompt(self, ctx: PromptContext) -> PromptResult:
        """Render the forms dialog on real :0 (or DENY fast when zenity absent)."""
        if not isinstance(ctx, PromptContext):
            return PromptResult(decision="Deny", granted=False, operator_text="bad context -> DENY")
        # FR-38: emit prompt_shown from the prompter site (Gui path; only own dialog, never foreign; Step 12 exact uniform payload incl. run_id, identical to Fake path)
        try:
            sink = getattr(self, "_audit_sink", None)
            if sink and hasattr(sink, "emit"):
                from .models import WorkerEvent, WorkerEventType
                from datetime import datetime, timezone
                sink.emit(WorkerEvent(ts=datetime.now(timezone.utc).isoformat(), run_id=str(getattr(ctx, "run_id", "r")), event_type=WorkerEventType.PERMISSION_PROMPT_SHOWN.value, payload={"run_id": str(getattr(ctx, "run_id", "r")), "pid": getattr(ctx, "pid", None), "action_type": getattr(ctx, "action_type", None), "policy": getattr(ctx, "policy", None), "ask_mode": getattr(ctx, "ask_mode", None)}))
        except Exception:
            pass

        if not self._zenity_available():
            # Side-effect free contract (FR-35): no Popen, no env read beyond which,
            # no dialog, instant DENY. Tests and CI machines without zenity are safe.
            return PromptResult(
                decision="Deny",
                grant_scope=None,
                operator_text="zenity not found -> DENY (fail-closed)",
                granted=False,
            )

        # Build the question text (concise; operator sees pid/app/rationale/run).
        title = f"AgentOrch Real-GUI Permission — {ctx.run_id}"
        text = (
            f"Agent wants: {ctx.action_type}\n"
            f"Target: app='{ctx.target_app_title or 'unknown'}' pid={ctx.pid} win={ctx.window_id or 'n/a'}\n"
            f"Rationale: {ctx.rationale or '(none provided)'}\n"
            f"Policy={ctx.policy} ask_mode={ctx.ask_mode}\n\n"
            "Select grant scope below. Type any steering instructions in the box."
        )

        # zenity --forms invocation (exact four scopes + free-text entry).
        # We pass --timeout (supported on modern zenity) + python communicate backstop.
        to_sec = max(5, self.timeout_ms // 1000)
        argv = [
            "zenity",
            "--forms",
            f"--title={title}",
            f"--text={text}",
            "--add-combo=Decision",
            f"--combo-values={self._CHOICES}",
            "--add-entry=Your instructions (optional)",
            f"--timeout={to_sec}",
        ]

        # Real operator environment (never call get_isolated_env / xauth sanitizers).
        # DISPLAY/XAUTHORITY/WAYLAND_DISPLAY are left exactly as the parent process sees them.
        env = os.environ.copy()

        # Launch the dialog. We use direct Popen (required to capture stdout of --forms;
        # ProcessSupervisor.spawn always DEVNULLs stdio for its fire-and-forget contract).
        # We replicate the exact start_new_session + preexec rlimit hardening from the
        # supervisor so the zenity remains a killable, rlimited owned child (FR-39).
        # The supervisor (if passed) is retained for the dependency-injection contract and
        # any future ownership-registration extension; terminate_tree ancestry sweeps plus
        # session shutdown will ensure cleanup of this short-lived dialog. The env is the
        # *real* operator :0 (never isolated cookie) — the narrow audited exception.
        # No foreign window is ever addressed; only our own forms dialog appears on :0.
        def _preexec_rlimits() -> None:
            """Apply the same NPROC floor the supervisor uses, but only to the zenity child."""
            try:
                import resource

                cur_nsoft, cur_nhard = resource.getrlimit(resource.RLIMIT_NPROC)
                if cur_nhard > 0 and cur_nhard > 4096:
                    resource.setrlimit(resource.RLIMIT_NPROC, (4096, min(cur_nhard, 8192)))
            except Exception:
                pass  # best-effort; the kernel backstop is defense-in-depth only

        try:
            proc = subprocess.Popen(
                argv,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
                preexec_fn=_preexec_rlimits,
            )

            out_b, _ = proc.communicate(timeout=to_sec + 2.0)
            rc = proc.returncode

            if rc != 0:
                # Cancel, close, timeout, or zenity error -> fail-closed DENY (FR-35)
                return PromptResult(
                    decision="Deny",
                    grant_scope=None,
                    operator_text="operator cancelled / timeout / zenity error -> DENY (FR-35)",
                    granted=False,
                )

            text_out = (out_b or b"").decode("utf-8", errors="replace").strip()
            # --forms with one combo + one entry emits: "Choice|typed text"
            parts = [p.strip() for p in text_out.split("|", 1)]
            choice = parts[0] if parts and parts[0] else "Deny"
            op_text = parts[1] if len(parts) > 1 else ""

            grant_scope: Optional[str] = None
            granted = False
            cl = choice.lower()
            if "once" in cl:
                grant_scope = GrantScope.ACTION.value
                granted = True
            elif "run" in cl:
                grant_scope = GrantScope.PROCESS_RUN.value
                granted = True
            elif "10" in cl or "min" in cl:
                grant_scope = GrantScope.PROCESS_TTL.value
                granted = True
            # anything else (including explicit "Deny") -> DENY, granted=False

            return PromptResult(
                decision=choice,
                grant_scope=grant_scope,
                operator_text=op_text,
                granted=granted,
            )

        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            return PromptResult(
                decision="Deny",
                grant_scope=None,
                operator_text="python-level timeout -> DENY (FR-35 fail-closed)",
                granted=False,
            )
        except Exception as e:
            # Missing DISPLAY, permission, zenity segfault, etc. -> DENY (fail-closed)
            return PromptResult(
                decision="Deny",
                grant_scope=None,
                operator_text=f"prompter failure ({type(e).__name__}) -> DENY (FR-35)",
                granted=False,
            )


# ------------------------------------------------------------------
# Hermetic self-test (python -m agy_orchestrator.computer_use.gui_prompt).
# Exercises ONLY FakePrompter. Never constructs GuiPrompter, never calls which/zenity,
# never reads :0, never spawns anything. Covers the pre-config, raise, queue, and
# call-recording paths required by FR-33/34/35/36/39 tests.
# ------------------------------------------------------------------
if __name__ == "__main__":
    # All of the following must pass with zero host GUI side effects.
    fp = FakePrompter()
    ctx = PromptContext(
        run_id="20260529-000000-0",
        pid=4242,
        action_type="click",
        policy="full",
        ask_mode="on",
        target_app_title="Terminal",
        window_id="0xabc",
        rationale="test foreign baseline regression",
    )

    # Default is deny
    r0 = fp.prompt(ctx)
    assert r0.granted is False and r0.decision == "Deny"
    assert fp.call_count == 1 and len(fp.calls) == 1

    # Scripted allow + free text (FR-34)
    fp.queue(grant_scope=GrantScope.PROCESS_RUN.value, granted=True, operator_text="use the terminal I left open")
    r1 = fp.prompt(ctx)
    assert r1.granted is True and r1.grant_scope == "PROCESS_RUN"
    assert "use the terminal" in r1.operator_text
    assert fp.call_count == 2

    # ACTION single-use scope
    fp.queue(grant_scope=GrantScope.ACTION.value, granted=True)
    r2 = fp.prompt(ctx)
    assert r2.grant_scope == "ACTION"
    assert fp.call_count == 3

    # Simulate raise path (FR-35)
    fp.simulate_raise()
    try:
        _ = fp.prompt(ctx)
        assert False, "expected raise"
    except TimeoutError as te:
        assert "FR-35" in str(te)
    assert fp.raised is True and fp.call_count == 4

    # New foreign pid always gets its own prompt (call recording proves it)
    ctx2 = PromptContext(run_id=ctx.run_id, pid=9999, action_type="type", policy="children", ask_mode="on")
    fp.queue("Deny")
    r3 = fp.prompt(ctx2)
    assert r3.granted is False
    assert fp.calls[-1].pid == 9999
    assert fp.call_count == 5

    # Protocol shape check (construction only; .prompt never called on Gui here)
    _ = GuiPrompter  # type exists and is importable

    print("gui_prompt.py Fake-only self-test OK (4 scopes + raise + queue + recording; zero zenity/:0)")

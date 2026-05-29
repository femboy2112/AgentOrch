"""CapabilityBroker (Step 7) — runtime sensor + action capability probe.

Implements FR-05 (graceful AT-SPI/PyGObject degradation), FR-15 (capability
matrix + readiness returned by is_available() / probe()), and supports the
exact CapabilityReport shape from models.py (used by WorkerSession, adapter,
SessionController, and capability.probe events).

Detections (all side-effect free, no X server / :0 interaction):
- PyGObject / Atspi: try import + require_version + from gi.repository (broad except)
- ocr: tesseract binary via shutil.which
- geometry: xdotool + wmctrl + xwininfo (for window/element discovery)
- action_exec: xdotool (for xdotool-based ActionExecutor on isolated display)
- dom: playwright package (import) or playwright CLI on PATH
- psutil: always (hard dep of ProcessSupervisor); graceful note if absent

readiness:
  "ready"     — full stack: atspi + ocr + dom + geometry + action_exec (+psutil) all present.
                This is the only non-degraded state.
  "degraded"  — partial (e.g. atspi missing per FR-05, ocr/dom/geometry incomplete,
                or psutil absent); still functional for remaining paths (FR-05, reliability).
  "unavailable" — no action_exec (xdotool absent → cannot act at all). Zero perception
                channels with action_exec present yields degraded + "severely limited" note
                (allows blind launch_app/hotkey/wait scenarios; graceful per contract).

notes[] populated for every missing piece (actionable operator guidance).
degraded flag mirrors readiness != "ready" (plus explicit atspi=False case).

is_available() is the delegation target for the future ComputerUseWorkerAdapter.
probe() is the implementation called at every run start (and on sensor faults).

Pure stdlib + optional imports. Zero GUI, zero X11 calls, zero real-:0 side effects.
Matches style of safety.py / process_supervisor.py / utils.py.
"""

from __future__ import annotations

import builtins
import shutil
import sys
from typing import Any, Callable, Dict, List, Optional

from .models import CapabilityReport, Readiness


class CapabilityBroker:
    """Runtime capability detection and degrade-state reporter (FR-05, FR-15).

    probe() is the single source of truth for what the worker can do in the
    current environment. It is called early in SessionController.create_session
    and its result is stored in WorkerSession.capabilities and emitted as a
    capability.probe audit event.
    """

    def probe(self) -> CapabilityReport:
        """Return a validated CapabilityReport reflecting current host capabilities.

        Never raises on missing optional components — all detection is defensive.
        """
        notes: List[str] = []

        # ------------------------------------------------------------------
        # AT-SPI / PyGObject (FR-05: must degrade gracefully, never hard-fail)
        # ------------------------------------------------------------------
        atspi = False
        try:
            import gi  # type: ignore

            gi.require_version("Atspi", "2.0")
            from gi.repository import Atspi  # noqa: F401  # type: ignore

            atspi = True
        except Exception as exc:
            notes.append(
                f"PyGObject/AT-SPI unavailable ({type(exc).__name__}); "
                "degraded to OCR+geometry per FR-05"
            )

        # ------------------------------------------------------------------
        # OCR: tesseract binary (OCRCollector shells out; pytesseract optional)
        # ------------------------------------------------------------------
        ocr = bool(shutil.which("tesseract"))
        if not ocr:
            notes.append("tesseract binary not on PATH; OCR text channel unavailable")

        # ------------------------------------------------------------------
        # Geometry + action execution tools (xdotool is the executor; others for discovery)
        # ------------------------------------------------------------------
        has_xdotool = bool(shutil.which("xdotool"))
        has_wmctrl = bool(shutil.which("wmctrl"))
        has_xwininfo = bool(shutil.which("xwininfo"))

        geometry = has_xdotool and has_wmctrl and has_xwininfo
        if not geometry:
            missing_geom = [
                name
                for name, present in [
                    ("xdotool", has_xdotool),
                    ("wmctrl", has_wmctrl),
                    ("xwininfo", has_xwininfo),
                ]
                if not present
            ]
            missing_str = ", ".join(missing_geom)
            notes.append(f"geometry tools incomplete: missing {missing_str}")

        action_exec = has_xdotool  # xdotool provides click, type, key, etc. on isolated $DISPLAY
        if not action_exec:
            notes.append("xdotool not on PATH; ActionExecutor cannot inject input (FR-24 paths blocked)")

        # ------------------------------------------------------------------
        # DOM: Playwright (python package preferred; CLI fallback for detection)
        # ------------------------------------------------------------------
        dom = False
        try:
            import playwright  # noqa: F401  # type: ignore

            dom = True
        except Exception:
            if shutil.which("playwright"):
                dom = True
            else:
                notes.append(
                    "playwright not installed (pip install playwright && playwright install); "
                    "DOM snapshots unavailable"
                )

        # ------------------------------------------------------------------
        # psutil (ProcessSupervisor hard dep for rlimits + tree enforcement, hardening #3)
        # ------------------------------------------------------------------
        psutil_ok = False
        try:
            import psutil  # noqa: F401

            psutil_ok = True
        except Exception:
            notes.append("psutil unavailable — resource caps (hardening #3) will be inoperative")

        # ------------------------------------------------------------------
        # Readiness + degraded computation (exact contract)
        # ------------------------------------------------------------------
        has_text_perception = atspi or ocr or dom
        has_any_perception = has_text_perception or geometry  # geometry alone is weak but present

        if not action_exec:
            readiness = Readiness.UNAVAILABLE.value
            degraded = True
        elif not has_text_perception:
            readiness = Readiness.DEGRADED.value
            degraded = True
            if not has_any_perception:
                notes.append("no perception channels available (AT-SPI/OCR/DOM/geometry); worker severely limited")
        else:
            # Action possible + at least one text channel
            missing_core = []
            if not atspi:
                missing_core.append("atspi")
            if not ocr:
                missing_core.append("ocr")
            if not dom:
                missing_core.append("dom")
            if not geometry:
                missing_core.append("geometry")
            if missing_core:
                readiness = Readiness.DEGRADED.value
                degraded = True
            else:
                readiness = Readiness.READY.value
                degraded = False

        if not psutil_ok:
            degraded = True
            if readiness == Readiness.READY.value:
                readiness = Readiness.DEGRADED.value

        # Ensure notes is None when empty (matches dataclass default + serialization)
        notes_list: Optional[List[str]] = notes if notes else None

        report = CapabilityReport(
            atspi=atspi,
            ocr=ocr,
            geometry=geometry,
            dom=dom,
            action_exec=action_exec,
            degraded=degraded,
            readiness=readiness,
            notes=notes_list,
        )
        return report


# -----------------------------------------------------------------------------
# Public helpers (adapter delegation point + direct calls from SessionController)
# -----------------------------------------------------------------------------

def probe() -> CapabilityReport:
    """Return current capability matrix. Preferred entry point for run startup.

    CapabilityBroker.probe() is the implementation per Step 7 instruction.
    """
    return CapabilityBroker().probe()


def probe_capabilities() -> CapabilityReport:
    """Alias expected by package __init__.py surface (Step 7 integration point).

    Delegates to the canonical probe(). Adapter and SessionController may use
    either name; both return the exact CapabilityReport shape.
    """
    return probe()


def is_available() -> CapabilityReport:
    """Adapter-facing is_available() implementation.

    Future ComputerUseWorkerAdapter.is_available() MUST delegate to this
    (or re-export it) so that AgentOrch harness sees the full graded report
    rather than a bare bool. Matches the contract in DESIGN §5 table.
    """
    return probe()


# Optional helper for tests / harness that want the raw matrix without the dataclass
def get_capability_matrix() -> Dict[str, Any]:
    """Return a plain dict snapshot (useful for quick health checks)."""
    r = probe()
    return {
        "atspi": r.atspi,
        "ocr": r.ocr,
        "geometry": r.geometry,
        "dom": r.dom,
        "action_exec": r.action_exec,
        "degraded": r.degraded,
        "readiness": r.readiness,
        "notes": r.notes or [],
    }


# Step 7 complete: CapabilityBroker + probe()/is_available() fully wired (FR-05/15).
# The implementation satisfies the exact contract, graceful gi handling, tool
# detection matrix (tesseract / xdotool+wmctrl+xwininfo / playwright / psutil),
# Readiness/degraded/notes, and zero X / real-:0 side effects. Tests in
# test_computer_use_capability.py cover the mandated gi-monkeypatch scenario.

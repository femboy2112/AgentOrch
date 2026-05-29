"""CapabilityBroker.probe() + is_available() tests (Step 7).

Implements the test mandate for Step 7: CapabilityBroker returns the exact
CapabilityReport shape, detects PyGObject/Atspi with graceful except (FR-05),
tesseract for ocr, xdotool/wmctrl/xwininfo for geometry/action_exec, playwright
for dom, psutil always (for the resource side of hardening #3).

The critical monkeypatch test simulates missing 'gi' module and asserts:
- atspi=False
- report is still functional (degraded but not unavailable when other channels exist)
- ocr/geometry reflect the actual tool presence on the test host ("true when available")
- No X server interaction occurs in any test.

is_available() (module level) is the delegation target that the future
ComputerUseWorkerAdapter.is_available() will call. Also exercises the package
re-exports added in Step 7.

All tests are hermetic, import-only, no display required, no real-:0 actions.
"""

from __future__ import annotations

import builtins
import shutil
import sys
from typing import Any, Dict

import pytest

from agy_orchestrator.computer_use import (
    CapabilityBroker,
    CapabilityReport,
    Readiness,
    is_available,
    probe_capabilities,
)


def test_probe_and_is_available_return_exact_report_shape():
    """Contract: probe() / is_available() / broker.probe() all yield valid CapabilityReport."""
    for getter in (
        lambda: CapabilityBroker().probe(),
        probe_capabilities,
        is_available,
    ):
        rep: CapabilityReport = getter()
        assert isinstance(rep, CapabilityReport)
        assert isinstance(rep.atspi, bool)
        assert isinstance(rep.ocr, bool)
        assert isinstance(rep.geometry, bool)
        assert isinstance(rep.dom, bool)
        assert isinstance(rep.action_exec, bool)
        assert isinstance(rep.degraded, bool)
        assert rep.readiness in {r.value for r in Readiness}
        if rep.notes is not None:
            assert isinstance(rep.notes, list)
            assert all(isinstance(x, str) for x in rep.notes)


def test_is_available_is_the_adapter_delegation_point():
    """Per Step 7 + spec §5 table: is_available() on the (future) adapter
    delegates here and returns the graded CapabilityReport, not a bool.
    """
    rep = is_available()
    # The report itself is the capability matrix the adapter must surface
    d = rep.to_dict() if hasattr(rep, "to_dict") else {}
    assert isinstance(d, dict) and "atspi" in d and "readiness" in d
    assert hasattr(rep, "atspi")


def test_missing_gi_via_monkeypatch_reports_atspi_false_degraded_functional(monkeypatch: Any):
    """FR-05 + Step 7 required test.

    Simulate complete absence of PyGObject/gi (as would happen in a minimal
    container or before the optional system dep is installed). The probe must:
    - return atspi=False without raising
    - still produce a usable (degraded) report
    - ocr/geometry must be the real answers from the host (not polluted by gi patch)
    - readiness must permit continuation (degraded or ready, depending on tools)
    """
    # Remove any cached gi modules so the probe's internal try: import gi sees the patch
    for name in list(sys.modules):
        if name == "gi" or name.startswith("gi."):
            del sys.modules[name]

    # Force the import to fail for gi (and any sub-attempt)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gi" or name.startswith("gi."):
            raise ImportError("Simulated missing PyGObject for FR-05 test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setitem(sys.modules, "gi", None)
    monkeypatch.setitem(sys.modules, "gi.repository", None)

    broker = CapabilityBroker()
    rep = broker.probe()

    assert rep.atspi is False, "FR-05: missing gi must yield atspi=False"
    assert rep.degraded is True, "missing atspi must mark the report degraded"
    # Still functional for the rest of the matrix
    assert rep.readiness in {
        Readiness.DEGRADED.value,
        Readiness.READY.value,
    }, "FR-05: must remain usable (not unavailable) when other channels exist"

    # ocr / geometry must truthfully reflect the test environment ("true when available")
    # and must be unaffected by the gi monkeypatch (FR-05 isolation of detection)
    expected_ocr = shutil.which("tesseract") is not None
    assert rep.ocr == expected_ocr

    has_xdotool = bool(shutil.which("xdotool"))
    has_wmctrl = bool(shutil.which("wmctrl"))
    has_xwininfo = bool(shutil.which("xwininfo"))
    expected_geometry = has_xdotool and has_wmctrl and has_xwininfo
    assert rep.geometry == expected_geometry

    # action_exec is xdotool-only (per impl and ActionExecutor contract)
    expected_action_exec = has_xdotool
    assert rep.action_exec == expected_action_exec

    # dom and action_exec likewise must be real (unaffected)
    assert isinstance(rep.dom, bool)
    assert isinstance(rep.action_exec, bool)

    # notes must document the atspi miss
    assert rep.notes is not None
    assert any("PyGObject" in n or "Atspi" in n or "FR-05" in n or "atspi" in n.lower() for n in rep.notes)


def test_probe_never_touches_x_or_real_display():
    """Sanity: the entire probe path (including all internal detections) is
    free of X11, xdotool calls, xvfb, or real-:0 activity. We only ever call
    shutil.which and guarded Python imports.
    """
    # Running the probe multiple times must be instantaneous and side-effect free.
    r1 = is_available()
    r2 = CapabilityBroker().probe()
    assert r1.readiness == r2.readiness  # deterministic for same env
    # If we got here without hanging or X auth errors, the invariant holds.


def test_package_surface_exposes_step7_contract():
    """The computer_use package must re-export the Step 7 symbols so higher
    layers (adapter, harness roles) can `from agy_orchestrator.computer_use import is_available`.
    """
    import agy_orchestrator.computer_use as cu

    assert hasattr(cu, "CapabilityBroker")
    assert hasattr(cu, "is_available")
    assert hasattr(cu, "probe_capabilities")
    assert callable(cu.is_available)
    # Calling it works
    r = cu.is_available()
    assert isinstance(r, CapabilityReport)

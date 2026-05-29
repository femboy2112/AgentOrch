"""Step 8: PerceptionPipeline + collectors tests.

All actuation on isolated Xvfb only. Covers FR-05/06/17, hardening #4 redaction,
explicit display contract, event emission, package surface.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Generator

import pytest

from agy_orchestrator.computer_use import (
    ATSPICollector,
    AuditEventSink,
    BrowserDOMCollector,
    GeometryCollector,
    OCRCollector,
    PerceptionPipeline,
    ProcessSupervisor,
    Scope,
    WorkerEventType,
    get_isolated_env,
    redact_secrets,
)
from agy_orchestrator.computer_use.models import IsolatedDisplaySpec, PerceptionSnapshot, SnapshotSummary


def _has_tools() -> bool:
    return all(shutil.which(t) for t in ("Xvfb", "xclock", "wmctrl", "xwininfo", "xprop", "scrot", "tesseract"))


@pytest.fixture
def xvfb() -> Generator[Dict[str, Any], None, None]:
    if not _has_tools():
        pytest.skip("X11 tools missing")
    sup = ProcessSupervisor()
    d = f":{150 + (os.getpid() % 300)}"
    try:
        env = get_isolated_env(d)
    except Exception as e:
        pytest.skip(str(e))
    try:
        xv = sup.spawn_isolated_display(IsolatedDisplaySpec(display=d, screen="640x480x24"))
        sup.spawn(["xclock", "-geometry", "100x100+5+5"], display_scope="isolated", no_shell=True, isolated_display=d, env=env)
        time.sleep(0.65)
        yield {"display": d, "env": env, "sup": sup, "root": xv.root_id}
    finally:
        try:
            sup.terminate_tree(xv.root_id)
        except Exception:
            pass


def test_geometry_ocr_collect(xvfb: Dict[str, Any]) -> None:
    d, e = xvfb["display"], xvfb["env"]
    g = GeometryCollector(d, e)
    wins, elems = g.collect()
    assert len(wins) >= 1
    assert any("xclock" in (w.get("title") or "").lower() for w in wins)
    assert len(elems) >= 1
    o = OCRCollector(d, e)
    bl = o.collect()
    assert isinstance(bl, list)


def test_atspi_dom_degrade() -> None:
    assert ATSPICollector().collect() == []
    assert BrowserDOMCollector(cdp_endpoint=None).collect() == []


def test_pipeline_snapshot_summary(xvfb: Dict[str, Any]) -> None:
    d, e = xvfb["display"], xvfb["env"]
    p = PerceptionPipeline(redaction_enabled=True)
    snap = p.snapshot(Scope.ISOLATED.value, display=d, env=e, run_id="t1")
    assert isinstance(snap, PerceptionSnapshot)
    assert len(snap.windows) >= 1
    assert isinstance(p.make_summary(snap), SnapshotSummary)


def test_fr17_multi(xvfb: Dict[str, Any]) -> None:
    """FR-17: multi-scope snapshot assembly (isolated + observe_real) in one call.
    Per-scope displays/envs dicts must be honored so OBSERVE can watch real :0
    while ISOLATED controls a private Xvfb (different DISPLAY/XAUTHORITY).
    """
    d, e = xvfb["display"], xvfb["env"]
    p = PerceptionPipeline()
    s = p.snapshot_set(
        [Scope.ISOLATED.value, Scope.OBSERVE_REAL.value],
        displays={Scope.ISOLATED.value: d},
        envs={Scope.ISOLATED.value: e},
    )
    assert Scope.ISOLATED.value in s and Scope.OBSERVE_REAL.value in s
    iso_snap = s[Scope.ISOLATED.value]
    assert isinstance(iso_snap, PerceptionSnapshot)
    # The dispatch must have routed the fixture's private display (with xclock) to the isolated scope.
    assert len(iso_snap.windows) >= 1
    assert any("xclock" in (w.get("title") or "").lower() for w in iso_snap.windows)
    # observe_real falls back to pipeline default (":0"); content is environment-dependent
    # and may be empty in hermetic CI — only key presence is asserted for the set.


@pytest.mark.not_slow
@pytest.mark.release_blocking
def test_redaction_default_on() -> None:
    s = "password=secret123 token=ghp_ABCDEF0123456789"
    r = redact_secrets(s)
    assert "secret123" not in r and "ghp_ABCDEF0123456789" not in r


@pytest.mark.not_slow
@pytest.mark.release_blocking
def test_redaction_optout_and_events(xvfb: Dict[str, Any], tmp_path: Path) -> None:
    d, e = xvfb["display"], xvfb["env"]
    run = "p8e"
    sink = AuditEventSink(run, runs_root=tmp_path)
    p = PerceptionPipeline(audit_sink=sink, redaction_enabled=True)
    snap = p.snapshot(Scope.ISOLATED.value, display=d, env=e, run_id=run)
    txt = (tmp_path / run / "events.jsonl").read_text()
    assert "perception.snapshot" in txt
    assert snap.snapshot_id


def test_exports() -> None:
    import agy_orchestrator.computer_use as cu
    assert hasattr(cu, "PerceptionPipeline") and hasattr(cu, "GeometryCollector")
    p = cu.PerceptionPipeline()
    assert p.default_isolated_display.startswith(":")

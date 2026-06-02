"""Issue #64: the persisted verify_step log was a fixed ~2 KB TAIL, dropping the
pytest tracebacks (emitted before the short summary). The fix persists a sibling
verify_step<N>_iter<M>.full.log with the FULL output (head+tail bounded) so the
assertion detail of a thrashing step survives.
"""
from __future__ import annotations

import asyncio

import pytest

from agy_orchestrator.execution.verifier import QualityVerifier


@pytest.mark.not_slow
def test_head_tail_keeps_both_ends():
    text = "HEAD_MARKER" + ("x" * 5000) + "TAIL_MARKER"
    out = QualityVerifier._head_tail(text, 400)
    assert "HEAD_MARKER" in out and "TAIL_MARKER" in out
    assert "elided" in out
    assert len(out) < len(text)


@pytest.mark.not_slow
def test_head_tail_passthrough_under_cap():
    text = "short output"
    assert QualityVerifier._head_tail(text, 1000) == text


@pytest.mark.not_slow
def test_full_log_preserves_traceback_dropped_by_tail(tmp_path):
    # A command that prints a big traceback block FIRST, then >2 KB of filler, then
    # the FAILED summary last. The 2 KB tail .log loses the traceback; the .full.log
    # must retain it.
    # The marker is assembled at runtime (B=BUG) so the literal "THE_REAL_BUG"
    # appears ONLY in the OUTPUT, never in the command string echoed into the log
    # header — otherwise it would show up in both files for the wrong reason.
    script = (
        "B=BUG; "
        "echo Traceback most recent call last; "
        "echo E assert 2 == 3 THE_REAL_$B; "
        "for i in $(seq 1 200); do echo filler line $i ................................; done; "
        "echo FAILED tests/test_x.py::test_one; "
        "exit 1"
    )
    v = QualityVerifier([script])
    v.run_dir = str(tmp_path)
    v.step_index = 0

    res = asyncio.run(v.verify(working_directory=str(tmp_path)))
    assert res.ok is False

    capped = tmp_path / "verify_step0_iter1.log"
    full = tmp_path / "verify_step0_iter1.full.log"
    assert capped.exists() and full.exists()

    capped_txt = capped.read_text()
    full_txt = full.read_text()
    # The traceback (head of stdout) is dropped by the tail-only quick-glance file…
    assert "THE_REAL_BUG" not in capped_txt
    # …but preserved in the full log.
    assert "THE_REAL_BUG" in full_txt
    assert "FAILED tests/test_x.py::test_one" in full_txt


@pytest.mark.not_slow
def test_failed_tests_parsed_from_full_output(tmp_path):
    # FAILED line appears early (before >2 KB of filler), so a tail-only parse would
    # miss it; parsing the full output must still surface it in the event stream.
    events = []
    script = (
        "echo 'FAILED tests/test_early.py::test_zero - boom'; "
        "for i in $(seq 1 200); do echo filler $i ........................................; done; "
        "exit 1"
    )
    v = QualityVerifier([script])
    v.run_dir = str(tmp_path)
    v.step_index = 0
    v.event_callback = lambda e: events.append(e)

    asyncio.run(v.verify(working_directory=str(tmp_path)))
    blob = str(events)
    assert "tests/test_early.py::test_zero" in blob

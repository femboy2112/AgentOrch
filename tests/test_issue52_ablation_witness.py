"""Hermetic tests for the programmatic ablation-witness hook (issue #52).

The reconcile station's load-bearing witness used to be whatever number the LLM
self-reported. ``--ablation-cmd`` instead MEASURES it: run a cheap witness command
twice in a throwaway worktree (clean, then with ``AGY_ABLATE=<mech>`` set so the
project disables the component) and record the real with/without delta. A model
that CLAIMS load-bearing but whose measured delta is 0 is flipped to dead.

The agent is mocked (canned JSON reply, no LLM/network); the witness "command" is a
tiny shell script that varies its output based on ``AGY_ABLATE`` — no real worker
CLI, no network. ``working_directory`` is a non-git tmp dir so ``candidate_workspace``
takes the copy path (no git required).
"""
from __future__ import annotations

import asyncio
import json
import textwrap

from agy_orchestrator.workflows.reconcile import (
    ReconciliationResult,
    ReconciliationReview,
    Witness,
    _parse_witness_signal,
)
from agy_orchestrator.core.agent import AgentInstance


class _StubAgent(AgentInstance):
    """An AgentInstance whose run_async returns a canned reply."""

    def __init__(self, reply: str, prompt: str = ""):
        super().__init__(prompt=prompt)
        self._reply = reply

    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input=None) -> str:
        return self._reply


def _run(reply: str, work_dir, **kwargs) -> ReconciliationResult:
    agent = _StubAgent(reply)
    station = ReconciliationReview(
        agent, goal="build a live-learning agent",
        working_directory=str(work_dir), **kwargs,
    )
    return asyncio.run(station.execute())


# A witness shell command: prints a DIFFERENT number when AGY_ABLATE matches the
# given mechanism (component disabled -> signal moves), the SAME number otherwise
# (component absent / not honored -> signal stays put). {MECH} is substituted by
# the station with the mechanism name.
_MOVES_CMD = (
    'python3 -c "import os,sys; '
    'sys.stdout.write(\'WITNESS:0.0\') if os.environ.get(\'AGY_ABLATE\')==\'{MECH}\' '
    'else sys.stdout.write(\'WITNESS:0.91\')"'
)
# A witness that NEVER moves regardless of ablation (dead wiring): the project does
# not honor AGY_ABLATE for this mechanism, so the signal is constant.
_STATIC_CMD = (
    'python3 -c "import sys; sys.stdout.write(\'WITNESS:0.42\')"'
)


# --------------------------------------------------------------------------- #
# parse helper: last number / WITNESS tag
# --------------------------------------------------------------------------- #
def test_parse_witness_signal_prefers_tag_then_last_number():
    assert _parse_witness_signal("noise 1 2 WITNESS: 3.5 trailing") == 3.5
    assert _parse_witness_signal("only bare 7 then 8.25") == 8.25
    assert _parse_witness_signal("no numbers here") is None
    assert _parse_witness_signal("") is None


# --------------------------------------------------------------------------- #
# (success path) a real moving witness: measured delta replaces the self-report,
# load_bearing stands, witness.measured is True.
# --------------------------------------------------------------------------- #
def test_measured_moving_witness_confirms_load_bearing(tmp_path):
    (tmp_path / "src.py").write_text("# the assembled code\n")
    reply = json.dumps({
        "reconciled": True,
        "findings": [{
            "name": "surprise",
            "classification": "load_bearing",
            "location": "src.py:1",
            "witness": {"value": 5.0, "description": "model says it moves"},
            "evidence": "called from train()",
        }],
    })
    result = _run(reply, tmp_path, ablation_cmd=_MOVES_CMD)
    assert result.reconciled is True
    (f,) = result.findings_list
    assert f.classification == "load_bearing"
    assert f.witness.measured is True
    # baseline 0.91, ablated 0.0 -> delta 0.91 (NOT the model's self-reported 5.0)
    assert abs(f.witness.value - 0.91) < 1e-9


# --------------------------------------------------------------------------- #
# (failure/edge path the issue describes) model CLAIMS load-bearing, but the
# measured delta is 0 -> mismatch flips the finding to exists_not_load_bearing.
# --------------------------------------------------------------------------- #
def test_mismatch_flips_to_exists_not_load_bearing(tmp_path):
    (tmp_path / "src.py").write_text("# the assembled code\n")
    reply = json.dumps({
        "reconciled": True,
        "findings": [{
            "name": "surprise",
            "classification": "load_bearing",  # model LIES that it moves
            "location": "src.py:1",
            "witness": {"value": 9.0, "description": "model claims a big move"},
            "evidence": "supposedly called from train()",
        }],
    })
    result = _run(reply, tmp_path, ablation_cmd=_STATIC_CMD)
    # Measurement says the signal did NOT move -> dead wiring; verdict flips.
    assert result.reconciled is False
    (f,) = result.findings_list
    assert f.classification == "exists_not_load_bearing"
    assert f.is_dead is True
    assert f.sub_kind is not None  # a dead finding must carry a sub_kind
    assert f.witness.measured is True
    assert f.witness.value == 0


# --------------------------------------------------------------------------- #
# per-mechanism override map wins over the shared template.
# --------------------------------------------------------------------------- #
def test_per_mechanism_cmd_map_override(tmp_path):
    (tmp_path / "src.py").write_text("# code\n")
    reply = json.dumps({
        "reconciled": True,
        "findings": [{
            "name": "surprise",
            "classification": "load_bearing",
            "location": "src.py:1",
            "witness": {"value": 1.0, "description": ""},
        }],
    })
    # Shared template would MOVE (look load-bearing); the override is STATIC (dead),
    # so the override must be the one that runs -> flip to dead.
    result = _run(
        reply, tmp_path,
        ablation_cmd=_MOVES_CMD,
        ablation_cmd_map={"surprise": _STATIC_CMD},
    )
    (f,) = result.findings_list
    assert f.classification == "exists_not_load_bearing"
    assert f.witness.measured is True


# --------------------------------------------------------------------------- #
# (backward-compat) no --ablation-cmd: witness is the model's self-report,
# unmeasured, byte-identical to the pre-#52 path.
# --------------------------------------------------------------------------- #
def test_no_ablation_cmd_is_self_report_only(tmp_path):
    reply = json.dumps({
        "reconciled": True,
        "findings": [{
            "name": "surprise",
            "classification": "load_bearing",
            "location": "src.py:1",
            "witness": {"value": 3.0, "description": "self-reported"},
        }],
    })
    result = _run(reply, tmp_path)  # no ablation_cmd
    (f,) = result.findings_list
    assert f.classification == "load_bearing"
    assert f.witness.value == 3.0          # untouched self-report
    assert f.witness.measured is False
    # to_dict always carries the measured flag (additive field).
    wdict = result.to_dict()["findings"][0]["witness"]
    assert wdict == {"value": 3.0, "description": "self-reported", "measured": False}


# --------------------------------------------------------------------------- #
# (best-effort) an unmeasurable witness command (no number emitted) leaves the
# self-report intact and never crashes the station.
# --------------------------------------------------------------------------- #
def test_unmeasurable_witness_keeps_self_report(tmp_path):
    (tmp_path / "src.py").write_text("# code\n")
    reply = json.dumps({
        "reconciled": True,
        "findings": [{
            "name": "surprise",
            "classification": "load_bearing",
            "location": "src.py:1",
            "witness": {"value": 4.0, "description": "model claim"},
        }],
    })
    no_number = 'python3 -c "import sys; sys.stdout.write(\'no signal here\')"'
    result = _run(reply, tmp_path, ablation_cmd=no_number)
    (f,) = result.findings_list
    # Could not measure -> self-report preserved, not flipped.
    assert f.classification == "load_bearing"
    assert f.witness.value == 4.0
    assert f.witness.measured is False


# --------------------------------------------------------------------------- #
# the station runs the witness READ-ONLY: the live working_directory is untouched
# even if a (hypothetical) witness wrote files (it runs in a throwaway worktree).
# --------------------------------------------------------------------------- #
def test_witness_runs_in_throwaway_not_live_tree(tmp_path):
    (tmp_path / "src.py").write_text("# code\n")
    reply = json.dumps({
        "reconciled": True,
        "findings": [{
            "name": "surprise",
            "classification": "load_bearing",
            "location": "src.py:1",
            "witness": {"value": 1.0, "description": ""},
        }],
    })
    # This witness drops a marker file in its cwd; if cwd were the live tree the
    # marker would persist there. It must NOT — the witness runs in a copy.
    side_effect = (
        'python3 -c "import os; '
        'open(\'ABLATION_SIDE_EFFECT\',\'w\').close(); '
        'print(0.0 if os.environ.get(\'AGY_ABLATE\') else 1.0)"'
    )
    _run(reply, tmp_path, ablation_cmd=side_effect)
    assert not (tmp_path / "ABLATION_SIDE_EFFECT").exists()

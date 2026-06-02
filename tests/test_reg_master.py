"""Regression tests for agy_orchestrator/workflows/master.py checkpoint loading.

Subsystem: workflows/master.py — plan/ToT/adversarial/checkpoint/compaction.

Hermetic: no network, no real worker CLI. A stub AgentInstance subclass stands
in for any agent. Each test pins a confirmed-and-fixed corrupt-checkpoint defect
so a regression re-introducing the crash is caught.

Covered findings:
  * master-checkpoint-1 — a top-level JSON value that is NOT an object (list,
    number, string, bool) is valid JSON, so json.load succeeds; the subsequent
    ``data.get("key")`` then raised AttributeError and aborted the whole master
    run. It must instead be tolerated like any other corrupt checkpoint
    (logged + None, start fresh).
  * master-checkpoint-2 — a non-integer ``completed`` field, and a non-list
    ``tasks`` field, must likewise be tolerated (return None), never crash.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import pytest

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.workflows.master import MasterWorkflow


class _Stub(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["x"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input=None):
        return ["true"]

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        return "summary-line"


def _make(checkpoint_path):
    return MasterWorkflow(
        model="m", effort="low", branches=1,
        agent_class=_Stub, checkpoint_path=checkpoint_path,
    )


def _write(tmp_path, obj):
    p = str(tmp_path / "ckpt.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return p


def _key(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


# ----- master-checkpoint-1: non-object top-level JSON ----------------------- #
@pytest.mark.parametrize("payload", [[1, 2, 3], 42, "hello", 3.14, True, None])
def test_non_object_toplevel_checkpoint_returns_none(tmp_path, payload):
    p = _write(tmp_path, payload)
    m = _make(p)
    # Must NOT raise AttributeError — tolerated, start fresh.
    assert m._load_checkpoint("p") is None


# ----- master-checkpoint-2: non-integer 'completed' ------------------------- #
@pytest.mark.parametrize("bad", ["NaN", "", "abc", None, [1], {"x": 1}])
def test_non_integer_completed_returns_none(tmp_path, bad):
    p = _write(tmp_path, {
        "key": _key("p"), "tasks": ["a", "b"],
        "completed": bad, "base_fingerprint": None,
    })
    m = _make(p)
    assert m._load_checkpoint("p") is None


# ----- master-checkpoint-2: non-list 'tasks' (e.g. a dict) ------------------ #
@pytest.mark.parametrize("bad_tasks", [{"a": 1}, "notalist", 5])
def test_non_list_tasks_returns_none(tmp_path, bad_tasks):
    p = _write(tmp_path, {
        "key": _key("p"), "tasks": bad_tasks,
        "completed": 0, "base_fingerprint": None,
    })
    m = _make(p)
    # A truthy non-list tasks must not flow downstream into tasks[i].
    assert m._load_checkpoint("p") is None


# ----- positive control: a well-formed resumable checkpoint still resumes --- #
def test_wellformed_checkpoint_resumes(tmp_path):
    p = _write(tmp_path, {
        "key": _key("p"), "tasks": ["a", "b", "c"],
        "completed": 1, "base_fingerprint": None,
    })
    m = _make(p)
    result = m._load_checkpoint("p")
    assert result is not None
    tasks, completed = result[0], result[1]
    assert tasks == ["a", "b", "c"]
    assert completed == 1

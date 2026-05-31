"""Issue #31 — worker process-cleanup must not take down the orchestrator, and
a master/pat run killed mid-step must be salvageable.

Two layers, two fixes:
  * Fix option 2 (remove the trigger): the worker preamble forbids re-running the
    full test/build gate and forbids `pkill`/`kill`-by-name. A worker that ran
    `make check` itself and then pkilled the "stale" jobs by name matched and
    killed the harness scope (exit-144).
  * Fix option 4 (salvage on death): master/pat dispatches checkpoint to a stable,
    instruction-keyed path so re-dispatching the same instruction resumes from the
    last completed step instead of restarting; the checkpoint is removed on a clean
    finish so completed/other runs start fresh.
"""
from __future__ import annotations

import json

from harness.dispatch import (
    CHECKPOINT_DIR,
    WORKER_PREAMBLE,
    _build_prompt,
    _master_checkpoint_path,
)
from agy_orchestrator.workflows.master import MasterWorkflow


# --- Fix option 2: the preamble removes the destructive-cleanup trigger --- #

def test_preamble_forbids_running_the_full_gate():
    text = WORKER_PREAMBLE.lower()
    assert "do not run the full" in text
    assert "make check" in text
    # The harness, not the worker, owns verification.
    assert "harness" in text and "verify" in text


def test_preamble_forbids_pkill_by_name():
    text = WORKER_PREAMBLE.lower()
    assert "pkill" in text
    assert "killall" in text
    # The point: a name/pattern kill can match the orchestrator itself.
    assert "orchestrator" in text


def test_built_prompt_carries_the_process_discipline():
    # The discipline must reach the worker via the assembled prompt, not just live
    # in a constant — every mode builds the worker prompt through _build_prompt.
    prompt = _build_prompt("add a flag to the parser", None)
    assert "pkill" in prompt
    assert "make check" in prompt


# --- Fix option 4: stable, instruction-keyed salvage checkpoint --- #

def test_checkpoint_path_is_stable_per_instruction():
    p1 = _master_checkpoint_path("build feature X")
    p2 = _master_checkpoint_path("build feature X")
    p3 = _master_checkpoint_path("build feature Y")
    assert p1 == p2, "same instruction must map to the same checkpoint (resumable)"
    assert p1 != p3, "different instructions must not collide"
    assert str(CHECKPOINT_DIR) in p1


def _wf(tmp_path, **kw):
    return MasterWorkflow(
        model="m",
        effort="low",
        checkpoint_path=str(tmp_path / "ck.json"),
        **kw,
    )


def test_resume_picks_up_after_last_completed_step(tmp_path):
    wf = _wf(tmp_path)
    prompt = "do the whole thing"
    wf._save_checkpoint(prompt, ["step A", "step B", "step C"], 2, "ctx-so-far", "sess-1")
    resumed = wf._load_checkpoint(prompt)
    assert resumed is not None
    tasks, completed, ctx, sess = resumed
    assert completed == 2  # resumes at step 3 (index 2), not from scratch
    assert tasks == ["step A", "step B", "step C"]
    assert ctx == "ctx-so-far"
    assert sess == "sess-1"


def test_completed_checkpoint_does_not_resume(tmp_path):
    wf = _wf(tmp_path)
    prompt = "finished project"
    wf._save_checkpoint(prompt, ["only step"], 1, "ctx", "sess")
    assert wf._load_checkpoint(prompt) is None


def test_checkpoint_for_a_different_instruction_is_ignored(tmp_path):
    wf = _wf(tmp_path)
    wf._save_checkpoint("instruction A", ["s1", "s2"], 1, "ctx", "sess")
    # A different instruction hashes to a different key -> no accidental resume.
    assert wf._load_checkpoint("instruction B") is None


def test_remove_checkpoint_clears_the_file(tmp_path):
    wf = _wf(tmp_path)
    wf._save_checkpoint("p", ["s1"], 0, "ctx", "sess")
    path = tmp_path / "ck.json"
    assert path.exists()
    # Sanity: it's the structure _load_checkpoint reads back.
    assert "tasks" in json.loads(path.read_text())
    wf._remove_checkpoint()
    assert not path.exists()
    wf._remove_checkpoint()  # idempotent: removing a missing checkpoint is a no-op

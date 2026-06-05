"""Issue #78 — a real codex usage/quota wall is reported ONLY in the rollout JSONL
(``token_count.rate_limits.primary.used_percent``), never on stderr, so the
stderr-only classifier (``is_usage_wall``) misses it: a walled codex exits code 1
with EMPTY stderr and is retried 3x/step + re-led every step for the whole window.

``CodexAgent._augment_failure_stderr`` now reads the rollout out-of-band (located
via the ``thread_id`` from the ``thread.started`` stdout event) and folds a
synthetic ``usage limit`` marker into stderr when used_percent crosses the
threshold, so ``is_usage_wall`` fires and the fallback layer demotes/cycles codex.

These are hermetic: a fake rollout file under a tmp CODEX_HOME, no network/codex.
"""
from __future__ import annotations

import json
import os

import pytest

from agy_orchestrator.core.agent import (
    is_model_unavailable,
    is_usage_wall,
)
from agy_orchestrator.core.agents.codex_agent import (
    CodexAgent,
    codex_usage_wall_marker,
    extract_codex_model_error,
    extract_codex_thread_id,
    find_codex_rollout,
    read_codex_rate_limits,
)

THREAD_ID = "019e9556-3358-7fe0-869f-aef4207e569e"


def _stdout_with_thread(tid: str = THREAD_ID) -> str:
    return "\n".join([
        json.dumps({"type": "thread.started", "thread_id": tid}),
        json.dumps({"type": "turn.started"}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 5}}),
    ])


def _write_rollout(codex_home, tid: str, used_percent: float,
                   limit_name="GPT-5.3-Codex-Spark", resets_at=1780633615) -> str:
    day = os.path.join(codex_home, "sessions", "2026", "06", "04")
    os.makedirs(day, exist_ok=True)
    path = os.path.join(day, f"rollout-2026-06-04T21-05-45-{tid}.jsonl")
    rows = [
        {"type": "event_msg", "payload": {"type": "token_count",
         "rate_limits": {"limit_name": None,
                         "primary": {"used_percent": 0.0, "window_minutes": 300,
                                     "resets_at": resets_at},
                         "secondary": {"used_percent": 0.0, "window_minutes": 10080,
                                       "resets_at": resets_at}}}},
        # The LAST token_count is the one that matters — it reflects the final wall.
        {"type": "event_msg", "payload": {"type": "token_count",
         "rate_limits": {"limit_name": limit_name,
                         "primary": {"used_percent": used_percent, "window_minutes": 300,
                                     "resets_at": resets_at},
                         "secondary": {"used_percent": 12.0, "window_minutes": 10080,
                                       "resets_at": resets_at}}}},
    ]
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def test_extract_thread_id_from_stdout():
    assert extract_codex_thread_id(_stdout_with_thread()) == THREAD_ID
    assert extract_codex_thread_id("not json\nplain text") is None


def test_find_and_read_rollout(tmp_path):
    home = str(tmp_path / ".codex")
    path = _write_rollout(home, THREAD_ID, used_percent=100.0)
    found = find_codex_rollout(THREAD_ID, codex_home=home)
    assert found == path
    rl = read_codex_rate_limits(found)
    assert rl is not None and rl["primary"]["used_percent"] == 100.0
    assert rl["limit_name"] == "GPT-5.3-Codex-Spark"


def test_marker_fires_at_wall_not_below_threshold():
    walled = {"primary": {"used_percent": 100.0, "resets_at": 1}, "limit_name": "spark"}
    fine = {"primary": {"used_percent": 40.0, "resets_at": 1}, "limit_name": "spark"}
    m = codex_usage_wall_marker(walled, threshold=99.0)
    assert m is not None and "usage limit" in m.lower()
    assert is_usage_wall(m)  # the classifier the fallback layer consults
    assert codex_usage_wall_marker(fine, threshold=99.0) is None
    # Secondary (weekly) window walled counts too.
    weekly = {"primary": {"used_percent": 3.0}, "secondary": {"used_percent": 100.0}}
    assert codex_usage_wall_marker(weekly, threshold=99.0) is not None
    # Disabled detector (threshold <= 0) never fires.
    assert codex_usage_wall_marker(walled, threshold=0.0) is None


def test_augment_failure_stderr_injects_marker_when_walled(tmp_path, monkeypatch):
    home = str(tmp_path / ".codex")
    _write_rollout(home, THREAD_ID, used_percent=100.0)
    monkeypatch.setenv("CODEX_HOME", home)
    agent = CodexAgent(prompt="x", model="standard")
    # The exact incident shape: code-1 failure, EMPTY stderr.
    out = agent._augment_failure_stderr(_stdout_with_thread(), "")
    assert is_usage_wall(out), (
        "a real codex wall (rollout used_percent=100) must be classified as a "
        "usage wall even though stderr was empty (issue #78)"
    )
    assert getattr(agent, "usage_wall_resets_at", None) == 1780633615


def test_augment_no_wall_below_threshold_is_noop(tmp_path, monkeypatch):
    home = str(tmp_path / ".codex")
    _write_rollout(home, THREAD_ID, used_percent=37.0)
    monkeypatch.setenv("CODEX_HOME", home)
    agent = CodexAgent(prompt="x", model="standard")
    out = agent._augment_failure_stderr(_stdout_with_thread(), "boom")
    assert out == "boom"
    assert not is_usage_wall(out)


def test_augment_graceful_when_no_thread_or_rollout(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    agent = CodexAgent(prompt="x", model="standard")
    # No thread.started in stdout -> cannot locate rollout -> stderr unchanged.
    assert agent._augment_failure_stderr("plain non-json output", "orig") == "orig"
    # thread present but no rollout file on disk -> still graceful.
    assert agent._augment_failure_stderr(_stdout_with_thread(), "orig") == "orig"


# --- dynamic model-availability detection (operator request) ----------------- #

# The exact codex stdout shape for an inaccessible model (empty stderr).
_UNAVAILABLE_STDOUT = "\n".join([
    json.dumps({"type": "thread.started", "thread_id": THREAD_ID}),
    json.dumps({"type": "turn.started"}),
    json.dumps({"type": "error", "message": json.dumps({
        "type": "error", "status": 400, "error": {
            "type": "invalid_request_error",
            "message": "The 'gpt-5.2' model is not supported when using Codex with a ChatGPT account."}})}),
    json.dumps({"type": "turn.failed", "error": {"message":
        "The 'gpt-5.2' model is not supported when using Codex with a ChatGPT account."}}),
])


def test_extract_codex_model_error():
    msg = extract_codex_model_error(_UNAVAILABLE_STDOUT)
    assert msg is not None and "not supported" in msg
    assert is_model_unavailable(msg)
    # A clean success stream has no error.
    assert extract_codex_model_error(_stdout_with_thread()) is None


def test_augment_folds_model_unavailable_into_stderr():
    agent = CodexAgent(prompt="x", model="gpt-5.2")
    out = agent._augment_failure_stderr(_UNAVAILABLE_STDOUT, "")  # empty stderr, like the real failure
    assert is_model_unavailable(out), (
        "an inaccessible-model 400 (reported only on codex stdout) must classify as "
        "model-unavailable so the fallback layer prunes it"
    )
    # It must NOT be mis-read as a usage wall (a different model would serve).
    assert not is_usage_wall(out)


def test_model_unavailable_distinct_from_usage_wall():
    assert is_model_unavailable("The 'gpt-5.2' model is not supported when using Codex")
    assert not is_model_unavailable("rate limit exceeded")
    assert not is_model_unavailable("usage limit reached")
    # A usage-phrased error never reads as model-unavailable.
    assert not is_model_unavailable("429 too many requests")


def test_detector_disabled_via_env(tmp_path, monkeypatch):
    home = str(tmp_path / ".codex")
    _write_rollout(home, THREAD_ID, used_percent=100.0)
    monkeypatch.setenv("CODEX_HOME", home)
    monkeypatch.setenv("AGY_CODEX_USAGE_WALL_PERCENT", "0")
    agent = CodexAgent(prompt="x", model="standard")
    assert agent._augment_failure_stderr(_stdout_with_thread(), "") == ""

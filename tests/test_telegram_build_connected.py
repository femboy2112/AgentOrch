"""Hermetic tests for the build-CONNECTED Telegram render layer (no network).

These cover the operator-requested upgrade: worker spin-ups + adversarial
rounds now surface at the DEFAULT (``normal``) verbosity, annotated with the
build context (current step + phase) tracked on :class:`TelegramNotifier`.

Everything is hermetic — no real Telegram, no real token/chat id. The HTTP
layer is monkeypatched (``capture_client``) so no request is ever made, and a
fake key/ids are used throughout. Mirrors the fixtures of ``test_telegram.py``.
"""
from __future__ import annotations

import pytest

from harness import telegram as tg


FAKE_TOKEN = "123456:FAKE-TEST-TOKEN-not-real"
FAKE_CHAT_ID = 444555666


@pytest.fixture
def fake_key(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_KEY", FAKE_TOKEN)
    return FAKE_TOKEN


@pytest.fixture
def capture_client(monkeypatch):
    """A TelegramClient whose _post is replaced to capture payloads (no I/O)."""
    sent = []

    def fake_post(self, method, params, *, timeout=None):
        sent.append({"method": method, "params": dict(params)})
        if method == "getUpdates":
            return {"ok": True, "result": []}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(tg.TelegramClient, "_post", fake_post)
    return sent


def _agent_started(worker="codex", model="gpt-5.3-codex-spark", effort="high"):
    # worker/model/effort are stamped on the TOP LEVEL by the event bus, NOT
    # inside data (per dashboard/event_bus._normalize_worker_event).
    return {
        "kind": "lifecycle",
        "run_id": "R",
        "worker": worker,
        "model": model,
        "effort": effort,
        "data": {"event": "agent_started", "detail": {}},
    }


def _adv(action, *, outcome=None, iteration=1, iteration_total=5,
         to_worker=None, model=None):
    orch = {
        "phase": "adversarial",
        "action": action,
        "iteration": iteration,
        "iteration_total": iteration_total,
    }
    if outcome is not None:
        orch["outcome"] = outcome
    if to_worker is not None:
        orch["to_worker"] = to_worker
    if model is not None:
        orch["model"] = model
    return {"kind": "lifecycle", "run_id": "R", "data": {"orchestration": orch}}


# --------------------------------------------------------------------------- #
# Worker spin-up ("a model spun up")
# --------------------------------------------------------------------------- #
def test_spinup_renders_at_normal_with_worker_model_effort():
    msg = tg.render_event(
        _agent_started(), verbosity="normal", mode="master", run_id="R"
    )
    assert msg is not None
    assert "codex" in msg
    assert "gpt-5.3-codex-spark" in msg
    assert "high" in msg


def test_spinup_is_silent_at_quiet():
    msg = tg.render_event(
        _agent_started(), verbosity="quiet", mode="master", run_id="R"
    )
    assert msg is None


# --------------------------------------------------------------------------- #
# Adversarial rounds
# --------------------------------------------------------------------------- #
def test_iteration_started_renders_draft_at_normal():
    msg = tg.render_event(
        _adv("iteration_started", model="gpt-5.3-codex-spark"),
        verbosity="normal", mode="master", run_id="R",
    )
    assert msg is not None
    assert "Draft" in msg
    # carries the generator's model for that round
    assert "gpt-5.3-codex-spark" in msg


def test_iteration_completed_continue_reads_as_revised():
    msg = tg.render_event(
        _adv("iteration_completed", outcome="continue"),
        verbosity="normal", mode="master", run_id="R",
    )
    assert msg is not None
    assert "revised" in msg


@pytest.mark.parametrize("outcome", ["approved", "verified"])
def test_iteration_completed_positive_verdict(outcome):
    msg = tg.render_event(
        _adv("iteration_completed", outcome=outcome),
        verbosity="normal", mode="master", run_id="R",
    )
    assert msg is not None
    assert "approved" in msg
    assert "revised" not in msg


def test_iteration_completed_verifier_timeout_is_infra_glyph():
    msg = tg.render_event(
        _adv("iteration_completed", outcome="verifier_timeout"),
        verbosity="normal", mode="master", run_id="R",
    )
    assert msg is not None
    # infra glyph (⚠️), NOT a benign "revised" verdict
    assert "⚠️" in msg
    assert "revised" not in msg


def test_generator_rotation_carries_target_at_normal():
    msg = tg.render_event(
        _adv("generator_rotation", to_worker="agy"),
        verbosity="normal", mode="master", run_id="R",
    )
    assert msg is not None
    assert "Rotation" in msg
    assert "agy" in msg


# --------------------------------------------------------------------------- #
# Build-context: spin-up after a step start carries the step it belongs to
# --------------------------------------------------------------------------- #
def test_notifier_spinup_carries_step_context(fake_key, capture_client):
    client = tg.TelegramClient()
    notifier = tg.TelegramNotifier(
        run_id="R", mode="master", verbosity="normal",
        client=client, chat_ids=[FAKE_CHAT_ID],
    )
    # Step 2/5 starts ...
    notifier({
        "kind": "lifecycle", "run_id": "R",
        "data": {"orchestration": {
            "phase": "step", "action": "started",
            "step_index": 2, "step_total": 5, "step_title": "Wire the API"}},
    })
    # ... then a worker spins up inside it.
    notifier(_agent_started())
    notifier.flush(timeout=5.0)

    sends = [s for s in capture_client if s["method"] == "sendMessage"]
    # the spin-up is the latest sent message
    spinup = sends[-1]["params"]["text"]
    assert "codex" in spinup  # it really is the spin-up line
    # and it carries the build context (step position or title)
    assert "Step 2/5" in spinup or "Wire the API" in spinup


# --------------------------------------------------------------------------- #
# HTML safety
# --------------------------------------------------------------------------- #
def test_step_title_with_html_and_amp_is_escaped(fake_key, capture_client):
    client = tg.TelegramClient()
    notifier = tg.TelegramNotifier(
        run_id="R", mode="master", verbosity="normal",
        client=client, chat_ids=[FAKE_CHAT_ID],
    )
    notifier({
        "kind": "lifecycle", "run_id": "R",
        "data": {"orchestration": {
            "phase": "step", "action": "started",
            "step_index": 1, "step_total": 2,
            "step_title": "<script>x</script> & y"}},
    })
    notifier.flush(timeout=5.0)
    text = capture_client[-1]["params"]["text"]
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert "&amp; y" in text


def test_pure_render_escapes_context_title(fake_key, capture_client):
    # Drive a spin-up annotated with a malicious step title; the suffix must be
    # escaped (context is interpolated through _e in the render path).
    client = tg.TelegramClient()
    notifier = tg.TelegramNotifier(
        run_id="R", mode="master", verbosity="normal",
        client=client, chat_ids=[FAKE_CHAT_ID],
    )
    notifier({
        "kind": "lifecycle", "run_id": "R",
        "data": {"orchestration": {
            "phase": "step", "action": "started",
            "step_index": 3, "step_total": 9,
            "step_title": "<b>evil</b> & co"}},
    })
    notifier(_agent_started())
    notifier.flush(timeout=5.0)
    spinup = capture_client[-1]["params"]["text"]
    # The step position is safe; ensure no raw injected tag leaked via context.
    assert "<b>evil</b>" not in spinup
    assert "Step 3/9" in spinup


# --------------------------------------------------------------------------- #
# Exception isolation — render_event never raises; notifier never raises
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [None, 42, "string", [], {"kind": "lifecycle"}])
def test_render_event_on_malformed_returns_none_never_raises(bad):
    # Must not raise and must decline unknown/garbage shapes.
    assert tg.render_event(bad, verbosity="normal", mode="m", run_id="R") is None


def test_notifier_call_on_garbage_never_raises(fake_key, capture_client):
    client = tg.TelegramClient()
    notifier = tg.TelegramNotifier(
        run_id="R", mode="master", verbosity="debug",
        client=client, chat_ids=[FAKE_CHAT_ID],
    )
    # None, wrong types, malformed orchestration — none may raise.
    for garbage in (None, 42, "junk", {"kind": "lifecycle", "data": "notadict"},
                    {"data": {"orchestration": "nope"}}, {}):
        notifier(garbage)  # must not raise
    notifier.flush(timeout=5.0)


# --------------------------------------------------------------------------- #
# Spam gate: only the canonical per-CLI-call spin-up renders, NOT the per-turn
# adapter events (dashboard codex 'turn.started' / claude 'message_start'),
# which would otherwise emit many duplicate "spun up" lines for one worker call.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("detail", ["turn.started", "message_start"])
def test_per_turn_agent_started_is_not_rendered(detail):
    ev = _agent_started()
    ev["data"] = {"event": "agent_started", "detail": detail}
    assert tg.render_event(ev, verbosity="normal", mode="master", run_id="R") is None


def test_canonical_spinup_still_renders_with_empty_detail():
    # detail={} (canonical) and detail missing both render the spin-up line.
    for data in ({"event": "agent_started", "detail": {}},
                 {"event": "agent_started"}):
        ev = _agent_started()
        ev["data"] = data
        msg = tg.render_event(ev, verbosity="normal", mode="master", run_id="R")
        assert msg is not None and "spun up" in msg

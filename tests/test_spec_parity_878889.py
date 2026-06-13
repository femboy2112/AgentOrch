from __future__ import annotations

from typing import Any, Dict, List, Optional

from agy_orchestrator.core.agent import AgentInstance

import harness.spec as hs
from harness.effort_overrides import resolve_overrides


class _Stub(AgentInstance):
    def __init__(self, reply: str, *, prompt: str = "", last_provider: Optional[str] = None):
        super().__init__(prompt=prompt)
        self._reply = reply
        if last_provider is not None:
            self.last_provider = last_provider

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


def _recorder_agent(
    reply: str,
    *,
    calls: List[tuple],
    chain,
    overrides: Optional[Dict[str, Dict[str, str]]],
    watchdog_scale: float,
    watchdog_max_bytes: Optional[int],
    post_construct_hook,
) -> _Stub:
    calls.append((tuple(chain), overrides, watchdog_scale, watchdog_max_bytes))
    agent = _Stub(reply)
    if post_construct_hook is not None:
        post_construct_hook(agent, chain[0], {"model": "m", "effort": "high"})
    return agent



def test_spec_overrides_reach_builds(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "RUNS_DIR", tmp_path)
    calls: List[tuple] = []

    def fake_build(
        chain,
        *,
        prompt="",
        fallback=True,
        cycles=2,
        codex_config=None,
        post_construct_hook=None,
        overrides=None,
        watchdog_scale=1.0,
        watchdog_max_bytes=None,
        **_kw,
    ):
        reply = "# DESIGN\nbody" if len(calls) == 0 else "APPROVED"
        return _recorder_agent(
            reply,
            calls=calls,
            chain=chain,
            overrides=overrides,
            watchdog_scale=watchdog_scale,
            watchdog_max_bytes=watchdog_max_bytes,
            post_construct_hook=post_construct_hook,
        )

    monkeypatch.setattr(hs.roles, "build_role_agent", fake_build)

    hs.generate_spec(
        "g",
        architect_overrides={"codex": {"model": "gpt-5.5", "effort": "max"}},
        critic_overrides={"agy": {"effort": "high"}},
        watchdog_scale=2.0,
        watchdog_max_bytes=900000,
        run_id="ov",
    )

    assert len(calls) >= 2
    assert calls[0][0][0] == "codex"
    assert calls[0][1] == {"codex": {"model": "gpt-5.5", "effort": "max"}}
    assert calls[0][2] == 2.0
    assert calls[0][3] == 900000
    assert calls[1][1] == {"agy": {"effort": "high"}}


def test_issue87_resolve_profile_max_for_architect_chain():
    resolved = resolve_overrides(
        generator_chain=["codex", "agy"],
        critic_chain=["agy", "codex"],
        mode="",
        profile="max",
    )
    assert resolved.generator["codex"] == {"model": "gpt-5.5", "effort": "max"}


class _RecorderNotifier:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.finished_meta: Optional[Dict[str, Any]] = None
        self.finished_calls = 0

    def __call__(self, event: Dict[str, Any]) -> None:
        self.events.append(event)

    def finished(self, meta: Dict[str, Any]) -> None:
        self.finished_calls += 1
        self.finished_meta = meta


def test_spec_telegram_notifier_added_and_finished(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "RUNS_DIR", tmp_path)

    def fake_build(
        chain,
        *,
        prompt="",
        fallback=True,
        cycles=2,
        codex_config=None,
        post_construct_hook=None,
        **_kw,
    ):
        reply = "# DESIGN\nbody" if chain[0] == "codex" else "APPROVED"
        agent = _Stub(reply)
        if post_construct_hook is not None:
            post_construct_hook(agent, chain[0], {"model": "m", "effort": "high"})
        return agent

    monkeypatch.setattr(hs.roles, "build_role_agent", fake_build)

    calls = []
    notifier = _RecorderNotifier()

    def fake_notifier_builder(*, run_id, mode, enabled, verbosity, instruction):
        calls.append(
            {
                "run_id": run_id,
                "mode": mode,
                "enabled": enabled,
                "verbosity": verbosity,
                "instruction": instruction,
            }
        )
        return notifier

    import harness.dispatch as hdispatch

    monkeypatch.setattr(hdispatch, "_build_telegram_notifier", fake_notifier_builder)

    hs.generate_spec("goal", run_id="tg1", telegram=True)

    assert calls and calls[0]["mode"] == "spec"
    assert calls[0]["enabled"] is True
    assert notifier.events
    assert notifier.finished_calls == 1
    assert notifier.finished_meta is not None
    assert notifier.finished_meta.get("mode") == "spec"


def test_spec_telegram_notifier_not_constructed_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "RUNS_DIR", tmp_path)

    def fake_build(
        chain,
        *,
        prompt="",
        fallback=True,
        cycles=2,
        codex_config=None,
        post_construct_hook=None,
        **_kw,
    ):
        reply = "# DESIGN\nbody" if chain[0] == "codex" else "APPROVED"
        agent = _Stub(reply)
        if post_construct_hook is not None:
            post_construct_hook(agent, chain[0], {"model": "m", "effort": "high"})
        return agent

    monkeypatch.setattr(hs.roles, "build_role_agent", fake_build)

    calls = []

    def fake_notifier_builder(*, run_id, mode, enabled, verbosity, instruction):
        calls.append(
            {
                "run_id": run_id,
                "mode": mode,
                "enabled": enabled,
                "verbosity": verbosity,
                "instruction": instruction,
            }
        )
        return None

    import harness.dispatch as hdispatch

    monkeypatch.setattr(hdispatch, "_build_telegram_notifier", fake_notifier_builder)

    hs.generate_spec("goal", run_id="tg0", telegram=False)

    assert calls and calls[0]["enabled"] is False
    assert calls[0]["mode"] == "spec"


def _make_architect_chain_builder(last_provider: Optional[str]):
    state = {"i": 0}

    def fake_build(
        chain,
        *,
        prompt="",
        fallback=True,
        cycles=2,
        codex_config=None,
        post_construct_hook=None,
        **_kw,
    ):
        if state["i"] == 0:
            reply = "# DESIGN\nbody"
            agent = _Stub(reply, last_provider=last_provider)
        else:
            reply = "APPROVED"
            agent = _Stub(reply)
        state["i"] += 1
        if post_construct_hook is not None:
            post_construct_hook(agent, chain[0], {"model": "m", "effort": "high"})
        return agent

    return fake_build


def test_spec_architect_demotion_is_surface_and_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "RUNS_DIR", tmp_path)

    monkeypatch.setattr(hs.roles, "build_role_agent", _make_architect_chain_builder("GrokAgent"))
    result = hs.generate_spec(
        "g",
        architect_chain=["codex", "grok"],
        critic_chain=["agy"],
        run_id="dem",
    )
    assert result.architect_author == "grok"
    assert result.architect_demoted is True


def test_spec_architect_non_demoted_when_lead_authored(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "RUNS_DIR", tmp_path)

    monkeypatch.setattr(hs.roles, "build_role_agent", _make_architect_chain_builder("CodexAgent"))
    result = hs.generate_spec(
        "g",
        architect_chain=["codex", "grok"],
        critic_chain=["agy"],
        run_id="nondem",
    )
    assert result.architect_author == "codex"
    assert result.architect_demoted is False


# --- #89: the telegram finish card mirrors the CLI authored-by line --------- #
def test_summary_card_shows_authored_by(tmp_path, monkeypatch):
    """A spec finish card surfaces who authored, and flags a lead demotion."""
    import harness.telegram as tg

    monkeypatch.setenv("AGY_TELEGRAM_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(tmp_path / "users.json"))
    monkeypatch.setenv("TELEGRAM_BOT_KEY", "123456:FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE")

    n = tg.TelegramNotifier(
        run_id="spec1", mode="spec", verbosity="normal",
        client=tg.TelegramClient(), chat_ids=["4242"],
        instruction="design a thing",
    )

    base = {"mode": "spec", "success": True, "duration_s": 12.0,
            "quality": {"confidence": "approved"}}

    # Demoted: the runner-up authored — card names it + the demotion warning.
    demoted = n._summary_card({**base, "architect_author": "grok",
                               "architect_demoted": True})
    assert "authored by" in demoted
    assert "grok" in demoted
    assert "demoted" in demoted

    # Lead authored: card names the author, no demotion flag.
    ok = n._summary_card({**base, "architect_author": "codex",
                          "architect_demoted": False})
    assert "authored by" in ok and "codex" in ok
    assert "demoted" not in ok

    # A `do` run (no architect_author key) renders no authored-by line.
    plain = n._summary_card({"mode": "master", "success": True, "duration_s": 5.0})
    assert "authored by" not in plain

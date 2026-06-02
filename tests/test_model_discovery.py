"""Tests for the shared model-discovery helper + softened model validation (issue #46).

Fully hermetic: every subprocess is stubbed via monkeypatch — NO real CLI call,
NO network. Covers dynamic fetch success, fallback on every failure mode, TTL
memoization (second call within TTL does not re-shell), the list_argv=None path,
and the unknown-codex-model passthrough (no raise).
"""
import asyncio

import pytest

from agy_orchestrator.core import model_discovery
from agy_orchestrator.core.agents.codex_agent import CodexAgent
from agy_orchestrator.core.agents.grok_agent import GrokAgent, _parse_grok_models
from agy_orchestrator.core.agents.claude_agent import ClaudeAgent
from agy_orchestrator.core.agents.agy_agent import AgyAgent


@pytest.fixture(autouse=True)
def _clear_cache():
    model_discovery.reset_cache()
    yield
    model_discovery.reset_cache()


class _FakeProc:
    def __init__(self, stdout=b"", returncode=0, *, hang=False, raise_on_comm=None):
        self._stdout = stdout
        self.returncode = returncode
        self._hang = hang
        self._raise = raise_on_comm
        self.killed = False

    async def communicate(self):
        if self._hang:
            await asyncio.sleep(3600)
        if self._raise is not None:
            raise self._raise
        return self._stdout, b""

    def kill(self):
        self.killed = True


def _patch_spawn(monkeypatch, proc=None, *, raise_on_spawn=None, record=None):
    async def fake_create(*argv, **kwargs):
        if record is not None:
            record.append(argv)
        if raise_on_spawn is not None:
            raise raise_on_spawn
        return proc

    monkeypatch.setattr(
        model_discovery.asyncio, "create_subprocess_exec", fake_create
    )


# --- discover_models: success + every fallback path ------------------------- #

def test_dynamic_fetch_success(monkeypatch):
    proc = _FakeProc(stdout=b"Available models:\n  * model-a\n  * model-b\n")
    _patch_spawn(monkeypatch, proc)
    out = asyncio.run(model_discovery.discover_models(
        "k", list_argv=["x", "models"],
        parse=lambda s: [ln.strip(" *") for ln in s.splitlines() if ln.startswith("  ")],
        fallback=["fb"],
    ))
    assert out == ["model-a", "model-b"]


def test_fallback_on_missing_binary(monkeypatch):
    _patch_spawn(monkeypatch, raise_on_spawn=FileNotFoundError("no such cli"))
    out = asyncio.run(model_discovery.discover_models(
        "k", list_argv=["nope", "models"], parse=lambda s: ["X"], fallback=["fb"],
    ))
    assert out == ["fb"]


def test_fallback_on_nonzero_exit(monkeypatch):
    _patch_spawn(monkeypatch, _FakeProc(stdout=b"boom", returncode=2))
    out = asyncio.run(model_discovery.discover_models(
        "k", list_argv=["x", "models"], parse=lambda s: ["X"], fallback=["fb"],
    ))
    assert out == ["fb"]


def test_fallback_on_empty_output(monkeypatch):
    # cmd succeeds but parser yields nothing -> static fallback.
    _patch_spawn(monkeypatch, _FakeProc(stdout=b"\n\n", returncode=0))
    out = asyncio.run(model_discovery.discover_models(
        "k", list_argv=["x", "models"], parse=lambda s: [], fallback=["fb"],
    ))
    assert out == ["fb"]


def test_fallback_on_parse_error(monkeypatch):
    def boom(_s):
        raise ValueError("bad parse")

    _patch_spawn(monkeypatch, _FakeProc(stdout=b"data", returncode=0))
    out = asyncio.run(model_discovery.discover_models(
        "k", list_argv=["x", "models"], parse=boom, fallback=["fb"],
    ))
    assert out == ["fb"]


def test_fallback_on_timeout(monkeypatch):
    proc = _FakeProc(hang=True)
    _patch_spawn(monkeypatch, proc)
    out = asyncio.run(model_discovery.discover_models(
        "k", list_argv=["x", "models"], parse=lambda s: ["X"],
        fallback=["fb"], timeout=0.05,
    ))
    assert out == ["fb"]
    assert proc.killed is True


def test_fallback_on_communicate_error(monkeypatch):
    # Process spawns, then communicate() raises a non-TimeoutError -> fallback.
    proc = _FakeProc(raise_on_comm=RuntimeError("drain blew up"))
    _patch_spawn(monkeypatch, proc)
    out = asyncio.run(model_discovery.discover_models(
        "k", list_argv=["x", "models"], parse=lambda s: ["X"], fallback=["fb"],
    ))
    assert out == ["fb"]


def test_list_argv_none_skips_subprocess(monkeypatch):
    record = []
    _patch_spawn(monkeypatch, _FakeProc(stdout=b"X"), record=record)
    out = asyncio.run(model_discovery.discover_models(
        "k", list_argv=None, parse=lambda s: ["X"], fallback=["fb"],
    ))
    assert out == ["fb"]
    assert record == []  # never shelled out


# --- TTL memoization -------------------------------------------------------- #

def test_ttl_caches_and_does_not_reshell(monkeypatch):
    record = []
    _patch_spawn(
        monkeypatch,
        _FakeProc(stdout=b"Available models:\n  * m1\n"),
        record=record,
    )
    parse = lambda s: [ln.strip(" *") for ln in s.splitlines() if ln.startswith("  ")]

    async def go():
        a = await model_discovery.discover_models(
            "memo", list_argv=["x", "models"], parse=parse, fallback=["fb"], ttl=600)
        b = await model_discovery.discover_models(
            "memo", list_argv=["x", "models"], parse=parse, fallback=["fb"], ttl=600)
        return a, b

    a, b = asyncio.run(go())
    assert a == b == ["m1"]
    assert len(record) == 1  # second call hit the cache, did not re-shell


def test_reset_cache_forces_reshell(monkeypatch):
    record = []
    _patch_spawn(monkeypatch, _FakeProc(stdout=b"  * m1\n"), record=record)
    parse = lambda s: [ln.strip(" *") for ln in s.splitlines() if ln.startswith("  ")]
    asyncio.run(model_discovery.discover_models(
        "rk", list_argv=["x", "models"], parse=parse, fallback=["fb"]))
    model_discovery.reset_cache("rk")
    asyncio.run(model_discovery.discover_models(
        "rk", list_argv=["x", "models"], parse=parse, fallback=["fb"]))
    assert len(record) == 2


def test_concurrent_callers_shell_out_once(monkeypatch):
    # Two callers racing the same cli_key must share a single shell-out: the
    # per-key lock + double-check-under-lock collapses the stampede to one spawn.
    record = []

    class _SlowProc(_FakeProc):
        async def communicate(self):
            await asyncio.sleep(0.05)  # hold the lock long enough to race
            return self._stdout, b""

    _patch_spawn(monkeypatch, _SlowProc(stdout=b"  * m1\n"), record=record)
    parse = lambda s: [ln.strip(" *") for ln in s.splitlines() if ln.startswith("  ")]

    async def go():
        return await asyncio.gather(
            model_discovery.discover_models(
                "race", list_argv=["x", "models"], parse=parse, fallback=["fb"]),
            model_discovery.discover_models(
                "race", list_argv=["x", "models"], parse=parse, fallback=["fb"]),
        )

    a, b = asyncio.run(go())
    assert a == b == ["m1"]
    assert len(record) == 1  # only one shell-out despite two concurrent callers


def test_expired_ttl_reshells(monkeypatch):
    record = []
    _patch_spawn(monkeypatch, _FakeProc(stdout=b"  * m1\n"), record=record)
    parse = lambda s: [ln.strip(" *") for ln in s.splitlines() if ln.startswith("  ")]
    asyncio.run(model_discovery.discover_models(
        "ek", list_argv=["x", "models"], parse=parse, fallback=["fb"], ttl=0.0))
    # ttl=0 means the entry is already expired by the next monotonic tick.
    asyncio.run(model_discovery.discover_models(
        "ek", list_argv=["x", "models"], parse=parse, fallback=["fb"], ttl=0.0))
    assert len(record) == 2


# --- adapter wiring --------------------------------------------------------- #

def test_grok_adapter_dynamic(monkeypatch):
    _patch_spawn(monkeypatch, _FakeProc(
        stdout=b"Available models:\n  * grok-build\n  * grok-fast\n"))
    out = asyncio.run(GrokAgent.get_available_models())
    assert out == ["grok-build", "grok-fast"]


def test_grok_adapter_fallback(monkeypatch):
    _patch_spawn(monkeypatch, raise_on_spawn=FileNotFoundError())
    out = asyncio.run(GrokAgent.get_available_models())
    assert out == ["grok-build"]


def test_grok_parser_filters_headers():
    stdout = "Available models:\n  * grok-build\nLogged in as foo\n"
    assert _parse_grok_models(stdout) == ["grok-build"]


def test_grok_parser_keeps_model_named_substring():
    # A bullet model name containing the substring "model" must NOT be dropped:
    # only the literal header/footer lines are filtered (anchored, not substring).
    stdout = "Available models:\n  * grok-model-x\n  * grok-build\nLogged in as foo\n"
    assert _parse_grok_models(stdout) == ["grok-model-x", "grok-build"]


def test_codex_adapter_static_fallback_no_shell(monkeypatch):
    record = []
    _patch_spawn(monkeypatch, _FakeProc(stdout=b"X"), record=record)
    out = asyncio.run(CodexAgent.get_available_models())
    assert out == list(CodexAgent.AVAILABLE_MODELS)
    assert record == []  # codex has no models subcommand; never shells out


def test_claude_adapter_static_fallback_no_shell(monkeypatch):
    record = []
    _patch_spawn(monkeypatch, _FakeProc(stdout=b"X"), record=record)
    out = asyncio.run(ClaudeAgent.get_available_models())
    assert out == ["haiku", "sonnet", "opus"]
    assert record == []


def test_agy_adapter_static_fallback_no_shell(monkeypatch):
    record = []
    _patch_spawn(monkeypatch, _FakeProc(stdout=b"X"), record=record)
    out = asyncio.run(AgyAgent.get_available_models())
    assert "Gemini 3.1 Pro (High)" in out
    assert record == []


# --- softened validation: unknown model passthrough, no raise --------------- #

def test_unknown_codex_model_passthrough_no_raise():
    from harness import effort_overrides as eo
    notes = []
    # Does not raise; returns the value unchanged and records an advisory note.
    out = eo.validate_model("codex", "gpt-renamed-99", where="--codex-model", notes=notes)
    assert out == "gpt-renamed-99"
    assert any("gpt-renamed-99" in n for n in notes)


def test_known_codex_model_unchanged_no_note():
    from harness import effort_overrides as eo
    notes = []
    known = CodexAgent.AVAILABLE_MODELS[0]
    out = eo.validate_model("codex", known, where="--codex-model", notes=notes)
    assert out == known
    assert notes == []  # a known model is silent — existing behavior preserved

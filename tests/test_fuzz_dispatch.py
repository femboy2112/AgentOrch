"""Hermetic fuzz / property / edge-case tests for the dispatch orchestration glue.

Targets the PURE, side-effect-free functions in ``harness/dispatch.py`` plus the
``dashboard.event_bus.EventBus`` fanout and ``agy_orchestrator.execution.pipeline``
helpers. Everything here runs with fakes / temp files only — NO real worker CLI is
spawned and NO network is touched (HARD RULE). Each test is well under a second.

These assert CORRECT, robust behaviour and all PASS. Genuine defects found while
fuzzing are reported separately as findings with their own repros (a known bug is
NOT committed as a failing test).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from harness.dispatch import (
    _as_int,
    _cache_read_ratio,
    _clean_chain_steps,
    _decide_reconcile_status,
    _derive_verifier_delta,
    _glob_to_regex,
    _summarize_token_usage,
    evaluate_path_policy,
    load_plan,
    load_plan_steps,
    plan_file_sha256,
)
from agy_orchestrator.execution.graph_plan import ChainPlan
from agy_orchestrator.execution.pipeline import LinearPipeline, ParallelSwarm
from agy_orchestrator.core.agent import AgentInstance
from dashboard.event_bus import EventBus, _normalize_worker_event, _to_branch


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class _FakeAgent(AgentInstance):
    """Hermetic AgentInstance double: never spawns a process or hits the net."""

    def __init__(self, *, out="ok", raises=False, echo_input=False):
        self._out = out
        self._raises = raises
        self._echo = echo_input

    async def run_async(self, piped_input=None):
        if self._raises:
            raise RuntimeError(f"boom:{self._out}")
        if self._echo:
            return f"{self._out}<-{piped_input}"
        return self._out

    # Abstract-method stubs (never invoked in these tests).
    def build_command(self):  # pragma: no cover - not exercised
        return []

    def get_available_models(self):  # pragma: no cover
        return []

    def get_model_usage(self):  # pragma: no cover
        return {}


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# _glob_to_regex / evaluate_path_policy                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "glob,path,matches",
    [
        ("*.py", "a.py", True),
        ("*.py", "sub/a.py", False),          # * stays within one segment
        ("**/*.py", "a.py", True),            # **/ matches zero leading dirs
        ("**/*.py", "x/y/a.py", True),
        ("**", "anything/at/all", True),
        ("src/**", "src/a/b.c", True),
        ("?.txt", "a.txt", True),
        ("?.txt", "ab.txt", False),
        ("./a.py", "a.py", True),             # leading ./ is stripped
    ],
)
def test_glob_to_regex_semantics(glob, path, matches):
    assert bool(_glob_to_regex(glob).match(path)) is matches


def test_glob_regex_metachars_are_literal():
    # A glob containing regex metachars must be treated literally, not as a
    # regex (a '.' is a literal dot, '+' / '(' are literals).
    pat = _glob_to_regex("a+b(c).py")
    assert pat.match("a+b(c).py")
    assert not pat.match("aXbYc.py")


def test_evaluate_path_policy_protect_and_allow():
    paths = ["src/a.py", "secrets/key.pem", "docs/readme.md"]
    v = evaluate_path_policy(paths, protect_globs=["secrets/**"], allow_globs=None)
    assert [e["path"] for e in v] == ["secrets/key.pem"]

    # Allowlist: anything outside every allow glob is a violation.
    v2 = evaluate_path_policy(paths, protect_globs=None, allow_globs=["src/**"])
    assert sorted(e["path"] for e in v2) == ["docs/readme.md", "secrets/key.pem"]


def test_evaluate_path_policy_empty_and_whitespace_globs_are_noops():
    # Empty lists, blank-string globs, and None globs must never raise and must
    # produce no violations (a whitespace-only glob is dropped, not matched as ^$).
    assert evaluate_path_policy(["x"], protect_globs=[], allow_globs=[]) == []
    assert evaluate_path_policy(["x"], protect_globs=["   "], allow_globs=None) == []
    assert evaluate_path_policy([], protect_globs=["**"], allow_globs=None) == []
    assert evaluate_path_policy(["x"], protect_globs=None, allow_globs=None) == []


def test_evaluate_path_policy_normalizes_separators():
    # Windows-style separators in a snapshot path still match a POSIX glob.
    v = evaluate_path_policy(["secrets\\k.pem"], protect_globs=["secrets/**"])
    assert v and v[0]["path"] == "secrets\\k.pem"


def test_evaluate_path_policy_protect_wins_over_allow():
    # A path that matches BOTH a protect and an allow glob is still a violation
    # (denylist is evaluated first and short-circuits).
    v = evaluate_path_policy(
        ["src/secret.py"], protect_globs=["**/secret.py"], allow_globs=["src/**"]
    )
    assert len(v) == 1 and "protected" in v[0]["reason"]


# --------------------------------------------------------------------------- #
# _as_int / _cache_read_ratio                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        (5, 5),
        (0, 0),
        ("7", 7),
        (-3, None),            # negative collapses to None
        (None, None),
        (True, None),          # bool is explicitly excluded
        (False, None),
        (float("nan"), None),  # NaN can't become an int -> None, not a crash
        (float("inf"), None),
        ("not a number", None),
        (3.9, 3),
    ],
)
def test_as_int_robust(value, expected):
    assert _as_int(value) == expected


@pytest.mark.parametrize(
    "cache,inp,expected",
    [
        (50, 50, 0.5),
        (0, 100, 0.0),
        (100, 0, 1.0),
        (None, 5, None),
        (5, None, None),
        (0, 0, None),          # zero denominator -> None, not ZeroDivisionError
    ],
)
def test_cache_read_ratio(cache, inp, expected):
    assert _cache_read_ratio(cache, inp) == expected


# --------------------------------------------------------------------------- #
# _summarize_token_usage — corrupt / adversarial events.jsonl                  #
# --------------------------------------------------------------------------- #
def test_summarize_token_usage_skips_garbage_lines(tmp_path):
    p = _write(
        tmp_path,
        "events.jsonl",
        "\n".join(
            [
                "not json at all",
                "",
                "   ",
                "[1,2,3]",  # valid json, not a dict -> skipped
                "42",       # valid json, not a dict -> skipped
                json.dumps({"kind": "message"}),  # not a usage event
                json.dumps({"kind": "usage", "data": {"usage_kind": "summary"}}),  # wrong sub-kind
                json.dumps(
                    {
                        "kind": "usage",
                        "worker": "codex",
                        "data": {"usage_kind": "call", "input_tokens": 10, "output_tokens": 3},
                    }
                ),
            ]
        ),
    )
    out = _summarize_token_usage(p)
    assert out["total_calls"] == 1
    assert out["per_worker"]["codex"]["input_tokens"] == 10
    assert out["grand_total"]["input_tokens"] == 10


def test_summarize_token_usage_nan_inf_tokens_do_not_crash(tmp_path):
    # NaN / Infinity in a token field (json.loads accepts them) must be dropped
    # cleanly via _as_int, never propagate to a ValueError in the ratio math.
    p = _write(
        tmp_path,
        "events.jsonl",
        "\n".join(
            [
                '{"kind":"usage","worker":"w","data":{"usage_kind":"call","input_tokens":NaN,"total_tokens":Infinity}}',
                '{"kind":"usage","worker":"w","data":{"usage_kind":"call","input_tokens":4,"cache_read_tokens":4}}',
            ]
        ),
    )
    out = _summarize_token_usage(p)
    # First call's NaN/Inf tokens dropped; only the clean call counts.
    assert out["total_calls"] == 2
    w = out["per_worker"]["w"]
    assert w["input_tokens"] == 4
    assert w["cache_read_tokens"] == 4
    assert w["cache_read_ratio"] == 0.5


def test_summarize_token_usage_empty_file(tmp_path):
    p = _write(tmp_path, "events.jsonl", "")
    out = _summarize_token_usage(p)
    assert out["total_calls"] == 0
    assert out["per_worker"] == {}
    assert out["grand_total"]["input_tokens"] is None


def test_summarize_token_usage_mixed_token_source(tmp_path):
    p = _write(
        tmp_path,
        "events.jsonl",
        "\n".join(
            [
                json.dumps({"kind": "usage", "worker": "w", "data": {"usage_kind": "call", "token_source": "cli", "input_tokens": 1}}),
                json.dumps({"kind": "usage", "worker": "w", "data": {"usage_kind": "call", "token_source": "unavailable", "input_tokens": 1}}),
            ]
        ),
    )
    out = _summarize_token_usage(p)
    assert out["per_worker"]["w"]["token_source"] == "mixed"


def test_summarize_token_usage_control_chars_and_unicode(tmp_path):
    # Embedded unicode + control chars in a worker name must not crash; the row
    # is still aggregated under that (stringified) key.
    weird = "wörker " + chr(27) + "[31m"
    p = _write(
        tmp_path,
        "events.jsonl",
        json.dumps(
            {"kind": "usage", "worker": weird, "data": {"usage_kind": "call", "input_tokens": 2}}
        ),
    )
    out = _summarize_token_usage(p)
    assert out["total_calls"] == 1
    assert sum(r["input_tokens"] for r in out["per_worker"].values()) == 2


# --------------------------------------------------------------------------- #
# load_plan / load_plan_steps / _clean_chain_steps                            #
# --------------------------------------------------------------------------- #
def test_load_plan_bare_list(tmp_path):
    p = _write(tmp_path, "plan.json", json.dumps(["step one", "step two"]))
    plan = load_plan(p)
    assert isinstance(plan, ChainPlan)
    assert plan.steps == ["step one", "step two"]
    assert load_plan_steps(p) == ["step one", "step two"]


def test_load_plan_steps_object(tmp_path):
    p = _write(tmp_path, "plan.json", json.dumps({"steps": ["a", "b"]}))
    assert load_plan(p).steps == ["a", "b"]


@pytest.mark.parametrize(
    "content",
    [
        "42",                       # bare number
        "null",
        "true",
        '"a string"',
        "NaN",                      # float, not a list/dict
        "Infinity",
        "{}",                       # dict with neither nodes nor steps
        '{"foo": 1}',
    ],
)
def test_load_plan_wrong_shape_raises_valueerror(tmp_path, content):
    p = _write(tmp_path, "plan.json", content)
    with pytest.raises(ValueError):
        load_plan(p)


@pytest.mark.parametrize(
    "content",
    [
        "[]",                       # empty list
        '{"steps": []}',            # empty steps
        '[""]',                     # blank step
        '["   "]',                  # whitespace-only step
        '[123]',                    # non-string step
        '[{"x": 1}]',               # dict step
        '[null]',                   # null step
        '[["nested"]]',             # nested-list step
    ],
)
def test_load_plan_bad_steps_raises_valueerror(tmp_path, content):
    p = _write(tmp_path, "plan.json", content)
    with pytest.raises(ValueError):
        load_plan(p)


def test_load_plan_both_nodes_and_steps_raises(tmp_path):
    p = _write(tmp_path, "plan.json", json.dumps({"nodes": [], "steps": ["a"]}))
    with pytest.raises(ValueError):
        load_plan(p)


def test_load_plan_missing_file_raises_valueerror(tmp_path):
    with pytest.raises(ValueError):
        load_plan(tmp_path / "does-not-exist.json")


def test_load_plan_truncated_json_raises_valueerror(tmp_path):
    p = _write(tmp_path, "plan.json", '["a", "b"')  # truncated
    with pytest.raises(ValueError):
        load_plan(p)


def test_load_plan_empty_file_raises_valueerror(tmp_path):
    p = _write(tmp_path, "plan.json", "")
    with pytest.raises(ValueError):
        load_plan(p)


def test_load_plan_preserves_unicode_and_control_steps(tmp_path):
    steps = ["café ✓ step", "tab\tand newline-free"]
    p = _write(tmp_path, "plan.json", json.dumps(steps))
    assert load_plan(p).steps == steps


def test_load_plan_huge_step_list(tmp_path):
    steps = [f"step {i}" for i in range(5000)]
    p = _write(tmp_path, "plan.json", json.dumps(steps))
    assert len(load_plan(p).steps) == 5000


def test_clean_chain_steps_direct():
    assert _clean_chain_steps(["a", "b"], "x") == ["a", "b"]
    for bad in (None, [], "notalist", [1], [""], [None]):
        with pytest.raises(ValueError):
            _clean_chain_steps(bad, "x")


# --------------------------------------------------------------------------- #
# plan_file_sha256                                                             #
# --------------------------------------------------------------------------- #
def test_plan_file_sha256_stable(tmp_path):
    p = _write(tmp_path, "plan.json", '["a"]')
    import hashlib

    assert plan_file_sha256(p) == hashlib.sha256(b'["a"]').hexdigest()


def test_plan_file_sha256_missing_raises_valueerror(tmp_path):
    with pytest.raises(ValueError):
        plan_file_sha256(tmp_path / "nope.json")


# --------------------------------------------------------------------------- #
# _decide_reconcile_status                                                     #
# --------------------------------------------------------------------------- #
def test_decide_reconcile_status_matrix():
    f = _decide_reconcile_status
    assert f(False, False, True, "warn", True) == "skipped:not_enabled"
    assert f(True, True, True, "warn", True) == "skipped:plan_only"
    assert f(True, False, False, "warn", True) == "skipped:no_output"
    assert f(True, False, True, "fail", False) == "skipped:verifier_not_green"
    # fail disposition but verifier green -> run
    assert f(True, False, True, "fail", True) == "run"
    # warn disposition runs even when verifier not green (read-only trace)
    assert f(True, False, True, "warn", False) == "run"


# --------------------------------------------------------------------------- #
# _derive_verifier_delta                                                       #
# --------------------------------------------------------------------------- #
class _VR:
    def __init__(self, ok):
        self.ok = ok


def test_derive_verifier_delta():
    assert _derive_verifier_delta(None, True) is None
    assert _derive_verifier_delta(_VR(True), True) == "preserved"
    assert _derive_verifier_delta(_VR(True), False) == "regressed"
    assert _derive_verifier_delta(_VR(False), True) == "fixed"
    assert _derive_verifier_delta(_VR(False), False) == "unchanged"


# --------------------------------------------------------------------------- #
# EventBus — sink fanout, raising sink, removal, ids                           #
# --------------------------------------------------------------------------- #
def test_eventbus_sink_fanout_and_raising_sink_isolated():
    bus = EventBus()
    seen_a = []
    seen_b = []

    def good(ev):
        seen_a.append(ev)

    def bad(ev):
        raise RuntimeError("sink failure must not break the bus")

    bus.add_sink("r", bad)   # a sink that always raises
    bus.add_sink("r", good)
    bus.add_sink("r", bad)
    pub = bus.publisher_for("r", worker="w", model="m", effort="e")
    pub({"kind": "message", "text": "hi"})
    # The raising sink is swallowed; the good sink still received the event.
    assert len(seen_a) == 1
    assert seen_a[0]["text"] == "hi"
    # The queue still recorded the event (not dropped by the sink failure).
    assert len(bus.queues["r"]) == 1


def test_eventbus_publish_after_close_is_dropped():
    bus = EventBus()
    pub = bus.publisher_for("r", worker="w", model="m", effort="e")
    pub({"kind": "message"})
    bus.close("r")
    pub({"kind": "message"})  # dropped: run is closed
    assert len(bus.queues["r"]) == 1


def test_eventbus_event_ids_monotonic_under_default_publish():
    bus = EventBus()
    pub = bus.publisher_for("r", worker="w", model="m", effort="e")
    for _ in range(5):
        pub({"kind": "message"})
    ids = [e["_event_id"] for e in bus.queues["r"]]
    assert ids == [0, 1, 2, 3, 4]


def test_eventbus_unknown_kind_demoted_to_stderr():
    ev = _normalize_worker_event(
        {"kind": "totally-made-up", "data": {"a": 1}},
        run_id="r", worker="w", model=None, effort=None, branch=None,
    )
    assert ev["kind"] == "stderr"


def test_eventbus_normalize_non_dict_data():
    ev = _normalize_worker_event(
        {"kind": "usage", "data": [1, 2, 3]},
        run_id="r", worker="w", model=None, effort=None, branch=None,
    )
    assert ev["data"] == {}


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (3, 3),
        (True, 1),
        ("5", 5),
        ("bad", None),
        (float("nan"), None),
        (float("inf"), None),
    ],
)
def test_to_branch_robust(value, expected):
    assert _to_branch(value) == expected


def test_eventbus_subscribe_replays_buffered_then_yields_live():
    bus = EventBus()
    pub = bus.publisher_for("r", worker="w", model="m", effort="e")
    pub({"kind": "message", "text": "first"})

    async def drive():
        agen = bus.subscribe("r")
        first = await agen.__anext__()
        pub({"kind": "message", "text": "second"})
        bus.close("r")
        rest = [e async for e in agen]
        return first, rest

    first, rest = asyncio.run(drive())
    assert first["text"] == "first"
    assert [e["text"] for e in rest] == ["second"]
    # The yielded clean events never leak the internal _event_id.
    assert "_event_id" not in first


def test_eventbus_replay_jsonl_tolerates_corrupt_lines(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text(
        "\n".join(
            [
                "garbage not json",
                "",
                "[1,2]",  # json but not a dict
                json.dumps({"kind": "message", "run_id": "r", "worker": "w", "text": "ok"}),
            ]
        ),
        encoding="utf-8",
    )
    out = EventBus.replay_jsonl(p)
    assert len(out) == 1 and out[0]["text"] == "ok"


def test_eventbus_replay_missing_file_returns_empty(tmp_path):
    assert EventBus.replay_jsonl(tmp_path / "absent.jsonl") == []
    assert EventBus.replay_events(tmp_path / "absent.jsonl") == []


# --------------------------------------------------------------------------- #
# Pipeline — LinearPipeline / ParallelSwarm                                    #
# --------------------------------------------------------------------------- #
def test_linear_pipeline_pipes_input():
    lp = LinearPipeline([_FakeAgent(out="A", echo_input=True),
                         _FakeAgent(out="B", echo_input=True)])
    out = asyncio.run(lp.execute("seed"))
    assert out == "B<-A<-seed"


def test_linear_pipeline_empty_returns_seed():
    lp = LinearPipeline([])
    assert asyncio.run(lp.execute("seed")) == "seed"
    assert asyncio.run(lp.execute()) is None


def test_parallel_swarm_returns_only_survivors():
    sw = ParallelSwarm([_FakeAgent(out="x", raises=True), _FakeAgent(out="good")])
    assert asyncio.run(sw.execute("in")) == ["good"]


def test_parallel_swarm_empty_returns_empty_list():
    assert asyncio.run(ParallelSwarm([]).execute()) == []


def test_parallel_swarm_all_fail_reraises_first():
    sw = ParallelSwarm([_FakeAgent(out="a", raises=True),
                        _FakeAgent(out="b", raises=True)])
    with pytest.raises(RuntimeError):
        asyncio.run(sw.execute())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

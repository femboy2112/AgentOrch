"""C4 tests — CLI: `serve` / `queue` / `do` routing (additive, opt-in).

Hermetic: `dispatch` / `dispatch_async` are monkeypatched (never a real worker
call), the broker is faked via `harness.cli.broker_client`, and no real socket is
bound. The headline test is the BACKWARD-COMPAT guarantee — `do` with neither
--queue nor --direct and no broker running calls the real `dispatch` with exactly
the same args it would have today.
"""
from __future__ import annotations

import pytest

from harness import cli
from harness.dispatch import DispatchResult


def _result(**over) -> DispatchResult:
    base = dict(
        run_id="RID-1",
        run_dir="runs/RID-1",
        mode="adversarial",
        generator="codex",
        critic="agy",
        success=True,
        duration_s=1.0,
    )
    base.update(over)
    return DispatchResult(**base)


# --------------------------------------------------------------------------- #
# arg parsing
# --------------------------------------------------------------------------- #
def test_serve_parses_with_cap():
    ns = _parse(["serve", "--cap", "3"])
    assert ns.command == "serve"
    assert ns.cap == 3


def test_serve_cap_defaults_none():
    ns = _parse(["serve"])
    assert ns.cap is None


def test_queue_parses():
    ns = _parse(["queue"])
    assert ns.command == "queue"


def test_do_queue_flag_parses():
    ns = _parse(["do", "hi", "--queue"])
    assert ns.queue is True
    assert ns.direct is False
    assert ns.detach is False


def test_do_direct_flag_parses():
    ns = _parse(["do", "hi", "--direct"])
    assert ns.direct is True
    assert ns.queue is False


def test_do_detach_flag_parses():
    ns = _parse(["do", "hi", "--queue", "--detach"])
    assert ns.detach is True


def test_do_queue_and_direct_mutually_exclusive():
    # argparse mutually-exclusive group -> SystemExit at parse time.
    with pytest.raises(SystemExit):
        _parse(["do", "hi", "--queue", "--direct"])


def test_do_defaults_no_routing_flags():
    ns = _parse(["do", "hi"])
    assert ns.queue is False
    assert ns.direct is False
    assert ns.detach is False


def _parse(argv):
    import argparse

    # Reconstruct the parser by calling main() up to parse — easiest is to build a
    # throwaway parser identical to main()'s. Instead, drive main() with a func
    # that records the namespace and short-circuits.
    captured = {}

    def _spy(args):
        captured["ns"] = args
        return 0

    # Patch every subcommand func to the spy so parse + dispatch returns the ns.
    import harness.cli as _cli

    saved = (_cli._cmd_do, _cli._cmd_serve, _cli._cmd_queue, _cli._cmd_runs,
             _cli._cmd_show, _cli._cmd_spec, _cli._cmd_dashboard)
    _cli._cmd_do = _spy
    _cli._cmd_serve = _spy
    _cli._cmd_queue = _spy
    _cli._cmd_runs = _spy
    _cli._cmd_show = _spy
    _cli._cmd_spec = _spy
    _cli._cmd_dashboard = _spy
    try:
        _cli.main(argv)
    finally:
        (_cli._cmd_do, _cli._cmd_serve, _cli._cmd_queue, _cli._cmd_runs,
         _cli._cmd_show, _cli._cmd_spec, _cli._cmd_dashboard) = saved
    return captured["ns"]


# --------------------------------------------------------------------------- #
# BACKWARD COMPAT — the load-bearing test
# --------------------------------------------------------------------------- #
def _expected_default_kwargs():
    """The exact kwargs a plain `do "hello"` forwards to dispatch() today."""
    return dict(
        mode="adversarial",
        context=None,
        generator_chain=None,
        critic_chain=None,
        fallback=True,
        cycles=2,
        max_iterations=5,
        branches=3,
        test_cmd=None,
        verifier_mem_max=None,
        candidate_setup=None,
        git_pr=False,
        resume_policy="auto",
        protect_paths=None,
        allow_paths=None,
        plan_only=False,
        plan_steps=None,
        plan_source=None,
        plan_expect_sha=None,
        plan_graph=None,
        max_parallel_nodes=None,
        merge_policy="reconcile",
        run_stall_abort=None,
        notify=None,
        notify_cmd=None,
        heartbeat_interval=None,
        telegram_enabled=None,
        telegram_verbosity=None,
        gen_effort=None,
        gen_model=None,
        critic_effort=None,
        critic_model=None,
        architect_effort=None,
        architect_model=None,
        codex_model=None,
        effort_map=None,
        model_map=None,
        effort_profile=None,
        watchdog_scale=None,
        watchdog_max_bytes=None,
        watchdog_stall=None,
        max_parallel_workers=None,
        worker_mem_max=None,
        baseline_gate=False,
        reconcile=False,
        reconcile_disposition=None,
        ablation_cmd=None,
        web_search=False,
        mission_critical=False,
        spec=None,
        out_dir=None,
        computer_use_mode=None,
        computer_use_task_priority=None,
        computer_use_budgets=None,
        real_gui_policy=None,
        ask_mode=None,
        browser_engine="bing",
        browser_display=None,
    )


def test_do_no_flags_no_broker_calls_dispatch_unchanged(monkeypatch):
    """`do` with neither flag and no broker running -> the real dispatch with the
    SAME args it would have today (the byte-for-byte backward-compat invariant)."""
    calls = {}

    def fake_dispatch(instruction, **kwargs):
        calls["instruction"] = instruction
        calls["kwargs"] = kwargs
        return _result()

    # No broker reachable -> auto routing must pick the local path.
    monkeypatch.setattr(cli, "dispatch", fake_dispatch)
    monkeypatch.setattr(cli.broker_client, "is_running", lambda *a, **k: False)

    rc = cli.main(["do", "hello world"])
    assert rc == 0
    assert calls["instruction"] == "hello world"
    assert calls["kwargs"] == _expected_default_kwargs()


def test_do_direct_runs_local_even_when_broker_up(monkeypatch):
    """--direct forces local dispatch even if a broker is reachable; the broker
    must NOT be consulted for submit/is_running on the direct path."""
    calls = {}

    def fake_dispatch(instruction, **kwargs):
        calls["kwargs"] = kwargs
        return _result()

    def boom(*a, **k):
        raise AssertionError("broker must not be touched on --direct")

    monkeypatch.setattr(cli, "dispatch", fake_dispatch)
    monkeypatch.setattr(cli.broker_client, "is_running", lambda *a, **k: True)
    monkeypatch.setattr(cli.broker_client, "submit", boom)

    rc = cli.main(["do", "hello", "--direct"])
    assert rc == 0
    assert calls["kwargs"] == _expected_default_kwargs()


# --------------------------------------------------------------------------- #
# routing -> broker
# --------------------------------------------------------------------------- #
def test_do_auto_routes_to_broker_when_running(monkeypatch):
    """Auto path: broker reachable -> submit + wait + print the SAME result card.
    dispatch() (local) must NOT be called."""
    submitted = {}

    def fake_submit(instruction, kwargs, pools):
        submitted["instruction"] = instruction
        submitted["kwargs"] = kwargs
        submitted["pools"] = pools
        return "JOB-1"

    def fake_status(job_id):
        return {
            "id": job_id, "status": "done", "run_id": None,
            "result": {"success": True, "run_id": None, "error": None},
            "kwargs": {"mode": "adversarial"}, "instruction": "hello",
        }

    monkeypatch.setattr(cli, "dispatch", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("local dispatch must not run when routed to broker")))
    monkeypatch.setattr(cli.broker_client, "is_running", lambda *a, **k: True)
    monkeypatch.setattr(cli.broker_client, "submit", fake_submit)
    monkeypatch.setattr(cli.broker_client, "status", fake_status)

    rc = cli.main(["do", "hello"])
    assert rc == 0
    assert submitted["instruction"] == "hello"
    # default chains -> codex/agy/grok pools, sorted.
    assert submitted["pools"] == ["agy", "codex", "grok"]
    # json-safe projection: run_id stripped, no GraphPlan object.
    assert "run_id" not in submitted["kwargs"]
    assert submitted["kwargs"]["plan_graph"] is None


def test_do_queue_requires_broker(monkeypatch):
    """--queue with no broker -> clear error, nonzero exit, no dispatch call."""
    monkeypatch.setattr(cli, "dispatch", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("dispatch must not run when --queue errors")))
    monkeypatch.setattr(cli.broker_client, "is_running", lambda *a, **k: False)

    rc = cli.main(["do", "hello", "--queue"])
    assert rc == 1


def test_do_detach_prints_job_id_and_returns(monkeypatch, capsys):
    """--detach submits and prints the job id only (no wait/status poll)."""
    monkeypatch.setattr(cli.broker_client, "is_running", lambda *a, **k: True)
    monkeypatch.setattr(cli.broker_client, "submit",
                        lambda instruction, kwargs, pools: "JOB-XYZ")

    def boom(*a, **k):
        raise AssertionError("--detach must not poll status")

    monkeypatch.setattr(cli.broker_client, "status", boom)

    rc = cli.main(["do", "hello", "--queue", "--detach"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "JOB-XYZ" in out


def test_do_detach_incompatible_with_direct(monkeypatch):
    monkeypatch.setattr(cli, "dispatch", lambda *a, **k: _result())
    rc = cli.main(["do", "hello", "--direct", "--detach"])
    assert rc == 1


def test_do_pools_follow_custom_chains(monkeypatch):
    submitted = {}

    monkeypatch.setattr(cli.broker_client, "is_running", lambda *a, **k: True)

    def fake_submit(instruction, kwargs, pools):
        submitted["pools"] = pools
        return "J"

    monkeypatch.setattr(cli.broker_client, "submit", fake_submit)
    monkeypatch.setattr(cli.broker_client, "status", lambda j: {
        "id": j, "status": "done", "run_id": None,
        "result": {"success": True}, "kwargs": {}, "instruction": "hi",
    })

    rc = cli.main(["do", "hi", "--queue", "--generator", "codex", "--critic", "claude"])
    assert rc == 0
    assert submitted["pools"] == ["claude", "codex"]


def test_do_broker_failed_job_returns_nonzero(monkeypatch):
    monkeypatch.setattr(cli.broker_client, "is_running", lambda *a, **k: True)
    monkeypatch.setattr(cli.broker_client, "submit",
                        lambda instruction, kwargs, pools: "J")
    monkeypatch.setattr(cli.broker_client, "status", lambda j: {
        "id": j, "status": "failed", "run_id": None,
        "result": {"success": False, "error": "boom"},
        "kwargs": {"mode": "direct"}, "instruction": "hi",
    })
    rc = cli.main(["do", "hi", "--queue"])
    assert rc == 1


# --------------------------------------------------------------------------- #
# pool derivation + result-card reconstruction helpers
# --------------------------------------------------------------------------- #
def test_derive_pools_defaults():
    assert cli._derive_pools(None, None) == ["agy", "codex", "grok"]


def test_derive_pools_drops_computer_use():
    assert cli._derive_pools(["computer-use"], ["codex"]) == ["codex"]


def test_broker_kwargs_is_json_safe(tmp_path):
    import json

    kw = {"run_id": "x", "out_dir": tmp_path, "mode": "direct", "plan_graph": None}
    proj = cli._broker_kwargs(kw)
    assert "run_id" not in proj
    assert proj["out_dir"] == str(tmp_path)
    json.dumps(proj)  # must not raise


def test_result_from_meta_roundtrip(tmp_path, monkeypatch):
    import json

    run_dir = tmp_path / "RID-9"
    run_dir.mkdir()
    meta = {
        "run_id": "RID-9", "run_dir": str(run_dir), "mode": "direct",
        "generator": "codex", "critic": None, "success": True, "duration_s": 2.5,
        "unknown_future_field": "ignored",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    res = cli._result_from_meta("RID-9")
    assert res is not None
    assert res.run_id == "RID-9"
    assert res.success is True


def test_result_from_meta_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    assert cli._result_from_meta("NOPE") is None


# --------------------------------------------------------------------------- #
# queue command
# --------------------------------------------------------------------------- #
def test_cmd_queue_no_broker(monkeypatch, capsys):
    monkeypatch.setattr(cli.broker_client, "is_running", lambda *a, **k: False)
    rc = cli.main(["queue"])
    assert rc == 0
    assert "no broker" in capsys.readouterr().out.lower()


def test_cmd_queue_lists_jobs(monkeypatch, capsys):
    monkeypatch.setattr(cli.broker_client, "is_running", lambda *a, **k: True)
    monkeypatch.setattr(cli.broker_client, "list_jobs", lambda *a, **k: [
        {"id": "J1", "status": "running", "kwargs": {"mode": "master"},
         "instruction": "build the thing with many words " * 5},
        {"id": "J2", "status": "queued", "kwargs": {"mode": "direct"},
         "instruction": "small"},
    ])
    rc = cli.main(["queue"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "J1" in out and "J2" in out
    assert "master" in out and "direct" in out
    assert "running" in out and "queued" in out


# --------------------------------------------------------------------------- #
# serve command — singleton guard wiring (no real loop run)
# --------------------------------------------------------------------------- #
def test_cmd_serve_refuses_rival(monkeypatch, capsys):
    """serve exits nonzero with a clear message when a broker already owns the
    socket (singleton guard surfaces BrokerAlreadyRunning)."""
    import harness.broker as broker_mod

    class FakeServer:
        async def serve_forever(self):
            raise broker_mod.BrokerAlreadyRunning(4242, "/tmp/x.sock")

    monkeypatch.setattr(broker_mod, "make_server", lambda **k: FakeServer())
    rc = cli.main(["serve"])
    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "already running" in err


def test_cmd_serve_passes_cap(monkeypatch):
    import harness.broker as broker_mod

    seen = {}

    class FakeServer:
        async def serve_forever(self):
            return None

    def fake_make_server(**kwargs):
        seen.update(kwargs)
        return FakeServer()

    monkeypatch.setattr(broker_mod, "make_server", fake_make_server)
    rc = cli.main(["serve", "--cap", "5"])
    assert rc == 0
    assert seen.get("cap") == 5

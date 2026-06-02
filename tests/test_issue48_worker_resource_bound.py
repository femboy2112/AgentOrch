"""Issue #48 — the ``--test-cmd -n2`` bound governs only the orchestrator's own
verifier; a generation/refinement step running its OWN pytest inherited the
target repo's ``-n auto`` addopts (one xdist worker per core) with no BLAS thread
pins, hitting ~5.6 GB inside the build scope.

The worker CLI subprocess is the single env every such command descends from, so
``apply_worker_resource_bounds`` pins it. These tests pin: default bounds,
respect-existing precedence, the PYTEST_ADDOPTS append/override, the xdist knob,
and the opt-out — plus that ``_child_env`` actually carries the bounds.
"""

from typing import List, Optional

from agy_orchestrator.core.agent import (
    AgentInstance,
    apply_worker_resource_bounds,
)


class _StubAgent(AgentInstance):
    @classmethod
    async def get_available_models(cls):
        return ["stub"]

    @classmethod
    async def get_model_usage(cls, model):
        return 100.0

    def build_command(self, piped_input: Optional[str] = None) -> List[str]:
        return ["true"]


def test_thread_pins_set_when_absent(monkeypatch):
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    out = apply_worker_resource_bounds({})
    assert out["OPENBLAS_NUM_THREADS"] == "1"
    assert out["OMP_NUM_THREADS"] == "1"
    assert out["MKL_NUM_THREADS"] == "1"
    assert out["NUMEXPR_NUM_THREADS"] == "1"


def test_existing_thread_pin_is_respected(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    # An operator/parent value must win over our default.
    out = apply_worker_resource_bounds({"OMP_NUM_THREADS": "4"})
    assert out["OMP_NUM_THREADS"] == "4"
    # Unset siblings still get pinned.
    assert out["OPENBLAS_NUM_THREADS"] == "1"


def test_pytest_addopts_bounds_xdist_when_unset(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.delenv("AGY_WORKER_PYTEST_XDIST", raising=False)
    out = apply_worker_resource_bounds({})
    assert out["PYTEST_ADDOPTS"] == "-n 2"


def test_pytest_addopts_appends_to_preserve_existing(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.delenv("AGY_WORKER_PYTEST_XDIST", raising=False)
    # The repo's existing addopts (incl. a runaway -n auto) are preserved, but our
    # -n is appended LAST so pytest takes it (last -n wins).
    out = apply_worker_resource_bounds({"PYTEST_ADDOPTS": "-q -n auto"})
    assert out["PYTEST_ADDOPTS"] == "-q -n auto -n 2"


def test_pytest_xdist_knob(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.setenv("AGY_WORKER_PYTEST_XDIST", "4")
    assert apply_worker_resource_bounds({})["PYTEST_ADDOPTS"] == "-n 4"


def test_pytest_xdist_zero_forces_serial(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.setenv("AGY_WORKER_PYTEST_XDIST", "0")
    assert apply_worker_resource_bounds({})["PYTEST_ADDOPTS"] == "-p no:xdist"


def test_opt_out_returns_env_unchanged(monkeypatch):
    monkeypatch.setenv("AGY_WORKER_RESOURCE_BOUND", "0")
    src = {"PYTEST_ADDOPTS": "-n auto"}
    out = apply_worker_resource_bounds(src)
    assert out is src  # untouched, no pins, no addopts rewrite
    assert "OMP_NUM_THREADS" not in out


def test_child_env_carries_bounds_by_default(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.delenv("AGY_WORKER_PYTEST_XDIST", raising=False)
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    env = _StubAgent(prompt="x")._child_env()
    assert env is not None
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["PYTEST_ADDOPTS"] == "-n 2"


def test_child_env_extra_env_wins_over_bounds(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    agent = _StubAgent(prompt="x")
    agent.extra_env = {"OMP_NUM_THREADS": "8", "BROWSER": "/bin/true"}
    env = agent._child_env()
    assert env is not None
    assert env["OMP_NUM_THREADS"] == "8"  # explicit extra_env overrides the pin
    assert env["BROWSER"] == "/bin/true"

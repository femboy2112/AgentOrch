"""Issue #39 — opt-in memory cap on the verifier subprocess.

A heavy --test-cmd can OOM a small host and, because the verifier is a child of
the orchestrator, take the harness down with it. --verifier-mem-max runs the
test command inside its OWN transient systemd scope so a spike is OOM-killed in
that scope (the orchestrator survives) and is classified as "exceeded resource
budget", distinct from "tests failed".

These tests are hermetic: they never spawn a real systemd scope. They exercise
the argv construction, the graceful no-systemd fallback, and the OOM
classification (by forcing the cap-active flag and running a plain command that
exits with the OOM return code).
"""
from __future__ import annotations

import asyncio

import pytest

from agy_orchestrator.execution.verifier import QualityVerifier, VerifierResult


def _run(coro):
    return asyncio.run(coro)


# --- cap inactive by default --- #

def test_no_cap_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("AGY_VERIFIER_MEM_MAX", raising=False)
    v = QualityVerifier(test_commands=["true"])
    assert v.mem_max is None
    assert v._cap_active is False
    assert v._exec_argv("true") is None          # -> plain shell path


def test_uncapped_fallback_when_systemd_absent(tmp_path, monkeypatch):
    # Cap requested but no usable user scope manager -> degrade to uncapped + warn.
    monkeypatch.setattr(QualityVerifier, "_systemd_scope_available", staticmethod(lambda: False))
    v = QualityVerifier(test_commands=["true"], mem_max="3G")
    assert v.mem_max == "3G"
    assert v._cap_active is False
    # Still runs (uncapped) and verifies normally.
    res = _run(v.verify(str(tmp_path)))
    assert res.ok is True


# --- argv construction --- #

def test_exec_argv_builds_capped_scope(monkeypatch):
    monkeypatch.setattr(QualityVerifier, "_systemd_scope_available", staticmethod(lambda: True))
    v = QualityVerifier(test_commands=["pytest -q"], mem_max="2G")
    assert v._cap_active is True
    argv = v._exec_argv("pytest -q")
    assert argv[0] == "systemd-run"
    assert "--user" in argv and "--scope" in argv and "--collect" in argv
    assert "MemoryMax=2G" in argv
    assert "MemorySwapMax=0" in argv            # hard ceiling by default (no thrash)
    assert argv[-3:] == ["/bin/sh", "-c", "pytest -q"]


def test_swap_max_override(monkeypatch):
    monkeypatch.setattr(QualityVerifier, "_systemd_scope_available", staticmethod(lambda: True))
    monkeypatch.setenv("AGY_VERIFIER_SWAP_MAX", "1G")
    v = QualityVerifier(test_commands=["x"], mem_max="2G")
    assert "MemorySwapMax=1G" in v._exec_argv("x")


# --- OOM classification --- #

def test_oom_returncode_under_cap_is_resource_exceeded(tmp_path, monkeypatch):
    # Force cap-active but run via plain shell (no real systemd) so we can drive
    # the OOM return code deterministically. rc 137 under a cap => resource budget.
    v = QualityVerifier(test_commands=["exit 137"], mem_max="1G")
    v._cap_active = True
    monkeypatch.setattr(v, "_exec_argv", lambda cmd: None)   # run via shell
    res = _run(v.verify(str(tmp_path)))
    assert res.ok is False
    assert res.resource_exceeded is True
    assert "resource budget" in res.message.lower()


def test_ordinary_failure_not_misclassified(tmp_path, monkeypatch):
    v = QualityVerifier(test_commands=["exit 1"], mem_max="1G")
    v._cap_active = True
    monkeypatch.setattr(v, "_exec_argv", lambda cmd: None)
    res = _run(v.verify(str(tmp_path)))
    assert res.ok is False
    assert res.resource_exceeded is False
    assert "exit code 1" in res.message


def test_oom_code_without_cap_is_plain_failure(tmp_path, monkeypatch):
    # Same 137 code but no cap active -> NOT a resource-budget classification.
    monkeypatch.delenv("AGY_VERIFIER_MEM_MAX", raising=False)
    v = QualityVerifier(test_commands=["exit 137"])
    assert v._cap_active is False
    res = _run(v.verify(str(tmp_path)))
    assert res.ok is False
    assert res.resource_exceeded is False


def test_result_field_defaults_false():
    assert VerifierResult(ok=False).resource_exceeded is False

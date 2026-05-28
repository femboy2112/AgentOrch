from __future__ import annotations

import asyncio
import re

from agy_orchestrator.execution.verifier import QualityVerifier, VerifierResult


def test_verifier_result_unpacks_as_two_tuple():
    result = VerifierResult(ok=False, message="boom")
    ok, msg = result
    assert ok is False
    assert msg == "boom"


def test_verifier_result_truthiness_follows_ok():
    assert bool(VerifierResult(ok=True)) is True
    assert bool(VerifierResult(ok=False)) is False


def test_quality_verifier_returns_structured_on_failure(tmp_path):
    verifier = QualityVerifier(["bash -c 'echo verifier-fail >&2; false'"])
    result = asyncio.run(verifier.verify(working_directory=str(tmp_path)))

    assert result.ok is False
    assert result.returncode != 0
    assert "verifier-fail" in result.stderr_tail
    assert result.error_hash is not None
    assert re.fullmatch(r"[0-9a-f]{16}", result.error_hash)

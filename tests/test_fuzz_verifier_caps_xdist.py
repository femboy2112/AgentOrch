"""Fuzz tests for execution/verifier.py — QualityVerifier subprocess/parse/caps/process-group.

Hermetic: drives the pure helpers directly with randomized inputs. No subprocess,
no network, no worker CLI. Asserts the structural invariants that the two fixed
defects violated:

  * _clamp_xdist must NEVER rewrite an embedded `-n<digits>` that is not a
    token-start xdist flag (verifier-1), and must never lengthen the command.
  * _head_tail must NEVER return output longer than the input for a positive cap
    (verifier-2).
"""
import os
import random
import string

import pytest

from agy_orchestrator.execution.verifier import QualityVerifier


@pytest.fixture
def verifier(monkeypatch):
    monkeypatch.delenv("AGY_WORKER_RESOURCE_BOUND", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    return QualityVerifier(test_commands=["x"])


def _rand_word(rng):
    n = rng.randint(1, 12)
    return "".join(rng.choice(string.ascii_letters + string.digits + "-_/.") for _ in range(n))


def test_fuzz_clamp_does_not_touch_embedded_substrings(verifier):
    rng = random.Random(1234)
    for _ in range(2000):
        # Build a command where `-n<digits>` only ever appears EMBEDDED inside a
        # larger non-whitespace token (so it is never a real flag start).
        digits = str(rng.randint(0, 999))
        prefix = "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(1, 6)))
        token = f"{prefix}-n{digits}suffix{rng.randint(0,9)}"
        words = [_rand_word(rng) for _ in range(rng.randint(0, 4))]
        words.insert(rng.randint(0, len(words)), token)
        cmd = "pytest " + " ".join(words)
        out = verifier._clamp_xdist(cmd)
        # The embedded token (which is never preceded by whitespace+flag-start)
        # must survive verbatim, so the command is unchanged.
        assert out == cmd, f"embedded substring corrupted: {cmd!r} -> {out!r}"


def test_fuzz_clamp_only_clamps_genuine_above_budget(verifier):
    rng = random.Random(99)
    cpu = os.cpu_count()  # 4 (monkeypatched)
    for _ in range(2000):
        k = rng.randint(0, 64)
        flag = rng.choice(["-n", "--numprocesses"])
        sep = rng.choice([" ", "", "="])
        # "-n=" / "--numprocesses" with empty sep then digits are all valid forms.
        cmd = f"pytest {flag}{sep}{k} -q"
        out = verifier._clamp_xdist(cmd)
        if k <= cpu:
            assert out == cmd  # within budget: untouched
        else:
            assert f"{flag}{sep}{cpu}" in out  # clamped to cpu
            assert str(k) not in out.split("-q")[0]  # original count gone
        # Never longer than a sane reconstruction (no runaway growth).
        assert len(out) <= len(cmd)


def test_fuzz_head_tail_never_exceeds_input(verifier):
    rng = random.Random(7)
    for _ in range(3000):
        length = rng.randint(0, 400)
        text = "".join(rng.choice("ab\n ") for _ in range(length))
        cap = rng.randint(-3, 50)
        out = QualityVerifier._head_tail(text, cap)
        if cap <= 0:
            assert out == text  # unbounded passthrough
        else:
            # The core invariant the bug violated: a "cap" must never make the
            # output LONGER than the raw input (that defeats its purpose). The
            # head+tail marker overhead is allowed to exceed `cap` itself for
            # generous budgets, but the framed result must always shrink the text.
            assert len(out) <= len(text), (
                f"cap defeated: cap={cap} len(text)={len(text)} len(out)={len(out)}"
            )
            if cap == 1:
                # The specific verifier-2 trigger: must honour a 1-char cap.
                assert len(out) <= 1

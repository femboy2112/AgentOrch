from __future__ import annotations

from harness.dispatch import _build_prompt


def test_build_prompt_without_spec_omits_spec_block():
    prompt = _build_prompt("do the thing", None)
    assert "do the thing" in prompt
    assert "Approved design specification" not in prompt


def test_build_prompt_injects_spec_as_authoritative():
    spec = "# Design\nUse a ring buffer."
    prompt = _build_prompt("implement it", None, spec)
    assert "Approved design specification (authoritative)" in prompt
    assert "Use a ring buffer." in prompt
    assert "implement it" in prompt


def test_build_prompt_spec_precedes_context():
    spec = "SPEC-BODY"
    prompt = _build_prompt("inst", "CONTEXT-BODY", spec)
    assert "SPEC-BODY" in prompt
    assert "CONTEXT-BODY" in prompt
    # Authoritative spec comes before the lower-priority free-form context.
    assert prompt.index("SPEC-BODY") < prompt.index("CONTEXT-BODY")

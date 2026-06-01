# Spec: textstats utility (benchmark fixture)

## Overview
A tiny, dependency-free text-statistics helper used purely as a fixed benchmark
input. The prose here is deliberately representative of a real FloodSpec so the
context-projection findings can be A/B'd against a realistic spec size.

## Goal
Provide deterministic word/character statistics over arbitrary unicode strings.

## Requirements
- `word_count(s)` returns the number of whitespace-delimited tokens.
- `char_histogram(s)` returns a mapping of each character to its frequency.
- Pure standard library; no third-party dependencies.

## Components & Interfaces
- `word_count(s: str) -> int`
- `char_histogram(s: str) -> dict[str, int]`

## Data Models
- Histogram: `dict[str, int]`, keys are single characters, values are counts >= 1.

## Constraints & Guardrails
- Must handle empty input (returns 0 / {}).
- Must be unicode-correct (no byte-level counting).
- No global mutable state.

## Alternatives considered
- A `collections.Counter`-based histogram (acceptable) vs a hand-rolled loop.

## Open questions
- None for the benchmark fixture.

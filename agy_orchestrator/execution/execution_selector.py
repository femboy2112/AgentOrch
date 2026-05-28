"""Execution-based candidate selection for code-generation tasks.

For test-graded coding tasks, this selector scores each candidate by real pytest
outcomes instead of an LLM judge. Soundness comes from isolation: each candidate
text is re-extracted into fresh code and executed in its own temporary sandbox,
so candidates are compared on independent disk state rather than a shared repo
workspace.
"""
from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

_THINK_RE = re.compile(r"^\s*<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCED_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_COLLECTION_ERROR_RE = re.compile(r"error\s+during\s+collection|error\s+collecting", re.IGNORECASE)
_SUMMARY_COUNT_RE = re.compile(
    r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed|deselected|rerun|reruns)",
    re.IGNORECASE,
)


class ExecutionSelector:
    """Select among K candidate texts by sandboxed pytest pass counts.

    Each candidate is evaluated in an isolated temporary directory by writing the
    extracted code to ``solution.py`` (configurable) and running a fixed test
    suite against it. Because every candidate gets a fresh sandbox, ranking is
    sound even when candidates were originally produced without disk isolation.
    """

    def __init__(
        self,
        test_source: str,
        solution_filename: str = "solution.py",
        timeout: float = 60.0,
    ):
        self.test_source = test_source
        self.solution_filename = solution_filename
        self.timeout = timeout

    @staticmethod
    def extract_code(text: str) -> str:
        """Extract the first fenced code block, else return cleaned raw text."""
        match = _FENCED_CODE_RE.search(text)
        if match:
            return match.group(1).strip()
        return _THINK_RE.sub("", text, count=1).strip()

    def _score_one(self, code: str) -> tuple[int, int]:
        """Run one candidate in an isolated sandbox and return (passed, total)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / self.solution_filename).write_text(code, encoding="utf-8")
            (tmp_path / "test_solution.py").write_text(
                textwrap.dedent(self.test_source),
                encoding="utf-8",
            )

            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "--no-header", "-x"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return (0, 0)
            except Exception:
                return (0, 0)

            output = f"{proc.stdout}\n{proc.stderr}"
            if _COLLECTION_ERROR_RE.search(output):
                return (0, 0)
            if proc.returncode not in (0, 1):
                return (0, 0)

            passed = 0
            total = 0
            for count_str, status in _SUMMARY_COUNT_RE.findall(output):
                count = int(count_str)
                normalized = status.lower()
                if normalized == "passed":
                    passed += count
                if normalized in {
                    "passed",
                    "failed",
                    "error",
                    "errors",
                    "skipped",
                    "xfailed",
                    "xpassed",
                    "rerun",
                    "reruns",
                }:
                    total += count

            if total <= 0:
                return (0, 0)
            return (passed, total)

    async def select(self, candidates: list[str]) -> tuple[int, list[tuple[int, int]]]:
        """Score candidates concurrently and return (best_index, all_scores)."""
        if not candidates:
            raise ValueError("ExecutionSelector.select needs at least one candidate")

        tasks = [
            asyncio.to_thread(self._score_one, self.extract_code(candidate))
            for candidate in candidates
        ]
        scores = await asyncio.gather(*tasks)

        best_index = max(range(len(scores)), key=lambda idx: (scores[idx][0], -idx))
        return best_index, scores

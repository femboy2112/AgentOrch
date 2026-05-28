#!/usr/bin/env python3
"""Cloud-worker code-quality bench — compare grok vs claude vs codex vs agy.

Points an objective, hidden-pytest grader at the cloud worker CLIs so we can see
how a newly-integrated worker (e.g. grok) stacks up against the incumbents on
real, auto-graded code generation.

Each task asks for ONE self-contained function in a fenced code block; we extract
it and run a hidden pytest suite. The signal is pass/fail of real tests, never a
model's self-rating. Easy tasks check baseline correctness; HARD tasks (LeetCode
medium/hard: expression eval, path canonicalization, wildcard matching) are the
discriminators between strong models.

Single-shot only (no critic loop) — we're measuring each worker's raw one-pass
quality head-to-head, with each provider on its default "best" model.

Usage:
    python scripts/cloud_eval.py                         # all workers, all tasks
    python scripts/cloud_eval.py --workers grok claude   # subset of workers
    python scripts/cloud_eval.py --tasks eval_expr wildcard --timeout 240
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---- easy baseline tasks --------------------------------------------------
# Each task: name -> (prompt, test_src). The prompt asks for ONE python code
# block defining the named symbol; test_src imports `solution` and asserts.
EASY_TASKS = {
    "roman_to_int": (
        "Write a Python function `roman_to_int(s: str) -> int` that converts a "
        "Roman numeral string (e.g. 'MCMXciv' upper/lower) to its integer value. "
        "Handle subtractive pairs (IV=4, IX=9, XL=40, XC=90, CD=400, CM=900).",
        """
        from solution import roman_to_int
        def test_basic():
            assert roman_to_int("III") == 3
            assert roman_to_int("IV") == 4
            assert roman_to_int("IX") == 9
            assert roman_to_int("LVIII") == 58
            assert roman_to_int("MCMXCIV") == 1994
        def test_case_insensitive():
            assert roman_to_int("mcmxciv") == 1994
        """,
    ),
    "balanced": (
        "Write a Python function `is_balanced(s: str) -> bool` that returns True "
        "iff the brackets in s are balanced. Consider (), [], and {}. Ignore any "
        "non-bracket characters.",
        """
        from solution import is_balanced
        def test_balanced():
            assert is_balanced("()[]{}") is True
            assert is_balanced("(a[b]{c})") is True
            assert is_balanced("([)]") is False
            assert is_balanced("(") is False
            assert is_balanced("") is True
            assert is_balanced("]") is False
        """,
    ),
    "merge_intervals": (
        "Write a Python function `merge_intervals(intervals: list[list[int]]) -> "
        "list[list[int]]` that merges all overlapping intervals and returns them "
        "sorted by start. E.g. [[1,3],[2,6],[8,10],[15,18]] -> [[1,6],[8,10],[15,18]].",
        """
        from solution import merge_intervals
        def test_merge():
            assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
            assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
            assert merge_intervals([[1,4],[0,4]]) == [[0,4]]
            assert merge_intervals([]) == []
            assert merge_intervals([[1,4],[2,3]]) == [[1,4]]
        """,
    ),
    "parse_ini": (
        "Write a Python function `parse_ini(text: str) -> dict` that parses a "
        "simple INI string into a dict of {section: {key: value}}. Lines like "
        "'[section]' open a section; 'key = value' set keys (strip whitespace); "
        "lines starting with '#' or ';' are comments; blank lines ignored. Keys "
        "before any section go under the '' (empty string) section.",
        """
        from solution import parse_ini
        def test_parse():
            text = "g = 1\\n# comment\\n[a]\\nx = 10\\ny=20\\n; c\\n[b]\\nz = hi"
            out = parse_ini(text)
            assert out[""]["g"] == "1"
            assert out["a"]["x"] == "10"
            assert out["a"]["y"] == "20"
            assert out["b"]["z"] == "hi"
        """,
    ),
}


def extract_code(text: str) -> str:
    """Pull the first python/code fenced block; fall back to the whole text."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Some models emit a bare block; strip a leading <think>…</think>.
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return body


def run_test(code: str, test_src: str) -> tuple[bool, str]:
    """Write code + test to a temp dir, run pytest, return (passed, tail)."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "solution.py").write_text(code, encoding="utf-8")
        (Path(d) / "test_solution.py").write_text(textwrap.dedent(test_src), encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--no-header", "-x", "--tb=short"],
                cwd=d, capture_output=True, text=True, timeout=60,
            )
        except subprocess.TimeoutExpired:
            return False, "pytest timed out (likely infinite loop in generated code)"
        out = (proc.stdout + proc.stderr).strip()
        if proc.returncode == 0:
            return True, "ok"
        idx = out.find("FAILURES")
        detail = out[idx:] if idx != -1 else out
        return False, detail[-1200:]

# ---- discriminating tasks (LeetCode medium/hard) -------------------------
# Verified graders: a correct reference solution passes each test suite.
HARD_TASKS = {
    "eval_expr": (
        "Write a Python function `eval_expr(expr: str) -> int` that evaluates a "
        "basic arithmetic expression string with +, -, *, / operators, parentheses, "
        "and arbitrary spaces. Standard precedence (* and / bind tighter than + and "
        "-), left-to-right. Division is integer division that truncates toward zero. "
        "All inputs and intermediate results are non-negative integers.",
        """
        from solution import eval_expr
        def test_eval():
            assert eval_expr("3+2*2") == 7
            assert eval_expr(" 3/2 ") == 1
            assert eval_expr("(1+(4+5+2)-3)+(6+8)") == 23
            assert eval_expr("2*(5+5*2)/3+(6/2+8)") == 21
            assert eval_expr("14-3/2") == 13
        """,
    ),
    "simplify_path": (
        "Write a Python function `simplify_path(path: str) -> str` that converts an "
        "absolute Unix-style file path into its canonical form. Collapse '.' (current "
        "dir), '..' (parent dir — pops the previous component, no-op at root), and any "
        "number of consecutive slashes. The result starts with a single '/' and has no "
        "trailing slash (except the root itself, which is '/').",
        """
        from solution import simplify_path
        def test_simplify():
            assert simplify_path("/home/") == "/home"
            assert simplify_path("/../") == "/"
            assert simplify_path("/home//foo/") == "/home/foo"
            assert simplify_path("/a/./b/../../c/") == "/c"
            assert simplify_path("/a/../../b/../c//.//") == "/c"
            assert simplify_path("/") == "/"
        """,
    ),
    "wildcard": (
        "Write a Python function `is_match(s: str, p: str) -> bool` implementing "
        "wildcard pattern matching over the ENTIRE string s. '?' matches any single "
        "character; '*' matches any sequence of characters including the empty "
        "sequence. The match must cover all of s (not a substring).",
        """
        from solution import is_match
        def test_wild():
            assert is_match("aa","a") is False
            assert is_match("aa","*") is True
            assert is_match("cb","?a") is False
            assert is_match("adceb","*a*b") is True
            assert is_match("acdcb","a*c?b") is False
            assert is_match("","*") is True
            assert is_match("","") is True
        """,
    ),
    # --- "brutal" tier: fiddly spec/formatting where frontier models diverge ---
    "int_to_words": (
        "Write a Python function `int_to_words(n: int) -> str` that converts a "
        "non-negative integer (0 <= n < 2**31) into its English-words representation, "
        "capitalized per word and space-separated, with NO trailing/leading spaces and "
        "no commas. Use Thousand/Million/Billion grouping. E.g. 0 -> 'Zero'; 123 -> "
        "'One Hundred Twenty Three'; 1234567 -> 'One Million Two Hundred Thirty Four "
        "Thousand Five Hundred Sixty Seven'; 1000010 -> 'One Million Ten'.",
        """
        from solution import int_to_words
        def test_itw():
            assert int_to_words(0)=="Zero"
            assert int_to_words(100)=="One Hundred"
            assert int_to_words(123)=="One Hundred Twenty Three"
            assert int_to_words(12345)=="Twelve Thousand Three Hundred Forty Five"
            assert int_to_words(1000000)=="One Million"
            assert int_to_words(1000010)=="One Million Ten"
            assert int_to_words(1234567)=="One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"
            assert int_to_words(2147483647)=="Two Billion One Hundred Forty Seven Million Four Hundred Eighty Three Thousand Six Hundred Forty Seven"
        """,
    ),
    "full_justify": (
        "Write a Python function `full_justify(words: list[str], maxWidth: int) -> "
        "list[str]` that fully (left+right) justifies the given words into lines of "
        "exactly maxWidth characters. Pack as many words per line as fit (words "
        "separated by at least one space). Distribute extra spaces as evenly as "
        "possible; if spaces don't divide evenly, the LEFT slots get more. A line with "
        "a single word, and the LAST line, are LEFT-justified (single spaces between "
        "words) and padded with trailing spaces to maxWidth.",
        """
        from solution import full_justify
        def test_fj():
            assert full_justify(["This","is","an","example","of","text","justification."],16)==[
                "This    is    an","example  of text","justification.  "]
            assert full_justify(["What","must","be","acknowledgment","shall","be"],16)==[
                "What   must   be","acknowledgment  ","shall be        "]
            assert full_justify(["Science","is","what","we","understand","well","enough","to",
                "explain","to","a","computer.","Art","is","everything","else","we","do"],20)==[
                "Science  is  what we","understand      well","enough to explain to",
                "a  computer.  Art is","everything  else  we","do                  "]
        """,
    ),
}

ALL_TASKS = {**EASY_TASKS, **HARD_TASKS}
EASY = set(EASY_TASKS)

# Cloud workers are agentic (they default to editing files / running tools). For
# a pure generation bench we forbid that and demand a single fenced block.
CLOUD_SUFFIX = (
    "\n\nIMPORTANT: Do NOT create or edit any files, and do NOT run any commands "
    "or tools. Reply with ONLY one Python code block (```python ... ```) containing "
    "the complete function plus any imports it needs — no prose, no tests, no "
    "example usage."
)


async def run_one(worker: str, task: str, prompt: str, test_src: str) -> dict:
    """Generate with one worker, grade with the hidden test. Never raises."""
    from harness import roles
    t0 = time.time()
    try:
        agent = roles.build_role_agent([worker], prompt=prompt + CLOUD_SUFFIX,
                                       fallback=False)
        raw = await agent.run_async()
        code = extract_code(raw)
        ok, tail = run_test(code, test_src)
        return {"worker": worker, "task": task, "ok": ok,
                "t": time.time() - t0, "tail": "" if ok else tail[:80],
                "empty": not code.strip()}
    except Exception as exc:
        return {"worker": worker, "task": task, "ok": False,
                "t": time.time() - t0, "tail": f"{type(exc).__name__}: {exc}"[:80],
                "empty": False}


async def run(args) -> None:
    names = args.tasks or list(ALL_TASKS)
    workers = args.workers
    model_of = _describe_models(workers)
    print("cloud code-quality bench — single-shot, hidden-pytest graded")
    print("workers: " + ", ".join(f"{w} ({model_of[w]})" for w in workers))
    print(f"tasks:   {len(names)}  ({sum(t in EASY for t in names)} easy, "
          f"{sum(t not in EASY for t in names)} hard)   timeout={int(os.environ.get('AGY_TIMEOUT','0'))}s\n")

    results: dict = {w: {} for w in workers}
    # Per task, fan out across workers concurrently (independent backends).
    for task in names:
        prompt, test_src = ALL_TASKS[task]
        rows = await asyncio.gather(*[
            run_one(w, task, prompt, test_src) for w in workers
        ])
        cells = []
        for r in rows:
            results[r["worker"]][task] = r
            mark = "PASS" if r["ok"] else "FAIL"
            cells.append(f"{r['worker']}=[{mark}] {r['t']:4.0f}s")
        tag = "easy" if task in EASY else "HARD"
        print(f"{task:16} {tag}  " + " | ".join(cells))
        for r in rows:
            if not r["ok"] and r["tail"]:
                why = "EMPTY OUTPUT" if r["empty"] else r["tail"]
                print(f"    {r['worker']:8} ✗ {why}")

    # Scoreboard
    print("\n=== scoreboard ===")
    hdr = f"{'worker':10} {'easy':>7} {'hard':>7} {'total':>7} {'avg s':>7}"
    print(hdr)
    print("-" * len(hdr))
    rankable = []
    for w in workers:
        rs = results[w]
        easy_ok = sum(rs[t]["ok"] for t in names if t in EASY)
        hard_ok = sum(rs[t]["ok"] for t in names if t not in EASY)
        n_easy = sum(t in EASY for t in names)
        n_hard = sum(t not in EASY for t in names)
        total = easy_ok + hard_ok
        avg = sum(rs[t]["t"] for t in names) / max(len(names), 1)
        rankable.append((total, hard_ok, -avg, w))
        print(f"{w:10} {f'{easy_ok}/{n_easy}':>7} {f'{hard_ok}/{n_hard}':>7} "
              f"{f'{total}/{len(names)}':>7} {avg:7.0f}")
    rankable.sort(reverse=True)
    print("\nranking (total, then hard, then speed): "
          + " > ".join(r[3] for r in rankable))


def _describe_models(workers) -> dict:
    from harness import roles
    out = {}
    for w in workers:
        try:
            _, cfg = roles._cfg_for_token(w)
            out[w] = str(cfg.get("model", "?"))
        except Exception:
            out[w] = "?"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", nargs="*", default=["grok", "claude", "codex", "agy"],
                    help="cloud workers to compare (default: grok claude codex agy)")
    ap.add_argument("--tasks", nargs="*", choices=list(ALL_TASKS),
                    help="subset of tasks (default: all easy+hard)")
    ap.add_argument("--timeout", type=int, default=300,
                    help="per-call wall-clock ceiling (sets AGY_TIMEOUT)")
    args = ap.parse_args()
    os.environ["AGY_TIMEOUT"] = str(args.timeout)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

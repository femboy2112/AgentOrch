from __future__ import annotations

from scripts.cloud_eval import COMPLEXITY_BUDGETS, grade_complexity


LINEAR_BALANCED = """
def is_balanced(s: str) -> bool:
    pairs = {')': '(', ']': '[', '}': '{'}
    opens = set(pairs.values())
    stack = []
    for ch in s:
        if ch in opens:
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
"""


QUADRATIC_BALANCED = """
def is_balanced(s: str) -> bool:
    chars = [ch for ch in s if ch in '()[]{}']
    pairs = {'(': ')', '[': ']', '{': '}'}
    changed = True
    while changed:
        changed = False
        i = 0
        while i + 1 < len(chars):
            if chars[i] in pairs and pairs[chars[i]] == chars[i + 1]:
                # Two list.pop(i) calls (Python-level O(n) each) rather than a
                # C-accelerated del slice, so the O(n^2) shape is visible in
                # wall-time at modest n and the measured slope stays well clear
                # of the budget threshold (no flaky boundary at exp ~1.55).
                chars.pop(i)
                chars.pop(i)
                changed = True
                break
            i += 1
    return not chars
"""


EXPONENTIAL_DECODE_WAYS = """
def num_decodings(s: str) -> int:
    if not s:
        return 1
    if s[0] == '0':
        return 0
    total = num_decodings(s[1:])
    if len(s) >= 2 and 10 <= int(s[:2]) <= 26:
        total += num_decodings(s[2:])
    return total
"""


def _patch_sizes(monkeypatch, task: str, sizes: list[int]) -> None:
    budget = dict(COMPLEXITY_BUDGETS[task])
    budget["sizes"] = sizes
    monkeypatch.setitem(COMPLEXITY_BUDGETS, task, budget)


def test_grade_complexity_accepts_linear_balanced(monkeypatch) -> None:
    _patch_sizes(monkeypatch, "balanced", [1000, 2000, 4000, 8000])

    result = grade_complexity(LINEAR_BALANCED, "balanced", hard_timeout=8)

    assert result["applicable"] is True
    assert result["ok_import"] is True
    assert result["within_budget"] is True
    assert result["timed_out"] is False
    assert result["exponent"] is not None
    assert 0.5 <= result["exponent"] <= 1.5


def test_grade_complexity_rejects_quadratic_balanced(monkeypatch) -> None:
    _patch_sizes(monkeypatch, "balanced", [1000, 2000, 4000, 8000])

    result = grade_complexity(QUADRATIC_BALANCED, "balanced", hard_timeout=10)

    assert result["applicable"] is True
    assert result["ok_import"] is True
    assert result["within_budget"] is False
    assert result["timed_out"] is False
    assert result["exponent"] is not None
    assert result["exponent"] > 1.55
    assert result["label"] in {"O(n^2)", "O(n^3)", "super-polynomial"}


def test_grade_complexity_times_out_exponential_decode(monkeypatch) -> None:
    _patch_sizes(monkeypatch, "decode_ways", [45, 90])

    result = grade_complexity(EXPONENTIAL_DECODE_WAYS, "decode_ways", hard_timeout=2)

    assert result["applicable"] is True
    assert result["ok_import"] is True
    assert result["timed_out"] is True
    assert result["within_budget"] is False
    assert result["label"] == "non-terminating-or-superlinear"


def test_grade_complexity_reports_import_or_symbol_errors() -> None:
    syntax = grade_complexity("def is_balanced(:\n    pass\n", "balanced", hard_timeout=2)
    missing = grade_complexity("def other_name(s):\n    return True\n", "balanced", hard_timeout=2)

    assert syntax["applicable"] is True
    assert syntax["ok_import"] is False
    assert syntax["within_budget"] is None
    assert "SyntaxError" in syntax["notes"]
    assert missing["applicable"] is True
    assert missing["ok_import"] is False
    assert missing["within_budget"] is None
    assert "AttributeError" in missing["notes"]


def test_grade_complexity_not_applicable() -> None:
    result = grade_complexity("def f():\n    return None\n", "roman_to_int", hard_timeout=2)

    assert result["applicable"] is False
    assert result["within_budget"] is None
    assert result["timed_out"] is False

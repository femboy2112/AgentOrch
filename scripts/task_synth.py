#!/usr/bin/env python3
"""Task synthesizer for latency-vs-prompt-characteristics measurement.

Composes the difficulty-tagged, hidden-test-graded PRIMITIVES from cloud_eval into a
single deliverable parameterized along three orthogonal, a-priori-measurable axes, so
we can measure completion_time / tokens / quality as a FUNCTION of prompt characteristics
(not as point estimates on fixed tasks). Everything stays objectively pytest-graded.

Axes:
  * n_units    — how many functions to produce (output-SCALE axis; dominant latency term).
  * difficulty — each unit drawn from a tier: easy / hard / brutal (DIFFICULTY axis).
  * structure  — single_file | multi_file | interdependent (REALISM/structure axis):
       single_file    : all N functions in one solution.py.
       multi_file     : each function in its own mod_<i>.py (cross-file naming overhead).
       interdependent : multi_file PLUS a pipeline.py exposing dispatch(name, *args) that
                        imports and routes to each unit — forces cross-file contracts +
                        an integration test. This is the structural realism that drives
                        real latency while staying gradable.

Each synthesized task carries a FEATURE VECTOR (the a-priori predictors a router could
key on) and an objective grader. Intended consumer: scripts/calibrate.py sweeps configs
over sampled tasks and fits time/quality = f(features, model, effort) -> the calibration
model behind adaptive start-tier routing.

Run `python scripts/task_synth.py` to self-test the grading machinery.
"""
from __future__ import annotations

import random
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cloud_eval import BRUTAL_TASKS, EASY_TASKS, HARD_TASKS

_IMPORT_RE = re.compile(r"from\s+solution\s+import\s+(\w+)")
_FILE_BLOCK_RE = re.compile(r"#\s*file:\s*(\S+)\s*\n```(?:python)?\s*\n(.*?)```",
                            re.DOTALL | re.IGNORECASE)
_FENCED_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_SUMMARY_RE = re.compile(r"(\d+)\s+(passed|failed|error|errors)", re.IGNORECASE)

_TIER_WEIGHT = {"easy": 1, "hard": 2, "brutal": 3}
_TIER_LOC = {"easy": 8, "hard": 20, "brutal": 30}  # rough reference line counts


def _primitives() -> dict:
    """name -> (prompt, test_src, symbol, tier) for every graded primitive."""
    out = {}
    for tier, pool in (("easy", EASY_TASKS), ("hard", HARD_TASKS), ("brutal", BRUTAL_TASKS)):
        for name, (prompt, test_src) in pool.items():
            m = _IMPORT_RE.search(test_src)
            if m:
                out[name] = (prompt, test_src, m.group(1), tier)
    return out


PRIMITIVES = _primitives()


@dataclass
class SynthTask:
    prompt: str
    features: dict
    units: list = field(repr=False)        # [(name, prompt, test_src, symbol, tier)]
    structure: str = "single_file"

    def grade(self, response_text: str) -> tuple[int, int]:
        """Write the model's file(s) + composed tests into a sandbox, run pytest,
        return (passed, total). Robust to extraction/timeout/collection failures -> (0,0)."""
        files = _extract_files(response_text, self.structure)
        if not files:
            return (0, 0)
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            for fname, code in files.items():
                (dpath / fname).write_text(code, encoding="utf-8")
            # Compose one test file per unit, importing from the right module.
            for i, (_, _, test_src, symbol, _) in enumerate(self.units):
                module = "solution" if self.structure == "single_file" else f"mod_{i}"
                test = re.sub(r"from\s+solution\s+import", f"from {module} import",
                              textwrap.dedent(test_src))
                (dpath / f"test_u{i}.py").write_text(test, encoding="utf-8")
            if self.structure == "interdependent":
                (dpath / "test_pipeline.py").write_text(self._integration_test(), encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "--no-header"],
                    cwd=d, capture_output=True, text=True, timeout=90,
                )
            except Exception:
                return (0, 0)
            out = f"{proc.stdout}\n{proc.stderr}"
            if "error during collection" in out.lower():
                return (0, 0)
            passed = total = 0
            for n, status in _SUMMARY_RE.findall(out):
                n = int(n)
                if status.lower() == "passed":
                    passed += n
                total += n
            return (passed, total) if total else (0, 0)

    def _integration_test(self) -> str:
        """dispatch(symbol, *args) must route to each unit — checks cross-file wiring."""
        lines = ["from pipeline import dispatch"]
        for _, _, test_src, symbol, _ in self.units:
            # Reuse the first assert of each unit's suite, routed through dispatch.
            for asrt in re.findall(r"assert\s+(\w+)\((.*?)\)\s*(==|is)\s*(.+)", textwrap.dedent(test_src)):
                fn, args, op, exp = asrt
                lines.append(f"def test_dispatch_{symbol}():")
                lines.append(f"    assert dispatch({symbol!r}, {args}) {op} {exp}")
                break
        return "\n".join(lines) + "\n"


def _extract_files(text: str, structure: str) -> dict:
    """Parse model output into {filename: code}. multi_file/interdependent expect
    '# file: name.py' headers; single_file takes the first fenced block as solution.py."""
    blocks = {fn: code.strip() for fn, code in _FILE_BLOCK_RE.findall(text)}
    if structure == "single_file":
        if blocks:
            # Prefer a solution.py header; else first block.
            return {"solution.py": blocks.get("solution.py", next(iter(blocks.values())))}
        m = _FENCED_RE.search(text)
        return {"solution.py": m.group(1).strip()} if m else {}
    return blocks


def synthesize(n_units: int = 3, difficulty: str = "mixed",
               structure: str = "single_file", seed: int = 0) -> SynthTask:
    """Build one parameterized, graded task. difficulty: easy|hard|brutal|mixed."""
    rng = random.Random(seed)
    pool = [k for k, v in PRIMITIVES.items() if difficulty in ("mixed", v[3])]
    if not pool:
        raise ValueError(f"no primitives for difficulty={difficulty}")
    chosen = [rng.choice(pool) for _ in range(n_units)]
    units = [(c, *PRIMITIVES[c]) for c in chosen]

    # Build the prompt per structure.
    specs = []
    for i, (_, p, _, symbol, _) in enumerate(units):
        loc = f" in file `mod_{i}.py`" if structure != "single_file" else ""
        specs.append(f"{i+1}. (function `{symbol}`{loc}) {p}")
    spec_block = "\n\n".join(specs)
    if structure == "single_file":
        instr = ("Output ONE Python code block containing all the functions above. "
                 "Start the block with no prose.")
    else:
        files = ", ".join(f"`mod_{i}.py`" for i in range(n_units))
        extra = ("" if structure == "multi_file" else
                 " Also output `# file: pipeline.py` defining `dispatch(name, *args)` "
                 "that imports each function above and calls the one whose name matches "
                 "`name`, returning its result.")
        instr = (f"Output each file in its OWN fenced block, each immediately preceded by "
                 f"a line `# file: <filename>`. Create {files} (one function each).{extra}")

    prompt = (f"Implement the following {n_units} independent Python function(s). "
              f"Each must use EXACTLY the given name.\n\n{spec_block}\n\n{instr}\n\n"
              "Do NOT include tests or example usage.")

    tiers = [u[4] for u in units]
    features = {
        "n_units": n_units,
        "n_files": (1 if structure == "single_file"
                    else n_units + (1 if structure == "interdependent" else 0)),
        "structure": structure,
        "difficulty": difficulty,
        "difficulty_score_mean": round(sum(_TIER_WEIGHT[t] for t in tiers) / len(tiers), 2),
        "difficulty_score_max": max(_TIER_WEIGHT[t] for t in tiers),
        "est_loc": sum(_TIER_LOC[t] for t in tiers),
        "prompt_chars": len(prompt),
    }
    return SynthTask(prompt=prompt, features=features, units=units, structure=structure)


# --------------------------------------------------------------------------- #
# Self-test: validate the grading machinery with reference implementations.
# --------------------------------------------------------------------------- #
_REF = {
    "roman_to_int": "def roman_to_int(s):\n s=s.upper();v={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000};t=0;p=0\n for c in reversed(s):\n  x=v[c];t+=-x if x<p else x;p=x\n return t",
    "is_balanced": "def is_balanced(s):\n pr={')':'(',']':'[','}':'{'};st=[]\n for c in s:\n  if c in '([{':st.append(c)\n  elif c in pr:\n   if not st or st.pop()!=pr[c]:return False\n return not st",
    "candy": "def candy(r):\n n=len(r)\n if not n:return 0\n c=[1]*n\n for i in range(1,n):\n  if r[i]>r[i-1]:c[i]=c[i-1]+1\n for i in range(n-2,-1,-1):\n  if r[i]>r[i+1]:c[i]=max(c[i],c[i+1]+1)\n return sum(c)",
}


def _selftest() -> None:
    ok = True
    # single_file: 2 easy units, reference code in one block.
    t = synthesize(n_units=2, difficulty="easy", structure="single_file", seed=1)
    syms = [u[3] for u in t.units]
    if all(s in _REF for s in syms):
        block = "```python\n" + "\n".join(_REF[s] for s in syms) + "\n```"
        p, tot = t.grade(block)
        print(f"single_file {syms}: {p}/{tot} {'OK' if p == tot and tot > 0 else 'FAIL'}")
        ok &= (p == tot and tot > 0)
    # multi_file: force a known set via seed search.
    t = synthesize(n_units=2, difficulty="easy", structure="multi_file", seed=1)
    syms = [u[3] for u in t.units]
    if all(s in _REF for s in syms):
        resp = "\n".join(f"# file: mod_{i}.py\n```python\n{_REF[s]}\n```"
                         for i, s in enumerate(syms))
        p, tot = t.grade(resp)
        print(f"multi_file  {syms}: {p}/{tot} {'OK' if p == tot and tot > 0 else 'FAIL'}")
        ok &= (p == tot and tot > 0)
    # wrong code -> not full pass.
    t = synthesize(n_units=1, difficulty="easy", structure="single_file", seed=3)
    p, tot = t.grade("```python\ndef " + t.units[0][3] + "(*a,**k): return -999\n```")
    print(f"wrong-code  {t.units[0][3]}: {p}/{tot} {'OK' if p < max(tot,1) else 'FAIL'}")
    ok &= (p < max(tot, 1))
    # feature vector sanity
    t = synthesize(n_units=5, difficulty="mixed", structure="interdependent", seed=7)
    print("features:", t.features)
    ok &= (t.features["n_units"] == 5 and t.features["n_files"] == 6)
    print("SELFTEST", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    _selftest()

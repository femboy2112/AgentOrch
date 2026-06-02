"""Hermetic fuzz / property / edge-case tests for workflows/reconcile.py.

These assert ROBUST behaviour against adversarial model replies and witness
inputs. No real worker CLI, no network: the only agent is a stub whose
``run_async`` returns a canned string, and the only subprocesses are
``sh``/``echo``/``sleep`` via the ablation hook.

Axes covered:
  * _extract_json_blob: None/empty/whitespace/no-brace/unbalanced/nested/fenced.
  * _coerce_float / _parse_witness_signal: bool, NaN/Inf strings, embedded numbers,
    huge strings, non-numeric, None.
  * _parse_finding: missing name, OOV classification, sub_kind normalization,
    file/line fallback, control chars in name.
  * parse(): non-dict top level, findings-not-a-list, contradictory reconciled,
    derived-reconciled-never-trusting-the-model, DEEPLY NESTED blob (RecursionError
    must NOT escape — regression guard for the fix).
  * Witness/Result invariants: is_dead, substance gate, mark_starved, to_dict.
  * Ablation: cmd substitution, timeout, worktree failure, mismatch flip, signal
    parsing — all hermetic via sh.
  * execute(): never crashes on a garbage reply; reprompt path.
"""
from __future__ import annotations

import asyncio
import json
import math

import pytest

from agy_orchestrator.workflows import reconcile as R


# --------------------------------------------------------------------------- #
# stub agent
# --------------------------------------------------------------------------- #
class StubAgent:
    """Minimal AgentInstance stand-in: returns canned replies, never spawns."""

    def __init__(self, replies):
        if isinstance(replies, str):
            replies = [replies]
        self._replies = list(replies)
        self.model = "stub"
        self.effort = "high"
        self.prompt = ""
        self.calls = 0

    async def run_async(self):
        self.calls += 1
        if not self._replies:
            return ""
        if len(self._replies) == 1:
            return self._replies[0]
        return self._replies.pop(0)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# _extract_json_blob
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [None, "", "   ", "\n\t ", "no braces here", "[1,2,3]", "}{", "{ unbalanced",
     "```json\n\n```", "prefix } } }"],
)
def test_extract_blob_no_object_returns_none(text):
    assert R._extract_json_blob(text) is None


def test_extract_blob_nested_balanced():
    assert R._extract_json_blob('lead {"a": {"b": 2}} tail') == '{"a": {"b": 2}}'


def test_extract_blob_brace_inside_string_not_counted():
    # A literal "}" inside a JSON string must not prematurely close the object.
    blob = R._extract_json_blob('{"a": "has } brace", "b": 1}')
    assert blob == '{"a": "has } brace", "b": 1}'
    assert json.loads(blob) == {"a": "has } brace", "b": 1}


def test_extract_blob_fenced():
    blob = R._extract_json_blob('```json\n{"x": 1}\n```')
    assert json.loads(blob) == {"x": 1}


def test_extract_blob_huge_prose_then_object_is_fast():
    text = ("blah " * 50000) + '{"findings": []}'
    blob = R._extract_json_blob(text)
    assert json.loads(blob) == {"findings": []}


# --------------------------------------------------------------------------- #
# _coerce_float
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        (True, 1.0),
        (False, 0.0),
        (3, 3.0),
        (-2.5, -2.5),
        (0, 0.0),
        ("nan", 0.0),          # the bare token "nan" has no \d -> 0.0
        ("Infinity", 0.0),     # likewise
        ("-3.5", -3.5),
        ("abc-3.5def", -3.5),  # embedded number
        (None, 0.0),
        ([1, 2], 0.0),
        ({"a": 1}, 0.0),
        ("no number", 0.0),
    ],
)
def test_coerce_float(value, expected):
    assert R._coerce_float(value) == expected


def test_coerce_float_huge_string_no_hang():
    assert R._coerce_float("x" * 200000 + "7") == 7.0


# --------------------------------------------------------------------------- #
# _parse_witness_signal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "stdout,expected",
    [
        (None, None),
        ("", None),
        ("no numbers", None),
        ("foo 3 bar 7", 7.0),
        ("WITNESS: 4.2 trailing 9", 4.2),
        ("WITNESS:1\nWITNESS:2", 2.0),     # last tag wins
        ("-0", -0.0),
        ("value 1e9", 9.0),                # regex captures digit runs, not exponents
    ],
)
def test_parse_witness_signal(stdout, expected):
    got = R._parse_witness_signal(stdout)
    if expected is None:
        assert got is None
    else:
        assert got == expected


# --------------------------------------------------------------------------- #
# _parse_witness / Witness invariants
# --------------------------------------------------------------------------- #
def test_parse_witness_variants():
    assert R._parse_witness(None) == R.Witness()
    assert R._parse_witness(True).value == 1.0
    assert R._parse_witness(5).value == 5.0
    assert R._parse_witness("3.2").value == 3.2
    w = R._parse_witness({"value": 2, "description": "moved"})
    assert w.value == 2.0 and w.description == "moved"
    # non-string/number description coerced to str, never crashes
    w2 = R._parse_witness({"value": "x", "description": 123})
    assert w2.value == 0.0 and w2.description == "123"


def test_witness_is_dead_only_at_zero():
    assert R.Witness(value=0.0).is_dead is True
    assert R.Witness(value=0.5).is_dead is False
    assert R.Witness(value=-1.0).is_dead is False


# --------------------------------------------------------------------------- #
# _parse_finding
# --------------------------------------------------------------------------- #
def test_parse_finding_no_name_returns_none():
    assert R._parse_finding({"classification": "absent"}) is None
    assert R._parse_finding({"name": "   "}) is None
    assert R._parse_finding({}) is None


def test_parse_finding_numeric_name_coerced():
    f = R._parse_finding({"name": 123, "classification": "absent"})
    assert f is not None and f.name == "123"


def test_parse_finding_oov_classification_without_witness_is_dead():
    # Unknown classification + no positive witness => conservatively dead.
    f = R._parse_finding({"name": "m", "classification": "weird"})
    assert f.classification == "exists_not_load_bearing"
    assert f.sub_kind == "uncalled"  # dead finding always carries a sub_kind


def test_parse_finding_oov_classification_with_positive_witness_load_bearing():
    f = R._parse_finding(
        {"name": "m", "classification": "???", "witness": {"value": 4}}
    )
    assert f.classification == "load_bearing"


def test_parse_finding_invalid_sub_kind_dropped_but_dead_defaults():
    f = R._parse_finding(
        {"name": "m", "classification": "exists_not_load_bearing", "sub_kind": "bogus"}
    )
    assert f.sub_kind == "uncalled"


def test_parse_finding_file_line_fallback():
    f = R._parse_finding(
        {"name": "m", "classification": "absent", "file": "a/b.py", "line": 42}
    )
    assert f.location == "a/b.py:42"
    f2 = R._parse_finding(
        {"name": "m", "classification": "absent", "file": "a/b.py"}
    )
    assert f2.location == "a/b.py"


def test_parse_finding_control_chars_in_name_survive():
    f = R._parse_finding({"name": "a\x00b\tc", "classification": "absent"})
    assert f is not None and "\x00" in f.name


# --------------------------------------------------------------------------- #
# parse(): robustness + derived-reconciled-never-trusts-the-model
# --------------------------------------------------------------------------- #
def _review(reply, disposition="warn"):
    return R.ReconciliationReview(StubAgent(reply), goal="g", disposition=disposition)


@pytest.mark.parametrize(
    "reply",
    [
        '{"findings": "oops"}',
        '{"findings": null}',
        '{"findings": [1, 2, "x"]}',
        '{"findings": {}}',
        '{}',
        '{"reconciled": true}',
    ],
)
def test_parse_no_findings_never_reconciled(reply):
    r = _review(reply).parse(reply)
    assert r.parse_error is None
    assert r.findings_list == []
    assert r.reconciled is False           # empty trace is hollow, not "all clear"
    assert r.is_substantive is False


def test_parse_non_object_reply_is_parse_error():
    for reply in ("[1,2,3]", "42", "plain prose", ""):
        r = _review(reply).parse(reply)
        assert r.parse_error is not None
        assert r.reconciled is False


def test_parse_reconciled_derived_not_trusted():
    # Model claims reconciled=true while reporting a dead mechanism: must NOT pass.
    reply = json.dumps(
        {
            "reconciled": True,
            "findings": [
                {"name": "m", "classification": "exists_not_load_bearing",
                 "sub_kind": "stub_constant", "location": "f.py:1"}
            ],
        }
    )
    r = _review(reply).parse(reply)
    assert r.reconciled is False
    assert len(r.findings()) == 1


def test_parse_all_load_bearing_reconciled_true():
    reply = json.dumps(
        {
            "reconciled": False,  # model says false; we derive true
            "findings": [
                {"name": "m", "classification": "load_bearing",
                 "witness": {"value": 3}}
            ],
        }
    )
    r = _review(reply).parse(reply)
    assert r.reconciled is True
    assert r.is_substantive is True


def test_parse_deeply_nested_blob_does_not_crash():
    # Regression guard: a pathologically deep JSON blob used to raise an
    # UNHANDLED RecursionError out of parse(); it must become a parse_error.
    n = 20000
    reply = '{"k":' + "[" * n + "]" * n + "}"
    r = _review(reply).parse(reply)
    assert r.parse_error is not None
    assert r.reconciled is False


def test_parse_invalid_json_blob_is_parse_error():
    r = _review('{"findings": [oops not json]}').parse('{"findings": [oops not json]}')
    assert r.parse_error is not None and "invalid JSON" in r.parse_error


# --------------------------------------------------------------------------- #
# Result invariants: substance gate, starvation, dispositions, to_dict
# --------------------------------------------------------------------------- #
def test_substance_status_variants():
    starved = R.ReconciliationResult(reconciled=True).mark_starved("stalled")
    assert starved.substance_status() == "ran:critic_starved"
    assert starved.reconciled is False
    assert starved.is_substantive is False

    perr = R.ReconciliationResult(reconciled=False, parse_error="x")
    assert perr.substance_status() == "ran:parse_error"

    empty = R.ReconciliationResult(reconciled=False, findings_list=[])
    assert empty.substance_status() == "ran:empty"

    one = R.ReconciliationResult(
        reconciled=True,
        findings_list=[R.MechanismFinding(name="m", classification="load_bearing")],
    )
    assert one.substance_status() == "ran:1_findings"
    assert one.is_substantive is True


def test_mark_starved_empty_reason_defaults():
    r = R.ReconciliationResult(reconciled=True).mark_starved("")
    assert r.starved_reason == "critic_starved"
    assert r.reconciled is False


def test_should_fail_run_only_on_fail_disposition():
    dead = R.MechanismFinding(name="m", classification="exists_not_load_bearing",
                              sub_kind="uncalled")
    warn = R.ReconciliationResult(reconciled=False, findings_list=[dead],
                                  disposition="warn")
    assert warn.should_fail_run is False
    fail = R.ReconciliationResult(reconciled=False, findings_list=[dead],
                                  disposition="fail")
    assert fail.should_fail_run is True


def test_to_dict_roundtrips_and_is_strict_json_for_normal_values():
    reply = json.dumps(
        {
            "findings": [
                {"name": "m", "classification": "load_bearing",
                 "witness": {"value": 2, "description": "moves"},
                 "location": "f.py:9", "evidence": "called at f.py:9"},
                {"name": "d", "classification": "exists_not_load_bearing",
                 "sub_kind": "stub_constant", "witness": {"value": 0}},
            ]
        }
    )
    r = _review(reply).parse(reply)
    d = r.to_dict()
    # strict (allow_nan=False) JSON must serialize cleanly for ordinary values.
    s = json.dumps(d, allow_nan=False)
    back = json.loads(s)
    assert back["substantive"] is True
    assert len(back["findings"]) == 2


def test_disposition_validation():
    with pytest.raises(ValueError):
        R.ReconciliationReview(StubAgent("{}"), goal="g", disposition="nonsense")


# --------------------------------------------------------------------------- #
# execute(): never crash, reprompt path, hollow handling
# --------------------------------------------------------------------------- #
def test_execute_on_garbage_reply_yields_hollow_not_crash():
    rev = R.ReconciliationReview(StubAgent(["no json at all", "still no json"]),
                                 goal="g")
    r = _run(rev.execute())
    assert r.parse_error is not None
    assert r.reconciled is False
    assert r.is_substantive is False


def test_execute_reprompt_recovers():
    good = json.dumps({"findings": [
        {"name": "m", "classification": "load_bearing", "witness": {"value": 1}}]})
    rev = R.ReconciliationReview(StubAgent(["prose only no json", good]), goal="g")
    r = _run(rev.execute())
    assert r.parse_error is None
    assert r.reconciled is True
    assert rev.agent.calls == 2


def test_execute_deeply_nested_reply_does_not_crash():
    n = 20000
    bad = '{"k":' + "[" * n + "]" * n + "}"
    rev = R.ReconciliationReview(StubAgent([bad, bad]), goal="g")
    r = _run(rev.execute())  # must not raise RecursionError
    assert r.parse_error is not None
    assert r.reconciled is False


def test_execute_empty_reply():
    rev = R.ReconciliationReview(StubAgent(["", ""]), goal="g")
    r = _run(rev.execute())
    assert r.reconciled is False
    assert r.parse_error is not None


# --------------------------------------------------------------------------- #
# Ablation hook (#52) — hermetic via sh
# --------------------------------------------------------------------------- #
def test_ablation_cmd_substitution():
    rev = R.ReconciliationReview(
        StubAgent("{}"), goal="g",
        ablation_cmd="echo {MECH} {mech}",
        ablation_cmd_map={"special": "run --only={MECH}"},
    )
    assert rev._ablation_cmd_for("foo") == "echo foo foo"
    assert rev._ablation_cmd_for("special") == "run --only=special"
    rev2 = R.ReconciliationReview(StubAgent("{}"), goal="g")
    assert rev2._ablation_cmd_for("foo") is None  # no template => None


def test_run_witness_signal_and_timeout(tmp_path):
    rev = R.ReconciliationReview(StubAgent("{}"), goal="g", ablation_timeout=0.3)

    async def go():
        sig = await rev._run_witness("echo 42", str(tmp_path), ablate=None)
        nonzero = await rev._run_witness("echo 7; exit 1", str(tmp_path), ablate=None)
        empty = await rev._run_witness("true", str(tmp_path), ablate=None)
        timedout = await rev._run_witness("sleep 5; echo 9", str(tmp_path), ablate=None)
        return sig, nonzero, empty, timedout

    sig, nonzero, empty, timedout = _run(go())
    assert sig == 42.0
    assert nonzero == 7.0          # exit code irrelevant; the number is the signal
    assert empty is None           # no number => "could not measure", not 0
    assert timedout is None        # timeout => None, process killed, no hang


def test_run_witness_ablate_env_isolation(tmp_path):
    rev = R.ReconciliationReview(StubAgent("{}"), goal="g", ablation_timeout=10)
    cmd = 'if [ -n "$AGY_ABLATE" ]; then echo 1; else echo 9; fi'

    async def go():
        clean = await rev._run_witness(cmd, str(tmp_path), ablate=None)
        ablated = await rev._run_witness(cmd, str(tmp_path), ablate="mech")
        return clean, ablated

    clean, ablated = _run(go())
    assert clean == 9.0 and ablated == 1.0


def test_ablation_mismatch_flips_to_dead(tmp_path):
    # Model says load_bearing, but the measured signal does not move => flip to dead.
    cmd = "echo 5"  # identical clean & ablated => delta 0
    rev = R.ReconciliationReview(
        StubAgent("{}"), goal="g", working_directory=str(tmp_path),
        ablation_cmd=cmd, prefer_worktree=False, ablation_timeout=10,
    )
    f = R.MechanismFinding(name="mech", classification="load_bearing",
                           witness=R.Witness(value=3.0))
    res = R.ReconciliationResult(reconciled=True, findings_list=[f])
    out = _run(rev._apply_ablation_witnesses(res))
    flipped = out.findings_list[0]
    assert flipped.classification == "exists_not_load_bearing"
    assert flipped.witness.measured is True
    assert out.reconciled is False  # re-derived after the flip


def test_ablation_real_delta_keeps_load_bearing(tmp_path):
    cmd = 'if [ -n "$AGY_ABLATE" ]; then echo 0; else echo 10; fi'  # delta 10
    rev = R.ReconciliationReview(
        StubAgent("{}"), goal="g", working_directory=str(tmp_path),
        ablation_cmd=cmd, prefer_worktree=False, ablation_timeout=10,
    )
    f = R.MechanismFinding(name="mech", classification="load_bearing",
                           witness=R.Witness(value=0.0))
    res = R.ReconciliationResult(reconciled=True, findings_list=[f])
    out = _run(rev._apply_ablation_witnesses(res))
    kept = out.findings_list[0]
    assert kept.classification == "load_bearing"
    assert kept.witness.value == 10.0 and kept.witness.measured is True
    assert out.reconciled is True


def test_ablation_unmeasurable_keeps_self_report(tmp_path):
    # Command emits no number => cannot measure => self-report untouched.
    cmd = "echo no-number-here"
    rev = R.ReconciliationReview(
        StubAgent("{}"), goal="g", working_directory=str(tmp_path),
        ablation_cmd=cmd, prefer_worktree=False, ablation_timeout=10,
    )
    f = R.MechanismFinding(name="mech", classification="load_bearing",
                           witness=R.Witness(value=3.0))
    res = R.ReconciliationResult(reconciled=True, findings_list=[f])
    out = _run(rev._apply_ablation_witnesses(res))
    kept = out.findings_list[0]
    assert kept.classification == "load_bearing"
    assert kept.witness.measured is False
    assert kept.witness.value == 3.0


def test_ablation_workspace_failure_is_best_effort(tmp_path):
    # working_directory does not exist => candidate_workspace fails; the verdict
    # must be returned untouched (best-effort), never crash.
    rev = R.ReconciliationReview(
        StubAgent("{}"), goal="g",
        working_directory=str(tmp_path / "does-not-exist"),
        ablation_cmd="echo 5", prefer_worktree=False, ablation_timeout=10,
    )
    f = R.MechanismFinding(name="mech", classification="load_bearing",
                           witness=R.Witness(value=3.0))
    res = R.ReconciliationResult(reconciled=True, findings_list=[f])
    out = _run(rev._apply_ablation_witnesses(res))
    assert out.findings_list[0].classification == "load_bearing"
    assert out.findings_list[0].witness.measured is False


def test_ablation_noop_when_no_findings(tmp_path):
    rev = R.ReconciliationReview(
        StubAgent("{}"), goal="g", working_directory=str(tmp_path),
        ablation_cmd="echo 5", prefer_worktree=False,
    )
    res = R.ReconciliationResult(reconciled=False, findings_list=[])
    out = _run(rev._apply_ablation_witnesses(res))
    assert out.findings_list == []
    assert out.reconciled is False


def test_ablation_disabled_is_noop():
    rev = R.ReconciliationReview(StubAgent("{}"), goal="g")
    assert rev.ablation_enabled is False
    f = R.MechanismFinding(name="m", classification="load_bearing")
    res = R.ReconciliationResult(reconciled=True, findings_list=[f])
    out = _run(rev._apply_ablation_witnesses(res))
    assert out is res  # untouched


# --------------------------------------------------------------------------- #
# build_prompt / reprompt: never crash on odd goal/diff
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("goal", ["", "   ", "x" * 50000, "unicode ☃ \x00 nul"])
def test_build_prompt_robust(goal):
    rev = R.ReconciliationReview(StubAgent("{}"), goal=goal, diff="\x1b[31mansi\x1b[0m")
    p = rev.build_prompt()
    assert isinstance(p, str) and R.RECONCILE_MANDATE_PREAMBLE in p
    assert isinstance(rev._json_only_reprompt(), str)


# --------------------------------------------------------------------------- #
# reconcile-2: a non-finite (NaN/Inf) or overflowing witness value must coerce to
# 0.0 (dead) — it is not a real signal. Otherwise `nan == 0` is False, so a NaN
# would read as load-bearing (the exact dead-wiring-passing-as-load-bearing case
# this station exists to catch) AND corrupt the durable reconcile.json with a bare
# `NaN`/`Infinity` token that strict JSON readers reject; a huge int would crash
# parse() with an unhandled OverflowError.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), 10 ** 400, -(10 ** 400)],
)
def test_coerce_float_nonfinite_and_overflow_is_dead(value):
    out = R._coerce_float(value)
    assert out == 0.0                         # not a real signal => dead
    assert math.isfinite(out)


def test_nan_witness_reads_dead_not_load_bearing():
    # A NaN witness must be a DEAD witness (value 0), not silently load-bearing.
    reply = ('{"findings": [{"name": "m", "classification": "load_bearing", '
             '"witness": {"value": NaN}}]}')
    r = _review(reply).parse(reply)
    f = r.findings_list[0]
    assert f.witness.value == 0.0
    assert f.witness.is_dead is True          # was False pre-fix (nan == 0 is False)


def test_nan_witness_unknown_classification_is_dead():
    # No classification + NaN witness: the value>0 gate must NOT bless it.
    reply = '{"findings": [{"name": "m", "witness": {"value": NaN}}]}'
    r = _review(reply).parse(reply)
    f = r.findings_list[0]
    assert f.classification == "exists_not_load_bearing"
    assert r.reconciled is False


def test_to_dict_strict_json_compliant_for_nan_inf_overflow_witness():
    # The persisted reconcile.json (json.dumps allow_nan default) must stay valid
    # JSON even when the model reports NaN/Infinity/overflow witness values.
    for token in ("NaN", "Infinity", "-Infinity", "1" + "0" * 400):
        reply = ('{"findings": [{"name": "m", "classification": "load_bearing", '
                 '"witness": {"value": ' + token + "}}]}")
        r = _review(reply).parse(reply)
        assert r.parse_error is None          # no crash on the huge int
        # allow_nan=False == what a strict / non-Python JSON reader enforces.
        json.dumps(r.to_dict(), allow_nan=False)


def test_execute_nan_witness_reply_does_not_crash_and_persists_clean():
    reply = ('{"findings": [{"name": "m", "classification": "load_bearing", '
             '"witness": {"value": NaN}}]}')
    rev = R.ReconciliationReview(StubAgent([reply, reply]), goal="g")
    r = _run(rev.execute())
    assert r.parse_error is None
    assert r.findings_list[0].witness.value == 0.0
    json.dumps(r.to_dict(), allow_nan=False)  # durable artifact is strict-valid


def test_overflow_int_witness_does_not_crash_parse():
    # Pre-fix: float(10**400) raised an unhandled OverflowError out of parse().
    reply = ('{"findings": [{"name": "m", "classification": "load_bearing", '
             '"witness": {"value": 1' + "0" * 400 + "}}]}")
    r = _review(reply).parse(reply)             # must not raise
    assert r.parse_error is None
    assert r.findings_list[0].witness.value == 0.0


def test_parse_witness_signal_overflow_is_dead_not_inf():
    # A witness command that emits a huge number must not yield an inf signal that
    # would later corrupt reconcile.json; it coerces to 0.0.
    assert R._parse_witness_signal("WITNESS: 1" + "0" * 400) == 0.0
    assert R._parse_witness_signal("1" + "0" * 400) == 0.0

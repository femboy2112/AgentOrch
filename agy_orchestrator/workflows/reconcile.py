"""ReconciliationReview — the goal-vs-runtime Integration-Skeptic station (issue #43).

WHY this exists: the build loop optimizes "pass the tests" and faithfully delivers
exactly that — including, when the tests don't encode the goal, code that is green
and inert. Every other quality mechanism in AgentOrch is *downstream of the test
suite* and therefore inherits its blind spots:

  - the planner holds the goal but never observes the running system;
  - the implementer writes code but doesn't run + introspect the assembled whole;
  - ``QualityVerifier`` knows only ``test_commands`` (goal-blind);
  - ``AdversarialReview`` reviews the static artifact for catastrophic-safety, and
    dead-but-plausible wiring sails through a plausibility check.

No agent in the loop simultaneously holds ``(goal ∧ observed-runtime-behavior)``
*with a mandate to diff them*. That corner is exactly where the
"exists-but-not-load-bearing" defect class lives: a component with real code,
correct names, and a passing unit test that is nonetheless dead / stubbed /
untrained / bypassed / never on the live path.

This station fills the gap. It runs AFTER a build converges and the verifier is
GREEN, holds both the goal and the assembled code (via ``working_directory``), and
for each spec-named mechanism TRACES whether it is actually on the live execution
path — demanding ``file:line`` evidence and an ablation *witness* rather than a
plausibility judgement.

INDEPENDENCE (acceptance #5): the station is READ-ONLY w.r.t. the build artifact
and the test suite. Its prompt forbids editing code/tests — it may only run / trace
/ probe. Its verdict is a DISTINCT status (``ReconciliationResult``), reported
*alongside*, NEVER folded into, ``VerifierResult.ok`` (acceptance #4). Default
disposition is "warn": surface loudly as a distinct status + durable artifact, but
do NOT fail the run until the station is trusted.

Borrows ``adversarial.py``'s think-stripping / event-callback machinery, but with a
goal-vs-runtime mandate instead of a catastrophic-artifact one. Pure, deterministic
parsing; the only non-determinism is the single LLM call (which tests mock).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agy_orchestrator.core.agent import AgentInstance

logger = logging.getLogger(__name__)

# Reasoning-model <think> block; stripped before parsing so a model that muses
# about findings inside its scratchpad can't leak half-formed JSON into the parse.
# Mirrors adversarial.py's _THINK_RE.
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

# The three top-level classifications a mechanism can receive.
CLASSIFICATIONS = ("load_bearing", "exists_not_load_bearing", "absent")

# The required sub-kind for an ``exists_not_load_bearing`` finding — the specific
# way a present-but-dead component fails to be load-bearing.
SUB_KINDS = (
    "uncalled",         # defined, never invoked on the real path
    "stub_constant",    # hardcoded return (surprise()->0.0) shadowing a real impl
    "untrained",        # random-init weights, never in an optimizer / never backward()
    "bypassed_proxy",   # learned signal silently replaced by a cruder count/proxy
    "mocked_none",      # None/default-off, never enabled at scale
    "saturated",        # near-constant, non-discriminative output for all inputs
)

# Dispositions: what a non-reconciled verdict does to the run.
DISPOSITIONS = ("warn", "fail", "open-task")

# The mandate prompt's spine. Forces execution-tracing (not plausibility), file:line
# evidence, a structured ablation witness, and the read-only independence constraint.
RECONCILE_MANDATE_PREAMBLE = (
    "You are the RECONCILIATION / INTEGRATION-SKEPTIC station. The build already "
    "converged and the programmatic test suite is GREEN — your job is NOT to re-run "
    "tests or judge plausibility. Tests passing tells you nothing here: dead, "
    "stubbed, or bypassed code routinely has green unit tests.\n\n"
    "Your single mandate: for EACH mechanism the goal names as load-bearing, TRACE "
    "whether it is actually on the LIVE execution path. Find the real call site or "
    "declare it uncalled. Do NOT ask 'does this look like a plausible implementation' "
    "— that question lets dead-but-present code pass. Ask instead 'is this component "
    "actually REACHED when the real entry point runs, and would removing it change a "
    "live/at-scale signal?'\n\n"
    "INDEPENDENCE (hard constraint): you are READ-ONLY with respect to the build "
    "artifact and the test suite. You MUST NOT edit, patch, rewrite, or delete any "
    "source or test file, and you MUST NOT weaken or relabel any test. You may only "
    "run, trace, and probe. You cannot move the goalposts; you verify against the "
    "goal with no authority to redefine it.\n\n"
    "For each mechanism, classify it as exactly one of:\n"
    "  - load_bearing: invoked on the real path AND ablating it would change a "
    "live/at-scale witness.\n"
    "  - exists_not_load_bearing: present and unit-tested but DEAD. Give the specific "
    "sub_kind (one of: uncalled, stub_constant, untrained, bypassed_proxy, "
    "mocked_none, saturated) and a file:line.\n"
    "  - absent: the goal requires it and it does not exist.\n\n"
    "TEST-ESTABLISHED LOAD-BEARING (do NOT flag as dead): a component whose load-"
    "bearingness is established by the TEST SUITE is not dead wiring. (1) A test "
    "MODULE is invoked by pytest, not by production code — never call it uncalled/dead "
    "because the cert/runtime entrypoint doesn't reach it; that is the expected, "
    "correct state for a test. (2) A deliberately test-gated control/lesion/ablation "
    "arm — off the default live path BY DESIGN and reached by a dedicated passing test "
    "that asserts its behaviour — is load-bearing FOR ITS CLAIM; classify it "
    "load_bearing, not uncalled/bypassed_proxy. Only flag exists_not_load_bearing when "
    "NOTHING exercises the component — neither the runtime path NOR a test.\n\n"
    "LOAD-BEARING WITNESS: for each mechanism report an ablation witness — ablate the "
    "component and state whether a live/at-scale signal moves. witness value 0 means "
    "DEAD WIRING (a bug masquerading as honest-incomplete); witness > 0 with the "
    "metric still low means LEGITIMATELY INCOMPLETE (wired, science/training not there "
    "yet) — that is acceptable, do not flag it.\n\n"
    "Respond with a single JSON object and nothing else:\n"
    "{\n"
    '  "reconciled": <bool>,\n'
    '  "findings": [\n'
    "    {\n"
    '      "name": "<mechanism name>",\n'
    '      "classification": "load_bearing|exists_not_load_bearing|absent",\n'
    '      "sub_kind": "<one of the sub_kinds, or null>",\n'
    '      "location": "path/to/file.py:LINE or null",\n'
    '      "witness": {"value": <number>, "description": "<what moved / didn\'t>"},\n'
    '      "evidence": "<the call site you traced, or why it is uncalled>"\n'
    "    }\n"
    "  ]\n"
    "}\n"
)


@dataclass
class Witness:
    """A load-bearing ablation witness for one mechanism.

    ``value`` is the magnitude by which a live/at-scale signal moved when the
    component was ablated. ``value == 0`` => dead wiring (surface loudly);
    ``value > 0`` with a low metric => legitimately incomplete (acceptable).
    ``description`` is the agent's prose for what it observed.

    ``measured`` is False for the LLM's self-reported witness (the default) and
    True once a programmatic ``--ablation-cmd`` run has REPLACED ``value`` with a
    real with/without-component delta (#52). A measured witness is trusted over
    the model's self-report; an unmeasured one is only a claim.
    """

    value: float = 0.0
    description: str = ""
    measured: bool = False

    @property
    def is_dead(self) -> bool:
        """True iff ablation moved nothing — the component is not load-bearing."""
        return self.value == 0


@dataclass
class MechanismFinding:
    """One spec-named mechanism's reconciliation verdict.

    ``classification`` is one of CLASSIFICATIONS. ``sub_kind`` is required (one of
    SUB_KINDS) when classification is ``exists_not_load_bearing`` and is None
    otherwise. ``location`` is ``file:line`` evidence (None for absent). ``witness``
    carries the ablation signal. ``evidence`` is the traced call site / reason.
    """

    name: str
    classification: str
    sub_kind: Optional[str] = None
    location: Optional[str] = None
    witness: Witness = field(default_factory=Witness)
    evidence: str = ""
    # Set when a finding the model called ``exists_not_load_bearing`` is EXCUSED
    # because its own location is a pytest-collected TEST MODULE — uncalled by runtime
    # BY DESIGN, not dead wiring (#81). The slug is ``"test_module"``. An excused
    # finding is kept in the trace for transparency but is NOT counted as runtime-dead.
    excused: Optional[str] = None

    @property
    def is_dead(self) -> bool:
        """True iff this mechanism exists but is not load-bearing on the RUNTIME path
        (a real defect). An EXCUSED finding — load-bearing-for-its-claim via the test
        suite — is not a runtime defect and does NOT count as dead (#81)."""
        return self.classification == "exists_not_load_bearing" and not self.excused


@dataclass
class ReconciliationResult:
    """The station's verdict — a DISTINCT status, never a ``VerifierResult``.

    ``reconciled`` is True iff there are NO ``exists_not_load_bearing`` findings
    (absent mechanisms are reported but do not, by themselves, flip ``reconciled``;
    they are a planning gap, not dead wiring — callers can inspect them via
    ``absent_findings()``). ``disposition`` controls how a non-reconciled verdict
    affects the run ("warn" by default — report loudly, do NOT fail).

    Deliberately NOT a VerifierResult and NOT auto-truthy: a caller must read
    ``.reconciled`` explicitly, so the goal-vs-runtime verdict can never be
    silently folded into the verifier's ``ok`` (acceptance #4 / #6f).
    """

    reconciled: bool
    findings_list: List[MechanismFinding] = field(default_factory=list)
    disposition: str = "warn"
    raw: str = ""
    parse_error: Optional[str] = None
    # The recon agent's trace was killed/exhausted (watchdog trip or fallback-chain
    # exhaustion) before it could produce a real trace (#59). A starved trace is
    # HOLLOW — it carries no substance, so it must never bless dead wiring as
    # reconciled. ``starved_reason`` is the slug (e.g. "stalled", "fallback_exhausted")
    # for logs / the substance status; None when the trace ran to completion.
    starved_reason: Optional[str] = None

    def mark_starved(self, reason: str) -> "ReconciliationResult":
        """Flag this verdict HOLLOW because the trace agent was starved (#59).

        Records ``starved_reason`` AND forces ``reconciled=False`` regardless of how
        the parse landed: a killed/exhausted critic chain that happened to emit a
        well-formed (or empty) JSON blob must never read as "all clear". Returns
        ``self`` for chaining. ``reason`` must be a non-empty slug.
        """
        self.starved_reason = reason or "critic_starved"
        self.reconciled = False
        return self

    @property
    def is_starved(self) -> bool:
        """True iff the recon agent was killed/exhausted before it could trace (#59)."""
        return self.starved_reason is not None

    @property
    def is_substantive(self) -> bool:
        """True iff this trace actually examined mechanisms (#51).

        A reconcile is substantive only when it parsed cleanly, was not starved,
        and traced at least one mechanism. The hollow variants — a quota-walled /
        watchdog-killed critic (``starved``), an unparseable reply (``parse_error``),
        or a clean-but-empty trace (zero mechanisms) — are NOT substantive and must
        not contribute ``reconciled=true``.
        """
        return (
            not self.is_starved
            and self.parse_error is None
            and len(self.findings_list) >= 1
        )

    def substance_status(self) -> str:
        """Classify the RESULT into a reconcile_status variant (#51 / #59).

        Returns one of (most-degenerate first):
          * ``ran:critic_starved`` — the trace agent was killed/exhausted (#59);
          * ``ran:parse_error``    — the reply carried no parseable verdict;
          * ``ran:empty``          — parsed cleanly but zero mechanisms traced;
          * ``ran:<N>_findings``   — parsed cleanly and traced N>=1 mechanisms.

        The non-``ran:<N>_findings`` variants are HOLLOW (not substantive): CI on
        such a run is green on a reconcile that never traced a mechanism.
        """
        if self.is_starved:
            return "ran:critic_starved"
        if self.parse_error is not None:
            return "ran:parse_error"
        if not self.findings_list:
            return "ran:empty"
        return f"ran:{len(self.findings_list)}_findings"

    def findings(self) -> List[MechanismFinding]:
        """The DEAD-wired findings (``exists_not_load_bearing``) — the actionable set.

        This is the accessor the issue calls for: the list of mechanisms that exist
        but aren't load-bearing, each carrying its sub_kind + file:line + witness.
        """
        return [f for f in self.findings_list if f.is_dead]

    def absent_findings(self) -> List[MechanismFinding]:
        """Mechanisms the goal requires that do not exist at all."""
        return [f for f in self.findings_list if f.classification == "absent"]

    def excused_findings(self) -> List[MechanismFinding]:
        """Findings the model called dead but EXCUSED as a test module (#81).

        These were classified ``exists_not_load_bearing`` by the trace yet point at a
        pytest-collected test module (``excused == "test_module"``); a test being
        uncalled by production code is by design. They are surfaced for transparency
        but do not flip ``reconciled`` or appear in the actionable ``findings()`` set."""
        return [f for f in self.findings_list
                if f.classification == "exists_not_load_bearing" and f.excused]

    def load_bearing_findings(self) -> List[MechanismFinding]:
        """Mechanisms confirmed on the live path with a moving witness."""
        return [f for f in self.findings_list if f.classification == "load_bearing"]

    @property
    def verdict(self) -> str:
        """A short distinct status string for logs/artifacts (never the verifier's)."""
        return "reconciled" if self.reconciled else "not_reconciled"

    @property
    def should_fail_run(self) -> bool:
        """True only when disposition is 'fail' AND the verdict is not reconciled.

        Under the default "warn" disposition this is always False — the station
        surfaces loudly but never fails the run.
        """
        return self.disposition == "fail" and not self.reconciled

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for the durable ``runs/<id>/reconcile.json`` artifact."""
        return {
            "reconciled": self.reconciled,
            "verdict": self.verdict,
            "disposition": self.disposition,
            "parse_error": self.parse_error,
            # #51/#59: whether this trace actually examined mechanisms, and the
            # substance variant (ran:<N>_findings / ran:empty / ran:parse_error /
            # ran:critic_starved). A reader can gate on these without re-deriving.
            "substantive": self.is_substantive,
            "substance_status": self.substance_status(),
            "starved_reason": self.starved_reason,
            "findings": [
                {
                    "name": f.name,
                    "classification": f.classification,
                    "sub_kind": f.sub_kind,
                    "excused": f.excused,
                    "location": f.location,
                    "witness": {
                        "value": f.witness.value,
                        "description": f.witness.description,
                        "measured": f.witness.measured,
                    },
                    "evidence": f.evidence,
                }
                for f in self.findings_list
            ],
        }


def _strip_thinking(text: str) -> str:
    """Drop any ``<think>...</think>`` block (mirrors adversarial.py)."""
    return _THINK_RE.sub("", text or "")


def _extract_json_blob(text: str) -> Optional[str]:
    """Pull the JSON object out of a model reply, tolerating fences + prose.

    Strategy (most-specific first):
      1. a ```json ... ``` (or bare ``` ... ```) fenced block;
      2. otherwise the first balanced ``{...}`` object by brace-matching (so a
         trailing "Hope this helps!" or a leading sentence doesn't break parsing).
    Returns the candidate JSON string, or None if no object-looking span is found.
    """
    cleaned = _strip_thinking(text).strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.IGNORECASE | re.DOTALL)
    if fence:
        inner = fence.group(1).strip()
        if inner:
            cleaned = inner

    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]
    return None


# Control characters that even ``strict=False`` cannot tolerate — ones that land
# OUTSIDE a JSON string (between tokens). \t \n \r and any control char INSIDE a
# string value are accepted by ``strict=False``; everything else in 0x00-0x1F is
# stray noise we strip only as a last resort. (#74)
_STRAY_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _loads_repaired(blob: str) -> Optional[Any]:
    """Best-effort repair of near-valid reconcile JSON (#74).

    Returns the parsed object, or None when the blob is GENUINELY malformed
    (structural error) — so real corruption still reads as a parse_error.

    A reconcile agent occasionally emits a literal control character (an unescaped
    newline/tab) inside a JSON string value, which strict ``json.loads`` rejects
    with "Invalid control character at ...". That is a cosmetic formatting glitch,
    not a semantic defect. Two repair passes, most-faithful first:
      1. ``json.loads(strict=False)`` — permits control characters inside string
         values, preserving the value verbatim;
      2. strip stray non-whitespace control chars (those outside any string, which
         even ``strict=False`` rejects) and retry leniently.
    """
    for attempt in (blob, _STRAY_CONTROL_RE.sub("", blob)):
        try:
            return json.loads(attempt, strict=False)
        except (json.JSONDecodeError, RecursionError):
            continue
    return None


def _finite_or_dead(value: float) -> float:
    """Map a non-finite witness magnitude (NaN/±Inf) to 0.0 (dead).

    A NaN/Inf is not a real ablation signal — it carries no numeric witness, so it
    must read as DEAD WIRING (value 0), not as load-bearing. Crucially this also
    keeps a non-JSON-compliant ``NaN``/``Infinity`` token out of the durable
    ``reconcile.json`` artifact: ``Witness.is_dead`` checks ``value == 0`` and
    ``nan == 0`` is False, so a NaN witness would otherwise silently bless a
    mechanism as load-bearing AND corrupt the persisted JSON (strict readers reject
    the bare ``NaN``)."""
    if value != value or value in (float("inf"), float("-inf")):
        return 0.0
    return value


def _coerce_float(value: Any) -> float:
    """Best-effort numeric coercion for a witness value; non-numeric -> 0.0 (dead).

    Non-finite results (NaN, ±Inf) and values too large to represent as a float are
    coerced to 0.0 (dead): they are not a real numeric witness, and they must never
    crash parsing (OverflowError on a giant int) or corrupt the durable
    ``reconcile.json`` (a bare ``NaN``/``Infinity`` token is not valid JSON)."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        try:
            return _finite_or_dead(float(value))
        except (OverflowError, ValueError):
            # e.g. an int too large to convert to a float -> no real signal -> dead.
            return 0.0
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if m:
            try:
                return _finite_or_dead(float(m.group(0)))
            except (ValueError, OverflowError):
                return 0.0
    return 0.0


# A witness command reports its signal as a number on stdout — by convention the
# LAST numeric token (so a verbose harness can log freely and still emit the metric
# last). ``WITNESS:<n>`` is honored first when present so a command can be explicit.
_WITNESS_TAG_RE = re.compile(r"WITNESS:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_witness_signal(stdout: str) -> Optional[float]:
    """Extract the numeric witness signal a witness command emitted on stdout.

    Prefers an explicit ``WITNESS:<n>`` tag (last occurrence wins so a re-run that
    prints progress can still pin the metric); otherwise falls back to the LAST
    bare number in the output. Returns None when no number is present at all (so
    the caller can tell "ran but emitted no signal" apart from "signal == 0").
    """
    tags = _WITNESS_TAG_RE.findall(stdout or "")
    if tags:
        try:
            return _finite_or_dead(float(tags[-1]))
        except (ValueError, OverflowError):
            pass
    nums = _NUMBER_RE.findall(stdout or "")
    if nums:
        try:
            return _finite_or_dead(float(nums[-1]))
        except (ValueError, OverflowError):
            return None
    return None


def _parse_witness(raw: Any) -> Witness:
    """Parse a witness field that may be a dict, a bare number, or missing."""
    if isinstance(raw, dict):
        return Witness(
            value=_coerce_float(raw.get("value", 0.0)),
            description=str(raw.get("description", "") or ""),
        )
    if isinstance(raw, (int, float, str)):
        return Witness(value=_coerce_float(raw), description="")
    return Witness()


def _parse_finding(raw: Dict[str, Any]) -> Optional[MechanismFinding]:
    """Parse one finding dict into a MechanismFinding, normalizing fields.

    Robust to a model that omits sub_kind for a dead finding, supplies an
    out-of-vocabulary classification, or stuffs the location into 'file'/'line'
    instead of 'location'. Returns None only when there is no usable name.
    """
    name = str(raw.get("name", "") or "").strip()
    if not name:
        return None

    classification = str(raw.get("classification", "") or "").strip().lower()
    if classification not in CLASSIFICATIONS:
        # Unknown/blank classification: treat as load_bearing only when an
        # explicit positive witness backs it; otherwise be conservative and call
        # it dead so a malformed reply can't hide a defect.
        witness_peek = _parse_witness(raw.get("witness"))
        classification = "load_bearing" if witness_peek.value > 0 else "exists_not_load_bearing"

    sub_kind = raw.get("sub_kind")
    sub_kind = str(sub_kind).strip().lower() if sub_kind not in (None, "") else None
    if sub_kind is not None and sub_kind not in SUB_KINDS:
        sub_kind = None

    location = raw.get("location")
    if not location:
        f = raw.get("file")
        ln = raw.get("line")
        if f and ln not in (None, ""):
            location = f"{f}:{ln}"
        elif f:
            location = str(f)
    location = str(location).strip() if location else None

    witness = _parse_witness(raw.get("witness"))

    finding = MechanismFinding(
        name=name,
        classification=classification,
        sub_kind=sub_kind,
        location=location,
        witness=witness,
        evidence=str(raw.get("evidence", "") or ""),
    )

    # A dead finding MUST carry a sub_kind; if the model didn't give a valid one,
    # default to the most generic ("uncalled") so the finding stays actionable.
    if finding.is_dead and finding.sub_kind is None:
        finding.sub_kind = "uncalled"
    return finding


# --- test-module excusal (#81) -------------------------------------------------
# A reconcile traces reachability from the PRODUCTION / cert entrypoint ONLY, so a
# TEST MODULE is misread as dead wiring: it is uncalled by runtime BY DESIGN (pytest
# invokes it, not the cert entrypoint), yet flagging it ``uncalled`` flips
# reconciled=False and hard-fails the build under disposition=fail. A test module
# being uncalled by production code is the EXPECTED, correct state — never a defect.
# We excuse such findings here (kept in the trace for transparency, not counted as
# dead wiring).
#
# We deliberately do NOT try to deterministically excuse the issue's OTHER case — a
# test-gated control/lesion arm living in PRODUCTION code. Distinguishing such an arm
# (load-bearing-for-its-claim because a test drives its gate and ASSERTS a behavioural
# difference) from a genuine dead stub that merely happens to have a unit test is a
# SEMANTIC judgement about what the test asserts — not something a grep can do safely.
# An adversarial fuzz pass confirmed any text-matching heuristic over-suppresses real
# dead wiring (a stub with a unit test + "lesion"/"control" prose would be excused),
# which is the exact silent-hollow-win this station exists to prevent. That case is
# handled at the right layer instead: the RECONCILE_MANDATE_PREAMBLE instructs the
# tracing agent (which can read the test and judge its assertion) to classify a
# test-gated control arm as load_bearing rather than emitting a dead finding at all.

# pytest's default collection globs: a finding at one of these IS a test module,
# regardless of directory. A file merely living under a tests/ dir but NOT matching
# (fixtures, helpers, data, non-pytest shims) is NOT treated as a test — it may be
# production code, and excusing it would silently bless dead wiring.
_TEST_FILE_RE = re.compile(r"(?:^|/)(?:test_[^/]*\.py|[^/]*_test\.py|conftest\.py)$")


def _location_path(location: Optional[str]) -> Optional[str]:
    """The file part of a ``path:line[:col]`` location string (forward-slashed), or None."""
    if not location:
        return None
    loc = str(location).strip()
    # Drop a pytest nodeid tail (file.py::Class::method) before the :line suffix, so a
    # legitimately test-located finding reported as a nodeid is still recognized.
    loc = loc.split("::", 1)[0].strip()
    m = re.match(r"^(.*?\.py)(?::\d+)*$", loc, re.IGNORECASE)
    return (m.group(1) if m else loc).replace("\\", "/")


def _is_test_path(location: Optional[str]) -> bool:
    """True iff ``location`` is a pytest-COLLECTED test module (by filename convention:
    ``test_*.py`` / ``*_test.py`` / ``conftest.py``).

    Intentionally strict: a file merely under a ``tests/`` (or ``test/``) directory but
    NOT matching the collection globs is NOT a test — it could be production code, and
    excusing it would silently bless dead wiring (the over-suppression the station
    exists to prevent)."""
    path = _location_path(location)
    if not path:
        return False
    return bool(_TEST_FILE_RE.search(path.lower()))


class ReconciliationReview:
    """Post-verifier station: trace each spec-named mechanism to the live path.

    Constructor mirrors ``AdversarialReview``'s shape. The agent does the tracing;
    this class owns the prompt mandate, robust output parsing, the distinct verdict
    object, and the warn-only-by-default disposition.

    Args:
        agent: an ``AgentInstance`` (already model/effort-configured by the caller's
            routing) that performs the trace. Its ``run_async()`` is awaited once.
        goal: the spec / intent text — what the build was supposed to ACHIEVE. This
            is the side the rest of the loop never holds at runtime.
        verifier: optional, accepted for symmetry with the other workflows so the
            caller can pass the same object; this station NEVER folds its verdict
            into the verifier's ``ok`` (it is informational here only).
        working_directory: where the assembled code lives — threaded so a cross-repo
            dispatch (out_dir != PROJECT_ROOT) traces the right tree.
        disposition: "warn" (default) | "fail" | "open-task". Controls only how a
            non-reconciled verdict affects the run; "warn" never fails it.
        diff: optional changed-surface text, for scoping which mechanisms to trace.
        event_callback: optional, no-op-safe lifecycle sink (mirrors adversarial.py).
        max_iterations: kept for signature symmetry; the trace is a single pass, so
            values > 1 do not loop (the station is deterministic given the reply).
        ablation_cmd: optional programmatic ablation-witness template (#52). When
            given, the station MEASURES each mechanism's load-bearing witness instead
            of trusting the model's self-report: it runs the command twice in a
            throwaway worktree (once clean, once with ``AGY_ABLATE=<mech>`` set so the
            project disables that component) and records the real signal delta in
            ``Witness.value``. The template may contain ``{MECH}`` (substituted with
            the mechanism name); whatever the command emits as its last number (or a
            ``WITNESS:<n>`` tag) is the signal. Read-only: it never mutates the build
            tree. None (default) => byte-identical to the pre-#52 self-report path.
        ablation_cmd_map: optional ``{mechanism_name: cmd}`` override map; a mechanism
            listed here uses its own witness command instead of ``ablation_cmd``.
        ablation_timeout: per-command wall-clock cap in seconds for a witness run
            (each of the clean + ablated runs); None => AGY_ABLATION_TIMEOUT or 600.
        prefer_worktree: forwarded to the throwaway-workspace isolation (clean repos
            get a fast git worktree; dirty/non-git fall back to a copy).
    """

    def __init__(
        self,
        agent: AgentInstance,
        goal: str,
        verifier: Optional[object] = None,
        working_directory: str = ".",
        disposition: str = "warn",
        diff: Optional[str] = None,
        event_callback: Optional[Callable[[dict], None]] = None,
        max_iterations: int = 1,
        ablation_cmd: Optional[str] = None,
        ablation_cmd_map: Optional[Dict[str, str]] = None,
        ablation_timeout: Optional[float] = None,
        prefer_worktree: bool = True,
    ):
        if disposition not in DISPOSITIONS:
            raise ValueError(
                f"disposition must be one of {DISPOSITIONS} (got {disposition!r})"
            )
        self.agent = agent
        self.goal = goal
        self.verifier = verifier
        self.working_directory = working_directory
        self.disposition = disposition
        self.diff = diff
        self.event_callback = event_callback
        self.max_iterations = max_iterations
        # Programmatic ablation-witness hook (#52). OPT-IN: None leaves the station
        # byte-identical to the LLM-self-report path.
        self.ablation_cmd = ablation_cmd or None
        self.ablation_cmd_map = dict(ablation_cmd_map or {})
        if ablation_timeout is None:
            import os
            ablation_timeout = float(os.environ.get("AGY_ABLATION_TIMEOUT", "600") or 0)
        self.ablation_timeout = ablation_timeout
        self.prefer_worktree = prefer_worktree
        # Populated by execute() for the run ledger / dashboard.
        self.result: Optional[ReconciliationResult] = None
        self.reconciled: Optional[bool] = None

    @property
    def ablation_enabled(self) -> bool:
        """True iff a programmatic ablation witness is configured (#52)."""
        return bool(self.ablation_cmd or self.ablation_cmd_map)

    # --- event plumbing (no-op-safe; mirrors adversarial.py) -------------------
    def _emit_orchestration(self, **fields) -> None:
        cb = self.event_callback
        if cb is None:
            return
        orchestration = {"workflow": "reconcile"}
        for key, value in fields.items():
            if value is not None:
                orchestration[key] = value
        try:
            cb(
                {
                    "kind": "lifecycle",
                    "data": {
                        "event": "orchestration_transition",
                        "orchestration": orchestration,
                    },
                }
            )
        except Exception:
            pass

    def build_prompt(self) -> str:
        """Assemble the trace prompt: mandate + goal + working dir (+ optional diff)."""
        parts = [
            RECONCILE_MANDATE_PREAMBLE,
            "\n--- GOAL (what this build was supposed to ACHIEVE) ---\n",
            self.goal.strip(),
            "\n\n--- ASSEMBLED CODE ---\n",
            f"The assembled code is in: {self.working_directory}\n"
            "Read it, run its real entry point, and trace each goal-named mechanism "
            "to the live path. Use read-only probes only.\n",
        ]
        if self.diff:
            parts.append(
                "\n--- CHANGED SURFACE (for scoping) ---\n" + self.diff.strip() + "\n"
            )
        return "".join(parts)

    def _json_only_reprompt(self) -> str:
        """The trace prompt + a strict JSON-only instruction, used to recover from a
        first reply that carried no parseable JSON object (#68)."""
        return (
            self.build_prompt()
            + "\n\n--- OUTPUT CONTRACT (STRICT) ---\n"
            "Your previous reply contained NO JSON object and could not be parsed. "
            "Reply with ONLY a single JSON object for the verdict — no prose, no "
            "explanation, no markdown fences, no thinking. The first character of "
            "your reply must be '{' and the last must be '}'.\n"
        )

    def parse(self, reply: str) -> ReconciliationResult:
        """Parse a model reply into a ReconciliationResult.

        Tolerates ```json fences, thinking blocks, and surrounding prose. The
        top-level ``reconciled`` is RECOMPUTED from the findings (True iff no
        ``exists_not_load_bearing``) so a model that contradicts itself ("reconciled:
        true" alongside a stub_constant finding) cannot bless dead wiring.
        """
        blob = _extract_json_blob(reply)
        parse_error: Optional[str] = None
        findings: List[MechanismFinding] = []
        if blob is None:
            parse_error = "no JSON object found in reply"
        else:
            data = {}
            try:
                data = json.loads(blob)
            except json.JSONDecodeError as exc:
                # A reconcile reply sometimes embeds a RAW control character (an
                # unescaped newline/tab) inside a JSON string value, which strict
                # json.loads rejects with "Invalid control character at ...". That
                # is a cosmetic formatting glitch, not a semantic defect — failing
                # on it disabled the anti-hollow-win gate on a mission-critical run
                # (#74). Repair before giving up; only GENUINELY malformed structure
                # stays a parse_error (so a real corruption still reads as hollow).
                repaired = _loads_repaired(blob)
                if repaired is None:
                    parse_error = f"invalid JSON: {exc}"
                    data = {}
                else:
                    data = repaired
                    logger.info(
                        "reconcile: repaired near-valid JSON (%s) — verdict "
                        "recovered instead of bailing hollow (#74).", exc.msg)
            except RecursionError as exc:
                # A pathologically deep ``{...}``/``[...]`` blob blows Python's
                # parser recursion limit. That is a malformed reply, not a station
                # bug — treat it as a parse_error rather than letting the
                # RecursionError escape parse()/execute() and crash the run.
                parse_error = f"invalid JSON: input nested too deeply ({exc})"
                data = {}
            raw_findings = data.get("findings") if isinstance(data, dict) else None
            if isinstance(raw_findings, list):
                for raw in raw_findings:
                    if isinstance(raw, dict):
                        parsed = _parse_finding(raw)
                        if parsed is not None:
                            findings.append(parsed)

        # reconciled is DERIVED from the findings, not trusted from the model.
        reconciled = not any(f.is_dead for f in findings)
        # A HOLLOW reply must not silently pass as "all clear" (#51): neither an
        # unparseable verdict NOR a clean-but-EMPTY trace (zero mechanisms examined)
        # may contribute reconciled=true. Only a trace that actually examined >= 1
        # mechanism and found no dead wiring earns reconciled=true.
        if not findings:
            reconciled = False

        return ReconciliationResult(
            reconciled=reconciled,
            findings_list=findings,
            disposition=self.disposition,
            raw=reply,
            parse_error=parse_error,
        )

    def _ablation_cmd_for(self, mechanism: str) -> Optional[str]:
        """The witness command to run for ``mechanism`` (#52), or None.

        A per-mechanism entry in ``ablation_cmd_map`` wins over the shared
        ``ablation_cmd`` template; ``{MECH}`` (and ``{mech}``) is substituted with
        the mechanism name in either."""
        template = self.ablation_cmd_map.get(mechanism) or self.ablation_cmd
        if not template:
            return None
        return template.replace("{MECH}", mechanism).replace("{mech}", mechanism)

    async def _run_witness(
        self, cmd: str, cwd: str, ablate: Optional[str]
    ) -> Optional[float]:
        """Run ONE witness command in ``cwd`` and return its numeric signal (#52).

        When ``ablate`` is set, ``AGY_ABLATE=<mechanism>`` is added to the child env
        so a project that honors the flag disables that component for this run. The
        signal is the last number (or a ``WITNESS:<n>`` tag) the command prints on
        stdout; None when it fails, times out, or emits no number — the caller treats
        a missing signal as "could not measure", NOT as 0."""
        import os
        env = dict(os.environ)
        if ablate:
            env["AGY_ABLATE"] = ablate
        else:
            # Never leak an outer AGY_ABLATE into the clean baseline run.
            env.pop("AGY_ABLATE", None)
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=cwd, env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            comm = proc.communicate()
            if self.ablation_timeout and self.ablation_timeout > 0:
                out, _err = await asyncio.wait_for(comm, self.ablation_timeout)
            else:
                out, _err = await comm
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), 5)
            except Exception:
                pass
            logger.warning(
                "Ablation witness timed out after %ss (ablate=%s): %s",
                self.ablation_timeout, ablate, cmd,
            )
            return None
        except Exception as exc:
            logger.warning("Ablation witness failed to launch (ablate=%s): %s — %s",
                           ablate, cmd, exc)
            return None
        return _parse_witness_signal(out.decode(errors="replace"))

    async def _measure_finding(self, finding: MechanismFinding, cwd: str) -> None:
        """Measure ``finding``'s real ablation delta in ``cwd`` and reconcile it (#52).

        Runs the mechanism's witness command twice — clean, then with
        ``AGY_ABLATE=<name>`` — and writes the absolute signal delta into
        ``finding.witness.value`` (marking it ``measured``). When the model CLAIMED
        the mechanism load-bearing (or reported a positive witness) but the
        measurement shows the signal did NOT move (delta == 0), the finding is
        flipped to ``exists_not_load_bearing`` and logged loudly — the measurement
        overrides the self-report. A run that can't produce a number on both passes
        is left as the model reported it (we don't have a contradicting measurement)."""
        cmd = self._ablation_cmd_for(finding.name)
        if not cmd:
            return
        claimed_load_bearing = (
            finding.classification == "load_bearing" or finding.witness.value > 0
        )
        baseline = await self._run_witness(cmd, cwd, ablate=None)
        ablated = await self._run_witness(cmd, cwd, ablate=finding.name)
        if baseline is None or ablated is None:
            logger.warning(
                "Ablation witness for %r produced no measurable signal "
                "(baseline=%s, ablated=%s); keeping the self-reported witness.",
                finding.name, baseline, ablated,
            )
            return

        delta = abs(baseline - ablated)
        finding.witness.value = delta
        finding.witness.measured = True
        finding.witness.description = (
            f"measured ablation delta {delta:g} "
            f"(baseline={baseline:g}, ablated[{finding.name}]={ablated:g})"
        )
        self._emit_orchestration(
            phase="reconcile",
            action="ablation_measured",
            mechanism=finding.name,
            measured_delta=delta,
        )

        if claimed_load_bearing and delta == 0:
            # Model said it moves; measurement says it doesn't. Trust the
            # measurement: this is dead wiring masquerading as load-bearing.
            logger.warning(
                "Ablation MISMATCH for %r: model reported load-bearing but the "
                "measured signal did not move (delta=0, baseline=ablated=%g). "
                "Flipping finding to exists_not_load_bearing.",
                finding.name, baseline,
            )
            finding.classification = "exists_not_load_bearing"
            if finding.sub_kind is None:
                finding.sub_kind = "uncalled"
            self._emit_orchestration(
                phase="reconcile",
                action="ablation_mismatch",
                mechanism=finding.name,
                measured_delta=delta,
            )

    async def _apply_ablation_witnesses(
        self, result: ReconciliationResult
    ) -> ReconciliationResult:
        """Measure every finding's witness in a throwaway worktree, then re-derive.

        Read-only w.r.t. the build tree: the witness commands run inside an isolated
        ``candidate_workspace`` copy/worktree, never the live ``working_directory``.
        After measuring, ``reconciled`` is recomputed because a mismatch may have
        flipped a finding to dead. The whole step is best-effort: a workspace or
        command failure logs and leaves the self-reported verdict untouched."""
        if not self.ablation_enabled or not result.findings_list:
            return result
        # Lazy import: agy_orchestrator.execution is a peer; keep reconcile importable
        # even where the isolation backend's deps (git) are unusual.
        from agy_orchestrator.execution.workspace import candidate_workspace

        try:
            async with candidate_workspace(
                self.working_directory,
                candidate_id="ablate",
                prefer_worktree=self.prefer_worktree,
            ) as (ws_path, _backend):
                for finding in result.findings_list:
                    try:
                        await self._measure_finding(finding, str(ws_path))
                    except Exception as exc:  # one bad witness can't sink the rest
                        logger.warning(
                            "Ablation measurement raised for %r (continuing): %s",
                            finding.name, exc,
                        )
        except Exception as exc:  # best-effort: never fail a build on the witness hook
            logger.warning(
                "Ablation-witness workspace failed (%s); keeping self-reported "
                "witnesses.", exc,
            )
            return result

        # A flip to dead may have changed the verdict; re-derive it the same way
        # parse() does (a hollow/empty trace still cannot read as reconciled).
        result.reconciled = bool(result.findings_list) and not any(
            f.is_dead for f in result.findings_list
        )
        return result

    def _excuse_test_exercised_findings(
        self, result: ReconciliationResult
    ) -> ReconciliationResult:
        """Excuse dead findings that point at a pytest-collected TEST MODULE (#81).

        The classifier traces reachability from the cert/production entrypoint ONLY, so
        a test module is misread as dead wiring (``uncalled``) → flip reconciled=False →
        hard-fail under disposition=fail. A test module being uncalled by production
        code is the EXPECTED, correct state, never a defect. Mark such findings
        ``excused`` (kept in the trace, not counted as dead) and re-derive the verdict.

        Scope is deliberately narrow: only a finding whose own ``location`` is a pytest
        module (``test_*.py`` / ``*_test.py`` / ``conftest.py``) is excused — a strictly
        decidable fact. The issue's other case (a test-gated control/lesion arm in
        PRODUCTION code) is NOT excused here: telling such an arm apart from a genuine
        dead stub that merely has a unit test requires judging what the test ASSERTS,
        which a grep cannot do without over-suppressing real dead wiring; that case is
        steered at the source by the mandate preamble instead. Genuine runtime-dead and
        MEASURED-dead findings are untouched. Best-effort — never raises into execute()."""
        dead = [f for f in result.findings_list
                if f.classification == "exists_not_load_bearing" and not f.excused]
        if not dead:
            return result
        changed = False
        for f in dead:
            # A test being uncalled by production code is BY DESIGN. (A pytest module is
            # never ablation-measured, so excusing it regardless of witness.measured is
            # sound — _is_test_path matches only true collection-glob filenames.)
            if _is_test_path(f.location):
                f.excused = "test_module"
                changed = True
        if changed:
            excused = result.excused_findings()
            logger.warning(
                "Reconciliation: excused %d test-module finding(s) — uncalled by "
                "runtime BY DESIGN, not dead wiring (#81): %s",
                len(excused),
                ", ".join(f"{f.name}[{f.excused}]" for f in excused),
            )
            self._emit_orchestration(
                phase="reconcile", action="excused_test_module",
                excused_count=len(excused),
            )
            # Re-derive: a build with NO remaining runtime-dead findings reconciles
            # (the hollow-empty rule still holds — an empty trace can't read green; a
            # starved/hollow verdict can never be resurrected to green by excusal).
            result.reconciled = (
                not result.is_starved
                and bool(result.findings_list)
                and not any(f.is_dead for f in result.findings_list)
            )
        return result

    async def _cycle_providers_for_verdict(self) -> Optional[ReconciliationResult]:
        """#79: rotate the fallback chain's LEAD through every remaining provider,
        re-asking (JSON-only) until one returns a parseable verdict.

        A reconcile ``parse_error`` is a successful produce of prose-not-JSON, so the
        ``FallbackAgent`` (which only cycles on a produce-FAILURE) never advances —
        the lead provider's formatting flakiness alone would false-fail an otherwise
        fully-verified run under ``disposition=fail``. We drive the rotation here with
        the same ``rotate_offset`` lever the adversarial loop uses (#65). Returns the
        first parseable result, or the last parse_error result if none parse, or
        ``None`` when the agent is not a multi-provider chain (nothing to cycle).
        Best-effort: never raises into the caller; restores ``rotate_offset``."""
        chain = list(getattr(self.agent, "_chain", []) or [])
        if len(chain) <= 1:
            return None  # single provider: no sibling to try
        prior_offset = int(getattr(self.agent, "rotate_offset", 0) or 0)
        last: Optional[ReconciliationResult] = None
        try:
            # Try each OTHER provider as lead exactly once (offsets 1..N-1).
            for rot in range(1, len(chain)):
                provider_name = chain[rot % len(chain)].__name__
                try:
                    self.agent.rotate_offset = rot
                    self._emit_orchestration(
                        phase="reconcile", action="provider_cycle",
                        to_worker=provider_name, reason="parse_error", attempt=rot,
                    )
                    self.agent.prompt = self._json_only_reprompt()
                    reply = await self.agent.run_async()
                    candidate = self.parse(reply)
                except Exception as exc:  # a walled/failed provider -> try the next
                    logger.warning(
                        "Reconciliation provider-cycle to %s failed: %s",
                        provider_name, exc,
                    )
                    continue
                last = candidate
                if candidate.parse_error is None:
                    logger.warning(
                        "Reconciliation recovered a parseable verdict from %s after "
                        "the lead provider emitted unparseable JSON twice (#79).",
                        provider_name,
                    )
                    return candidate
        finally:
            try:
                self.agent.rotate_offset = prior_offset
            except Exception:
                pass
        return last

    async def execute(self) -> ReconciliationResult:
        """Run the single trace pass and return the distinct verdict object."""
        logger.info(
            "Reconciliation station: tracing goal-named mechanisms to the live path "
            "in %s (disposition=%s)",
            self.working_directory,
            self.disposition,
        )
        self._emit_orchestration(
            phase="reconcile",
            action="trace_started",
            disposition=self.disposition,
            model=getattr(self.agent, "model", None),
            effort=getattr(self.agent, "effort", None),
        )

        self.agent.prompt = self.build_prompt()
        reply = await self.agent.run_async()
        result = self.parse(reply)

        # #68: a reply with no parseable JSON object means the anti-hollow-win
        # trace produced NO verdict — the station would otherwise silently no-op
        # (disposition=warn) while the run still reports "verified" from the unit
        # verifier alone, exactly the hollow-win this station exists to catch.
        # Re-ask ONCE with a strict JSON-only reprompt before accepting a
        # parse_error (most are the model wrapping the verdict in prose).
        if result.parse_error is not None:
            logger.warning(
                "Reconciliation reply had no parseable verdict (%s); re-prompting "
                "for JSON-only output before declaring parse_error (#68).",
                result.parse_error,
            )
            self._emit_orchestration(
                phase="reconcile", action="reprompt_json", reason=result.parse_error,
            )
            try:
                self.agent.prompt = self._json_only_reprompt()
                reply2 = await self.agent.run_async()
                result2 = self.parse(reply2)
                if result2.parse_error is None:
                    result = result2
                else:
                    # #79: the #68 re-prompt re-asks the SAME lead provider. A
                    # parse_error is a SUCCESSFUL produce (prose, not JSON), so the
                    # fallback chain never cycled on its own — one provider's JSON-
                    # formatting flakiness would false-fail an otherwise-verified run
                    # under disposition=fail without ever asking the OTHER providers.
                    # Cycle the lead through the remaining providers (mirroring the
                    # produce-failure path + the adversarial #65 rotation) until one
                    # yields a parseable verdict, or all are exhausted.
                    cycled = await self._cycle_providers_for_verdict()
                    if cycled is not None and cycled.parse_error is None:
                        result = cycled
                    else:
                        logger.error(
                            "Reconciliation STILL unparseable after a JSON-only "
                            "re-prompt AND cycling every provider in the chain — the "
                            "anti-hollow-win check did NOT verify this run; treating "
                            "as hollow (disposition=%s).",
                            self.disposition,
                        )
            except Exception as exc:  # best-effort: keep the original parse_error
                logger.warning("Reconciliation JSON-only re-prompt failed: %s", exc)

        # Programmatic ablation-witness hook (#52): when a witness command is
        # configured, MEASURE each mechanism's load-bearing signal in a throwaway
        # worktree and cross-check the model's self-report. Opt-in + best-effort; a
        # no-op when no command is set, so the verdict is byte-identical otherwise.
        if self.ablation_enabled:
            result = await self._apply_ablation_witnesses(result)

        # #81: the trace only follows the cert/production entrypoint, so a TEST MODULE
        # (uncalled by runtime by design) is misread as dead wiring and hard-fails
        # disposition=fail. Excuse those (kept in the trace, not counted dead) and
        # re-derive the verdict. Runtime-dead findings stay dead; test-gated control
        # arms in production code are handled by the mandate preamble, not here.
        try:
            result = self._excuse_test_exercised_findings(result)
        except Exception as exc:  # best-effort: never fail a build on the excusal pass
            logger.warning("Reconciliation test-exercise excusal raised (continuing): %s", exc)

        self.result = result
        self.reconciled = result.reconciled

        dead = result.findings()
        if dead:
            logger.warning(
                "Reconciliation: %d exists-but-not-load-bearing finding(s) (%s). "
                "Verdict=%s, disposition=%s.",
                len(dead),
                ", ".join(f"{f.name}:{f.sub_kind}@{f.location}" for f in dead),
                result.verdict,
                result.disposition,
            )
        else:
            logger.info("Reconciliation: no dead wiring found (verdict=%s).", result.verdict)

        self._emit_orchestration(
            phase="reconcile",
            action="trace_completed",
            outcome=result.verdict,
            reconciled=result.reconciled,
            dead_count=len(dead),
            disposition=result.disposition,
        )
        return result

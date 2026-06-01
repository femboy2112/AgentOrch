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
    """

    value: float = 0.0
    description: str = ""

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

    @property
    def is_dead(self) -> bool:
        """True iff this mechanism exists but is not load-bearing (a real defect)."""
        return self.classification == "exists_not_load_bearing"


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

    def findings(self) -> List[MechanismFinding]:
        """The DEAD-wired findings (``exists_not_load_bearing``) — the actionable set.

        This is the accessor the issue calls for: the list of mechanisms that exist
        but aren't load-bearing, each carrying its sub_kind + file:line + witness.
        """
        return [f for f in self.findings_list if f.is_dead]

    def absent_findings(self) -> List[MechanismFinding]:
        """Mechanisms the goal requires that do not exist at all."""
        return [f for f in self.findings_list if f.classification == "absent"]

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
            "findings": [
                {
                    "name": f.name,
                    "classification": f.classification,
                    "sub_kind": f.sub_kind,
                    "location": f.location,
                    "witness": {
                        "value": f.witness.value,
                        "description": f.witness.description,
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


def _coerce_float(value: Any) -> float:
    """Best-effort numeric coercion for a witness value; non-numeric -> 0.0 (dead)."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return 0.0
    return 0.0


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
        # Populated by execute() for the run ledger / dashboard.
        self.result: Optional[ReconciliationResult] = None
        self.reconciled: Optional[bool] = None

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
            try:
                data = json.loads(blob)
            except json.JSONDecodeError as exc:
                parse_error = f"invalid JSON: {exc}"
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
        # A reply we couldn't parse is NOT reconciled-by-default: an unreadable
        # verdict must not silently pass as "all clear".
        if parse_error is not None and not findings:
            reconciled = False

        return ReconciliationResult(
            reconciled=reconciled,
            findings_list=findings,
            disposition=self.disposition,
            raw=reply,
            parse_error=parse_error,
        )

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

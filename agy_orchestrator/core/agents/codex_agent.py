import glob
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.core.model_discovery import discover_models

logger = logging.getLogger(__name__)


def _codex_models_cache_path() -> Path:
    """Path to codex's own server-fetched model roster cache.

    codex writes the account's available models (with a ``priority`` rank and the
    per-model supported reasoning efforts) to ``~/.codex/models_cache.json`` — the
    same roster its model picker shows. Reading it is true dynamic detection: it
    reflects exactly what THIS account can use (e.g. retired/ungranted models like
    gpt-5.2 simply aren't present), with no hardcoded list and no TUI scraping.
    Overridable via ``AGY_CODEX_MODELS_CACHE`` (mainly so tests stay hermetic)."""
    override = os.environ.get("AGY_CODEX_MODELS_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".codex" / "models_cache.json"


def codex_models_from_cache() -> List[str]:
    """Codex's selectable models, ordered by codex's own ``priority`` (best first).

    Reads :func:`_codex_models_cache_path`. Returns the slugs of models the picker
    lists (``visibility == "list"``; hidden helpers like ``codex-auto-review`` are
    excluded), sorted by ascending ``priority``. Returns ``[]`` on ANY failure
    (missing/corrupt cache, unexpected shape) so callers fall back to the static
    list — never raises."""
    try:
        raw = _codex_models_cache_path().read_text(encoding="utf-8")
        data = json.loads(raw)
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return []
        picked = []
        for m in models:
            if not isinstance(m, dict):
                continue
            slug = m.get("slug") or m.get("id")
            if not slug:
                continue
            if m.get("visibility") not in (None, "list"):
                continue  # 'hide' (codex-auto-review) and unknowns are not pickable
            prio = m.get("priority")
            picked.append((prio if isinstance(prio, (int, float)) else math.inf, str(slug)))
        # Stable sort by priority; models without a priority sink to the end.
        picked.sort(key=lambda t: t[0])
        # De-dup preserving order.
        out: List[str] = []
        for _, slug in picked:
            if slug not in out:
                out.append(slug)
        return out
    except FileNotFoundError:
        return []
    except Exception as exc:  # corrupt cache / odd shape — never raise
        logger.debug("codex models cache unreadable: %s", exc)
        return []

# A codex usage/quota wall is reported ONLY in the rollout JSONL's token_count
# event (rate_limits.primary/secondary.used_percent), never on stderr — so the
# stderr-only classifier misses it and a walled codex is retried 3x/step and
# re-led every step for the whole window (issue #78). We detect it out-of-band and
# fold a synthetic marker into stderr. Default threshold is deliberately high (a
# real wall reports 100.0); overridable for tuning/tests. 0 disables the detector.
def _codex_usage_wall_percent() -> float:
    try:
        v = float(os.environ.get("AGY_CODEX_USAGE_WALL_PERCENT", "99") or "99")
    except ValueError:
        v = 99.0
    return v


def _codex_home() -> str:
    # codex honours CODEX_HOME; default ~/.codex. Sessions live under
    # <home>/sessions/YYYY/MM/DD/rollout-<ts>-<thread_id>.jsonl.
    return os.environ.get("CODEX_HOME") or os.path.join(os.path.expanduser("~"), ".codex")


def extract_codex_thread_id(raw_stdout: str) -> Optional[str]:
    """Pull the codex session/thread id from a --json stdout stream.

    codex --json emits a ``{"type":"thread.started","thread_id":"<uuid>"}`` event
    first; that uuid is the suffix of the rollout filename. Tolerates older
    ``session.created``/``session_configured`` shapes too. Returns None when the
    stream carries no start event (e.g. codex died before a session existed)."""
    for line in raw_stdout.splitlines():
        line = line.strip()
        if not line.startswith("{") or "thread" not in line and "session" not in line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if not isinstance(ev, dict):
            continue
        etype = str(ev.get("type") or ev.get("event") or "")
        if etype in ("thread.started", "session.created", "session_configured", "session.started"):
            sess = ev.get("session")
            nested = sess.get("id") if isinstance(sess, dict) else None
            tid = (
                ev.get("thread_id")
                or ev.get("session_id")
                or ev.get("id")
                or nested
            )
            if tid:
                return str(tid)
    return None


def find_codex_rollout(thread_id: str, codex_home: Optional[str] = None) -> Optional[str]:
    """Locate the rollout JSONL for ``thread_id`` (the uuid suffixes the filename).

    Unique per call, so safe under concurrent ToT branches. Returns the newest
    match (there is normally exactly one) or None."""
    if not thread_id:
        return None
    home = codex_home or _codex_home()
    pattern = os.path.join(home, "sessions", "*", "*", "*", f"rollout-*-{thread_id}.jsonl")
    matches = glob.glob(pattern)
    if not matches:
        # Fallback: some codex builds use a flatter sessions/ layout.
        matches = glob.glob(os.path.join(home, "sessions", "**", f"rollout-*-{thread_id}.jsonl"),
                            recursive=True)
    if not matches:
        return None
    try:
        return max(matches, key=os.path.getmtime)
    except OSError:
        return matches[0]


def read_codex_rate_limits(rollout_path: str) -> Optional[dict]:
    """Return the LAST token_count event's ``rate_limits`` dict from a rollout, or None.

    Reads the whole (small) JSONL and keeps the final rate_limits seen; tolerant of
    truncated/garbage lines (a mid-write kill can't crash detection)."""
    last: Optional[dict] = None
    try:
        with open(rollout_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{") or "rate_limits" not in line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if not isinstance(ev, dict):
                    continue
                payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else ev
                rl = payload.get("rate_limits") if isinstance(payload, dict) else None
                if isinstance(rl, dict):
                    last = rl
    except OSError:
        return None
    return last


def extract_codex_model_error(raw_stdout: str) -> Optional[str]:
    """Return codex's error message when the --json stream reports a failed turn,
    else None. codex emits model-access / request errors ONLY here (empty stderr):
        {"type":"error","message":"...400...is not supported..."}
        {"type":"turn.failed","error":{"message":"..."}}
    Returns the innermost message string so the classifier can inspect it."""
    found: Optional[str] = None
    for line in raw_stdout.splitlines():
        line = line.strip()
        if not line.startswith("{") or ("error" not in line and "failed" not in line):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if not isinstance(ev, dict):
            continue
        etype = str(ev.get("type") or ev.get("event") or "")
        if etype in ("error", "turn.failed", "response.failed", "item.failed"):
            msg = ev.get("message")
            err = ev.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or msg
            if msg:
                found = str(msg)  # keep the LAST (most specific) failure message
    return found


def codex_usage_wall_marker(rate_limits: Optional[dict], threshold: float) -> Optional[str]:
    """Synthetic stderr marker when codex's rate_limits show a wall, else None.

    Walls when EITHER the primary (rolling, ~5h) OR secondary (weekly) window is at
    ``used_percent >= threshold``. The returned string contains the literal
    ``usage limit`` so ``is_usage_wall`` matches it; it also carries the window's
    ``limit_name``/``resets_at`` so a reader (and the fallback layer) can see which
    model walled and when it clears. threshold<=0 disables (returns None)."""
    if not isinstance(rate_limits, dict) or threshold <= 0:
        return None
    worst_pct = -1.0
    worst: dict = {}
    for key in ("primary", "secondary"):
        win = rate_limits.get(key)
        if not isinstance(win, dict):
            continue
        pct = win.get("used_percent")
        if isinstance(pct, (int, float)) and math.isfinite(pct) and pct > worst_pct:
            worst_pct = float(pct)
            worst = win
    if worst_pct < threshold:
        return None
    limit_name = rate_limits.get("limit_name") or rate_limits.get("limit_id") or "codex"
    resets_at = worst.get("resets_at")
    return (
        f"usage limit reached (codex rate_limits.used_percent={worst_pct} "
        f"limit_name={limit_name} resets_at={resets_at})"
    )


class CodexAgent(AgentInstance):
    # Single source of truth for codex model names (used by the async API below
    # and synchronously by harness/effort_overrides.py for --codex-model
    # validation without spinning an event loop).
    # Last-resort static roster, used ONLY when codex's own models_cache.json is
    # unreadable (see codex_models_from_cache). Kept account-accurate (priority
    # order; no gpt-5.2 / bare gpt-5.3-codex — neither is granted on the current
    # account). The cache is the real source of truth; this is the safety net.
    AVAILABLE_MODELS: List[str] = [
        "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex-spark",
    ]

    @classmethod
    def candidate_models(cls) -> List[str]:
        """Codex's selectable models, best-first. Dynamic (codex's own cache) with
        the static :attr:`AVAILABLE_MODELS` as the never-empty fallback."""
        return codex_models_from_cache() or list(cls.AVAILABLE_MODELS)

    @classmethod
    async def get_available_models(cls) -> List[str]:
        # codex has no model-list subcommand, but it caches its server-fetched
        # roster to ~/.codex/models_cache.json (priority-ordered, account-accurate).
        # Read THAT for dynamic detection; the static AVAILABLE_MODELS is the
        # fallback when the cache is absent. Routed through discover_models
        # (list_argv=None) to keep the memoized discovery path uniform across
        # adapters; ``fallback`` is the dynamic-then-static resolved roster.
        return await discover_models(
            "codex",
            list_argv=None,
            parse=lambda _stdout: [],
            fallback=cls.candidate_models(),
        )

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        # Codex exposes no machine-readable remaining-usage value, so report
        # "full" and let real quota exhaustion be handled by the fallback layer
        # (a usage wall surfaces as a non-zero exit -> fail over). Spawning
        # `codex --usage` only to discard its output was pure overhead.
        return 100.0

    def filter_stderr(self, stderr: str) -> str:
        lines = stderr.splitlines()
        filtered = [ln for ln in lines if "network" not in ln.lower() and "timeout" not in ln.lower()]
        return "\n".join(filtered)

    def _full_prompt(self, piped_input: Optional[str] = None) -> str:
        full_prompt = self.prompt
        if piped_input:
            full_prompt += f"\n\n[Context]:\n{piped_input}"
        return full_prompt

    def _stdin_bytes(self, piped_input: Optional[str] = None) -> bytes:
        # Deliver the prompt on stdin (codex exec reads it when '-' is the prompt
        # arg) so large diff-feedback prompts can't hit MAX_ARG_STRLEN (128 KiB).
        return self._full_prompt(piped_input).encode()

    def _postprocess(self, raw_stdout: str) -> str:
        # Under --json the agent's answer is no longer plain stdout (that's now a
        # JSONL event stream). Prefer codex's --output-last-message file (the clean
        # final message); fall back to concatenating message-event text from the
        # JSONL; last resort, return raw_stdout unchanged (e.g. a pre-turn error).
        path = getattr(self, "_codex_last_msg_path", None)
        if path:
            try:
                text = ""
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                if text.strip():
                    return text
            except Exception:
                pass
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        msg = self._message_text_from_jsonl(raw_stdout)
        return msg if msg is not None else raw_stdout

    def _augment_failure_stderr(self, raw_stdout: str, stderr: str) -> str:
        # A real codex quota wall reports nothing on stderr — exhaustion lives only
        # in the rollout JSONL (rate_limits.used_percent). Read it out-of-band and,
        # when walled, append a synthetic 'usage limit' marker so is_usage_wall()
        # fires and the fallback layer demotes/cycles codex instead of retrying a
        # dead window 3x/step + re-leading it every step (issue #78). Already
        # carries the marker? leave it. Best-effort: any miss returns stderr as-is
        # (graceful degrade to the prior, undetected-wall behavior).
        low = (stderr or "").lower()
        # (1) Model inaccessible to this account — codex reports it ONLY in the
        # --json stdout error event (empty stderr). Fold it in so is_model_unavailable
        # fires and the fallback layer PRUNES this model and tries another (dynamic
        # model detection / operator request). Cheap (no disk) so check it first.
        if "is not supported" not in low and "does not have access" not in low:
            err_msg = extract_codex_model_error(raw_stdout)
            if err_msg:
                from agy_orchestrator.core.agent import (
                    is_model_unavailable as _is_unavail,
                    is_usage_wall as _is_usage,
                )
                # A 'usage'/'rate limit' phrased error still routes to the usage path.
                if _is_unavail(err_msg) and not _is_usage(err_msg):
                    logger.warning("[codex] model unavailable (from stdout): %s", err_msg)
                    return f"{stderr}\n{err_msg}".strip() if stderr else err_msg
        # (2) Out-of-band quota wall — codex reports exhaustion only in the rollout
        # JSONL (rate_limits.used_percent), never on stderr (issue #78).
        if "usage limit" in low:
            return stderr
        threshold = _codex_usage_wall_percent()
        if threshold <= 0:
            return stderr
        thread_id = extract_codex_thread_id(raw_stdout)
        if not thread_id:
            return stderr
        rollout = find_codex_rollout(thread_id)
        if not rollout:
            return stderr
        rate_limits = read_codex_rate_limits(rollout)
        marker = codex_usage_wall_marker(rate_limits, threshold)
        if not marker:
            return stderr
        # Stash structured fields so a caller can skip the walled model until reset.
        try:
            primary = (rate_limits or {}).get("primary") or {}
            self.usage_wall_resets_at = primary.get("resets_at")
            self.usage_wall_limit_name = (rate_limits or {}).get("limit_name")
        except Exception:
            pass
        logger.warning("[codex] out-of-band quota wall from rollout %s: %s", rollout, marker)
        return f"{stderr}\n{marker}".strip() if stderr else marker

    @staticmethod
    def _message_text_from_jsonl(raw_stdout: str) -> Optional[str]:
        # Reconstruct the assistant message from codex --json events. Mirrors the
        # message shapes the dashboard codex adapter handles: a top-level
        # {"type":"message"|"item.message"|"response.output_text.delta", "text"|"delta"|"message"}
        # or a nested {"item": {"type":"message", "text": ...}}.
        parts: List[str] = []
        saw_message = False
        for line in raw_stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if not isinstance(ev, dict):
                continue
            etype = str(ev.get("type") or ev.get("event") or "")
            if etype in ("message", "item.message", "response.output_text.delta"):
                txt = ev.get("text") or ev.get("delta") or ev.get("message")
                if txt:
                    parts.append(str(txt))
                    saw_message = True
                continue
            item = ev.get("item")
            if isinstance(item, dict) and item.get("type") == "message" and item.get("text"):
                parts.append(str(item["text"]))
                saw_message = True
        if not saw_message:
            return None
        return "".join(parts)

    def _events_from_stdout_line(self, line: str) -> List[dict]:
        try:
            from dashboard.adapters import AdapterCtx, parse_codex_stream_line
        except Exception:
            return []
        ctx = getattr(self, "_adapter_ctx", None)
        if ctx is None:
            ctx = AdapterCtx()
            self._adapter_ctx = ctx
        return parse_codex_stream_line(line, ctx)

    def _events_from_stderr_line(self, line: str) -> List[dict]:
        try:
            from dashboard.adapters import AdapterCtx, parse_codex_stream_line
        except Exception:
            return super()._events_from_stderr_line(line)
        ctx = getattr(self, "_adapter_ctx", None)
        if ctx is None:
            ctx = AdapterCtx()
            self._adapter_ctx = ctx
        parsed = parse_codex_stream_line(line, ctx)
        if parsed:
            return parsed
        return super()._events_from_stderr_line(line)

    def _extract_usage(self, raw_stdout: str, raw_stderr: str) -> dict:
        # codex usage is available only when JSON turn events are present.
        usage = {}
        for line in raw_stdout.splitlines():
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            event_type = payload.get("type") or payload.get("event")
            if event_type == "turn.completed":
                row = payload.get("usage") or payload.get("data", {}).get("usage") or {}
                if isinstance(row, dict):
                    usage = row
        if not usage:
            import re
            # codex prints the "tokens used\n<N>" summary on STDERR, not stdout;
            # scan both and take the last match (the final per-call total).
            matches = re.findall(
                r"tokens used(?:[:\s]+)([\d,]+)",
                f"{raw_stdout}\n{raw_stderr}",
                re.IGNORECASE,
            )
            if matches:
                try:
                    total = int(matches[-1].replace(",", ""))
                    return {
                        "token_source": "cli",
                        "input_tokens": None,
                        "output_tokens": None,
                        "cache_read_tokens": None,
                        "total_tokens": total,
                    }
                except Exception:
                    pass
            return {
                "token_source": "unavailable",
                "input_tokens": None,
                "output_tokens": None,
                "cache_read_tokens": None,
                "total_tokens": None,
            }
        raw_input = usage.get("input_tokens") or usage.get("in_tokens")
        cache_read = (
            usage.get("cache_read_tokens")
            or usage.get("cached_input_tokens")
            or usage.get("cached_tokens")
        )
        # codex/OpenAI report ``input_tokens`` INCLUSIVE of the cached prompt
        # tokens (the probe: total input stays ~constant while cached grows on a
        # warm prefix). The rest of the stack — and dispatch._cache_read_ratio —
        # treats input_tokens as the FRESH, reprocessed input (the Anthropic /
        # claude convention, cache-exclusive). Normalize codex to match so the
        # hit-rate is correct and cross-provider sums line up: fresh = total - cached.
        fresh_input = raw_input
        # JSON can legally carry non-finite floats (json.loads accepts 'NaN', and
        # '1e400' -> inf), and int(nan)/int(inf) raise ValueError/OverflowError.
        # Guard with isfinite() so a poisoned token count can't crash usage
        # extraction and silently discard the whole (otherwise valid) usage row.
        if (
            isinstance(raw_input, (int, float))
            and isinstance(cache_read, (int, float))
            and math.isfinite(raw_input)
            and math.isfinite(cache_read)
        ):
            fresh_input = max(int(raw_input) - int(cache_read), 0)
        return {
            "token_source": "cli",
            "input_tokens": fresh_input,
            "output_tokens": usage.get("output_tokens") or usage.get("out_tokens"),
            "cache_read_tokens": cache_read,
            "total_tokens": usage.get("total_tokens"),
        }

    def _last_message_path(self) -> str:
        # Stable per-instance path so build_command and _postprocess agree. Unique
        # per (pid, instance) so concurrent ToT branches (separate instances) and
        # parallel processes don't collide. codex overwrites it each run; we read
        # and unlink it in _postprocess.
        path = getattr(self, "_codex_last_msg_path", None)
        if not path:
            path = os.path.join(
                tempfile.gettempdir(), f"codex_lastmsg_{os.getpid()}_{id(self)}.txt"
            )
            self._codex_last_msg_path = path
        return path

    # Resolved identity for observers — MUST mirror build_command's aliasing so
    # telegram/dashboard show the model codex actually runs, not the "standard"
    # alias (which is meaningless to the operator) or the "max" effort tier
    # (which codex sends as reasoning_effort=xhigh).
    @classmethod
    def resolve_effective_model(cls, model: Optional[str]) -> Optional[str]:
        if model and model != "standard":
            return model
        # "standard" and the unset default both land on the ChatGPT-account
        # model the command-builder pins (bare gpt-5.3-codex 400s; see
        # build_command). Surface that real name.
        return "gpt-5.3-codex-spark"

    @classmethod
    def resolve_effective_effort(cls, effort: Optional[str]) -> Optional[str]:
        return "xhigh" if effort == "max" else effort

    def build_command(self, piped_input: Optional[str] = None) -> List[str]:
        cmd = [
            "codex",
            "exec",
            # --json makes codex emit JSONL events to stdout, which is the ONLY way
            # it reports structured per-call usage (turn.completed -> input/output/
            # cached_input_tokens). Without it we get only a "tokens used: N" total on
            # stderr (cache_read invisible). It also feeds the dashboard codex adapter
            # real stream events. The agent's final text no longer lands on stdout as
            # plain text, so -o/--output-last-message captures it cleanly instead (see
            # _postprocess).
            "--json",
            "--output-last-message",
            self._last_message_path(),
            "--skip-git-repo-check",
        ]
        # #91: a plan-only / planner invocation MUST be provably read-only. Codex's
        # built-in apply_patch/write_file tools would otherwise let the model write a
        # partial implementation into the out-dir during what the operator was told is
        # a no-write dry-run (the planner prompt embeds the raw "Implement X" request,
        # and codex dutifully starts building). The read-only sandbox makes those tool
        # writes PHYSICALLY impossible — a hard guarantee, not a prompt the model can
        # ignore. The normal (execute) path keeps the externally-sandboxed bypass so
        # accepted build steps can actually edit. --output-last-message still writes
        # codex's own result file (harness-owned temp path), which the sandbox does
        # not gate.
        if getattr(self, "read_only", False):
            cmd.extend(["--sandbox", "read-only"])
        else:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")

        if self.model and self.model != "standard":
            cmd.extend(["--model", self.model])
        elif self.model == "standard":
            # ChatGPT-account codex (codex-cli) REJECTS bare "gpt-5.3-codex" with a
            # 400 "model is not supported when using Codex with a ChatGPT account";
            # the account-valid Codex default is the "-spark" variant. Pinning the
            # bare name 400'd every call -> blank "failed after 3 attempts" -> silent
            # fallback off codex (looked like a usage wall, wasn't one).
            cmd.extend(["--model", "gpt-5.3-codex-spark"])

        if hasattr(self, "effort") and self.effort:
            effort = "xhigh" if self.effort == "max" else self.effort
            cmd.extend(["-c", f"model_reasoning_effort=\"{effort}\""])

        # Arbitrary `-c key=value` config overrides (e.g. "tools.web_search=true").
        for entry in getattr(self, "config_overrides", None) or []:
            cmd.extend(["-c", entry])

        for k, v in self.additional_flags.items():
            cmd.extend([f"--{k}", str(v)])

        # '-' tells codex exec to read the prompt from stdin (see _stdin_bytes).
        cmd.append("-")
        return cmd

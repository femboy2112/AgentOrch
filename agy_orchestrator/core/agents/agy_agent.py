"""Antigravity (`agy`) worker.

agy has no ``--model``/``--effort`` flags: the model is chosen in its interactive
``/model`` picker and persisted to ``~/.gemini/antigravity-cli/settings.json`` as a
display name (e.g. ``"Gemini 3.1 Pro (High)"``, effort baked into the suffix). To
pin a model for a headless ``agy --print`` run we therefore write that settings
field before invoking agy and restore it afterward.

Because settings.json is GLOBAL, concurrent agy calls would clobber each other's
model, so model-pinned runs are serialized with a cross-process file lock — i.e.
parallel agy branches run one-at-a-time while pinned (the price of agy's global
model selection). Runs that don't request a known model skip all of this and use
agy's current default.
"""
import asyncio
import fcntl
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from agy_orchestrator.core.agent import AgentInstance
from agy_orchestrator.core.model_discovery import discover_models

logger = logging.getLogger(__name__)

# Static fallback roster (agy selects its model via settings.json, not a flag or
# a models subcommand, so this display-name list IS the source of truth).
_AGY_FALLBACK_MODELS = [
    "Gemini 3.1 Pro (High)", "Gemini 3.1 Pro (Low)",
    "Gemini 3.5 Flash (High)", "Gemini 3.5 Flash (Medium)", "Gemini 3.5 Flash (Low)",
    "Claude Opus 4.6 (Thinking)", "Claude Sonnet 4.6 (Thinking)",
    "GPT-OSS 120B (Medium)",
]

_SETTINGS_PATH = Path.home() / ".gemini" / "antigravity-cli" / "settings.json"
_LOCK_PATH = Path(tempfile.gettempdir()) / "agentorch-agy-model.lock"

# Verified roster (from the live /model picker). Display name is what goes into
# settings.json["model"]. Effort is encoded in the suffix.
GEMINI_PRO_EFFORTS = {"low": "Low", "high": "High"}            # Pro: Low/High only
GEMINI_FLASH_EFFORTS = {"low": "Low", "medium": "Medium", "high": "High"}


def resolve_agy_model(model: Optional[str], effort: Optional[str]) -> Optional[str]:
    """Map an orchestrator (model, effort) pair to an exact agy picker display
    name, or return None to leave agy on its current default."""
    if not model:
        return None
    m = model.strip()
    # Already a full display name? Pass through.
    if "(" in m and any(m.startswith(p) for p in ("Gemini", "Claude", "GPT-OSS")):
        return m
    key = m.lower()
    eff = (effort or "").lower()
    if key in ("pro", "gemini-pro", "gemini 3.1 pro", "gemini-3.1-pro"):
        return f"Gemini 3.1 Pro ({GEMINI_PRO_EFFORTS.get(eff, 'High')})"
    if key in ("flash", "gemini-flash", "gemini 3.5 flash", "gemini-3.5-flash"):
        return f"Gemini 3.5 Flash ({GEMINI_FLASH_EFFORTS.get(eff, 'Medium')})"
    if key in ("opus", "claude-opus", "claude opus 4.6"):
        return "Claude Opus 4.6 (Thinking)"
    if key in ("sonnet", "claude-sonnet", "claude sonnet 4.6"):
        return "Claude Sonnet 4.6 (Thinking)"
    if key in ("gpt-oss", "oss", "gpt-oss 120b"):
        return "GPT-OSS 120B (Medium)"
    return None  # "standard"/unknown -> don't pin


class AgyAgent(AgentInstance):
    def __init__(
        self,
        prompt: str,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        input_files: Optional[List[str]] = None,
        output_files: Optional[List[str]] = None,
        additional_flags: Optional[Dict[str, str]] = None
    ):
        super().__init__(prompt, model, additional_flags)
        self.effort = effort
        self.input_files = input_files or []
        self.output_files = output_files or []
        # Neuter the browser launcher for headless agy runs: on token expiry agy
        # would otherwise open Firefox for Google OAuth, which the operator can't
        # complete (the subprocess stdin isn't a terminal to paste the code into)
        # and which hangs the call. Pointing BROWSER at a no-op makes expiry a
        # clean fast failure that the fallback chain (agy->codex->grok) rolls over.
        # Re-auth is done out-of-band via an interactive `agy -i` session.
        self.extra_env = {"BROWSER": "/bin/true"}

    @classmethod
    async def get_available_models(cls) -> List[str]:
        # agy has no model-list subcommand (model is chosen in its interactive
        # /model picker), so we don't shell out (``list_argv=None``); the picker
        # roster is the static fallback. Routed through discover_models to keep the
        # discovery/union path uniform + memoized across adapters.
        return await discover_models(
            "agy",
            list_argv=None,
            parse=lambda _stdout: [],
            fallback=_AGY_FALLBACK_MODELS,
        )

    @classmethod
    async def get_model_usage(cls, model: str) -> float:
        return 100.0

    def build_command(self, piped_input: Optional[str] = None) -> List[str]:
        # Generic "be excellent / be performant" exhortations were dropped: a
        # prompt-ablation (2026-05-30, codex, partial-credit bench) showed they
        # buy no measurable quality. The interface-CORRECTNESS line stays (it's
        # plausibly load-bearing for multi-file work the bench doesn't exercise),
        # as does the NO SUDO safety constraint.
        injected_prompt = (
            "System constraints:\n"
            "- CORRECTNESS: Double-check that identifiers, signatures, and interfaces match exactly across files and components.\n"
            "- NO SUDO: Do NOT use `sudo` under any circumstances.\n"
        )
        if self.input_files:
            injected_prompt += f"- Read these files: {', '.join(self.input_files)}\n"
        if self.output_files:
            injected_prompt += f"- Ensure these files are created: {', '.join(self.output_files)}\n"

        injected_prompt += f"\n{self.prompt}"
        if piped_input:
            injected_prompt += f"\n\n[Piped Context from previous step]:\n{piped_input}"

        cmd = ["agy", "--print", injected_prompt, "--dangerously-skip-permissions"]
        cmd += ["--print-timeout", os.environ.get("AGY_PRINT_TIMEOUT", "300s")]
        for k, v in self.additional_flags.items():
            cmd.extend([f"--{k}", str(v)])
        return cmd

    def _events_from_stderr_line(self, line: str) -> List[dict]:
        try:
            from dashboard.adapters import parse_agy_stderr
        except Exception:
            return super()._events_from_stderr_line(line)
        return parse_agy_stderr(line)

    def _events_from_stdout_complete(self, raw_stdout: str) -> List[dict]:
        try:
            from dashboard.adapters import parse_agy_stdout
        except Exception:
            return super()._events_from_stdout_complete(raw_stdout)
        return parse_agy_stdout(raw_stdout)

    def _postprocess(self, raw_stdout: str) -> str:
        return raw_stdout

    # --- model pinning via settings.json (see module docstring) ----------------
    @staticmethod
    def _read_settings_model() -> Optional[str]:
        try:
            return json.loads(_SETTINGS_PATH.read_text()).get("model")
        except Exception:
            return None

    @staticmethod
    def _write_settings_model(model_name: Optional[str]) -> None:
        try:
            data = json.loads(_SETTINGS_PATH.read_text()) if _SETTINGS_PATH.exists() else {}
        except Exception:
            data = {}
        if model_name is None:
            data.pop("model", None)
        else:
            data["model"] = model_name
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(_SETTINGS_PATH.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, _SETTINGS_PATH)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    async def run_async(self, piped_input: Optional[str] = None) -> str:
        display = resolve_agy_model(self.model, self.effort)
        if not display:
            # No recognized model -> use agy's current default, no settings touch.
            return await super().run_async(piped_input)

        loop = asyncio.get_running_loop()
        lock_fh = open(_LOCK_PATH, "w")

        def _acquire():
            fcntl.flock(lock_fh, fcntl.LOCK_EX)

        await loop.run_in_executor(None, _acquire)
        prev = self._read_settings_model()
        try:
            logger.info("agy: pinning model -> %s (was %s)", display, prev)
            self._write_settings_model(display)
            return await super().run_async(piped_input)
        finally:
            # Restore the user's prior model so interactive agy is unaffected.
            try:
                self._write_settings_model(prev)
            except Exception:
                pass
            try:
                fcntl.flock(lock_fh, fcntl.LOCK_UN)
            finally:
                lock_fh.close()

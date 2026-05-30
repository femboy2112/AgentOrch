"""Utilities for the computer-use worker (Step 1).

OBSERVE-mode redaction (hardening requirement #4):
- Default-ON secret redaction for all real-:0-scope text before it ever reaches
  a reasoning CLI prompt (window titles, OCR/AT-SPI/DOM text, terminal content).
- Per-run opt-out via enabled=False (explicit, auditable).
- Must be wired into PerceptionPipeline + ReasoningInput construction for OBSERVE.

Never leaks: password, secret, token, key, OAuth markers, or KEY=VALUE env-like pairs.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Pattern

# Pre-compiled patterns (case-insensitive where appropriate)
# Order matters: more specific first.
_REDACTION_PATTERNS: list[tuple[Pattern[str], str]] = [
    # High-value explicit secret markers (password=..., secret: ..., token=...)
    (re.compile(r'(?i)\b(password|passwd|secret|api[_-]?key|auth[_-]?token|access[_-]?token|refresh[_-]?token|private[_-]?key)\s*[:=]\s*\S+', re.IGNORECASE), r'\1=***REDACTED***'),
    # OAuth / bearer / Authorization headers (eat the full value incl. following JWTs etc)
    (re.compile(r'(?i)\b(authorization|oauth|bearer)\s*[:=]?\s*\S+(?:\s+\S+)?', re.IGNORECASE), r'\1 ***REDACTED***'),
    # Generic high-entropy token-like after key words
    (re.compile(r'(?i)\b(token|key|secret|credential)\s*[:=]\s*[A-Za-z0-9_\-\.\/\+=]{8,}', re.IGNORECASE), r'\1=***REDACTED***'),
    # Any ALL_CAPS_ENV=VALUE style pair (catches exported secrets, AWS_*, etc.)
    # We keep the KEY= but scrub the value. Non-capturing for the = part.
    (re.compile(r'\b([A-Z][A-Z0-9_]{2,})\s*=\s*([^\s;"\']+)', re.IGNORECASE), r'\1=***REDACTED***'),
    # Quoted variants "KEY=val" or 'KEY=val'
    (re.compile(r'(["\'])([A-Z][A-Z0-9_]{2,})\s*=\s*[^"\']+\1', re.IGNORECASE), r'\1\2=***REDACTED***\1'),
    # JWT-like or long base64-ish secrets standing alone after common prefixes
    (re.compile(r'(?i)(sk-[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{10,}|xoxb-[A-Za-z0-9-]{10,})'), '***REDACTED***'),
]


def redact_secrets(text: str, *, enabled: bool = True) -> str:
    """Scrub secrets from text (default ON for OBSERVE hardening #4).

    Every real-:0 scope text (titles, OCR, AT-SPI, DOM, terminal) MUST pass
    through this before inclusion in any ReasoningInput / prompt sent to
    claude/codex reasoner CLIs. Default is redact-on; explicit opt-out only
    for special runs (audited via the enabled flag).

    Hardening guarantee: when enabled=True, no planted password/secret/token/key/
    OAuth value or KEY=VALUE pair ever appears in the output string.

    :param enabled: False for per-run opt-out (still logged in events).
    """
    if not enabled:
        return text if text is not None else ""
    if text is None or not isinstance(text, str):
        return ""
    if not text:
        return text

    result = text
    for pattern, repl in _REDACTION_PATTERNS:
        result = pattern.sub(repl, result)
    return result


def is_redaction_enabled_for_run(overrides: Optional[Dict[str, Any]] = None) -> bool:
    """Helper for later SessionController / run config (default True).

    Future: may read env or per-run flag; for Step 1 the default is always on.
    """
    if overrides and "redact_secrets" in overrides:
        return bool(overrides["redact_secrets"])
    return True

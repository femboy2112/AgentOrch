"""Shared passthrough chat/session primitives.

This module is intentionally lightweight and import-safe (no network, no
subprocesses, no broker calls). It centralizes prompt construction, atomic
session persistence, and minimal run capture for the `harness ask`/`harness chat`
paths.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from harness import dispatch, roles


SESSION_CAPABLE = {"claude", "grok"}
RUN_ID_TS_FORMAT = "%Y%m%d-%H%M%S-%f"

# Run-local passthrough session store path (under runs/, not user data outside the repo).
SESSIONS_DIR = dispatch.RUNS_DIR / ".sessions"

# Keep filenames bounded and deterministic. Spec only requires “bounded”.
SESSION_ID_MAX_LEN = 64

_SESSION_ID_SANITIZE = re.compile(r"[^a-z0-9_-]")


@dataclass
class PassthroughSession:
    """Persisted passthrough conversation state."""

    session_id: str
    provider: str
    provider_chain: List[str]
    # Optional fields default to None so a partially-populated or older-schema
    # session file (e.g. one written before a field existed, or hand-edited)
    # still loads instead of failing construction and silently reading as None
    # (fuzz finding: missing-field sessions returned None from load_session).
    model: Optional[str] = None
    effort: Optional[str] = None
    provider_session_id: Optional[str] = None
    turns: List[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PassthroughSession":
        # Guard non-dict payloads with a clean, catchable error rather than an
        # AttributeError on ``.items()`` (load_session/list_sessions already
        # catch ValueError); only the always-required identity fields
        # (session_id/provider/provider_chain) lacking a value still raises.
        if not isinstance(data, dict):
            raise ValueError("session data must be a dict")
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class PassthroughResult:
    """Return contract for one passthrough turn."""

    reply: str
    provider_used: str
    session_id: Optional[str] = None
    error: Optional[str] = None


class SessionStore:
    """Atomic persistence helpers for passthrough session files."""

    @staticmethod
    def session_file(session_id: str) -> Path:
        slug = _sanitize_session_id(session_id)
        return SESSIONS_DIR / f"{slug}.json"

    @staticmethod
    def save_session(session: PassthroughSession) -> None:
        p = SessionStore.session_file(session.session_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(session.to_dict(), indent=2)
        # Per-write UNIQUE temp name. A fixed ``<id>.json.tmp`` races when the
        # SAME session is saved concurrently (e.g. two rapid Telegram /chat turns
        # each spawn a detached `harness ask --session tg-<chat>`): both write the
        # one tmp, the first replace() consumes it, the second replace() fails
        # FileNotFoundError and the cross-writer `tmp.exists()` unlink can delete a
        # peer's tmp (fuzz finding). A unique suffix gives each writer its own tmp;
        # concurrent replaces are individually atomic (last writer wins, no crash).
        tmp = p.with_name(f"{p.name}.tmp.{uuid.uuid4().hex}")
        try:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(p)
        finally:
            # Only ever our OWN tmp; after a successful replace it is already gone.
            try:
                tmp.unlink()
            except OSError:
                pass

    @staticmethod
    def load_session(session_id: str) -> Optional[PassthroughSession]:
        try:
            p = SessionStore.session_file(session_id)
            data = json.loads(p.read_text(encoding="utf-8"))
            return PassthroughSession.from_dict(data)
        except (AttributeError, FileNotFoundError, ValueError, TypeError, OSError):
            return None

    @staticmethod
    def list_sessions() -> List[PassthroughSession]:
        if not SESSIONS_DIR.exists():
            return []

        sessions: List[PassthroughSession] = []
        for p in sorted(SESSIONS_DIR.glob("*.json")):
            if not p.is_file():
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                sessions.append(PassthroughSession.from_dict(data))
            except (AttributeError, FileNotFoundError, ValueError, TypeError, OSError):
                continue
        return sessions

    @staticmethod
    def delete_session(session_id: str) -> None:
        try:
            SessionStore.session_file(session_id).unlink()
        except (FileNotFoundError, ValueError, OSError):
            return


def _sanitize_session_id(session_id: str, *, max_len: int = SESSION_ID_MAX_LEN) -> str:
    """Return a safe bounded session slug.

    Security/robustness policy:
      - must be a string
      - must not be empty after normalization
      - reject traversal and explicit path separators
      - restrict chars to [a-z0-9_-]
      - trim to bounded length
    """
    if not isinstance(session_id, str):
        raise ValueError("session_id must be a string")

    raw = session_id.strip().lower()
    if not raw or raw in {".", ".."}:
        raise ValueError("invalid session id")
    if "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError("invalid session id")
    if os.path.sep in raw or (os.path.altsep and os.path.altsep in raw):
        raise ValueError("invalid session id")

    slug = _SESSION_ID_SANITIZE.sub("-", raw)
    if max_len > 0:
        slug = slug[:max_len]
    # Require at least one alphanumeric char in the slug. Without this, any
    # punctuation-only name collapses to all-dashes (``!@#`` and ``---`` both ->
    # ``---``), silently COLLIDING on one file (fuzz finding). Rejecting the
    # degenerate class makes those names error cleanly instead of aliasing.
    if not slug or not re.search(r"[a-z0-9]", slug):
        raise ValueError("invalid session id")
    return slug


def _rendered_line(role: str, text: str) -> str:
    if role == "user":
        return f"User: {text}"
    if role == "assistant":
        return f"Assistant: {text}"
    return ""


def render_transcript(
    turns: Sequence[Dict[str, Any]],
    *,
    max_turns: int = 12,
    max_chars: int = 8000,
) -> str:
    """Render a bounded transcript for soft resume.

    Returns:
        A one-line instruction followed by up to ``max_turns`` turns, dropping
        oldest entries first until the full rendered text is under ``max_chars``.
    """
    prefix = "Continue this conversation:"
    if max_chars <= 0:
        return ""
    if len(prefix) >= max_chars:
        return prefix[:max_chars]

    if max_turns <= 0:
        return prefix

    rendered: List[str] = []
    for turn in list(turns)[-max_turns:]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "")).lower()
        if role not in {"user", "assistant"}:
            continue
        rendered.append(_rendered_line(role, str(turn.get("text", ""))))
        if not rendered[-1]:
            rendered.pop()

    if not rendered:
        return prefix

    while rendered:
        rendered_text = "\n".join(rendered)
        if len(f"{prefix}\n{rendered_text}") <= max_chars:
            break
        rendered.pop(0)

    if not rendered:
        return prefix

    body = "\n".join(rendered)
    return f"{prefix}\n{body}"


def _normalise_provider_name(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        if value.endswith("Agent"):
            value = re.sub(r"Agent$", "", value)
            return value.lower()
        return value.lower()
    value_name = getattr(value, "__name__", "")
    if value_name:
        if value_name.endswith("Agent"):
            return value_name[:-5].lower()
    return ""


def _run_capture(
    run_dir: Path,
    *,
    provider: str,
    model: Optional[str],
    effort: Optional[str],
    session_id: Optional[str],
    success: bool,
    duration_s: float,
    reply_chars: int,
    prompt: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    dispatch._atomic_write_text(run_dir / "prompt.txt", prompt, encoding="utf-8")
    dispatch._atomic_write_text(
        run_dir / "meta.json",
        json.dumps(
            {
                "mode": "ask",
                "provider": provider,
                "model": model,
                "effort": effort,
                "session_id": session_id,
                "success": bool(success),
                "duration_s": duration_s,
                "reply_chars": reply_chars,
            }
        ),
        encoding="utf-8",
    )


def _next_capture_run_dir() -> Path:
    return dispatch.RUNS_DIR / datetime.now().strftime(RUN_ID_TS_FORMAT)[:-3]


async def run_turn(
    prompt: str,
    *,
    provider_chain: List[str],
    model: Optional[str] = None,
    effort: Optional[str] = None,
    session: Optional[PassthroughSession] = None,
    fallback: bool = True,
    web_search: bool = False,
    out_dir: Optional[str] = None,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    capture: bool = True,
) -> PassthroughResult:
    """Run one lightweight passthrough turn."""
    if not provider_chain:
        return PassthroughResult(
            reply="",
            provider_used="",
            session_id=None,
            error="provider_chain must be non-empty",
        )

    start_s = time.time()
    lead = provider_chain[0]
    if session is not None and session.provider != lead:
        session.provider = lead
        session.provider_session_id = None
    resume_native = (
        session is not None
        and lead in SESSION_CAPABLE
        and session.provider == lead
        and bool(session.provider_session_id)
    )

    sent_prompt = prompt
    if session is not None and session.turns and not resume_native:
        sent_prompt = f"{render_transcript(session.turns)}\n{prompt}"

    native_session_id = session.provider_session_id if resume_native else None
    codex_config = ["tools.web_search=true"] if web_search else None
    provider_used = lead
    reply = ""
    resolved_session_id = None
    run_dir = _next_capture_run_dir() if capture else None

    try:
        agent = roles.build_role_agent(
            provider_chain,
            prompt=sent_prompt,
            model=model,
            effort=effort,
            fallback=fallback,
            codex_config=codex_config,
            session_id=native_session_id,
            fork_session=False,
        )
        if out_dir is not None:
            agent.cwd = out_dir
        if event_callback is not None:
            agent.event_callback = event_callback

        reply = str(await agent.run_async())
        provider_hint = _normalise_provider_name(getattr(agent, "last_provider", None))
        provider_used = provider_hint or lead
        duration_s = time.time() - start_s

        if capture:
            try:
                reply_text = str(reply)
                _run_capture(
                    run_dir,
                    provider=provider_used,
                    model=model,
                    effort=effort,
                    session_id=session.session_id if session else None,
                    success=True,
                    duration_s=duration_s,
                    reply_chars=len(reply_text),
                    prompt=sent_prompt,
                )
            except Exception:
                pass

        if session is not None:
            resolved_session_id = session.provider_session_id
            now = time.time()
            provider_session_id = getattr(agent, "session_id", None)
            if isinstance(provider_session_id, str) and provider_session_id:
                session.provider = lead
                session.provider_session_id = provider_session_id
                resolved_session_id = provider_session_id
            session.turns.append({"role": "user", "text": prompt, "ts": now})
            session.turns.append({"role": "assistant", "text": str(reply), "ts": now})
            session.updated_at = now
            SessionStore.save_session(session)
        else:
            resolved_session_id = getattr(agent, "session_id", None)

        return PassthroughResult(
            reply=reply,
            provider_used=provider_used,
            session_id=resolved_session_id,
            error=None,
        )

    except Exception as exc:  # pragma: no cover - integration behavior
        duration_s = time.time() - start_s
        if capture:
            try:
                _run_capture(
                    run_dir,
                    provider=provider_used,
                    model=model,
                    effort=effort,
                    session_id=session.session_id if session else None,
                    success=False,
                    duration_s=duration_s,
                    reply_chars=0,
                    prompt=sent_prompt,
                )
            except Exception:
                pass
        resolved_session_id = None
        if session is not None:
            resolved_session_id = session.provider_session_id
        return PassthroughResult(
            reply="",
            provider_used=provider_used,
            session_id=resolved_session_id,
            error=str(exc),
        )

"""Shared, memoized model-list discovery for worker CLIs (issue #46).

Hardcoded model literals + hard validation broke dispatch the moment a CLI
update/account change renamed or retired a model: the rejected model surfaced as
a blank failure and masqueraded as a usage wall. This module replaces the
"static list is the gate" posture with "discover when possible, static fallback
always, never raise":

  * :func:`discover_models` shells out to a CLI's model-list subcommand, parses
    stdout with a caller-supplied parser, and returns the parsed names — or the
    caller's static ``fallback`` on ANY failure (binary missing, nonzero exit,
    empty output, parse error, timeout). It NEVER raises.
  * Results are memoized process-wide per ``cli_key`` with a TTL so a hot suite
    of dispatches re-shells at most once per ``ttl`` window. An ``asyncio.Lock``
    per key keeps concurrent callers from stampeding the subprocess.
  * Discovery is lazy and fully fallback-covered, so unit tests stay hermetic and
    fast: a CLI that isn't installed simply yields the static fallback with no
    network and no hang.

The result (discovered-when-available, else the static fallback — never both
merged) is *advisory*: callers use it to surface the model list they expose,
never as a hard gate (see harness validation, which now warns + passes an unknown
model through rather than rejecting it).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default per-call subprocess timeout (seconds). A model-list subcommand is a
# trivial, near-instant query; if it hangs we abandon it and fall back rather
# than block a dispatch.
_DEFAULT_SUBPROCESS_TIMEOUT = 10.0

# Default memoization TTL (seconds). Model rosters change at most on a CLI
# update, so a 10-minute window is plenty fresh while keeping a busy suite of
# dispatches from re-shelling on every call.
_DEFAULT_TTL = 600.0

# Process-level cache: cli_key -> (models, monotonic_expiry). Guarded per-key by
# a lock in ``_locks`` so concurrent callers for the same CLI don't stampede.
_cache: Dict[str, Tuple[List[str], float]] = {}
_locks: Dict[str, asyncio.Lock] = {}


def _lock_for(cli_key: str) -> asyncio.Lock:
    lock = _locks.get(cli_key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[cli_key] = lock
    return lock


def reset_cache(cli_key: Optional[str] = None) -> None:
    """Clear the memoization cache (test hook).

    ``cli_key=None`` clears everything; a specific key clears just that entry.
    Locks are left in place (cheap, and dropping them mid-flight would race)."""
    if cli_key is None:
        _cache.clear()
    else:
        _cache.pop(cli_key, None)


async def _run_list_cmd(list_argv: List[str], timeout: float) -> Optional[str]:
    """Run the model-list subcommand; return decoded stdout or None on failure.

    Swallows every failure mode (binary missing, nonzero exit, timeout) and
    returns None so the caller falls back. Never raises."""
    try:
        process = await asyncio.create_subprocess_exec(
            *list_argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:  # binary missing / not executable / OS error
        logger.debug("model discovery: failed to spawn %r: %s", list_argv, exc)
        return None
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.debug("model discovery: %r timed out after %.1fs", list_argv, timeout)
        try:
            process.kill()
            # Reap the killed child so it can't linger as a zombie or emit an
            # asyncio "subprocess still running" ResourceWarning at GC. Bounded
            # and fully swallowed — this must never raise (never-raise contract).
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except Exception:
            pass
        return None
    except Exception as exc:
        logger.debug("model discovery: %r errored: %s", list_argv, exc)
        return None
    if process.returncode != 0:
        logger.debug("model discovery: %r exited %s", list_argv, process.returncode)
        return None
    try:
        return stdout.decode(errors="replace")
    except Exception:
        return None


async def discover_models(
    cli_key: str,
    *,
    list_argv: Optional[List[str]],
    parse: Callable[[str], List[str]],
    fallback: List[str],
    ttl: float = _DEFAULT_TTL,
    timeout: float = _DEFAULT_SUBPROCESS_TIMEOUT,
) -> List[str]:
    """Return the discovered model list for ``cli_key``, or the static ``fallback``.

    Shells out to ``list_argv``, parses stdout via ``parse``, and returns the
    parsed names on success; on ANY failure (missing binary, nonzero exit, empty
    output, parse error, timeout) returns ``fallback``. NEVER raises.

    Results are memoized per ``cli_key`` for ``ttl`` seconds (``time.monotonic``),
    so a second call within the window does not re-shell. Pass ``list_argv=None``
    for a CLI with no model-list subcommand (e.g. claude/agy): discovery is
    skipped and ``fallback`` is returned + memoized, keeping the call path uniform.

    The returned list is always a fresh copy the caller may mutate freely.
    """
    now = time.monotonic()
    cached = _cache.get(cli_key)
    if cached is not None and cached[1] > now:
        return list(cached[0])

    async with _lock_for(cli_key):
        # Re-check under the lock: a concurrent caller may have just populated it.
        now = time.monotonic()
        cached = _cache.get(cli_key)
        if cached is not None and cached[1] > now:
            return list(cached[0])

        models = list(fallback)
        if list_argv:
            stdout = await _run_list_cmd(list_argv, timeout)
            if stdout is not None:
                try:
                    parsed = parse(stdout)
                except Exception as exc:
                    logger.debug("model discovery: parse error for %s: %s", cli_key, exc)
                    parsed = []
                if parsed:
                    models = list(parsed)
                else:
                    logger.debug(
                        "model discovery: %s yielded no models; using static fallback",
                        cli_key,
                    )

        _cache[cli_key] = (list(models), time.monotonic() + ttl)
        return list(models)

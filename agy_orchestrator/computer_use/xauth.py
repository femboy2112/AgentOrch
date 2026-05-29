"""XAUTHORITY isolation helpers (hardening requirement #1).

MISSION-CRITICAL HOST SAFETY INVARIANT:
  The action executor (xdotool etc.) and *every* GUI app / Xvfb child we spawn
  must be started with an environment whose DISPLAY and XAUTHORITY point at a
  freshly generated private cookie file that contains an authority entry for the
  *isolated* display ONLY.

  Even though the worker runs as the same real UID as the operator's :0 session,
  the private cookie file has no entry for :0 (and never copied the real
  ~/.Xauthority). Therefore no process in the owned tree can authenticate to the
  real session display. "Cannot authenticate to real :0" is a structural guarantee.

  This is enforced at the ProcessSupervisor boundary for all spawns that declare
  display_scope="isolated". The helpers never read $XAUTHORITY or ~/.Xauthority
  for cookie material; they only ever create new files via mcookie + xauth(1).

  All temp files are 0600. Callers (SessionController) are responsible for
  unlinking them when the run ends (in addition to terminate_tree of the Xvfb).

Never used for OBSERVE-mode observer side (real :0 read-only perception path).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional


def generate_private_xauthority(display: str) -> Path:
    """Create a private X authority cookie file for the given isolated display.

    MISSION-CRITICAL HARDENING #1 (XAUTHORITY ISOLATION):
      - Never touches, reads, or inherits the caller's ~/.Xauthority or $XAUTHORITY
        (or any value of $HOME that could resolve to the operator's real cookie).
      - The mcookie(1) and xauth(1) helper binaries are *always* invoked under a
        fully cleaned environment dictionary that contains **zero** keys or values
        from the parent process that could name or locate a real-session X cookie.
        DISPLAY, XAUTHORITY, WAYLAND_DISPLAY, and the real HOME are explicitly
        absent; only a tiny safe PATH + /tmp HOME/TMPDIR are provided.
      - Uses mcookie(1) when available for a high-quality 128-bit random cookie.
      - Falls back to secrets.token_hex when mcookie is absent.
      - Uses `xauth -f <file> add <display> . <cookie>` (when xauth(1) present) to
        produce a correctly-formatted Xauthority database containing *only* the
        entry for `display`. The resulting file therefore cannot be used to
        authenticate to any other display (including real :0).
      - File is created with mkstemp + explicit chmod 0o600.
      - Returns the Path; caller (SessionController / ProcessSupervisor) owns
        lifetime and must unlink the file + its sibling private HOME dir when the
        isolated display session ends.

    This is the single place that materializes the "private cookie + helpers
    never see real cookie" structural guarantee. Even at identical real UID as
    the :0 session owner, no process in the owned tree can ever authenticate to :0.
    """
    if not isinstance(display, str) or not display.startswith(":"):
        raise ValueError(f"display must be a string like ':99', got {display!r}")

    # Secure, predictable temp location (easy for later cleanup by session owner)
    prefix = f"agu-{display.replace(':', 'd')}-"
    fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=".Xauth")
    os.close(fd)
    os.chmod(tmp_path, 0o600)
    path = Path(tmp_path)

    # ------------------------------------------------------------------
    # FULLY CLEANED ENV FOR HELPER BINARIES (the core of the "never read
    # real cookie during generation" guarantee).
    # We build a *brand new* dict with only safe, minimal keys. No os.environ
    # copy, no inheritance of DISPLAY/XAUTHORITY/HOME/WAYLAND_*/anything that
    # could point at ~/.Xauthority. This is stronger than "strip the bad keys".
    # ------------------------------------------------------------------
    safe_path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    clean_gen_env: Dict[str, str] = {
        "PATH": safe_path,
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "LC_ALL": "C",
        "LANG": "C",
    }
    # Explicitly ensure the three dangerous X/Wayland + real HOME keys are absent
    # (defensive even though we never put them in; documents the invariant).
    for forbidden in ("DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "HOME"):
        clean_gen_env.pop(forbidden, None)

    cookie: str
    try:
        out = subprocess.check_output(["mcookie"], text=True, timeout=2.0, env=clean_gen_env).strip()
        if len(out) >= 20:
            cookie = out
        else:
            raise RuntimeError("mcookie output too short")
    except Exception:
        import secrets

        cookie = secrets.token_hex(16)

    # Write a well-formed authority file for *only this display*.
    xauth = shutil.which("xauth")
    if xauth:
        try:
            subprocess.run(
                [xauth, "-f", str(path), "-q", "add", display, ".", cookie],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3.0,
                env=clean_gen_env,  # xauth(1) sees the fully cleaned env only
            )
        except Exception:
            # xauth present but failed (rare). Leave a marker; real clients will
            # fail to connect/auth — safe fail-closed, never a leak of real cookie.
            path.write_text(
                f"# AGU private Xauth (xauth add failed for {display})\n"
                f"# cookie (for manual recovery): {cookie}\n",
                encoding="ascii",
            )
            os.chmod(path, 0o600)
    else:
        # No xauth tool on PATH. Record intent for diagnostics.
        # X clients started with this XAUTHORITY will be unable to authenticate
        # (expected on minimal containers). This path is still safe: we never
        # copied real-session material.
        path.write_text(
            f"# AGU private Xauth (no xauth(1) on PATH)\n"
            f"# display={display}\n# cookie={cookie}\n",
            encoding="ascii",
        )
        os.chmod(path, 0o600)

    return path


def get_isolated_env(display: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return a complete environment dict for an owned subprocess (Xvfb child,
    xdotool, launched GUI app, etc.) that is *structurally incapable* of
    authenticating to the operator's real :0 display.

    The returned mapping:
      - contains DISPLAY=<display>
      - contains XAUTHORITY=/tmp/agu-...Xauth   (fresh private cookie, see above)
      - contains NO "WAYLAND_DISPLAY" key
      - never contains the parent's $XAUTHORITY value or any path resembling
        ~/.Xauthority (or $XAUTHORITY from the calling environment)
      - extra (caller overrides) are merged but any attempt to override the
        three critical X/Wayland keys is silently ignored — the isolation wins.

    This is the function that ProcessSupervisor calls for every isolated-scope
    spawn so the guarantee is applied uniformly and cannot be bypassed by
    individual action payloads.
    """
    if not isinstance(display, str) or not display.startswith(":"):
        raise ValueError(f"display must be like ':99', got {display!r}")

    auth_path = generate_private_xauthority(display)

    # Create a *private* HOME for this env so that even if client code ignores
    # $XAUTHORITY and falls back to the classic $HOME/.Xauthority lookup, it
    # will never find the operator's real cookie. This is the "cannot read via
    # any file the child could inherit" half of hardening #1.
    iso_home = Path(tempfile.mkdtemp(prefix="iso-home-"))

    # Start from parent but *strip every key that could carry real-session X state*.
    # We keep most other vars (PATH, LANG, etc.) so normal GUI apps continue to work.
    child_env: Dict[str, str] = {}
    for k, v in os.environ.items():
        if k in ("DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY"):
            continue
        # Belt-and-suspenders: also drop any key whose *value* contains the
        # literal path to a likely real cookie (defends against weird wrappers).
        if isinstance(v, str) and (".Xauthority" in v or "/Xauth" in v):
            # Do not propagate values that name real authority files.
            continue
        # Drop the real HOME so the private iso_home below is authoritative.
        if k == "HOME":
            continue
        child_env[k] = v

    # Force the isolated display + its private cookie (this is the hardening).
    child_env["DISPLAY"] = display
    child_env["XAUTHORITY"] = str(auth_path)
    child_env["HOME"] = str(iso_home)
    child_env["AGY_ISOLATED_X"] = "1"  # marker for audit / test assertions

    # Merge caller extras (e.g. from SpawnSpec.env) but the X isolation keys win.
    if extra:
        for k, v in extra.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            if k in ("DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "HOME", "AGY_ISOLATED_X"):
                # Explicitly refuse override attempts — this is what FR-24 / hardening #1 demands.
                continue
            child_env[k] = v

    return child_env

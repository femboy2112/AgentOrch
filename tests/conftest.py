from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator

import pytest

from agy_orchestrator.computer_use import ProcessSupervisor, get_isolated_env
from agy_orchestrator.computer_use.models import IsolatedDisplaySpec

ROOT = Path(__file__).resolve().parent.parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


@pytest.fixture(autouse=True)
def _isolate_telegram_state(tmp_path, monkeypatch):
    """Hermetic isolation: never let a test read or write the OPERATOR's real
    Telegram state/whitelist files (DEFAULT_STATE_PATH / DEFAULT_USERS_PATH under
    ~/tgbot/data/). Without this, TelegramNotifier._live_state() reads the LIVE
    bot_state.json, so a real mute/quiet-window entry for the test FAKE_CHAT_ID
    (444555666) suppresses delivery — making the notifier tests fail depending on
    the wall-clock (in/out of the quiet window) and pytest collection order. Point
    both paths at an empty per-test tmp dir; a test may still override either.
    """
    monkeypatch.setenv("AGY_TELEGRAM_STATE", str(tmp_path / "bot_state.json"))
    monkeypatch.setenv("AGY_TELEGRAM_USERS", str(tmp_path / "users.json"))


def _has_xvfb() -> bool:
    return shutil.which("Xvfb") is not None


@pytest.fixture
def isolated_xvfb(request: pytest.FixtureRequest) -> Generator[Dict[str, Any], None, None]:
    """Guaranteed-cleanup isolated Xvfb session fixture for computer-use tests.

    - Picks a high, low-collision display number based on pid.
    - Spawns via ProcessSupervisor.spawn_isolated_display (owned tree, killable).
    - Yields dict with display, env (private XAUTH + HOME), sup, root_id.
    - ALWAYS tears down the full owned tree (Xvfb + any launched children) in
      finally + request.addfinalizer, even if test crashes or raises.
    - Skips cleanly if Xvfb missing (no real :0 ever involved).
    - All tests using this stay hermetic: never touch real :0, never inherit
      operator X cookie (hardening #1 enforced by get_isolated_env).
    """
    if not _has_xvfb():
        pytest.skip("Xvfb not on PATH; isolated-only computer-use tests require it")

    sup = ProcessSupervisor()
    display = f":{120 + (os.getpid() % 200)}"
    env = None
    xv = None
    root_id = None

    def _cleanup() -> None:
        # Double guarantee: explicit terminate + registry walk
        if root_id:
            try:
                sup.terminate_tree(root_id)
            except Exception:
                pass
        # Any stragglers registered under this sup
        for rid in list(sup._registry.keys()):
            try:
                sup.terminate_tree(rid)
            except Exception:
                pass

    # Register cleanup before any spawn so even early failures are covered
    # (pytest will still call finalizer on exception paths)
    # We also wrap in try/finally below for belt-and-suspenders.

    # Register pytest finalizer *before* spawn so crashes, asserts, or KeyboardInterrupt
    # still trigger full owned-tree kill (Xvfb + grandchildren via killpg in terminate_tree).
    request.addfinalizer(_cleanup)

    try:
        env = get_isolated_env(display)
        xv = sup.spawn_isolated_display(
            IsolatedDisplaySpec(display=display, screen="640x480x24", timeout_ms=2500)
        )
        root_id = xv.root_id
        # Give Xvfb a moment to be ready for clients (xdotool etc.)
        time.sleep(0.35)
        yield {"display": display, "env": env, "sup": sup, "root_id": root_id}
    finally:
        _cleanup()

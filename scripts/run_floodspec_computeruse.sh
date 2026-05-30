#!/usr/bin/env bash
# Phase 1: generate the computer-use agent design doc via FloodSpec.
# Architect=codex, critic=agy (defaults) — cross-family, avoids claude
# (account-sharing rule, since a claude session is driving). Writes
# runs/<id>/spec.md and a copy at COMPUTER_USE_DESIGN.md.
#
# Run from the AgentOrch repo root with the venv active (or it uses the
# venv python directly below).

set -uo pipefail
cd /home/leah/AgentOrch

PY=/home/leah/AgentOrch/.venv/bin/python

"$PY" -m harness spec "A GUI computer-use agent worker for AgentOrch that perceives the screen by programmatic reconstruction (accessibility tree + window geometry + OCR + browser DOM) instead of pixel vision, driven by the existing text CLI workers" \
  --max-iterations 8 \
  -c "Two perception modes: (a) ISOLATED — a private Xvfb virtual display the agent both perceives and acts in (default for testing); (b) OBSERVE — may perceive the operator's real :0 display (screenshots, accessibility tree, window geometry, including visible terminals) but is STRUCTURALLY INCAPABLE of injecting any input (clicks, keystrokes, drags) into the real session. Actionable control is confined to the isolated display; the real session is strictly read-only" \
  -c "Must not read, signal, or interfere with any process it did not itself spawn; no global X input grabs; no killing or sending signals to foreign processes (other claude/codex/grok/agy/orchestrator instances must be untouched)" \
  -c "Must not be able to crash or destabilize the host: bounded total action/step count, per-action timeouts, CPU/memory/process-count resource caps (psutil), and a self-contained, fully killable subprocess tree" \
  -c "Perception is programmatic text reconstruction only (AT-SPI accessibility tree via PyGObject, wmctrl/xdotool/xprop geometry, tesseract OCR fallback, browser DOM via Playwright/CDP); no pixel-vision model, no Anthropic SDK, no API keys" \
  -c "Reasoning runs through the claude and codex CLIs over their OAuth logins — claude is the lead engine for high-priority tasks, codex is the primary fallback; integrate as a standard AgentOrch worker (adapter/role pattern) with an is_available() probe that degrades gracefully (AT-SPI -> OCR+geometry if PyGObject absent) and streams actions into runs/<id>/events.jsonl for the dashboard" \
  -c "Every action must be auditable: logged with target element handle, coordinates, and the model's stated rationale; destructive or irreversible actions gated behind an explicit confirmation/dry-run step" \
  -o COMPUTER_USE_DESIGN.md

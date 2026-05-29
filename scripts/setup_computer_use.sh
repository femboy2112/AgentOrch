#!/usr/bin/env bash
# Install dependencies for AgentOrch's computer-use agent worker.
#
# Perception is PROGRAMMATIC (accessibility tree + window geometry + OCR +
# browser DOM) — no pixel-vision model, no Anthropic SDK, no API keys. This
# script installs the system tools + Python packages that perception relies on.
#
# Safe to re-run. apt steps need sudo; pip goes into AgentOrch/.venv.
# Failures in any step are reported but do NOT abort the verify block at the end,
# so you always get a clear READY / INCOMPLETE picture.

set -uo pipefail

VENV="/home/leah/AgentOrch/.venv"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv python not found at $PY" >&2
  echo "Edit VENV at the top of this script to point at your venv." >&2
  exit 1
fi

note() { printf '\n=== %s ===\n' "$*"; }
warn() { printf '!!! %s\n' "$*" >&2; }

# --- 1. apt: OCR engine + AT-SPI runtime + build deps for PyGObject/pycairo ---
note "1/4 apt packages (sudo)"
if command -v apt-get >/dev/null; then
  sudo apt-get update || warn "apt-get update failed (continuing)"
  sudo apt-get install -y \
    tesseract-ocr \
    at-spi2-core gir1.2-atspi-2.0 \
    libgirepository1.0-dev gobject-introspection libcairo2-dev \
    pkg-config python3-dev build-essential \
    || warn "some apt packages failed to install"
else
  warn "apt-get not found — translate package names for your distro"
fi

# --- 2. pip into the venv (PyGObject capped <3.52 to match libgirepository1.0) -
note "2/4 pip packages into $VENV"
"$PY" -m pip install --upgrade "PyGObject<3.52" pycairo pytesseract psutil playwright \
  || warn "pip install failed for one or more packages (PyGObject is the usual culprit — see notes)"

# --- 3. browser for DOM/CDP perception (playwright's bundled chromium) ---------
note "3/4 playwright chromium"
"$PY" -m playwright install chromium || warn "playwright chromium download failed"
sudo "$PY" -m playwright install-deps chromium || warn "playwright system-lib install failed"

# --- 4. verify ----------------------------------------------------------------
note "4/4 verify"
"$PY" - <<'EOF'
import shutil
bins = ["Xvfb", "xvfb-run", "xdotool", "scrot", "wmctrl",
        "xprop", "xwininfo", "tesseract"]
miss = [b for b in bins if not shutil.which(b)]

mods = []
for m in ("PIL", "pytesseract", "psutil", "playwright"):
    try:
        __import__(m)
    except Exception as e:
        mods.append(f"{m}({e.__class__.__name__})")

try:
    import gi
    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi  # noqa: F401
    atspi = "OK"
except Exception as e:
    atspi = f"FAIL({e.__class__.__name__})"
    mods.append("gi/Atspi")

print("binaries missing       :", miss or "none")
print("AT-SPI (a11y tree)     :", atspi)
print("py modules missing     :", mods or "none")
print()
if not miss and not mods:
    print("=> READY — all computer-use perception deps present")
else:
    print("=> INCOMPLETE — see above")
    if "gi/Atspi" in mods:
        print("   note: AT-SPI is the PREMIUM perception tier, not a hard gate.")
        print("         The worker degrades to geometry + OCR if it's absent.")
EOF

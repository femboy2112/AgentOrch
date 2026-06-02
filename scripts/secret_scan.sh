#!/usr/bin/env bash
# Local one-shot secret scan mirroring the CI gate (.github/workflows/secret-scan.yml).
#
# Scans the full git history with the repo's .gitleaks.toml config. Use it for a
# manual baseline (e.g. before a public push) without waiting on CI.
#
# Resolution order for the gitleaks engine:
#   1. a `gitleaks` binary on PATH
#   2. docker (ghcr.io/gitleaks/gitleaks) if available
#   3. otherwise: print install instructions and exit non-zero.
#
# Exit code is gitleaks' own: 0 = clean, 1 = leaks found.
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

CONFIG=".gitleaks.toml"
REPORT="${1:-}"   # optional: path to write a JSON report

args=(detect --source "." --config "$CONFIG" --redact --verbose)
if [[ -n "$REPORT" ]]; then
  args+=(--report-format json --report-path "$REPORT")
fi

if command -v gitleaks >/dev/null 2>&1; then
  echo "==> gitleaks (native) over full history"
  exec gitleaks "${args[@]}"
elif command -v docker >/dev/null 2>&1; then
  echo "==> gitleaks (docker) over full history"
  exec docker run --rm -v "$REPO_ROOT:/repo" -w /repo \
    ghcr.io/gitleaks/gitleaks:latest "${args[@]}"
else
  cat >&2 <<'EOF'
ERROR: gitleaks not found and docker unavailable.

Install one of:
  * binary : https://github.com/gitleaks/gitleaks/releases  (then put on PATH)
  * go     : go install github.com/gitleaks/gitleaks/v8@latest
  * brew   : brew install gitleaks

CI still enforces the scan on every push regardless of local tooling.
EOF
  exit 2
fi

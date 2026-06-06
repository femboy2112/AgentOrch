# AgentOrch reference documentation

Accurate, source-checked reference for using AgentOrch — both from the command
line and as a Python library. Each page is generated against the actual code and
adversarially verified, so flags, defaults, signatures, and env vars trace to
source rather than to prose.

| Page | Audience | Covers |
|---|---|---|
| [CLI reference](cli.md) | Operators running AgentOrch from a terminal | Every subcommand and flag of `python -m harness` and `python -m agy_orchestrator`, the workflow modes, provider chains, exit codes, and plan-JSON shapes. |
| [Python API](python-api.md) | Programmers importing AgentOrch as a library | The `AgentInstance` contract, `make_fallback_agent`, `QualityVerifier`/`VerifierResult`, every workflow class and its `execute(...)` signature, the ledger, and graph plans. |
| [Configuration](configuration.md) | Operators tuning runtime behavior | The full `AGY_*` / `TELEGRAM_BOT_KEY` / `CODEX_HOME` environment-variable surface, grouped by subsystem, with defaults and accepted values. |
| [Telegram commands](telegram-commands.md) | Users driving builds from their phone | Every bot command (`/build`, `/cancel`, `/retry`, `/run`, `/notify`, `/clear`, …), the whitelist/auth model, verbosity levels, and inline callbacks. |

For LLM coding agents, the root [`llms.txt`](../../llms.txt) is a dense,
token-efficient digest of this entire surface and is the best first read.

## See also

- [`README.md`](../../README.md) — project overview and quick start.
- [`AGENTS.md`](../../AGENTS.md) — integrating AgentOrch as a callable tool, and the rules for workers dispatched inside the repo.
- [`CLAUDE.md`](../../CLAUDE.md) — operator/maintainer guide and runtime rules.
- Design specs in [`docs/`](../) — architecture and feature designs (e.g. [graph execution](../graph-execution-design.md), [reconciliation station](../reconciliation-station.md), [Telegram](../telegram.md)).

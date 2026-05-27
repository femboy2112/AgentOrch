# AGENTS.md — instructions for worker agents (codex, agy, claude, grok)

You are a coding worker invoked inside this repository. Read this before editing.

## What this project is

AgentOrch builds software by orchestrating worker CLIs. It is a cloud-only
multi-agent orchestrator (`agy_orchestrator/`) plus an operator harness
(`harness/`) that dispatches coding work and captures every run. The current
goal is to upgrade and polish it toward a public release.

## Hard rules

- Make changes **directly on disk** in the current working directory. Implement
  the instruction; do not merely describe it and do not ask clarifying questions.
- **Never run `sudo`.** Never modify anything outside the folder you were told to
  work in.
- Prefer **stdlib / zero-dependency** solutions unless told otherwise. If a
  dependency is essential, state why.
- Match the existing code style of nearby files. Keep changes minimal and scoped.
- End your reply with a short list of files you created/modified and why.

## Verification

If a test or build command applies, ensure it passes before finishing. The
project tests run with `python -m pytest -q`.

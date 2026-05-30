# Token Telemetry Coverage

Per-call usage telemetry is now emitted to `runs/<id>/events.jsonl` as `kind: "usage"`
with `data.usage_kind="call"`, then rolled up into `runs/<id>/meta.json -> tokens`.

Worker coverage in normal non-interactive CLI output:

- `claude`: **Yes** (`--output-format json` / `stream-json` includes `usage`).
- `codex`: **Conditional** (`turn.completed` usage is parsed when JSON turn events are present).
- `grok`: **Usually no** (normal payload typically omits `usage`; parsed if present).
- `agy`: **No** (plain text output; no machine-readable usage fields).

When usage is not available, events record:

- `token_source: "unavailable"`
- `input_tokens: null`
- `output_tokens: null`
- `cache_read_tokens: null`

Manual interactive probes (for operators) when unavailable:

- `codex` (force JSON events): `codex exec --json -`
- `agy`: use `agy -i` and inspect provider-side usage/console reporting
- `grok`: use interactive/verbose mode if provider exposes usage in-session

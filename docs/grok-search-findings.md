# GROK_SEARCH_FINDINGS.md

**Date of investigation**: 2026-05-28
**Context**: Diagnosis of why `grok` (the xAI Grok Build TUI / agentic CLI) crashes or fails when its `web_search` tool is used — especially "deep" web searches — even when the user specifies fewer than 5 domains in config. Investigation performed inside the AgentOrch repository without modifying any source files during the root-cause phase.

## Executive Summary

The harness (`python -m harness do ... --web-search`) and the `agy_orchestrator` layer have **no meaningful support** for controlling or hardening the `web_search` tool when the generator/critic is `grok`.

- `--web-search` only ever affects the `codex` worker.
- For `grok`, web search is always enabled by default. The only toggle (`web_search=False`) only adds `--disable-web-search`.
- There is no path to pass domain allowlists, max-turns, web-search model overrides, or result budgets through the dispatch/roles layer to a `GrokAgent`.
- The external `grok` binary (v0.2.3) only documents `allowed_domains` under `[toolset.web_fetch]`. There is **no** `[toolset.web_search]` section and no documented domain limiting for the `web_search` tool itself.
- When the inner `grok` agent performs deep / multi-turn web research (multiple `web_search` + `web_fetch` cycles), it quickly exceeds the model's default turn limit. The binary emits a `{"type":"error", ... "max_turns exceeded"}` envelope on **stdout** (often with exit code 0).
- `GrokAgent._extract_text` / `_postprocess` happily returns this error blob as if it were normal agent output. The orchestrator treats the dispatch as "successful."
- Attempting to configure domains under the wrong section (or at all for `web_search`) either has no effect or causes the `grok` binary itself to choke during config parsing / tool initialization.

Result: "calling your CLI with websearch tool use" either produces silent garbage, hits turn limits, or crashes at startup when people try the only visible domain knob from the docs.

## Detailed Findings

### 1. Harness / Dispatch Layer (only codex is affected)

- [harness/cli.py:150](harness/cli.py:150) — `--web-search` flag.
- [harness/dispatch.py:167](harness/dispatch.py:167):
  ```python
  codex_config = ["tools.web_search=true"] if web_search else None
  ```
- [harness/roles.py:55](harness/roles.py:55) and 85 — `config_overrides` only ever applied to `codex` via `_cfg_for_token`.
- `build_role_agent`, `build_master_agent_class`, `_configs_for`, etc. never inject anything web-search-related for `grok`.
- `describe_chain` also ignores web-search state.

When the generator chain includes `grok` (the default secondary), `--web-search` is a no-op for that provider.

### 2. GrokAgent Implementation

File: [agy_orchestrator/core/agents/grok_agent.py](agy_orchestrator/core/agents/grok_agent.py)

Key points:
- `__init__` (line 39) accepts `web_search: bool = True`.
- Only effect (line 88):
  ```python
  if not self.web_search:
      cmd.append("--disable-web-search")
  ```
- `_build_cmd` (lines 72-96) produces the exact invocation the orchestrator uses:
  ```
  grok --prompt-file <tmp> --output-format json
       --always-approve --permission-mode bypassPermissions
       [-m MODEL]
  ```
  No `--max-turns`, no web-related config, no domain flags.
- Output parsing (lines 149-172):
  - `_extract_text` does `json.loads` on the whole stdout, then a greedy `re.search(r"\{.*\}", s, re.DOTALL)` fallback.
  - Only extracts top-level `"text"`.
  - `_extract_session_id` is a simple regex.
  - No handling for `{"type":"error"...}` envelopes, tool-use side-effects, or large "thought" fields containing search results.
- `additional_flags` are passed through, but the harness/roles layer never populates anything useful for web search or turn limits.
- `filter_stderr` only drops two very specific benign lines.

### 3. External `grok` Binary (v0.2.3) Behavior & Documentation

The real CLI lives at `~/.grok/bin/grok` (symlinked from `~/.local/bin/grok`).

From `--help`, user-guide docs, and `~/.grok/README.md` (the canonical reference the environment instructs us to read):

- Only web-search-related CLI flag: `--disable-web-search`.
- Config surface for domains (only one place):
  ```toml
  [toolset.web_fetch]
  allowed_domains = ["docs.rs", "x.ai"]   # overrides built-in ~84-domain allowlist
  ```
  (See: 05-configuration.md:101, README ~1310 and ~2285.)
- `web_search` tool has **no** equivalent `allowed_domains`, no documented depth / result limit, and no `[toolset.web_search]` section.
- `[models] web_search = "..."` only selects which model the *tool* calls (Responses API).
- In headless mode (exactly what the orchestrator uses via `--prompt-file --output-format json`), the agent loop counts every tool call / reasoning step toward an internal `max_turns`.
- When deep web research happens (`web_search` → follow-up searches → `web_fetch` on results → more searches), the limit is exceeded.
- The binary emits on stdout (with rc=0 in observed cases):
  ```json
  {"type":"error","message":"Internal error: \"max_turns exceeded: limit is 3, but got 4 messages\""}
  ```
  Plus ERROR lines on stderr (including the "failed to watch root recursively" noise that GrokAgent partially filters).

- `web_search` is enabled by default and has full network access even under restrictive sandboxes (docs confirm this).

### 4. Why "Even <5 Domains" Still Crashes

Users naturally look at the only documented `allowed_domains` example and try to apply it to the tool they actually want to constrain (`web_search`):

- Placing it under `[toolset.web_search]`, at root level, or under `[tools]` produces an unrecognized section/key.
- The `grok` binary's config loader (TOML + validation at startup) treats this as fatal in many cases → immediate crash / early exit before any prompt is processed.
- Even if it were accepted under the correct section, it would only affect `web_fetch`, not the `web_search` tool the agent calls first during "deep" research.
- The agent still performs unbounded turn loops of search + context injection. Domains don't reduce turn count; they only gate individual `web_fetch` approvals.

### 5. Interaction With AgentOrch Workflows

- `master`, `adversarial`, `cascade`, etc. all go through the same `GrokAgent` or fallback wrapper.
- No per-provider turn budgets, no web-specific compaction, no special error detection for `grok` + search tool failures.
- The quality ledger / verifier path sees either garbage output or a "successful" run that actually emitted an internal error.
- Long-running dispatches (the ones most likely to benefit from web search) are exactly the ones that hit the turn wall first.

### 6. Minor Parsing Edge Cases Exposed by Web Search

- Greedy `\{.*\}` regex can capture the wrong object when logs + multiple JSON fragments appear (web search results sometimes surface in "thought" or ancillary output).
- Error envelopes with rc=0 are treated as successful agent text.
- Large search result text inside the JSON is never a problem for extraction of `"text"`, but it *does* bloat the inner agent's context across turns, accelerating the max_turns problem.

## Affected Files (read-only during diagnosis)

- `harness/cli.py`
- `harness/dispatch.py`
- `harness/roles.py`
- `agy_orchestrator/core/agents/grok_agent.py`
- `agy_orchestrator/core/agent.py` (base run loop + timeout)
- `~/.grok/docs/user-guide/05-configuration.md`
- `~/.grok/docs/user-guide/14-headless-mode.md`
- `~/.grok/docs/user-guide/18-sandbox.md`
- `~/.grok/README.md` (primary reference)
- `~/.grok/config.toml` (user's actual config — inspected, not modified)

## How to Reproduce (for future verification)

```bash
# Minimal headless trigger that exercises web_search
echo 'Use web_search to find the latest stable pytest version. Reply with only the version number.' > /tmp/p.txt

grok --prompt-file /tmp/p.txt --output-format json \
     --always-approve --permission-mode bypassPermissions \
     --max-turns 3   # force the error path quickly
```

Then feed the resulting stdout into `GrokAgent._extract_text` — you will see either the clean answer or the raw error JSON.

With a real coding task that benefits from up-to-date info (and no artificial `--max-turns`), the agent will usually exceed the default limit after a handful of search/fetch cycles.

## Recommended Direction (not implemented here)

- Expose `--max-turns`, web-search toggles, and (if the binary ever supports it) domain lists via `GrokAgent` and thread them from `dispatch`/`roles` (similar to how `codex_config` works today).
- Make `_postprocess` / `_extract_text` detect `{"type":"error"...}` and surface a proper `RuntimeError` so the existing retry + fallback machinery kicks in.
- Add documentation or a guard in the harness CLI that warns when a `grok` generator is chosen together with `--web-search` (or with tasks likely to need deep research).
- Consider a small wrapper that injects a turn budget or "search lightly" instruction when web tools are active.

This file captures the complete diagnosis performed on 2026-05-28. No source files in the repository were edited during the investigation itself.

---

## Appendix: Clarification from 400 Responses API Error (2026-05-28 follow-up)

**New evidence provided by user (from Claude runs + session logs):**

When `--max-turns 1000` (and other local mitigations) were tried, the process still failed immediately with a **400 Bad Request** from the backend:

> `Responses API returned 400 Bad Request: {"code":"Client specified an invalid argument","error":"A maximum of 5 domains can be allowed, but 6 were provided."}`

### Exact location of the failure (from session transcripts)

Found in:
`~/.grok/sessions/%2Ftmp%2Fagentorch_research/019e6be9-e9b7-7f91-8175-4f03351a1eb8/chat_history.jsonl`

```json
{"type":"tool_result","tool_call_id":"call-8e27720f-a92d-4bca-883f-3302eca2075d-1",
 "content":"Tool `web_search` failed: Responses API returned 400 Bad Request: 
 {\"code\":\"Client specified an invalid argument\",\"error\":\"A maximum of 5 domains can be allowed, but 6 were provided.\"}"}
```

The model’s own later reasoning in the same transcript:

> “The first web_search failed because I put too many domains (max 5). The other three succeeded...”

### Root cause refinement

- The limit is **server-side** on the `web_search` tool in the Responses API (the backend the `grok` CLI uses for its search model, configured via `[models] web_search`).
- The `grok` binary exposes the tool to the model; the model itself can (and sometimes does) supply an `allowed_domains` (or equivalent) array when it decides to call `web_search`.
- In this case the model generated a list containing **6 domains** on its first tool call.
- This is **dynamic / model-driven**, not (solely) coming from the user’s static `~/.grok/config.toml`.
  - `[toolset.web_fetch].allowed_domains` only affects the separate `web_fetch` tool.
  - The model can independently decide to constrain `web_search` at call time for “careful research.”
- Strong prompts of the form **“Do DEEP web research (USE web_search / web_fetch extensively — verify with current 2024-2026 sources...”** are reliable triggers. The model responds by being cautious and over-specifying the domain list.

### Why previous symptoms were misleading

- Earlier “max_turns exceeded” errors (the `{"type":"error"...}` JSON with rc=0) were secondary symptoms that appeared when the agent was allowed to continue after partial failures or on lighter prompts.
- The real hard blocker on deep research tasks is this 400 on the very first qualifying `web_search` call.
- User attempts to pre-declare a short domain list (<5) in config had no effect on the model-generated list passed at tool invocation time.
- `--max-turns` (any value) cannot help because the rejection occurs at the tool-definition / tool-call layer before the agent loop’s turn counter is meaningfully exercised.

### Connection to AgentOrch harness usage

When `GrokAgent` is used as a generator (via `roles.py` → `build_role_agent`, `dispatch.py`, etc.):

- The `WORKER_PREAMBLE` + any instruction that implies “use fresh sources” or “research thoroughly” can cause the inner `grok` agent to emit the same style of prompt.
- The rich context loaded at startup (AGENTS.md, CLAUDE.md, 12+ user skills/plugins shown by `grok inspect`, marketplace plugins, etc.) gives the model additional material from which to assemble a 6+ domain list.
- Nothing in `GrokAgent._build_cmd`, `additional_flags`, or the fallback wrapper can observe or sanitize the internal `web_search` tool parameters the binary sends to the Responses API.
- The only control the orchestrator currently has is the binary `--disable-web-search` flag. There is no way to request “web_search with at most 5 domains” or “web_search in unrestricted mode.”

### Updated implications

- “grok fails when trying to do deep websearches” is now fully explained: the failure is an unrecoverable 400 on the search tool itself when the model over-provisions `allowed_domains`.
- This is orthogonal to the local turn-limit and output-parsing issues documented in the main body.
- Any future mitigation in AgentOrch would require either:
  - Instructing the model not to use domain restrictions (or to keep lists ≤5),
  - Prepending system-level constraints before the user prompt,
  - Or (ideally) the `grok` CLI exposing a flag / config to cap or disable domain lists on `web_search`.

This appendix was added on 2026-05-28 after the new 400 evidence and session-log analysis. The core diagnosis in the sections above remains accurate; this provides the missing precise failure mode.

---

**End of findings (including appendix).**

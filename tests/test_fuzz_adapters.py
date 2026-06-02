"""Hermetic fuzz / property / edge-case tests for the worker adapters.

These NEVER spawn a real worker CLI and NEVER touch the network. They drive the
pure-ish build/parse methods (build_command, _stdin_bytes, _postprocess, the
JSON-event parsers, usage/session extraction) directly with adversarial inputs
and assert robust behaviour. Each committed test PASSES; genuine defects found
during fuzzing are reported separately with their own repros (not committed as
failing tests).
"""
import json

import pytest

from agy_orchestrator.core.agents.codex_agent import CodexAgent
from agy_orchestrator.core.agents.claude_agent import ClaudeAgent, MODEL_ALIASES
from agy_orchestrator.core.agents.grok_agent import GrokAgent, _parse_grok_models
from agy_orchestrator.core.agents.agy_agent import AgyAgent, resolve_agy_model
from agy_orchestrator.core.agents.mock_agent import MockAgent, mock_agent_class_for


# --------------------------------------------------------------------------- #
# Shared adversarial input corpus.
# --------------------------------------------------------------------------- #
WEIRD_STRINGS = [
    "",
    " ",
    "\n\n\n",
    "\t\r\n",
    "x" * 100_000,                      # huge
    "unicode ☃ \U0001f600 \x00 mid-null",
    "\x1b[31mANSI red\x1b[0m",          # ANSI escape
    "line1\nline2\nno trailing newline",
    "{not valid json",
    "}{][",
    "null", "true", "42", "3.14", "[]", "{}",
    "\ud800 lone surrogate",            # surrogate (Python str can hold it)
    "ç̃ combining marks ä̈",
]

GARBAGE_JSON_LINES = [
    "",
    "   ",
    "not json at all",
    "{",
    "{]",
    '{"unterminated": ',
    "[1, 2, 3]",          # valid json, not a dict
    "42",
    "null",
    '{"type": 123}',      # type not a string
    '{"type": null}',
]


# --------------------------------------------------------------------------- #
# CodexAgent
# --------------------------------------------------------------------------- #
class TestCodexBuildCommand:
    def test_build_command_standard_maps_to_spark(self):
        a = CodexAgent("p", model="standard")
        cmd = a.build_command()
        assert cmd[0:2] == ["codex", "exec"]
        # standard must expand to the ChatGPT-account-valid spark model
        i = cmd.index("--model")
        assert cmd[i + 1] == "gpt-5.3-codex-spark"
        assert cmd[-1] == "-"

    def test_build_command_max_effort_maps_to_xhigh(self):
        a = CodexAgent("p")
        a.effort = "max"
        cmd = a.build_command()
        joined = " ".join(cmd)
        assert 'model_reasoning_effort="xhigh"' in joined

    def test_build_command_none_model_omits_model_flag(self):
        a = CodexAgent("p", model=None)
        cmd = a.build_command()
        assert "--model" not in cmd

    def test_build_command_weird_additional_flags_stringified(self):
        # control chars / unicode in flag values must not crash command build
        a = CodexAgent("p", additional_flags={"weird": "va\nl\x1b[0m☃"})
        cmd = a.build_command()
        assert "--weird" in cmd
        assert cmd[cmd.index("--weird") + 1] == "va\nl\x1b[0m☃"

    def test_build_command_non_str_flag_value_stringified(self):
        a = CodexAgent("p", additional_flags={"n": 5, "f": 1.5, "b": True})
        cmd = a.build_command()
        # every value coerced via str()
        assert "5" in cmd and "1.5" in cmd and "True" in cmd

    def test_last_message_path_stable_per_instance(self):
        a = CodexAgent("p")
        assert a._last_message_path() == a._last_message_path()

    def test_config_overrides_none_is_safe(self):
        a = CodexAgent("p")
        a.config_overrides = None
        # build must not crash when overrides is None
        a.build_command()


class TestCodexStdin:
    def test_stdin_bytes_includes_prompt_and_context(self):
        a = CodexAgent("PROMPT")
        b = a._stdin_bytes("CTX")
        assert b"PROMPT" in b and b"CTX" in b

    def test_stdin_bytes_huge_prompt(self):
        a = CodexAgent("x" * 200_000)
        assert len(a._stdin_bytes()) >= 200_000

    def test_stdin_bytes_unicode_roundtrip(self):
        a = CodexAgent("snow ☃ face \U0001f600")
        assert "☃" in a._stdin_bytes().decode()


class TestCodexParse:
    @pytest.mark.parametrize("raw", GARBAGE_JSON_LINES)
    def test_message_text_from_jsonl_never_crashes(self, raw):
        # individual garbage lines must be ignored, not raise
        assert CodexAgent._message_text_from_jsonl(raw) is None or isinstance(
            CodexAgent._message_text_from_jsonl(raw), str
        )

    def test_message_text_from_jsonl_concatenates_messages(self):
        raw = "\n".join([
            json.dumps({"type": "message", "text": "hello "}),
            "garbage line that is not json",
            json.dumps({"type": "message", "delta": "world"}),
        ])
        assert CodexAgent._message_text_from_jsonl(raw) == "hello world"

    def test_message_text_from_jsonl_nested_item(self):
        raw = json.dumps({"item": {"type": "message", "text": "nested"}})
        assert CodexAgent._message_text_from_jsonl(raw) == "nested"

    def test_message_text_no_message_returns_none(self):
        raw = json.dumps({"type": "turn.completed", "usage": {}})
        assert CodexAgent._message_text_from_jsonl(raw) is None

    def test_postprocess_falls_back_to_jsonl(self):
        a = CodexAgent("p")  # no _codex_last_msg_path set
        raw = json.dumps({"type": "message", "text": "answer"})
        assert a._postprocess(raw) == "answer"

    def test_postprocess_returns_raw_when_no_message(self):
        a = CodexAgent("p")
        assert a._postprocess("plain pre-turn error") == "plain pre-turn error"

    def test_extract_usage_integer_tokens(self):
        a = CodexAgent("p")
        raw = json.dumps({
            "type": "turn.completed",
            "usage": {"input_tokens": 100, "cache_read_tokens": 30,
                      "output_tokens": 10, "total_tokens": 110},
        })
        u = a._extract_usage(raw, "")
        assert u["token_source"] == "cli"
        # fresh = total_input - cached
        assert u["input_tokens"] == 70
        assert u["cache_read_tokens"] == 30

    def test_extract_usage_stderr_tokens_used_fallback(self):
        a = CodexAgent("p")
        u = a._extract_usage("", "tokens used: 1,234\n")
        assert u["total_tokens"] == 1234

    def test_extract_usage_no_data_unavailable(self):
        a = CodexAgent("p")
        u = a._extract_usage("nothing here", "nothing here")
        assert u["token_source"] == "unavailable"
        assert u["total_tokens"] is None

    def test_extract_usage_non_dict_usage_field(self):
        a = CodexAgent("p")
        raw = json.dumps({"type": "turn.completed", "usage": [1, 2, 3]})
        # usage isn't a dict -> ignored, falls through to unavailable
        u = a._extract_usage(raw, "")
        assert u["token_source"] == "unavailable"


# --------------------------------------------------------------------------- #
# ClaudeAgent
# --------------------------------------------------------------------------- #
class TestClaudeBuildCommand:
    def test_standard_maps_to_sonnet(self):
        a = ClaudeAgent("p", model="standard")
        cmd = a._build_base_cmd()
        assert cmd[cmd.index("--model") + 1] == "sonnet"
        assert MODEL_ALIASES["standard"] == "sonnet"

    def test_none_model_omits_flag(self):
        a = ClaudeAgent("p", model=None)
        assert "--model" not in a._build_base_cmd()

    def test_default_output_format_json(self):
        a = ClaudeAgent("p")
        cmd = a._build_base_cmd()
        assert cmd[cmd.index("--output-format") + 1] == "json"

    def test_session_resume_and_fork(self):
        a = ClaudeAgent("p", session_id="abc", fork_session=True)
        cmd = a._build_base_cmd()
        assert "--resume" in cmd and "abc" in cmd and "--fork-session" in cmd


class TestClaudeParse:
    def test_extract_session_id_single_object(self):
        sid = "12345678-1234-1234-1234-123456789abc"
        raw = json.dumps({"session_id": sid})
        assert ClaudeAgent._extract_session_id(raw) == sid

    def test_extract_session_id_pretty_multiline(self):
        sid = "12345678-1234-1234-1234-123456789abc"
        raw = json.dumps({"session_id": sid, "x": {"y": 1}}, indent=2)
        assert ClaudeAgent._extract_session_id(raw) == sid

    @pytest.mark.parametrize("raw", GARBAGE_JSON_LINES + WEIRD_STRINGS)
    def test_extract_session_id_garbage_returns_none_or_str(self, raw):
        out = ClaudeAgent._extract_session_id(raw)
        assert out is None or isinstance(out, str)

    def test_extract_result_text_result_blob(self):
        raw = json.dumps({"type": "result", "result": "THE ANSWER"})
        assert ClaudeAgent._extract_result_text(raw) == "THE ANSWER"

    @pytest.mark.parametrize("raw", WEIRD_STRINGS)
    def test_extract_result_text_never_crashes(self, raw):
        out = ClaudeAgent._extract_result_text(raw)
        assert isinstance(out, str)

    def test_postprocess_captures_session(self):
        sid = "12345678-1234-1234-1234-123456789abc"
        a = ClaudeAgent("p")
        raw = json.dumps({"type": "result", "result": "ok", "session_id": sid})
        text = a._postprocess(raw)
        assert text == "ok"
        assert a.session_id == sid

    def test_extract_usage_from_object(self):
        a = ClaudeAgent("p")
        raw = json.dumps({"usage": {"input_tokens": 5, "output_tokens": 7,
                                    "cache_read_input_tokens": 2, "total_tokens": 14}})
        u = a._extract_usage(raw, "")
        assert u["input_tokens"] == 5 and u["cache_read_tokens"] == 2

    @pytest.mark.parametrize("raw", WEIRD_STRINGS)
    def test_extract_usage_never_crashes(self, raw):
        u = ClaudeAgent("p")._extract_usage(raw, "")
        assert u["token_source"] in ("cli", "unavailable")

    def test_stdin_bytes_huge(self):
        a = ClaudeAgent("z" * 150_000)
        assert len(a._stdin_bytes()) >= 150_000


# --------------------------------------------------------------------------- #
# GrokAgent
# --------------------------------------------------------------------------- #
class TestGrokBuildCmd:
    def test_build_cmd_basic(self):
        a = GrokAgent("p", model="grok-build")
        cmd = a._build_cmd("/tmp/x")
        assert cmd[0] == "grok"
        assert "--prompt-file" in cmd and "/tmp/x" in cmd
        assert "-m" in cmd and "grok-build" in cmd

    def test_build_cmd_standard_model_omits_m(self):
        a = GrokAgent("p", model="standard")
        assert "-m" not in a._build_cmd("/tmp/x")

    def test_build_cmd_disable_web_search(self):
        a = GrokAgent("p", web_search=False)
        assert "--disable-web-search" in a._build_cmd("/tmp/x")

    def test_effort_not_sent_for_grok_build(self):
        a = GrokAgent("p", model="grok-build")
        a.effort = "high"
        assert "--effort" not in a._build_cmd("/tmp/x")


class TestGrokParse:
    def test_extract_text_from_json(self):
        raw = json.dumps({"text": "the reply", "sessionId": "s1"})
        assert GrokAgent._extract_text(raw) == "the reply"

    def test_extract_text_with_log_prefix(self):
        raw = "INFO some log\n" + json.dumps({"text": "reply"})
        assert GrokAgent._extract_text(raw) == "reply"

    def test_extract_text_plain_passthrough(self):
        assert GrokAgent._extract_text("just plain text") == "just plain text"

    @pytest.mark.parametrize("raw", WEIRD_STRINGS)
    def test_extract_text_never_crashes(self, raw):
        assert isinstance(GrokAgent._extract_text(raw), str)

    def test_extract_session_id(self):
        raw = json.dumps({"sessionId": "sess-42", "text": "x"})
        assert GrokAgent._extract_session_id(raw) == "sess-42"

    @pytest.mark.parametrize("raw", WEIRD_STRINGS)
    def test_extract_session_id_never_crashes(self, raw):
        out = GrokAgent._extract_session_id(raw)
        assert out is None or isinstance(out, str)

    @pytest.mark.parametrize("raw", WEIRD_STRINGS)
    def test_extract_usage_never_crashes(self, raw):
        u = GrokAgent("p")._extract_usage(raw, "")
        assert u["token_source"] in ("cli", "unavailable")

    def test_extract_usage_non_dict_usage(self):
        raw = json.dumps({"usage": [1, 2], "text": "x"})
        u = GrokAgent("p")._extract_usage(raw, "")
        assert u["token_source"] == "unavailable"


class TestGrokModelParse:
    def test_parse_models_basic(self):
        out = _parse_grok_models("Available models:\n  * grok-build\n  * grok-4\nLogged in as foo\n")
        assert "grok-build" in out and "grok-4" in out
        assert all("Logged" not in m for m in out)

    def test_parse_models_empty(self):
        assert _parse_grok_models("") == []

    def test_parse_models_model_substring_kept(self):
        # a model name containing "model" must still be collected
        out = _parse_grok_models("Available models:\n  * grok-model-x\n")
        assert "grok-model-x" in out


# --------------------------------------------------------------------------- #
# AgyAgent
# --------------------------------------------------------------------------- #
class TestAgyResolveModel:
    @pytest.mark.parametrize("model,effort,expected", [
        (None, None, None),
        ("standard", "high", None),
        ("pro", "low", "Gemini 3.1 Pro (Low)"),
        ("pro", None, "Gemini 3.1 Pro (High)"),
        ("flash", "medium", "Gemini 3.5 Flash (Medium)"),
        ("flash", "bogus-effort", "Gemini 3.5 Flash (Medium)"),  # default
        ("opus", None, "Claude Opus 4.6 (Thinking)"),
        ("Gemini 3.1 Pro (High)", None, "Gemini 3.1 Pro (High)"),  # passthrough
    ])
    def test_resolve(self, model, effort, expected):
        assert resolve_agy_model(model, effort) == expected

    def test_resolve_whitespace_model(self):
        assert resolve_agy_model("  pro  ", "high") == "Gemini 3.1 Pro (High)"

    def test_resolve_empty_string(self):
        assert resolve_agy_model("", "high") is None


class TestAgyBuildCommand:
    def test_build_command_injects_constraints(self):
        a = AgyAgent("do the thing")
        cmd = a.build_command()
        assert cmd[0] == "agy" and "--print" in cmd
        injected = cmd[cmd.index("--print") + 1]
        assert "NO SUDO" in injected and "do the thing" in injected

    def test_build_command_with_piped_input(self):
        a = AgyAgent("task")
        cmd = a.build_command("prior output")
        injected = cmd[cmd.index("--print") + 1]
        assert "prior output" in injected

    def test_build_command_input_output_files(self):
        a = AgyAgent("task", input_files=["a.py"], output_files=["b.py"])
        injected = a.build_command()[a.build_command().index("--print") + 1]
        assert "a.py" in injected and "b.py" in injected

    def test_postprocess_passthrough(self):
        a = AgyAgent("p")
        assert a._postprocess("raw\noutput") == "raw\noutput"

    def test_write_settings_model_tolerates_garbage_file(self, tmp_path, monkeypatch):
        # _read_settings_model must never raise on a corrupt settings file
        import agy_orchestrator.core.agents.agy_agent as mod
        bad = tmp_path / "settings.json"
        bad.write_text("{ this is not json")
        monkeypatch.setattr(mod, "_SETTINGS_PATH", bad)
        assert AgyAgent._read_settings_model() is None
        # writing into a dir that doesn't exist yet works (mkdir parents)
        sub = tmp_path / "deep" / "settings.json"
        monkeypatch.setattr(mod, "_SETTINGS_PATH", sub)
        AgyAgent._write_settings_model("Gemini 3.1 Pro (High)")
        assert json.loads(sub.read_text())["model"] == "Gemini 3.1 Pro (High)"


# --------------------------------------------------------------------------- #
# MockAgent
# --------------------------------------------------------------------------- #
class TestMockAgent:
    def test_build_command_real_subprocess_shape(self):
        a = MockAgent("p")
        cmd = a.build_command()
        assert cmd[0] == "sh" and cmd[1] == "-c"
        assert "AGY_MOCK_OUT" in a.extra_env

    def test_mock_output_planner_json(self):
        a = MockAgent("Return a valid JSON list of strings please")
        out = a._mock_output()
        assert json.loads(out) == ["Implement the core module.", "Add a focused test for it."]

    def test_mock_output_default_ends_approved(self):
        a = MockAgent("normal task")
        assert a._mock_output().strip().endswith("APPROVED")

    def test_extract_usage_none_prompt_safe(self):
        a = MockAgent.__new__(MockAgent)
        a.prompt = None
        u = a._extract_usage("out", "")
        assert u["input_tokens"] >= 0 and u["token_source"] == "cli"

    def test_mock_class_for_stable(self):
        c1 = mock_agent_class_for("codex")
        c2 = mock_agent_class_for("codex")
        assert c1 is c2 and c1._MOCK_WORKER == "codex"

    def test_mock_sleep_seconds_nonnegative(self, monkeypatch):
        monkeypatch.setenv("AGY_BENCH_MOCK_SLEEP", "not-a-number")
        a = MockAgent("p")
        # bad env must fall back to default, never crash or go negative
        assert a._mock_sleep_seconds() >= 0.0

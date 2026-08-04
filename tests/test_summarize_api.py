"""Tests for the Claude API call path, against a stub SDK.

Covers request construction, chunk-then-synthesize flow, and - most importantly -
that every failure mode surfaces as a SummarizeError instead of a half-written
summary.md. Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scribe.config import DEFAULTS, Config, _deep_merge  # noqa: E402
from scribe.summarize import SummarizeError, summarize_session  # noqa: E402


class FakeUsage:
    def __init__(self, input_tokens=1200, output_tokens=350):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeMessage:
    def __init__(self, text="## TL;DR\nIt went fine.", stop_reason="end_turn"):
        self.content = [FakeTextBlock(text)]
        self.stop_reason = stop_reason
        self.stop_details = None
        self.usage = FakeUsage()
        self._request_id = "req_test"


def build_fake_anthropic():
    """A stand-in for the `anthropic` package, with the exception hierarchy."""
    mod = types.ModuleType("anthropic")

    class APIError(Exception):
        pass

    class APIStatusError(APIError):
        def __init__(self, message, status_code=500):
            super().__init__(message)
            self.status_code = status_code
            self.type = "api_error"
            self.request_id = "req_err"

    class AuthenticationError(APIStatusError):
        pass

    class NotFoundError(APIStatusError):
        pass

    class BadRequestError(APIStatusError):
        pass

    class RateLimitError(APIStatusError):
        def __init__(self, message):
            super().__init__(message, 429)
            self.response = types.SimpleNamespace(headers={"retry-after": "30"})

    class APIConnectionError(APIError):
        pass

    class _Stream:
        def __init__(self, client, kwargs):
            self._client = client
            self._kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            self._client.calls.append(self._kwargs)
            if self._client.raises is not None:
                raise self._client.raises
            responses = self._client.responses
            index = min(len(self._client.calls) - 1, len(responses) - 1)
            return responses[index]

    class _Messages:
        def __init__(self, client):
            self._client = client

        def stream(self, **kwargs):
            return _Stream(self._client, kwargs)

    class Anthropic:
        instances: list = []

        def __init__(self, *args, **kwargs):
            self.calls: list[dict] = []
            self.responses = [FakeMessage()]
            self.raises = None
            self.messages = _Messages(self)
            Anthropic.instances.append(self)

    mod.APIError = APIError
    mod.APIStatusError = APIStatusError
    mod.AuthenticationError = AuthenticationError
    mod.NotFoundError = NotFoundError
    mod.BadRequestError = BadRequestError
    mod.RateLimitError = RateLimitError
    mod.APIConnectionError = APIConnectionError
    mod.Anthropic = Anthropic
    return mod


TRANSCRIPT = (
    "# Budget sync\n\n## Transcript\n\n"
    "**[00:00:00] Me:** Can you hear me?\n\n"
    "**[00:00:05] Others:** Loud and clear.\n"
)


class SummarizeTestCase(unittest.TestCase):
    def setUp(self):
        self.fake = build_fake_anthropic()
        self._saved = sys.modules.get("anthropic")
        sys.modules["anthropic"] = self.fake
        self.fake.Anthropic.instances = []

        self._tmp = tempfile.TemporaryDirectory()
        self.session = Path(self._tmp.name) / "2026-08-03_1400_budget-sync"
        self.session.mkdir(parents=True)
        (self.session / "transcript.md").write_text(TRANSCRIPT, encoding="utf-8")
        (self.session / "meta.json").write_text(
            json.dumps({"title": "Budget sync", "duration_seconds": 930}), encoding="utf-8"
        )

    def tearDown(self):
        if self._saved is not None:
            sys.modules["anthropic"] = self._saved
        else:
            sys.modules.pop("anthropic", None)
        self._tmp.cleanup()

    def cfg(self, **overrides):
        return Config(_deep_merge(copy.deepcopy(DEFAULTS), {"summarize": overrides}))

    @property
    def client(self):
        return self.fake.Anthropic.instances[-1]


class TestRequestConstruction(SummarizeTestCase):
    def test_single_call_for_a_short_transcript(self):
        result = summarize_session(self.session, self.cfg())

        self.assertEqual(result.chunk_count, 1)
        self.assertEqual(len(self.client.calls), 1)
        call = self.client.calls[0]
        self.assertEqual(call["model"], "claude-sonnet-4-6")
        self.assertEqual(call["max_tokens"], 8000)
        self.assertEqual(call["thinking"], {"type": "adaptive"})
        self.assertNotIn("output_config", call)  # effort defaults to null
        self.assertIn("Can you hear me?", call["messages"][0]["content"])
        self.assertIn("## Action Items", call["system"])

    def test_thinking_and_effort_follow_config(self):
        summarize_session(self.session, self.cfg(thinking="off", effort="high"))
        call = self.client.calls[0]
        self.assertEqual(call["thinking"], {"type": "disabled"})
        self.assertEqual(call["output_config"], {"effort": "high"})

    def test_template_and_model_overrides_win(self):
        summarize_session(
            self.session, self.cfg(), template="sales-call", model="claude-opus-4-6"
        )
        call = self.client.calls[0]
        self.assertEqual(call["model"], "claude-opus-4-6")
        self.assertIn("sales call", call["system"])

    def test_template_from_meta_json_is_used_when_not_overridden(self):
        (self.session / "meta.json").write_text(
            json.dumps({"title": "Chem 201", "template": "class-lecture"}), encoding="utf-8"
        )
        result = summarize_session(self.session, self.cfg())
        self.assertEqual(result.template, "class-lecture")
        self.assertIn("class or lecture", self.client.calls[0]["system"])

    def test_unknown_template_is_rejected_before_any_api_call(self):
        with self.assertRaises(SummarizeError):
            summarize_session(self.session, self.cfg(), template="stand-up")
        self.assertEqual(self.fake.Anthropic.instances, [])


class TestOutput(SummarizeTestCase):
    def test_summary_file_has_header_and_body(self):
        result = summarize_session(self.session, self.cfg())
        text = result.path.read_text(encoding="utf-8")

        self.assertEqual(result.path, self.session / "summary.md")
        self.assertIn("# Summary — Budget sync", text)
        self.assertIn("Template: General", text)
        self.assertIn("Model: claude-sonnet-4-6", text)
        self.assertIn("Duration: 15 min", text)
        self.assertIn("It went fine.", text)

    def test_usage_is_accumulated(self):
        result = summarize_session(self.session, self.cfg())
        self.assertEqual(result.usage.calls, 1)
        self.assertEqual(result.usage.input_tokens, 1200)
        self.assertEqual(result.usage.output_tokens, 350)


class TestChunking(SummarizeTestCase):
    def test_long_transcript_is_chunked_then_synthesized(self):
        paragraphs = [f"**[00:{i:02d}:00] Me:** {'word ' * 60}".strip() for i in range(20)]
        (self.session / "transcript.md").write_text("\n\n".join(paragraphs), encoding="utf-8")

        result = summarize_session(self.session, self.cfg(max_chunk_chars=1000))

        self.assertGreater(result.chunk_count, 1)
        # one call per chunk, plus the synthesis pass
        self.assertEqual(len(self.client.calls), result.chunk_count + 1)
        self.assertEqual(result.usage.calls, result.chunk_count + 1)

        chunk_call, final_call = self.client.calls[0], self.client.calls[-1]
        self.assertIn("part 1 of", chunk_call["system"])
        self.assertIn("digests of consecutive parts", final_call["system"])
        self.assertIn("## Notable Moments", final_call["system"])
        self.assertTrue(result.path.exists())


class TestErrorsAreSurfaced(SummarizeTestCase):
    def _expect_error(self, exc, *fragments):
        summarize_session_cfg = self.cfg()
        # Prime the client, then make the call fail.
        client_holder = {}

        original = self.fake.Anthropic

        class Failing(original):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.raises = exc
                client_holder["client"] = self

        self.fake.Anthropic = Failing
        try:
            with self.assertRaises(SummarizeError) as ctx:
                summarize_session(self.session, summarize_session_cfg)
        finally:
            self.fake.Anthropic = original
        message = str(ctx.exception)
        for fragment in fragments:
            self.assertIn(fragment, message)
        self.assertFalse((self.session / "summary.md").exists())

    def test_auth_error(self):
        self._expect_error(self.fake.AuthenticationError("bad key", 401), "ANTHROPIC_API_KEY")

    def test_model_not_found(self):
        self._expect_error(self.fake.NotFoundError("no such model", 404), "summarize.model")

    def test_bad_request_hints_at_thinking_and_effort(self):
        self._expect_error(self.fake.BadRequestError("thinking not supported", 400), "thinking")

    def test_rate_limit_reports_retry_after(self):
        self._expect_error(self.fake.RateLimitError("slow down"), "429", "30")

    def test_connection_error(self):
        self._expect_error(self.fake.APIConnectionError("no route to host"), "could not reach")

    def test_missing_credentials_gives_actionable_guidance(self):
        # With no key at all the SDK raises TypeError while building the request,
        # not an APIError - it must still come back as a clean SummarizeError.
        self._expect_error(
            TypeError(
                "Could not resolve authentication method. Expected one of api_key, "
                "auth_token, or credentials to be set."
            ),
            "ANTHROPIC_API_KEY",
        )

    def test_server_error_includes_status(self):
        self._expect_error(self.fake.APIStatusError("overloaded", 529), "529")

    def test_truncated_summary_is_an_error_not_a_silent_partial(self):
        summarize_session_cfg = self.cfg()
        original = self.fake.Anthropic

        class Truncating(original):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.responses = [FakeMessage("## TL;DR\nIt was going we", "max_tokens")]

        self.fake.Anthropic = Truncating
        try:
            with self.assertRaises(SummarizeError) as ctx:
                summarize_session(self.session, summarize_session_cfg)
        finally:
            self.fake.Anthropic = original
        self.assertIn("truncated", str(ctx.exception))
        self.assertIn("max_output_tokens", str(ctx.exception))
        self.assertFalse((self.session / "summary.md").exists())

    def test_refusal_is_reported(self):
        original = self.fake.Anthropic

        class Refusing(original):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.responses = [FakeMessage("", "refusal")]

        self.fake.Anthropic = Refusing
        try:
            with self.assertRaises(SummarizeError) as ctx:
                summarize_session(self.session, self.cfg())
        finally:
            self.fake.Anthropic = original
        self.assertIn("declined", str(ctx.exception))

    def test_missing_transcript_is_reported_before_any_api_call(self):
        (self.session / "transcript.md").unlink()
        with self.assertRaises(SummarizeError) as ctx:
            summarize_session(self.session, self.cfg())
        self.assertIn("no transcript", str(ctx.exception))
        self.assertEqual(self.fake.Anthropic.instances, [])

    def test_empty_transcript_is_reported(self):
        (self.session / "transcript.md").write_text("   \n", encoding="utf-8")
        with self.assertRaises(SummarizeError) as ctx:
            summarize_session(self.session, self.cfg())
        self.assertIn("empty", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

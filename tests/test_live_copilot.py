"""Tests for scribe/live/copilot.py - the live suggestion engine.

Runs against a stubbed Anthropic client, so no credentials, no network and no
spend. Covers the transcript buffer, the char caps, the debounce/
cancel-and-replace rule, the new-speech gate, error surfacing, and the
privacy posture of suggestions.jsonl.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scribe.live.context import ReferenceContext  # noqa: E402
from scribe.live.copilot import (  # noqa: E402
    NOTHING,
    CopilotSettings,
    SuggestionEngine,
    TranscriptBuffer,
    build_prompt,
    cap_text,
)
from scribe.live.stream import LiveChunk  # noqa: E402


def chunk(track="system", speaker="Others", start=0.0, end=8.0, text="hello there"):
    return LiveChunk(track=track, speaker=speaker, window_start=start,
                     window_end=end, text=text)


# --------------------------------------------------------------------------
# Stub SDK
# --------------------------------------------------------------------------


class FakeStream:
    def __init__(self, pieces, on_iter=None, error=None):
        self._pieces = pieces
        self._on_iter = on_iter
        self._error = error
        self.closed = False

    def __enter__(self):
        if self._error:
            raise self._error
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    @property
    def text_stream(self):
        for piece in self._pieces:
            if self._on_iter:
                self._on_iter()
            yield piece


class FakeClient:
    """Records the requests it was given and replays scripted output."""

    def __init__(self, pieces=("Ask about ", "rush pricing."), on_iter=None, error=None):
        self.requests = []
        self._pieces = pieces
        self._on_iter = on_iter
        self._error = error
        outer = self

        class _Messages:
            def stream(self, **kwargs):
                outer.requests.append(kwargs)
                return FakeStream(outer._pieces, outer._on_iter, outer._error)

            def create(self, **kwargs):
                outer.requests.append(kwargs)
                block = type("B", (), {"type": "text", "text": "CONDENSED"})()
                return type("M", (), {"content": [block]})()

        self.messages = _Messages()


# --------------------------------------------------------------------------
# Pure pieces
# --------------------------------------------------------------------------


class TestTranscriptBuffer(unittest.TestCase):
    def test_renders_speaker_prefixed_lines(self):
        buf = TranscriptBuffer()
        buf.add(chunk(track="mic", speaker="Me", end=4.0, text="what's your turnaround"))
        buf.add(chunk(track="system", speaker="Others", end=6.0, text="about two weeks"))

        rendered = buf.recent(60, 4000)
        self.assertEqual(rendered,
                         "Me: what's your turnaround\nOthers: about two weeks")

    def test_consecutive_duplicate_from_one_track_is_dropped(self):
        # Windows overlap by design, so the same sentence can arrive twice.
        buf = TranscriptBuffer()
        self.assertTrue(buf.add(chunk(text="same words")))
        self.assertFalse(buf.add(chunk(text="same words")))
        self.assertEqual(buf.recent(60, 4000), "Others: same words")

    def test_the_same_words_from_the_other_track_are_kept(self):
        buf = TranscriptBuffer()
        buf.add(chunk(track="mic", speaker="Me", text="yes exactly"))
        self.assertTrue(buf.add(chunk(track="system", speaker="Others", text="yes exactly")))

    def test_empty_text_is_ignored(self):
        buf = TranscriptBuffer()
        self.assertFalse(buf.add(chunk(text="   ")))
        self.assertEqual(buf.recent(60, 4000), "")

    def test_only_the_recent_window_is_returned(self):
        buf = TranscriptBuffer()
        buf.add(chunk(end=10.0, text="ancient history"))
        buf.add(chunk(track="mic", speaker="Me", end=100.0, text="right now"))

        rendered = buf.recent(30, 4000)
        self.assertIn("right now", rendered)
        self.assertNotIn("ancient history", rendered)

    def test_a_window_shorter_than_one_chunk_still_returns_something(self):
        buf = TranscriptBuffer()
        buf.add(chunk(end=50.0, text="the only thing said"))
        self.assertIn("the only thing said", buf.recent(0.1, 4000))

    def test_char_cap_keeps_the_freshest_end(self):
        buf = TranscriptBuffer()
        for i in range(50):
            buf.add(chunk(track="mic" if i % 2 else "system",
                          speaker="Me" if i % 2 else "Others",
                          end=float(i), text=f"line number {i}"))
        rendered = buf.recent(1000, 120)

        self.assertLessEqual(len(rendered), 120)
        self.assertIn("line number 49", rendered)
        self.assertNotIn("line number 0 ", rendered)

    def test_new_chars_tracks_speech_since_the_last_mark(self):
        buf = TranscriptBuffer()
        buf.add(chunk(text="abcd"))
        self.assertEqual(buf.new_chars(), 4)
        buf.mark()
        self.assertEqual(buf.new_chars(), 0)
        buf.add(chunk(text="efghij"))
        self.assertEqual(buf.new_chars(), 6)


class TestCaps(unittest.TestCase):
    def test_short_text_is_untouched(self):
        text, truncated = cap_text("short and sweet", 500)
        self.assertEqual(text, "short and sweet")
        self.assertFalse(truncated)

    def test_long_text_is_cut_at_a_word_boundary_with_an_ellipsis(self):
        text, truncated = cap_text("word " * 200, 50)
        self.assertTrue(truncated)
        self.assertLessEqual(len(text), 53)
        self.assertTrue(text.endswith("..."))
        self.assertNotIn("wor...", text)  # not mid-word

    def test_prompt_includes_transcript_and_context(self):
        prompt = build_prompt("<reference_material>PRICES</reference_material>",
                              "Others: how much for 50 hoodies")
        self.assertIn("PRICES", prompt)
        self.assertIn("how much for 50 hoodies", prompt)

    def test_prompt_without_context_still_works(self):
        prompt = build_prompt("", "Others: hello")
        self.assertIn("hello", prompt)
        self.assertNotIn("reference_material", prompt)


class TestSettings(unittest.TestCase):
    def test_defaults_are_fast_by_design(self):
        settings = CopilotSettings()
        self.assertIn("haiku", settings.model.lower())
        self.assertLessEqual(settings.interval_seconds, 10)

    def test_reads_the_live_section(self):
        from scribe.config import DEFAULTS, Config, _deep_merge

        cfg = Config(_deep_merge(DEFAULTS, {"live": {
            "suggestion_model": "claude-sonnet-4-6",
            "suggest_interval_seconds": 3,
            "max_suggestion_chars": 120,
            "suggestions_enabled": False,
        }}))
        settings = CopilotSettings.from_config(cfg)

        self.assertEqual(settings.model, "claude-sonnet-4-6")
        self.assertEqual(settings.interval_seconds, 3)
        self.assertEqual(settings.max_suggestion_chars, 120)
        self.assertFalse(settings.enabled)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


class EngineTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.session = Path(self._tmp.name)
        self.deltas: list[str] = []
        self.done: list = []
        self.errors: list[str] = []

    def tearDown(self):
        self._tmp.cleanup()

    def engine(self, client, **overrides):
        settings = CopilotSettings(**{"interval_seconds": 999, **overrides})
        return SuggestionEngine(
            self.session, settings, ReferenceContext(text="<reference_material>P</reference_material>"),
            client=client,
            on_delta=self.deltas.append,
            on_done=self.done.append,
            on_error=self.errors.append,
        )

    def log_records(self):
        path = self.session / "suggestions.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestOneSuggestion(EngineTestCase):
    def test_produces_streams_and_logs_a_suggestion(self):
        client = FakeClient(pieces=("Ask about ", "rush pricing."))
        eng = self.engine(client)
        eng.buffer.add(chunk(text="what about a rush order"))

        eng._one("timer", eng.buffer.recent(60, 4000))

        self.assertEqual(self.deltas, ["Ask about ", "rush pricing."])
        suggestion = self.done[0]
        self.assertEqual(suggestion.text, "Ask about rush pricing.")
        self.assertEqual(suggestion.trigger, "timer")
        self.assertIsNotNone(suggestion.first_token_ms)
        self.assertIsNotNone(suggestion.total_ms)

        record = self.log_records()[0]
        self.assertEqual(record["trigger"], "timer")
        self.assertEqual(record["suggestion"], "Ask about rush pricing.")

    def test_request_uses_the_configured_model_and_caps_output(self):
        client = FakeClient()
        eng = self.engine(client, model="claude-haiku-4-5", max_output_tokens=123)
        eng.buffer.add(chunk(text="hello"))
        eng._one("timer", "Others: hello")

        request = client.requests[0]
        self.assertEqual(request["model"], "claude-haiku-4-5")
        self.assertEqual(request["max_tokens"], 123)
        # Speed matters more than depth here: no thinking, no effort knobs.
        self.assertNotIn("thinking", request)
        self.assertNotIn("output_config", request)

    def test_nothing_useful_is_suppressed_rather_than_shown(self):
        eng = self.engine(FakeClient(pieces=(NOTHING,)))
        eng._one("timer", "Others: uh")

        self.assertEqual(self.done[0].text, "")
        self.assertEqual(self.log_records()[0]["suggestion"], "")

    def test_overlong_output_is_hard_capped(self):
        eng = self.engine(FakeClient(pieces=("x" * 200,) * 20), max_suggestion_chars=100)
        eng._one("timer", "Others: talk")

        suggestion = self.done[0]
        self.assertLessEqual(len(suggestion.text), 104)
        self.assertTrue(suggestion.truncated)

    def test_api_error_is_surfaced_and_logged_not_swallowed(self):
        import anthropic

        error = anthropic.APIConnectionError(request=None)
        eng = self.engine(FakeClient(error=error))
        eng._one("timer", "Others: hello")

        suggestion = self.done[0]
        self.assertIsNotNone(suggestion.error)
        self.assertIn("could not reach", suggestion.error)
        self.assertEqual(self.errors, [suggestion.error])
        self.assertIsNotNone(self.log_records()[0]["error"])
        self.assertEqual(eng.errors, 1)

    def test_missing_credentials_gives_actionable_guidance(self):
        error = TypeError("Could not resolve authentication method. Expected one of "
                          "api_key, auth_token, or credentials to be set.")
        eng = self.engine(FakeClient(error=error))
        eng._one("timer", "Others: hello")

        self.assertIn("ANTHROPIC_API_KEY", self.done[0].error)


class TestRepeatedFailureBackoff(EngineTestCase):
    def _failing_engine(self):
        error = TypeError("Could not resolve authentication method.")
        return self.engine(FakeClient(error=error), interval_seconds=0.01, min_new_chars=1)

    def test_auto_timer_stands_down_after_repeated_failures(self):
        eng = self._failing_engine()
        for _ in range(eng.AUTO_DISABLE_AFTER_ERRORS):
            eng._one("timer", "Others: hi")

        self.assertTrue(eng.auto_disabled)
        eng._last_sent = 0.0
        eng.buffer.add(chunk(text="plenty of fresh speech to trigger on"))
        self.assertFalse(eng._due(), "kept retrying a call that fails every time")

    def test_every_failure_is_still_surfaced_before_standing_down(self):
        eng = self._failing_engine()
        for _ in range(eng.AUTO_DISABLE_AFTER_ERRORS):
            eng._one("timer", "Others: hi")

        # Each failure reported, plus the one-off "pausing" notice - nothing
        # is swallowed, it just stops repeating forever.
        self.assertEqual(len(self.errors), eng.AUTO_DISABLE_AFTER_ERRORS + 1)
        self.assertIn("pausing automatic suggestions", self.errors[-1])

    def test_a_success_resets_the_failure_streak(self):
        eng = self.engine(FakeClient(), interval_seconds=0.01)
        eng._consecutive_errors = 2
        eng._one("timer", "Others: hi")

        self.assertEqual(eng._consecutive_errors, 0)
        self.assertFalse(eng.auto_disabled)


class TestPrivacy(EngineTestCase):
    def test_suggestions_log_never_contains_the_transcript(self):
        spoken = "our budget is exactly forty two thousand dollars"
        eng = self.engine(FakeClient())
        eng.buffer.add(chunk(text=spoken))
        eng._one("timer", eng.buffer.recent(60, 4000))

        raw = (self.session / "suggestions.jsonl").read_text(encoding="utf-8")
        self.assertNotIn(spoken, raw, "spoken content leaked into suggestions.jsonl")
        # Counts only - enough to reason about prompt size and cost.
        self.assertGreater(self.log_records()[0]["transcript_chars"], 0)


class TestDebounce(EngineTestCase):
    def test_a_newer_trigger_supersedes_the_in_flight_call(self):
        started = threading.Event()
        release = threading.Event()

        def block_mid_stream():
            started.set()
            release.wait(5)

        client = FakeClient(pieces=("part one ", "part two"), on_iter=block_mid_stream)
        eng = self.engine(client)
        eng.buffer.add(chunk(text="something worth answering"))

        worker = threading.Thread(target=eng._one, args=("timer", "Others: hi"))
        worker.start()
        self.assertTrue(started.wait(5), "stream never started")

        eng.trigger("manual")  # arrives mid-flight
        release.set()
        worker.join(10)

        self.assertTrue(self.done[0].superseded)
        self.assertEqual(eng.superseded, 1)

    def test_trigger_marks_a_pending_reason_for_the_worker(self):
        eng = self.engine(FakeClient())
        eng.trigger("manual")
        self.assertEqual(eng._pending, "manual")

    def test_shutdown_mid_stream_reads_as_cancelled_not_superseded(self):
        started = threading.Event()
        release = threading.Event()

        def block_mid_stream():
            started.set()
            release.wait(5)

        eng = self.engine(FakeClient(pieces=("a ", "b"), on_iter=block_mid_stream))
        worker = threading.Thread(target=eng._one, args=("timer", "Others: hi"))
        worker.start()
        self.assertTrue(started.wait(5))

        eng._stop.set()  # what stop() does
        release.set()
        worker.join(10)

        self.assertTrue(self.done[0].cancelled)
        self.assertFalse(self.done[0].superseded)
        self.assertEqual(eng.superseded, 0, "shutdown counted as a supersede")


class TestClientReuse(EngineTestCase):
    def test_the_client_is_built_once_and_reused(self):
        # A fresh client per suggestion would re-resolve credentials and drop
        # the HTTP connection pool - a new TLS handshake on the latency path.
        import scribe.live.copilot as copilot

        built = []

        def counting_build():
            built.append(1)
            return FakeClient()

        original = copilot.build_client
        copilot.build_client = counting_build
        try:
            settings = CopilotSettings(interval_seconds=999)
            eng = SuggestionEngine(self.session, settings, None,
                                   on_done=self.done.append)
            eng._one("timer", "Others: one")
            eng._one("timer", "Others: two")
            eng._one("timer", "Others: three")
        finally:
            copilot.build_client = original

        self.assertEqual(len(built), 1, f"built the client {len(built)} times")
        self.assertEqual(len(self.done), 3)


class TestAutoTimerGate(EngineTestCase):
    def test_not_due_without_enough_new_speech(self):
        eng = self.engine(FakeClient(), interval_seconds=0, min_new_chars=40)
        eng._last_sent = 0.0
        eng.buffer.add(chunk(text="hi"))  # 2 chars
        self.assertFalse(eng._due(), "fired an API call on two characters of speech")

    def test_due_once_enough_new_speech_has_arrived(self):
        eng = self.engine(FakeClient(), interval_seconds=0.01, min_new_chars=10)
        eng._last_sent = 0.0  # monotonic() is far past this, so the interval has elapsed
        eng.buffer.add(chunk(text="a much longer sentence than the gate"))
        self.assertTrue(eng._due())

    def test_interval_of_zero_or_less_disables_the_auto_timer(self):
        # Not "as fast as possible" - that would be a billing accident.
        for interval in (0.0, -1.0):
            eng = self.engine(FakeClient(), interval_seconds=interval, min_new_chars=1)
            eng._last_sent = 0.0
            eng.buffer.add(chunk(text="lots of new speech here"))
            self.assertFalse(eng._due(), f"interval {interval} should disable the timer")

    def test_silence_after_a_suggestion_does_not_retrigger(self):
        eng = self.engine(FakeClient(), interval_seconds=0.01, min_new_chars=5)
        eng._last_sent = 0.0
        eng.buffer.add(chunk(text="enough new speech"))
        self.assertTrue(eng._due())
        eng.buffer.mark()  # what the worker does when it sends
        self.assertFalse(eng._due(), "would have billed a call for silence")


class TestLifecycle(EngineTestCase):
    def test_disabled_engine_never_starts_a_thread_or_calls_the_api(self):
        client = FakeClient()
        eng = self.engine(client, enabled=False)
        eng.start()
        eng.trigger("manual")
        time.sleep(0.3)
        stats = eng.stop()

        self.assertEqual(client.requests, [])
        self.assertEqual(stats["calls"], 0)

    def test_manual_trigger_runs_end_to_end_through_the_worker(self):
        client = FakeClient(pieces=("Mention the ", "rush fee."))
        eng = self.engine(client)
        eng.buffer.add(chunk(text="can you do it faster"))
        eng.start()
        eng.trigger("manual")

        deadline = time.time() + 5
        while not self.done and time.time() < deadline:
            time.sleep(0.02)
        stats = eng.stop()

        self.assertEqual(self.done[0].text, "Mention the rush fee.")
        self.assertEqual(self.done[0].trigger, "manual")
        self.assertEqual(stats["delivered"], 1)
        self.assertIsNotNone(stats["median_first_token_ms"])

    def test_stats_report_model_and_counts(self):
        eng = self.engine(FakeClient())
        eng.buffer.add(chunk(text="hello"))
        eng._one("timer", "Others: hello")
        stats = eng.stop()

        self.assertEqual(stats["calls"], 1)
        self.assertEqual(stats["errors"], 0)
        self.assertEqual(stats["model"], eng.settings.model)


if __name__ == "__main__":
    unittest.main()

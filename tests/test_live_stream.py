"""Tests for the pure logic in scribe/live/stream.py: window slicing/overlap,
the widen/narrow backpressure decision, and the non-invasive queue tap.

Deliberately excludes _TrackTap/LiveTranscriber, which call real whisper and
real threads - the same split this repo already uses for
scribe.transcribe.merge (pure, tested here-style) vs. scribe.transcribe.engine
(whisper-calling I/O, untested by this suite). Proving the live transcript
keeps up on real hardware is the explicit real-call test this phase is gated
on, not something a fast unit test can stand in for.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import json
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scribe.live.stream import (  # noqa: E402
    BackpressureTracker,
    LiveChunk,
    LiveSettings,
    LiveTranscriber,
    WindowAccumulator,
    _TrackTap,
    tee_track_buffers,
)


def pcm(seconds: float, rate: int, value: int = 100) -> bytes:
    """A block of constant-value int16 mono PCM, easy to assert on by count
    and by value rather than needing real speech-shaped audio."""
    n = int(round(seconds * rate))
    return np.full(n, value, dtype=np.int16).tobytes()


class TestWindowAccumulatorBoundaries(unittest.TestCase):
    def test_construction_rejects_overlap_not_less_than_window(self):
        with self.assertRaises(ValueError):
            WindowAccumulator(16000, window_seconds=8.0, overlap_seconds=8.0)
        with self.assertRaises(ValueError):
            WindowAccumulator(16000, window_seconds=8.0, overlap_seconds=9.0)

    def test_no_window_until_enough_audio(self):
        acc = WindowAccumulator(16000, window_seconds=1.0, overlap_seconds=0.2)
        acc.feed(pcm(0.5, 16000))
        self.assertEqual(acc.pop_ready(), [])

    def test_first_window_at_exact_boundary(self):
        acc = WindowAccumulator(16000, window_seconds=1.0, overlap_seconds=0.2)
        acc.feed(pcm(1.0, 16000))
        windows = acc.pop_ready()
        self.assertEqual(len(windows), 1)
        w = windows[0]
        self.assertAlmostEqual(w.start, 0.0)
        self.assertAlmostEqual(w.end, 1.0, places=2)
        self.assertEqual(w.audio.size, 16000)
        self.assertEqual(w.audio.dtype, np.float32)

    def test_second_window_starts_at_the_hop_not_the_window_length(self):
        # window=1.0s, overlap=0.2s -> hop=0.8s. Feed 2 full windows' worth in
        # separate feed() calls so this also proves feed() accumulates across
        # calls rather than only working on a single contiguous buffer.
        acc = WindowAccumulator(16000, window_seconds=1.0, overlap_seconds=0.2)
        acc.feed(pcm(1.0, 16000))
        first = acc.pop_ready()
        self.assertEqual(len(first), 1)
        acc.feed(pcm(0.8, 16000))  # exactly one more hop's worth
        second = acc.pop_ready()
        self.assertEqual(len(second), 1)
        self.assertAlmostEqual(second[0].start, 0.8, places=2)
        self.assertAlmostEqual(second[0].end, 1.8, places=2)

    def test_overlap_region_is_shared_between_consecutive_windows(self):
        # The last 0.2s of window 1 and the first 0.2s of window 2 must be
        # the literal same audio - that's what "overlap" means here. A
        # varying signal (not a constant block) makes this a real check: a
        # wrongly-computed hop offset would show up as a value mismatch
        # instead of trivially "matching" because everything's the same number.
        rate = 16000
        n = int(1.8 * rate)  # window (1.0s) + hop (0.8s) = exactly 2 windows
        varying = (np.arange(n, dtype=np.int32) % 3000).astype(np.int16).tobytes()

        acc = WindowAccumulator(rate, window_seconds=1.0, overlap_seconds=0.2)
        acc.feed(varying)
        windows = acc.pop_ready()
        self.assertEqual(len(windows), 2)
        w1, w2 = windows
        overlap_frames = int(0.2 * rate)
        np.testing.assert_array_equal(w1.audio[-overlap_frames:], w2.audio[:overlap_frames])

    def test_multiple_windows_ready_in_one_call_share_the_backlog_count(self):
        # Simulates a caller that fell behind: a lot of audio arrived between
        # two pop_ready() calls, so several windows become ready at once.
        acc = WindowAccumulator(16000, window_seconds=1.0, overlap_seconds=0.2)
        acc.feed(pcm(3.0, 16000))  # far more than one window+hop
        windows = acc.pop_ready()
        self.assertGreaterEqual(len(windows), 2)
        self.assertTrue(all(w.backlog == len(windows) for w in windows))

    def test_single_window_per_call_has_backlog_one(self):
        acc = WindowAccumulator(16000, window_seconds=1.0, overlap_seconds=0.2)
        acc.feed(pcm(1.0, 16000))
        windows = acc.pop_ready()
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].backlog, 1)

    def test_flush_returns_the_partial_tail(self):
        acc = WindowAccumulator(16000, window_seconds=1.0, overlap_seconds=0.2)
        acc.feed(pcm(0.3, 16000))
        tail = acc.flush()
        self.assertIsNotNone(tail)
        self.assertEqual(tail.audio.size, int(0.3 * 16000))
        self.assertAlmostEqual(tail.start, 0.0)

    def test_flush_is_none_when_nothing_was_ever_fed(self):
        acc = WindowAccumulator(16000, window_seconds=1.0, overlap_seconds=0.2)
        self.assertIsNone(acc.flush())

    def test_flush_is_none_after_a_pop_leaves_nothing_when_overlap_is_zero(self):
        # With overlap=0, a pop consumes exactly one window's worth and
        # nothing is retained - this is the one case where "just popped" and
        # "truly drained" coincide.
        acc = WindowAccumulator(16000, window_seconds=1.0, overlap_seconds=0.0)
        acc.feed(pcm(1.0, 16000))
        acc.pop_ready()
        self.assertIsNone(acc.flush())

    def test_flush_after_a_pop_surfaces_the_retained_overlap_tail(self):
        # With overlap>0, popping a window only consumes hop_frames
        # (window - overlap); the last overlap_seconds is DELIBERATELY kept
        # as the start of the next window. That tail is real unconsumed
        # audio, and flush() must still surface it at Stop rather than
        # silently dropping the caller's last fraction of a second of speech.
        acc = WindowAccumulator(16000, window_seconds=1.0, overlap_seconds=0.2)
        acc.feed(pcm(1.0, 16000))
        acc.pop_ready()
        tail = acc.flush()
        self.assertIsNotNone(tail)
        self.assertAlmostEqual(tail.audio.size / 16000, 0.2, places=2)

    def test_feed_accepts_bytes_or_ndarray_equivalently(self):
        acc_bytes = WindowAccumulator(16000, window_seconds=1.0, overlap_seconds=0.2)
        acc_array = WindowAccumulator(16000, window_seconds=1.0, overlap_seconds=0.2)
        raw = pcm(1.0, 16000, value=42)
        acc_bytes.feed(raw)
        acc_array.feed(np.frombuffer(raw, dtype=np.int16))
        np.testing.assert_array_equal(
            acc_bytes.pop_ready()[0].audio, acc_array.pop_ready()[0].audio
        )


class TestWindowAccumulatorWidenNarrow(unittest.TestCase):
    def test_widen_grows_window_and_caps_overlap_proportionally(self):
        acc = WindowAccumulator(16000, window_seconds=8.0, overlap_seconds=1.0)
        acc.widen(factor=2.0)
        self.assertAlmostEqual(acc.window_seconds, 16.0)
        self.assertLessEqual(acc.overlap_seconds, 16.0 * 0.4)

    def test_widen_respects_cap(self):
        acc = WindowAccumulator(16000, window_seconds=8.0, overlap_seconds=1.0)
        acc.widen(factor=10.0, cap=20.0)
        self.assertEqual(acc.window_seconds, 20.0)

    def test_narrow_moves_toward_floor_but_not_past_it(self):
        acc = WindowAccumulator(16000, window_seconds=8.0, overlap_seconds=1.0)
        acc.widen(factor=3.0)  # 24.0s
        acc.narrow(0.5, floor_window=8.0, floor_overlap=1.0)  # -> 12.0s, still above floor
        self.assertAlmostEqual(acc.window_seconds, 12.0)
        acc.narrow(0.5, floor_window=8.0, floor_overlap=1.0)  # -> 6.0s, clamped to floor 8.0
        self.assertAlmostEqual(acc.window_seconds, 8.0)
        self.assertAlmostEqual(acc.overlap_seconds, 1.0)

    def test_narrow_settles_exactly_on_the_configured_overlap_at_the_floor(self):
        acc = WindowAccumulator(16000, window_seconds=8.0, overlap_seconds=1.0)
        acc.widen(factor=1.2)  # small widen, overlap likely untouched by the 40% cap
        for _ in range(10):
            acc.narrow(0.9, floor_window=8.0, floor_overlap=1.0)
        self.assertAlmostEqual(acc.window_seconds, 8.0)
        self.assertAlmostEqual(acc.overlap_seconds, 1.0)


class TestBackpressureTracker(unittest.TestCase):
    def test_widens_only_after_consecutive_slow_windows(self):
        bp = BackpressureTracker(widen_after=3)
        self.assertIsNone(bp.observe(1.2))
        self.assertIsNone(bp.observe(1.1))
        self.assertEqual(bp.observe(1.05), "widen")
        self.assertTrue(bp.widened)

    def test_a_single_fast_window_resets_the_slow_streak(self):
        bp = BackpressureTracker(widen_after=3)
        self.assertIsNone(bp.observe(1.2))
        self.assertIsNone(bp.observe(1.1))
        self.assertIsNone(bp.observe(0.9))  # caught up once - resets the streak
        self.assertIsNone(bp.observe(1.2))
        self.assertIsNone(bp.observe(1.1))  # only 2 in a row again, not 3
        self.assertFalse(bp.widened)

    def test_never_narrows_without_having_widened_first(self):
        bp = BackpressureTracker(narrow_after=2, recover_below=0.6)
        for _ in range(20):
            self.assertIsNone(bp.observe(0.1))  # comfortably fast, but never was widened
        self.assertFalse(bp.widened)

    def test_narrows_after_consecutive_recovered_windows_once_widened(self):
        bp = BackpressureTracker(widen_after=1, narrow_after=2, recover_below=0.6)
        self.assertEqual(bp.observe(1.5), "widen")
        self.assertTrue(bp.widened)
        self.assertIsNone(bp.observe(0.3))
        self.assertEqual(bp.observe(0.2), "narrow")
        self.assertFalse(bp.widened)

    def test_mid_range_rtf_neither_widens_nor_narrows_and_resets_recovery(self):
        bp = BackpressureTracker(widen_after=1, narrow_after=2, recover_below=0.6)
        bp.observe(1.5)  # widen
        self.assertIsNone(bp.observe(0.3))  # 1 toward recovery
        self.assertIsNone(bp.observe(0.8))  # not <recover_below - resets the streak
        self.assertIsNone(bp.observe(0.3))  # back to 1 toward recovery, not 2
        self.assertTrue(bp.widened)  # still widened - never reached narrow_after


class TestBacklogBound(unittest.TestCase):
    """The tap's queue must not grow without bound when whisper can't keep up.

    These drive _TrackTap.feed()/_trim_backlog() WITHOUT starting the worker
    thread, which is exactly the "whisper is infinitely far behind" case: the
    queue is never drained, so only the bound itself can stop it growing.
    """

    RATE = 48000
    CHANNELS = 2

    def _tap(self, max_backlog_seconds: float):
        track = SimpleNamespace(
            label="system",
            buffers=queue.Queue(),
            device=SimpleNamespace(sample_rate=self.RATE, channels=self.CHANNELS),
        )
        settings = LiveSettings(max_backlog_seconds=max_backlog_seconds, model="tiny")
        self.warnings = []
        # model=None is safe: nothing here starts the thread that would use it.
        return _TrackTap(track, settings, None, lambda _c: None,
                         lambda t, m: self.warnings.append((t, m)))

    def _one_second(self) -> bytes:
        return b"\x01\x00" * (self.RATE * self.CHANNELS)

    def test_backlog_under_the_cap_is_kept_intact(self):
        tap = self._tap(5.0)
        for _ in range(4):
            tap.feed(self._one_second())

        self.assertEqual(tap._queue.qsize(), 4)
        self.assertEqual(tap.dropped_seconds, 0.0)

    def test_oldest_audio_is_dropped_once_over_the_cap(self):
        tap = self._tap(3.0)
        for marker in range(6):
            # Distinguishable buffers so we can prove WHICH ones survived.
            tap.feed(bytes([marker, 0]) * (self.RATE * self.CHANNELS))

        remaining = []
        while not tap._queue.empty():
            remaining.append(tap._queue.get_nowait()[0])

        # Three seconds' worth kept, and it's the NEWEST three - a live view is
        # only useful if it's current.
        self.assertEqual(remaining, [3, 4, 5])
        self.assertAlmostEqual(tap.dropped_seconds, 3.0, places=3)

    def test_memory_stays_bounded_across_a_long_backlog(self):
        tap = self._tap(10.0)
        for _ in range(300):  # five minutes of audio, never drained
            tap.feed(self._one_second())

        self.assertLessEqual(tap._queue.qsize(), 10)
        self.assertAlmostEqual(tap.dropped_seconds, 290.0, places=1)

    def test_feed_never_blocks(self):
        # The audio callback thread calls feed(); if it ever blocked, capture
        # itself would stall. Far more input than the cap must still return.
        tap = self._tap(2.0)
        for _ in range(100):
            tap.feed(self._one_second())  # would hang here if feed() blocked
        self.assertLessEqual(tap._queue.qsize(), 2)

    def test_trimming_never_swallows_the_stop_sentinel(self):
        tap = self._tap(2.0)
        tap.feed(self._one_second())
        tap._queue.put(None)  # what stop() sends
        for _ in range(5):
            tap.feed(self._one_second())

        drained = []
        while not tap._queue.empty():
            drained.append(tap._queue.get_nowait())
        self.assertIn(None, drained, "stop sentinel was dropped by backlog trimming")

    def test_drop_warning_names_the_track_and_reassures_about_the_recording(self):
        tap = self._tap(1.0)
        for _ in range(4):
            tap.feed(self._one_second())
        tap._report_drops()

        self.assertEqual(len(self.warnings), 1)
        track_label, message = self.warnings[0]
        self.assertEqual(track_label, "system")
        self.assertIn("recording itself is unaffected", message)

    def test_drop_warnings_are_rate_limited(self):
        tap = self._tap(1.0)
        for _ in range(10):
            tap.feed(self._one_second())
            tap._report_drops()
        self.assertEqual(len(self.warnings), 1, "drop warnings should not spam per chunk")


class TestLiveSettingsFromConfig(unittest.TestCase):
    def test_defaults_favour_keeping_up_over_accuracy(self):
        # Both tracks share one model and faster-whisper serializes concurrent
        # calls, so the live default has to leave room for two of them.
        self.assertEqual(LiveSettings().model, "base")

    def test_reads_the_live_section_and_falls_back_to_whisper_for_device(self):
        from scribe.config import DEFAULTS, Config, _deep_merge

        cfg = Config(_deep_merge(DEFAULTS, {
            "live": {"window_seconds": 12.0, "model": "small", "max_backlog_seconds": 45.0},
            "whisper": {"device": "cuda", "language": "en"},
        }))
        settings = LiveSettings.from_config(cfg)

        self.assertEqual(settings.window_seconds, 12.0)
        self.assertEqual(settings.model, "small")
        self.assertEqual(settings.max_backlog_seconds, 45.0)
        self.assertEqual(settings.device, "cuda")
        self.assertEqual(settings.language, "en")

    def test_config_defaults_are_self_consistent(self):
        from scribe.config import DEFAULTS, Config

        settings = LiveSettings.from_config(Config(DEFAULTS))
        self.assertEqual(settings.model, DEFAULTS["live"]["model"])
        self.assertLess(settings.overlap_seconds, settings.window_seconds)


class TestTeeTrackBuffers(unittest.TestCase):
    def _fake_track(self):
        return SimpleNamespace(buffers=queue.Queue())

    def test_sink_sees_every_item_and_the_queue_still_gets_it(self):
        track = self._fake_track()
        seen = []
        tee_track_buffers(track, seen.append)

        track.buffers.put(b"a")
        track.buffers.put(b"b")

        self.assertEqual(seen, [b"a", b"b"])
        self.assertEqual(track.buffers.get(), b"a")
        self.assertEqual(track.buffers.get(), b"b")

    def test_none_sentinel_reaches_the_queue_but_not_the_sink(self):
        track = self._fake_track()
        seen = []
        tee_track_buffers(track, seen.append)

        track.buffers.put(b"a")
        track.buffers.put(None)

        self.assertEqual(seen, [b"a"])  # sentinel filtered out of the tap
        self.assertEqual(track.buffers.get(), b"a")
        self.assertIsNone(track.buffers.get())  # writer thread's stop signal still works

    def test_a_sink_exception_does_not_block_the_original_put(self):
        track = self._fake_track()

        def broken_sink(_item):
            raise RuntimeError("live tap exploded")

        tee_track_buffers(track, broken_sink)
        track.buffers.put(b"still gets written")  # must not raise
        self.assertEqual(track.buffers.get(), b"still gets written")

    def test_restore_removes_the_tap(self):
        track = self._fake_track()
        seen = []
        restore = tee_track_buffers(track, seen.append)

        track.buffers.put(b"tapped")
        restore()
        track.buffers.put(b"not tapped")

        self.assertEqual(seen, [b"tapped"])
        self.assertEqual(track.buffers.get(), b"tapped")
        self.assertEqual(track.buffers.get(), b"not tapped")


class TestLiveTranscriberLogging(unittest.TestCase):
    """LiveTranscriber._handle_chunk (log-write + on_chunk callback) has no
    dependency on whisper or threads, so it's tested directly here rather
    than only indirectly through a real (slow, non-deterministic) whisper
    run. A real-audio smoke run against the cached "tiny" model separately
    confirmed the surrounding pipeline end to end, but produced zero chunks
    - synthetic sine-wave audio isn't speech, so whisper correctly emits
    nothing for it (double-checked directly against the model with vad
    both on and off) - which is exactly why this path needs its own
    deterministic coverage rather than relying on that run alone.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.session_dir = Path(self._tmp.name) / "session"  # does not exist yet

    def tearDown(self):
        self._tmp.cleanup()

    def test_handle_chunk_appends_a_json_line_and_calls_on_chunk(self):
        received = []
        live = LiveTranscriber(self.session_dir, LiveSettings(), on_chunk=received.append)
        chunk = LiveChunk(
            track="mic", speaker="Me", window_start=0.0, window_end=8.0, text="hello world"
        )

        live._handle_chunk(chunk)

        log_path = self.session_dir / "live_transcript.jsonl"
        self.assertTrue(log_path.exists(), "log file must be created on first chunk")
        lines = log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["track"], "mic")
        self.assertEqual(record["speaker"], "Me")
        self.assertEqual(record["text"], "hello world")
        self.assertIn("logged_at", record)
        self.assertEqual(received, [chunk])

    def test_multiple_chunks_append_multiple_lines_in_order(self):
        live = LiveTranscriber(self.session_dir, LiveSettings())
        for i in range(3):
            live._handle_chunk(
                LiveChunk(track="mic", speaker="Me", window_start=i, window_end=i + 1, text=f"n{i}")
            )
        lines = (self.session_dir / "live_transcript.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(l)["text"] for l in lines], ["n0", "n1", "n2"])

    def test_works_with_no_on_chunk_callback_supplied(self):
        live = LiveTranscriber(self.session_dir, LiveSettings())  # on_chunk defaults to a no-op
        live._handle_chunk(
            LiveChunk(track="system", speaker="Others", window_start=0.0, window_end=1.0, text="hi")
        )
        self.assertTrue((self.session_dir / "live_transcript.jsonl").exists())

    def test_attach_twice_raises(self):
        live = LiveTranscriber(self.session_dir, LiveSettings())
        live._attached = True  # simulate a completed attach() without needing whisper
        with self.assertRaises(RuntimeError):
            live.attach(SimpleNamespace(tracks=[]))


if __name__ == "__main__":
    unittest.main()

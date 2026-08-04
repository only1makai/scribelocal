"""Tests for the merge/dedup pass - the part that keeps real transcripts clean.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scribe.transcribe.merge import (  # noqa: E402
    DedupSettings,
    format_timestamp,
    merge_segments,
    normalize,
    render_markdown,
    text_similarity,
    time_overlap,
)
from scribe.transcribe.segments import LevelTrack, Segment, annotate_levels  # noqa: E402


def seg(track, speaker, start, end, text, rms=-20.0, no_speech=0.05, logprob=-0.2):
    return Segment(
        track=track,
        speaker=speaker,
        start=start,
        end=end,
        text=text,
        avg_logprob=logprob,
        no_speech_prob=no_speech,
        rms_dbfs=rms,
    )


def build(mic, system):
    """Annotate both tracks the way transcribe_session does, then merge."""
    annotate_levels(mic, None)
    annotate_levels(system, None)
    return merge_segments({"mic": mic, "system": system}, DedupSettings())


class TestTextMatching(unittest.TestCase):
    def test_normalize_strips_case_punctuation_and_accents(self):
        self.assertEqual(normalize("Yeah, that works!"), "yeah that works")
        self.assertEqual(normalize("  Café   RÉSUMÉ  "), "cafe resume")

    def test_transcription_variants_score_as_the_same_speech(self):
        a = normalize("So the deadline is next Friday, right?")
        b = normalize("so the deadline is next friday right")
        self.assertGreaterEqual(text_similarity(a, b), 0.82)

    def test_different_sentences_score_low(self):
        a = normalize("Let's start with the budget numbers.")
        b = normalize("I'll send the contract over tomorrow.")
        self.assertLess(text_similarity(a, b), 0.82)

    def test_containment_catches_differently_split_segments(self):
        long = normalize("the numbers came in at forty two thousand this quarter")
        short = normalize("the numbers came in at forty two thousand")
        self.assertEqual(text_similarity(long, short), 1.0)

    def test_short_filler_does_not_match_by_containment(self):
        # "yeah" appears inside longer text constantly; containment must not fire.
        self.assertLess(text_similarity(normalize("yeah"), normalize("yeah okay so anyway")), 1.0)

    def test_time_overlap_is_relative_to_the_shorter_segment(self):
        a = seg("mic", "Me", 0.0, 10.0, "a")
        b = seg("system", "Others", 8.0, 10.0, "b")
        self.assertAlmostEqual(time_overlap(a, b), 1.0)
        self.assertEqual(time_overlap(a, seg("system", "Others", 20.0, 22.0, "c")), 0.0)


class TestSilenceGate(unittest.TestCase):
    def test_near_silent_segments_are_dropped(self):
        mic = [
            seg("mic", "Me", 0.0, 2.0, "Real speech here", rms=-18.0),
            seg("mic", "Me", 30.0, 32.0, "Thanks for watching!", rms=-72.0),
        ]
        kept, stats = build(mic, [])
        self.assertEqual([s.text for s in kept], ["Real speech here"])
        self.assertEqual(stats.dropped_silence, 1)

    def test_segments_whisper_flags_as_non_speech_are_dropped(self):
        mic = [
            seg("mic", "Me", 0.0, 2.0, "Real speech here", rms=-18.0),
            seg("mic", "Me", 9.0, 11.0, "you", rms=-19.0, no_speech=0.95),
        ]
        kept, stats = build(mic, [])
        self.assertEqual([s.text for s in kept], ["Real speech here"])
        self.assertEqual(stats.dropped_silence, 1)


class TestCrossTrackDedup(unittest.TestCase):
    def test_bleed_through_is_dropped_in_both_directions(self):
        # My voice is loud on the mic and faint on the loopback; their voice is
        # the other way round. Each sentence should survive exactly once.
        mic = [
            seg("mic", "Me", 0.0, 2.5, "Hey, can you hear me okay?", rms=-18.0),
            seg("mic", "Me", 3.0, 5.2, "Yeah I can hear you fine", rms=-42.0),
            seg("mic", "Me", 6.0, 9.0, "Great, let's start with the budget", rms=-17.0),
        ]
        system = [
            seg("system", "Others", 0.1, 2.6, "Hey can you hear me ok", rms=-40.0),
            seg("system", "Others", 3.1, 5.3, "Yeah, I can hear you fine.", rms=-14.0),
        ]
        kept, stats = build(mic, system)

        self.assertEqual(stats.dropped_duplicate, 2)
        self.assertEqual(
            [(s.speaker, s.text) for s in kept],
            [
                ("Me", "Hey, can you hear me okay?"),
                ("Others", "Yeah, I can hear you fine."),
                ("Me", "Great, let's start with the budget"),
            ],
        )

    def test_split_segments_are_deduped_against_one_long_segment(self):
        system = [
            seg(
                "system",
                "Others",
                20.0,
                25.0,
                "So the numbers came in at forty two thousand this quarter",
                rms=-15.0,
            ),
        ]
        mic = [
            seg("mic", "Me", 20.1, 23.0, "So the numbers came in at forty two thousand",
                rms=-44.0),
            seg("mic", "Me", 23.1, 25.1, "this quarter", rms=-45.0),
            seg("mic", "Me", 26.0, 28.0, "Wow, that's higher than I expected", rms=-16.0),
        ]
        kept, stats = build(mic, system)
        self.assertEqual(stats.dropped_duplicate, 2)
        self.assertEqual(
            [(s.speaker, s.text) for s in kept],
            [
                ("Others", "So the numbers came in at forty two thousand this quarter"),
                ("Me", "Wow, that's higher than I expected"),
            ],
        )

    def test_repetition_within_one_track_is_kept(self):
        # Someone repeating themselves is not bleed-through - both stay.
        mic = [
            seg("mic", "Me", 0.0, 2.0, "Can you hear me?", rms=-18.0),
            seg("mic", "Me", 4.0, 6.0, "Can you hear me?", rms=-18.0),
        ]
        kept, stats = build(mic, [])
        self.assertEqual(stats.dropped_duplicate, 0)
        self.assertEqual(len(kept), 2)

    def test_similar_text_far_apart_in_time_is_kept(self):
        mic = [seg("mic", "Me", 0.0, 2.0, "Let's circle back on pricing", rms=-18.0)]
        system = [seg("system", "Others", 600.0, 602.0, "Let's circle back on pricing",
                      rms=-18.0)]
        kept, stats = build(mic, system)
        self.assertEqual(stats.dropped_duplicate, 0)
        self.assertEqual(len(kept), 2)

    def test_dedup_can_be_disabled(self):
        mic = [seg("mic", "Me", 0.0, 2.0, "Hey there", rms=-40.0)]
        system = [seg("system", "Others", 0.1, 2.1, "Hey there", rms=-14.0)]
        annotate_levels(mic, None)
        annotate_levels(system, None)
        kept, stats = merge_segments(
            {"mic": mic, "system": system}, DedupSettings(enabled=False)
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(stats.dropped_duplicate, 0)

    def test_single_track_session_needs_no_dedup(self):
        mic = [seg("mic", "Me", 0.0, 2.0, "Solo recording", rms=-18.0)]
        kept, stats = merge_segments({"mic": mic}, DedupSettings())
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats.dropped_duplicate, 0)

    def test_confidence_breaks_ties_when_levels_match(self):
        mic = [seg("mic", "Me", 0.0, 2.0, "The invoice is due Friday", rms=-20.0,
                   logprob=-1.4)]
        system = [seg("system", "Others", 0.1, 2.1, "The invoice is due Friday", rms=-20.0,
                      logprob=-0.1)]
        kept, _ = build(mic, system)
        self.assertEqual([s.speaker for s in kept], ["Others"])


class TestRendering(unittest.TestCase):
    def test_timestamps_are_hh_mm_ss(self):
        self.assertEqual(format_timestamp(0), "00:00:00")
        self.assertEqual(format_timestamp(3671), "01:01:11")

    def test_consecutive_turns_from_one_speaker_become_one_paragraph(self):
        segments = [
            seg("mic", "Me", 0.0, 2.0, "First bit."),
            seg("mic", "Me", 2.1, 4.0, "Second bit."),
            seg("system", "Others", 5.0, 7.0, "Their reply."),
        ]
        markdown = render_markdown(segments, title="Test session")
        self.assertIn("**[00:00:00] Me:** First bit. Second bit.", markdown)
        self.assertIn("**[00:00:05] Others:** Their reply.", markdown)

    def test_empty_transcript_still_renders(self):
        markdown = render_markdown([], title="Silent session")
        self.assertIn("# Silent session", markdown)
        self.assertIn("(no speech detected)", markdown)


class TestLevelTrack(unittest.TestCase):
    def test_levels_track_loud_and_quiet_regions(self):
        rate = 16000
        quiet = np.zeros(rate, dtype=np.int16)
        loud = (np.sin(np.linspace(0, 400 * np.pi, rate)) * 16000).astype(np.int16)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(quiet.tobytes() + loud.tobytes())

            levels = LevelTrack.from_wav(path)
            self.assertLess(levels.dbfs(0.0, 1.0), -100.0)  # digital silence
            self.assertGreater(levels.dbfs(1.0, 2.0), -12.0)  # ~-6 dBFS sine
            self.assertEqual(levels.dbfs(50.0, 51.0), -120.0)  # past the end


if __name__ == "__main__":
    unittest.main()

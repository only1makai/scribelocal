"""Merge/dedup over a whole call, not one contrived pair at a time.

`tests/test_merge.py` pins each rule down in isolation. This file runs the
entire event-hoodie sales call from `demo/sales_call_script.md` through
`merge_segments` with the shipped defaults, because the failure that actually
bites is not a broken rule - it is two working rules interacting badly across
twenty-four turns of real conversation.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# `discover tests` puts this directory on the path already; naming the module
# directly (python -m unittest tests.test_merge_sales_call) does not.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scribe.transcribe.merge import (  # noqa: E402
    DedupSettings,
    merge_segments,
    normalize,
    render_markdown,
    text_similarity,
)
from scribe.transcribe.segments import annotate_levels  # noqa: E402

from fixtures import sales_call  # noqa: E402


def merge(settings: DedupSettings | None = None):
    """Annotate both tracks the way transcribe_session does, then merge."""
    mic = sales_call.mic_track()
    system = sales_call.system_track()
    annotate_levels(mic, None)
    annotate_levels(system, None)
    return merge_segments({"mic": mic, "system": system}, settings or DedupSettings())


class TestWholeCallSurvivesTheMerge(unittest.TestCase):
    def test_every_turn_appears_exactly_once_in_order(self):
        kept, _stats = merge()
        self.assertEqual(
            [(s.speaker, s.text) for s in kept], sales_call.expected_transcript()
        )

    def test_nothing_a_human_said_is_lost(self):
        kept, _stats = merge()
        spoken = {normalize(text) for _speaker, text in sales_call.expected_transcript()}
        self.assertEqual({normalize(s.text) for s in kept}, spoken)

    def test_counts_add_up(self):
        kept, stats = merge()
        # 24 turns kept; 8 bleed copies and 2 whisper artifacts dropped.
        self.assertEqual(stats.kept, 24)
        self.assertEqual(stats.dropped_duplicate, 8)
        self.assertEqual(stats.dropped_silence, 2)
        self.assertEqual(stats.per_track_kept, {"mic": 12, "system": 12})
        self.assertEqual(len(kept), stats.kept)

    def test_the_call_reads_as_a_two_sided_conversation(self):
        kept, _stats = merge()
        speakers = [s.speaker for s in kept]
        self.assertEqual(speakers[0], "Me")
        # Dana and Priya alternate for the whole call; no turn is attributed to
        # the wrong side, which is what a dedup that keeps the wrong copy looks
        # like from the outside.
        self.assertEqual(speakers, ["Me", "Others"] * 12)


class TestTheHardCasesInThisCall(unittest.TestCase):
    def test_the_split_turn_is_deduped_against_its_whole_copy(self):
        # Turn 5 is one segment on the mic and two on the loopback. Both halves
        # must go, and the whole must stay.
        kept, _stats = merge()
        texts = [s.text for s in kept]
        whole = next(text for text in texts if text.startswith("Fifty to seventy"))
        self.assertIn("front and back print included.", whole)
        self.assertEqual(
            [t for t in texts if t.startswith("base hoodies run twenty-eight")], []
        )

    def test_each_half_of_the_split_turn_matches_by_containment(self):
        whole = normalize(sales_call.DANA_TURNS[2][3])
        for _start, _end, half in sales_call.DANA_BLEED_ONTO_SYSTEM[:2]:
            self.assertEqual(text_similarity(whole, normalize(half)), 1.0)

    def test_the_hallucination_after_the_call_is_dropped(self):
        kept, _stats = merge()
        self.assertNotIn("Thanks for watching!", [s.text for s in kept])

    def test_the_non_speech_filler_is_dropped(self):
        kept, _stats = merge()
        self.assertNotIn("you", [s.text for s in kept])

    def test_priya_repeating_the_event_date_is_not_treated_as_bleed(self):
        # "October eleventh" lands twice, 74 seconds apart, both on the loopback.
        # One source cannot bleed onto itself, and the gap is far outside the
        # tolerance window either way.
        kept, _stats = merge()
        mentions = [s.text for s in kept if "October eleventh" in s.text]
        self.assertEqual(len(mentions), 2)

    def test_friday_is_committed_to_by_both_sides(self):
        # Priya commits at 135.9 and Dana confirms at 141.7. Adjacent, similar
        # topic, opposite tracks - exactly the shape a too-loose dedup eats.
        kept, _stats = merge()
        fridays = [(s.speaker, s.text) for s in kept if "Friday" in s.text]
        self.assertEqual([speaker for speaker, _text in fridays], ["Others", "Me", "Others"])


class TestTheDedupPassIsWhatIsDoingTheWork(unittest.TestCase):
    def test_without_dedup_the_call_double_prints(self):
        kept, stats = merge(DedupSettings(enabled=False))
        self.assertEqual(stats.dropped_duplicate, 0)
        self.assertEqual(stats.dropped_silence, 0)
        # 24 real turns + 8 bleed copies + 2 whisper artifacts.
        self.assertEqual(len(kept), 34)

    def test_the_loud_copy_is_the_one_that_survives(self):
        kept, _stats = merge()
        for seg in kept:
            self.assertGreater(
                seg.prominence_db,
                -10.0,
                f"a bleed copy survived instead of the real one: {seg.text!r}",
            )


class TestRenderedTranscript(unittest.TestCase):
    def test_the_transcript_reports_what_the_merge_removed(self):
        kept, stats = merge()
        markdown = render_markdown(
            kept,
            title="hoodie quote call",
            duration_seconds=154.0,
            stats=stats,
        )
        self.assertIn("removed 8 cross-track duplicate(s)", markdown)
        self.assertIn("and 2 silent/non-speech segment(s)", markdown)
        self.assertIn("mic: 12, system: 12", markdown)

    def test_the_pricing_the_summary_needs_is_in_the_transcript(self):
        kept, stats = merge()
        markdown = render_markdown(kept, title="hoodie quote call", stats=stats)
        # The sales-call template is asked for budget, timeline and commitments;
        # none of that can be extracted if dedup ate the turn carrying it.
        for needle in (
            "twenty-eight dollars apiece",
            "fifteen hundred dollars",
            "rush is fifteen percent on top",
            "twelve business days",
            "decision by Friday",
        ):
            self.assertIn(needle, markdown)


if __name__ == "__main__":
    unittest.main()

"""Tests for chunking, templates, and summary config handling.

These cover everything up to the API boundary; the call itself needs a key.
Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scribe.summarize import SummarizeError, _thinking_param, chunk_transcript  # noqa: E402
from scribe.summarize.templates import TEMPLATES, get_template  # noqa: E402

EXPECTED_TEMPLATES = {"general", "class-lecture", "work-shift", "club-meeting", "sales-call"}
EXPECTED_SECTIONS = [
    "## TL;DR",
    "## Key Decisions",
    "## Action Items",
    "## Open Questions",
    "## Notable Moments",
]


class TestTemplates(unittest.TestCase):
    def test_all_five_templates_exist(self):
        self.assertEqual(set(TEMPLATES), EXPECTED_TEMPLATES)

    def test_every_template_asks_for_every_section(self):
        for key in EXPECTED_TEMPLATES:
            prompt = get_template(key).system_prompt()
            for heading in EXPECTED_SECTIONS:
                self.assertIn(heading, prompt, f"{key} is missing {heading}")

    def test_action_items_ask_for_an_owner(self):
        self.assertIn("**Owner**", get_template("general").system_prompt())

    def test_templates_differ_in_focus(self):
        focuses = {get_template(k).focus for k in EXPECTED_TEMPLATES}
        self.assertEqual(len(focuses), len(EXPECTED_TEMPLATES))

    def test_default_and_unknown_lookups(self):
        self.assertEqual(get_template(None).key, "general")
        self.assertEqual(get_template("Sales-Call").key, "sales-call")
        with self.assertRaises(KeyError):
            get_template("stand-up")


class TestChunking(unittest.TestCase):
    def test_short_transcript_is_one_chunk(self):
        text = "**[00:00:00] Me:** hello\n\n**[00:00:05] Others:** hi"
        self.assertEqual(chunk_transcript(text, 150_000), [text])

    def test_long_transcript_splits_on_paragraph_boundaries(self):
        paragraphs = [f"**[00:0{i}:00] Me:** {'word ' * 40}".strip() for i in range(9)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_transcript(text, 500)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.startswith("**["), "a chunk began mid-turn")
        # Nothing may be lost or duplicated in the split.
        self.assertEqual("\n\n".join(chunks), text)

    def test_oversized_paragraph_is_not_truncated(self):
        huge = "**[00:00:00] Me:** " + ("x" * 900)
        chunks = chunk_transcript(f"short one\n\n{huge}\n\nshort two", 200)
        self.assertIn(huge, chunks)
        self.assertEqual("\n\n".join(chunks), f"short one\n\n{huge}\n\nshort two")


class TestThinkingParam(unittest.TestCase):
    def test_accepted_values(self):
        self.assertEqual(_thinking_param("adaptive"), {"type": "adaptive"})
        self.assertEqual(_thinking_param("off"), {"type": "disabled"})
        self.assertIsNone(_thinking_param(None))
        self.assertIsNone(_thinking_param("none"))

    def test_typo_is_reported_rather_than_ignored(self):
        with self.assertRaises(SummarizeError):
            _thinking_param("addaptive")


if __name__ == "__main__":
    unittest.main()

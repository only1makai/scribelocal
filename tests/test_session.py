"""Tests for session.py's meta.json handling - in particular the `mode`
field's back-compat default, since sessions written before live mode existed
have no `mode` key at all and must still read as "record", not "unknown".

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scribe.session import list_sessions  # noqa: E402


class TestModeBackCompat(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.notes = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _make(self, name: str, meta: dict | None = None):
        session = self.notes / name
        session.mkdir(parents=True)
        (session / "meta.json").write_text(json.dumps(meta or {}), encoding="utf-8")
        return session

    def test_missing_mode_key_reads_as_record(self):
        # Exactly what a pre-live-mode session's meta.json looks like: no
        # `mode` key was ever written, because the key didn't exist yet.
        self._make("2026-01-01_0900_old", {"title": "Old session", "status": "recorded"})
        sessions = list_sessions(self.notes)
        self.assertEqual(sessions[0]["mode"], "record")

    def test_explicit_record_mode_is_preserved(self):
        self._make("2026-08-01_0900_a", {"mode": "record"})
        sessions = list_sessions(self.notes)
        self.assertEqual(sessions[0]["mode"], "record")

    def test_explicit_live_mode_is_preserved(self):
        self._make("2026-08-01_0900_b", {"mode": "live"})
        sessions = list_sessions(self.notes)
        self.assertEqual(sessions[0]["mode"], "live")

    def test_session_with_no_meta_json_at_all_reads_as_record(self):
        # Even more degraded than "no mode key": no meta.json file at all.
        # list_sessions() already tolerates this for other fields; mode must
        # fall back the same way rather than raising.
        (self.notes / "2026-01-01_0900_bare").mkdir(parents=True)
        sessions = list_sessions(self.notes)
        self.assertEqual(sessions[0]["mode"], "record")

    def test_mode_does_not_affect_other_derived_fields(self):
        # A live session's meta.json shape must otherwise look identical to a
        # record session's for anything that isn't mode-specific - the whole
        # point is that live is a superset, not a fork.
        session = self._make(
            "2026-08-01_0900_live",
            {"mode": "live", "title": "Standup", "status": "recorded"},
        )
        (session / "audio_mic.wav").write_bytes(b"RIFF")
        (session / "transcript.md").write_text("# t", encoding="utf-8")

        sessions = list_sessions(self.notes)
        s = sessions[0]
        self.assertEqual(s["mode"], "live")
        self.assertTrue(s["recorded"])
        self.assertTrue(s["transcribed"])
        self.assertFalse(s["summarized"])
        self.assertEqual(s["title"], "Standup")


if __name__ == "__main__":
    unittest.main()

"""Tests for the inbox watcher.

Processing is stubbed - these cover detection, stability gating, timestamping,
and the move-before-process guarantee, not transcription itself.
Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scribe.config import DEFAULTS, Config, _deep_merge  # noqa: E402
from scribe.inbox import (  # noqa: E402
    InboxSettings,
    InboxWatcher,
    recorded_at,
    title_from_filename,
)


class InboxTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.notes = root / "MeetingNotes"
        self.inbox = self.notes / "inbox"
        self.inbox.mkdir(parents=True)

        self.cfg = Config(_deep_merge(DEFAULTS, {
            "paths": {"notes_dir": str(self.notes), "inbox_dir": str(self.inbox)},
        }))
        self.processed: list[Path] = []
        self.log: list[str] = []
        # stable_seconds=0 so most tests don't have to sleep; the gating test
        # sets its own value.
        self.settings = InboxSettings(stable_seconds=0.0)

    def tearDown(self):
        self._tmp.cleanup()

    def make_watcher(self, process=None, settings=None):
        return InboxWatcher(
            notify=self.log.append,
            process=process if process is not None else self.record_process,
            settings=settings or self.settings,
            config_loader=lambda: self.cfg,
        )

    def record_process(self, session_dir: Path, template: str | None) -> None:
        self.processed.append(session_dir)

    def drop(self, name: str, content: bytes = b"RIFF fake audio", *, mtime=None) -> Path:
        path = self.inbox / name
        path.write_bytes(content)
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path


class TestFilenameAndTimestamp(InboxTestCase):
    def test_titles_are_readable(self):
        self.assertEqual(title_from_filename(Path("New Recording 12.m4a")), "New Recording 12")
        self.assertEqual(title_from_filename(Path("team_standup.mp3")), "team standup")
        self.assertEqual(title_from_filename(Path("2026-08-03_voice-memo.m4a")),
                         "2026-08-03 voice memo")

    def test_punctuation_only_filename_still_gets_a_title(self):
        self.assertEqual(title_from_filename(Path("___.m4a")), "imported recording")

    def test_timestamp_comes_from_the_file_not_from_now(self):
        # A memo recorded days ago and synced today must file under its own date.
        old = datetime(2026, 7, 29, 14, 5).timestamp()
        path = self.drop("old memo.m4a", mtime=old)
        self.assertEqual(recorded_at(path), datetime.fromtimestamp(old))

    def test_session_folder_uses_the_recording_date(self):
        old = datetime(2026, 7, 29, 14, 5).timestamp()
        self.drop("old memo.m4a", mtime=old)
        self.make_watcher().scan_once()

        self.assertEqual(len(self.processed), 1)
        self.assertEqual(self.processed[0].name, "2026-07-29_1405_old-memo")


class TestDetection(InboxTestCase):
    def test_common_phone_formats_are_picked_up(self):
        for name in ["memo.m4a", "memo.mp3", "memo.wav"]:
            self.drop(name)
        self.make_watcher().scan_once()
        self.assertEqual(len(self.processed), 3)
        self.assertEqual(list(self.inbox.iterdir()), [])

    def test_non_audio_is_ignored(self):
        for name in ["notes.txt", "photo.jpg", "archive.zip"]:
            self.drop(name)
        self.make_watcher().scan_once()
        self.assertEqual(self.processed, [])
        self.assertEqual(len(list(self.inbox.iterdir())), 3)

    def test_partial_and_placeholder_files_are_ignored(self):
        for name in ["memo.m4a.part", "memo.m4a.crdownload", ".memo.m4a.icloud", ".hidden.m4a"]:
            self.drop(name)
        self.make_watcher().scan_once()
        self.assertEqual(self.processed, [])
        self.assertEqual(len(list(self.inbox.iterdir())), 4)

    def test_zero_byte_placeholder_is_ignored(self):
        self.drop("empty.m4a", content=b"")
        self.make_watcher().scan_once()
        self.assertEqual(self.processed, [])
        self.assertTrue((self.inbox / "empty.m4a").exists())


class TestStability(InboxTestCase):
    def test_a_growing_file_is_not_imported_until_it_settles(self):
        settings = InboxSettings(stable_seconds=0.25)
        watcher = self.make_watcher(settings=settings)
        path = self.drop("syncing.m4a", b"partial")

        # First sighting: recorded, never imported on the same pass.
        watcher.scan_once()
        self.assertEqual(self.processed, [])
        self.assertTrue(path.exists())

        # Still growing - the stability clock restarts.
        time.sleep(0.3)
        path.write_bytes(b"partial + more data")
        watcher.scan_once()
        self.assertEqual(self.processed, [], "imported a file that was still growing")
        self.assertTrue(path.exists())

        # Unchanged for long enough - now it may be imported.
        time.sleep(0.3)
        watcher.scan_once()
        self.assertEqual(len(self.processed), 1)
        self.assertFalse(path.exists())

    def test_forgotten_once_removed(self):
        watcher = self.make_watcher(settings=InboxSettings(stable_seconds=5.0))
        path = self.drop("gone.m4a")
        watcher.scan_once()
        path.unlink()
        watcher.scan_once()
        self.assertEqual(watcher._seen, {})


class TestImport(InboxTestCase):
    def test_file_is_moved_before_processing_runs(self):
        seen_state = {}

        def process(session_dir: Path, template: str | None) -> None:
            # At this point the inbox must already be empty and the audio must
            # already be in the session folder - that is what makes a watcher
            # restart mid-job safe.
            seen_state["inbox_empty"] = list(self.inbox.iterdir()) == []
            seen_state["audio"] = [p.name for p in session_dir.glob("audio_*")]

        self.drop("memo.m4a")
        self.make_watcher(process=process).scan_once()

        self.assertTrue(seen_state["inbox_empty"], "file was still in inbox during processing")
        self.assertEqual(seen_state["audio"], ["audio_phone.m4a"])

    def test_original_extension_is_preserved(self):
        self.drop("memo.MP3")
        self.make_watcher().scan_once()
        self.assertEqual([p.name for p in self.processed[0].glob("audio_*")],
                         ["audio_phone.mp3"])

    def test_meta_json_marks_the_session_as_imported(self):
        old = datetime(2026, 7, 29, 14, 5).timestamp()
        self.drop("Team Standup.m4a", mtime=old)
        self.make_watcher().scan_once()

        meta = json.loads((self.processed[0] / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["title"], "Team Standup")
        self.assertEqual(meta["source"], "inbox")
        self.assertEqual(meta["original_filename"], "Team Standup.m4a")
        self.assertEqual(meta["status"], "recorded")
        self.assertTrue(meta["started_at"].startswith("2026-07-29T14:05"))
        self.assertIn("scribelocal_version", meta)

    def test_two_files_never_share_a_session_folder(self):
        stamp = datetime(2026, 7, 29, 14, 5).timestamp()
        self.drop("memo.m4a", b"first", mtime=stamp)
        self.make_watcher().scan_once()
        self.drop("memo.m4a", b"second", mtime=stamp)
        self.make_watcher().scan_once()

        self.assertEqual(len(self.processed), 2)
        self.assertNotEqual(self.processed[0], self.processed[1])
        self.assertEqual(self.processed[1].name, "2026-07-29_1405_memo-2")
        self.assertEqual((self.processed[0] / "audio_phone.m4a").read_bytes(), b"first")
        self.assertEqual((self.processed[1] / "audio_phone.m4a").read_bytes(), b"second")

    def test_imported_session_appears_in_the_session_list(self):
        from scribe.session import list_sessions

        self.drop("memo.m4a")
        self.make_watcher().scan_once()

        sessions = list_sessions(self.notes)
        self.assertEqual(len(sessions), 1)
        self.assertTrue(sessions[0]["recorded"], "import not counted as recorded")
        self.assertFalse(sessions[0]["transcribed"])

    def test_inbox_folder_is_created_if_missing(self):
        import shutil

        shutil.rmtree(self.inbox)
        self.make_watcher().scan_once()
        self.assertTrue(self.inbox.is_dir())


class TestFailureHandling(InboxTestCase):
    def test_processing_failure_keeps_the_audio_and_does_not_raise(self):
        from scribe.transcribe import TranscriptionError

        def boom(session_dir: Path, template: str | None) -> None:
            raise TranscriptionError("whisper exploded")

        self.drop("memo.m4a")
        watcher = self.make_watcher(process=boom)
        imported = watcher.scan_once()  # must not raise

        self.assertEqual(len(imported), 1)
        self.assertTrue((imported[0] / "audio_phone.m4a").exists())
        joined = "\n".join(self.log)
        self.assertIn("whisper exploded", joined)
        self.assertIn("scribe process", joined)  # tells the user how to retry

    def test_unexpected_error_does_not_kill_the_watcher(self):
        def boom(session_dir: Path, template: str | None) -> None:
            raise RuntimeError("something odd")

        self.drop("a.m4a")
        watcher = self.make_watcher(process=boom)
        watcher.scan_once()

        self.drop("b.m4a")
        watcher.scan_once()  # still working after the failure
        self.assertEqual(len(list(self.notes.glob("*/audio_phone.m4a"))), 2)


class TestSettings(InboxTestCase):
    def test_extensions_are_normalized_from_config(self):
        cfg = Config(_deep_merge(DEFAULTS, {"inbox": {"extensions": ["M4A", ".Mp3", "wav"]}}))
        self.assertEqual(InboxSettings.from_config(cfg).extensions, [".m4a", ".mp3", ".wav"])

    def test_defaults_cover_the_required_formats(self):
        extensions = InboxSettings.from_config(Config(DEFAULTS)).extensions
        for required in (".m4a", ".mp3", ".wav"):
            self.assertIn(required, extensions)


if __name__ == "__main__":
    unittest.main()

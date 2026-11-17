"""Tests for the CLI command functions in scribe/cli.py.

Sandboxes PROJECT_ROOT the same way tests/test_server.py does, and swaps in a
fake recorder so nothing here opens real audio devices.
Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scribe import config as config_module  # noqa: E402

CONFIG_YAML = """\
paths:
  notes_dir: {notes_dir}
  inbox_dir: {notes_dir}/inbox

audio:
  mic_device: null
  loopback_device: null
  chunk_frames: 4096

whisper:
  model: medium
  device: auto
  compute_type: auto

live:
  window_seconds: 8
  overlap_seconds: 1
  model: small

summarize:
  model: claude-sonnet-4-6
  template: general
"""


class FakeDevice:
    def __init__(self, name="Fake Device", sample_rate=48000):
        self.name = name
        self.sample_rate = sample_rate
        self.index = 0
        self.channels = 2


class FakeTrack:
    def __init__(self, label, device_name):
        self.label = label
        self.device = FakeDevice(device_name)

    def level(self):
        return 0.1


class FakeRecorder:
    """Stands in for DualRecorder so tests never open real audio devices.

    Matches DualRecorder's public surface exactly (constructor kwargs,
    .warning, .tracks, .start/.elapsed/.stop) so cli.py's `from
    .audio.recorder import DualRecorder` picks this up unmodified when the
    test monkeypatches scribe.audio.recorder.DualRecorder before calling a
    cmd_* function - the same technique tests/test_server.py already uses.
    """

    instances: list = []

    def __init__(self, session_dir, mic_device=None, loopback_device=None, chunk_frames=4096):
        self.session_dir = Path(session_dir)
        self.warning = None
        self.tracks = [FakeTrack("mic", "Fake Mic"), FakeTrack("system", "Fake Loopback")]
        self.started = False
        FakeRecorder.instances.append(self)

    def start(self):
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.started = True

    def elapsed(self):
        return 0.05

    def stop(self):
        for track in self.tracks:
            (self.session_dir / f"audio_{track.label}.wav").write_bytes(b"RIFF")
        return {
            t.label: {
                "path": str(self.session_dir / f"audio_{t.label}.wav"),
                "device": t.device.name,
                "sample_rate": t.device.sample_rate,
                "seconds": 0.05,
            }
            for t in self.tracks
        }


class CliTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.notes = self.root / "notes"
        self.notes.mkdir()
        (self.root / "config.yaml").write_text(
            CONFIG_YAML.format(notes_dir=self.notes.as_posix()), encoding="utf-8"
        )
        self._saved_root = config_module.PROJECT_ROOT
        config_module.PROJECT_ROOT = self.root

        import scribe.audio.recorder as recorder_module

        FakeRecorder.instances = []
        self._saved_recorder_cls = recorder_module.DualRecorder
        recorder_module.DualRecorder = FakeRecorder

    def tearDown(self):
        import scribe.audio.recorder as recorder_module

        config_module.PROJECT_ROOT = self._saved_root
        recorder_module.DualRecorder = self._saved_recorder_cls
        self._tmp.cleanup()

    def only_session_dir(self) -> Path:
        entries = [e for e in self.notes.iterdir() if e.is_dir() and e.name != "inbox"]
        self.assertEqual(len(entries), 1, f"expected exactly one session dir, found {entries}")
        return entries[0]


class TestCmdRecordMode(CliTestCase):
    def test_record_writes_mode_record(self):
        from scribe.cli import cmd_record

        args = argparse.Namespace(title="Team sync", duration=0.05, template=None)
        rc = cmd_record(args)
        self.assertEqual(rc, 0)

        session_dir = self.only_session_dir()
        meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["mode"], "record")
        self.assertEqual(meta["status"], "recorded")
        self.assertEqual(meta["title"], "Team sync")


class FakeLiveTranscriber:
    """Stands in for scribe.live.stream.LiveTranscriber so this test never
    loads a real whisper model - the same reason FakeRecorder exists above.
    Real-hardware behavior (does the live transcript keep up) is exactly the
    thing 07_BUILD_SEQUENCE.md gates on a real call, not a unit test.
    """

    instances: list = []

    def __init__(self, session_dir, settings, on_chunk=None, on_warning=None):
        self.session_dir = Path(session_dir)
        self.settings = settings
        self.on_chunk = on_chunk
        self.on_warning = on_warning
        # Whether attach() ran before the recorder had actually started -
        # the ordering LiveTranscriber.attach()'s own docstring promises.
        self.attach_saw_recorder_not_yet_started = None
        self.stopped = False
        FakeLiveTranscriber.instances.append(self)

    def attach(self, recorder):
        self.attach_saw_recorder_not_yet_started = not getattr(recorder, "started", False)
        if self.on_chunk:
            from scribe.live.stream import LiveChunk

            self.on_chunk(
                LiveChunk(
                    track="mic", speaker="Me", window_start=0.0, window_end=1.0, text="hello there"
                )
            )
        if self.on_warning:
            self.on_warning("system", "test warning")

    def stop(self):
        self.stopped = True
        return {"mic": {"windows": 1, "chunks_fed": 5}, "system": {"windows": 1, "chunks_fed": 5}}


class LiveTestCase(CliTestCase):
    def setUp(self):
        super().setUp()
        import scribe.live.stream as stream_module

        FakeLiveTranscriber.instances = []
        self._saved_live_cls = stream_module.LiveTranscriber
        stream_module.LiveTranscriber = FakeLiveTranscriber

    def tearDown(self):
        import scribe.live.stream as stream_module

        stream_module.LiveTranscriber = self._saved_live_cls
        super().tearDown()


class TestCmdLive(LiveTestCase):
    def test_live_writes_mode_live_with_the_same_shape_as_record(self):
        from scribe.cli import cmd_live

        args = argparse.Namespace(title="Standup", duration=0.05, template=None,
                                  context=None, no_suggest=True)
        rc = cmd_live(args)
        self.assertEqual(rc, 0)

        session_dir = self.only_session_dir()
        meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["mode"], "live")
        # Same value record mode writes - a live session is a superset, not
        # a fork, so `scribe process` must treat it identically afterward.
        self.assertEqual(meta["status"], "recorded")
        self.assertEqual(meta["title"], "Standup")
        self.assertEqual(meta["live_model"], "small")  # from config.yaml's live.model
        self.assertIn("mic", meta["live_stats"])
        self.assertIn("system", meta["live_stats"])

        # The full WAVs exist exactly like a record session's - nothing
        # about live mode changes what DualRecorder writes to disk.
        self.assertTrue((session_dir / "audio_mic.wav").exists())
        self.assertTrue((session_dir / "audio_system.wav").exists())

        self.assertEqual(len(FakeLiveTranscriber.instances), 1)
        live = FakeLiveTranscriber.instances[0]
        self.assertTrue(
            live.attach_saw_recorder_not_yet_started,
            "attach() must run before recorder.start(), per its own docstring",
        )
        self.assertTrue(live.stopped)

    def test_live_attach_failure_is_surfaced_and_leaves_no_empty_session(self):
        from scribe.cli import cmd_live
        from scribe.transcribe import TranscriptionError

        class BrokenLiveTranscriber(FakeLiveTranscriber):
            def attach(self, recorder):
                raise TranscriptionError(
                    'faster-whisper is not installed. Install it with:\n'
                    '  pip install -e ".[transcribe]"'
                )

        import scribe.live.stream as stream_module

        stream_module.LiveTranscriber = BrokenLiveTranscriber

        args = argparse.Namespace(title="Doomed", duration=0.05, template=None,
                                  context=None, no_suggest=True)
        rc = cmd_live(args)

        self.assertEqual(rc, 1)
        self.assertEqual(
            list(self.notes.iterdir()), [], "an empty session dir was left behind"
        )


class TestDeviceSelectionErrors(CliTestCase):
    """A configured device that can't be resolved stops the command cleanly."""

    MESSAGE = 'No microphone matching "Yeti" (audio.mic_device). Available:\n  [0] Microphone (Brio 101)'

    def setUp(self):
        super().setUp()
        import scribe.audio.recorder as recorder_module
        from scribe.audio.devices import DeviceSelectionError

        message = self.MESSAGE

        class UnresolvableRecorder(FakeRecorder):
            def __init__(self, *args, **kwargs):
                raise DeviceSelectionError(message)

        recorder_module.DualRecorder = UnresolvableRecorder

    def _run(self, fn, args):
        import contextlib
        import io

        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = fn(args)
        return rc, err.getvalue()

    def test_record_exits_2_with_the_message_and_no_empty_session(self):
        from scribe.cli import cmd_record

        rc, err = self._run(
            cmd_record, argparse.Namespace(title="Lecture", duration=0.05, template=None)
        )
        self.assertEqual(rc, 2)
        self.assertIn(self.MESSAGE, err)
        self.assertNotIn("Traceback", err)
        self.assertEqual(list(self.notes.iterdir()), [], "an empty session dir was left behind")

    def test_live_exits_2_with_the_message_and_no_empty_session(self):
        from scribe.cli import cmd_live

        rc, err = self._run(cmd_live, argparse.Namespace(
            title="Lecture", duration=0.05, template=None, context=None, no_suggest=True))
        self.assertEqual(rc, 2)
        self.assertIn(self.MESSAGE, err)
        self.assertEqual(list(self.notes.iterdir()), [], "an empty session dir was left behind")

    def test_existing_session_folder_with_files_is_never_removed(self):
        # create_session_dir() reuses a same-minute, same-title folder. If that
        # folder already holds a recording, a device error must leave it alone.
        from scribe.cli import cmd_record
        from scribe.session import create_session_dir

        existing = create_session_dir(self.notes, "Lecture")
        (existing / "audio_mic.wav").write_bytes(b"RIFF")

        rc, _ = self._run(
            cmd_record, argparse.Namespace(title="Lecture", duration=0.05, template=None)
        )
        self.assertEqual(rc, 2)
        self.assertTrue((existing / "audio_mic.wav").exists())


if __name__ == "__main__":
    unittest.main()

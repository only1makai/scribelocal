"""Tests for the web layer.

Runs against a sandboxed PROJECT_ROOT (its own config.yaml and notes dir) and a
fake recorder, so nothing here touches real audio devices or your real settings.
Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from scribe import config as config_module  # noqa: E402
from scribe.server import jobs as jobs_module  # noqa: E402
from scribe.server.app import create_app  # noqa: E402
from scribe.server.jobs import JobManager  # noqa: E402
from scribe.server.recording import RecordingManager  # noqa: E402

CONFIG_YAML = """\
# ScribeLocal configuration. All paths support ~ expansion.

paths:
  notes_dir: {notes_dir}        # session folders land here
  inbox_dir: {notes_dir}/inbox  # phone-sync drop folder

audio:
  retention_days: 7        # auto-delete WAVs after N days
  mic_device: null         # null = system default input
  chunk_frames: 4096       # frames per buffer written to disk

whisper:
  model: medium            # tiny | base | small | medium | large-v3
  device: auto             # auto | cpu | cuda

diarization:
  enabled: false           # needs HF_TOKEN in .env; slow

summarize:
  model: claude-sonnet-4-6
  template: general        # general | class-lecture | ...

server:
  host: 127.0.0.1
  port: 8321
"""


class ServerTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.notes = self.root / "notes"
        self.notes.mkdir()
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(
            CONFIG_YAML.format(notes_dir=self.notes.as_posix()), encoding="utf-8"
        )

        self._saved_root = config_module.PROJECT_ROOT
        config_module.PROJECT_ROOT = self.root

        # Fresh managers per test so job/recording state can't leak between them.
        self._saved_jobs = jobs_module.job_manager
        jobs_module.job_manager = JobManager()
        import scribe.server.app as app_module
        import scribe.server.recording as recording_module

        self._saved_recorder = recording_module.recorder_state
        recording_module.recorder_state = RecordingManager()
        self._saved_app_jobs = app_module.job_manager
        self._saved_app_rec = app_module.recorder_state
        app_module.job_manager = jobs_module.job_manager
        app_module.recorder_state = recording_module.recorder_state

        self.client = TestClient(create_app())

    def tearDown(self):
        import scribe.server.app as app_module
        import scribe.server.recording as recording_module

        config_module.PROJECT_ROOT = self._saved_root
        jobs_module.job_manager = self._saved_jobs
        recording_module.recorder_state = self._saved_recorder
        app_module.job_manager = self._saved_app_jobs
        app_module.recorder_state = self._saved_app_rec
        self._tmp.cleanup()

    def make_session(self, name="2026-08-03_1400_budget-sync", *, transcript=None,
                     summary=None, meta=None, audio=True):
        session = self.notes / name
        session.mkdir(parents=True, exist_ok=True)
        if audio:
            (session / "audio_mic.wav").write_bytes(b"RIFF....WAVEfmt ")
        if transcript is not None:
            (session / "transcript.md").write_text(transcript, encoding="utf-8")
        if summary is not None:
            (session / "summary.md").write_text(summary, encoding="utf-8")
        payload = {"title": "Budget sync", "duration_seconds": 930, "status": "recorded"}
        payload.update(meta or {})
        (session / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
        return session


class TestPagesAndReads(ServerTestCase):
    def test_index_is_served(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("scribe", res.text.lower())
        self.assertIn("<title>ScribeLocal</title>", res.text)

    def test_templates_match_the_summarizer(self):
        keys = {t["key"] for t in self.client.get("/api/templates").json()}
        self.assertEqual(
            keys, {"general", "class-lecture", "work-shift", "club-meeting", "sales-call"}
        )

    def test_sessions_list_reports_status(self):
        self.make_session("2026-08-01_0900_one")
        self.make_session("2026-08-02_1000_two", transcript="# t", summary="# s")

        data = self.client.get("/api/sessions").json()
        by_name = {s["name"]: s for s in data["sessions"]}
        self.assertEqual(data["notes_dir"], str(self.notes))
        self.assertTrue(by_name["2026-08-01_0900_one"]["recorded"])
        self.assertFalse(by_name["2026-08-01_0900_one"]["transcribed"])
        self.assertTrue(by_name["2026-08-02_1000_two"]["summarized"])

    def test_session_detail_returns_documents(self):
        self.make_session(transcript="**[00:00:01] Me:** hi", summary="## TL;DR\nfine")
        detail = self.client.get("/api/sessions/2026-08-03_1400_budget-sync").json()

        self.assertEqual(detail["title"], "Budget sync")
        self.assertIn("hi", detail["transcript"])
        self.assertIn("TL;DR", detail["summary"])
        self.assertEqual([a["name"] for a in detail["audio"]], ["audio_mic.wav"])

    def test_missing_documents_are_null_not_errors(self):
        self.make_session()
        detail = self.client.get("/api/sessions/2026-08-03_1400_budget-sync").json()
        self.assertIsNone(detail["transcript"])
        self.assertIsNone(detail["summary"])

    def test_unknown_session_is_404(self):
        self.assertEqual(self.client.get("/api/sessions/nope").status_code, 404)

    def test_path_traversal_is_refused(self):
        outside = self.root / "secret.md"
        outside.write_text("private", encoding="utf-8")
        for bad in ["../secret.md", "..", "a/b", "a\\b"]:
            res = self.client.get(f"/api/sessions/{bad}")
            self.assertIn(res.status_code, (400, 404), f"{bad} was not refused")


class TestSettings(ServerTestCase):
    def test_api_key_status_never_exposes_the_key(self):
        import os

        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-super-secret-value"
        try:
            res = self.client.get("/api/settings")
            body = res.text
            self.assertTrue(res.json()["api_key"]["configured"])
            self.assertNotIn("super-secret-value", body)
            self.assertNotIn("sk-ant", body)
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    def test_settings_reports_current_config(self):
        settings = self.client.get("/api/settings").json()
        self.assertEqual(settings["whisper"]["model"], "medium")
        self.assertEqual(settings["audio"]["retention_days"], 7)
        self.assertFalse(settings["diarization"]["enabled"])
        self.assertIn("tiny", settings["whisper"]["choices"])

    def test_saving_settings_preserves_config_comments(self):
        res = self.client.put(
            "/api/settings",
            json={"whisper_model": "small", "retention_days": 30, "diarization_enabled": True},
        )
        self.assertEqual(res.status_code, 200)

        text = self.config_path.read_text(encoding="utf-8")
        self.assertIn("model: small", text)
        self.assertIn("retention_days: 30", text)
        self.assertIn("enabled: true", text)
        # The comments that make config.yaml worth hand-editing must survive.
        self.assertIn("# tiny | base | small | medium | large-v3", text)
        self.assertIn("# needs HF_TOKEN in .env; slow", text)
        self.assertIn("# session folders land here", text)

        reloaded = self.client.get("/api/settings").json()
        self.assertEqual(reloaded["whisper"]["model"], "small")
        self.assertEqual(reloaded["audio"]["retention_days"], 30)
        self.assertTrue(reloaded["diarization"]["enabled"])

    def test_unrelated_settings_are_untouched(self):
        self.client.put("/api/settings", json={"whisper_model": "tiny"})
        text = self.config_path.read_text(encoding="utf-8")
        self.assertIn("model: claude-sonnet-4-6", text)  # summarize.model, not whisper.model
        self.assertIn("port: 8321", text)

    def test_invalid_whisper_model_is_rejected(self):
        res = self.client.put("/api/settings", json={"whisper_model": "enormous"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("medium", res.json()["detail"])
        self.assertIn("model: medium", self.config_path.read_text(encoding="utf-8"))

    def test_negative_retention_is_rejected(self):
        self.assertEqual(
            self.client.put("/api/settings", json={"retention_days": -5}).status_code, 422
        )

    def test_diarization_without_token_returns_a_note(self):
        import os

        os.environ.pop("HF_TOKEN", None)
        res = self.client.put("/api/settings", json={"diarization_enabled": True}).json()
        self.assertTrue(any("HF_TOKEN" in note for note in res["notes"]))


class TestProcessing(ServerTestCase):
    def wait_for_job(self, job_id, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["state"] != "running":
                return job
            time.sleep(0.02)
        self.fail("job did not finish in time")

    def test_process_runs_the_shared_pipeline(self):
        self.make_session()
        calls = {}

        def fake_process(session_dir, cfg, **kwargs):
            calls.update(kwargs)
            calls["session"] = session_dir.name
            kwargs["notify"]("working")
            from scribe.pipeline import ProcessOutcome

            return ProcessOutcome(session_dir=session_dir)

        original = jobs_module.process_session
        jobs_module.process_session = fake_process
        try:
            started = self.client.post(
                "/api/sessions/2026-08-03_1400_budget-sync/process",
                json={"transcribe": True, "summarize": False, "force": True,
                      "template": "sales-call"},
            ).json()
            job = self.wait_for_job(started["id"])
        finally:
            jobs_module.process_session = original

        self.assertEqual(job["state"], "done")
        self.assertIn("working", job["lines"])
        self.assertEqual(calls["session"], "2026-08-03_1400_budget-sync")
        self.assertTrue(calls["transcribe"])
        self.assertFalse(calls["summarize"])
        self.assertTrue(calls["force"])
        self.assertEqual(calls["template"], "sales-call")

    def test_transcription_error_surfaces_on_the_job(self):
        # No audio files at all - transcribe_session raises, and the browser
        # must see why rather than a job that silently ends.
        self.make_session(audio=False)
        started = self.client.post(
            "/api/sessions/2026-08-03_1400_budget-sync/process",
            json={"transcribe": True, "summarize": False},
        ).json()
        job = self.wait_for_job(started["id"])

        self.assertEqual(job["state"], "error")
        self.assertIn("no audio files", job["error"])
        self.assertTrue(any("ERROR:" in line for line in job["lines"]))

    def test_summarize_without_transcript_surfaces_the_error(self):
        self.make_session()
        started = self.client.post(
            "/api/sessions/2026-08-03_1400_budget-sync/process",
            json={"transcribe": False, "summarize": True},
        ).json()
        job = self.wait_for_job(started["id"])
        self.assertEqual(job["state"], "error")
        self.assertIn("no transcript", job["error"])

    def test_second_job_is_refused_while_one_runs(self):
        self.make_session()
        release = {"go": False}

        def slow_process(session_dir, cfg, **kwargs):
            while not release["go"]:
                time.sleep(0.01)
            from scribe.pipeline import ProcessOutcome

            return ProcessOutcome(session_dir=session_dir)

        original = jobs_module.process_session
        jobs_module.process_session = slow_process
        try:
            first = self.client.post(
                "/api/sessions/2026-08-03_1400_budget-sync/process", json={}
            )
            self.assertEqual(first.status_code, 200)
            second = self.client.post(
                "/api/sessions/2026-08-03_1400_budget-sync/process", json={}
            )
            self.assertEqual(second.status_code, 409)
            self.assertIn("already processing", second.json()["detail"])
        finally:
            release["go"] = True
            self.wait_for_job(first.json()["id"])
            jobs_module.process_session = original

    def test_empty_process_request_is_rejected(self):
        self.make_session()
        res = self.client.post(
            "/api/sessions/2026-08-03_1400_budget-sync/process",
            json={"transcribe": False, "summarize": False},
        )
        self.assertEqual(res.status_code, 400)

    def test_unknown_template_is_rejected(self):
        self.make_session()
        res = self.client.post(
            "/api/sessions/2026-08-03_1400_budget-sync/process",
            json={"template": "stand-up"},
        )
        self.assertEqual(res.status_code, 400)


class FakeDevice:
    def __init__(self, name):
        self.name = name
        self.sample_rate = 48000
        self.index = 0
        self.channels = 2


class FakeTrack:
    def __init__(self, label, device_name):
        self.label = label
        self.device = FakeDevice(device_name)
        self._level = 0.42

    def level(self):
        return self._level


class FakeRecorder:
    """Stands in for DualRecorder so tests never open real audio devices."""

    instances: list = []

    def __init__(self, session_dir, mic_device=None, loopback_device=None, chunk_frames=4096):
        self.session_dir = Path(session_dir)
        self.warning = None
        self.tracks = [FakeTrack("mic", "Fake Mic"), FakeTrack("system", "Fake Loopback")]
        self.started = False
        FakeRecorder.instances.append(self)

    def start(self):
        self.started = True

    def elapsed(self):
        return 12.5

    def stop(self):
        for track in self.tracks:
            (self.session_dir / f"audio_{track.label}.wav").write_bytes(b"RIFF")
        return {
            t.label: {
                "path": str(self.session_dir / f"audio_{t.label}.wav"),
                "device": t.device.name,
                "sample_rate": 48000,
                "seconds": 12.5,
            }
            for t in self.tracks
        }


class TestRecording(ServerTestCase):
    def setUp(self):
        super().setUp()
        import scribe.audio.recorder as recorder_module

        FakeRecorder.instances = []
        self._saved_cls = recorder_module.DualRecorder
        recorder_module.DualRecorder = FakeRecorder

    def tearDown(self):
        import scribe.audio.recorder as recorder_module

        recorder_module.DualRecorder = self._saved_cls
        super().tearDown()

    def test_idle_status(self):
        status = self.client.get("/api/status").json()
        self.assertFalse(status["recording"]["recording"])
        self.assertIsNone(status["job"])

    def test_record_start_stop_writes_a_session(self):
        started = self.client.post(
            "/api/record/start", json={"title": "Team sync", "template": "work-shift"}
        )
        self.assertEqual(started.status_code, 200)
        self.assertTrue(started.json()["recording"])

        status = self.client.get("/api/status").json()["recording"]
        self.assertTrue(status["recording"])
        self.assertEqual(status["elapsed"], 12.5)
        self.assertEqual([t["label"] for t in status["tracks"]], ["mic", "system"])
        self.assertAlmostEqual(status["tracks"][0]["level"], 0.42)

        done = self.client.post("/api/record/stop", json={})
        self.assertEqual(done.status_code, 200)
        name = done.json()["session"]

        session_dir = self.notes / name
        meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["title"], "Team sync")
        self.assertEqual(meta["template"], "work-shift")
        self.assertEqual(meta["status"], "recorded")
        self.assertEqual(meta["duration_seconds"], 12.5)
        self.assertIn("team-sync", name)

        self.assertFalse(self.client.get("/api/status").json()["recording"]["recording"])

    def test_double_start_is_refused(self):
        self.client.post("/api/record/start", json={})
        second = self.client.post("/api/record/start", json={})
        self.assertEqual(second.status_code, 409)
        self.client.post("/api/record/stop", json={})

    def test_stop_without_recording_is_refused(self):
        res = self.client.post("/api/record/stop", json={})
        self.assertEqual(res.status_code, 409)
        self.assertIn("no recording", res.json()["detail"])

    def test_processing_is_refused_while_recording(self):
        self.make_session()
        self.client.post("/api/record/start", json={})
        try:
            res = self.client.post(
                "/api/sessions/2026-08-03_1400_budget-sync/process", json={}
            )
            self.assertEqual(res.status_code, 409)
        finally:
            self.client.post("/api/record/stop", json={})

    def test_unknown_template_is_rejected_before_recording(self):
        res = self.client.post("/api/record/start", json={"template": "stand-up"})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(FakeRecorder.instances, [])

    def test_failed_start_leaves_no_empty_session_behind(self):
        import scribe.audio.recorder as recorder_module

        class Unavailable(FakeRecorder):
            def __init__(self, *args, **kwargs):
                raise OSError("no default input device")

        recorder_module.DualRecorder = Unavailable
        res = self.client.post("/api/record/start", json={"title": "doomed"})

        self.assertEqual(res.status_code, 409)
        self.assertIn("no default input device", res.json()["detail"])
        self.assertEqual(list(self.notes.iterdir()), [], "an empty session dir was orphaned")
        self.assertEqual(self.client.get("/api/sessions").json()["sessions"], [])


if __name__ == "__main__":
    unittest.main()

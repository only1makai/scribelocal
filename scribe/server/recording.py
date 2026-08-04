"""Process-wide recording state for the web UI.

Wraps the existing DualRecorder rather than reimplementing capture: the browser
gets the same tracks, the same level meters, and the same meta.json the CLI
writes. A DualRecorder owns real audio devices and is single-use (stop() calls
PyAudio.terminate()), so exactly one lives here at a time and a fresh one is
built per recording.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from ..session import create_session_dir, write_meta


class RecordingError(RuntimeError):
    """Raised when a start/stop request doesn't fit the current state."""


class RecordingManager:
    """Serializes access to the one recorder this process may own."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recorder = None
        self._session_dir: Path | None = None
        self._title: str | None = None
        self._template: str | None = None
        self._started_iso: str | None = None
        self._last: dict | None = None  # summary of the recording that just ended

    @property
    def is_recording(self) -> bool:
        return self._recorder is not None

    def start(self, cfg, title: str | None, template: str | None) -> dict:
        from ..audio.recorder import DualRecorder

        with self._lock:
            if self._recorder is not None:
                raise RecordingError("a recording is already running")

            session_title = (title or "meeting").strip() or "meeting"
            session_dir = create_session_dir(cfg.notes_dir, session_title)
            try:
                recorder = DualRecorder(
                    session_dir,
                    mic_device=cfg.get("audio.mic_device"),
                    loopback_device=cfg.get("audio.loopback_device"),
                    chunk_frames=cfg.get("audio.chunk_frames", 4096),
                )
                recorder.start()
            except Exception as exc:  # device busy, no input device, ...
                # Don't leave an empty session folder behind for a recording
                # that never happened - it would show up in the sessions list.
                try:
                    if not any(session_dir.iterdir()):
                        session_dir.rmdir()
                except OSError:
                    pass
                raise RecordingError(f"could not start recording: {exc}") from exc

            self._recorder = recorder
            self._session_dir = session_dir
            self._title = session_title
            self._template = template or cfg.get("summarize.template")
            self._started_iso = datetime.now(timezone.utc).isoformat()
            self._last = None
            return self._status_locked()

    def stop(self) -> dict:
        with self._lock:
            if self._recorder is None:
                raise RecordingError("no recording is running")

            recorder = self._recorder
            session_dir = self._session_dir
            assert session_dir is not None
            try:
                stats = recorder.stop()
            finally:
                # Whatever happens, this manager must not stay wedged holding a
                # half-dead recorder - the next start() has to be able to run.
                self._recorder = None

            duration = max((t["seconds"] for t in stats.values()), default=0.0)
            write_meta(
                session_dir,
                title=self._title,
                started_at=self._started_iso,
                duration_seconds=duration,
                template=self._template,
                tracks=stats,
                status="recorded",
            )
            self._last = {
                "session": session_dir.name,
                "path": str(session_dir),
                "duration_seconds": duration,
                "tracks": stats,
                "template": self._template,
            }
            self._session_dir = None
            self._title = None
            self._template = None
            self._started_iso = None
            return self._last

    def status(self) -> dict:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> dict:
        if self._recorder is None:
            return {"recording": False, "elapsed": 0.0, "tracks": [], "last": self._last}
        return {
            "recording": True,
            "elapsed": round(self._recorder.elapsed(), 2),
            "session": self._session_dir.name if self._session_dir else None,
            "title": self._title,
            "template": self._template,
            "warning": self._recorder.warning,
            "tracks": [
                {
                    "label": track.label,
                    "device": track.device.name,
                    "sample_rate": track.device.sample_rate,
                    "level": round(track.level(), 4),
                }
                for track in self._recorder.tracks
            ],
            "last": None,
        }


recorder_state = RecordingManager()

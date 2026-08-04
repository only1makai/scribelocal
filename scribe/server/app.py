"""FastAPI app for the local ScribeLocal UI.

This layer is a thin wrapper: recording goes through DualRecorder via
RecordingManager, processing goes through the same `process_session` the CLI
calls, and settings are written back to config.yaml. Nothing here reimplements
capture, transcription, or summarization - if the browser can do it, so can the
CLI, and both take the identical code path.

Binds to 127.0.0.1 by default. There is no authentication: this is a personal
tool serving your own recordings on your own machine. Don't expose it publicly.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ..config import load_config, set_config_values
from ..inbox import InboxSettings, InboxWatcher
from ..session import list_sessions
from ..summarize import TEMPLATES
from .jobs import JobError, job_manager
from .recording import RecordingError, recorder_state

STATIC_DIR = Path(__file__).resolve().parent / "static"
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]


class StartRecording(BaseModel):
    title: str | None = None
    template: str | None = None


class ProcessRequest(BaseModel):
    transcribe: bool = True
    summarize: bool = True
    force: bool = False
    template: str | None = None


class SettingsUpdate(BaseModel):
    whisper_model: str | None = None
    retention_days: int | None = Field(default=None, ge=0, le=3650)
    diarization_enabled: bool | None = None


def _run_inbox_watcher(stop: threading.Event, log: Any) -> None:
    """Inbox watcher for the server: imports go through the shared job queue.

    Routing imports through JobManager (rather than processing them inline)
    means a phone import can't run whisper at the same time as a reprocess
    started from the UI, and its progress shows up in the same job log.
    """

    def submit(session_dir: Path, template: str | None) -> None:
        cfg = load_config()
        summarize = bool(cfg.get("inbox.summarize", True))
        while not stop.is_set():
            if recorder_state.is_recording:
                stop.wait(3.0)  # never compete with a live capture
                continue
            try:
                job_manager.start(
                    session_dir, transcribe=True, summarize=summarize, template=template
                )
                return
            except JobError:
                stop.wait(3.0)  # another job is running; queue behind it

    watcher = InboxWatcher(notify=log, process=submit)
    watcher.run(stop)


def create_app() -> FastAPI:
    watcher_stop = threading.Event()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        thread: threading.Thread | None = None
        try:
            cfg = load_config()
            if InboxSettings.from_config(cfg).enabled:
                thread = threading.Thread(
                    target=_run_inbox_watcher,
                    args=(watcher_stop, lambda m: print(m, flush=True)),
                    name="scribe-inbox",
                    daemon=True,
                )
                thread.start()
        except Exception as exc:  # a broken inbox must not stop the web UI
            print(f"inbox: watcher did not start ({exc})", flush=True)
        try:
            yield
        finally:
            watcher_stop.set()
            if thread is not None:
                thread.join(timeout=5)

    app = FastAPI(title="ScribeLocal", docs_url=None, redoc_url=None, lifespan=lifespan)

    # ---------------------------------------------------------------- helpers

    def session_dir_for(name: str) -> Path:
        """Resolve a session folder, refusing anything outside notes_dir.

        The CLI's resolve_session() accepts arbitrary paths, which is right for a
        terminal but wrong for an HTTP surface - here a session name is only ever
        a folder directly inside notes_dir.
        """
        cfg = load_config()
        notes_dir = cfg.notes_dir
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise HTTPException(status_code=400, detail=f"invalid session name: {name!r}")
        candidate = (notes_dir / name).resolve()
        try:
            candidate.relative_to(notes_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="session is outside notes_dir") from None
        if not candidate.is_dir():
            raise HTTPException(status_code=404, detail=f"no session named {name!r}")
        return candidate

    def read_if_present(path: Path) -> str | None:
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"(could not read {path.name}: {exc})"

    # ------------------------------------------------------------------ pages

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> JSONResponse:
        return JSONResponse({}, status_code=204)

    # ---------------------------------------------------------------- reading

    @app.get("/api/status")
    def get_status() -> dict[str, Any]:
        """Everything the record view polls: capture state plus any live job."""
        job = job_manager.current()
        return {
            "recording": recorder_state.status(),
            "job": job.as_dict() if job else None,
        }

    @app.get("/api/templates")
    def get_templates() -> list[dict[str, str]]:
        return [{"key": t.key, "label": t.label} for t in TEMPLATES.values()]

    @app.get("/api/sessions")
    def get_sessions() -> dict[str, Any]:
        cfg = load_config()
        sessions = list_sessions(cfg.notes_dir)
        for item in sessions:
            meta = item.get("meta") or {}
            item["duration_seconds"] = meta.get("duration_seconds")
            item["started_at"] = meta.get("started_at")
            item["template"] = meta.get("template")
        return {"notes_dir": str(cfg.notes_dir), "sessions": sessions}

    @app.get("/api/sessions/{name}")
    def get_session(name: str) -> dict[str, Any]:
        session_dir = session_dir_for(name)
        meta_path = session_dir / "meta.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            import json

            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                meta = {}

        audio = [
            {"name": p.name, "bytes": p.stat().st_size}
            for p in sorted(session_dir.glob("audio_*"))
            if p.is_file()
        ]
        return {
            "name": session_dir.name,
            "path": str(session_dir),
            "title": meta.get("title", session_dir.name),
            "meta": meta,
            "audio": audio,
            "transcript": read_if_present(session_dir / "transcript.md"),
            "summary": read_if_present(session_dir / "summary.md"),
        }

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return job.as_dict()

    # ---------------------------------------------------------------- actions

    @app.post("/api/record/start")
    def start_recording(body: StartRecording) -> dict[str, Any]:
        cfg = load_config()
        if body.template and body.template not in TEMPLATES:
            raise HTTPException(status_code=400, detail=f"unknown template: {body.template}")
        job = job_manager.current()
        if job is not None and job.state == "running":
            # Whisper would fight the recorder for CPU and risk dropped frames.
            raise HTTPException(
                status_code=409,
                detail=f"'{job.session}' is still processing - wait for it to finish",
            )
        try:
            return recorder_state.start(cfg, body.title, body.template)
        except RecordingError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/record/stop")
    def stop_recording() -> dict[str, Any]:
        try:
            return recorder_state.stop()
        except RecordingError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/sessions/{name}/process")
    def process(name: str, body: ProcessRequest) -> dict[str, Any]:
        session_dir = session_dir_for(name)
        if not body.transcribe and not body.summarize:
            raise HTTPException(status_code=400, detail="nothing to do")
        if body.template and body.template not in TEMPLATES:
            raise HTTPException(status_code=400, detail=f"unknown template: {body.template}")
        if recorder_state.is_recording:
            raise HTTPException(
                status_code=409, detail="cannot process while a recording is running"
            )
        try:
            job = job_manager.start(
                session_dir,
                transcribe=body.transcribe,
                summarize=body.summarize,
                template=body.template,
                force=body.force,
            )
        except JobError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return job.as_dict()

    # --------------------------------------------------------------- settings

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        cfg = load_config()  # also loads .env, so API-key presence is accurate
        return {
            "whisper": {
                "model": cfg.get("whisper.model"),
                "choices": WHISPER_MODELS,
                "device": cfg.get("whisper.device"),
                "compute_type": cfg.get("whisper.compute_type"),
            },
            "audio": {"retention_days": cfg.get("audio.retention_days")},
            "diarization": {
                "enabled": bool(cfg.get("diarization.enabled")),
                # Presence only - the token itself never leaves the server.
                "hf_token_configured": bool(os.environ.get("HF_TOKEN")),
            },
            "api_key": {
                # Never the key, not even a prefix: only whether one is set.
                "configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
                "variable": "ANTHROPIC_API_KEY",
            },
            "summarize": {
                "model": cfg.get("summarize.model"),
                "template": cfg.get("summarize.template"),
            },
            "inbox": {
                "enabled": bool(cfg.get("inbox.enabled", True)),
                "dir": str(cfg.inbox_dir),
            },
            "paths": {"notes_dir": str(cfg.notes_dir)},
        }

    @app.put("/api/settings")
    def put_settings(body: SettingsUpdate) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        if body.whisper_model is not None:
            if body.whisper_model not in WHISPER_MODELS:
                raise HTTPException(
                    status_code=400,
                    detail=f"whisper model must be one of: {', '.join(WHISPER_MODELS)}",
                )
            updates["whisper.model"] = body.whisper_model
        if body.retention_days is not None:
            updates["audio.retention_days"] = body.retention_days
        if body.diarization_enabled is not None:
            updates["diarization.enabled"] = body.diarization_enabled

        if updates:
            try:
                set_config_values(updates)
            except (OSError, ValueError) as exc:
                raise HTTPException(
                    status_code=500, detail=f"could not write config.yaml: {exc}"
                ) from exc

        settings = get_settings()
        notes: list[str] = []
        if body.diarization_enabled and not settings["diarization"]["hf_token_configured"]:
            notes.append("diarization is on but HF_TOKEN is not set - it will be skipped")
        settings["notes"] = notes
        return settings

    @app.get("/api/devices")
    def get_devices() -> dict[str, Any]:
        """Audio device readout - the first thing to check when a track is missing."""
        if recorder_state.is_recording:
            return {"available": False, "reason": "recording in progress", "devices": []}
        try:
            from ..audio.devices import list_devices

            devices = [
                {
                    "index": d.index,
                    "name": d.name,
                    "channels": d.channels,
                    "sample_rate": d.sample_rate,
                    "is_loopback": d.is_loopback,
                }
                for d in list_devices()
            ]
        except Exception as exc:
            return {"available": False, "reason": str(exc), "devices": []}
        return {
            "available": True,
            "devices": devices,
            "has_loopback": any(d["is_loopback"] for d in devices),
        }

    return app

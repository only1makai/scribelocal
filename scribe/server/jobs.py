"""Background processing jobs for the web UI.

Transcription takes minutes and summarization makes network calls, so neither
can run inside a request. Each job runs on its own thread and streams the same
progress lines the CLI prints (both go through the `notify` callback the
transcribe/summarize modules already accept); the browser polls for them.

One job at a time, deliberately: whisper is happy to eat every core, and two
concurrent runs on one machine are slower than two sequential ones.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import load_config
from ..pipeline import process_session
from ..summarize import SummarizeError
from ..transcribe import TranscriptionError


class JobError(RuntimeError):
    """Raised when a job cannot be accepted."""


@dataclass
class Job:
    id: str
    session: str
    transcribe: bool
    summarize: bool
    force: bool
    template: str | None = None
    state: str = "running"  # running | done | error
    lines: list[str] = field(default_factory=list)
    error: str | None = None
    outcome: dict | None = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: str | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "session": self.session,
            "transcribe": self.transcribe,
            "summarize": self.summarize,
            "force": self.force,
            "template": self.template,
            "state": self.state,
            "lines": list(self.lines),
            "error": self.error,
            "outcome": self.outcome,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._current: str | None = None

    def current(self) -> Job | None:
        with self._lock:
            return self._jobs.get(self._current) if self._current else None

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)[:limit]

    def start(
        self,
        session_dir: Path,
        *,
        transcribe: bool = True,
        summarize: bool = True,
        template: str | None = None,
        force: bool = False,
    ) -> Job:
        with self._lock:
            running = self._jobs.get(self._current) if self._current else None
            if running is not None and running.state == "running":
                raise JobError(
                    f"already processing '{running.session}' - wait for it to finish"
                )
            job = Job(
                id=uuid.uuid4().hex[:12],
                session=session_dir.name,
                transcribe=transcribe,
                summarize=summarize,
                force=force,
                template=template,
            )
            self._jobs[job.id] = job
            self._current = job.id

        thread = threading.Thread(
            target=self._run,
            args=(job, session_dir),
            name=f"scribe-job-{job.id}",
            daemon=True,
        )
        thread.start()
        return job

    def _run(self, job: Job, session_dir: Path) -> None:
        def say(message: str) -> None:
            job.lines.append(message)

        try:
            # Config is re-read per job so a settings change (e.g. a smaller
            # whisper model) applies to the next run without a server restart.
            cfg = load_config()
            outcome = process_session(
                session_dir,
                cfg,
                transcribe=job.transcribe,
                summarize=job.summarize,
                template=job.template,
                force=job.force,
                notify=say,
            )
            job.outcome = outcome.as_dict()
            job.state = "done"
            say("finished")
        except (TranscriptionError, SummarizeError) as exc:
            # Expected, actionable failures: missing model, no API key, refusal.
            job.state = "error"
            job.error = str(exc)
            say(f"ERROR: {exc}")
        except Exception as exc:  # pragma: no cover - unexpected, still must surface
            job.state = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            say(f"ERROR: {job.error}")
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat()


job_manager = JobManager()

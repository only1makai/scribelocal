"""Inbox watcher: audio dropped in inbox/ becomes a normal session.

Point your phone's sync client (iCloud, OneDrive, Syncthing - not our problem)
at `paths.inbox_dir`. Anything audio that lands there is imported and run
through the same `process_session` the CLI and web UI use, so a voice memo ends
up indistinguishable from a session recorded on this laptop.

Two things make this safe against a sync client rather than a local copy:

* **Stability check.** A file is ignored until its size and mtime stop changing
  for `stable_seconds`. Syncing a 200 MB memo takes a while, and half a file
  transcribes into garbage.
* **Move on detection.** The audio is moved into its session folder *before*
  processing starts, so a watcher that restarts mid-job never sees it again.
  The work is resumable (`scribe process <session>`), the import is not
  repeatable.

Polling rather than filesystem events: cloud-sync clients materialize files in
ways that don't always raise watch events, and polling makes the stability
check natural. The cost is one directory listing every couple of seconds.
"""

from __future__ import annotations

import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import load_config
from .pipeline import process_session
from .session import create_session_dir, write_meta
from .summarize import SummarizeError
from .transcribe import TranscriptionError

Notify = Callable[[str], None]
ProcessFn = Callable[[Path, str | None], None]  # (session_dir, template)

# What a phone might plausibly drop in here. Decoding is PyAV's problem (it
# bundles FFmpeg), so this list is about intent, not codec support.
DEFAULT_EXTENSIONS = [
    ".m4a",  # iPhone Voice Memos default
    ".mp3",
    ".wav",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".mp4",
    ".m4b",
    ".amr",  # older Android recorders
    ".wma",
]

# Partial-download markers. A file still being written by a sync client often
# carries one of these, and .icloud placeholders aren't audio at all yet.
PARTIAL_SUFFIXES = {".part", ".partial", ".crdownload", ".download", ".tmp", ".icloud"}


@dataclass
class InboxSettings:
    enabled: bool = True
    poll_seconds: float = 2.0
    stable_seconds: float = 3.0
    summarize: bool = True
    template: str | None = None
    extensions: list[str] = field(default_factory=lambda: list(DEFAULT_EXTENSIONS))

    @classmethod
    def from_config(cls, cfg) -> "InboxSettings":
        d = cls()
        raw = cfg.get("inbox.extensions") or DEFAULT_EXTENSIONS
        extensions = [
            ("." + str(e).lower().lstrip(".")) for e in raw if str(e).strip()
        ]
        return cls(
            enabled=bool(cfg.get("inbox.enabled", d.enabled)),
            poll_seconds=float(cfg.get("inbox.poll_seconds", d.poll_seconds)),
            stable_seconds=float(cfg.get("inbox.stable_seconds", d.stable_seconds)),
            summarize=bool(cfg.get("inbox.summarize", d.summarize)),
            template=cfg.get("inbox.template"),
            extensions=extensions,
        )


def title_from_filename(path: Path) -> str:
    """A readable session title from a filename.

    "New Recording 12.m4a" -> "New Recording 12"
    "2026-08-03_voice-memo.m4a" -> "2026-08-03 voice memo"
    """
    stem = re.sub(r"[_]+", " ", path.stem)
    # Only hyphens between letters become spaces - "voice-memo" is two words,
    # but "2026-08-03" is a date and should survive intact.
    stem = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    return stem or "imported recording"


def recorded_at(path: Path) -> datetime:
    """Best guess at when the audio was recorded, not when it synced.

    A copied or synced file usually keeps its original mtime while ctime becomes
    the moment it landed on this disk, so the earlier of the two is the better
    estimate. Sync delay can be hours; using "now" would file a Tuesday lecture
    under Thursday.
    """
    stat = path.stat()
    return datetime.fromtimestamp(min(stat.st_mtime, stat.st_ctime))


def human_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size / 1048576:.1f} MB"


class InboxWatcher:
    """Polls the inbox folder and imports whatever settles there."""

    def __init__(
        self,
        *,
        notify: Notify | None = None,
        process: ProcessFn | None = None,
        settings: InboxSettings | None = None,
        config_loader: Callable[[], object] = load_config,
    ) -> None:
        self._notify = notify or (lambda _msg: None)
        self._process = process
        self._settings = settings
        self._load_config = config_loader
        # path -> (size, mtime, first time we saw it unchanged)
        self._seen: dict[Path, tuple[int, float, float]] = {}
        self._stop = threading.Event()

    # ------------------------------------------------------------------ setup

    def settings(self, cfg) -> InboxSettings:
        return self._settings or InboxSettings.from_config(cfg)

    def ensure_inbox(self, cfg) -> Path:
        inbox = cfg.inbox_dir
        inbox.mkdir(parents=True, exist_ok=True)
        return inbox

    # ------------------------------------------------------------- scanning

    def _is_candidate(self, path: Path, settings: InboxSettings) -> bool:
        if not path.is_file():
            return False
        name = path.name
        if name.startswith(".") or name.startswith("~"):
            return False  # hidden, or an Office-style lock file
        suffixes = {s.lower() for s in path.suffixes}
        if suffixes & PARTIAL_SUFFIXES:
            return False
        return path.suffix.lower() in settings.extensions

    def stable_files(self, inbox: Path, settings: InboxSettings) -> list[Path]:
        """Files that have stopped changing for long enough to be safe to move.

        Two independent ways to qualify:

        * The file was already quiet when we first saw it - its mtime is older
          than `stable_seconds`, so nothing has touched it recently. This is the
          common case for anything that finished syncing before the watcher
          started, and it's what makes `scribe watch --once` useful.
        * We have watched it across polls and its size and mtime never moved.
          This is what catches a file actively being written right now.
        """
        now = time.monotonic()
        wall_now = time.time()
        ready: list[Path] = []
        present: set[Path] = set()

        try:
            entries = sorted(inbox.iterdir())
        except OSError:
            return ready

        for path in entries:
            if not self._is_candidate(path, settings):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue  # vanished mid-scan
            if stat.st_size == 0:
                continue  # placeholder the sync client hasn't filled in yet

            present.add(path)
            fingerprint = (stat.st_size, stat.st_mtime)
            previous = self._seen.get(path)
            if previous is None or (previous[0], previous[1]) != fingerprint:
                self._seen[path] = (stat.st_size, stat.st_mtime, now)
                # Untouched for longer than the settle window already? Then it
                # isn't mid-write and there's nothing to wait for. Clamp at zero:
                # filesystem timestamps can sit slightly ahead of the wall clock,
                # and a negative "age" must not make a settled file look busy.
                if max(0.0, wall_now - stat.st_mtime) >= settings.stable_seconds:
                    ready.append(path)
                continue
            if now - previous[2] >= settings.stable_seconds:
                ready.append(path)

        # Forget files that are gone, so a name reused later starts fresh.
        for path in list(self._seen):
            if path not in present:
                del self._seen[path]
        return ready

    # ------------------------------------------------------------- importing

    def import_file(self, path: Path, cfg, settings: InboxSettings) -> Path | None:
        """Move one file into a fresh session folder. Returns the session dir."""
        try:
            size = path.stat().st_size
            when = recorded_at(path)
        except OSError:
            return None  # gone between scan and import

        title = title_from_filename(path)
        session_dir = create_session_dir(cfg.notes_dir, title, when, unique=True)
        destination = session_dir / f"audio_phone{path.suffix.lower()}"

        try:
            shutil.move(str(path), str(destination))
        except (OSError, shutil.Error) as exc:
            # Usually the sync client still holds a lock; try again next poll.
            self._notify(f"inbox: could not move {path.name} yet ({exc})")
            try:
                if not any(session_dir.iterdir()):
                    session_dir.rmdir()
            except OSError:
                pass
            self._seen.pop(path, None)
            return None

        write_meta(
            session_dir,
            title=title,
            started_at=when.astimezone().isoformat(),
            template=settings.template or cfg.get("summarize.template"),
            source="inbox",
            original_filename=path.name,
            imported_at=datetime.now(timezone.utc).isoformat(),
            tracks={"phone": {"path": str(destination), "bytes": size}},
            status="recorded",
        )
        self._notify(f"inbox: picked up {path.name} ({human_bytes(size)})")
        self._notify(f"inbox: -> {session_dir.name}")
        return session_dir

    def process(self, session_dir: Path, cfg, settings: InboxSettings) -> None:
        if self._process is not None:
            self._process(session_dir, settings.template)
            return
        process_session(
            session_dir,
            cfg,
            transcribe=True,
            summarize=settings.summarize,
            template=settings.template,
            notify=lambda m: self._notify(f"  {m}"),
        )

    # ----------------------------------------------------------------- loops

    def scan_once(self) -> list[Path]:
        """One poll: import and process everything that has settled."""
        cfg = self._load_config()
        settings = self.settings(cfg)
        inbox = self.ensure_inbox(cfg)
        imported: list[Path] = []

        for path in self.stable_files(inbox, settings):
            session_dir = self.import_file(path, cfg, settings)
            if session_dir is None:
                continue
            imported.append(session_dir)
            try:
                self.process(session_dir, cfg, settings)
            except (TranscriptionError, SummarizeError) as exc:
                # The audio is already safe inside the session folder, so this
                # is recoverable: fix the cause and re-run `scribe process`.
                self._notify(f"inbox: ERROR processing {session_dir.name}: {exc}")
                self._notify(f"inbox: audio is safe in {session_dir.name} - "
                             f"retry with: scribe process {session_dir.name}")
            except Exception as exc:  # never let one bad file kill the watcher
                self._notify(f"inbox: ERROR processing {session_dir.name}: "
                             f"{type(exc).__name__}: {exc}")
        return imported

    def run(self, stop: threading.Event | None = None) -> None:
        """Poll until stopped. Blocks; run in a thread for the web server."""
        stop = stop or self._stop
        cfg = self._load_config()
        settings = self.settings(cfg)
        inbox = self.ensure_inbox(cfg)
        self._notify(f"inbox: watching {inbox}")

        while not stop.is_set():
            try:
                self.scan_once()
            except Exception as exc:  # a transient FS error must not end the loop
                self._notify(f"inbox: scan failed ({type(exc).__name__}: {exc})")
            stop.wait(settings.poll_seconds)

    def stop(self) -> None:
        self._stop.set()

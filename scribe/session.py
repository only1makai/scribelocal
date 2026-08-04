"""Session folders under notes_dir: YYYY-MM-DD_HHMM_<title>/ with meta.json."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from . import __version__


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w\-]+", "-", title.strip()).strip("-").lower()
    return slug or "untitled"


def create_session_dir(
    notes_dir: Path, title: str, when: datetime | None = None, *, unique: bool = False
) -> Path:
    """Create a session folder named `YYYY-MM-DD_HHMM_<slug>`.

    `when` overrides the timestamp - imported recordings are stamped with when
    they were actually recorded, not when they happened to be noticed. `unique`
    appends a counter rather than reusing an existing folder, so two imports
    that resolve to the same minute and title can't land on top of each other.
    """
    stamp = (when or datetime.now()).strftime("%Y-%m-%d_%H%M")
    base = f"{stamp}_{slugify(title)}"
    session_dir = notes_dir / base
    if unique:
        counter = 2
        while session_dir.exists():
            session_dir = notes_dir / f"{base}-{counter}"
            counter += 1
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def write_meta(session_dir: Path, **fields) -> None:
    meta_path = session_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("scribelocal_version", __version__)
    meta.update(fields)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def resolve_session(notes_dir: Path, token: str) -> Path:
    """Find a session folder from a path, an exact name, or a unique prefix."""
    candidate = Path(token).expanduser()
    if candidate.is_dir():
        return candidate

    direct = notes_dir / token
    if direct.is_dir():
        return direct

    if not notes_dir.exists():
        raise FileNotFoundError(f"notes directory does not exist: {notes_dir}")

    entries = [e for e in sorted(notes_dir.iterdir()) if e.is_dir() and e.name != "inbox"]
    lowered = token.lower()
    matches = [e for e in entries if e.name.lower().startswith(lowered)]
    if not matches:
        matches = [e for e in entries if lowered in e.name.lower()]

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no session matching '{token}' in {notes_dir}")
    names = "\n  ".join(e.name for e in matches)
    raise FileNotFoundError(f"'{token}' matches {len(matches)} sessions:\n  {names}")


def list_sessions(notes_dir: Path) -> list[dict]:
    sessions = []
    if not notes_dir.exists():
        return sessions
    for entry in sorted(notes_dir.iterdir(), reverse=True):
        if not entry.is_dir() or entry.name == "inbox":
            continue
        meta_path = entry / "meta.json"
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sessions.append(
            {
                "dir": str(entry),
                "name": entry.name,
                "title": meta.get("title", entry.name),
                # Any audio counts: laptop captures write audio_mic/audio_system,
                # inbox imports write audio_phone with the original extension.
                "recorded": any(entry.glob("audio_*")),
                "transcribed": (entry / "transcript.md").exists(),
                "summarized": (entry / "summary.md").exists(),
                "meta": meta,
            }
        )
    return sessions

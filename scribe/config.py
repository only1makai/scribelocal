"""Config loading: config.yaml merged over defaults, plus .env secrets."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "paths": {
        "notes_dir": "~/Documents/MeetingNotes",
        "inbox_dir": "~/Documents/MeetingNotes/inbox",
    },
    "audio": {
        "retention_days": 7,
        "mic_device": None,
        "loopback_device": None,
        "chunk_frames": 4096,
    },
    "whisper": {
        "model": "medium",
        "device": "auto",
        "compute_type": "auto",
        "language": None,
    },
    "merge": {
        "dedup": True,
        "time_tolerance": 2.0,
        "min_overlap": 0.3,
        "similarity": 0.82,
        "silence_floor_dbfs": -50.0,
        "max_no_speech": 0.6,
    },
    "inbox": {
        "enabled": True,
        "poll_seconds": 2.0,
        "stable_seconds": 3.0,
        "summarize": True,
        "template": None,
        "extensions": [
            ".m4a", ".mp3", ".wav", ".aac", ".flac",
            ".ogg", ".opus", ".mp4", ".m4b", ".amr", ".wma",
        ],
    },
    "diarization": {"enabled": False},
    "summarize": {
        "model": "claude-sonnet-4-6",
        "template": "general",
        "max_chunk_chars": 150_000,
        "max_output_tokens": 8_000,
        "thinking": "adaptive",
        "effort": None,
    },
    "server": {"host": "127.0.0.1", "port": 8321},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        """Config lookup by dotted path, e.g. cfg.get('audio.retention_days')."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def notes_dir(self) -> Path:
        return Path(self.get("paths.notes_dir")).expanduser()

    @property
    def inbox_dir(self) -> Path:
        return Path(self.get("paths.inbox_dir")).expanduser()


def load_config(path: Path | None = None) -> Config:
    load_dotenv(PROJECT_ROOT / ".env")
    config_path = path or PROJECT_ROOT / "config.yaml"
    user_data: dict[str, Any] = {}
    if config_path.exists():
        user_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return Config(_deep_merge(DEFAULTS, user_data))


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    # Quote anything YAML would otherwise reinterpret as a non-string.
    if text == "" or re.search(r"[:#\[\]{},&*?|>'\"%@`]", text) or text.lower() in {
        "true", "false", "null", "yes", "no", "on", "off",
    }:
        return '"' + text.replace('"', '\\"') + '"'
    return text


def set_config_values(updates: dict[str, Any], path: Path | None = None) -> None:
    """Persist dotted-path settings (e.g. "whisper.model") to config.yaml.

    Edits the matching line in place so the file's comments and layout survive -
    config.yaml is meant to stay hand-editable, and a full YAML re-dump would
    strip every explanatory comment in it. Keys the file doesn't already contain
    fall back to a merged re-dump.
    """
    config_path = path or PROJECT_ROOT / "config.yaml"
    if not updates:
        return

    if config_path.exists():
        lines = config_path.read_text(encoding="utf-8").splitlines()
        remaining = dict(updates)
        for dotted, value in list(updates.items()):
            section, _, key = dotted.rpartition(".")
            index = _find_key_line(lines, section, key)
            if index is None:
                continue
            line = lines[index]
            indent = line[: len(line) - len(line.lstrip())]
            comment = ""
            match = re.search(r"\s+#.*$", line)
            if match:
                comment = match.group(0)
            lines[index] = f"{indent}{key}: {_format_scalar(value)}{comment}"
            remaining.pop(dotted)
        if not remaining:
            config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return

    # Fallback: no file yet, or a key that isn't written down anywhere.
    existing: dict[str, Any] = {}
    if config_path.exists():
        existing = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for dotted, value in updates.items():
        node = existing
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot set '{dotted}': '{part}' is not a section")
        node[parts[-1]] = value
    config_path.write_text(
        yaml.safe_dump(existing, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _find_key_line(lines: list[str], section: str, key: str) -> int | None:
    """Index of `key:` inside `section:` (dotted sections supported), or None."""
    depth = 0
    start = 0
    for part in [p for p in section.split(".") if p]:
        found = None
        for i in range(start, len(lines)):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(lines[i]) - len(lines[i].lstrip())
            if indent < depth * 2 and i > start:
                break  # left the enclosing section
            if indent == depth * 2 and stripped.split(":")[0].strip() == part:
                found = i
                break
        if found is None:
            return None
        start, depth = found + 1, depth + 1

    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(lines[i]) - len(lines[i].lstrip())
        if indent < depth * 2:
            return None  # ran past the end of the section
        if indent == depth * 2 and stripped.split(":")[0].strip() == key:
            return i
    return None

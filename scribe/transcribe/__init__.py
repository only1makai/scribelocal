"""Transcription: two WAVs in, one deduped transcript.md out.

    audio_mic.wav    --whisper-->  segments (speaker "Me")     --+
                                                                 +--> merge/dedup --> transcript.md
    audio_system.wav --whisper-->  segments (speaker "Others")  --+
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .diarize import DiarizationError, apply_diarization, diarize_track
from .engine import ProgressFn, TranscriptionError, transcribe_track
from .merge import DedupSettings, MergeStats, merge_segments, render_markdown
from .segments import LevelTrack, Segment, annotate_levels

__all__ = [
    "DiarizationError",
    "TranscriptionError",
    "TranscriptResult",
    "transcribe_session",
]

# Which file is whose voice. The mic is you; the loopback is everyone else.
# "phone" is an imported recording (see scribe.inbox) - one device in the room
# heard everybody, so it gets a neutral label rather than being called "Me".
# Patterns are globs because imports keep their original extension (.m4a, ...).
TRACKS: list[tuple[str, str, str]] = [  # (label, filename glob, speaker)
    ("mic", "audio_mic.wav", "Me"),
    ("system", "audio_system.wav", "Others"),
    ("phone", "audio_phone.*", "Phone"),
]


@dataclass
class TranscriptResult:
    path: Path
    segments: list[Segment]
    stats: MergeStats
    warnings: list[str]


def transcribe_session(
    session_dir: Path,
    cfg,
    *,
    on_progress: ProgressFn | None = None,
    notify: Callable[[str], None] | None = None,
) -> TranscriptResult:
    """Transcribe every track in a session and write transcript.md."""
    say = notify or (lambda _msg: None)
    warnings: list[str] = []

    available: list[tuple[str, Path, str]] = []
    for label, pattern, speaker in TRACKS:
        matches = sorted(session_dir.glob(pattern))
        if matches:
            available.append((label, matches[0], speaker))
    if not available:
        raise TranscriptionError(
            f"no audio files in {session_dir} "
            f"(expected audio_mic.wav, audio_system.wav, and/or audio_phone.*)"
        )

    settings = DedupSettings.from_config(cfg)
    diarization_on = bool(cfg.get("diarization.enabled", False))
    tracks: dict[str, list[Segment]] = {}

    for label, wav_path, speaker in available:
        say(f"  [{label}] transcribing {wav_path.name} ...")
        segments = transcribe_track(
            wav_path,
            label,
            speaker,
            model_size=cfg.get("whisper.model", "medium"),
            device=cfg.get("whisper.device", "auto"),
            compute_type=cfg.get("whisper.compute_type", "auto"),
            language=cfg.get("whisper.language"),
            on_progress=on_progress,
        )

        # Levels drive the dedup pass, so they are read even when dedup is off -
        # they cost one streaming pass over the audio and land in the stats.
        # Any failure here is non-fatal by design: losing levels costs dedup
        # accuracy, and must never cost the transcript itself.
        try:
            levels = LevelTrack.from_file(wav_path)
        except Exception as exc:
            warnings.append(f"could not measure levels for {wav_path.name}: {exc}")
            levels = None
        annotate_levels(segments, levels)

        if diarization_on:
            try:
                say(f"  [{label}] diarizing ...")
                apply_diarization(segments, diarize_track(wav_path), speaker)
            except DiarizationError as exc:
                # Report it, keep the transcript: a diarization failure costs
                # per-speaker labels, not the whole session's words.
                warnings.append(f"diarization skipped for {label}: {exc}")

        tracks[label] = segments
        say(f"  [{label}] {len(segments)} segment(s)")

    merged, stats = merge_segments(tracks, settings)

    meta = _read_meta(session_dir)
    markdown = render_markdown(
        merged,
        title=meta.get("title") or session_dir.name,
        started_at=meta.get("started_at"),
        duration_seconds=meta.get("duration_seconds"),
        stats=stats,
    )
    out_path = session_dir / "transcript.md"
    out_path.write_text(markdown, encoding="utf-8")

    return TranscriptResult(out_path, merged, stats, warnings)


def _read_meta(session_dir: Path) -> dict:
    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

"""Optional pyannote-audio speaker diarization.

Off by default (config: diarization.enabled). Channel labels already tell you
who is who at the track level - "Me" vs "Others" - so diarization only earns its
keep when several people share one track: a room full of people around one mic,
or a conference call where "Others" is four different voices.

When it runs, labels become "Others (Speaker 1)", "Others (Speaker 2)", ... A
track that turns out to hold a single speaker keeps its plain label.
"""

from __future__ import annotations

import os
from pathlib import Path

from .segments import Segment


class DiarizationError(RuntimeError):
    """Raised when diarization is requested but cannot run."""


def diarize_track(wav_path: Path, hf_token: str | None = None) -> list[tuple[float, float, str]]:
    """Return [(start, end, speaker_id), ...] turns for one WAV."""
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise DiarizationError(
            "diarization is enabled but HF_TOKEN is not set. Add it to .env, or set "
            "diarization.enabled: false in config.yaml."
        )
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise DiarizationError(
            'pyannote.audio is not installed. Install it with:\n  pip install -e ".[diarize]"'
        ) from exc

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )
        annotation = pipeline(str(wav_path))
    except Exception as exc:
        raise DiarizationError(f"diarization failed for {wav_path.name}: {exc}") from exc

    return [
        (float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def apply_diarization(
    segments: list[Segment], turns: list[tuple[float, float, str]], base_label: str
) -> None:
    """Refine speaker labels in place using diarization turns.

    Each segment takes the label of whichever diarized turn it overlaps most.
    If the whole track resolves to one speaker, the base label is left alone -
    "Others (Speaker 1)" is just noise when there is only ever one of them.
    """
    if not segments or not turns:
        return

    assignments: dict[int, str] = {}
    for index, seg in enumerate(segments):
        best_speaker, best_overlap = None, 0.0
        for start, end, speaker in turns:
            overlap = min(seg.end, end) - max(seg.start, start)
            if overlap > best_overlap:
                best_speaker, best_overlap = speaker, overlap
        if best_speaker is not None:
            assignments[index] = best_speaker

    distinct = sorted(set(assignments.values()))
    if len(distinct) < 2:
        return

    numbering = {speaker: n for n, speaker in enumerate(distinct, start=1)}
    for index, speaker in assignments.items():
        segments[index].speaker = f"{base_label} (Speaker {numbering[speaker]})"

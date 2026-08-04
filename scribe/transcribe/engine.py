"""faster-whisper wrapper: one WAV in, a list of Segments out.

The model is loaded once per process and reused across tracks - loading `medium`
twice would double both the wait and the memory for no benefit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .segments import Segment

_MODEL_CACHE: dict[tuple[str, str, str], object] = {}

ProgressFn = Callable[[str, float, float], None]  # (track label, seconds done, duration)


class TranscriptionError(RuntimeError):
    """Raised when the transcription backend is missing or fails."""


def load_model(model_size: str, device: str = "auto", compute_type: str = "auto"):
    """Load (and cache) a faster-whisper model."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise TranscriptionError(
            "faster-whisper is not installed. Install it with:\n"
            '  pip install -e ".[transcribe]"'
        ) from exc

    # faster-whisper spells these differently from our config vocabulary.
    ct2_compute = "default" if compute_type == "auto" else compute_type
    key = (model_size, device, ct2_compute)
    if key not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[key] = WhisperModel(model_size, device=device, compute_type=ct2_compute)
        except Exception as exc:  # model download / CUDA / compute-type failures
            raise TranscriptionError(
                f"could not load whisper model '{model_size}' "
                f"(device={device}, compute_type={ct2_compute}): {exc}"
            ) from exc
    return _MODEL_CACHE[key]


def transcribe_track(
    wav_path: Path,
    label: str,
    speaker: str,
    *,
    model_size: str = "medium",
    device: str = "auto",
    compute_type: str = "auto",
    language: str | None = None,
    vad_filter: bool = True,
    on_progress: ProgressFn | None = None,
) -> list[Segment]:
    """Transcribe one track. `label` is "mic"/"system", `speaker` the display name."""
    if not wav_path.exists():
        raise TranscriptionError(f"missing audio file: {wav_path}")

    model = load_model(model_size, device, compute_type)
    try:
        raw_segments, info = model.transcribe(
            str(wav_path),
            language=language,
            vad_filter=vad_filter,
            beam_size=5,
            word_timestamps=False,
        )
    except Exception as exc:
        raise TranscriptionError(f"transcription failed for {wav_path.name}: {exc}") from exc

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    segments: list[Segment] = []
    # raw_segments is a generator - work is done lazily as we iterate.
    for raw in raw_segments:
        text = (raw.text or "").strip()
        if not text:
            continue
        segments.append(
            Segment(
                track=label,
                speaker=speaker,
                start=float(raw.start),
                end=float(raw.end),
                text=text,
                avg_logprob=getattr(raw, "avg_logprob", None),
                no_speech_prob=getattr(raw, "no_speech_prob", None),
            )
        )
        if on_progress is not None:
            on_progress(label, float(raw.end), duration)
    return segments

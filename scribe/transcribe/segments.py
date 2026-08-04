"""Segment model plus per-track loudness envelopes.

The merge/dedup pass needs to know how loud each spoken segment was *relative to
its own track*. Absolute levels are useless for that comparison: a mic and a
WASAPI loopback have unrelated gain staging, so -30 dBFS on one means nothing
next to -30 dBFS on the other. What does compare cleanly is prominence - how far
a segment sits above that track's own median speech level. Real speech towers
over its track's median; audio bleeding in from the speakers sits well below it.
"""

from __future__ import annotations

import math
import statistics
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SILENCE_DBFS = -120.0  # floor reported for digital silence


@dataclass
class Segment:
    """One utterance from one track."""

    track: str  # "mic", "system", or "phone"
    speaker: str  # display label: "Me", "Others", "Others (Speaker 2)", ...
    start: float  # seconds from session start
    end: float
    text: str
    avg_logprob: float | None = None  # whisper confidence, higher (less negative) is better
    no_speech_prob: float | None = None  # whisper's own "this isn't speech" score
    # None means "level unknown" (the audio couldn't be measured), which is not
    # the same as "silent" - callers must not treat it as quiet, or every
    # segment of an unmeasurable file gets dropped by the silence gate.
    rms_dbfs: float | None = None
    prominence_db: float = 0.0  # rms_dbfs minus the track's median speech level
    dropped_by: str | None = None  # set when the merge pass discards this segment

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class LevelTrack:
    """Coarse RMS envelope of a WAV, one value per `frame_seconds` of audio.

    Built by streaming the file so a two-hour recording costs a few hundred KB
    of memory instead of the gigabyte the raw samples would take.
    """

    frame_seconds: float
    frames: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    @classmethod
    def from_file(cls, path: Path, frame_seconds: float = 0.1) -> "LevelTrack":
        """Measure any audio file the transcriber can read.

        Laptop recordings are 16-bit PCM WAV and go through the cheap `wave`
        reader. Phone imports (.m4a, .mp3, ...) are decoded with PyAV, which
        faster-whisper already depends on and which bundles its own FFmpeg.
        """
        if path.suffix.lower() == ".wav":
            try:
                return cls.from_wav(path, frame_seconds)
            except (wave.Error, EOFError, ValueError):
                pass  # e.g. a 24-bit or float WAV - let the decoder handle it
        return cls.from_decoded(path, frame_seconds)

    @classmethod
    def from_decoded(cls, path: Path, frame_seconds: float = 0.1) -> "LevelTrack":
        """Level envelope via PyAV, streamed so long files stay cheap."""
        import av  # provided by faster-whisper; only needed for non-WAV input

        rate = 16000
        block = max(1, int(rate * frame_seconds))
        full_scale = float(1 << 15)
        levels: list[float] = []
        carry = np.zeros(0, dtype=np.float32)

        def drain(samples: np.ndarray) -> None:
            nonlocal carry
            carry = np.concatenate([carry, samples]) if carry.size else samples
            usable = (carry.size // block) * block
            if usable:
                blocks = carry[:usable].reshape(-1, block).astype(np.float64)
                levels.extend(np.sqrt((blocks**2).mean(axis=1)) / full_scale)
                carry = carry[usable:]

        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=rate)
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError(f"{path.name}: no audio stream")
            stream = container.streams.audio[0]
            for frame in container.decode(stream):
                for chunk in resampler.resample(frame):
                    drain(chunk.to_ndarray().reshape(-1).astype(np.float32))
            for chunk in resampler.resample(None):  # flush
                drain(chunk.to_ndarray().reshape(-1).astype(np.float32))
        if carry.size:
            levels.append(float(np.sqrt((carry.astype(np.float64) ** 2).mean()) / full_scale))
        return cls(frame_seconds, np.asarray(levels, dtype=np.float32))

    @classmethod
    def from_wav(cls, path: Path, frame_seconds: float = 0.1) -> "LevelTrack":
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            if width != 2:
                raise ValueError(f"{path.name}: expected 16-bit PCM, got {width * 8}-bit")

            frames_per_block = max(1, int(rate * frame_seconds))
            full_scale = float(1 << 15)
            levels: list[float] = []
            while True:
                raw = wav.readframes(frames_per_block * 64)  # read in batches, score per block
                if not raw:
                    break
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1)
                usable = (samples.size // frames_per_block) * frames_per_block
                if usable:
                    blocks = samples[:usable].reshape(-1, frames_per_block)
                    levels.extend(np.sqrt((blocks**2).mean(axis=1)) / full_scale)
                tail = samples[usable:]
                if tail.size:
                    levels.append(float(np.sqrt((tail**2).mean()) / full_scale))
        return cls(frame_seconds, np.asarray(levels, dtype=np.float32))

    def dbfs(self, start: float, end: float) -> float:
        """RMS level over [start, end) in dBFS."""
        if self.frames.size == 0:
            return SILENCE_DBFS
        lo = max(0, int(start / self.frame_seconds))
        hi = min(self.frames.size, max(lo + 1, math.ceil(end / self.frame_seconds)))
        if lo >= self.frames.size:
            return SILENCE_DBFS
        window = self.frames[lo:hi]
        rms = float(np.sqrt((window.astype(np.float64) ** 2).mean()))
        if rms <= 0:
            return SILENCE_DBFS
        return max(SILENCE_DBFS, 20.0 * math.log10(rms))


def annotate_levels(segments: list[Segment], levels: LevelTrack | None) -> None:
    """Fill in rms_dbfs and prominence_db for one track's segments, in place.

    With no level data the segments keep rms_dbfs=None ("unknown") and a
    prominence of 0, which leaves them neutral to both the silence gate and the
    dedup ranking rather than looking like silence.
    """
    if not segments:
        return
    if levels is not None:
        for seg in segments:
            seg.rms_dbfs = levels.dbfs(seg.start, seg.end)

    audible = [s.rms_dbfs for s in segments if s.rms_dbfs is not None and s.rms_dbfs > SILENCE_DBFS]
    if not audible:
        return
    reference = statistics.median(audible)
    for seg in segments:
        if seg.rms_dbfs is not None:
            seg.prominence_db = seg.rms_dbfs - reference

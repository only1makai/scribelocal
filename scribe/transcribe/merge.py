"""Merge two transcribed tracks into one timeline, without double-printing speech.

Why this exists: the two tracks are not acoustically independent. The mic hears
whatever the speakers are playing, and if the output device is monitoring the
mic, the loopback hears you. Whisper faithfully transcribes both copies, so a
naive merge prints every sentence twice - once as "Me" and once as "Others".

Two passes fix it:

1. Silence gate - drop segments that sit near the noise floor, or that whisper
   itself flagged as probably-not-speech. These are the hallucinations whisper
   produces over near-silence ("Thank you.", "Thanks for watching!").
2. Cross-track dedup - find segments on opposite tracks that say near-enough the
   same thing at overlapping times, and keep only the copy from the source that
   heard it properly. "Properly" means loudest relative to its own track's
   median speech level, with whisper's confidence as the tiebreaker.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher

from .segments import Segment

# Containment matching ("is B just a piece of A?") needs enough text to be
# meaningful - otherwise every "yeah" in the session matches every other one.
MIN_CONTAINMENT_CHARS = 12


@dataclass
class DedupSettings:
    enabled: bool = True
    time_tolerance: float = 2.0  # seconds of start-time slop when pairing segments
    min_overlap: float = 0.3  # fraction of the shorter segment that must overlap
    similarity: float = 0.82  # 0-1 text similarity to call two segments the same speech
    silence_floor_dbfs: float = -50.0  # drop segments quieter than this
    max_no_speech: float = 0.6  # drop segments whisper flags as non-speech

    @classmethod
    def from_config(cls, cfg) -> "DedupSettings":
        d = cls()
        return cls(
            enabled=bool(cfg.get("merge.dedup", d.enabled)),
            time_tolerance=float(cfg.get("merge.time_tolerance", d.time_tolerance)),
            min_overlap=float(cfg.get("merge.min_overlap", d.min_overlap)),
            similarity=float(cfg.get("merge.similarity", d.similarity)),
            silence_floor_dbfs=float(cfg.get("merge.silence_floor_dbfs", d.silence_floor_dbfs)),
            max_no_speech=float(cfg.get("merge.max_no_speech", d.max_no_speech)),
        )


@dataclass
class MergeStats:
    kept: int = 0
    dropped_silence: int = 0
    dropped_duplicate: int = 0
    per_track_kept: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "kept": self.kept,
            "dropped_silence": self.dropped_silence,
            "dropped_duplicate": self.dropped_duplicate,
            "per_track_kept": self.per_track_kept,
        }


def normalize(text: str) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace.

    Two tracks rarely transcribe bleed-through identically - one hears
    "Yeah, that works." and the other "yeah that works". Comparing on a
    normalized form keeps those from looking like different sentences.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    lowered = stripped.casefold()
    cleaned = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def text_similarity(a: str, b: str) -> float:
    """0-1 similarity between two normalized strings."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    # Whisper splits utterances differently per track: one long segment on the
    # mic can arrive as two shorter ones on the loopback. A clean containment
    # is the same speech, even though the ratio would score it poorly.
    if len(short) >= MIN_CONTAINMENT_CHARS and short in long:
        return 1.0
    return SequenceMatcher(None, short, long).ratio()


def time_overlap(a: Segment, b: Segment) -> float:
    """Overlap as a fraction of the shorter segment (0-1)."""
    overlap = min(a.end, b.end) - max(a.start, b.start)
    if overlap <= 0:
        return 0.0
    shortest = min(a.duration, b.duration)
    return 1.0 if shortest <= 0 else min(1.0, overlap / shortest)


def _quality(seg: Segment) -> tuple[float, float, int]:
    """Ranking key for "which copy of this speech is the real one".

    Prominence first (bleed-through is quiet relative to its own track's
    speech), then whisper's confidence, then length as a final tiebreak.
    """
    return (
        round(seg.prominence_db, 1),
        seg.avg_logprob if seg.avg_logprob is not None else -99.0,
        len(seg.text),
    )


def apply_silence_gate(segments: list[Segment], settings: DedupSettings) -> int:
    """Mark near-silent / non-speech segments as dropped. Returns the count."""
    dropped = 0
    for seg in segments:
        if seg.dropped_by:
            continue
        # An unmeasured level is unknown, not silent - never gate on it.
        too_quiet = (
            seg.rms_dbfs is not None and seg.rms_dbfs < settings.silence_floor_dbfs
        )
        not_speech = (
            seg.no_speech_prob is not None and seg.no_speech_prob > settings.max_no_speech
        )
        if too_quiet or not_speech:
            seg.dropped_by = "silence" if too_quiet else "no-speech"
            dropped += 1
    return dropped


def dedup_across_tracks(segments: list[Segment], settings: DedupSettings) -> int:
    """Drop cross-track duplicates, keeping the louder/cleaner copy.

    Pairs are scored first and resolved best-match-first, so a segment that
    looks like a weak match for one neighbour and a strong match for another is
    settled by the strong pairing.
    """
    live = sorted((s for s in segments if not s.dropped_by), key=lambda s: s.start)
    normalized = [normalize(s.text) for s in live]
    candidates: list[tuple[float, int, int]] = []

    for i, a in enumerate(live):
        # Segments are start-ordered, so once b starts later than a's end plus
        # the tolerance window, nothing further can pair with a.
        horizon = a.end + settings.time_tolerance
        for j in range(i + 1, len(live)):
            b = live[j]
            if b.start > horizon:
                break
            if a.track == b.track:
                continue  # one source can't talk over itself
            starts_together = abs(a.start - b.start) <= settings.time_tolerance
            if not starts_together and time_overlap(a, b) < max(settings.min_overlap, 1e-9):
                continue
            score = text_similarity(normalized[i], normalized[j])
            if score >= settings.similarity:
                candidates.append((score, i, j))

    dropped = 0
    for _score, i, j in sorted(candidates, key=lambda c: -c[0]):
        a, b = live[i], live[j]
        if a.dropped_by or b.dropped_by:
            continue  # already resolved by a stronger pairing
        loser = b if _quality(a) >= _quality(b) else a
        loser.dropped_by = "duplicate"
        dropped += 1
    return dropped


def merge_segments(
    tracks: dict[str, list[Segment]], settings: DedupSettings
) -> tuple[list[Segment], MergeStats]:
    """Combine per-track segments into one chronological transcript."""
    all_segments: list[Segment] = []
    for track_segments in tracks.values():
        all_segments.extend(track_segments)
    all_segments.sort(key=lambda s: (s.start, s.track))

    stats = MergeStats()
    if settings.enabled:
        stats.dropped_silence = apply_silence_gate(all_segments, settings)
        if len(tracks) > 1:
            stats.dropped_duplicate = dedup_across_tracks(all_segments, settings)

    kept = [s for s in all_segments if not s.dropped_by]
    stats.kept = len(kept)
    for seg in kept:
        stats.per_track_kept[seg.track] = stats.per_track_kept.get(seg.track, 0) + 1
    return kept, stats


def format_timestamp(seconds: float) -> str:
    total = int(max(0.0, seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def render_markdown(
    segments: list[Segment],
    *,
    title: str,
    started_at: str | None = None,
    duration_seconds: float | None = None,
    stats: MergeStats | None = None,
) -> str:
    """Render the merged timeline as transcript.md."""
    lines = [f"# {title}", ""]

    meta_bits = []
    if started_at:
        meta_bits.append(_pretty_timestamp(started_at))
    if duration_seconds:
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        meta_bits.append(f"{minutes} min {seconds:02d} s")
    if stats:
        tracks = ", ".join(f"{name}: {n}" for name, n in sorted(stats.per_track_kept.items()))
        if tracks:
            meta_bits.append(f"segments — {tracks}")
    if meta_bits:
        lines += ["*" + " · ".join(meta_bits) + "*", ""]

    if stats and (stats.dropped_duplicate or stats.dropped_silence):
        lines += [
            f"*Merge pass removed {stats.dropped_duplicate} cross-track duplicate(s) "
            f"and {stats.dropped_silence} silent/non-speech segment(s).*",
            "",
        ]

    lines += ["## Transcript", ""]
    if not segments:
        lines += ["*(no speech detected)*", ""]
        return "\n".join(lines)

    # Collapse consecutive turns from the same speaker into one paragraph so the
    # transcript reads like a conversation rather than a log of 4-second chunks.
    current_speaker: str | None = None
    buffer: list[str] = []
    block_start = 0.0

    def flush() -> None:
        if buffer:
            lines.append(f"**[{format_timestamp(block_start)}] {current_speaker}:** " + " ".join(buffer))
            lines.append("")

    for seg in segments:
        if seg.speaker != current_speaker:
            flush()
            current_speaker = seg.speaker
            block_start = seg.start
            buffer = []
        buffer.append(seg.text)
    flush()
    return "\n".join(lines)


def _pretty_timestamp(iso: str) -> str:
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M")

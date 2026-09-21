"""Real-time transcription tap for live sessions.

Reuses DualRecorder completely unmodified for on-disk capture -
audio_mic.wav/audio_system.wav are written exactly as in record mode, so the
full WAVs still exist for the record-quality transcript.md pass at Stop. This
module does not touch scribe/audio/recorder.py at all. Instead it taps a COPY
of the same PCM buffers the recorder's own writer threads are already
draining to disk: Track.buffers is a plain queue.Queue that recorder.py
already exposes as a public field, and this module wraps its .put() so every
chunk handed to the writer thread is also forwarded to a live-transcription
worker, before calling the original .put() unchanged, with the exact same
arguments, in the same call. The writer thread's behavior - what it receives,
in what order, with what timing - is therefore identical to record mode:
nothing here can make a WAV frame arrive late, get duplicated, or get
skipped, because the original .put() call happens exactly as it always did.

Two transcript qualities exist per live session, matching the design:
  - The rolling one this module produces, from a smaller/faster model
    (live.model, default "small") on short overlapping windows. This is for
    responsiveness in the live view and is NEVER written to transcript.md.
  - The full-quality one, produced by the existing, untouched
    scribe.transcribe.transcribe_session() at Stop, exactly like record mode.
    That remains the sole source of truth for transcript.md.

The windowing/backpressure math (WindowAccumulator, BackpressureTracker) is
kept pure - no threads, no whisper, no wall-clock reads inside those classes
- so it's unit-testable the same way scribe.transcribe.merge's dedup logic
is: as plain input/output, without real audio devices or a real model.
_TrackTap is the only piece that touches real queues/threads/whisper/timing,
matching this codebase's existing split between tested pure logic
(merge.py, segments.py) and untested I/O glue (engine.py, transcribe/__init__.py).
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from ..audio.recorder import DualRecorder, Track
from ..transcribe.engine import load_model

Notify = Callable[[str], None]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class LiveSettings:
    window_seconds: float = 8.0
    overlap_seconds: float = 1.0
    # "base" rather than "small": both tracks are transcribed concurrently but
    # share one model, and faster-whisper serializes concurrent calls, so live
    # mode's real cost is roughly double a single track's. Measured on 12 cores
    # with 8s windows and both tracks running: base 0.55x real time (keeps up),
    # small 1.12x (falls behind, then auto-widens to recover).
    model: str = "base"
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = None
    # Hard ceiling on un-transcribed audio held in memory per track. Widening
    # windows is the first defence when whisper falls behind, but it isn't a
    # guarantee - on a slow machine, or with a big model, the backlog can still
    # grow faster than it drains. This bounds that: past this many seconds, the
    # OLDEST audio is dropped from the live pass. The recording itself is
    # untouched (the WAVs are written by the recorder's own thread), so a drop
    # costs live-view text, never the final transcript.
    max_backlog_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.overlap_seconds >= self.window_seconds:
            raise ValueError(
                f"live.overlap_seconds ({self.overlap_seconds}) must be less than "
                f"live.window_seconds ({self.window_seconds})"
            )

    @classmethod
    def from_config(cls, cfg) -> "LiveSettings":
        d = cls()
        return cls(
            window_seconds=float(cfg.get("live.window_seconds", d.window_seconds)),
            overlap_seconds=float(cfg.get("live.overlap_seconds", d.overlap_seconds)),
            model=str(cfg.get("live.model", d.model)),
            max_backlog_seconds=float(
                cfg.get("live.max_backlog_seconds", d.max_backlog_seconds)
            ),
            # Device/compute_type are shared with the main whisper settings
            # rather than duplicated - "auto" behaves the same for a small
            # model as a medium one, and giving live its own copy would just
            # be one more place these could quietly drift apart.
            device=str(cfg.get("whisper.device", d.device)),
            compute_type=str(cfg.get("whisper.compute_type", d.compute_type)),
            language=cfg.get("whisper.language"),
        )


# ---------------------------------------------------------------------------
# Pure windowing (no threads, no whisper, no wall-clock - unit-testable)
# ---------------------------------------------------------------------------


@dataclass
class Window:
    """One window of audio ready to hand to whisper."""

    start: float  # seconds since this track's tap began
    end: float
    audio: np.ndarray  # float32 mono @ 16kHz, range [-1, 1]
    # How many windows became ready in the SAME pop_ready() call as this one.
    # >1 means audio arrived faster than windows were drained between two
    # pop_ready() calls - the caller fell behind, almost certainly stuck
    # inside a whisper.transcribe() call for longer than one window's worth
    # of real time. See BackpressureTracker for what a caller does with this.
    backlog: int = 1


class WindowAccumulator:
    """Turns a stream of mono 16kHz int16 PCM chunks into overlapping,
    whisper-ready float32 windows.

    Pure and synchronous: feed() only appends samples, pop_ready() only
    slices what's already buffered. No timing assumption of any kind - the
    caller decides when to call pop_ready(), and backlog on the returned
    Windows reflects only how many became ready between two such calls.
    """

    def __init__(self, sample_rate: int, window_seconds: float, overlap_seconds: float):
        if overlap_seconds >= window_seconds:
            raise ValueError("overlap_seconds must be less than window_seconds")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.overlap_seconds = overlap_seconds
        self._carry = np.zeros(0, dtype=np.int16)
        self._elapsed = 0.0  # seconds already emitted into windows

    def _window_frames(self) -> int:
        return max(1, int(round(self.window_seconds * self.sample_rate)))

    def _hop_frames(self) -> int:
        hop_seconds = self.window_seconds - self.overlap_seconds
        return max(1, int(round(hop_seconds * self.sample_rate)))

    def feed(self, pcm_int16_mono: bytes | np.ndarray) -> None:
        """Append mono int16 PCM already at this accumulator's sample_rate."""
        if isinstance(pcm_int16_mono, (bytes, bytearray)):
            samples = np.frombuffer(pcm_int16_mono, dtype=np.int16)
        else:
            samples = np.asarray(pcm_int16_mono, dtype=np.int16)
        if samples.size == 0:
            return
        self._carry = (
            np.concatenate([self._carry, samples]) if self._carry.size else samples.copy()
        )

    def pop_ready(self) -> list[Window]:
        """Return zero or more windows now fully buffered, oldest first."""
        window_frames = self._window_frames()
        hop_frames = self._hop_frames()
        out: list[Window] = []
        while self._carry.size >= window_frames:
            win_samples = self._carry[:window_frames]
            audio = win_samples.astype(np.float32) / 32768.0
            out.append(
                Window(
                    start=self._elapsed,
                    end=self._elapsed + window_frames / self.sample_rate,
                    audio=audio,
                )
            )
            self._carry = self._carry[hop_frames:]
            self._elapsed += hop_frames / self.sample_rate
        for w in out:
            w.backlog = len(out)
        return out

    def flush(self) -> Window | None:
        """Emit whatever's left (shorter than a full window), for use at
        Stop so the last few seconds of speech aren't silently dropped."""
        if self._carry.size == 0:
            return None
        audio = self._carry.astype(np.float32) / 32768.0
        w = Window(
            start=self._elapsed,
            end=self._elapsed + self._carry.size / self.sample_rate,
            audio=audio,
            backlog=1,
        )
        self._carry = np.zeros(0, dtype=np.int16)
        return w

    def widen(self, factor: float = 1.5, cap: float | None = None) -> None:
        """Grow window_seconds (and shrink overlap proportionally) so future
        windows cost fewer whisper calls per second of audio. Used when
        whisper can't keep up - see BackpressureTracker."""
        new_window = self.window_seconds * factor
        if cap is not None:
            new_window = min(new_window, cap)
        self.window_seconds = new_window
        self.overlap_seconds = min(self.overlap_seconds, new_window * 0.4)

    def narrow(self, factor: float, floor_window: float, floor_overlap: float) -> None:
        """Shrink back toward the originally configured window/overlap once
        whisper has comfortably caught up. Never goes below the floor - this
        only undoes a previous widen(), it doesn't tune below what was asked
        for in config."""
        self.window_seconds = max(self.window_seconds * factor, floor_window)
        if self.window_seconds <= floor_window + 1e-9:
            self.overlap_seconds = floor_overlap
        else:
            self.overlap_seconds = min(self.overlap_seconds, self.window_seconds * 0.4)


@dataclass
class BackpressureTracker:
    """Decides when a track's window should widen or narrow, from a stream
    of real-time-factor (RTF) observations: how long whisper took to
    transcribe a window, divided by how many seconds of audio that window
    covered. RTF > 1 means whisper is losing ground against real time.

    Pure decision policy - no timing, no threading, no whisper of its own -
    so the thresholds are unit-testable as a plain sequence of inputs and
    outputs, the same way WindowAccumulator's slicing is.
    """

    widen_after: int = 2  # consecutive RTF>1.0 windows before widening
    narrow_after: int = 5  # consecutive comfortably-caught-up windows before narrowing
    recover_below: float = 0.6  # RTF must drop below this to count toward narrowing
    _behind: int = field(default=0, repr=False)
    _caught_up: int = field(default=0, repr=False)
    widened: bool = field(default=False, repr=False)

    def observe(self, rtf: float) -> str | None:
        """Feed one window's RTF; returns "widen", "narrow", or None."""
        if rtf > 1.0:
            self._behind += 1
            self._caught_up = 0
            if self._behind >= self.widen_after:
                self._behind = 0
                self.widened = True
                return "widen"
            return None

        self._behind = 0
        if self.widened and rtf < self.recover_below:
            self._caught_up += 1
            if self._caught_up >= self.narrow_after:
                self._caught_up = 0
                self.widened = False
                return "narrow"
        else:
            self._caught_up = 0
        return None


# ---------------------------------------------------------------------------
# Tapping the recorder without modifying it
# ---------------------------------------------------------------------------


def tee_track_buffers(track: Track, sink: Callable[[bytes], None]) -> Callable[[], None]:
    """Wrap track.buffers.put so every chunk handed to the disk-writer
    thread is also forwarded to `sink`, before the original .put() runs
    unchanged with the same arguments. Returns a restore() function.

    This wraps the bound .put method on the ONE existing queue.Queue
    instance track.buffers already is - it does not replace track.buffers
    with a different object, and it does not touch anything in
    scribe/audio/recorder.py. queue.Queue has no __slots__ restricting
    instance attributes, so this is a supported, if unusual, use of a
    perfectly ordinary Python object.
    """
    original_put = track.buffers.put

    def tee_put(item, *args, **kwargs):
        if item is not None:  # None is the writer thread's own stop sentinel
            try:
                sink(item)
            except Exception:
                pass  # a live-tap failure must never block or break the disk write
        return original_put(item, *args, **kwargs)

    track.buffers.put = tee_put

    def restore() -> None:
        track.buffers.put = original_put

    return restore


# ---------------------------------------------------------------------------
# Live chunks
# ---------------------------------------------------------------------------

TRACK_SPEAKERS = {"mic": "Me", "system": "Others"}


@dataclass
class LiveChunk:
    """One incremental piece of live transcript, emitted as it's produced."""

    track: str  # "mic" | "system"
    speaker: str  # "Me" | "Others"
    window_start: float  # seconds since this track's tap began
    window_end: float
    text: str
    lagging: bool = False  # true once this track has been auto-widened

    def as_dict(self) -> dict:
        return {
            "track": self.track,
            "speaker": self.speaker,
            "window_start": round(self.window_start, 2),
            "window_end": round(self.window_end, 2),
            "text": self.text,
            "lagging": self.lagging,
        }


OnChunk = Callable[[LiveChunk], None]
OnWarning = Callable[[str, str], None]  # (track_label, message)


# ---------------------------------------------------------------------------
# Threaded per-track worker (I/O glue - not unit-tested directly, same as
# scribe.transcribe.engine's whisper-calling code)
# ---------------------------------------------------------------------------


class _TrackTap:
    """Owns one track's raw-audio queue, resampling, windowing, and whisper
    calls, on a dedicated worker thread."""

    def __init__(
        self,
        track: Track,
        settings: LiveSettings,
        model,
        on_chunk: OnChunk,
        on_warning: OnWarning,
    ):
        self.track = track
        self.settings = settings
        self.model = model
        self.on_chunk = on_chunk
        self.on_warning = on_warning
        self.channels = min(2, track.device.channels)
        self.rate = track.device.sample_rate
        self._queue: "queue.Queue[bytes | None]" = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name=f"live-{track.label}", daemon=True
        )
        self.window_count = 0
        self.chunk_count = 0
        # Backlog accounting. bytes_per_second assumes int16 (2 bytes), which is
        # what recorder.py opens every stream with (SAMPLE_WIDTH = 2).
        self._bytes_per_second = max(1, self.rate * self.channels * 2)
        self._max_backlog_bytes = int(settings.max_backlog_seconds * self._bytes_per_second)
        self._backlog_lock = threading.Lock()
        self._pending_bytes = 0
        self._dropped_bytes = 0  # since the last warning
        self.dropped_seconds = 0.0  # cumulative, for the session stats
        self._last_drop_warning = 0.0

    def feed(self, chunk: bytes) -> None:
        """Accept one raw PCM buffer from the recorder's callback thread.

        This runs on PyAudio's callback thread, so it must never block: a
        bounded queue.Queue is not an option here, because its put() would
        block once full and stall audio capture itself. Instead the queue stays
        unbounded and is trimmed from the front, which bounds memory without
        ever making the caller wait.
        """
        with self._backlog_lock:
            self._pending_bytes += len(chunk)
        self._queue.put(chunk)
        self._trim_backlog()

    def _trim_backlog(self) -> None:
        """Drop the OLDEST queued audio until the backlog is under the cap.

        Oldest-first because a live transcript is only useful if it's current;
        when whisper can't keep up, the recent audio is the audio worth having.
        """
        while True:
            with self._backlog_lock:
                if self._pending_bytes <= self._max_backlog_bytes:
                    return
            try:
                oldest = self._queue.get_nowait()
            except queue.Empty:
                return
            if oldest is None:
                self._queue.put(None)  # never swallow the worker's stop sentinel
                return
            with self._backlog_lock:
                self._pending_bytes -= len(oldest)
                self._dropped_bytes += len(oldest)
                self.dropped_seconds += len(oldest) / self._bytes_per_second

    def _report_drops(self) -> None:
        """Warn about dropped audio, rate-limited, from the WORKER thread.

        Deliberately not called from feed(): that runs on the audio callback
        thread, and on_warning ultimately does console or socket I/O.
        """
        with self._backlog_lock:
            dropped = self._dropped_bytes
            if not dropped:
                return
            now = time.monotonic()
            if now - self._last_drop_warning < 10.0:
                return
            self._dropped_bytes = 0
            self._last_drop_warning = now
            total = self.dropped_seconds
        self.on_warning(
            self.track.label,
            f"live transcription is too far behind - dropped "
            f"{dropped / self._bytes_per_second:.0f}s of audio from the live view "
            f"({total:.0f}s total). The recording itself is unaffected; "
            f"transcript.md at Stop still covers everything. "
            f"Consider a smaller live.model.",
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30)

    def _transcribe(self, audio: np.ndarray) -> str:
        raw_segments, _info = self.model.transcribe(
            audio,
            language=self.settings.language,
            vad_filter=True,
            # Greedy decoding (vs. engine.py's beam_size=5 for the final
            # pass): this is the "cheap" rolling transcript, and beam search
            # is one of the more expensive knobs to turn up on CPU. Accuracy
            # here is explicitly secondary to keeping up in real time -
            # transcript.md at Stop still gets the full beam_size=5 pass.
            beam_size=1,
            word_timestamps=False,
        )
        parts = [(seg.text or "").strip() for seg in raw_segments]
        return " ".join(p for p in parts if p).strip()

    def _run(self) -> None:
        try:
            import av
        except ImportError as exc:
            self.on_warning(
                self.track.label,
                f"live transcription unavailable: PyAV not installed ({exc}). "
                'Install it with:  pip install -e ".[transcribe]"',
            )
            # Drain the queue without processing so the backlog accounting
            # stays balanced (and never reports phantom drops) while the
            # recorder's own capture carries on unaffected.
            while True:
                item = self._queue.get()
                if item is None:
                    return
                with self._backlog_lock:
                    self._pending_bytes -= len(item)
                    self._dropped_bytes = 0

        # input_layout describes the SOURCE frame (matches what this track was
        # actually opened with - mono mic, or stereo system/loopback).
        # resampler is ALWAYS built to OUTPUT "mono": whisper wants one
        # channel, and PyAV's resampler does the stereo->mono downmix and the
        # sample-rate conversion in a single pass when its target layout is
        # "mono" regardless of the input layout. These must not be the same
        # variable - an earlier version of this file used one `layout` for
        # both, which silently built a resampler that output STEREO for any
        # 2-channel track: feed() then received twice the sample count per
        # real second of audio, windows closed in half the intended real
        # time, and every "mono" sample it fed to whisper was actually an
        # interleaved L/R pair read as if it were one channel. Caught before
        # the real-hardware test by a smoke run against the cached "tiny"
        # model: the system (stereo) track produced ~2x the window count of
        # the mic (mono) track for the same input duration - the signature
        # of doubled output rate. Fixed by keeping the two layouts separate.
        input_layout = "mono" if self.channels == 1 else "stereo"
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        acc = WindowAccumulator(16000, self.settings.window_seconds, self.settings.overlap_seconds)
        floor_window = self.settings.window_seconds
        floor_overlap = self.settings.overlap_seconds
        # Never widen past 4x the configured window (capped at 60s) - "drop
        # to longer windows... rather than falling behind unboundedly" means
        # there must be a ceiling, not that windows can grow forever.
        widen_cap = min(floor_window * 4.0, 60.0)
        backpressure = BackpressureTracker()

        while True:
            item = self._queue.get()
            if item is None:
                break
            with self._backlog_lock:
                self._pending_bytes -= len(item)
            self.chunk_count += 1
            self._report_drops()

            try:
                samples = np.frombuffer(item, dtype=np.int16)
                frame = av.AudioFrame.from_ndarray(
                    samples.reshape(1, -1), format="s16", layout=input_layout
                )
                frame.sample_rate = self.rate
                for resampled in resampler.resample(frame):
                    acc.feed(resampled.to_ndarray().reshape(-1))
            except Exception as exc:  # a bad buffer must not kill the tap
                self.on_warning(self.track.label, f"resample error: {exc}")
                continue

            for window in acc.pop_ready():
                self.window_count += 1
                t0 = time.monotonic()
                try:
                    text = self._transcribe(window.audio)
                except Exception as exc:  # whisper failing must not kill the tap
                    self.on_warning(self.track.label, f"transcription error: {exc}")
                    continue
                infer_seconds = time.monotonic() - t0

                if text:
                    self.on_chunk(
                        LiveChunk(
                            track=self.track.label,
                            speaker=TRACK_SPEAKERS.get(self.track.label, self.track.label),
                            window_start=window.start,
                            window_end=window.end,
                            text=text,
                            lagging=backpressure.widened,
                        )
                    )

                window_seconds = max(window.end - window.start, 1e-6)
                rtf = infer_seconds / window_seconds
                decision = backpressure.observe(rtf)
                if decision == "widen":
                    acc.widen(cap=widen_cap)
                    self.on_warning(
                        self.track.label,
                        f"falling behind (took {infer_seconds:.1f}s to process "
                        f"{window_seconds:.1f}s of audio) - widening windows to "
                        f"{acc.window_seconds:.1f}s",
                    )
                elif decision == "narrow":
                    acc.narrow(0.75, floor_window, floor_overlap)
                    self.on_warning(
                        self.track.label,
                        f"caught up - window size back to {acc.window_seconds:.1f}s",
                    )

        tail = acc.flush()
        if tail is not None:
            try:
                text = self._transcribe(tail.audio)
            except Exception as exc:
                self.on_warning(self.track.label, f"transcription error: {exc}")
                text = ""
            if text:
                self.window_count += 1
                self.on_chunk(
                    LiveChunk(
                        track=self.track.label,
                        speaker=TRACK_SPEAKERS.get(self.track.label, self.track.label),
                        window_start=tail.start,
                        window_end=tail.end,
                        text=text,
                        lagging=backpressure.widened,
                    )
                )


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


class LiveTranscriber:
    """Attaches to a DualRecorder's tracks and produces a live, lower-
    fidelity transcript while the session runs, writing every chunk to
    <session_dir>/live_transcript.jsonl for later review (per
    07_BUILD_SEQUENCE.md's requirement that the live log be saved to the
    session folder, not just shown on screen and discarded).

    Does not write transcript.md and never will - that file is owned
    exclusively by scribe.transcribe.transcribe_session() at Stop.
    """

    def __init__(
        self,
        session_dir: Path,
        settings: LiveSettings,
        *,
        on_chunk: OnChunk | None = None,
        on_warning: OnWarning | None = None,
    ):
        self.session_dir = session_dir
        self.settings = settings
        self._on_chunk = on_chunk or (lambda _c: None)
        self._on_warning = on_warning or (lambda _t, _m: None)
        self._taps: list[_TrackTap] = []
        self._restores: list[Callable[[], None]] = []
        self._log_path = session_dir / "live_transcript.jsonl"
        self._log_lock = threading.Lock()
        self._attached = False

    def _log(self, chunk: LiveChunk) -> None:
        record = chunk.as_dict()
        record["logged_at"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(record)
        with self._log_lock:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def _handle_chunk(self, chunk: LiveChunk) -> None:
        self._log(chunk)
        self._on_chunk(chunk)

    def attach(self, recorder: DualRecorder) -> None:
        """Load the live model once (shared across tracks, matching
        engine.load_model's own "load once per process" principle) and tap
        every track. Must be called BEFORE recorder.start() so no audio is
        captured before the tap is in place - though because recorder.py's
        callback looks up track.buffers.put freshly on every call rather
        than capturing a reference ahead of time, attaching after start()
        would still work; before is simply the version with zero possible
        gap at the very beginning of the recording."""
        if self._attached:
            raise RuntimeError("LiveTranscriber.attach() called twice")
        model = load_model(self.settings.model, self.settings.device, self.settings.compute_type)
        for track in recorder.tracks:
            tap = _TrackTap(track, self.settings, model, self._handle_chunk, self._on_warning)
            restore = tee_track_buffers(track, tap.feed)
            self._taps.append(tap)
            self._restores.append(restore)
        for tap in self._taps:
            tap.start()
        self._attached = True

    def stop(self) -> dict:
        """Stop every tap and restore the recorder's queues untouched.

        Call this AFTER recorder.stop(), not before: recorder.stop() joins
        the writer threads, which guarantees every .put() (and therefore
        every tee) has already fired, so the taps see every last buffer the
        recorder captured, including the final partial one, before this
        drains and flushes them.
        """
        for restore in self._restores:
            restore()
        for tap in self._taps:
            tap.stop()
        return {
            tap.track.label: {
                "windows": tap.window_count,
                "chunks_fed": tap.chunk_count,
                # >0 means the live pass couldn't keep up and shed old audio.
                # Recorded WAVs and transcript.md are unaffected either way.
                "dropped_seconds": round(tap.dropped_seconds, 1),
            }
            for tap in self._taps
        }

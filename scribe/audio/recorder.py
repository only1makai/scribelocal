"""Dual-track recorder: microphone + WASAPI loopback, chunked WAV writing.

Each track runs its own PyAudio callback stream. Callbacks push raw buffers
onto a queue; a writer thread drains the queue to a wave file, so audio is
never held in memory beyond a few buffers (safe for 1-2 hr sessions).
"""

from __future__ import annotations

import queue
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio

from .devices import DeviceInfo, get_default_loopback, get_default_mic

SAMPLE_FORMAT = pyaudio.paInt16
SAMPLE_WIDTH = 2  # bytes, matches paInt16


@dataclass
class Track:
    label: str  # "mic" or "system"
    device: DeviceInfo
    path: Path
    stream: pyaudio.Stream | None = None
    wav: wave.Wave_write | None = None
    buffers: "queue.Queue[bytes | None]" = field(default_factory=queue.Queue)
    writer: threading.Thread | None = None
    frames_written: int = 0
    peak: float = 0.0  # 0..1, decays; read by UI/CLI for level meters

    def level(self) -> float:
        return self.peak


class DualRecorder:
    """Records mic and system audio to <session_dir>/audio_mic.wav and
    audio_system.wav. If no loopback device is found, records mic only and
    sets .warning."""

    def __init__(
        self,
        session_dir: Path,
        mic_device: int | None = None,
        loopback_device: int | None = None,
        chunk_frames: int = 4096,
    ):
        self.session_dir = session_dir
        self.chunk_frames = chunk_frames
        self.warning: str | None = None
        self.started_at: float | None = None
        self._p = pyaudio.PyAudio()
        self._keepalive: pyaudio.Stream | None = None
        self.tracks: list[Track] = []

        mic = get_default_mic(self._p, mic_device)
        self.tracks.append(Track("mic", mic, session_dir / "audio_mic.wav"))

        loopback = get_default_loopback(self._p, loopback_device)
        if loopback is None:
            self.warning = (
                "No WASAPI loopback device found - recording microphone only. "
                "The other side of calls will NOT be captured."
            )
        else:
            self.tracks.append(Track("system", loopback, session_dir / "audio_system.wav"))

    def start(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if any(t.label == "system" for t in self.tracks):
            self._start_keepalive()
        for track in self.tracks:
            self._start_track(track)
        self.started_at = time.time()

    def _start_keepalive(self) -> None:
        """Play continuous silence to the default output.

        WASAPI loopback only delivers packets while the endpoint is rendering;
        without this, the system track would start late and skip silent gaps,
        drifting out of sync with the mic track.
        """

        def silence(in_data, frame_count, time_info, status):
            return (b"\x00" * frame_count * SAMPLE_WIDTH * 2, pyaudio.paContinue)

        try:
            out = self._p.get_default_output_device_info()
            self._keepalive = self._p.open(
                format=SAMPLE_FORMAT,
                channels=2,
                rate=int(out["defaultSampleRate"]),
                output=True,
                output_device_index=out["index"],
                frames_per_buffer=self.chunk_frames,
                stream_callback=silence,
            )
        except OSError:
            self.warning = (
                (self.warning + " " if self.warning else "")
                + "Could not start silence keepalive - system track may drop silent gaps."
            )

    def _start_track(self, track: Track) -> None:
        channels = min(2, track.device.channels)
        wav = wave.open(str(track.path), "wb")
        wav.setnchannels(channels)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(track.device.sample_rate)
        track.wav = wav

        def writer() -> None:
            while True:
                buf = track.buffers.get()
                if buf is None:
                    break
                wav.writeframes(buf)  # appends to disk; nothing accumulates in RAM
                track.frames_written += len(buf) // (SAMPLE_WIDTH * channels)

        track.writer = threading.Thread(target=writer, name=f"writer-{track.label}", daemon=True)
        track.writer.start()

        def callback(in_data, frame_count, time_info, status):
            track.buffers.put(in_data)
            if in_data:
                samples = np.frombuffer(in_data, dtype=np.int16)
                peak = float(np.abs(samples).max()) / 32768 if samples.size else 0.0
            else:
                peak = 0.0
            track.peak = max(peak, track.peak * 0.8)  # decay for a readable meter
            return (None, pyaudio.paContinue)

        track.stream = self._p.open(
            format=SAMPLE_FORMAT,
            channels=channels,
            rate=track.device.sample_rate,
            input=True,
            input_device_index=track.device.index,
            frames_per_buffer=self.chunk_frames,
            stream_callback=callback,
        )

    def elapsed(self) -> float:
        return time.time() - self.started_at if self.started_at else 0.0

    def stop(self) -> dict:
        """Stop streams, flush writers, close files. Returns per-track stats."""
        for track in self.tracks:
            if track.stream is not None:
                track.stream.stop_stream()
                track.stream.close()
        if self._keepalive is not None:
            self._keepalive.stop_stream()
            self._keepalive.close()
        for track in self.tracks:
            track.buffers.put(None)  # sentinel: drain queue, then exit
            if track.writer is not None:
                track.writer.join(timeout=10)
            if track.wav is not None:
                track.wav.close()
        self._p.terminate()
        return {
            track.label: {
                "path": str(track.path),
                "device": track.device.name,
                "sample_rate": track.device.sample_rate,
                "seconds": round(track.frames_written / track.device.sample_rate, 2),
            }
            for track in self.tracks
        }

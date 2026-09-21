"""Render the demo sales call to audio with the Windows SAPI voices.

    python demo/make_demo_audio.py

Writes three 16 kHz mono 16-bit WAVs into demo/audio/:

    sales_call_playback.wav   both sides, one file - play this and record it
    sales_call_mic.wav        Dana loud, Priya bled in faint
    sales_call_system.wav     Priya loud, Dana bled in faint

The lines and their timings come from tests/fixtures/sales_call.py, which is the
same fixture the merge/dedup tests run on - so the demo recording and the test
suite are describing one conversation rather than two that drifted apart.

The two track files are a shortcut for iterating on the pipeline: drop them into
a session folder as audio_mic.wav / audio_system.wav and run `scribe process`.
They are not a substitute for the real demo, where you play the playback file
into the room and record it, because only a real capture produces real bleed,
real room noise, and honest merge stats.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from fixtures import sales_call  # noqa: E402

RATE = 16000  # what whisper resamples to anyway
DEAD_AIR_SECONDS = 6.0  # after the last turn, for the silence gate to chew on
MIN_GAP_SECONDS = 0.55  # silence between turns, so whisper segments on the turn
NOISE_FLOOR_DBFS = -66.0  # room tone; must stay under merge.silence_floor_dbfs
MAX_SAPI_RATE = 8  # SAPI tops out at 10; past 8 it stops sounding like speech

DANA_VOICE = "Microsoft David Desktop"
PRIYA_VOICE = "Microsoft Zira Desktop"

# How far under real speech the bleed sits, taken from the fixture so the audio
# and the segment levels agree on what "faint" means.
MIC_BLEED_DB = sales_call.BLEED_ON_MIC - sales_call.LOUD_MIC
SYSTEM_BLEED_DB = sales_call.BLEED_ON_SYSTEM - sales_call.LOUD_SYSTEM

SYNTH_PS1 = r"""
param([string]$Manifest)
Add-Type -AssemblyName System.Speech
$clips = Get-Content -Path $Manifest -Encoding UTF8 -Raw | ConvertFrom-Json
foreach ($clip in $clips) {
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $synth.SelectVoice($clip.voice)
    $synth.Rate = [int]$clip.rate
    $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
        16000,
        [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
        [System.Speech.AudioFormat.AudioChannel]::Mono)
    $synth.SetOutputToWaveFile($clip.path, $format)
    $synth.Speak($clip.text)
    $synth.Dispose()
}
"""


class SynthesisError(RuntimeError):
    pass


def _turns() -> list[dict]:
    """Every turn from both sides, in the order they are spoken."""
    turns = [
        {"n": n, "start": start, "end": end, "text": text,
         "who": "Dana", "voice": DANA_VOICE}
        for n, start, end, text in sales_call.DANA_TURNS
    ]
    turns += [
        {"n": n, "start": start, "end": end, "text": text,
         "who": "Priya", "voice": PRIYA_VOICE}
        for n, start, end, text in sales_call.PRIYA_TURNS
    ]
    turns.sort(key=lambda t: t["start"])
    return turns


def _slot_seconds(turns: list[dict], index: int) -> float:
    """How long turn `index` may speak for.

    The scripted gap between two turns minus `MIN_GAP_SECONDS`, because a turn
    that runs right up to the next one is the thing that breaks the merge. Left
    with no silence between them, whisper's own segmenter merges across the
    boundary and emits one segment holding the end of one speaker and the start
    of the other. Half a segment cannot match a whole one, so the dedup pass has
    nothing to act on and the turn is attributed to whoever spoke first.
    """
    turn = turns[index]
    if index + 1 < len(turns):
        return turns[index + 1]["start"] - turn["start"] - MIN_GAP_SECONDS
    return turn["end"] - turn["start"] + DEAD_AIR_SECONDS


def _synthesize(clips: list[dict], workdir: Path) -> None:
    """Render a batch of clips in one PowerShell session."""
    if not clips:
        return
    manifest = workdir / "manifest.json"
    manifest.write_text(json.dumps(clips), encoding="utf-8")
    script = workdir / "synth.ps1"
    script.write_text(SYNTH_PS1, encoding="utf-8")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive",
         "-ExecutionPolicy", "Bypass", "-File", str(script), "-Manifest", str(manifest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SynthesisError(
            "SAPI synthesis failed. Check that both demo voices are installed "
            f"(Settings > Time & language > Speech).\n{result.stderr.strip()}"
        )


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != RATE or wav.getsampwidth() != 2:
            raise SynthesisError(f"{path.name}: expected 16 kHz 16-bit mono")
        raw = wav.readframes(wav.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    # SAPI pads a beat of silence onto both ends of every utterance; trimming it
    # keeps a turn's audio where the script says the turn starts.
    return _trim_silence(samples)


def _trim_silence(samples: np.ndarray, threshold: float = 0.004) -> np.ndarray:
    loud = np.flatnonzero(np.abs(samples) > threshold)
    if loud.size == 0:
        return samples
    pad = int(0.03 * RATE)
    lo = max(0, int(loud[0]) - pad)
    hi = min(samples.size, int(loud[-1]) + pad)
    return samples[lo:hi]


def render_turns(turns: list[dict], workdir: Path) -> list[np.ndarray]:
    """Synthesize every turn, speeding a turn up if it overruns its slot."""
    audio: list[np.ndarray | None] = [None] * len(turns)
    pending = list(range(len(turns)))
    rate = 0

    while pending and rate <= MAX_SAPI_RATE:
        clips = [
            {"voice": turns[i]["voice"], "text": turns[i]["text"], "rate": rate,
             "path": str(workdir / f"turn_{turns[i]['n']:02d}_r{rate}.wav")}
            for i in pending
        ]
        _synthesize(clips, workdir)

        still_long: list[int] = []
        for i, clip in zip(pending, clips):
            samples = _read_wav(Path(clip["path"]))
            audio[i] = samples
            if samples.size / RATE > _slot_seconds(turns, i):
                still_long.append(i)
        pending = still_long
        rate += 1

    for i in pending:
        turn = turns[i]
        over = audio[i].size / RATE - _slot_seconds(turns, i)
        print(
            f"  ! turn {turn['n']} ({turn['who']}) runs {over:.1f}s past its slot "
            f"even at maximum speaking rate; it will overlap the next turn",
            file=sys.stderr,
        )
    return [a for a in audio if a is not None]


def lay_out(turns: list[dict], clips: list[np.ndarray], who: str | None = None,
            gain_db: float = 0.0) -> np.ndarray:
    """Place clips on the session timeline. `who` limits it to one speaker."""
    total = int((turns[-1]["start"] + DEAD_AIR_SECONDS) * RATE) + max(
        c.size for c in clips
    )
    track = np.zeros(total, dtype=np.float32)
    gain = 10.0 ** (gain_db / 20.0)
    for turn, clip in zip(turns, clips):
        if who is not None and turn["who"] != who:
            continue
        at = int(turn["start"] * RATE)
        track[at:at + clip.size] += clip * gain
    return track


def as_bleed(track: np.ndarray, gain_db: float, *, cutoff_hz: float,
             delay_ms: float) -> np.ndarray:
    """Make a clean track sound like it arrived the long way round.

    Attenuation alone is not enough. A copy that is merely quieter is still a
    pristine copy, and whisper transcribes it as accurately as the original -
    which leaves the dedup pass choosing between two equally good transcripts of
    the same words and picking the wrong one about as often as the right one.

    Bleed in the room is not pristine. It leaves a small speaker, crosses a
    couple of feet of air, and arrives late, dull and smeared. Reproducing that
    is what makes whisper transcribe the second copy badly, which is the signal
    the merge pass is actually built to act on.
    """
    delayed = np.concatenate(
        [np.zeros(int(delay_ms * RATE / 1000.0), dtype=np.float32), track]
    )[: track.size]

    # One-pole low-pass: laptop speakers and phone mics both roll off hard.
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / RATE)
    filtered = np.empty_like(delayed)
    running = 0.0
    for i, sample in enumerate(delayed):
        running += alpha * (sample - running)
        filtered[i] = running

    # A couple of early reflections - enough smear to blur word boundaries.
    smeared = filtered.copy()
    for reflect_ms, level in ((11.0, 0.32), (23.0, 0.18), (37.0, 0.09)):
        offset = int(reflect_ms * RATE / 1000.0)
        smeared[offset:] += filtered[:-offset] * level

    # Filtering and smearing both change the level, so set the final gain from
    # what came out rather than from what went in: `gain_db` is a promise about
    # how far under the original this copy lands, and the fixture's segment
    # levels are written against that promise.
    source_rms = float(np.sqrt(np.mean(track.astype(np.float64) ** 2)))
    bleed_rms = float(np.sqrt(np.mean(smeared.astype(np.float64) ** 2)))
    if bleed_rms <= 0.0 or source_rms <= 0.0:
        return smeared
    target = 10.0 ** (gain_db / 20.0)
    return smeared * float(source_rms * target / bleed_rms)


def with_room_tone(track: np.ndarray) -> np.ndarray:
    """Add a noise floor, because digital silence is not what a mic records."""
    rng = np.random.default_rng(20260910)
    amplitude = 10.0 ** (NOISE_FLOOR_DBFS / 20.0)
    return track + rng.normal(0.0, amplitude, track.size).astype(np.float32)


def write_wav(path: Path, track: np.ndarray) -> float:
    peak = float(np.max(np.abs(track)))
    if peak > 0.99:  # only ever attenuates; never inflates a quiet track
        track = track * (0.99 / peak)
    samples = np.clip(track * 32767.0, -32768, 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(samples.tobytes())
    return samples.size / RATE


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "audio"
    turns = _turns()
    print(f"Rendering {len(turns)} turns with SAPI ...")

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        try:
            clips = render_turns(turns, workdir)
        except SynthesisError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        dana = lay_out(turns, clips, who="Dana")
        priya = lay_out(turns, clips, who="Priya")

        both = with_room_tone(dana + priya)
        # Priya reaches the microphone through the laptop speakers and the room.
        mic = with_room_tone(
            dana + as_bleed(priya, MIC_BLEED_DB, cutoff_hz=2800.0, delay_ms=18.0)
        )
        # Dana reaches the loopback the shorter way, monitored back through the
        # output device, so that copy is late and dull but not room-smeared.
        system = with_room_tone(
            priya + as_bleed(dana, SYSTEM_BLEED_DB, cutoff_hz=3600.0, delay_ms=9.0)
        )

    for name, track in (
        ("sales_call_playback.wav", both),
        ("sales_call_mic.wav", mic),
        ("sales_call_system.wav", system),
    ):
        seconds = write_wav(out_dir / name, track)
        print(f"  {name}  {seconds // 60:.0f}m {seconds % 60:04.1f}s")

    print(f"\nWrote to {out_dir}")
    print("Play sales_call_playback.wav and record it, or copy the two track")
    print("files into a session folder as audio_mic.wav / audio_system.wav.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ScribeLocal

Local-first meeting notes for Windows. Records **both sides of a conversation** —
your microphone *and* whatever is coming out of your speakers — transcribes them
with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on your own
machine, and turns the transcript into a structured summary with the Claude API.

Audio and transcripts never leave your computer. The only thing that goes over
the network is the finished transcript, and only when you ask for a summary.

```
  audio_mic.wav    ──whisper──▶  "Me"      ──┐
                                              ├──▶ merge + dedup ──▶ transcript.md ──▶ summary.md
  audio_system.wav ──whisper──▶  "Others"  ──┘                                    (Claude API)

  inbox/*.m4a      ──whisper──▶  "Phone"   ─────▶ (same pipeline, same session layout)
```

**Windows only.** System-audio capture uses WASAPI loopback via
[PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch); there is no macOS or
Linux equivalent here.

---

## Setup

### 1. Python 3.12 and a virtual environment

Python **3.12** is recommended (3.11+ is required). Newer versions can be ahead
of the wheels `faster-whisper` and its `ctranslate2` backend publish.

```powershell
cd path\to\scribelocal
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If `py -3.12` isn't found, install Python 3.12 from
[python.org](https://www.python.org/downloads/) and re-run.

### 2. Install

The base install only captures audio. Everything else is an opt-in extra, so you
don't download a transcription backend you aren't going to use:

```powershell
pip install -e ".[transcribe,summarize,server]"
```

| Extra        | Pulls in                          | Needed for |
| ------------ | --------------------------------- | ---------- |
| *(base)*     | pyaudiowpatch, numpy, pyyaml, dotenv | `scribe record`, `scribe devices` |
| `transcribe` | faster-whisper                    | `scribe process` (transcription) |
| `summarize`  | anthropic                         | `scribe process` (summaries) |
| `server`     | fastapi, uvicorn                  | `scribe serve` (web UI) |
| `diarize`    | pyannote.audio                    | per-speaker labels within a track |

Using [uv](https://github.com/astral-sh/uv) instead:

```powershell
uv venv --python 3.12
uv pip install -e ".[transcribe,summarize,server]"
```

### 3. API key

```powershell
copy .env.example .env
```

Then edit `.env` and set `ANTHROPIC_API_KEY=sk-ant-...`. Recording and
transcription work without it; only summarization needs it. `.env` is
gitignored — keep it that way.

### 4. Verify your audio devices

**Do this before your first real recording.** See
[Verifying WASAPI loopback](#verifying-wasapi-loopback) below.

### 5. First run

```powershell
scribe record --title "test" --duration 20   # say something, play a video
scribe process test                          # transcribe + summarize
```

The first `scribe process` downloads the Whisper model (`medium` is ~1.5 GB);
later runs use the cached copy in `%USERPROFILE%\.cache\huggingface`.

---

## Verifying WASAPI loopback

System-audio capture only works if Windows exposes a **loopback** device for
your current output. Check with:

```powershell
scribe devices
```

```
 idx  type        rate  ch  name
  64  input      48000   2  Mic In (2- Elgato Wave:1)
  68  input      48000   1  Microphone (Brio 101)
  71  LOOPBACK   48000   2  Speakers (2- SRS-XB13 Stereo) [Loopback]
  74  LOOPBACK   48000   2  Speakers (Realtek(R) Audio) [Loopback]
```

You need **at least one `LOOPBACK` row**, and it should be the device you
actually listen through. If the list has no `LOOPBACK` rows at all, see
[No loopback device found](#no-loopback-device-found).

Then confirm both tracks actually capture sound:

```powershell
scribe record --title "loopback-check" --duration 15
```

While it runs, talk **and** play a video. Both meters should move — `mic` when
you talk, `system` when audio plays. Afterwards:

```powershell
scribe list
```

Check the session folder under `~\Documents\MeetingNotes\`: `audio_mic.wav` and
`audio_system.wav` should both be non-trivial in size, and both should have
audible content. A `system` track that stays silent means the loopback is
attached to an output you aren't using — set `audio.loopback_device` in
`config.yaml` to the right index from `scribe devices`.

---

## CLI reference

### `scribe devices`

Lists WASAPI input and loopback devices with their index, sample rate, and
channel count. Use the `idx` values for `audio.mic_device` /
`audio.loopback_device` in `config.yaml`.

### `scribe record`

Records until you press **Ctrl+C**. Writes `audio_mic.wav`, `audio_system.wav`,
and `meta.json` into a new timestamped folder under `notes_dir`.

| Flag | Description |
| ---- | ----------- |
| `--title`, `-t` | Session title; also used for the folder name (default: `meeting`) |
| `--duration`, `-d` | Stop automatically after N seconds instead of waiting for Ctrl+C |
| `--template` | Summary template to record in `meta.json` for later processing |

```powershell
scribe record -t "physics lecture" --template class-lecture
scribe record -t "unattended" -d 3600      # one hour, no babysitting
```

While recording it prints a live timer and a level meter per track. If nothing
is playing through your speakers the `system` meter will sit at zero — that's
expected, not a fault.

### `scribe process <session>`

Transcribes and summarizes an existing recording. `<session>` can be a folder
name, a full path, or any unique prefix (`scribe process 2026-08-03` works if
only one session starts with that).

| Flag | Description |
| ---- | ----------- |
| `--template` | Override the summary template for this run |
| `--model` | Override the Claude model id |
| `--force`, `-f` | Redo work even if `transcript.md` / `summary.md` already exist |
| `--transcribe-only` | Stop after writing `transcript.md` (no API call, no cost) |
| `--summarize-only` | Skip transcription and summarize the existing transcript |

```powershell
scribe process 2026-08-03_1400_physics-lecture
scribe process physics --summarize-only --template class-lecture
scribe process physics --force            # re-run everything from scratch
```

Existing outputs are **never overwritten without `--force`** — re-running is
safe and cheap.

### `scribe list`

Lists sessions newest-first with a status flag per stage:

```
[RTS]  2026-08-03_1400_physics-lecture     R = recorded
[RT-]  2026-08-02_0930_standup             T = transcribed
[R--]  2026-08-01_1600_client-call         S = summarized
```

### `scribe serve`

Runs the local web UI (see below), and the inbox watcher alongside it.

| Flag | Description |
| ---- | ----------- |
| `--host` | Bind address (default: `server.host`, `127.0.0.1`) |
| `--port`, `-p` | Port (default: `server.port`, `8321`) |

### `scribe watch`

Watches `inbox_dir` and imports any audio dropped there (see
[Phone sync inbox](#phone-sync-inbox)). Use this when you want imports without
running the web UI; `scribe serve` already includes a watcher.

| Flag | Description |
| ---- | ----------- |
| `--once` | Scan once and exit, instead of watching continuously |
| `--interval` | Seconds between scans (default: `inbox.poll_seconds`) |
| `--template` | Summary template for imports |
| `--no-summarize` | Transcribe imports but skip the API call |

```powershell
scribe watch                              # watch until Ctrl+C
scribe watch --once                       # drain whatever is sitting there now
scribe watch --template class-lecture      # everything from my phone is a lecture
```

---

## Web UI

```powershell
scribe serve
```

Then open <http://127.0.0.1:8321>. Four screens:

- **Record** — record/stop, live timer, level meters for both tracks, title and
  template.
- **Sessions** — every past recording with its recorded / transcribed /
  summarized status.
- **Session detail** — transcript and summary side by side, plus a Reprocess
  panel that re-runs transcription and/or summarization with live progress.
- **Settings** — Whisper model, retention, diarization toggle, API-key status,
  and your audio devices.

The UI is a wrapper, not a replacement: it calls the same functions the CLI
does, and anything you record or process in one shows up in the other. You can
run the UI and use the CLI at the same time.

**It has no authentication** and binds to localhost by default. It serves your
recordings, so don't expose it to a network you don't control.

---

## Phone sync inbox

Anything audio dropped into `inbox_dir` is imported automatically and run
through the same pipeline as a laptop recording: same session folder, same
`meta.json`, same `transcript.md` and `summary.md`.

```
inbox/New Recording 7.m4a
   │  moved on detection
   ▼
2026-07-30_0915_new-recording-7/
├── audio_phone.m4a     ← original file, original extension
├── meta.json           ← source: "inbox", original_filename, started_at
├── transcript.md
└── summary.md
```

Set your sync client (iCloud Drive, OneDrive, Syncthing — your choice, not
ScribeLocal's) to drop phone recordings into that folder. The watcher runs
inside `scribe serve`, or standalone via `scribe watch`.

### Testing it without a phone

**Yes — just copy an audio file into `inbox\` and watch.** Nothing about the
watcher cares where a file came from.

```powershell
scribe watch                                   # leave this running in one terminal
copy "C:\some\recording.wav" "$HOME\Documents\MeetingNotes\inbox\"
```

Within a couple of seconds:

```
inbox: picked up recording.wav (171 KB)
inbox: -> 2026-08-03_1727_recording
  [phone] transcribing audio_phone.wav ...
  [phone] 12 segment(s)
  merged 12 segment(s); dropped 0 duplicate(s), 0 silent/non-speech
  -> ...\2026-08-03_1727_recording\transcript.md
```

Then `scribe list` shows it, and it appears in the web UI like any other
session. Useful variations while testing:

```powershell
scribe watch --once                  # drain the folder and exit
scribe watch --no-summarize          # skip the API call (free, no key needed)
```

To check that the timestamp logic works, backdate a file before copying it in —
the session should be named for the file's own date, not today's:

```powershell
$f = Get-Item "$HOME\Documents\MeetingNotes\inbox\recording.wav"
$f.LastWriteTime = "2026-07-30 09:15"
```

An existing session folder from `scribe record` can be re-fed too: copy its
`audio_mic.wav` into `inbox\` under a different name and it imports as a new,
independent session.

### What it does, precisely

| Behaviour | Detail |
| --------- | ------ |
| **Formats** | `.m4a` (iPhone Voice Memos default), `.mp3`, `.wav`, `.aac`, `.flac`, `.ogg`, `.opus`, `.mp4`, `.m4b`, `.amr`, `.wma`. Configurable via `inbox.extensions`. Decoding is handled by PyAV, which bundles FFmpeg — nothing extra to install. |
| **Session title** | Derived from the filename: `New Recording 7.m4a` → *New Recording 7*, `team_standup.mp3` → *team standup*. Dates in filenames survive intact. |
| **Session timestamp** | The **file's** creation/modified time (whichever is earlier), not when the watcher noticed it. A memo synced three days late still files under the day you recorded it. |
| **Partial files** | A file is ignored until its size and mtime stop changing for `inbox.stable_seconds`. Half-synced audio is never transcribed. `.part`, `.crdownload`, `.icloud`, hidden and zero-byte files are skipped outright. |
| **Move on detection** | The audio is moved into its session folder *before* processing starts, so restarting the watcher mid-job can't double-process it. |
| **Speaker label** | Imports are labelled **Phone**, not *Me* — one device in a room heard everyone. For per-person labels, enable `diarization`, which turns them into *Phone (Speaker 1)*, *Phone (Speaker 2)*. |
| **On failure** | The audio is already safe inside its session folder. The watcher logs the error and keeps running; re-run with `scribe process <session>` once you've fixed the cause. |
| **Concurrency** | Under `scribe serve`, imports queue behind any running job and wait while a recording is live, so whisper never competes with a capture. |

Config lives under `inbox:` in [Configuration](#inbox) below.

---

## Configuration

All settings live in `config.yaml`. Anything you leave out falls back to the
defaults in `scribe/config.py`. Paths support `~`.

### paths

| Key | Default | Meaning |
| --- | ------- | ------- |
| `paths.notes_dir` | `~/Documents/MeetingNotes` | Where session folders are created |
| `paths.inbox_dir` | `~/Documents/MeetingNotes/inbox` | Watched drop folder — audio placed here is imported automatically (see [Phone sync inbox](#phone-sync-inbox)) |

### audio

| Key | Default | Meaning |
| --- | ------- | ------- |
| `audio.retention_days` | `7` | **Not yet enforced** — stored and editable, but nothing deletes old WAVs. Clean up manually for now. |
| `audio.mic_device` | `null` | `null` = system default input, or an index from `scribe devices` |
| `audio.loopback_device` | `null` | `null` = default output's loopback, or an index |
| `audio.chunk_frames` | `4096` | Frames per buffer written to disk. Raise if you see dropouts on a busy machine. |

### whisper

| Key | Default | Meaning |
| --- | ------- | ------- |
| `whisper.model` | `medium` | `tiny` \| `base` \| `small` \| `medium` \| `large-v3`. Bigger = more accurate and slower. `small` is a good CPU compromise. |
| `whisper.device` | `auto` | `auto` \| `cpu` \| `cuda` |
| `whisper.compute_type` | `auto` | `auto` \| `int8` \| `float16` \| … `int8` is much faster on CPU. |
| `whisper.language` | `null` | `null` autodetects; set e.g. `en` to skip detection and avoid mis-detection on short recordings |

### merge

Controls how the two tracks are combined. The mic hears your speakers and the
loopback can hear you, so the same sentence often lands on **both** tracks;
without this pass every line would appear twice, once per speaker.

| Key | Default | Meaning |
| --- | ------- | ------- |
| `merge.dedup` | `true` | `false` keeps every segment from both tracks (useful for debugging a bad merge) |
| `merge.time_tolerance` | `2.0` | Seconds of start-time slop when pairing segments across tracks |
| `merge.min_overlap` | `0.3` | Fraction of the shorter segment that must overlap in time |
| `merge.similarity` | `0.82` | 0–1 text similarity needed to call two segments the same speech. Lower catches more duplicates but risks eating real dialogue. |
| `merge.silence_floor_dbfs` | `-50` | Segments quieter than this are dropped (Whisper hallucinates over silence) |
| `merge.max_no_speech` | `0.6` | Drop segments Whisper itself scores as non-speech above this |

Which copy survives is decided by how loud a segment is **relative to its own
track's median speech level**, with Whisper's confidence as the tiebreak — real
speech towers over its track's median, bleed-through sits far below it.

### inbox

| Key | Default | Meaning |
| --- | ------- | ------- |
| `inbox.enabled` | `true` | `false` disables the watcher entirely (both in `scribe serve` and `scribe watch`) |
| `inbox.poll_seconds` | `2.0` | How often the folder is checked |
| `inbox.stable_seconds` | `3.0` | Size and mtime must hold still this long before a file is imported. Raise it if your sync client writes slowly in place. |
| `inbox.summarize` | `true` | `false` transcribes imports without calling the API |
| `inbox.template` | `null` | Template for imports; `null` uses `summarize.template` |
| `inbox.extensions` | `.m4a .mp3 .wav .aac .flac .ogg .opus .mp4 .m4b .amr .wma` | Which extensions count as audio |

### diarization

| Key | Default | Meaning |
| --- | ------- | ------- |
| `diarization.enabled` | `false` | Per-speaker labels *within* a track (`Others (Speaker 2)`, `Phone (Speaker 1)`). Needs `HF_TOKEN` in `.env` and the `diarize` extra; slow. Channel labels (Me / Others / Phone) work without it. |

### summarize

| Key | Default | Meaning |
| --- | ------- | ------- |
| `summarize.model` | `claude-sonnet-4-6` | Claude model id |
| `summarize.template` | `general` | `general` \| `class-lecture` \| `work-shift` \| `club-meeting` \| `sales-call` |
| `summarize.max_chunk_chars` | `150000` | Transcripts larger than this are chunked, summarized per chunk, then synthesized |
| `summarize.max_output_tokens` | `8000` | Per API call. Raise if summaries come back truncated. |
| `summarize.thinking` | `adaptive` | `adaptive` \| `off`. Model-dependent — set `off` for older models that reject it. |
| `summarize.effort` | `null` | `null` \| `low` \| `medium` \| `high`. Model-dependent. |

### server

| Key | Default | Meaning |
| --- | ------- | ------- |
| `server.host` | `127.0.0.1` | Bind address for `scribe serve`. Leave on localhost unless you know what you're doing. |
| `server.port` | `8321` | Port |

### Summary templates

Every template produces the same five sections — TL;DR, Key Decisions, Action
Items (with owners where identifiable), Open Questions, Notable Moments — but
weighs the transcript differently:

| Template | Prioritizes |
| -------- | ----------- |
| `general` | Whatever changes what someone does next |
| `class-lecture` | Concepts and explanations, worked examples, assessable material, assignments |
| `work-shift` | What happened on shift, what's outstanding for the next person, escalations |
| `club-meeting` | Motions, votes, role assignments, budget, event planning |
| `sales-call` | Needs, objections, budget/authority/timeline signals, commitments |

---

## Troubleshooting

### No loopback device found

`scribe record` prints:

```
WARNING: No WASAPI loopback device found - recording microphone only.
The other side of calls will NOT be captured.
```

The mic track still records; only system audio is missing. In order:

1. **Check for any loopback at all** — run `scribe devices` and look for
   `LOOPBACK` rows. Every active WASAPI output device should have one.
2. **Make sure an output device is actually enabled.** Right-click the speaker
   icon → *Sound settings* → *All sound devices*. A disabled or unplugged output
   has no loopback. Headphones that are off or disconnected are a common cause.
3. **Play something.** Some drivers only surface a loopback endpoint while the
   endpoint is active. Start music, then re-run `scribe devices`.
4. **Pin the device explicitly.** If a loopback exists but isn't the default,
   set its index in `config.yaml`:

   ```yaml
   audio:
     loopback_device: 71   # index from `scribe devices`
   ```

5. **Exclusive-mode apps.** Some DAWs, games, and conferencing tools grab the
   endpoint in exclusive mode, which blocks loopback capture. Close them, or
   turn off *Allow applications to take exclusive control of this device* in the
   device's advanced properties.
6. **Bluetooth headsets** in hands-free/headset mode often expose no usable
   loopback. Switch the profile to stereo/A2DP, or capture from the built-in
   speakers instead.

If the `system` track records but is **silent**, the loopback is attached to an
output you aren't listening through — pin the correct index as in step 4.

### Level meters don't render properly

`scribe record` draws its meters by rewriting one line with a carriage return
(`\r`). Some consoles don't handle that:

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| A new line for every update, scrolling forever | Console ignores `\r` (IDE run panes, VS Code *Debug Console*, Jupyter, CI logs) | Run in Windows Terminal, PowerShell, or `cmd` directly. In VS Code use the **Terminal** tab, not the Debug Console. |
| Meters wrap and smear across lines | Window too narrow for `mm:ss  mic [########----------------]  system [...]` | Widen the window to ~100 columns, or maximize it |
| Garbled `#` / box characters | Codepage or font substitution | The meters are plain ASCII `#` and `-`; try a different console font, or `chcp 65001` |
| Nothing appears until the recording ends | Output is being piped or redirected | Don't redirect if you want meters — recording still works fine, you just can't see levels |

The recording itself is unaffected in every case — the meters are cosmetic. If
your terminal is hostile, either use `--duration N` and skip watching, or use
the web UI (`scribe serve`), whose meters are rendered in the browser.

### `scribe process` fails with a credentials error

```
ERROR: no Anthropic credentials found. Copy .env.example to .env and set
ANTHROPIC_API_KEY (or run `ant auth login`), then try again.
```

Transcription doesn't need a key — `scribe process <session> --transcribe-only`
works offline. Only summarization calls the API.

### Transcription is very slow

Expected on CPU — `medium` typically takes about as long as the recording
itself, often longer. Options, in order of impact:

- Set `whisper.compute_type: int8` — usually a large CPU speedup for a small
  accuracy cost.
- Drop `whisper.model` to `small`.
- Set `whisper.language: en` to skip language detection.
- If you have an NVIDIA GPU with CUDA libraries installed, set
  `whisper.device: cuda`.

### The same sentence appears under both speakers

The dedup pass didn't catch a duplicate. Lower `merge.similarity` (e.g. `0.75`)
so near-matches still pair, or raise `merge.time_tolerance` if the two tracks
drift apart. Conversely, if **real dialogue is disappearing** — someone agreeing
with the exact words the other person just said — raise `merge.similarity`
toward `0.9`, or set `merge.dedup: false` and inspect the raw merge.

### A file in inbox/ isn't being picked up

Run `scribe watch --once` and see what it says. In order of likelihood:

1. **Extension isn't in the list.** Check `inbox.extensions` in `config.yaml`.
2. **It's a placeholder, not the file.** iCloud shows not-yet-downloaded files
   as `.filename.icloud` (hidden, zero-length). Open it once on this machine to
   force a real download. The watcher deliberately skips `.icloud`, `.part`,
   `.crdownload`, hidden, and zero-byte files.
3. **It's still syncing.** The watcher waits for the size and mtime to hold
   still for `inbox.stable_seconds`. A large file over a slow link takes a
   while — this is working as intended.
4. **Something is holding a lock on it.** You'll see
   `inbox: could not move <name> yet (...)`; it retries every poll.
5. **The watcher isn't running.** `scribe serve` includes one, but only if
   `inbox.enabled` is `true`. `scribe watch` shows the folder it's watching on
   startup — confirm it's the folder you're dropping into.

Don't run `scribe watch` and `scribe serve` against the same inbox at once.
It's safe (the move is atomic, so one of them simply loses the race) but the
logs get confusing.

### The web UI won't start

```
ERROR: the web UI needs FastAPI and uvicorn (...). Install them with:
  pip install -e ".[server]"
```

If the port is taken, `scribe serve --port 8400` or change `server.port`.

---

## Session folder layout

```
~/Documents/MeetingNotes/
├── inbox/                              drop phone recordings here
└── 2026-08-03_1400_physics-lecture/
    ├── audio_mic.wav       your microphone        ┐ laptop recording
    ├── audio_system.wav    system audio           ┘
    ├── audio_phone.m4a     imported recording     · inbox import (instead of the two above)
    ├── transcript.md       merged, deduped, timestamped
    ├── summary.md          Claude summary in the session's template
    └── meta.json           title, timings, template, merge stats, token usage
```

Everything is plain files: delete a folder to delete a session, and `meta.json`
is readable JSON if you want to script against it.

---

## Development

```powershell
python -m unittest discover tests
```

The suite covers the merge/dedup logic, summarization chunking and error
handling (against a stubbed SDK), and the web API (against a sandboxed config
and a fake recorder) — no audio devices, no API calls, no network.

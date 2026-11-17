# CLAUDE.md — ScribeLocal

Instructions for an autonomous coding agent working in `only1makai/scribelocal`.
Read this whole file before changing anything. If something you need is not in
this file, read the code. If the code doesn't settle it, ask the owner in the PR
description instead of guessing.

---

## What ScribeLocal is

A local-first meeting and call recorder for Windows:

1. Records **two tracks at once**: your microphone and system audio (WASAPI loopback).
2. Transcribes both tracks **on-device** with faster-whisper.
3. Merges the two transcripts into one, dropping cross-track duplicates and
   silence/non-speech segments.
4. Sends **only the final transcript text** to the Claude API for a structured
   summary. Raw audio never leaves the machine.

**Who it's for:** people who take calls and meetings and want structured notes
without uploading audio. The flagship use case is sales calls. The owner built
it while working as a sales rep taking real client calls. The owner also uses it
to record class lectures.

**It is also a public portfolio project** for internship applications.
README accuracy and verifiable numbers matter as much as the code.

---

## Tech stack

- **Python.** 3.14 is confirmed: full install plus the full test suite passes. A
  specific older version is not required.
- **Transcription:** `faster-whisper` (ctranslate2 backend), model `medium`, CPU.
- **Audio capture:** `pyaudiowpatch` (WASAPI loopback, Windows).
- **Summarization:** Anthropic Python SDK, model `claude-sonnet-4-6`.
- **Web UI / server:** FastAPI + uvicorn (`scribe serve`), app factory at
  `scribe.server.app.create_app()`.
- **Optional diarization:** `pyannote.audio` (per-speaker labels within a track).
- **Tests:** stdlib `unittest`. Server tests use Starlette's TestClient, which
  needs `httpx2`.

### `pyproject.toml` extras (current and verified)

| Extra | Contents |
|---|---|
| `transcribe` | `faster-whisper>=1.0` |
| `summarize` | `anthropic>=1.6.0` |
| `server` | `fastapi>=0.110`, `uvicorn>=0.29` |
| `diarize` | `pyannote.audio>=3.1` (optional, heavy) |
| `test` | `transcribe` + `summarize` + `server`, plus `httpx2>=2.0.0` |

---

## Install, run, test

```bash
# Normal use
pip install -e ".[transcribe,summarize,server]"
# Optional per-speaker labels
pip install -e ".[transcribe,summarize,server,diarize]"

# Tests: `.[test]` is the only supported path to a green suite
pip install -e ".[test]"
python -m unittest discover tests
```

**Expected on `main`: 240/240 tests passing** from a fresh clone and a fresh
venv. If you add tests, report the new count. Installing only `.[server]` and
running the suite fails 5 tests on missing `httpx2`. That is **expected by
design**, so do not "fix" it.

### Configuration

- `config.yaml` is located relative to the package, not the working directory,
  so `scribe` works from any folder. The API key is read from `.env`
  (`ANTHROPIC_API_KEY`). `.env` must never be committed.
- Known keys: `whisper.compute_type`, `whisper.device`, the whisper model name,
  the mic and loopback device settings, `summarize.thinking`, `summarize.effort`.
  Read `config.yaml` for exact key names before touching it.
- `scribe/live/stream.py` reads the same device/compute keys, so live mode
  inherits whatever transcription defaults are set.
- There is a user-facing reference doc at `docs/reference.md`.

### CLI

Subcommands: `list`, `devices`, `record`, `process`, `live`, `serve`, `watch`.

```bash
scribe devices                                   # list audio devices with index, type (input/LOOPBACK), rate, channels
scribe record --title <title> --template <name>  # Ctrl+C stops; optional --duration/-d N seconds
scribe process <session-name>                    # transcribe + merge + summarize
scribe process <session-name> --summarize-only   # re-run only the summary
scribe process <session-name> --transcribe-only --force   # re-transcribe, overwriting transcript.md
scribe list                                      # list sessions
scribe serve                                     # local web UI
scribe live                                      # live / copilot mode
```

- Templates: `sales-call` (flagship), `class-lecture`, `work-shift`,
  `club-meeting`, `general`. An unknown template name errors at summarize time,
  not at record time.
- A session is a folder named `YYYY-MM-DD_HHMM_<slug>`, where the slug is the
  title lowercased with `/` turned into `-`. The folder contains
  `audio_mic.wav`, `audio_system.wav`, `meta.json` (status, timestamps,
  `merge_stats`, timing/environment), `transcript.md`, and `summary.md`.
- `scribe process` matches on the folder name, not the original title.

### What you can and cannot verify

Real recording needs Windows audio hardware. You almost certainly cannot record
real audio. Verify with unit tests and mocked audio/devices. When a change also
needs a real-hardware check, say so in the PR and leave a checklist for the
owner. **Never claim a hardware result you did not produce.**

---

## Repo structure (known files only)

```
pyproject.toml
README.md
config.yaml
docs/reference.md
scribe/
  cli.py                     # entry point; cmd_record, process, etc.
  audio/recorder.py          # dual-track capture; stop() closes streams and finalizes WAVs
  summarize/__init__.py      # Anthropic call + error handling
  summarize/templates.py     # the five templates; sales-call is the flagship
  server/app.py              # FastAPI app (create_app)
  live/                      # live/copilot mode, incl. stream.py
demo/
  sales_call_script.md       # fictional sales call used for the demo
  make_demo_audio.py         # generates synthetic demo audio
tests/
  test_cli.py
  test_session.py
  test_server.py             # asserts a fake key ("sk-ant-super-secret-value") never appears in responses
  test_live_context.py
  test_live_copilot.py
  test_live_stream.py
  test_merge_sales_call.py   # hand-built fixture: 24/24 turns, 8 dupes + 2 silence artifacts dropped
```

Other modules exist, such as the merge logic, the web UI templates, and the
session resolver. Read them before assuming their structure.

---

## Coding conventions

- **Read before building.** Inspect the code you're changing first.
- **Propose before building anything non-trivial.** Put the design and tradeoffs
  in the PR description. Changes to the recorder, merge timing, config defaults,
  dependencies, or the on-disk session format always count as non-trivial.
- **One roadmap item per PR.** Keep PRs small.
- **Branches:** `feat/<short-name>` or `fix/<short-name>`, matching existing
  `feat/sales-call-demo-and-config` and `fix/install-friction`.
- **Dependencies go in the right extra.** Test-only dependencies go in `test`.
  Never add to base dependencies what one feature needs.
- **Tests use `unittest`.** Mock audio devices and the Anthropic client. New
  behavior gets tests.
- **Keep the full suite green** on a fresh `.[test]` install.
- **Raw output in every PR.** Paste the raw `python -m unittest discover tests`
  output. Any number you claim (file sizes, timings, tokens, cost) needs the raw
  command output beside it. This is a hard rule: early in this project,
  fabricated status reports (invented test counts, a config change that was never
  applied, invented timings) cost the owner real time.
- **Keep evidence sources separate.** The README's numbers come from three
  distinct sources: (1) the hand-built test fixture, (2) real faster-whisper on
  synthesized demo audio, and (3) a real human-recorded session. Never blend them
  into one claim.
- **No formatter or linter is documented.** Match surrounding style, and don't
  reformat code you aren't changing.

---

## Decisions already made (do not relitigate)

| Decision | Why |
|---|---|
| "Sales story, general engine": README leads with the sales-rep origin; all five templates stay | The origin story is credible and differentiated, and cutting four working templates isn't worth it. |
| Sales-call template lives in `scribe/summarize/templates.py` and is the most developed | Built around what a rep needs: objections, budget/timeline signals, commitments, next steps with an owner. |
| Shipped defaults: `device: cpu`, `compute_type: int8_float32`, model `medium` | `auto` resolved to float32 and was OOM-killed on the owner's machine. CUDA crashed (`cublas64_12.dll`). `docs/reference.md` already recommended int8 for CPU speed. |
| `.[server]` alone does not run the full test suite | By design. `.[test]` is the documented test path. |
| `anthropic>=1.6.0` floor | The exact version tested live. Normal installs upgrade cleanly. |
| `watchfiles` removed from `server` | It was never imported. |
| Raw audio never leaves the machine | This is the core privacy claim. |
| Storage moves to **16 kHz mono** WAV (approved, not yet built) | Whisper resamples to 16 kHz mono internally, so 48 kHz stereo capture only wastes disk (~1 GB/hour today). |
| Audio deletion and compression (Opus/FLAC) are **not** approved | The owner wants to keep audio for now. |

---

## Never do these

- **Never commit secrets:** API keys, tokens, `.env`, HF tokens. The repo passed
  a full git-history secrets audit before going public. Keep it that way.
- **Never commit real recordings, transcripts, summaries, `meta.json`, or session
  folders.** That means no real client names or calls and no class lectures
  (other people's voices). Demo and test content stays fictional.
- **Never push to `main`.** Always branch and open a PR.
- **Never force-push or rewrite history** on any shared branch.
- **Never change the license** or add license headers.
- **Never delete user data.** No code path may delete session folders or audio
  unless it's an explicit, opt-in, owner-approved feature. Don't add automatic
  deletion or retention.
- **Never change shipped defaults** (CPU, `int8_float32`, `medium`, summarize
  model) without explicit owner approval in the PR.
- **Never send raw audio off the device**, and never add telemetry or anything
  that phones home.
- **Never relitigate the decisions table.**
- **Never claim a result without raw output**, including test counts, sizes,
  timings, costs, and hardware behavior.
- **Never edit the README's verified numbers** unless you produced new raw
  measurements, and label every new number with its source.

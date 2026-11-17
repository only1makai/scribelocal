# ROADMAP.md — ScribeLocal

From the current state to "done." Each unchecked item is sized for **one PR**.
Items are ordered most important first within each phase, and phases are ordered
by priority. Read `CLAUDE.md` first. Its rules (no pushing to `main`, raw output
in every PR, no changes to defaults without approval) apply to every item here.

**Definition of done for the project:** a reliable recorder for hour-plus real
calls and lectures, public, tested, documented, and demoed, with every README
number backed by raw output.

Items marked **(owner)** are for the owner, not the agent. Items marked
**(needs decision)** are blocked on an open question at the bottom of this file.

---

## Phase 0 — Shipped (done)

- [x] Dual-track capture (mic + WASAPI loopback) working on real hardware
- [x] On-device transcription + cross-track merge/dedupe
- [x] Claude summarization with five templates; sales-call template built out as flagship
- [x] Config fixes: `device: cpu`, `compute_type: int8_float32`, summarize thinking/effort incompatibility fixed
- [x] Full git-history secrets audit (clean); repo made public
- [x] `pyproject.toml` install fix: `test` extra added, `watchfiles` dropped, `anthropic>=1.6.0`, README Python guidance corrected
- [x] `fix/install-friction` and `feat/sales-call-demo-and-config` merged; 211/211 tests pass from a fresh clone
- [x] Verified metrics in README: 42.92 s real session → 56.86 s transcription (1.32x realtime, measured twice); 1 API call, 695 in / 164 out tokens, ~$0.0045; test fixture 24/24 turns, 8 dupes + 2 silence dropped; synthetic-audio run 46 dupes → 24 turns
- [x] README sections: Why, How it works, Templates, Performance, Install, Usage, Privacy
- [x] Demo fixtures merged: `demo/sales_call_script.md`, `demo/make_demo_audio.py`

---

## Phase 1 — Recording reliability (highest priority)

Real-world use in November exposed these problems. Two class recordings opened
the wrong devices (an Oculus virtual mic and an Elgato loopback instead of the
configured Brio mic and Realtek loopback). One stop crashed on Ctrl+C. One
`process` call couldn't find its session.

- [ ] **1.1 Match audio devices by name, not only by index.**
  Device indices shift on Windows when devices or virtual-audio software change.
  - Config accepts a device **name** (substring match is fine) for mic and loopback; index stays supported for backward compatibility.
  - If the configured device isn't found, `scribe record` exits with a clear error listing available devices. It must not silently fall back to another device.
  - Existing configs keep working unchanged.
  - Unit tests with a mocked device list cover: exact match, substring match, no match, ambiguous match (multiple hits → error listing them).
  - PR includes a hardware checklist for the owner (run `scribe devices`, set names, confirm the banner shows the intended devices).

- [ ] **1.2 Clean shutdown on Ctrl+C.**
  A Ctrl+C during `recorder.stop()` raised `KeyboardInterrupt` inside `pyaudiowpatch` `pa.close()` (via `scribe/audio/recorder.py` `stop` → `track.stream.close()`).
  - A second Ctrl+C during shutdown no longer produces a traceback. Shutdown completes, or the user gets a clear message.
  - WAV headers are finalized before streams are torn down, so files are valid even if shutdown is interrupted.
  - Print a "Stopping… don't press Ctrl+C again" line while finalizing.
  - Tests simulate an interrupt during stop with mocked streams.

- [ ] **1.3 Dead-track warning during recording.**
  Both November class recordings had a flat system meter for the whole session, and it went unnoticed.
  - While recording, if either track stays below a silence threshold for 30 s (configurable), print a visible warning naming the track and the device.
  - The warning repeats at a sane interval rather than spamming.
  - Threshold logic is unit-tested with synthetic sample buffers.

- [ ] **1.4 Friendlier session lookup in `scribe process`.**
  Titles containing `/` are slugified, so `scribe process <original-title>` failed with "no session matching."
  - If no exact folder match exists, also try the slugified form of the argument, and fall back to a unique substring match.
  - If multiple sessions match, list them and exit without guessing.
  - `scribe record` prints the exact `scribe process <session-name>` command (it already does; keep it).
  - Tests cover slashes, case, unique and ambiguous partial matches.

- [ ] **1.5 Pre-flight device check command.**
  - `scribe record --check` (or `scribe devices --test`, whichever fits the CLI best) opens the configured devices for ~5 s and prints peak level per track plus pass/fail.
  - Exits non-zero if either track is silent.
  - Logic is tested with mocked streams; PR includes a hardware checklist.

---

## Phase 2 — Storage and long sessions

- [ ] **2.1 Save audio as 16 kHz mono WAV.** *(approved)*
  Current 48 kHz capture is about 1 GB per hour.
  - PR first states whether resampling happens at capture or at stop, and why that's safe for cross-track timestamp alignment in merge.
  - Both tracks are saved as 16 kHz, mono, 16-bit PCM WAV.
  - Existing 48 kHz sessions still process correctly.
  - Full `.[test]` suite passes; add a test that merge alignment is unchanged on resampled fixtures.
  - Report expected size per hour (computed), plus an owner checklist to record 60 s and report real before/after file sizes.
  - Out of scope: deleting audio, Opus/FLAC compression.

- [ ] **2.2 Actionable error when the Whisper model fails to load.**
  A 104-minute session failed at model load with `mkl_malloc: failed to allocate memory` (medium, cpu, int8_float32). This happens before any audio is read, so 2.1 will **not** fix it.
  - Catch model-load allocation failures and print guidance: free RAM, or retry with a smaller model.
  - Add a per-run override (e.g. `scribe process <session> --model small`) that does **not** change the config default.
  - The recorded audio is never touched by a failed run.
  - Tests mock the model loader to raise and assert the message and exit code.

- [ ] **2.3 Verify and document long-session behavior.**
  The longest verified end-to-end run is 42.92 s. Real use is 60–105 minutes.
  - Read the recorder and transcription code, and document in `docs/reference.md` any maximum duration, file-size limit, or memory growth with length (or state that none exists, citing the code).
  - If transcription memory grows with input length, open a follow-up issue with findings rather than fixing it in this PR.

- [ ] **2.4 Quiet the Hugging Face warnings on Windows.**
  The `HF_TOKEN` and symlink-cache warnings look like errors to users.
  - Suppress or downgrade them in a documented way (e.g., setting `HF_HUB_DISABLE_SYMLINKS_WARNING`), or explain them once in `docs/reference.md`.
  - No behavior change to model download or caching.

---

## Phase 3 — Live interface

- [ ] **3.1 Audit `scribe serve` (read-only, docs PR).**
  - Add `docs/ui-audit.md` describing: existing routes and pages; whether the web UI can start/stop recording, show levels, pick devices, run `process`, and browse sessions/transcripts; the effort to add live meters (websocket), a name-based device picker, and the dead-track warning; and a comparison with a Textual terminal app.
  - Include the raw output of starting `scribe serve` from a fresh `.[transcribe,summarize,server]` install.
  - No code changes.

- [ ] **3.2 Build the live interface.** *(needs decision: web UI vs Textual)*
  Must include: REC indicator, elapsed time, device names, per-track level meters, running file size, the dead-track warning from 1.3, and a pre-record device picker that uses 1.1's name matching.
  - Device switching **before** recording only. Mid-recording switching is out of scope (it breaks two-track timing).
  - New dependencies go in the correct extra.
  - Full suite passes; UI logic separated enough to unit-test.
  - PR includes screenshots or a recorded terminal session from the owner's hardware checklist.

---

## Phase 4 — Launch polish

- [ ] **4.1 Record the ~90-second demo video.** **(owner)**
  Real `scribe record` → `scribe process` flow using the fictional sales call in `demo/`. No real client audio or names.

- [ ] **4.2 Link the demo in the README Demo section.** *(blocked on 4.1)*
  - Demo section contains the video link and one line describing what it shows.

- [ ] **4.3 Write the README "What's next" line.** **(owner)**

- [ ] **4.4 Update README for Phases 1–3.**
  - Usage reflects any new flags/commands (name-based devices, check command, `--model` override, UI).
  - Any new number (e.g., storage per hour after 2.1) is added only with raw output and labeled with its source.
  - Existing verified numbers stay unchanged.

- [ ] **4.5 Add CI.** *(needs decision)*
  - GitHub Actions workflow runs `pip install -e ".[test]"` and `python -m unittest discover tests` on pull requests.
  - Runner OS per the open question below.

---

## Open questions for the owner

1. **License:** does the repo have a `LICENSE` file, and which license? (Agents must not add or change one.)
2. **Interface (3.2):** web UI or Textual? Wait for the 3.1 audit before deciding.
3. **CI (4.5):** do you want CI? Should it run on `windows-latest`, Linux, or both? (Capture is Windows-only; tests should be mockable, but nobody has confirmed the suite passes on Linux.)
4. **Realtime factor:** README uses 1.32x (Measure-Command, 42.92 s session). A Claude Code session also quoted 1.95x for a two-track session without raw output shown. Should the README keep 1.32x only, or should someone re-measure on a longer session?
5. **Device config:** which device names should the shipped `config.yaml` default to after 1.1, or should defaults stay `null`/auto with name matching only in personal configs?
6. **`.gitignore`:** does it already exclude `.env`, session folders, and any local recordings path? Should an agent add those entries if missing?
7. **Retention:** confirm that audio deletion and compression stay out of scope until you say otherwise.
8. **`diarize`:** should the README feature it, or leave it as a footnote?
9. **Minimum Python version:** what does `pyproject.toml` declare for `requires-python`, and should it be pinned?
10. **The Nov 5 and Nov 12 recordings:** did either capture usable system audio? The answer tells us whether 1.1 is the only cause or whether something else is wrong with loopback capture.

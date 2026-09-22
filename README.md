# ScribeLocal
 
A local-first meeting and call recorder that transcribes and summarizes on your own machine — your audio never leaves your laptop.
 
## Why I built this
 
I take real client calls as a sales rep for Fresh Prints. I wanted notes that captured objections, budget signals, and next steps without depending on a SaaS tool that ships my client conversations to a third-party server. So I built one myself: mic + system audio capture, transcription that runs entirely locally, and a summarizer that only ever sees the final text — never the recording.
 
The sales-call use case is what got this built, and it's the flagship template here, but the underlying engine is general-purpose. It also handles lecture notes, work shifts, club meetings, and general meetings.
 
## How it works
 
1. **Record** — captures both microphone and system audio simultaneously (so it captures both sides of a call, not just your mic).
2. **Transcribe, locally** — runs on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on your machine. Nothing about the audio or the raw transcript is sent anywhere.
3. **Summarize** — only the final transcript text is sent to Claude for summarization, structured according to whichever template you pick.
4. **Live "copilot" mode** — surfaces prompts in the local web UI while the call is still happening, not just after the fact.
```
your mic + system audio → local transcription (faster-whisper) → [only text crosses this line] → Claude → structured summary
```
 
## Templates
 
| Template | Use case |
|---|---|
| **sales-call** (flagship) | Objections raised, budget/timeline signals, commitments made, next steps with an owner |
| class-lecture | Structured lecture notes |
| work-shift | Shift handoff notes |
| club-meeting | Meeting minutes format |
| general | Anything else |
 
The sales-call template is the most developed of the five, by design — it's built around what a merch/apparel sales rep actually needs to capture on a real call, not a generic "meeting notes" format.
 
## Demo
 
[~90-second demo using a mocked sales call — link once recorded]
 
## Performance
 
An automated test suite validates the merge logic against a scripted 24-turn call: all 24 turns reconcile correctly, with 8 synthetic cross-track duplicates and 2 silence artifacts correctly dropped. Run against real faster-whisper transcription output on a synthesized call, the same logic correctly resolved 46 cross-track duplicates down to the same 24 turns. On a live captured recording, a 42.92-second session transcribes in 56.86s (1.32x realtime) and summarizes for ~$0.0045 per call.
 
| Metric | Value |
|---|---|
| Merge correctness (test suite) | 24/24 turns reconciled, 8 duplicates + 2 silence artifacts correctly dropped |
| Merge robustness (real transcription, synthetic call) | 46 cross-track duplicates correctly resolved to 24 turns |
| Real-world transcription time | 56.86s for a 42.92s recording (1.32x realtime), CPU, int8_float32 |
| Real-world summarization cost | ~$0.0045/call (695 input / 164 output tokens, Claude Sonnet 4.6) |
 
Only the final transcript text and summary ever leave your machine — transcription happens entirely offline.
 
## Install
 
```
pip install -e ".[transcribe,summarize,server]"
```
 
That gets you recording, local transcription, Claude-powered summarization, and the web UI (`scribe serve`).
 
Optional: add `diarize` for per-speaker labels within a single track (pulls in `pyannote.audio`, a heavier dependency, so it's opt-in):
 
```
pip install -e ".[transcribe,summarize,server,diarize]"
```
 
Running the test suite requires everything installed together — use the `test` extra rather than any partial combination:
 
```
pip install -e ".[test]"
python -m unittest discover tests
```
 
## Usage
 
```
scribe record --title my-call --template sales-call
# ... talk ...
# Ctrl+C to stop
 
scribe process my-call
```
 
`scribe live` runs copilot mode, surfacing prompts in the local web UI while the call is still happening.
 
## Privacy
 
- Audio and raw transcripts are never transmitted anywhere.
- Only the generated summary text is sent to Claude's API for the final structuring step.
- No real client data, audio, or names appear anywhere in this repo or its demo materials.
## What's next
 
- Additional refinement of the four non-flagship templates
- [any other roadmap items you want listed]

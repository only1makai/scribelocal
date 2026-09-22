# Demo script — event hoodie sales call

Synthetic two-sided sales call used for two things:

- the **demo source**: play it back and record it, so `scribe` captures a real
  two-track session with real bleed between mic and loopback;
- the **merge/dedup test fixture**: `tests/fixtures/sales_call.py` encodes this
  same conversation as `Segment` objects, so the test suite and the demo tell
  the same story.

Keep the two in sync. If a line changes here, change it in the fixture too.

## Casting

| Role | Speaker | Track | Voice (SAPI) | Label in transcript |
|---|---|---|---|---|
| Dana — merch rep (you) | Dana | `mic` | Microsoft David Desktop | `Me` |
| Priya — prospect | Priya | `system` | Microsoft Zira Desktop | `Others` |

Dana sells the hoodies, so Dana is the local speaker on the microphone. Priya
comes out of the speakers, so Priya lands on the WASAPI loopback.

## What this call is built to exercise

- **`sales-call` template** — needs, objections, budget/authority/timeline
  signals, and explicit commitments all appear, so the summary has something
  real to extract in every section.
- **Cross-track dedup** — several turns are loud on one track and faint on the
  other, in both directions.
- **Containment matching** — turn 5 is one long segment on Dana's track and
  arrives split in two on the loopback.
- **Silence gate** — a whisper hallucination over the dead air after the call
  ends, and a non-speech filler token mid-call.
- **Pricing continuity** — `$28` base and `rush +15%` match the numbers already
  used in `tests/test_live_context.py`, so a live-mode context doc built from
  the same pricing stays consistent.

## Script

Timings are the start of each turn in the recording, in seconds.

| # | @ | Speaker | Line |
|---|---|---|---|
| 1 | 0.0 | Dana | Hi Priya, thanks for hopping on. I saw your form come through for event hoodies — is this for the fall kickoff? |
| 2 | 6.9 | Priya | Yeah, exactly. We're doing our fall kickoff on October eleventh and we want hoodies for the whole crew. |
| 3 | 13.7 | Dana | Perfect. How many people are we talking about? |
| 4 | 16.9 | Priya | Right now the headcount is around fifty, but it could stretch to seventy if the sponsor tier fills. |
| 5 | 24.1 | Dana | Fifty to seventy is comfortably in our standard tier. Base hoodies run twenty-eight dollars apiece at that volume, front and back print included. |
| 6 | 33.3 | Priya | Twenty-eight is a little higher than what we paid last year. We were closer to twenty-two. |
| 7 | 39.1 | Dana | Understood. Twenty-two usually means a lighter blank — six ounce fleece instead of the eight and a half we use. I can quote both so you can see the difference. |
| 8 | 47.9 | Priya | That would help. Who actually signs off on this? I can approve up to fifteen hundred dollars myself, anything above that goes to our director. |
| 9 | 56.3 | Dana | Good to know. At fifty units the base order lands around fourteen hundred, so you'd be just under your own line. |
| 10 | 63.2 | Priya | And if we go to seventy? |
| 11 | 65.9 | Dana | Seventy at twenty-eight is nineteen sixty, so that one needs your director. |
| 12 | 70.8 | Priya | Okay. What's the turnaround? |
| 13 | 73.5 | Dana | Standard production is twelve business days from art approval. If you need it faster, rush is fifteen percent on top. |
| 14 | 81.1 | Priya | October eleventh is our hard date. When would art need to be locked? |
| 15 | 86.5 | Dana | Working backward, art locked by September twenty-second keeps you on standard pricing. After that we're into rush. |
| 16 | 93.7 | Priya | That's tight but doable. Our designer has the logo files already. |
| 17 | 98.9 | Dana | Send them over as vector if you have them — AI or EPS. If it's a PNG we can redraw for a one time forty dollar art fee. |
| 18 | 106.8 | Priya | One more thing — last year a chunk of the sizes came in wrong and we ate the cost. How do you handle that? |
| 19 | 114.7 | Dana | We send a size run sheet before production and you sign off on it. If we ship against a signed sheet and it's wrong, we reprint free. |
| 20 | 123.1 | Priya | That's better than what we had. |
| 21 | 126.3 | Dana | So here's what I'll do. I'll send two quotes today, fifty units and seventy units, both on the eight and a half ounce blank, plus a lighter blank option for comparison. |
| 22 | 135.9 | Priya | Send it to me and copy our director. I'll get you a decision by Friday. |
| 23 | 141.7 | Dana | Friday works. I'll also hold a production slot for the week of September twenty-third so we don't lose the standard rate while you decide. |
| 24 | 150.8 | Priya | Appreciate that. Talk Friday. |

Call ends at about 154 seconds. Leave several seconds of dead air after it — the
silence gate is there to handle exactly that, and the demo should show it working.

## Producing the audio

```powershell
python demo/make_demo_audio.py
```

Writes three files into `demo/audio/`:

| File | What it is |
|---|---|
| `sales_call_playback.wav` | both sides, one file — play this into the room and record it |
| `sales_call_mic.wav` | Dana loud, Priya mixed in faint — a ready-made `audio_mic.wav` |
| `sales_call_system.wav` | Priya loud, Dana mixed in faint — a ready-made `audio_system.wav` |

Two ways to use them:

**Record it for real** (what the demo should show). Play `sales_call_playback.wav`
through your speakers and run `scribe record --title "hoodie quote call"`,
talking over Dana's turns yourself if you want a genuinely mixed mic track. The
bleed is then real, and so are the dedup stats.

**Skip the recording** (fast path, for iterating on the pipeline). Copy the two
track files into a session folder as `audio_mic.wav` and `audio_system.wav`,
then `scribe process <session> --template sales-call`.

The generated WAVs are not committed — `*.wav` is already in `.gitignore`, and
they rebuild from this script in about a minute.

## What the pipeline does with it

Measured on the i7-1360P, `whisper.model: medium`, `device: cpu`,
`compute_type: int8_float32`, both tracks, transcription only:

| | |
|---|---|
| Audio | 165.3 s per track, both tracks transcribed |
| Wall clock | 321.6 s |
| Ratio | 1.95× realtime |
| Segments in | 48 mic + 49 system = 97 |
| Dropped as cross-track duplicates | 46 |
| Dropped by the silence gate | 1 |
| Segments kept | 50 — 27 mic, 23 system |
| Speaker blocks in `transcript.md` | 24, strictly alternating — one per scripted turn |

The two segment counts reconcile: 48 + 49 in, and 50 + 46 + 1 out.

That is a transcription-only number. It does not include the summarization call,
so it is not the figure the README wants.

Two things to know before running it:

- **`whisper.device: auto` fails on this machine.** ctranslate2 sees the iGPU,
  selects CUDA, and then dies at the first encode with
  `Library cublas64_12.dll is not found or cannot be loaded`. The model loads
  fine, so the failure lands mid-transcription rather than at startup. Set
  `device: cpu` explicitly.
- **The bleed is synthesized, not recorded.** `make_demo_audio.py` fakes the
  speaker-to-microphone path with a low-pass, a delay and a few reflections,
  which is enough to make whisper transcribe the second copy badly the way real
  bleed does. It is a stand-in for a real capture, not a replacement, and merge
  statistics taken from it should be described as such.

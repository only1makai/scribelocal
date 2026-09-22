"""The event-hoodie sales call from `demo/sales_call_script.md`, as segments.

One conversation, two uses: `demo/make_demo_audio.py` renders it to audio for
the demo, and this module hands the same call to the merge/dedup tests as the
`Segment` objects whisper would have produced from that audio.

The two tracks are deliberately not independent. Dana is on the microphone and
Priya comes out of the speakers, so each of them bleeds faintly onto the other's
track - which is the whole reason the dedup pass exists. The bleed copies here
carry the differences whisper really does produce on a quieter second pass:
punctuation that lands elsewhere, an occasional dropped word.

If a line changes in the script, change it here too.
"""

from __future__ import annotations

from scribe.transcribe.segments import Segment

# Levels are dBFS, chosen so each track's own median sits inside its loud
# speech: real speech lands near a prominence of zero, bleed ~25 dB below it.
LOUD_MIC = -18.0
LOUD_SYSTEM = -15.0
BLEED_ON_MIC = -42.0
BLEED_ON_SYSTEM = -41.0

# (turn, start, end, text) - Dana sells the hoodies, so Dana is "Me" on the mic.
DANA_TURNS: list[tuple[int, float, float, str]] = [
    (1, 0.0, 6.4, "Hi Priya, thanks for hopping on. I saw your form come through for "
                  "event hoodies - is this for the fall kickoff?"),
    (3, 13.7, 16.4, "Perfect. How many people are we talking about?"),
    (5, 24.1, 32.8, "Fifty to seventy is comfortably in our standard tier. Base hoodies "
                    "run twenty-eight dollars apiece at that volume, front and back "
                    "print included."),
    (7, 39.1, 47.4, "Understood. Twenty-two usually means a lighter blank - six ounce "
                    "fleece instead of the eight and a half we use. I can quote both so "
                    "you can see the difference."),
    (9, 56.3, 62.7, "Good to know. At fifty units the base order lands around fourteen "
                    "hundred, so you'd be just under your own line."),
    (11, 65.9, 70.3, "Seventy at twenty-eight is nineteen sixty, so that one needs your "
                     "director."),
    (13, 73.5, 80.6, "Standard production is twelve business days from art approval. If "
                     "you need it faster, rush is fifteen percent on top."),
    (15, 86.5, 93.2, "Working backward, art locked by September twenty-second keeps you "
                     "on standard pricing. After that we're into rush."),
    (17, 98.9, 106.3, "Send them over as vector if you have them - AI or EPS. If it's a "
                      "PNG we can redraw for a one time forty dollar art fee."),
    (19, 114.7, 122.6, "We send a size run sheet before production and you sign off on "
                       "it. If we ship against a signed sheet and it's wrong, we reprint "
                       "free."),
    (21, 126.3, 135.4, "So here's what I'll do. I'll send two quotes today, fifty units "
                       "and seventy units, both on the eight and a half ounce blank, "
                       "plus a lighter blank option for comparison."),
    (23, 141.7, 150.3, "Friday works. I'll also hold a production slot for the week of "
                       "September twenty-third so we don't lose the standard rate while "
                       "you decide."),
]

PRIYA_TURNS: list[tuple[int, float, float, str]] = [
    (2, 6.9, 13.2, "Yeah, exactly. We're doing our fall kickoff on October eleventh and "
                   "we want hoodies for the whole crew."),
    (4, 16.9, 23.6, "Right now the headcount is around fifty, but it could stretch to "
                    "seventy if the sponsor tier fills."),
    (6, 33.3, 38.6, "Twenty-eight is a little higher than what we paid last year. We "
                    "were closer to twenty-two."),
    (8, 47.9, 55.8, "That would help. Who actually signs off on this? I can approve up "
                    "to fifteen hundred dollars myself, anything above that goes to our "
                    "director."),
    (10, 63.2, 65.4, "And if we go to seventy?"),
    (12, 70.8, 73.0, "Okay. What's the turnaround?"),
    (14, 81.1, 86.0, "October eleventh is our hard date. When would art need to be "
                     "locked?"),
    (16, 93.7, 98.4, "That's tight but doable. Our designer has the logo files already."),
    (18, 106.8, 114.2, "One more thing - last year a chunk of the sizes came in wrong "
                       "and we ate the cost. How do you handle that?"),
    (20, 123.1, 125.8, "That's better than what we had."),
    (22, 135.9, 141.2, "Send it to me and copy our director. I'll get you a decision by "
                       "Friday."),
    (24, 150.8, 154.0, "Appreciate that. Talk Friday."),
]

# Priya's voice as the microphone hears it coming off the speakers.
PRIYA_BLEED_ONTO_MIC: list[tuple[float, float, str]] = [
    (6.9, 13.2, "Yeah exactly, we're doing our fall kickoff on October eleventh and we "
                "want hoodies for the whole crew"),
    (33.4, 38.6, "twenty eight is a little higher than what we paid last year, we were "
                 "closer to twenty two"),
    (106.9, 114.1, "One more thing, last year a chunk of the sizes came in wrong and we "
                   "ate the cost. How do you handle that?"),
    (135.9, 141.3, "Send it to me and copy our director, I'll get you a decision by "
                   "Friday"),
]

# Dana's voice as the loopback hears it, monitored back through the output
# device. Turn 5 arrives split in two here - the containment case the text
# matcher exists for, where neither half scores well on a plain ratio.
DANA_BLEED_ONTO_SYSTEM: list[tuple[float, float, str]] = [
    (24.1, 28.0, "Fifty to seventy is comfortably in our standard tier"),
    (28.1, 32.8, "base hoodies run twenty-eight dollars apiece at that volume, front and "
                 "back print included"),
    (73.6, 80.5, "Standard production is twelve business days from art approval. If you "
                 "need it faster rush is fifteen percent on top"),
    (126.4, 135.3, "So here's what I'll do, I'll send two quotes today, fifty units and "
                   "seventy units, both on the eight and a half ounce blank, plus a "
                   "lighter blank option for comparison"),
]

# Whisper's two standard failure modes on a real recording: a hallucinated
# sign-off over the dead air after the call, and a filler token it emits at
# speech level but flags itself as probably-not-speech.
HALLUCINATION_AFTER_HANGUP = (155.5, 157.0, "Thanks for watching!")
NON_SPEECH_FILLER = (62.9, 63.1, "you")


def _seg(track, speaker, start, end, text, rms, *, no_speech=0.04, logprob=-0.25):
    return Segment(
        track=track,
        speaker=speaker,
        start=start,
        end=end,
        text=text,
        avg_logprob=logprob,
        no_speech_prob=no_speech,
        rms_dbfs=rms,
    )


def mic_track() -> list[Segment]:
    """What whisper produces from audio_mic.wav: Dana loud, Priya bleeding in."""
    segments = [
        _seg("mic", "Me", start, end, text, LOUD_MIC)
        for _turn, start, end, text in DANA_TURNS
    ]
    # A quieter second copy is also a less confident one.
    segments += [
        _seg("mic", "Me", start, end, text, BLEED_ON_MIC, logprob=-0.95)
        for start, end, text in PRIYA_BLEED_ONTO_MIC
    ]
    start, end, text = NON_SPEECH_FILLER
    segments.append(
        _seg("mic", "Me", start, end, text, LOUD_MIC, no_speech=0.94, logprob=-1.6)
    )
    return sorted(segments, key=lambda s: s.start)


def system_track() -> list[Segment]:
    """What whisper produces from audio_system.wav: Priya loud, Dana bleeding in."""
    segments = [
        _seg("system", "Others", start, end, text, LOUD_SYSTEM)
        for _turn, start, end, text in PRIYA_TURNS
    ]
    segments += [
        _seg("system", "Others", start, end, text, BLEED_ON_SYSTEM, logprob=-0.9)
        for start, end, text in DANA_BLEED_ONTO_SYSTEM
    ]
    start, end, text = HALLUCINATION_AFTER_HANGUP
    segments.append(
        _seg("system", "Others", start, end, text, -71.0, no_speech=0.31, logprob=-1.1)
    )
    return sorted(segments, key=lambda s: s.start)


def expected_transcript() -> list[tuple[str, str]]:
    """(speaker, text) for every turn that should survive the merge, in order."""
    turns = [(start, "Me", text) for _turn, start, _end, text in DANA_TURNS]
    turns += [(start, "Others", text) for _turn, start, _end, text in PRIYA_TURNS]
    turns.sort(key=lambda t: t[0])
    return [(speaker, text) for _start, speaker, text in turns]

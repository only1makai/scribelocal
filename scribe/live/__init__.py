"""Live copilot mode: continuous transcription + grounded suggestions while a
session is running, alongside (not instead of) the existing record-then-
process flow.

Nothing in this package modifies scribe/audio/recorder.py, scribe/pipeline.py,
or scribe/summarize - a live session's audio_mic.wav/audio_system.wav are the
exact same files a record session writes, and the exact same
transcribe_session/summarize_session/process_session finish the job at Stop.
This package only adds the real-time layer on top: a cheap rolling transcript
for the live view (scribe.live.stream), reference-doc grounding
(scribe.live.context, phase 3), and the suggestion engine (scribe.live.copilot,
phase 3).
"""

from __future__ import annotations

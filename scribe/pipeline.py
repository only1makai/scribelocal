"""Shared transcribe-then-summarize orchestration.

The CLI (`scribe process`) and the web UI both drive processing through
`process_session`, so the two surfaces can't drift apart on skip rules, meta.json
bookkeeping, or the order things happen in. Neither surface reimplements the
work itself - this only sequences the existing transcribe/summarize modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .session import write_meta
from .summarize import summarize_session
from .transcribe import transcribe_session

Notify = Callable[[str], None]


@dataclass
class ProcessOutcome:
    """What a process run actually did, for the caller to render."""

    session_dir: Path
    transcript_path: Path | None = None
    transcript_skipped: bool = False
    merge_stats: dict | None = None
    warnings: list[str] = field(default_factory=list)
    summary_path: Path | None = None
    summary_skipped: bool = False
    summary_usage: dict | None = None
    template: str | None = None
    model: str | None = None

    def as_dict(self) -> dict:
        return {
            "session": self.session_dir.name,
            "transcript": str(self.transcript_path) if self.transcript_path else None,
            "transcript_skipped": self.transcript_skipped,
            "merge_stats": self.merge_stats,
            "warnings": self.warnings,
            "summary": str(self.summary_path) if self.summary_path else None,
            "summary_skipped": self.summary_skipped,
            "summary_usage": self.summary_usage,
            "template": self.template,
            "model": self.model,
        }


def process_session(
    session_dir: Path,
    cfg,
    *,
    transcribe: bool = True,
    summarize: bool = True,
    template: str | None = None,
    model: str | None = None,
    force: bool = False,
    notify: Notify | None = None,
) -> ProcessOutcome:
    """Transcribe and/or summarize one session.

    Existing outputs are left alone unless `force` is set. Errors from the
    underlying modules (TranscriptionError, SummarizeError) propagate untouched -
    callers decide how to present them.
    """
    say = notify or (lambda _msg: None)
    outcome = ProcessOutcome(session_dir=session_dir)

    if transcribe:
        transcript_path = session_dir / "transcript.md"
        if transcript_path.exists() and not force:
            say("transcript.md already exists - skipping (use force to redo)")
            outcome.transcript_path = transcript_path
            outcome.transcript_skipped = True
        else:
            result = transcribe_session(session_dir, cfg, notify=say)
            outcome.transcript_path = result.path
            outcome.merge_stats = result.stats.as_dict()
            outcome.warnings.extend(result.warnings)
            for warning in result.warnings:
                say(f"WARNING: {warning}")
            stats = result.stats
            say(
                f"merged {stats.kept} segment(s); dropped {stats.dropped_duplicate} "
                f"duplicate(s), {stats.dropped_silence} silent/non-speech"
            )
            say(f"-> {result.path}")
            write_meta(
                session_dir,
                status="transcribed",
                transcribed_at=datetime.now(timezone.utc).isoformat(),
                merge_stats=outcome.merge_stats,
            )

    if summarize:
        summary_path = session_dir / "summary.md"
        if summary_path.exists() and not force:
            say("summary.md already exists - skipping (use force to redo)")
            outcome.summary_path = summary_path
            outcome.summary_skipped = True
            return outcome

        result = summarize_session(session_dir, cfg, template=template, model=model, notify=say)
        outcome.summary_path = result.path
        outcome.summary_usage = result.usage.as_dict()
        outcome.template = result.template
        outcome.model = result.model
        usage = result.usage
        say(
            f"{usage.calls} API call(s), {usage.input_tokens:,} in / "
            f"{usage.output_tokens:,} out tokens"
        )
        say(f"-> {result.path}")
        write_meta(
            session_dir,
            status="summarized",
            summarized_at=datetime.now(timezone.utc).isoformat(),
            template=result.template,
            summary_model=result.model,
            summary_usage=outcome.summary_usage,
        )

    return outcome

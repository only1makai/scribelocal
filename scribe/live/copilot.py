"""Live suggestion engine - the copilot.

Consumes the rolling live transcript (Phase 2) plus your reference docs, and
produces short, glanceable suggestions while the call is happening.

The central design decision, per the phase brief: **suggestion latency is
decoupled from the transcription window.** The engine never waits for a clean
window boundary. It fires on demand against whatever text the rolling
transcript has produced *so far*, so a suggestion goes out the instant a
trigger fires even though transcription itself runs a beat behind. Windowing
stays exactly as Phase 2 left it.

Three things keep this from becoming an expensive noise generator:

* **Single in flight, cancel-and-replace.** Never two concurrent calls. A
  trigger arriving mid-call supersedes the running one rather than queueing
  behind it - for a live aid, the freshest context wins and a stale answer is
  worse than no answer.
* **New-speech gate.** The auto timer only fires when enough *new* transcript
  has accumulated, so silence doesn't bill you every interval.
* **A "(nothing)" escape.** The model is told to say exactly that when it has
  nothing worth adding, and the engine suppresses it. Without an out, a model
  asked for advice every few seconds will invent filler.

Speed over polish is deliberate here: the default model is a Haiku-class one,
thinking is left off, and output is hard-capped. The high-quality pass is
still `summarize.model` at Stop, untouched.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .context import ReferenceContext
from .stream import LiveChunk

# Callback surfaces. The engine never prints - the CLI (and later the web
# panel) decide how these render.
OnStart = Callable[[str], None]  # (trigger reason)
OnDelta = Callable[[str], None]  # (text fragment, as it streams)
OnDone = Callable[["Suggestion"], None]
OnError = Callable[[str], None]

# The model is told to emit exactly this when it has nothing worth saying.
NOTHING = "(nothing)"

SYSTEM_PROMPT = """\
You are a live copilot for someone who is IN a conversation right now. They are \
glancing at your output mid-sentence while still talking, so it must be readable \
in about one second.

You receive their reference material and a rough, partial transcript of the last \
minute or so. "Me" is the person you are helping; "Others" is whoever they are \
talking to.

The transcript comes from fast live speech recognition: expect mis-heard words, \
missing punctuation, and repeated phrases at chunk boundaries. Read through the \
errors; never comment on transcript quality.

Give at most ONE of these, whichever is most useful right now:
- A direct answer to a question Others just asked, grounded in the reference material.
- A specific fact from the reference material that is relevant to what was just said.
- A short talking point or question that moves the conversation forward.
- A flag about a risk, an inconsistency, or a commitment that was just made.

Rules:
- Under {max_chars} characters. One or two sentences. No preamble, no headings, \
no bullet lists, no markdown.
- Lead with the substance. Never start with "You could" or "Consider".
- When you use the reference material, name the source inline, e.g. (pricing sheet).
- Never invent a fact that is not in the reference material or the transcript. If \
the reference material does not cover it, say so in a few words.
- If you have nothing genuinely useful to add, reply with exactly: {nothing}

Output only the suggestion text."""


class CopilotError(RuntimeError):
    """Raised when the suggestion engine can't run at all."""


@dataclass
class CopilotSettings:
    enabled: bool = True
    # Haiku-class by default: this is the latency-critical path, and the brief
    # chose speed over polish. summarize.model stays the high-quality one.
    model: str = "claude-haiku-4-5"
    interval_seconds: float = 8.0
    max_suggestion_chars: int = 500
    # How much of the recent transcript to send. Bounded twice - by audio time
    # and by characters - because a fast talker can produce a lot of text in a
    # short window, and prompt size is latency.
    window_seconds: float = 45.0
    max_transcript_chars: int = 4000
    # Don't spend a call on a couple of words. Roughly a short sentence.
    min_new_chars: int = 40
    max_output_tokens: int = 200

    @classmethod
    def from_config(cls, cfg) -> "CopilotSettings":
        d = cls()
        return cls(
            enabled=bool(cfg.get("live.suggestions_enabled", d.enabled)),
            model=str(cfg.get("live.suggestion_model", d.model)),
            interval_seconds=float(
                cfg.get("live.suggest_interval_seconds", d.interval_seconds)
            ),
            max_suggestion_chars=int(
                cfg.get("live.max_suggestion_chars", d.max_suggestion_chars)
            ),
            window_seconds=float(cfg.get("live.suggestion_window_seconds", d.window_seconds)),
            max_transcript_chars=int(
                cfg.get("live.suggestion_transcript_chars", d.max_transcript_chars)
            ),
            min_new_chars=int(cfg.get("live.suggestion_min_new_chars", d.min_new_chars)),
            max_output_tokens=int(
                cfg.get("live.suggestion_max_tokens", d.max_output_tokens)
            ),
        )


@dataclass
class Suggestion:
    """One suggestion attempt, successful or not."""

    trigger: str  # "timer" | "manual" | "pause" (pause arrives in step 3)
    text: str = ""
    model: str = ""
    at: str = ""
    first_token_ms: float | None = None  # trigger -> first visible word
    total_ms: float | None = None
    transcript_chars: int = 0
    context_chars: int = 0
    truncated: bool = False
    superseded: bool = False  # a newer trigger replaced this one mid-stream
    cancelled: bool = False  # the session ended mid-stream
    error: str | None = None

    def as_log(self) -> dict:
        """The suggestions.jsonl record.

        Logs the model's OUTPUT and character COUNTS of what was sent - never
        the transcript text itself. The spoken content already lives in
        live_transcript.jsonl; duplicating it here would spread the same
        speech across more files for no benefit.
        """
        record = asdict(self)
        for key in ("first_token_ms", "total_ms"):
            if record[key] is not None:
                record[key] = round(record[key], 1)
        # "text" is the dataclass field; "suggestion" reads better in a log file
        # someone is scanning by eye afterwards.
        record["suggestion"] = record.pop("text")
        return record


# ---------------------------------------------------------------------------
# Transcript buffer (pure - no threads, no clock, no API)
# ---------------------------------------------------------------------------


class TranscriptBuffer:
    """Recent live-transcript text, bounded by audio time and characters.

    Recency is measured on the AUDIO timeline (chunk window_end), not the wall
    clock: both tracks' taps start together, so their timestamps are directly
    comparable, and using audio time keeps this class free of any clock and
    therefore trivially testable.
    """

    def __init__(self) -> None:
        self._entries: list[tuple[float, str, str]] = []  # (window_end, speaker, text)
        self._chars_since_mark = 0
        self._last_by_track: dict[str, str] = {}

    def add(self, chunk: LiveChunk) -> bool:
        """Record a chunk. Returns False if it was dropped as a duplicate.

        Consecutive windows overlap by design, so the same sentence can arrive
        twice from one track; an exact repeat adds nothing but prompt size.
        """
        text = (chunk.text or "").strip()
        if not text:
            return False
        if self._last_by_track.get(chunk.track) == text:
            return False
        self._last_by_track[chunk.track] = text
        self._entries.append((chunk.window_end, chunk.speaker, text))
        self._chars_since_mark += len(text)
        return True

    def new_chars(self) -> int:
        return self._chars_since_mark

    def mark(self) -> None:
        """Reset the new-speech counter (called when a suggestion is sent)."""
        self._chars_since_mark = 0

    def recent(self, window_seconds: float, max_chars: int) -> str:
        """Render the tail of the conversation as `Speaker: text` lines."""
        if not self._entries:
            return ""
        latest = max(end for end, _s, _t in self._entries)
        cutoff = latest - window_seconds
        lines = [
            f"{speaker}: {text}"
            for end, speaker, text in self._entries
            if end >= cutoff
        ]
        if not lines:  # window shorter than a single chunk - keep the last one
            end, speaker, text = self._entries[-1]
            lines = [f"{speaker}: {text}"]
        rendered = "\n".join(lines)
        if len(rendered) > max_chars:
            rendered = rendered[-max_chars:]  # keep the END - it's the freshest
            rendered = rendered.split("\n", 1)[-1]  # drop the half-line at the cut
        return rendered


def build_prompt(context_text: str, transcript: str) -> str:
    parts = []
    if context_text.strip():
        parts.append(context_text.strip())
    parts.append(
        "<transcript note=\"live speech recognition, may contain errors\">\n"
        f"{transcript.strip()}\n</transcript>"
    )
    parts.append("What is the single most useful thing to say or know right now?")
    return "\n\n".join(parts)


def cap_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Hard-cap a suggestion, trimming at a word boundary where possible."""
    text = text.strip()
    if len(text) <= max_chars:
        return text, False
    head = text[:max_chars]
    cut = head.rfind(" ")
    if cut > max_chars * 0.6:
        head = head[:cut]
    return head.rstrip(" ,;:-") + "...", True


# ---------------------------------------------------------------------------
# API access
# ---------------------------------------------------------------------------


def build_client():
    """Reuse the summarizer's credential resolution.

    Imported rather than reimplemented so there is exactly one place that
    knows how this project finds credentials. scribe/summarize is not modified
    by this phase - this only calls into it.
    """
    from ..summarize import _build_client, SummarizeError

    try:
        return _build_client()
    except SummarizeError as exc:
        raise CopilotError(str(exc)) from exc


def describe_api_error(exc: Exception) -> str:
    """Turn an SDK exception into something actionable in the live output."""
    import anthropic

    if isinstance(exc, anthropic.AuthenticationError):
        return ("Anthropic API rejected the credentials (401). Set ANTHROPIC_API_KEY "
                "in .env, or run `ant auth login`.")
    if isinstance(exc, anthropic.NotFoundError):
        return f"model not found (404) - check live.suggestion_model. {exc}"
    if isinstance(exc, anthropic.BadRequestError):
        return f"API rejected the request (400): {exc}"
    if isinstance(exc, anthropic.RateLimitError):
        retry = "unknown"
        response = getattr(exc, "response", None)
        if response is not None:
            retry = response.headers.get("retry-after", "unknown")
        return f"rate limited (429); retry after {retry}s"
    if isinstance(exc, anthropic.APIStatusError):
        return f"API error {exc.status_code}: {exc}"
    if isinstance(exc, anthropic.APIConnectionError):
        return f"could not reach the Anthropic API: {exc}"
    if isinstance(exc, TypeError) and "authentication" in str(exc).lower():
        # The SDK raises TypeError while building the request when no
        # credentials resolve at all - it never reaches the server.
        return ("no Anthropic credentials found. Copy .env.example to .env and set "
                "ANTHROPIC_API_KEY, or run `ant auth login`.")
    return f"{type(exc).__name__}: {exc}"


def make_distiller(settings: CopilotSettings, client=None) -> Callable[[str, int], str]:
    """A one-shot condenser for oversized reference docs (session start only).

    Uses the fast suggestion model on purpose: this runs while you're trying to
    join a call, so a 20-second wait to start recording would be worse than a
    slightly rougher summary.
    """

    def distill(text: str, target_chars: int) -> str:
        active = client or build_client()
        message = active.messages.create(
            model=settings.model,
            max_tokens=max(256, target_chars // 3),
            system=(
                "Condense the reference material for use as live prompt context. "
                f"Stay under {target_chars} characters. Keep concrete specifics - "
                "numbers, prices, names, dates, policies, exact terms - and drop "
                "narrative, examples and repetition. Output only the condensed "
                "material, organised under short headings."
            ),
            messages=[{"role": "user", "content": text}],
        )
        return "\n".join(b.text for b in message.content if b.type == "text").strip()

    return distill


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class SuggestionEngine:
    """Debounced, single-in-flight, cancel-and-replace suggestion worker."""

    def __init__(
        self,
        session_dir: Path,
        settings: CopilotSettings,
        context: ReferenceContext | None = None,
        *,
        client=None,
        on_start: OnStart | None = None,
        on_delta: OnDelta | None = None,
        on_done: OnDone | None = None,
        on_error: OnError | None = None,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.settings = settings
        self.context = context
        self._client = client
        self._on_start = on_start or (lambda _r: None)
        self._on_delta = on_delta or (lambda _t: None)
        self._on_done = on_done or (lambda _s: None)
        self._on_error = on_error or (lambda _m: None)

        self.buffer = TranscriptBuffer()
        self._log_path = self.session_dir / "suggestions.jsonl"
        self._log_lock = threading.Lock()

        self._cv = threading.Condition()
        self._pending: str | None = None
        self._supersede = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sent = 0.0

        self.suggestions: list[Suggestion] = []
        self.calls = 0
        self.errors = 0
        self.superseded = 0
        # A missing API key fails identically every time. Reporting it once per
        # interval for a whole call is noise, so the auto timer stands down
        # after a few in a row - the errors are still surfaced, just not on a
        # loop. Explicit triggers keep working.
        self._consecutive_errors = 0
        self.auto_disabled = False

    # ---------------------------------------------------------------- inputs

    def note_chunk(self, chunk: LiveChunk) -> None:
        """Feed one live-transcript chunk. Safe to call from the tap threads."""
        with self._cv:
            self.buffer.add(chunk)

    def trigger(self, reason: str = "manual") -> None:
        """Ask for a suggestion now. Non-blocking.

        If a call is already streaming it is superseded rather than queued:
        whatever comes back next should reflect the newest context.
        """
        with self._cv:
            self._pending = reason
            self._supersede = True
            self._cv.notify_all()

    # --------------------------------------------------------------- control

    def start(self) -> None:
        if not self.settings.enabled:
            return
        self._last_sent = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="live-copilot", daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        with self._cv:
            self._supersede = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=20)
        # `is not None`, not truthiness: a sub-millisecond latency rounds to 0.0
        # and would silently drop out of the median.
        latencies = [
            s.first_token_ms for s in self.suggestions if s.first_token_ms is not None
        ]
        return {
            "calls": self.calls,
            "delivered": len([s for s in self.suggestions if s.text and not s.error]),
            "errors": self.errors,
            "superseded": self.superseded,
            "model": self.settings.model,
            "median_first_token_ms": (
                round(sorted(latencies)[len(latencies) // 2], 1) if latencies else None
            ),
        }

    # ------------------------------------------------------------ scheduling

    def _due(self) -> bool:
        """Auto-timer gate: enough elapsed AND enough new speech to be worth it.

        An interval of 0 or less turns the auto timer off entirely, leaving
        only explicit triggers - it can't mean "as often as possible" without
        becoming a billing accident.
        """
        if self.settings.interval_seconds <= 0 or self.auto_disabled:
            return False
        if time.monotonic() - self._last_sent < self.settings.interval_seconds:
            return False
        return self.buffer.new_chars() >= self.settings.min_new_chars

    AUTO_DISABLE_AFTER_ERRORS = 3

    def _record_outcome(self, suggestion: Suggestion) -> None:
        """Stand the auto timer down after repeated identical failures."""
        if suggestion.error:
            self._consecutive_errors += 1
            if (
                self._consecutive_errors >= self.AUTO_DISABLE_AFTER_ERRORS
                and not self.auto_disabled
            ):
                self.auto_disabled = True
                self._on_error(
                    f"{self._consecutive_errors} suggestion calls failed in a row - "
                    "pausing automatic suggestions for this session. Fix the cause "
                    "and restart to re-enable."
                )
        elif not suggestion.superseded and not suggestion.cancelled:
            self._consecutive_errors = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                if self._pending is None:
                    self._cv.wait(timeout=0.2)
                reason = self._pending
                if reason is None and self._due():
                    reason = "timer"
                self._pending = None
                self._supersede = False
                if reason is None:
                    continue
                transcript = self.buffer.recent(
                    self.settings.window_seconds, self.settings.max_transcript_chars
                )
                if not transcript.strip():
                    continue
                self.buffer.mark()
            self._last_sent = time.monotonic()
            self._one(reason, transcript)

    # -------------------------------------------------------------- one call

    def _one(self, reason: str, transcript: str) -> None:
        context_text = self.context.text if self.context else ""
        suggestion = Suggestion(
            trigger=reason,
            model=self.settings.model,
            at=datetime.now(timezone.utc).isoformat(),
            transcript_chars=len(transcript),
            context_chars=len(context_text),
        )
        self.calls += 1
        self._on_start(reason)
        started = time.monotonic()

        try:
            # Build once and keep it. A fresh Anthropic() per suggestion would
            # redo credential resolution and, worse, throw away the HTTP
            # connection pool - paying a new TLS handshake on exactly the
            # metric this phase is optimising for.
            if self._client is None:
                self._client = build_client()
            client = self._client
        except CopilotError as exc:
            suggestion.error = str(exc)
            self.errors += 1
            self._on_error(suggestion.error)
            self._finish(suggestion)
            return

        pieces: list[str] = []
        try:
            with client.messages.stream(
                model=self.settings.model,
                max_tokens=self.settings.max_output_tokens,
                system=SYSTEM_PROMPT.format(
                    max_chars=self.settings.max_suggestion_chars, nothing=NOTHING
                ),
                messages=[{"role": "user", "content": build_prompt(context_text, transcript)}],
            ) as stream:
                for piece in stream.text_stream:
                    if self._is_superseded():
                        # Distinguish the two reasons a stream stops early:
                        # "a newer trigger won" is normal operation, "the call
                        # ended" is shutdown. Reporting shutdown as the former
                        # reads like a bug in the output.
                        if self._stop.is_set():
                            suggestion.cancelled = True
                        else:
                            suggestion.superseded = True
                            self.superseded += 1
                        break
                    if suggestion.first_token_ms is None and piece.strip():
                        suggestion.first_token_ms = (time.monotonic() - started) * 1000.0
                    pieces.append(piece)
                    self._on_delta(piece)
                    if sum(len(p) for p in pieces) >= self.settings.max_suggestion_chars:
                        break  # hard cap: stop paying for text we'd only trim
        except Exception as exc:
            suggestion.error = describe_api_error(exc)
            self.errors += 1
            self._on_error(suggestion.error)
            self._finish(suggestion)
            return

        suggestion.total_ms = (time.monotonic() - started) * 1000.0
        text, truncated = cap_text("".join(pieces), self.settings.max_suggestion_chars)
        suggestion.truncated = truncated
        # The model's explicit "nothing useful here" - not an error, not output.
        suggestion.text = "" if text.lower().startswith(NOTHING) else text
        self._finish(suggestion)

    def _is_superseded(self) -> bool:
        with self._cv:
            return self._supersede or self._stop.is_set()

    def _finish(self, suggestion: Suggestion) -> None:
        self.suggestions.append(suggestion)
        self._record_outcome(suggestion)
        self._log(suggestion)
        self._on_done(suggestion)

    def _log(self, suggestion: Suggestion) -> None:
        try:
            with self._log_lock:
                self.session_dir.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(suggestion.as_log()) + "\n")
        except OSError as exc:
            self._on_error(f"could not write suggestions.jsonl: {exc}")

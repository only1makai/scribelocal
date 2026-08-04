"""Summarization: transcript.md in, summary.md out, via the Claude API.

Long transcripts are chunked, each chunk is digested, and the digests are
synthesized into one final summary. Every API failure is raised as a
SummarizeError carrying the status, the API's own message, and the request id -
nothing is caught and quietly turned into an empty summary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .templates import DEFAULT_TEMPLATE, TEMPLATES, Template, get_template

__all__ = [
    "DEFAULT_TEMPLATE",
    "TEMPLATES",
    "SummarizeError",
    "SummaryResult",
    "summarize_session",
]

# Whisper output is dense; 150k chars is comfortably inside the context window
# while leaving room for the model's own reasoning and output.
DEFAULT_MAX_CHUNK_CHARS = 150_000
DEFAULT_MAX_OUTPUT_TOKENS = 8_000


class SummarizeError(RuntimeError):
    """Any failure that stopped a summary from being produced."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, message) -> None:
        usage = getattr(message, "usage", None)
        if usage is not None:
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.calls += 1

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class SummaryResult:
    path: Path
    model: str
    template: str
    chunk_count: int
    usage: Usage = field(default_factory=Usage)


def summarize_session(
    session_dir: Path,
    cfg,
    *,
    template: str | None = None,
    model: str | None = None,
    notify: Callable[[str], None] | None = None,
) -> SummaryResult:
    """Summarize a session's transcript.md and write summary.md."""
    say = notify or (lambda _msg: None)

    transcript_path = session_dir / "transcript.md"
    if not transcript_path.exists():
        raise SummarizeError(
            f"no transcript at {transcript_path}. Run transcription first "
            f"(scribe process {session_dir.name})."
        )
    transcript = transcript_path.read_text(encoding="utf-8").strip()
    if not transcript:
        raise SummarizeError(f"{transcript_path} is empty - nothing to summarize.")

    meta = _read_meta(session_dir)
    template_key = template or meta.get("template") or cfg.get("summarize.template")
    try:
        tmpl = get_template(template_key)
    except KeyError as exc:
        raise SummarizeError(str(exc)) from exc

    model_id = model or cfg.get("summarize.model")
    if not model_id:
        raise SummarizeError("no model configured (summarize.model in config.yaml)")

    max_chunk_chars = int(cfg.get("summarize.max_chunk_chars", DEFAULT_MAX_CHUNK_CHARS))
    max_tokens = int(cfg.get("summarize.max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS))
    thinking = _thinking_param(cfg.get("summarize.thinking", "adaptive"))
    effort = cfg.get("summarize.effort")

    client = _build_client()
    usage = Usage()
    chunks = chunk_transcript(transcript, max_chunk_chars)

    if len(chunks) == 1:
        say(f"  summarizing with {model_id} (template: {tmpl.key}) ...")
        body = _call(
            client,
            model=model_id,
            system=tmpl.system_prompt(),
            user=f"Here is the full transcript.\n\n<transcript>\n{transcript}\n</transcript>",
            max_tokens=max_tokens,
            thinking=thinking,
            effort=effort,
            usage=usage,
        )
    else:
        say(f"  transcript is {len(transcript):,} chars - summarizing in {len(chunks)} chunks")
        digests: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            say(f"  chunk {index}/{len(chunks)} ...")
            digests.append(
                _call(
                    client,
                    model=model_id,
                    system=_chunk_system_prompt(tmpl, index, len(chunks)),
                    user=(
                        f"Here is part {index} of {len(chunks)} of the transcript.\n\n"
                        f"<transcript_part>\n{chunk}\n</transcript_part>"
                    ),
                    max_tokens=max_tokens,
                    thinking=thinking,
                    effort=effort,
                    usage=usage,
                )
            )
        say("  synthesizing final summary ...")
        joined = "\n\n".join(
            f"<part index=\"{i}\">\n{d}\n</part>" for i, d in enumerate(digests, start=1)
        )
        body = _call(
            client,
            model=model_id,
            system=_synthesis_system_prompt(tmpl),
            user=(
                f"Here are the {len(digests)} partial digests, in chronological order. "
                f"Merge them into one summary of the whole conversation.\n\n{joined}"
            ),
            max_tokens=max_tokens,
            thinking=thinking,
            effort=effort,
            usage=usage,
        )

    out_path = session_dir / "summary.md"
    out_path.write_text(
        _render(body, meta=meta, session_dir=session_dir, template=tmpl, model=model_id),
        encoding="utf-8",
    )
    return SummaryResult(out_path, model_id, tmpl.key, len(chunks), usage)


def chunk_transcript(transcript: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries so no speaker turn is cut mid-sentence."""
    if max_chars <= 0 or len(transcript) <= max_chars:
        return [transcript]

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for block in transcript.split("\n\n"):
        block_size = len(block) + 2
        if current and size + block_size > max_chars:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        # A single paragraph over the limit still has to go somewhere; it lands
        # in a chunk of its own rather than being silently truncated.
        current.append(block)
        size += block_size
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _chunk_system_prompt(tmpl: Template, index: int, total: int) -> str:
    from .templates import BASE_RULES

    return (
        f"{BASE_RULES}\n\n{tmpl.focus}\n\n"
        f"You are reading part {index} of {total} of a longer transcript. Produce a "
        "detailed digest of THIS PART ONLY, under these headings: ## TL;DR, "
        "## Key Decisions, ## Action Items, ## Open Questions, ## Notable Moments.\n"
        "Keep `[hh:mm:ss]` timestamps on anything time-specific - a later pass merges "
        "these digests and needs them. Be comprehensive rather than brief: this digest "
        "replaces the raw text for everything that follows. If a thread is clearly still "
        "unfolding at the end of this part, note that instead of guessing how it resolves."
    )


def _synthesis_system_prompt(tmpl: Template) -> str:
    from .templates import BASE_RULES, SECTIONS

    return (
        f"{BASE_RULES}\n\n{tmpl.focus}\n\n"
        "You are given ordered digests of consecutive parts of one conversation. Merge "
        "them into a single summary of the whole thing. Collapse items that appear in "
        "more than one part into one entry, and let later parts override earlier ones "
        "when something was revisited or reversed - a decision that was later undone is "
        "not a decision. Keep the `[hh:mm:ss]` timestamps.\n\n" + SECTIONS
    )


def _thinking_param(setting) -> dict | None:
    """Map the config value to a thinking parameter, or None to omit it."""
    if setting is None:
        return None
    value = str(setting).strip().lower()
    if value in {"", "none", "omit"}:
        return None
    if value in {"off", "false", "disabled", "no"}:
        return {"type": "disabled"}
    if value in {"adaptive", "on", "true", "yes"}:
        return {"type": "adaptive"}
    raise SummarizeError(
        f"invalid summarize.thinking value '{setting}' (use: adaptive | off | none)"
    )


def _build_client():
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise SummarizeError(
            'the anthropic SDK is not installed. Install it with:\n  pip install -e ".[summarize]"'
        ) from exc
    try:
        # Credentials resolve from ANTHROPIC_API_KEY (loaded from .env by
        # load_config) or an `ant auth login` profile.
        return anthropic.Anthropic()
    except Exception as exc:
        raise SummarizeError(f"could not create the Anthropic client: {exc}") from exc


def _call(
    client,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    thinking: dict | None,
    effort: str | None,
    usage: Usage,
) -> str:
    """One Messages API call. Streams, so long transcripts don't hit a timeout."""
    import anthropic

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if thinking is not None:
        kwargs["thinking"] = thinking
    if effort:
        kwargs["output_config"] = {"effort": str(effort)}

    try:
        with client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()
    except anthropic.AuthenticationError as exc:
        raise SummarizeError(
            "Anthropic API rejected the credentials (401). Set ANTHROPIC_API_KEY in .env, "
            f"or run `ant auth login`. API said: {exc}"
        ) from exc
    except anthropic.NotFoundError as exc:
        raise SummarizeError(
            f"model '{model}' was not found (404). Check summarize.model in config.yaml. "
            f"API said: {exc}"
        ) from exc
    except anthropic.BadRequestError as exc:
        raise SummarizeError(
            f"Anthropic API rejected the request (400): {exc}\n"
            f"If this names 'thinking' or 'effort', model '{model}' does not support the "
            "configured value - set summarize.thinking: off and/or summarize.effort: null."
        ) from exc
    except anthropic.RateLimitError as exc:
        retry_after = exc.response.headers.get("retry-after", "unknown")
        raise SummarizeError(
            f"rate limited by the Anthropic API (429); retry after {retry_after}s. "
            f"API said: {exc}"
        ) from exc
    except anthropic.APIStatusError as exc:
        raise SummarizeError(
            f"Anthropic API error {exc.status_code} ({getattr(exc, 'type', 'unknown')}): {exc} "
            f"[request_id={getattr(exc, 'request_id', None)}]"
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise SummarizeError(f"could not reach the Anthropic API: {exc}") from exc
    except TypeError as exc:
        # With no credentials at all the SDK raises TypeError while building the
        # request - it never reaches the server, so it isn't an APIError and
        # would otherwise escape as a bare traceback.
        if "authentication" in str(exc).lower():
            raise SummarizeError(
                "no Anthropic credentials found. Copy .env.example to .env and set "
                "ANTHROPIC_API_KEY (or run `ant auth login`), then try again."
            ) from exc
        raise SummarizeError(f"bad request to the Anthropic SDK: {exc}") from exc

    usage.add(message)

    if message.stop_reason == "refusal":
        raise SummarizeError(
            "the model declined to summarize this transcript "
            f"(stop_reason=refusal, details={getattr(message, 'stop_details', None)})"
        )

    text = "\n".join(block.text for block in message.content if block.type == "text").strip()
    if not text:
        raise SummarizeError(
            f"the model returned no text (stop_reason={message.stop_reason}, "
            f"request_id={getattr(message, '_request_id', None)})"
        )
    if message.stop_reason == "max_tokens":
        # Truncation is a real failure of the deliverable, not a warning to bury:
        # a summary cut off mid-section will read as complete.
        raise SummarizeError(
            f"the summary hit the {max_tokens}-token output limit and was truncated. "
            "Raise summarize.max_output_tokens in config.yaml and re-run."
        )
    return text


def _render(body: str, *, meta: dict, session_dir: Path, template: Template, model: str) -> str:
    title = meta.get("title") or session_dir.name
    generated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    lines = [f"# Summary — {title}", ""]
    detail = [f"Template: {template.label}", f"Model: {model}", f"Generated: {generated}"]
    if meta.get("duration_seconds"):
        minutes = int(float(meta["duration_seconds"]) // 60)
        detail.insert(0, f"Duration: {minutes} min")
    lines += ["*" + " · ".join(detail) + "*", "", body.strip(), ""]
    return "\n".join(lines)


def _read_meta(session_dir: Path) -> dict:
    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

"""Reference-doc grounding for live suggestions.

Attach your own material at session start - a pricing sheet, a sales playbook,
interview notes - and it gets injected into every suggestion prompt, so the
copilot answers from YOUR facts instead of generic ones ("they asked about
rush orders, your sheet says 5-day turnaround is +15%").

Deliberately API-free: this module reads, bounds, and formats text, and takes
an optional `distiller` callable for the one case that needs a model (docs
bigger than the budget). That keeps the whole thing unit-testable without
credentials, and keeps the API surface in copilot.py where the rest of the
Anthropic access already lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

Notify = Callable[[str], None]
# (combined_text, target_chars) -> condensed text
Distiller = Callable[[str, int], str]

# Suffixes we'll read as text. Anything else is refused rather than guessed at:
# a PDF or .docx read as bytes would inject mojibake into every prompt.
TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text", ".rst", ".csv", ".json", ".yaml", ".yml"}

TRUNCATION_MARKER = "\n[... truncated to fit the context budget ...]"


class ContextError(RuntimeError):
    """Raised when an attached document can't be used at all."""


@dataclass
class ReferenceDoc:
    path: Path
    name: str
    text: str
    original_chars: int
    truncated: bool = False

    def as_meta(self) -> dict:
        """Metadata only - the document's CONTENTS never go in meta.json."""
        return {
            "path": str(self.path),
            "name": self.name,
            "chars": self.original_chars,
            "truncated": self.truncated,
        }


@dataclass
class ReferenceContext:
    docs: list[ReferenceDoc] = field(default_factory=list)
    text: str = ""  # the assembled block injected into prompts
    budget: int = 0
    distilled: bool = False  # true if a model condensed the docs to fit
    warnings: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return len(self.text)

    def is_empty(self) -> bool:
        return not self.text.strip()

    def as_meta(self) -> dict:
        return {
            "docs": [d.as_meta() for d in self.docs],
            "chars": self.chars,
            "budget": self.budget,
            "distilled": self.distilled,
        }


def read_documents(paths) -> tuple[list[ReferenceDoc], list[str]]:
    """Read attached files. Unreadable ones warn rather than abort the session -
    a bad path shouldn't stop you recording a call that's about to start."""
    docs: list[ReferenceDoc] = []
    warnings: list[str] = []
    for raw in paths or []:
        path = Path(raw).expanduser()
        if not path.exists():
            warnings.append(f"context file not found, skipping: {path}")
            continue
        if path.is_dir():
            warnings.append(f"context path is a directory, skipping: {path}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            warnings.append(
                f"unsupported context file type '{path.suffix}', skipping: {path.name} "
                f"(supported: {', '.join(sorted(TEXT_SUFFIXES))})"
            )
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            warnings.append(f"could not read {path.name}: {exc}")
            continue
        text = text.strip()
        if not text:
            warnings.append(f"context file is empty, skipping: {path.name}")
            continue
        docs.append(ReferenceDoc(path=path, name=path.name, text=text,
                                 original_chars=len(text)))
    return docs, warnings


def _truncate_at_boundary(text: str, limit: int) -> str:
    """Cut to `limit` chars, preferring a paragraph then a line boundary, so a
    chunk doesn't end mid-sentence."""
    if len(text) <= limit:
        return text
    room = max(0, limit - len(TRUNCATION_MARKER))
    head = text[:room]
    for sep in ("\n\n", "\n", ". "):
        cut = head.rfind(sep)
        if cut > room * 0.5:  # only honour a boundary that isn't wastefully early
            head = head[:cut]
            break
    return head.rstrip() + TRUNCATION_MARKER


def allocate_budget(sizes: list[int], budget: int) -> list[int]:
    """Fair-share split of `budget` across docs of the given sizes.

    Docs smaller than an equal share take only what they need and release the
    rest, so one short pricing sheet is never truncated just because it was
    attached alongside a long playbook.
    """
    n = len(sizes)
    if n == 0:
        return []
    allocation = [0] * n
    remaining = max(0, budget)
    pending = set(range(n))
    while pending:
        share = remaining // len(pending)
        fits = [i for i in pending if sizes[i] <= share]
        if not fits:
            for i in pending:
                allocation[i] = share
            break
        for i in fits:
            allocation[i] = sizes[i]
            remaining -= sizes[i]
            pending.discard(i)
    return allocation


def render(docs: list[ReferenceDoc]) -> str:
    """Format docs as the block injected into the suggestion prompt."""
    if not docs:
        return ""
    parts = ["<reference_material>"]
    for doc in docs:
        parts.append(f"\n## {doc.name}\n{doc.text.strip()}")
    parts.append("\n</reference_material>")
    return "\n".join(parts)


def build_context(
    paths,
    budget: int,
    *,
    distiller: Distiller | None = None,
    notify: Notify | None = None,
) -> ReferenceContext:
    """Load attached docs and fit them into `budget` characters.

    Under budget: used verbatim. Over budget: condensed once by `distiller` if
    one is supplied (a single session-start call, never on the live path), and
    if that isn't available or fails, truncated per-document with a warning.
    Falling back to truncation rather than failing matters - no API key should
    still leave you with a recording and a live transcript.
    """
    say = notify or (lambda _m: None)
    docs, warnings = read_documents(paths)
    context = ReferenceContext(docs=docs, budget=budget, warnings=list(warnings))
    for warning in warnings:
        say(f"context: WARNING {warning}")
    if not docs:
        return context

    total = sum(d.original_chars for d in docs)
    for doc in docs:
        say(f"context: loaded {doc.name} ({doc.original_chars:,} chars)")

    if total <= budget:
        context.text = render(docs)
        say(f"context: {total:,} chars within the {budget:,} budget - using verbatim")
        return context

    if distiller is not None:
        say(f"context: {total:,} chars exceeds the {budget:,} budget - condensing once...")
        try:
            condensed = distiller(render(docs), budget)
        except Exception as exc:
            # Any distiller failure degrades to truncation rather than taking
            # the session down with it.
            message = f"could not condense reference docs ({exc}); truncating instead"
            context.warnings.append(message)
            say(f"context: WARNING {message}")
        else:
            condensed = (condensed or "").strip()
            if condensed:
                context.distilled = True
                summary_doc = ReferenceDoc(
                    path=Path("(condensed)"),
                    name=f"summary of {len(docs)} reference doc(s)",
                    text=_truncate_at_boundary(condensed, budget),
                    original_chars=total,
                    truncated=len(condensed) > budget,
                )
                context.text = render([summary_doc])
                say(f"context: condensed to {context.chars:,} chars")
                return context
            message = "condenser returned nothing; truncating instead"
            context.warnings.append(message)
            say(f"context: WARNING {message}")

    allocation = allocate_budget([d.original_chars for d in docs], budget)
    for doc, limit in zip(docs, allocation):
        if doc.original_chars > limit:
            doc.text = _truncate_at_boundary(doc.text, limit)
            doc.truncated = True
    context.text = render(docs)
    trimmed = [d.name for d in docs if d.truncated]
    if trimmed:
        message = (
            f"reference docs exceeded the {budget:,} char budget - truncated: "
            f"{', '.join(trimmed)}"
        )
        context.warnings.append(message)
        say(f"context: WARNING {message}")
    return context

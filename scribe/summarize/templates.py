"""Summary templates.

Every template produces the same five sections - what changes is what the model
is told to treat as important. A lecture's "decisions" are the instructor's
rulings on scope and grading; a sales call's are commitments and next steps.
"""

from __future__ import annotations

from dataclasses import dataclass

SECTIONS = """Structure the summary with exactly these five sections, in this order,
using these exact Markdown headings:

## TL;DR
Three to five sentences. What happened, and what it means for the reader.

## Key Decisions
What was actually decided or settled. One bullet each, with enough context to
stand alone. If nothing was decided, write "None recorded."

## Action Items
One bullet per commitment, formatted `- [ ] **Owner** — task (deadline if stated)`.
Attribute an owner only when the transcript identifies one; otherwise use
`**Unassigned**`. Never invent a name. If there are none, write "None recorded."

## Open Questions
Questions raised but not answered, and unresolved disagreements. If none, write
"None recorded."

## Notable Moments
Anything worth going back to the recording for - a sharp explanation, a reversal,
a number that matters, a moment of tension. Include the `[hh:mm:ss]` timestamp
from the transcript where available."""

BASE_RULES = """You are summarizing a transcript of a real recorded conversation.

Ground rules:
- Speaker labels come from audio channels: "Me" is the person who made the
  recording, "Others" is everyone heard through their speakers. Names appear only
  when someone says them out loud.
- Work only from the transcript. Do not infer facts that are not in it, and do
  not smooth over gaps - transcripts contain mishearings, crosstalk, and dropped
  words.
- When the transcript is genuinely ambiguous about who said or committed to
  something, say so rather than guessing.
- Write plainly. No preamble, no "this transcript covers" throat-clearing -
  start with the first heading."""


@dataclass(frozen=True)
class Template:
    key: str
    label: str
    focus: str

    def system_prompt(self) -> str:
        return f"{BASE_RULES}\n\n{self.focus}\n\n{SECTIONS}"


TEMPLATES: dict[str, Template] = {
    "general": Template(
        key="general",
        label="General",
        focus=(
            "This is a general conversation. Weigh anything that changes what "
            "someone will do next above background discussion."
        ),
    ),
    "class-lecture": Template(
        key="class-lecture",
        label="Class lecture",
        focus=(
            "This is a class or lecture. Prioritize the concepts taught and how "
            "they were explained, worked examples, and anything flagged as "
            "assessable. Treat instructor rulings - scope, grading, deadlines, "
            "exam coverage - as decisions, and assignments as action items. "
            "Preserve technical terms, formulas, and cited sources exactly."
        ),
    ),
    "work-shift": Template(
        key="work-shift",
        label="Work shift",
        focus=(
            "This is a work shift or handover. Prioritize what happened on shift, "
            "what is still outstanding for whoever comes next, and anything that "
            "went wrong. Treat process changes and escalations as decisions, and "
            "follow-ups as action items. Preserve times, ticket or order numbers, "
            "and equipment or location names exactly."
        ),
    ),
    "club-meeting": Template(
        key="club-meeting",
        label="Club meeting",
        focus=(
            "This is a club or organization meeting. Prioritize motions, votes, "
            "role assignments, budget, and event planning. Treat anything agreed "
            "by the group as a decision, recording the outcome and any dissent. "
            "Preserve dates, dollar amounts, and event names exactly."
        ),
    ),
    "sales-call": Template(
        key="sales-call",
        label="Sales call",
        focus=(
            "This is a sales call. Prioritize what the prospect needs, objections "
            "raised and how they landed, budget/authority/timeline signals, and "
            "commitments made by either side. Treat pricing, scope, and next-step "
            "agreements as decisions. Distinguish clearly between what the "
            "prospect asked for and what was promised to them."
        ),
    ),
}

DEFAULT_TEMPLATE = "general"


def get_template(key: str | None) -> Template:
    """Look up a template by key, with a clear error listing valid options."""
    name = (key or DEFAULT_TEMPLATE).strip().lower()
    if name not in TEMPLATES:
        raise KeyError(f"unknown template '{key}'. Available: {', '.join(sorted(TEMPLATES))}")
    return TEMPLATES[name]

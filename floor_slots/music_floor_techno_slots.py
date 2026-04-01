"""Techno prompt fields for compositional Music Floor control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

PROMPT_FIELD_COUNT = 6
PROMPT_OPTION_COUNT = 6
DEFAULT_TECHNO_PROMPT_ANCHOR = "minimal techno"
DEFAULT_TECHNO_SCHEMA_NAME = "techno_fields_v1"


@dataclass(frozen=True)
class SlotChoice:
    label: str
    prompt: str


@dataclass(frozen=True)
class PromptField:
    name: str
    choices: Tuple[SlotChoice, ...]


TECHNO_PROMPT_FIELDS: Tuple[PromptField, ...] = (
    PromptField(
        name="Beat",
        choices=(
            SlotChoice("Four Floor", "steady four on the floor"),
            SlotChoice("Drive", "driving pulse"),
            SlotChoice("Roll", "rolling syncopation"),
            SlotChoice("Slow Burn", "slow-burning groove"),
            SlotChoice("Low Slung", "low-slung groove"),
            SlotChoice("Stomp", "straight warehouse stomp"),
        ),
    ),
    PromptField(
        name="Bass",
        choices=(
            SlotChoice("Sub Roll", "rolling sub bass"),
            SlotChoice("Rubber", "rubbery bassline"),
            SlotChoice("Acid", "subtle acid bass"),
            SlotChoice("Mono Pulse", "pulsing mono bass"),
            SlotChoice("Drone", "sub drone"),
            SlotChoice("Hollow", "hollow low-end throb"),
        ),
    ),
    PromptField(
        name="Perc",
        choices=(
            SlotChoice("Clicks", "clicky percussion"),
            SlotChoice("Hats", "shuffled hats"),
            SlotChoice("Rim Ticks", "dry rim ticks"),
            SlotChoice("Metallic", "metallic hat chatter"),
            SlotChoice("Dub Echoes", "dubby percussion echoes"),
            SlotChoice("Claps", "snappy clap accents"),
        ),
    ),
    PromptField(
        name="Texture",
        choices=(
            SlotChoice("Sequencer", "hypnotic sequencer"),
            SlotChoice("Stab", "filtered synth stab"),
            SlotChoice("Dub Chords", "muted dub chords"),
            SlotChoice("Blips", "minimal synth blips"),
            SlotChoice("Pad Smear", "detuned pad smear"),
            SlotChoice("Acid Frags", "acid line fragments"),
        ),
    ),
    PromptField(
        name="Space",
        choices=(
            SlotChoice("Punchy", "tight punchy compression"),
            SlotChoice("Hi-Fi", "clean hi-fi sheen"),
            SlotChoice("Analog", "warm analog blur"),
            SlotChoice("Reverb", "lush reverb tail"),
            SlotChoice("Warehouse", "dry warehouse room"),
            SlotChoice("Studio", "polished studio mix"),
        ),
    ),
    PromptField(
        name="Mood",
        choices=(
            SlotChoice("Hypnotic", "hypnotic, minor key tension"),
            SlotChoice("Brooding", "brooding, open fifth drones"),
            SlotChoice("Chromatic", "mysterious, chromatic movement"),
            SlotChoice("Focused", "focused, modal restraint"),
            SlotChoice("Suspended", "playful, suspended harmonies"),
            SlotChoice("Pressure", "dark, low-register pressure"),
        ),
    ),
)


def _validated_rows(selected_rows: Sequence[int]) -> List[int]:
    rows = [int(index) for index in selected_rows]
    if len(rows) != PROMPT_FIELD_COUNT:
        raise ValueError(
            f"Expected {PROMPT_FIELD_COUNT} selected rows, got {len(rows)}."
        )
    for field_index, row_index in enumerate(rows):
        if row_index < 0 or row_index >= PROMPT_OPTION_COUNT:
            raise ValueError(
                f"Field {field_index + 1} row must be between 0 and "
                f"{PROMPT_OPTION_COUNT - 1}; got {row_index}."
            )
    return rows


def field_names() -> List[str]:
    return [field.name for field in TECHNO_PROMPT_FIELDS]


def field_choice_labels() -> List[List[str]]:
    return [[choice.label for choice in field.choices] for field in TECHNO_PROMPT_FIELDS]


def field_choice_prompts() -> List[List[str]]:
    return [[choice.prompt for choice in field.choices] for field in TECHNO_PROMPT_FIELDS]


def selected_choice_labels(selected_rows: Sequence[int]) -> List[str]:
    rows = _validated_rows(selected_rows)
    return [
        field.choices[row_index].label
        for field, row_index in zip(TECHNO_PROMPT_FIELDS, rows)
    ]


def selected_choice_prompts(selected_rows: Sequence[int]) -> List[str]:
    rows = _validated_rows(selected_rows)
    return [
        field.choices[row_index].prompt
        for field, row_index in zip(TECHNO_PROMPT_FIELDS, rows)
    ]


def compose_techno_prompt(
    selected_rows: Sequence[int],
    *,
    anchor: str = DEFAULT_TECHNO_PROMPT_ANCHOR,
) -> str:
    normalized_anchor = " ".join((anchor or "").strip().split())
    if not normalized_anchor:
        raise ValueError("Prompt anchor cannot be empty.")
    parts = [normalized_anchor, *selected_choice_prompts(selected_rows)]
    return ", ".join(part for part in parts if part)


def format_techno_schema(
    *,
    anchor: str = DEFAULT_TECHNO_PROMPT_ANCHOR,
) -> str:
    lines = [
        f"schema: {DEFAULT_TECHNO_SCHEMA_NAME}",
        f"anchor: {anchor}",
    ]
    for field_index, field in enumerate(TECHNO_PROMPT_FIELDS, start=1):
        lines.append(f"{field_index}. {field.name}")
        for row_index, choice in enumerate(field.choices, start=1):
            lines.append(
                f"   R{row_index}: {choice.label} :: {choice.prompt}"
            )
    return "\n".join(lines)

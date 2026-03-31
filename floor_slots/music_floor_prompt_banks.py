"""Prompt banks for Music Floor slot selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROMPT_ROW_COUNT = 6
DEFAULT_PROMPT_BANK = "floor_journey"


@dataclass(frozen=True)
class PromptRow:
    label: str
    prompt: str


_PROMPT_BANKS: Dict[str, Tuple[PromptRow, ...]] = {
    "floor_journey": (
        PromptRow(
            label="Ambient Lift",
            prompt="ambient synthwave, glittering arpeggios, driving pulse, nostalgic, warm analog blur, 80s inspired, major key glow",
        ),
        PromptRow(
            label="Deep Pulse",
            prompt="deep house groove, soulful electric piano, steady four on the floor, warm, polished studio mix, modern streaming era, jazzy extensions",
        ),
        PromptRow(
            label="Minimal Motion",
            prompt="minimal techno pulse, hypnotic sequencer, motorik momentum, mysterious, clean hi-fi sheen, modern streaming era, minor key tension",
        ),
        PromptRow(
            label="Acid Drive",
            prompt="acid house, squelchy acid bass, rolling syncopation, playful, tight punchy compression, retro futurist, chromatic movement",
        ),
        PromptRow(
            label="Dub Horizon",
            prompt="dub techno haze, liquid synth pads, slow-burning groove, brooding, lush reverb tail, timeless soundtrack feel, open fifth drones",
        ),
        PromptRow(
            label="Garage Spark",
            prompt="future garage swing, glassy bell textures, half-time pocket, hopeful, modern wide stereo image, early 2000s inspired, modal harmony",
        ),
    ),
    "afterhours": (
        PromptRow(
            label="Noir Trip",
            prompt="trip hop noir, dusty jazz chords, slow-burning groove, melancholic, low-fi cassette texture, 90s inspired, rich seventh chords",
        ),
        PromptRow(
            label="DnB Pressure",
            prompt="drum and bass roller, heavy drums, fast tempo energy, icy, tight punchy compression, modern streaming era, minor key tension",
        ),
        PromptRow(
            label="Jungle Smoke",
            prompt="jungle breakbeats, broken drum machine, swinging breakbeat feel, brooding, raw live room feel, 90s inspired, chromatic movement",
        ),
        PromptRow(
            label="Electronica Veil",
            prompt="cinematic electronica, wide stereo strings, floating rubato mood, expansive, modern wide stereo image, timeless soundtrack feel, modal harmony",
        ),
        PromptRow(
            label="Soul Echo",
            prompt="neo soul pocket, soulful electric piano, mid-tempo head nod, intimate, warm analog blur, early 2000s inspired, jazzy extensions",
        ),
        PromptRow(
            label="Night House",
            prompt="deep house groove, glassy bell textures, steady four on the floor, mysterious, subtle sidechain pump, modern streaming era, rich seventh chords",
        ),
    ),
    "daylight": (
        PromptRow(
            label="Bliss Field",
            prompt="blissful ambient synth, liquid synth pads, slow-burning groove, hopeful, clean hi-fi sheen, retro futurist, major key glow",
        ),
        PromptRow(
            label="Dream Pop",
            prompt="dream pop shimmer, gentle fingerpicked guitar, mid-tempo head nod, dreamy, lush reverb tail, 90s inspired, major key glow",
        ),
        PromptRow(
            label="Chill Drift",
            prompt="chillwave drift, washed cassette hiss, floating rubato mood, nostalgic, low-fi cassette texture, 80s inspired, open fifth drones",
        ),
        PromptRow(
            label="Disco Sun",
            prompt="disco nu groove, rubbery bassline, steady four on the floor, uplifting, polished studio mix, modern streaming era, rich seventh chords",
        ),
        PromptRow(
            label="Pop Bloom",
            prompt="synthpop anthem, playful lead synth, driving pulse, expansive, clean hi-fi sheen, 80s inspired, major key glow",
        ),
        PromptRow(
            label="Fusion Light",
            prompt="jazz fusion workout, percussive mallets, rolling syncopation, playful, modern wide stereo image, timeless soundtrack feel, modal harmony",
        ),
    ),
    "minimal_techno_close": (
        PromptRow(
            label="Rubber Step",
            prompt="minimal techno, rubbery bassline, steady four on the floor, hypnotic, tight punchy compression, clean hi-fi sheen, minor key tension",
        ),
        PromptRow(
            label="Tick Roll",
            prompt="minimal techno, clicky percussion, rolling syncopation, focused, tight punchy compression, clean hi-fi sheen, minor key tension",
        ),
        PromptRow(
            label="Dub Slide",
            prompt="minimal techno, muted dub chords, slow-burning groove, brooding, lush reverb tail, warm analog blur, open fifth drones",
        ),
        PromptRow(
            label="Acid Loop",
            prompt="minimal techno, subtle acid bass, driving pulse, playful, clean hi-fi sheen, retro futurist, chromatic movement",
        ),
        PromptRow(
            label="Stab Glide",
            prompt="minimal techno, filtered synth stab, low-slung groove, mysterious, polished studio mix, modern streaming era, minor key tension",
        ),
        PromptRow(
            label="Hat Swing",
            prompt="minimal techno, shuffled hats, rolling sub bass, groovy, tight punchy compression, modern wide stereo image, minor key tension",
        ),
    ),
}


def available_prompt_banks() -> List[str]:
    return sorted(_PROMPT_BANKS)


def get_prompt_bank(bank_name: str) -> List[PromptRow]:
    normalized = (bank_name or "").strip() or DEFAULT_PROMPT_BANK
    if normalized not in _PROMPT_BANKS:
        raise SystemExit(
            f"Unknown --prompt-bank {normalized!r}. Available banks: {', '.join(available_prompt_banks())}"
        )
    return list(_PROMPT_BANKS[normalized])


def format_prompt_bank_listing() -> str:
    lines: List[str] = []
    for bank_name in available_prompt_banks():
        lines.append(bank_name)
        for index, row in enumerate(_PROMPT_BANKS[bank_name], start=1):
            lines.append(f"  {index}. {row.label}: {row.prompt}")
    return "\n".join(lines)


def _auto_label(prompt: str, index: int) -> str:
    cleaned = " ".join((prompt or "").strip().split())
    if not cleaned:
        return f"Custom {index + 1}"
    words = cleaned.split(" ")
    label = " ".join(words[:3])
    if len(label) > 22:
        label = label[:22].rstrip()
    return label or f"Custom {index + 1}"


def _rows_from_json(payload: object) -> List[PromptRow]:
    if not isinstance(payload, list):
        raise SystemExit("Prompt JSON must be a list of 6 strings or objects.")
    rows: List[PromptRow] = []
    for index, item in enumerate(payload):
        if isinstance(item, str):
            prompt = item.strip()
            label = _auto_label(prompt, index)
        elif isinstance(item, dict):
            prompt = str(item.get("prompt", "")).strip()
            label = str(item.get("label", "")).strip() or _auto_label(prompt, index)
        else:
            raise SystemExit("Prompt JSON entries must be strings or {label, prompt} objects.")
        if not prompt:
            raise SystemExit("Prompt entries cannot be empty.")
        rows.append(PromptRow(label=label, prompt=prompt))
    return rows


def _rows_from_text(text: str) -> List[PromptRow]:
    rows: List[PromptRow] = []
    for index, raw_line in enumerate(text.splitlines()):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "::" in line:
            label_text, prompt_text = line.split("::", 1)
            label = label_text.strip() or _auto_label(prompt_text, len(rows))
            prompt = prompt_text.strip()
        else:
            prompt = line
            label = _auto_label(prompt, len(rows))
        if not prompt:
            raise SystemExit(f"Prompt line {index + 1} is empty.")
        rows.append(PromptRow(label=label, prompt=prompt))
    return rows


def load_prompt_rows_from_file(path_text: str) -> List[PromptRow]:
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise SystemExit(f"Prompt file not found: {path}")
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid prompt JSON file {path}: {exc}") from exc
        rows = _rows_from_json(payload)
    else:
        rows = _rows_from_text(path.read_text(encoding="utf-8"))
    if len(rows) != PROMPT_ROW_COUNT:
        raise SystemExit(
            f"Prompt file {path} must define exactly {PROMPT_ROW_COUNT} rows; got {len(rows)}."
        )
    return rows


def resolve_prompt_rows(
    *,
    prompt_bank: str,
    prompt_file: str | None,
    row_prompts: Sequence[str],
    legacy_music_prompt: str | None = None,
) -> Tuple[str, List[PromptRow]]:
    rows = get_prompt_bank(prompt_bank)
    resolved_name = (prompt_bank or "").strip() or DEFAULT_PROMPT_BANK

    if prompt_file:
        rows = load_prompt_rows_from_file(prompt_file)
        resolved_name = f"file:{Path(prompt_file).expanduser().name}"

    overrides = [prompt.strip() for prompt in row_prompts if prompt.strip()]
    if legacy_music_prompt is not None and legacy_music_prompt.strip():
        overrides = [legacy_music_prompt.strip(), *overrides]
    if len(overrides) > PROMPT_ROW_COUNT:
        raise SystemExit(
            f"Received {len(overrides)} prompt overrides, but only {PROMPT_ROW_COUNT} rows exist."
        )
    for index, prompt in enumerate(overrides):
        rows[index] = PromptRow(label=_auto_label(prompt, index), prompt=prompt)
    if overrides and not prompt_file:
        resolved_name = f"{resolved_name}+custom"
    return resolved_name, rows


def prompt_rows_summary(rows: Iterable[PromptRow]) -> str:
    return " | ".join(f"{index}. {row.label}" for index, row in enumerate(rows, start=1))

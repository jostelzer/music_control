# AGENTS.md

## Scope

This repo drives Music Floor prompt selection and Magenta Realtime playback. Keep instructions here repo-specific and practical.

## Repo Map

- `music_floor_player.py`: headless Magenta Realtime player. Loads a prompt bank, tokenizes the active style prompt, and starts the audio session.
- `floor_slots/music_floor_demo.py`: floor/front projection demo with pose-driven column selection and Magenta server control.
- `floor_slots/music_floor_prompt_banks.py`: built-in six-row prompt banks plus prompt-file loading and validation.
- `.agents/`: local preview and artifact area. If something is rendered remotely and needs to be shown locally, copy it back here first.

## Dependency Layout

These scripts expect sibling repos, with `~/git/...` fallbacks:

- `magenta-realtime`
- `projector-mapping`
- `3d_pose_estimation_optitrack/src`
- `lunar_tools` or `lunar_tools_refact`

Do not change these lookup assumptions casually. They are part of the current workstation setup.

## Prompt Rules For Magenta Realtime

Magenta Realtime prompts work best as short, style-first descriptions rather than narrative paragraphs.

- Start with a clear anchor: genre, texture, or core vibe.
- Add one or two concrete musical modifiers.
- Good modifier types: groove, drum feel, bass character, harmonic color, production finish.
- Keep prompt language musical and instrumental.
- Avoid lyric writing, story beats, character descriptions, or long scenic prose.
- For realtime steering, prefer small edits or blends between related prompts instead of abrupt full-prompt replacements.
- When editing a bank, make rows distinct enough to matter but compatible enough to transition well.

Good modifier phrases in this repo style include:

- `heavy drums`
- `rubbery bassline`
- `slow-burning groove`
- `lush reverb tail`
- `driving pulse`
- `modern wide stereo image`
- `tight punchy compression`
- `warm analog blur`
- `jazzy extensions`
- `major key glow`
- `minor key tension`

## Prompt Bank Rules

- Banks and prompt files must define exactly 6 rows.
- Keep `PROMPT_ROW_COUNT` and the floor-grid row assumptions aligned if this ever changes.
- Text prompt files may use `Label :: prompt`.
- JSON prompt files may contain strings or objects shaped like `{ "label": "...", "prompt": "..." }`.
- Row labels should stay short and readable in the floor UI.
- Prefer prompt banks that feel curated, not random. Each row should have a clear musical identity.

## Examples

- `blissful ambient synth, dreamy, slow-burning groove, lush reverb tail`
- `ambient synthwave, heavy drums, driving pulse, modern wide stereo image`
- `funk jam, rubbery bassline, mid-tempo head nod, tight punchy compression`
- `dream pop shimmer, glittering arpeggios, expansive, warm analog blur`
- `jazz fusion workout, muted trumpet phrases, rolling syncopation, polished studio mix`

## Common One-Liners

- `python music_floor_player.py --help`
- `python music_floor_player.py --list-prompt-banks`
- `python music_floor_player.py --prompt-bank floor_journey`
- `python music_floor_player.py --prompt-file prompts.txt`
- `python floor_slots/music_floor_demo.py --help`
- `python floor_slots/music_floor_demo.py --preview-only --preview-dir .agents/preview_local`
- `python floor_slots/music_floor_techno_slots_demo.py --list-techno-schema`
- `python floor_slots/music_floor_techno_slots_demo.py --preview-only --preview-dir .agents/preview_techno_fields`

## Editing Guidance

- Prefer improving prompt banks and prompt-file workflows over adding ad hoc randomization.
- If you add prompts, keep them concise and musically legible.
- If you generate preview artifacts for review, keep them under `.agents/` locally.

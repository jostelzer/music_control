"""Prompt-space helpers for MusicCoCa surrogate experiments."""

from __future__ import annotations

import numpy as np


DEFAULT_PROMPT_SPACE: tuple[tuple[str, ...], ...] = (
    (
        'acid house',
        'ambient synthwave',
        'blissful ambient synth',
        'chillwave drift',
        'cinematic electronica',
        'deep house groove',
        'disco nu groove',
        'dream pop shimmer',
        'drum and bass roller',
        'dub techno haze',
        'future garage swing',
        'jazz fusion workout',
        'jungle breakbeats',
        'minimal techno',
        'minimal techno pulse',
        'neo soul pocket',
        'synthpop anthem',
        'trip hop noir',
    ),
    (
        'broken drum machine',
        'clicky percussion',
        'dusty jazz chords',
        'filtered synth stab',
        'gentle fingerpicked guitar',
        'glassy bell textures',
        'glittering arpeggios',
        'heavy drums',
        'hypnotic sequencer',
        'liquid synth pads',
        'muted dub chords',
        'percussive mallets',
        'playful lead synth',
        'rubbery bassline',
        'shuffled hats',
        'soulful electric piano',
        'squelchy acid bass',
        'subtle acid bass',
        'washed cassette hiss',
        'wide stereo strings',
    ),
    (
        'driving pulse',
        'fast tempo energy',
        'floating rubato mood',
        'half-time pocket',
        'low-slung groove',
        'mid-tempo head nod',
        'motorik momentum',
        'rolling sub bass',
        'rolling syncopation',
        'slow-burning groove',
        'steady four on the floor',
        'swinging breakbeat feel',
    ),
    (
        'brooding',
        'dreamy',
        'expansive',
        'focused',
        'groovy',
        'hopeful',
        'hypnotic',
        'icy',
        'intimate',
        'melancholic',
        'mysterious',
        'nostalgic',
        'playful',
        'uplifting',
        'warm',
    ),
    (
        'clean hi-fi sheen',
        'low-fi cassette texture',
        'lush reverb tail',
        'modern wide stereo image',
        'polished studio mix',
        'raw live room feel',
        'subtle sidechain pump',
        'tight punchy compression',
        'warm analog blur',
    ),
    (
        '80s inspired',
        '90s inspired',
        'clean hi-fi sheen',
        'early 2000s inspired',
        'modern streaming era',
        'modern wide stereo image',
        'retro futurist',
        'timeless soundtrack feel',
        'warm analog blur',
    ),
    (
        'chromatic movement',
        'jazzy extensions',
        'major key glow',
        'minor key tension',
        'modal harmony',
        'open fifth drones',
        'rich seventh chords',
    ),
)

UNKNOWN_TOKEN = '<unk>'


def prompt_to_feature_dict(prompt: str) -> dict[str, float]:
  parts = [part.strip() for part in prompt.split(',') if part.strip()]
  features: dict[str, float] = {}
  for index, phrase in enumerate(parts):
    features[f'pos={index}|{phrase}'] = 1.0
    features[f'phrase={phrase}'] = 1.0
  features[f'length={len(parts)}'] = 1.0
  return features


def prompt_to_ids(prompt: str) -> np.ndarray:
  parts = [part.strip() for part in prompt.split(',') if part.strip()]
  ids = np.zeros((len(DEFAULT_PROMPT_SPACE),), dtype=np.int32)
  for index in range(len(DEFAULT_PROMPT_SPACE)):
    phrase = parts[index] if index < len(parts) else UNKNOWN_TOKEN
    try:
      ids[index] = DEFAULT_PROMPT_SPACE[index].index(phrase) + 1
    except ValueError:
      ids[index] = 0
  return ids


def sample_prompt(
    rng: np.random.Generator,
    *,
    allow_suffix_drop: bool = False,
) -> str:
  parts = [choices[rng.integers(0, len(choices))] for choices in DEFAULT_PROMPT_SPACE]
  if allow_suffix_drop and rng.random() < 0.35:
    keep_count = int(rng.integers(5, len(parts) + 1))
    parts = parts[:keep_count]
  return ', '.join(parts)


def generate_unique_prompts(
    count: int,
    seed: int,
    *,
    allow_suffix_drop: bool = False,
) -> list[str]:
  rng = np.random.default_rng(seed)
  prompts: list[str] = []
  seen: set[str] = set()
  while len(prompts) < count:
    prompt = sample_prompt(rng, allow_suffix_drop=allow_suffix_drop)
    if prompt in seen:
      continue
    seen.add(prompt)
    prompts.append(prompt)
  return prompts

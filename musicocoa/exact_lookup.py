"""Exact lookup backend for structured prompt grammar."""

from __future__ import annotations

import dataclasses
import json
from math import prod
from pathlib import Path
from typing import Any

import numpy as np

from musicocoa.prompt_space import DEFAULT_PROMPT_SPACE


TOKENS_PER_STYLE = 6
TABLE_DTYPE = np.uint16


@dataclasses.dataclass(frozen=True)
class ExactPromptGrammar:
  phrases_by_slot: tuple[tuple[str, ...], ...] = DEFAULT_PROMPT_SPACE

  def __post_init__(self) -> None:
    object.__setattr__(
        self,
        '_cards',
        tuple(len(phrases) for phrases in self.phrases_by_slot),
    )
    radix = []
    stride = 1
    for card in reversed(self._cards):
      radix.append(stride)
      stride *= card
    object.__setattr__(self, '_strides', tuple(reversed(radix)))
    object.__setattr__(
        self,
        '_phrase_to_id',
        tuple(
            {phrase: index + 1 for index, phrase in enumerate(phrases)}
            for phrases in self.phrases_by_slot
        ),
    )

  @property
  def cards(self) -> tuple[int, ...]:
    return self._cards

  @property
  def strides(self) -> tuple[int, ...]:
    return self._strides

  @property
  def slot_count(self) -> int:
    return len(self.phrases_by_slot)

  @property
  def combination_count(self) -> int:
    return prod(self.cards)

  @property
  def table_shape(self) -> tuple[int, int]:
    return (self.combination_count, TOKENS_PER_STYLE)

  @property
  def table_nbytes(self) -> int:
    return self.combination_count * TOKENS_PER_STYLE * np.dtype(TABLE_DTYPE).itemsize

  def prompt_to_ids(self, prompt: str) -> np.ndarray | None:
    parts = [part.strip() for part in prompt.split(',') if part.strip()]
    if len(parts) != self.slot_count:
      return None
    ids = np.zeros((self.slot_count,), dtype=np.int32)
    for slot_index, phrase in enumerate(parts):
      phrase_id = self._phrase_to_id[slot_index].get(phrase)
      if phrase_id is None:
        return None
      ids[slot_index] = phrase_id
    return ids

  def ids_to_prompt(self, ids: np.ndarray) -> str:
    parts = []
    for slot_index, phrase_id in enumerate(ids.tolist()):
      if phrase_id <= 0 or phrase_id > self.cards[slot_index]:
        raise ValueError(f'invalid phrase id {phrase_id} for slot {slot_index}')
      parts.append(self.phrases_by_slot[slot_index][phrase_id - 1])
    return ', '.join(parts)

  def ids_to_flat_index(self, ids: np.ndarray) -> int:
    values = np.asarray(ids, dtype=np.int64).reshape(-1)
    if values.shape[0] != self.slot_count:
      raise ValueError(f'expected {self.slot_count} ids, got {values.shape[0]}')
    if np.any(values <= 0):
      raise ValueError('ids must be 1-based positive integers')
    if np.any(values > np.asarray(self.cards, dtype=np.int64)):
      raise ValueError('ids exceed slot cardinality')
    zero_based = values - 1
    return int(np.dot(zero_based, np.asarray(self.strides, dtype=np.int64)))

  def batch_ids_to_flat_index(self, ids: np.ndarray) -> np.ndarray:
    values = np.asarray(ids, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != self.slot_count:
      raise ValueError(f'expected [batch,{self.slot_count}] ids, got {values.shape}')
    if np.any(values <= 0):
      raise ValueError('ids must be 1-based positive integers')
    if np.any(values > np.asarray(self.cards, dtype=np.int64)[None, :]):
      raise ValueError('ids exceed slot cardinality')
    zero_based = values - 1
    return zero_based @ np.asarray(self.strides, dtype=np.int64)

  def flat_index_to_ids(self, flat_index: int) -> np.ndarray:
    if flat_index < 0 or flat_index >= self.combination_count:
      raise ValueError(f'flat index out of range: {flat_index}')
    ids = np.zeros((self.slot_count,), dtype=np.int32)
    remainder = int(flat_index)
    for slot_index, stride in enumerate(self.strides):
      card = self.cards[slot_index]
      digit = remainder // stride
      ids[slot_index] = int(digit) + 1
      remainder = remainder % stride
      if digit >= card:
        raise ValueError(f'invalid digit {digit} at slot {slot_index}')
    return ids

  def batch_flat_index_to_ids(self, flat_indices: np.ndarray) -> np.ndarray:
    values = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
    if np.any(values < 0) or np.any(values >= self.combination_count):
      raise ValueError('flat indices out of range')
    result = np.zeros((values.shape[0], self.slot_count), dtype=np.int32)
    remainder = values.copy()
    for slot_index, stride in enumerate(self.strides):
      digits = remainder // stride
      result[:, slot_index] = digits.astype(np.int32) + 1
      remainder = remainder % stride
    return result

  def metadata(self) -> dict[str, Any]:
    return {
        'slot_count': self.slot_count,
        'cards': list(self.cards),
        'combination_count': self.combination_count,
        'tokens_per_style': TOKENS_PER_STYLE,
        'dtype': np.dtype(TABLE_DTYPE).name,
        'table_nbytes': self.table_nbytes,
    }


class ExactLookupTable:
  """Memory-mapped exact lookup table for canonical grammar prompts."""

  def __init__(self, *, grammar: ExactPromptGrammar, table_path: str | Path, mmap_mode: str = 'r'):
    self.grammar = grammar
    self.table_path = Path(table_path).expanduser()
    self._table = np.memmap(
        self.table_path,
        dtype=TABLE_DTYPE,
        mode=mmap_mode,
        shape=self.grammar.table_shape,
    )

  @classmethod
  def create_empty(cls, *, grammar: ExactPromptGrammar, table_path: str | Path) -> 'ExactLookupTable':
    path = Path(table_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = np.memmap(
        path,
        dtype=TABLE_DTYPE,
        mode='w+',
        shape=grammar.table_shape,
    )
    table.flush()
    del table
    return cls(grammar=grammar, table_path=path, mmap_mode='r+')

  def flush(self) -> None:
    self._table.flush()

  def write_rows(self, flat_indices: np.ndarray, tokens: np.ndarray) -> None:
    row_ids = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
    token_values = np.asarray(tokens, dtype=TABLE_DTYPE)
    if token_values.shape != (row_ids.shape[0], TOKENS_PER_STYLE):
      raise ValueError(
          f'expected token shape {(row_ids.shape[0], TOKENS_PER_STYLE)}, got {token_values.shape}'
      )
    self._table[row_ids] = token_values

  def lookup_ids(self, ids: np.ndarray) -> np.ndarray:
    flat_index = self.grammar.ids_to_flat_index(ids)
    return np.asarray(self._table[flat_index], dtype=np.uint16)

  def lookup_ids_batch(self, ids: np.ndarray) -> np.ndarray:
    flat_indices = self.grammar.batch_ids_to_flat_index(ids)
    return np.asarray(self._table[flat_indices], dtype=np.uint16)

  def lookup_prompts(self, prompts: list[str]) -> tuple[np.ndarray, list[int]]:
    prompt_ids = []
    valid_indices = []
    for prompt_index, prompt in enumerate(prompts):
      ids = self.grammar.prompt_to_ids(prompt)
      if ids is None:
        continue
      prompt_ids.append(ids)
      valid_indices.append(prompt_index)
    if not prompt_ids:
      return np.zeros((0, TOKENS_PER_STYLE), dtype=np.uint16), []
    tokens = self.lookup_ids_batch(np.stack(prompt_ids, axis=0))
    return tokens, valid_indices


def write_metadata(path: str | Path, grammar: ExactPromptGrammar) -> None:
  target = Path(path).expanduser()
  target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(json.dumps(grammar.metadata(), indent=2, sort_keys=True), encoding='utf-8')

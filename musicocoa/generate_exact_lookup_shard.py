#!/usr/bin/env python3

"""Generates exact lookup table shards from the legacy teacher service."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
  from musicocoa.exact_lookup import ExactPromptGrammar
  from musicocoa.exact_lookup import TOKENS_PER_STYLE
  from musicocoa.exact_lookup import write_metadata
else:
  from .exact_lookup import ExactPromptGrammar
  from .exact_lookup import TOKENS_PER_STYLE
  from .exact_lookup import write_metadata


def teacher_embed_batch(teacher_url: str, prompts: list[str]) -> np.ndarray:
  body = json.dumps({'texts': prompts}).encode('utf-8')
  request = urllib.request.Request(
      f'{teacher_url.rstrip("/")}/embed_batch',
      data=body,
      headers={'Content-Type': 'application/json'},
  )
  try:
    with urllib.request.urlopen(request, timeout=600) as response:
      payload = json.loads(response.read().decode('utf-8'))
  except urllib.error.HTTPError as exc:
    detail = exc.read().decode('utf-8', errors='replace')
    raise RuntimeError(f'teacher request failed: {exc.code} {detail}') from exc
  return np.asarray([item['runtime_tokens'] for item in payload['items']], dtype=np.uint16)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--teacher-url', default='http://127.0.0.1:8770')
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--start-index', type=int, required=True)
  parser.add_argument('--end-index', type=int, required=True)
  parser.add_argument('--batch-size', type=int, default=256)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  grammar = ExactPromptGrammar()
  if args.start_index < 0 or args.end_index > grammar.combination_count:
    raise SystemExit('start/end indices out of range')
  if args.start_index >= args.end_index:
    raise SystemExit('start index must be < end index')

  output_dir = Path(args.output_dir).expanduser()
  output_dir.mkdir(parents=True, exist_ok=True)
  table_path = output_dir / f'exact_tokens_{args.start_index}_{args.end_index}.bin'
  meta_path = output_dir / f'exact_tokens_{args.start_index}_{args.end_index}.json'
  row_count = args.end_index - args.start_index

  shard = np.memmap(
      table_path,
      dtype=np.uint16,
      mode='w+',
      shape=(row_count, TOKENS_PER_STYLE),
  )

  for offset in range(0, row_count, args.batch_size):
    count = min(args.batch_size, row_count - offset)
    flat_indices = np.arange(args.start_index + offset, args.start_index + offset + count, dtype=np.int64)
    ids_batch = grammar.batch_flat_index_to_ids(flat_indices)
    prompts = [grammar.ids_to_prompt(ids) for ids in ids_batch]
    tokens = teacher_embed_batch(args.teacher_url, prompts)
    if tokens.shape != (count, TOKENS_PER_STYLE):
      raise RuntimeError(f'unexpected token shape {tokens.shape}')
    shard[offset : offset + count] = tokens
  shard.flush()
  del shard

  meta = {
      'start_index': args.start_index,
      'end_index': args.end_index,
      'row_count': row_count,
      'dtype': 'uint16',
      'tokens_per_style': TOKENS_PER_STYLE,
      'cards': list(grammar.cards),
  }
  meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')

  table_meta_path = output_dir / 'exact_lookup_metadata.json'
  if not table_meta_path.exists():
    write_metadata(table_meta_path, grammar)
  print(json.dumps({'table_path': str(table_path), 'meta_path': str(meta_path), **meta}, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

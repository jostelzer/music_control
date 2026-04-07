#!/usr/bin/env python3

"""Benchmarks exact grammar lookup throughput."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
  from musicocoa.exact_lookup import ExactLookupTable
  from musicocoa.exact_lookup import ExactPromptGrammar
  from musicocoa.prompt_space import generate_unique_prompts
else:
  from .exact_lookup import ExactLookupTable
  from .exact_lookup import ExactPromptGrammar
  from .prompt_space import generate_unique_prompts


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--prompt-count', type=int, default=250)
  parser.add_argument('--repeats', type=int, default=50)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--use-memmap', action='store_true')
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  grammar = ExactPromptGrammar()
  prompts = generate_unique_prompts(args.prompt_count, args.seed)
  prompt_ids = np.stack([grammar.prompt_to_ids(prompt) for prompt in prompts], axis=0)

  if args.use_memmap:
    with tempfile.TemporaryDirectory(prefix='musicocoa_exact_lookup_') as tmpdir:
      table_path = Path(tmpdir) / 'table.bin'
      lookup = ExactLookupTable.create_empty(grammar=grammar, table_path=table_path)
      lookup._table[:] = 0
      lookup.flush()
      timings_ms = []
      for _ in range(args.repeats):
        started = time.perf_counter()
        tokens, valid = lookup.lookup_prompts(prompts)
        elapsed = (time.perf_counter() - started) * 1000.0
        assert tokens.shape == (args.prompt_count, 6)
        assert len(valid) == args.prompt_count
        timings_ms.append(elapsed)
  else:
    started = time.perf_counter()
    flat_indices = grammar.batch_ids_to_flat_index(prompt_ids)
    index_elapsed = (time.perf_counter() - started) * 1000.0
    timings_ms = [index_elapsed]

  print(
      json.dumps(
          {
              'prompt_count': args.prompt_count,
              'repeats': args.repeats,
              'use_memmap': args.use_memmap,
              'mean_ms': float(np.mean(timings_ms)),
              'min_ms': float(np.min(timings_ms)),
              'max_ms': float(np.max(timings_ms)),
          },
          indent=2,
          sort_keys=True,
      )
  )
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

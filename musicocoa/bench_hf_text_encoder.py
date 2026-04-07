#!/usr/bin/env python3

"""Benchmarks pretrained Hugging Face text encoders for batched prompt throughput."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from transformers import AutoTokenizer

from musicocoa.prompt_space import generate_unique_prompts

try:
  import tensorflow as tf
except ImportError:  # pragma: no cover - optional backend
  tf = None

try:
  import torch
except ImportError:  # pragma: no cover - optional backend
  torch = None

try:
  from transformers import TFAutoModel
except ImportError:  # pragma: no cover - optional backend
  TFAutoModel = None

try:
  from transformers import AutoModel
except ImportError:  # pragma: no cover - optional backend
  AutoModel = None


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-name', required=True)
  parser.add_argument('--prompt-count', type=int, default=1000)
  parser.add_argument('--max-length', type=int, default=32)
  parser.add_argument('--repeats', type=int, default=10)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--allow-suffix-drop', action='store_true')
  parser.add_argument('--backend', choices=['auto', 'tf', 'torch'], default='auto')
  return parser.parse_args()


def benchmark_tf(args: argparse.Namespace, prompts: list[str]) -> dict[str, object]:
  if tf is None or TFAutoModel is None:
    raise RuntimeError('TensorFlow backend unavailable for this transformers build.')
  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  model = TFAutoModel.from_pretrained(args.model_name)
  devices = [device.name for device in tf.config.list_physical_devices()]

  encodings = tokenizer(
      prompts,
      padding=True,
      truncation=True,
      max_length=args.max_length,
      return_tensors='tf',
  )
  for _ in range(2):
    outputs = model(encodings, training=False).last_hidden_state
    pooled = tf.reduce_mean(outputs, axis=1)
    _ = pooled.numpy()

  timings_ms: list[float] = []
  for _ in range(args.repeats):
    started = time.perf_counter()
    encodings = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors='tf',
    )
    outputs = model(encodings, training=False).last_hidden_state
    pooled = tf.reduce_mean(outputs, axis=1)
    _ = pooled.numpy()
    timings_ms.append((time.perf_counter() - started) * 1000.0)
  return {
      'backend': 'tf',
      'devices': devices,
      'embedding_shape': list(pooled.shape),
      'mean_ms': float(np.mean(timings_ms)),
      'min_ms': float(np.min(timings_ms)),
      'max_ms': float(np.max(timings_ms)),
  }


def benchmark_torch(args: argparse.Namespace, prompts: list[str]) -> dict[str, object]:
  if torch is None or AutoModel is None:
    raise RuntimeError('PyTorch backend unavailable for this environment.')
  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  model = AutoModel.from_pretrained(args.model_name)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = model.to(device)
  model.eval()

  with torch.inference_mode():
    encodings = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors='pt',
    )
    encodings = {key: value.to(device) for key, value in encodings.items()}
    for _ in range(2):
      outputs = model(**encodings).last_hidden_state
      pooled = outputs.mean(dim=1)
      _ = pooled.detach().cpu().numpy()

    timings_ms: list[float] = []
    for _ in range(args.repeats):
      started = time.perf_counter()
      encodings = tokenizer(
          prompts,
          padding=True,
          truncation=True,
          max_length=args.max_length,
          return_tensors='pt',
      )
      encodings = {key: value.to(device) for key, value in encodings.items()}
      outputs = model(**encodings).last_hidden_state
      pooled = outputs.mean(dim=1)
      _ = pooled.detach().cpu().numpy()
      timings_ms.append((time.perf_counter() - started) * 1000.0)
  return {
      'backend': 'torch',
      'devices': [device],
      'embedding_shape': list(pooled.shape),
      'mean_ms': float(np.mean(timings_ms)),
      'min_ms': float(np.min(timings_ms)),
      'max_ms': float(np.max(timings_ms)),
  }


def main() -> int:
  args = parse_args()
  prompts = generate_unique_prompts(
      args.prompt_count,
      args.seed,
      allow_suffix_drop=args.allow_suffix_drop,
  )
  if args.backend == 'tf':
    result = benchmark_tf(args, prompts)
  elif args.backend == 'torch':
    result = benchmark_torch(args, prompts)
  else:
    try:
      result = benchmark_tf(args, prompts)
    except RuntimeError:
      result = benchmark_torch(args, prompts)
  result.update(
      {
          'model_name': args.model_name,
          'prompt_count': args.prompt_count,
      }
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

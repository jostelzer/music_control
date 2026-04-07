#!/usr/bin/env python3

"""Benchmarks local additive prompt backends against the legacy teacher."""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence

import numpy as np

if __package__ in (None, ''):
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
  from musicocoa.backend import BackendConfig
  from musicocoa.backend import MusicCoCaBackend
  from musicocoa.backend import rvq_dequantize
  from musicocoa.backend import rvq_quantization
  from musicocoa.prompt_space import DEFAULT_PROMPT_SPACE
  from musicocoa.prompt_space import generate_unique_prompts
else:
  from .backend import BackendConfig
  from .backend import MusicCoCaBackend
  from .backend import rvq_dequantize
  from .backend import rvq_quantization
  from .prompt_space import DEFAULT_PROMPT_SPACE
  from .prompt_space import generate_unique_prompts


@dataclasses.dataclass(frozen=True)
class TeacherSample:
  prompt: str
  embedding: np.ndarray
  runtime_tokens: np.ndarray


@dataclasses.dataclass(frozen=True)
class Report:
  model_name: str
  prompt_exact: float
  slot_accuracy: float
  mean_hamming: float
  mean_probe_count: float
  benchmark_ms_250: float


def teacher_embed_batch(
    teacher_url: str,
    prompts: Sequence[str],
    batch_size: int,
) -> list[TeacherSample]:
  records: list[TeacherSample] = []
  for start in range(0, len(prompts), batch_size):
    batch = list(prompts[start : start + batch_size])
    body = json.dumps(
        {
            'texts': batch,
            'include_embedding': True,
        }
    ).encode('utf-8')
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
    items = payload['items']
    for item in items:
      records.append(
          TeacherSample(
              prompt=str(item['text']),
              embedding=np.asarray(item['embedding'], dtype=np.float32),
              runtime_tokens=np.asarray(item['runtime_tokens'], dtype=np.int32),
          )
      )
  return records


def token_metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
  exact = float(np.mean(np.all(predicted == target, axis=1)))
  slot_accuracy = float(np.mean(predicted == target))
  mean_hamming = float(np.mean(np.sum(predicted != target, axis=1)))
  return exact, slot_accuracy, mean_hamming


def parse_prompt(prompt: str) -> list[str]:
  return [part.strip() for part in prompt.split(',') if part.strip()]


def set_slot(parts: Sequence[str], slot_index: int, value: str) -> str:
  updated = list(parts)
  updated[slot_index] = value
  return ', '.join(updated)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
  norms = np.linalg.norm(vectors, axis=1, keepdims=True)
  return vectors / np.maximum(norms, 1e-8)


def build_local_bank(base_prompt: str) -> tuple[list[str], dict[tuple[int, str], str]]:
  base_parts = parse_prompt(base_prompt)
  probe_prompts = [base_prompt]
  single_prompt_by_key: dict[tuple[int, str], str] = {}
  for slot_index, choices in enumerate(DEFAULT_PROMPT_SPACE):
    current = base_parts[slot_index]
    for choice in choices:
      if choice == current:
        continue
      prompt = set_slot(base_parts, slot_index, choice)
      probe_prompts.append(prompt)
      single_prompt_by_key[(slot_index, choice)] = prompt
  return probe_prompts, single_prompt_by_key


def sample_eval_prompts(
    base_prompt: str,
    *,
    count: int,
    seed: int,
    max_changed_slots: int,
) -> list[str]:
  rng = np.random.default_rng(seed)
  base_parts = parse_prompt(base_prompt)
  prompts: set[str] = set()
  while len(prompts) < count:
    slot_count = int(rng.integers(1, max_changed_slots + 1))
    slot_indices = rng.choice(len(DEFAULT_PROMPT_SPACE), size=slot_count, replace=False)
    updated = list(base_parts)
    changed = False
    for slot_index in slot_indices.tolist():
      choices = DEFAULT_PROMPT_SPACE[slot_index]
      current = updated[slot_index]
      alternatives = [choice for choice in choices if choice != current]
      replacement = alternatives[int(rng.integers(0, len(alternatives)))]
      updated[slot_index] = replacement
      changed = changed or replacement != current
    if changed:
      prompts.add(', '.join(updated))
  return list(prompts)


def predict_additive_embeddings(
    *,
    base_prompt: str,
    base_embedding: np.ndarray,
    single_embeddings: dict[tuple[int, str], np.ndarray],
    prompts: Sequence[str],
    shrink: float,
) -> tuple[np.ndarray, np.ndarray]:
  base_parts = parse_prompt(base_prompt)
  predicted = np.repeat(base_embedding.reshape(1, -1), len(prompts), axis=0)
  probe_counts = np.zeros((len(prompts),), dtype=np.int32)
  for row_index, prompt in enumerate(prompts):
    parts = parse_prompt(prompt)
    for slot_index, value in enumerate(parts):
      if value == base_parts[slot_index]:
        continue
      predicted[row_index] += shrink * (single_embeddings[(slot_index, value)] - base_embedding)
      probe_counts[row_index] += 1
  return normalize_rows(predicted.astype(np.float32)), probe_counts


def benchmark_predict(
    *,
    base_prompt: str,
    base_embedding: np.ndarray,
    single_embeddings: dict[tuple[int, str], np.ndarray],
    prompts: Sequence[str],
    codebooks: np.ndarray,
    shrink: float,
    repeats: int = 10,
) -> float:
  timings = []
  for _ in range(repeats):
    started = time.perf_counter()
    predicted_embeddings, _ = predict_additive_embeddings(
        base_prompt=base_prompt,
        base_embedding=base_embedding,
        single_embeddings=single_embeddings,
        prompts=prompts,
        shrink=shrink,
    )
    _ = rvq_quantization(predicted_embeddings, codebooks)[0][:, :6]
    timings.append((time.perf_counter() - started) * 1000.0)
  return float(np.mean(timings))


def run(args: argparse.Namespace) -> dict[str, object]:
  base_prompts = generate_unique_prompts(args.base_count, args.seed)
  backend = MusicCoCaBackend(
      BackendConfig(
          backend_name=args.backend_name,
          device=args.device,
          runtime_style_token_depth=args.runtime_style_token_depth,
      )
  )
  codebooks = backend._rvq_codebooks[: args.runtime_style_token_depth].copy()

  reports: list[Report] = []
  for shrink in args.shrink:
    all_predicted: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    all_probe_counts: list[np.ndarray] = []
    benchmark_ms_values: list[float] = []
    teacher_ms = 0.0

    for base_index, base_prompt in enumerate(base_prompts):
      probe_prompts, single_prompt_by_key = build_local_bank(base_prompt)
      eval_prompts = sample_eval_prompts(
          base_prompt,
          count=args.eval_count_per_base,
          seed=args.seed + 1000 + base_index,
          max_changed_slots=args.max_changed_slots,
      )
      query_prompts = list(dict.fromkeys(probe_prompts + eval_prompts))
      started = time.perf_counter()
      samples = teacher_embed_batch(args.teacher_url, query_prompts, args.teacher_batch_size)
      teacher_ms += (time.perf_counter() - started) * 1000.0
      sample_by_prompt = {sample.prompt: sample for sample in samples}

      base_sample = sample_by_prompt[base_prompt]
      single_embeddings = {
          key: sample_by_prompt[prompt].embedding for key, prompt in single_prompt_by_key.items()
      }
      predicted_embeddings, probe_counts = predict_additive_embeddings(
          base_prompt=base_prompt,
          base_embedding=base_sample.embedding,
          single_embeddings=single_embeddings,
          prompts=eval_prompts,
          shrink=shrink,
      )
      predicted_tokens = rvq_quantization(predicted_embeddings, codebooks)[0][:, : args.runtime_style_token_depth]
      target_tokens = np.asarray(
          [sample_by_prompt[prompt].runtime_tokens for prompt in eval_prompts],
          dtype=np.int32,
      )
      all_predicted.append(predicted_tokens)
      all_target.append(target_tokens)
      all_probe_counts.append(probe_counts)

      benchmark_prompts = eval_prompts[: min(args.benchmark_count, len(eval_prompts))]
      benchmark_ms_values.append(
          benchmark_predict(
              base_prompt=base_prompt,
              base_embedding=base_sample.embedding,
              single_embeddings=single_embeddings,
              prompts=benchmark_prompts,
              codebooks=codebooks,
              shrink=shrink,
          )
      )

    predicted = np.concatenate(all_predicted, axis=0)
    target = np.concatenate(all_target, axis=0)
    probe_counts = np.concatenate(all_probe_counts, axis=0)
    exact, slot_accuracy, mean_hamming = token_metrics(predicted, target)
    reports.append(
        Report(
            model_name=f'local_additive_shrink_{shrink:g}',
            prompt_exact=exact,
            slot_accuracy=slot_accuracy,
            mean_hamming=mean_hamming,
            mean_probe_count=float(np.mean(probe_counts)),
            benchmark_ms_250=float(np.mean(benchmark_ms_values)),
        )
    )

  reports_sorted = sorted(
      reports,
      key=lambda report: (
          report.prompt_exact,
          report.slot_accuracy,
          -report.mean_hamming,
          -report.benchmark_ms_250,
      ),
      reverse=True,
  )
  return {
      'teacher_url': args.teacher_url,
      'base_count': args.base_count,
      'eval_count_per_base': args.eval_count_per_base,
      'max_changed_slots': args.max_changed_slots,
      'reports': [dataclasses.asdict(report) for report in reports_sorted],
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--teacher-url', default='http://127.0.0.1:8770')
  parser.add_argument('--backend-name', default='cpu_compat')
  parser.add_argument('--device', default='/CPU:0')
  parser.add_argument('--runtime-style-token-depth', type=int, default=6)
  parser.add_argument('--teacher-batch-size', type=int, default=128)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--base-count', type=int, default=4)
  parser.add_argument('--eval-count-per-base', type=int, default=250)
  parser.add_argument('--benchmark-count', type=int, default=250)
  parser.add_argument('--max-changed-slots', type=int, default=3)
  parser.add_argument('--shrink', nargs='*', type=float, default=[0.5, 0.75, 1.0, 1.25])
  return parser.parse_args()


def main() -> int:
  result = run(parse_args())
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

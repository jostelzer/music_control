#!/usr/bin/env python3

"""Searches for fast prompt-to-style surrogates against the legacy teacher."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge

from musicocoa.backend import BackendConfig
from musicocoa.backend import MusicCoCaBackend
from musicocoa.backend import rvq_dequantize
from musicocoa.backend import rvq_quantization
from musicocoa.prompt_space import generate_unique_prompts
from musicocoa.prompt_space import prompt_to_feature_dict


@dataclasses.dataclass(frozen=True)
class TeacherSample:
  prompt: str
  runtime_tokens: np.ndarray


@dataclasses.dataclass(frozen=True)
class ModelReport:
  model_name: str
  prompt_exact: float
  slot_accuracy: float
  mean_hamming: float
  vector_cosine: float
  benchmark_ms_1000: float


def teacher_embed_batch(
    teacher_url: str,
    prompts: Sequence[str],
    batch_size: int,
) -> list[TeacherSample]:
  records: list[TeacherSample] = []
  for start in range(0, len(prompts), batch_size):
    batch = list(prompts[start : start + batch_size])
    body = json.dumps({'texts': batch}).encode('utf-8')
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
              runtime_tokens=np.asarray(item['runtime_tokens'], dtype=np.int32),
          )
      )
  return records


def dequantize_runtime_tokens(
    samples: Sequence[TeacherSample],
    codebooks: np.ndarray,
) -> np.ndarray:
  tokens = np.asarray([sample.runtime_tokens for sample in samples], dtype=np.int32)
  return rvq_dequantize(tokens, codebooks)


def cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
  numerator = np.sum(a * b, axis=1)
  denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
  denom = np.maximum(denom, 1e-8)
  return float(np.mean(numerator / denom))


def token_metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
  exact = float(np.mean(np.all(predicted == target, axis=1)))
  slot_accuracy = float(np.mean(predicted == target))
  mean_hamming = float(np.mean(np.sum(predicted != target, axis=1)))
  return exact, slot_accuracy, mean_hamming


def benchmark_model(
    predict_fn,
    benchmark_prompts: Sequence[str],
    repeats: int = 5,
) -> float:
  timings_ms: list[float] = []
  for _ in range(repeats):
    started = time.perf_counter()
    predict_fn(list(benchmark_prompts))
    timings_ms.append((time.perf_counter() - started) * 1000.0)
  return float(np.mean(timings_ms))


def build_additive_report(
    train_samples: Sequence[TeacherSample],
    eval_samples: Sequence[TeacherSample],
    codebooks: np.ndarray,
    alpha: float,
    benchmark_prompts: Sequence[str],
) -> ModelReport:
  vectorizer = DictVectorizer(sparse=False)
  x_train = vectorizer.fit_transform(
      [prompt_to_feature_dict(sample.prompt) for sample in train_samples]
  )
  y_train = dequantize_runtime_tokens(train_samples, codebooks)
  regressor = Ridge(alpha=alpha, fit_intercept=True)
  regressor.fit(x_train, y_train)

  x_eval = vectorizer.transform(
      [prompt_to_feature_dict(sample.prompt) for sample in eval_samples]
  )
  y_eval = dequantize_runtime_tokens(eval_samples, codebooks)
  predicted_vectors = regressor.predict(x_eval).astype(np.float32)
  predicted_tokens = rvq_quantization(predicted_vectors, codebooks)[0]
  target_tokens = np.asarray(
      [sample.runtime_tokens for sample in eval_samples], dtype=np.int32
  )
  exact, slot_accuracy, mean_hamming = token_metrics(predicted_tokens, target_tokens)

  def predict_prompts(prompts: list[str]) -> np.ndarray:
    x = vectorizer.transform([prompt_to_feature_dict(prompt) for prompt in prompts])
    predicted = regressor.predict(x).astype(np.float32)
    return rvq_quantization(predicted, codebooks)[0]

  return ModelReport(
      model_name=f'additive_ridge_alpha_{alpha:g}',
      prompt_exact=exact,
      slot_accuracy=slot_accuracy,
      mean_hamming=mean_hamming,
      vector_cosine=cosine_mean(predicted_vectors, y_eval),
      benchmark_ms_1000=benchmark_model(predict_prompts, benchmark_prompts),
  )


def build_hashing_report(
    train_samples: Sequence[TeacherSample],
    eval_samples: Sequence[TeacherSample],
    codebooks: np.ndarray,
    alpha: float,
    benchmark_prompts: Sequence[str],
) -> ModelReport:
  vectorizer = HashingVectorizer(
      analyzer='char_wb',
      ngram_range=(3, 5),
      n_features=2**14,
      alternate_sign=False,
      norm=None,
      lowercase=True,
  )
  x_train = vectorizer.transform([sample.prompt for sample in train_samples])
  y_train = dequantize_runtime_tokens(train_samples, codebooks)
  regressor = Ridge(alpha=alpha, fit_intercept=True, solver='lsqr')
  regressor.fit(x_train, y_train)

  x_eval = vectorizer.transform([sample.prompt for sample in eval_samples])
  y_eval = dequantize_runtime_tokens(eval_samples, codebooks)
  predicted_vectors = regressor.predict(x_eval).astype(np.float32)
  predicted_tokens = rvq_quantization(predicted_vectors, codebooks)[0]
  target_tokens = np.asarray(
      [sample.runtime_tokens for sample in eval_samples], dtype=np.int32
  )
  exact, slot_accuracy, mean_hamming = token_metrics(predicted_tokens, target_tokens)

  def predict_prompts(prompts: list[str]) -> np.ndarray:
    x = vectorizer.transform(prompts)
    predicted = regressor.predict(x).astype(np.float32)
    return rvq_quantization(predicted, codebooks)[0]

  return ModelReport(
      model_name=f'hashing_ridge_alpha_{alpha:g}',
      prompt_exact=exact,
      slot_accuracy=slot_accuracy,
      mean_hamming=mean_hamming,
      vector_cosine=cosine_mean(predicted_vectors, y_eval),
      benchmark_ms_1000=benchmark_model(predict_prompts, benchmark_prompts),
  )


def run(args: argparse.Namespace) -> dict[str, object]:
  total_count = args.train_count + args.eval_count + args.benchmark_count
  prompts = generate_unique_prompts(
      total_count,
      args.seed,
      allow_suffix_drop=args.allow_suffix_drop,
  )
  teacher_start = time.perf_counter()
  samples = teacher_embed_batch(args.teacher_url, prompts, args.teacher_batch_size)
  teacher_ms = (time.perf_counter() - teacher_start) * 1000.0

  backend = MusicCoCaBackend(
      BackendConfig(
          backend_name=args.backend_name,
          device=args.device,
          runtime_style_token_depth=args.runtime_style_token_depth,
      )
  )
  codebooks = backend._rvq_codebooks[: args.runtime_style_token_depth].copy()

  train_samples = samples[: args.train_count]
  eval_samples = samples[args.train_count : args.train_count + args.eval_count]
  benchmark_prompts = prompts[-args.benchmark_count :]

  reports: list[ModelReport] = []
  for alpha in args.additive_alpha:
    reports.append(
        build_additive_report(
            train_samples,
            eval_samples,
            codebooks,
            alpha,
            benchmark_prompts,
        )
    )
  for alpha in args.hashing_alpha:
    reports.append(
        build_hashing_report(
            train_samples,
            eval_samples,
            codebooks,
            alpha,
            benchmark_prompts,
        )
    )

  reports_sorted = sorted(
      reports,
      key=lambda report: (
          report.prompt_exact,
          report.slot_accuracy,
          -report.mean_hamming,
          report.vector_cosine,
          -report.benchmark_ms_1000,
      ),
      reverse=True,
  )
  return {
      'teacher_url': args.teacher_url,
      'teacher_label_ms': teacher_ms,
      'train_count': args.train_count,
      'eval_count': args.eval_count,
      'benchmark_count': args.benchmark_count,
      'reports': [dataclasses.asdict(report) for report in reports_sorted],
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--teacher-url', default='http://127.0.0.1:8770')
  parser.add_argument('--train-count', type=int, default=4000)
  parser.add_argument('--eval-count', type=int, default=1000)
  parser.add_argument('--benchmark-count', type=int, default=1000)
  parser.add_argument('--teacher-batch-size', type=int, default=128)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--backend-name', default='cpu_compat')
  parser.add_argument('--device', default='/CPU:0')
  parser.add_argument('--runtime-style-token-depth', type=int, default=6)
  parser.add_argument('--additive-alpha', nargs='*', type=float, default=[0.1, 1.0, 10.0])
  parser.add_argument('--hashing-alpha', nargs='*', type=float, default=[0.1, 1.0])
  parser.add_argument('--allow-suffix-drop', action='store_true')
  return parser.parse_args()


def main() -> int:
  result = run(parse_args())
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

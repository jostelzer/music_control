#!/usr/bin/env python3

"""Searches tree-based categorical backends against the legacy teacher."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from musicocoa.prompt_space import generate_unique_prompts
from musicocoa.prompt_space import prompt_to_ids


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


def token_metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
  exact = float(np.mean(np.all(predicted == target, axis=1)))
  slot_accuracy = float(np.mean(predicted == target))
  mean_hamming = float(np.mean(np.sum(predicted != target, axis=1)))
  return exact, slot_accuracy, mean_hamming


def benchmark_model(
    predict_fn,
    prompt_ids: np.ndarray,
    repeats: int = 10,
) -> float:
  timings = []
  for _ in range(repeats):
    started = time.perf_counter()
    _ = predict_fn(prompt_ids)
    timings.append((time.perf_counter() - started) * 1000.0)
  return float(np.mean(timings))


def fit_depthwise_classifier(
    model_factory,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, list[object], list[np.ndarray]]:
  predicted = np.zeros((x_eval.shape[0], y_train.shape[1]), dtype=np.int32)
  models = []
  class_values = []
  for depth in range(y_train.shape[1]):
    values, inverse = np.unique(y_train[:, depth], return_inverse=True)
    model = model_factory()
    model.fit(x_train, inverse)
    pred_index = model.predict(x_eval)
    predicted[:, depth] = values[pred_index]
    models.append(model)
    class_values.append(values)
  return predicted, models, class_values


def build_report(
    *,
    model_name: str,
    model_factory,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    benchmark_ids: np.ndarray,
) -> ModelReport:
  predicted, models, class_values = fit_depthwise_classifier(
      model_factory, x_train, y_train, x_eval
  )
  exact, slot_accuracy, mean_hamming = token_metrics(predicted, y_eval)

  def predict_fn(query_ids: np.ndarray) -> np.ndarray:
    result = np.zeros((query_ids.shape[0], y_train.shape[1]), dtype=np.int32)
    for depth, model in enumerate(models):
      pred_index = model.predict(query_ids)
      result[:, depth] = class_values[depth][pred_index]
    return result

  return ModelReport(
      model_name=model_name,
      prompt_exact=exact,
      slot_accuracy=slot_accuracy,
      mean_hamming=mean_hamming,
      benchmark_ms_1000=benchmark_model(predict_fn, benchmark_ids),
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

  train_samples = samples[: args.train_count]
  eval_samples = samples[args.train_count : args.train_count + args.eval_count]
  benchmark_prompts = prompts[-args.benchmark_count :]

  x_train = np.stack([prompt_to_ids(sample.prompt) for sample in train_samples], axis=0)
  y_train = np.asarray([sample.runtime_tokens for sample in train_samples], dtype=np.int32)
  x_eval = np.stack([prompt_to_ids(sample.prompt) for sample in eval_samples], axis=0)
  y_eval = np.asarray([sample.runtime_tokens for sample in eval_samples], dtype=np.int32)
  benchmark_ids = np.stack([prompt_to_ids(prompt) for prompt in benchmark_prompts], axis=0)

  reports: list[ModelReport] = []
  for max_iter in args.hgb_max_iter:
    for depth in args.hgb_max_depth:
      reports.append(
          build_report(
              model_name=f'hgb_iter{max_iter}_depth{depth}',
              model_factory=lambda max_iter=max_iter, depth=depth: HistGradientBoostingClassifier(
                  max_iter=max_iter,
                  max_depth=depth,
                  learning_rate=0.1,
                  categorical_features=np.ones((x_train.shape[1],), dtype=bool),
                  random_state=7,
              ),
              x_train=x_train,
              y_train=y_train,
              x_eval=x_eval,
              y_eval=y_eval,
              benchmark_ids=benchmark_ids,
          )
      )
  for n_estimators in args.extra_trees_estimators:
    reports.append(
        build_report(
            model_name=f'extra_trees_{n_estimators}',
            model_factory=lambda n_estimators=n_estimators: ExtraTreesClassifier(
                n_estimators=n_estimators,
                random_state=7,
                n_jobs=-1,
            ),
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            y_eval=y_eval,
            benchmark_ids=benchmark_ids,
        )
    )
  for n_estimators in args.random_forest_estimators:
    reports.append(
        build_report(
            model_name=f'random_forest_{n_estimators}',
            model_factory=lambda n_estimators=n_estimators: RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=7,
                n_jobs=-1,
            ),
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            y_eval=y_eval,
            benchmark_ids=benchmark_ids,
        )
    )

  reports_sorted = sorted(
      reports,
      key=lambda report: (
          report.prompt_exact,
          report.slot_accuracy,
          -report.mean_hamming,
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
  parser.add_argument('--train-count', type=int, default=10000)
  parser.add_argument('--eval-count', type=int, default=1500)
  parser.add_argument('--benchmark-count', type=int, default=1000)
  parser.add_argument('--teacher-batch-size', type=int, default=128)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--hgb-max-iter', nargs='*', type=int, default=[100, 200])
  parser.add_argument('--hgb-max-depth', nargs='*', type=int, default=[6, 10])
  parser.add_argument('--extra-trees-estimators', nargs='*', type=int, default=[400])
  parser.add_argument('--random-forest-estimators', nargs='*', type=int, default=[])
  parser.add_argument('--allow-suffix-drop', action='store_true')
  return parser.parse_args()


def main() -> int:
  result = run(parse_args())
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

#!/usr/bin/env python3

"""Searches sparse lexical retrieval backends against the legacy teacher."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

if __package__ in (None, ''):
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
  from musicocoa.prompt_space import generate_unique_prompts
else:
  from .prompt_space import generate_unique_prompts


PHRASE_SPLIT_RE = re.compile(r'\s*,\s*')


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
  benchmark_ms_250: float


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


def phrase_analyzer(text: str) -> list[str]:
  return [part.strip().lower() for part in PHRASE_SPLIT_RE.split(text) if part.strip()]


def build_vectorizer(kind: str) -> TfidfVectorizer:
  if kind == 'phrase':
    return TfidfVectorizer(
        analyzer=phrase_analyzer,
        lowercase=False,
        norm='l2',
        dtype=np.float32,
    )
  if kind == 'word':
    return TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        lowercase=True,
        norm='l2',
        dtype=np.float32,
    )
  if kind == 'char':
    return TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        lowercase=True,
        norm='l2',
        dtype=np.float32,
    )
  raise ValueError(f'unknown vectorizer kind: {kind}')


def sparse_topk(
    train_matrix,
    query_matrix,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
  scores = (query_matrix @ train_matrix.T).toarray().astype(np.float32)
  topk_idx = np.argpartition(scores, kth=scores.shape[1] - k, axis=1)[:, -k:]
  topk_scores = np.take_along_axis(scores, topk_idx, axis=1)
  order = np.argsort(topk_scores, axis=1)[:, ::-1]
  topk_idx = np.take_along_axis(topk_idx, order, axis=1)
  topk_scores = np.take_along_axis(topk_scores, order, axis=1)
  return topk_idx, topk_scores


def predict_sequence_vote(
    train_tokens: np.ndarray,
    topk_idx: np.ndarray,
    topk_scores: np.ndarray,
) -> np.ndarray:
  batch = topk_idx.shape[0]
  predicted = np.zeros((batch, train_tokens.shape[1]), dtype=np.int32)
  weights = np.exp(topk_scores - np.max(topk_scores, axis=1, keepdims=True))
  for query_index in range(batch):
    score_by_seq: dict[tuple[int, ...], float] = defaultdict(float)
    for neighbor_rank in range(topk_idx.shape[1]):
      seq = tuple(int(value) for value in train_tokens[topk_idx[query_index, neighbor_rank]].tolist())
      score_by_seq[seq] += float(weights[query_index, neighbor_rank])
    predicted[query_index] = np.asarray(max(score_by_seq.items(), key=lambda item: item[1])[0], dtype=np.int32)
  return predicted


def predict_slot_vote(
    train_tokens: np.ndarray,
    topk_idx: np.ndarray,
    topk_scores: np.ndarray,
) -> np.ndarray:
  batch, depth = topk_idx.shape[0], train_tokens.shape[1]
  predicted = np.zeros((batch, depth), dtype=np.int32)
  weights = np.exp(topk_scores - np.max(topk_scores, axis=1, keepdims=True))
  for query_index in range(batch):
    neighbors = train_tokens[topk_idx[query_index]]
    neighbor_weights = weights[query_index]
    for slot_index in range(depth):
      score_by_token: dict[int, float] = defaultdict(float)
      for neighbor_rank, token_value in enumerate(neighbors[:, slot_index].tolist()):
        score_by_token[int(token_value)] += float(neighbor_weights[neighbor_rank])
      predicted[query_index, slot_index] = max(score_by_token.items(), key=lambda item: item[1])[0]
  return predicted


def benchmark_backend(
    *,
    vectorizer: TfidfVectorizer,
    train_matrix,
    train_tokens: np.ndarray,
    benchmark_prompts: Sequence[str],
    predict_kind: str,
    k: int,
    repeats: int = 10,
) -> float:
  timings = []
  for _ in range(repeats):
    started = time.perf_counter()
    query_matrix = vectorizer.transform(list(benchmark_prompts))
    idx, scores = sparse_topk(train_matrix, query_matrix, k)
    if predict_kind == 'sequence_vote':
      _ = predict_sequence_vote(train_tokens, idx, scores)
    else:
      _ = predict_slot_vote(train_tokens, idx, scores)
    timings.append((time.perf_counter() - started) * 1000.0)
  return float(np.mean(timings))


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

  prompts_train = [sample.prompt for sample in train_samples]
  prompts_eval = [sample.prompt for sample in eval_samples]
  train_tokens = np.asarray([sample.runtime_tokens for sample in train_samples], dtype=np.int32)
  eval_tokens = np.asarray([sample.runtime_tokens for sample in eval_samples], dtype=np.int32)

  reports: list[ModelReport] = []
  for vectorizer_kind in args.vectorizer_kind:
    vectorizer = build_vectorizer(vectorizer_kind)
    build_start = time.perf_counter()
    train_matrix = vectorizer.fit_transform(prompts_train)
    eval_matrix = vectorizer.transform(prompts_eval)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    for k in args.knn_k:
      topk_idx, topk_scores = sparse_topk(train_matrix, eval_matrix, k)

      predicted_sequence = predict_sequence_vote(train_tokens, topk_idx, topk_scores)
      exact, slot_accuracy, mean_hamming = token_metrics(predicted_sequence, eval_tokens)
      reports.append(
          ModelReport(
              model_name=f'{vectorizer_kind}_sequence_vote_k{k}',
              prompt_exact=exact,
              slot_accuracy=slot_accuracy,
              mean_hamming=mean_hamming,
              benchmark_ms_250=benchmark_backend(
                  vectorizer=vectorizer,
                  train_matrix=train_matrix,
                  train_tokens=train_tokens,
                  benchmark_prompts=benchmark_prompts,
                  predict_kind='sequence_vote',
                  k=k,
              ),
          )
      )

      predicted_slot = predict_slot_vote(train_tokens, topk_idx, topk_scores)
      exact, slot_accuracy, mean_hamming = token_metrics(predicted_slot, eval_tokens)
      reports.append(
          ModelReport(
              model_name=f'{vectorizer_kind}_slot_vote_k{k}',
              prompt_exact=exact,
              slot_accuracy=slot_accuracy,
              mean_hamming=mean_hamming,
              benchmark_ms_250=benchmark_backend(
                  vectorizer=vectorizer,
                  train_matrix=train_matrix,
                  train_tokens=train_tokens,
                  benchmark_prompts=benchmark_prompts,
                  predict_kind='slot_vote',
                  k=k,
              ),
          )
      )

    print(
        json.dumps(
            {
                'vectorizer_kind': vectorizer_kind,
                'vectorizer_build_ms': build_ms,
                'feature_count': int(train_matrix.shape[1]),
            }
        ),
        file=sys.stderr,
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
  parser.add_argument('--benchmark-count', type=int, default=250)
  parser.add_argument('--teacher-batch-size', type=int, default=128)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--knn-k', nargs='*', type=int, default=[1, 2, 4, 8, 16, 32])
  parser.add_argument('--vectorizer-kind', nargs='*', default=['phrase', 'word', 'char'])
  parser.add_argument('--allow-suffix-drop', action='store_true')
  return parser.parse_args()


def main() -> int:
  result = run(parse_args())
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

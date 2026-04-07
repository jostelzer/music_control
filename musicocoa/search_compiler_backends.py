#!/usr/bin/env python3

"""Searches structured prompt-compiler backends against the legacy teacher."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import urllib.error
import urllib.request
from itertools import combinations
from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

from musicocoa.backend import BackendConfig
from musicocoa.backend import MusicCoCaBackend
from musicocoa.backend import rvq_dequantize
from musicocoa.backend import rvq_quantization
from musicocoa.prompt_space import DEFAULT_PROMPT_SPACE
from musicocoa.prompt_space import generate_unique_prompts
from musicocoa.prompt_space import prompt_to_feature_dict
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
  vector_cosine: float | None
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


def prompt_to_pairwise_feature_dict(prompt: str) -> dict[str, float]:
  parts = [part.strip() for part in prompt.split(',') if part.strip()]
  features = prompt_to_feature_dict(prompt)
  for left in range(len(parts)):
    for right in range(left + 1, len(parts)):
      features[f'pair={left},{right}|{parts[left]}|{parts[right]}'] = 1.0
  return features


def prompt_to_triple_feature_dict(prompt: str) -> dict[str, float]:
  parts = [part.strip() for part in prompt.split(',') if part.strip()]
  features = prompt_to_pairwise_feature_dict(prompt)
  for left in range(len(parts)):
    for mid in range(left + 1, len(parts)):
      for right in range(mid + 1, len(parts)):
        features[
            f'triple={left},{mid},{right}|{parts[left]}|{parts[mid]}|{parts[right]}'
        ] = 1.0
  return features


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
    prompts: Sequence[str],
    repeats: int = 5,
) -> float:
  timings_ms = []
  for _ in range(repeats):
    started = time.perf_counter()
    _ = predict_fn(list(prompts))
    timings_ms.append((time.perf_counter() - started) * 1000.0)
  return float(np.mean(timings_ms))


def build_pairwise_ridge_report(
    train_samples: Sequence[TeacherSample],
    eval_samples: Sequence[TeacherSample],
    codebooks: np.ndarray,
    alpha: float,
    benchmark_prompts: Sequence[str],
) -> ModelReport:
  vectorizer = DictVectorizer(sparse=False)
  x_train = vectorizer.fit_transform(
      [prompt_to_pairwise_feature_dict(sample.prompt) for sample in train_samples]
  )
  y_train = rvq_dequantize(
      np.asarray([sample.runtime_tokens for sample in train_samples], dtype=np.int32),
      codebooks,
  )
  regressor = Ridge(alpha=alpha, fit_intercept=True)
  regressor.fit(x_train, y_train)

  x_eval = vectorizer.transform(
      [prompt_to_pairwise_feature_dict(sample.prompt) for sample in eval_samples]
  )
  y_eval = rvq_dequantize(
      np.asarray([sample.runtime_tokens for sample in eval_samples], dtype=np.int32),
      codebooks,
  )
  predicted_vectors = regressor.predict(x_eval).astype(np.float32)
  predicted_tokens = rvq_quantization(predicted_vectors, codebooks)[0]
  target_tokens = np.asarray(
      [sample.runtime_tokens for sample in eval_samples], dtype=np.int32
  )
  exact, slot_accuracy, mean_hamming = token_metrics(predicted_tokens, target_tokens)

  def predict_prompts(prompts: list[str]) -> np.ndarray:
    x = vectorizer.transform([prompt_to_pairwise_feature_dict(prompt) for prompt in prompts])
    predicted = regressor.predict(x).astype(np.float32)
    return rvq_quantization(predicted, codebooks)[0]

  return ModelReport(
      model_name=f'pairwise_ridge_alpha_{alpha:g}',
      prompt_exact=exact,
      slot_accuracy=slot_accuracy,
      mean_hamming=mean_hamming,
      vector_cosine=cosine_mean(predicted_vectors, y_eval),
      benchmark_ms_1000=benchmark_model(predict_prompts, benchmark_prompts),
  )


def build_compiled_pairwise_ridge_report(
    train_samples: Sequence[TeacherSample],
    eval_samples: Sequence[TeacherSample],
    codebooks: np.ndarray,
    alpha: float,
    benchmark_prompts: Sequence[str],
) -> ModelReport:
  vectorizer = DictVectorizer(sparse=False)
  x_train = vectorizer.fit_transform(
      [prompt_to_pairwise_feature_dict(sample.prompt) for sample in train_samples]
  )
  y_train = rvq_dequantize(
      np.asarray([sample.runtime_tokens for sample in train_samples], dtype=np.int32),
      codebooks,
  )
  regressor = Ridge(alpha=alpha, fit_intercept=True)
  regressor.fit(x_train, y_train)

  coeffs = regressor.coef_.astype(np.float32)  # [dim, features]
  dim = coeffs.shape[0]
  bias = regressor.intercept_.astype(np.float32)
  feature_names = vectorizer.get_feature_names_out().tolist()
  phrase_to_id = [
      {phrase: index + 1 for index, phrase in enumerate(phrases)}
      for phrases in DEFAULT_PROMPT_SPACE
  ]
  unary_tables = [
      np.zeros((len(phrases) + 1, dim), dtype=np.float32)
      for phrases in DEFAULT_PROMPT_SPACE
  ]
  length_tables = np.zeros((len(DEFAULT_PROMPT_SPACE) + 1, dim), dtype=np.float32)
  pair_tables = {
      (left, right): np.zeros(
          (
              len(DEFAULT_PROMPT_SPACE[left]) + 1,
              len(DEFAULT_PROMPT_SPACE[right]) + 1,
              dim,
          ),
          dtype=np.float32,
      )
      for left in range(len(DEFAULT_PROMPT_SPACE))
      for right in range(left + 1, len(DEFAULT_PROMPT_SPACE))
  }
  phrase_contributions: dict[str, np.ndarray] = {}
  for feature_index, feature_name in enumerate(feature_names):
    contribution = coeffs[:, feature_index]
    if feature_name.startswith('pos='):
      header, phrase = feature_name.split('|', 1)
      position = int(header.split('=')[1])
      phrase_id = phrase_to_id[position].get(phrase)
      if phrase_id is not None:
        unary_tables[position][phrase_id] += contribution
    elif feature_name.startswith('phrase='):
      phrase = feature_name.split('=', 1)[1]
      phrase_contributions[phrase] = contribution
    elif feature_name.startswith('length='):
      length_tables[int(feature_name.split('=', 1)[1])] = contribution
    elif feature_name.startswith('pair='):
      header, left_phrase, right_phrase = feature_name.split('|', 2)
      pair_text = header.split('=')[1]
      left_position, right_position = [int(value) for value in pair_text.split(',')]
      left_id = phrase_to_id[left_position].get(left_phrase)
      right_id = phrase_to_id[right_position].get(right_phrase)
      if left_id is not None and right_id is not None:
        pair_tables[(left_position, right_position)][left_id, right_id] += contribution

  # Fold global phrase contributions into the per-position unary tables.
  for position, phrases in enumerate(DEFAULT_PROMPT_SPACE):
    for phrase, phrase_id in phrase_to_id[position].items():
      contribution = phrase_contributions.get(phrase)
      if contribution is not None:
        unary_tables[position][phrase_id] += contribution

  def predict_vectors_from_ids(query_ids: np.ndarray) -> np.ndarray:
    batch_size = query_ids.shape[0]
    predicted = np.repeat(bias[None, :], batch_size, axis=0)
    lengths = np.count_nonzero(query_ids, axis=1)
    predicted += length_tables[lengths]
    for position in range(query_ids.shape[1]):
      predicted += unary_tables[position][query_ids[:, position]]
    for (left, right), table in pair_tables.items():
      predicted += table[query_ids[:, left], query_ids[:, right]]
    return predicted

  def predict_prompts(prompts: list[str]) -> np.ndarray:
    query_ids = np.stack([prompt_to_ids(prompt) for prompt in prompts], axis=0)
    predicted = predict_vectors_from_ids(query_ids)
    return rvq_quantization(predicted.astype(np.float32), codebooks)[0]

  target_tokens = np.asarray(
      [sample.runtime_tokens for sample in eval_samples], dtype=np.int32
  )
  eval_ids = np.stack([prompt_to_ids(sample.prompt) for sample in eval_samples], axis=0)
  predicted_vectors = predict_vectors_from_ids(eval_ids)
  predicted_tokens = rvq_quantization(predicted_vectors.astype(np.float32), codebooks)[0]
  exact, slot_accuracy, mean_hamming = token_metrics(predicted_tokens, target_tokens)
  y_eval = rvq_dequantize(target_tokens, codebooks)

  return ModelReport(
      model_name=f'compiled_pairwise_ridge_alpha_{alpha:g}',
      prompt_exact=exact,
      slot_accuracy=slot_accuracy,
      mean_hamming=mean_hamming,
      vector_cosine=cosine_mean(predicted_vectors, y_eval),
      benchmark_ms_1000=benchmark_model(predict_prompts, benchmark_prompts),
  )


def build_compiled_pairwise_library_report(
    train_samples: Sequence[TeacherSample],
    eval_samples: Sequence[TeacherSample],
    codebooks: np.ndarray,
    alpha: float,
    benchmark_prompts: Sequence[str],
) -> ModelReport:
  vectorizer = DictVectorizer(sparse=False)
  x_train = vectorizer.fit_transform(
      [prompt_to_pairwise_feature_dict(sample.prompt) for sample in train_samples]
  )
  train_tokens = np.asarray(
      [sample.runtime_tokens for sample in train_samples], dtype=np.int32
  )
  y_train = rvq_dequantize(train_tokens, codebooks)
  regressor = Ridge(alpha=alpha, fit_intercept=True)
  regressor.fit(x_train, y_train)

  coeffs = regressor.coef_.astype(np.float32)
  dim = coeffs.shape[0]
  bias = regressor.intercept_.astype(np.float32)
  feature_names = vectorizer.get_feature_names_out().tolist()
  phrase_to_id = [
      {phrase: index + 1 for index, phrase in enumerate(phrases)}
      for phrases in DEFAULT_PROMPT_SPACE
  ]
  unary_tables = [
      np.zeros((len(phrases) + 1, dim), dtype=np.float32)
      for phrases in DEFAULT_PROMPT_SPACE
  ]
  length_tables = np.zeros((len(DEFAULT_PROMPT_SPACE) + 1, dim), dtype=np.float32)
  pair_tables = {
      (left, right): np.zeros(
          (
              len(DEFAULT_PROMPT_SPACE[left]) + 1,
              len(DEFAULT_PROMPT_SPACE[right]) + 1,
              dim,
          ),
          dtype=np.float32,
      )
      for left in range(len(DEFAULT_PROMPT_SPACE))
      for right in range(left + 1, len(DEFAULT_PROMPT_SPACE))
  }
  phrase_contributions: dict[str, np.ndarray] = {}
  for feature_index, feature_name in enumerate(feature_names):
    contribution = coeffs[:, feature_index]
    if feature_name.startswith('pos='):
      header, phrase = feature_name.split('|', 1)
      position = int(header.split('=')[1])
      phrase_id = phrase_to_id[position].get(phrase)
      if phrase_id is not None:
        unary_tables[position][phrase_id] += contribution
    elif feature_name.startswith('phrase='):
      phrase = feature_name.split('=', 1)[1]
      phrase_contributions[phrase] = contribution
    elif feature_name.startswith('length='):
      length_tables[int(feature_name.split('=', 1)[1])] = contribution
    elif feature_name.startswith('pair='):
      header, left_phrase, right_phrase = feature_name.split('|', 2)
      pair_text = header.split('=')[1]
      left_position, right_position = [int(value) for value in pair_text.split(',')]
      left_id = phrase_to_id[left_position].get(left_phrase)
      right_id = phrase_to_id[right_position].get(right_phrase)
      if left_id is not None and right_id is not None:
        pair_tables[(left_position, right_position)][left_id, right_id] += contribution

  for position, _phrases in enumerate(DEFAULT_PROMPT_SPACE):
    for phrase, phrase_id in phrase_to_id[position].items():
      contribution = phrase_contributions.get(phrase)
      if contribution is not None:
        unary_tables[position][phrase_id] += contribution

  unique_map: dict[tuple[int, ...], np.ndarray] = {}
  for token_row in train_tokens:
    token_key = tuple(int(value) for value in token_row.tolist())
    if token_key not in unique_map:
      unique_map[token_key] = token_row.copy()
  library_tokens = np.asarray(list(unique_map.values()), dtype=np.int32)
  library_vectors = rvq_dequantize(library_tokens, codebooks).astype(np.float32)
  library_norms = np.linalg.norm(library_vectors, axis=1, keepdims=True)
  library_norms = np.maximum(library_norms, 1e-8)
  library_unit = library_vectors / library_norms

  def predict_vectors_from_ids(query_ids: np.ndarray) -> np.ndarray:
    batch_size = query_ids.shape[0]
    predicted = np.repeat(bias[None, :], batch_size, axis=0)
    lengths = np.count_nonzero(query_ids, axis=1)
    predicted += length_tables[lengths]
    for position in range(query_ids.shape[1]):
      predicted += unary_tables[position][query_ids[:, position]]
    for (left, right), table in pair_tables.items():
      predicted += table[query_ids[:, left], query_ids[:, right]]
    return predicted

  def snap_vectors_to_library(predicted_vectors: np.ndarray) -> np.ndarray:
    predicted_norms = np.linalg.norm(predicted_vectors, axis=1, keepdims=True)
    predicted_norms = np.maximum(predicted_norms, 1e-8)
    predicted_unit = predicted_vectors / predicted_norms
    scores = predicted_unit @ library_unit.T
    nearest = np.argmax(scores, axis=1)
    return library_tokens[nearest]

  def predict_prompts(prompts: list[str]) -> np.ndarray:
    query_ids = np.stack([prompt_to_ids(prompt) for prompt in prompts], axis=0)
    predicted_vectors = predict_vectors_from_ids(query_ids)
    return snap_vectors_to_library(predicted_vectors.astype(np.float32))

  target_tokens = np.asarray(
      [sample.runtime_tokens for sample in eval_samples], dtype=np.int32
  )
  eval_ids = np.stack([prompt_to_ids(sample.prompt) for sample in eval_samples], axis=0)
  predicted_vectors = predict_vectors_from_ids(eval_ids).astype(np.float32)
  predicted_tokens = snap_vectors_to_library(predicted_vectors)
  exact, slot_accuracy, mean_hamming = token_metrics(predicted_tokens, target_tokens)
  y_eval = rvq_dequantize(target_tokens, codebooks)

  return ModelReport(
      model_name=f'compiled_pairwise_library_alpha_{alpha:g}_lib{library_tokens.shape[0]}',
      prompt_exact=exact,
      slot_accuracy=slot_accuracy,
      mean_hamming=mean_hamming,
      vector_cosine=cosine_mean(predicted_vectors, y_eval),
      benchmark_ms_1000=benchmark_model(predict_prompts, benchmark_prompts),
  )


def build_compiled_triple_ridge_report(
    train_samples: Sequence[TeacherSample],
    eval_samples: Sequence[TeacherSample],
    codebooks: np.ndarray,
    alpha: float,
    benchmark_prompts: Sequence[str],
) -> ModelReport:
  vectorizer = DictVectorizer()
  x_train = vectorizer.fit_transform(
      [prompt_to_triple_feature_dict(sample.prompt) for sample in train_samples]
  )
  train_tokens = np.asarray(
      [sample.runtime_tokens for sample in train_samples], dtype=np.int32
  )
  y_train = rvq_dequantize(train_tokens, codebooks)
  regressor = Ridge(alpha=alpha, fit_intercept=True)
  regressor.fit(x_train, y_train)

  coeffs = regressor.coef_.astype(np.float32)
  dim = coeffs.shape[0]
  bias = regressor.intercept_.astype(np.float32)
  feature_names = vectorizer.get_feature_names_out().tolist()
  phrase_to_id = [
      {phrase: index + 1 for index, phrase in enumerate(phrases)}
      for phrases in DEFAULT_PROMPT_SPACE
  ]
  unary_tables = [
      np.zeros((len(phrases) + 1, dim), dtype=np.float32)
      for phrases in DEFAULT_PROMPT_SPACE
  ]
  length_tables = np.zeros((len(DEFAULT_PROMPT_SPACE) + 1, dim), dtype=np.float32)
  pair_tables = {
      (left, right): np.zeros(
          (
              len(DEFAULT_PROMPT_SPACE[left]) + 1,
              len(DEFAULT_PROMPT_SPACE[right]) + 1,
              dim,
          ),
          dtype=np.float32,
      )
      for left in range(len(DEFAULT_PROMPT_SPACE))
      for right in range(left + 1, len(DEFAULT_PROMPT_SPACE))
  }
  triple_tables = {
      (left, mid, right): np.zeros(
          (
              len(DEFAULT_PROMPT_SPACE[left]) + 1,
              len(DEFAULT_PROMPT_SPACE[mid]) + 1,
              len(DEFAULT_PROMPT_SPACE[right]) + 1,
              dim,
          ),
          dtype=np.float32,
      )
      for left in range(len(DEFAULT_PROMPT_SPACE))
      for mid in range(left + 1, len(DEFAULT_PROMPT_SPACE))
      for right in range(mid + 1, len(DEFAULT_PROMPT_SPACE))
  }
  phrase_contributions: dict[str, np.ndarray] = {}
  for feature_index, feature_name in enumerate(feature_names):
    contribution = coeffs[:, feature_index]
    if feature_name.startswith('pos='):
      header, phrase = feature_name.split('|', 1)
      position = int(header.split('=')[1])
      phrase_id = phrase_to_id[position].get(phrase)
      if phrase_id is not None:
        unary_tables[position][phrase_id] += contribution
    elif feature_name.startswith('phrase='):
      phrase = feature_name.split('=', 1)[1]
      phrase_contributions[phrase] = contribution
    elif feature_name.startswith('length='):
      length_tables[int(feature_name.split('=', 1)[1])] = contribution
    elif feature_name.startswith('pair='):
      header, left_phrase, right_phrase = feature_name.split('|', 2)
      pair_text = header.split('=')[1]
      left_position, right_position = [int(value) for value in pair_text.split(',')]
      left_id = phrase_to_id[left_position].get(left_phrase)
      right_id = phrase_to_id[right_position].get(right_phrase)
      if left_id is not None and right_id is not None:
        pair_tables[(left_position, right_position)][left_id, right_id] += contribution
    elif feature_name.startswith('triple='):
      header, left_phrase, mid_phrase, right_phrase = feature_name.split('|', 3)
      triple_text = header.split('=')[1]
      left_position, mid_position, right_position = [
          int(value) for value in triple_text.split(',')
      ]
      left_id = phrase_to_id[left_position].get(left_phrase)
      mid_id = phrase_to_id[mid_position].get(mid_phrase)
      right_id = phrase_to_id[right_position].get(right_phrase)
      if left_id is not None and mid_id is not None and right_id is not None:
        triple_tables[(left_position, mid_position, right_position)][
            left_id, mid_id, right_id
        ] += contribution

  for position, _phrases in enumerate(DEFAULT_PROMPT_SPACE):
    for phrase, phrase_id in phrase_to_id[position].items():
      contribution = phrase_contributions.get(phrase)
      if contribution is not None:
        unary_tables[position][phrase_id] += contribution

  def predict_vectors_from_ids(query_ids: np.ndarray) -> np.ndarray:
    batch_size = query_ids.shape[0]
    predicted = np.repeat(bias[None, :], batch_size, axis=0)
    lengths = np.count_nonzero(query_ids, axis=1)
    predicted += length_tables[lengths]
    for position in range(query_ids.shape[1]):
      predicted += unary_tables[position][query_ids[:, position]]
    for (left, right), table in pair_tables.items():
      predicted += table[query_ids[:, left], query_ids[:, right]]
    for (left, mid, right), table in triple_tables.items():
      predicted += table[
          query_ids[:, left],
          query_ids[:, mid],
          query_ids[:, right],
      ]
    return predicted

  def predict_prompts(prompts: list[str]) -> np.ndarray:
    query_ids = np.stack([prompt_to_ids(prompt) for prompt in prompts], axis=0)
    predicted = predict_vectors_from_ids(query_ids)
    return rvq_quantization(predicted.astype(np.float32), codebooks)[0]

  target_tokens = np.asarray(
      [sample.runtime_tokens for sample in eval_samples], dtype=np.int32
  )
  eval_ids = np.stack([prompt_to_ids(sample.prompt) for sample in eval_samples], axis=0)
  predicted_vectors = predict_vectors_from_ids(eval_ids).astype(np.float32)
  predicted_tokens = rvq_quantization(predicted_vectors, codebooks)[0]
  exact, slot_accuracy, mean_hamming = token_metrics(predicted_tokens, target_tokens)
  y_eval = rvq_dequantize(target_tokens, codebooks)

  return ModelReport(
      model_name=f'compiled_triple_ridge_alpha_{alpha:g}',
      prompt_exact=exact,
      slot_accuracy=slot_accuracy,
      mean_hamming=mean_hamming,
      vector_cosine=cosine_mean(predicted_vectors, y_eval),
      benchmark_ms_1000=benchmark_model(predict_prompts, benchmark_prompts),
  )


def _vote_tokens(
    neighbor_tokens: np.ndarray,
    neighbor_distances: np.ndarray,
) -> np.ndarray:
  weights = 1.0 / (1.0 + neighbor_distances.astype(np.float32))
  predicted = np.zeros((neighbor_tokens.shape[0], neighbor_tokens.shape[2]), dtype=np.int32)
  for query_index in range(neighbor_tokens.shape[0]):
    for depth in range(neighbor_tokens.shape[2]):
      counter: dict[int, float] = {}
      for neighbor_index, token in enumerate(neighbor_tokens[query_index, :, depth]):
        counter[int(token)] = counter.get(int(token), 0.0) + float(
            weights[query_index, neighbor_index]
        )
      predicted[query_index, depth] = max(counter.items(), key=lambda item: item[1])[0]
  return predicted


def build_hamming_knn_report(
    train_samples: Sequence[TeacherSample],
    eval_samples: Sequence[TeacherSample],
    k: int,
    benchmark_prompts: Sequence[str],
) -> ModelReport:
  train_ids = np.stack([prompt_to_ids(sample.prompt) for sample in train_samples], axis=0)
  eval_ids = np.stack([prompt_to_ids(sample.prompt) for sample in eval_samples], axis=0)
  train_tokens = np.asarray(
      [sample.runtime_tokens for sample in train_samples], dtype=np.int32
  )
  target_tokens = np.asarray(
      [sample.runtime_tokens for sample in eval_samples], dtype=np.int32
  )

  def predict_ids(query_ids: np.ndarray) -> np.ndarray:
    # Shape: [queries, train, 7]
    mismatches = np.not_equal(query_ids[:, None, :], train_ids[None, :, :]).sum(axis=2)
    neighbor_indices = np.argpartition(mismatches, kth=k - 1, axis=1)[:, :k]
    neighbor_distances = np.take_along_axis(mismatches, neighbor_indices, axis=1)
    neighbor_tokens = train_tokens[neighbor_indices]
    return _vote_tokens(neighbor_tokens, neighbor_distances)

  predicted_tokens = predict_ids(eval_ids)
  exact, slot_accuracy, mean_hamming = token_metrics(predicted_tokens, target_tokens)

  def predict_prompts(prompts: list[str]) -> np.ndarray:
    query_ids = np.stack([prompt_to_ids(prompt) for prompt in prompts], axis=0)
    return predict_ids(query_ids)

  return ModelReport(
      model_name=f'hamming_knn_k{k}',
      prompt_exact=exact,
      slot_accuracy=slot_accuracy,
      mean_hamming=mean_hamming,
      vector_cosine=None,
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
  for alpha in args.ridge_alpha:
    reports.append(
        build_pairwise_ridge_report(
            train_samples,
            eval_samples,
            codebooks,
            alpha,
            benchmark_prompts,
        )
    )
    reports.append(
        build_compiled_pairwise_ridge_report(
            train_samples,
            eval_samples,
            codebooks,
            alpha,
            benchmark_prompts,
        )
    )
    reports.append(
        build_compiled_pairwise_library_report(
            train_samples,
            eval_samples,
            codebooks,
            alpha,
            benchmark_prompts,
        )
    )
    if args.include_triple:
      reports.append(
          build_compiled_triple_ridge_report(
              train_samples,
              eval_samples,
              codebooks,
              alpha,
              benchmark_prompts,
          )
      )
  for k in args.knn_k:
    reports.append(
        build_hamming_knn_report(
            train_samples,
            eval_samples,
            k,
            benchmark_prompts,
        )
    )

  reports_sorted = sorted(
      reports,
      key=lambda report: (
          report.prompt_exact,
          report.slot_accuracy,
          -report.mean_hamming,
          (report.vector_cosine or 0.0),
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
  parser.add_argument('--backend-name', default='cpu_compat')
  parser.add_argument('--device', default='/CPU:0')
  parser.add_argument('--runtime-style-token-depth', type=int, default=6)
  parser.add_argument('--ridge-alpha', nargs='*', type=float, default=[0.01, 0.1, 1.0])
  parser.add_argument('--knn-k', nargs='*', type=int, default=[1, 4, 8, 16])
  parser.add_argument('--include-triple', action='store_true')
  parser.add_argument('--allow-suffix-drop', action='store_true')
  return parser.parse_args()


def main() -> int:
  result = run(parse_args())
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

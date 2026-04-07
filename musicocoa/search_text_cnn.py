#!/usr/bin/env python3

"""Searches small text-CNN surrogates against the legacy teacher."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence

import numpy as np
import tensorflow as tf

from musicocoa.backend import BackendConfig
from musicocoa.backend import MusicCoCaBackend
from musicocoa.backend import rvq_dequantize
from musicocoa.backend import rvq_quantization
from musicocoa.prompt_space import generate_unique_prompts


@dataclasses.dataclass(frozen=True)
class TeacherSample:
  prompt: str
  runtime_tokens: np.ndarray


@dataclasses.dataclass(frozen=True)
class ModelReport:
  model_name: str
  epochs_run: int
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


def standardize_prompt(text: tf.Tensor) -> tf.Tensor:
  lowered = tf.strings.lower(text)
  lowered = tf.strings.regex_replace(lowered, r'[^a-z0-9, ]+', ' ')
  lowered = tf.strings.regex_replace(lowered, r',', ' ')
  lowered = tf.strings.regex_replace(lowered, r'\\s+', ' ')
  return tf.strings.strip(lowered)


def build_model(
    train_prompts: Sequence[str],
    *,
    sequence_length: int,
    embedding_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> tuple[tf.keras.Model, tf.keras.layers.TextVectorization]:
  vectorizer = tf.keras.layers.TextVectorization(
      standardize=standardize_prompt,
      split='whitespace',
      output_mode='int',
      output_sequence_length=sequence_length,
  )
  vectorizer.adapt(tf.data.Dataset.from_tensor_slices(list(train_prompts)).batch(256))

  inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name='prompt')
  token_ids = vectorizer(inputs)
  x = tf.keras.layers.Embedding(
      input_dim=len(vectorizer.get_vocabulary()),
      output_dim=embedding_dim,
      mask_zero=False,
      name='token_embedding',
  )(token_ids)
  conv3 = tf.keras.layers.Conv1D(
      hidden_dim, 3, padding='same', activation='gelu', name='conv3'
  )(x)
  conv5 = tf.keras.layers.Conv1D(
      hidden_dim, 5, padding='same', activation='gelu', name='conv5'
  )(x)
  x = tf.keras.layers.Concatenate(name='concat_convs')([conv3, conv5])
  x = tf.keras.layers.GlobalMaxPooling1D(name='global_max')(x)
  x = tf.keras.layers.Dense(hidden_dim, activation='gelu', name='hidden')(x)
  outputs = tf.keras.layers.Dense(output_dim, name='runtime_vector')(x)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
      loss=tf.keras.losses.MeanSquaredError(),
  )
  return model, vectorizer


def benchmark_model(
    model: tf.keras.Model,
    prompts: Sequence[str],
    codebooks: np.ndarray,
    repeats: int = 10,
) -> float:
  tensor = tf.constant([[prompt] for prompt in prompts], dtype=tf.string)
  infer = tf.function(model, reduce_retracing=True)
  _ = infer(tensor, training=False)
  timings_ms: list[float] = []
  for _ in range(repeats):
    started = time.perf_counter()
    predicted = infer(tensor, training=False).numpy().astype(np.float32)
    _ = rvq_quantization(predicted, codebooks)[0]
    timings_ms.append((time.perf_counter() - started) * 1000.0)
  return float(np.mean(timings_ms))


def run_one_model(
    *,
    model_name: str,
    train_prompts: Sequence[str],
    train_vectors: np.ndarray,
    eval_prompts: Sequence[str],
    eval_vectors: np.ndarray,
    eval_tokens: np.ndarray,
    benchmark_prompts: Sequence[str],
    codebooks: np.ndarray,
    sequence_length: int,
    embedding_dim: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
) -> ModelReport:
  model, _ = build_model(
      train_prompts,
      sequence_length=sequence_length,
      embedding_dim=embedding_dim,
      hidden_dim=hidden_dim,
      output_dim=train_vectors.shape[1],
  )
  callbacks = [
      tf.keras.callbacks.EarlyStopping(
          monitor='val_loss',
          patience=3,
          restore_best_weights=True,
      )
  ]
  train_tensor = tf.constant([[prompt] for prompt in train_prompts], dtype=tf.string)
  eval_tensor = tf.constant([[prompt] for prompt in eval_prompts], dtype=tf.string)
  history = model.fit(
      train_tensor,
      train_vectors,
      validation_data=(eval_tensor, eval_vectors),
      batch_size=batch_size,
      epochs=epochs,
      verbose=0,
      callbacks=callbacks,
  )
  predicted_vectors = model(eval_tensor, training=False).numpy().astype(np.float32)
  predicted_tokens = rvq_quantization(predicted_vectors, codebooks)[0]
  exact, slot_accuracy, mean_hamming = token_metrics(predicted_tokens, eval_tokens)
  return ModelReport(
      model_name=model_name,
      epochs_run=len(history.history['loss']),
      prompt_exact=exact,
      slot_accuracy=slot_accuracy,
      mean_hamming=mean_hamming,
      vector_cosine=cosine_mean(predicted_vectors, eval_vectors),
      benchmark_ms_1000=benchmark_model(model, benchmark_prompts, codebooks),
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

  train_prompts = [sample.prompt for sample in train_samples]
  eval_prompts = [sample.prompt for sample in eval_samples]
  train_tokens = np.asarray(
      [sample.runtime_tokens for sample in train_samples], dtype=np.int32
  )
  eval_tokens = np.asarray(
      [sample.runtime_tokens for sample in eval_samples], dtype=np.int32
  )
  train_vectors = rvq_dequantize(train_tokens, codebooks)
  eval_vectors = rvq_dequantize(eval_tokens, codebooks)

  reports = []
  for sequence_length in args.sequence_length:
    for embedding_dim in args.embedding_dim:
      for hidden_dim in args.hidden_dim:
        reports.append(
            run_one_model(
                model_name=(
                    f'text_cnn_len{sequence_length}_e{embedding_dim}_h{hidden_dim}'
                ),
                train_prompts=train_prompts,
                train_vectors=train_vectors,
                eval_prompts=eval_prompts,
                eval_vectors=eval_vectors,
                eval_tokens=eval_tokens,
                benchmark_prompts=benchmark_prompts,
                codebooks=codebooks,
                sequence_length=sequence_length,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                batch_size=args.batch_size,
                epochs=args.epochs,
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
  parser.add_argument('--eval-count', type=int, default=800)
  parser.add_argument('--benchmark-count', type=int, default=1000)
  parser.add_argument('--teacher-batch-size', type=int, default=128)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--backend-name', default='cpu_compat')
  parser.add_argument('--device', default='/CPU:0')
  parser.add_argument('--runtime-style-token-depth', type=int, default=6)
  parser.add_argument('--sequence-length', nargs='*', type=int, default=[32])
  parser.add_argument('--embedding-dim', nargs='*', type=int, default=[64])
  parser.add_argument('--hidden-dim', nargs='*', type=int, default=[128])
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--epochs', type=int, default=20)
  parser.add_argument('--allow-suffix-drop', action='store_true')
  return parser.parse_args()


def main() -> int:
  tf.random.set_seed(7)
  result = run(parse_args())
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

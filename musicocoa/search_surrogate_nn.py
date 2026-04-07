#!/usr/bin/env python3

"""Searches nonlinear categorical prompt surrogates against the legacy teacher."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence

import numpy as np
import tensorflow as tf

from musicocoa.prompt_space import DEFAULT_PROMPT_SPACE
from musicocoa.prompt_space import generate_unique_prompts
from musicocoa.prompt_space import prompt_to_ids


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


def prompts_to_ids(prompts: Sequence[str]) -> np.ndarray:
  return np.stack([prompt_to_ids(prompt) for prompt in prompts], axis=0).astype(np.int32)


def build_model(
    *,
    embedding_dim: int,
    hidden_dim: int,
    runtime_style_token_depth: int,
) -> tf.keras.Model:
  inputs = tf.keras.Input(
      shape=(len(DEFAULT_PROMPT_SPACE),),
      dtype=tf.int32,
      name='prompt_ids',
  )
  embeddings = []
  for index, phrases in enumerate(DEFAULT_PROMPT_SPACE):
    token_slice = tf.keras.layers.Lambda(
        lambda tensor, idx=index: tensor[:, idx], name=f'pick_{index}'
    )(inputs)
    embedded = tf.keras.layers.Embedding(
        input_dim=len(phrases) + 1,
        output_dim=embedding_dim,
        name=f'embed_{index}',
    )(token_slice)
    embeddings.append(embedded)
  x = tf.keras.layers.Concatenate(name='concat_embeddings')(embeddings)
  x = tf.keras.layers.Flatten(name='flatten_embeddings')(x)
  x = tf.keras.layers.Dense(hidden_dim, activation='gelu', name='hidden_0')(x)
  x = tf.keras.layers.Dense(hidden_dim, activation='gelu', name='hidden_1')(x)
  outputs = [
      tf.keras.layers.Dense(1024, name=f'token_{depth}')(x)
      for depth in range(runtime_style_token_depth)
  ]
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  losses = {
      f'token_{depth}': tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True
      )
      for depth in range(runtime_style_token_depth)
  }
  metrics = {
      f'token_{depth}': [tf.keras.metrics.SparseCategoricalAccuracy()]
      for depth in range(runtime_style_token_depth)
  }
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
      loss=losses,
      metrics=metrics,
  )
  return model


def benchmark_model(
    model: tf.keras.Model,
    prompts: Sequence[str],
    repeats: int = 10,
) -> float:
  infer = tf.function(model, reduce_retracing=True)
  prompt_ids = prompts_to_ids(prompts)
  tensor = tf.convert_to_tensor(prompt_ids, dtype=tf.int32)
  _ = infer(tensor, training=False)
  timings_ms: list[float] = []
  for _ in range(repeats):
    started = time.perf_counter()
    outputs = infer(tensor, training=False)
    _ = [tf.argmax(output, axis=1).numpy() for output in outputs]
    timings_ms.append((time.perf_counter() - started) * 1000.0)
  return float(np.mean(timings_ms))


def run_one_model(
    *,
    model_name: str,
    train_ids: np.ndarray,
    train_tokens: np.ndarray,
    eval_ids: np.ndarray,
    eval_tokens: np.ndarray,
    benchmark_prompts: Sequence[str],
    embedding_dim: int,
    hidden_dim: int,
    runtime_style_token_depth: int,
    batch_size: int,
    epochs: int,
) -> ModelReport:
  model = build_model(
      embedding_dim=embedding_dim,
      hidden_dim=hidden_dim,
      runtime_style_token_depth=runtime_style_token_depth,
  )
  callbacks = [
      tf.keras.callbacks.EarlyStopping(
          monitor='val_loss',
          patience=3,
          restore_best_weights=True,
      )
  ]
  history = model.fit(
      train_ids,
      [train_tokens[:, depth] for depth in range(runtime_style_token_depth)],
      validation_data=(
          eval_ids,
          [eval_tokens[:, depth] for depth in range(runtime_style_token_depth)],
      ),
      batch_size=batch_size,
      epochs=epochs,
      verbose=0,
      callbacks=callbacks,
  )
  logits = model(eval_ids, training=False)
  predicted_tokens = np.stack(
      [tf.argmax(output, axis=1).numpy() for output in logits],
      axis=1,
  ).astype(np.int32)
  exact, slot_accuracy, mean_hamming = token_metrics(predicted_tokens, eval_tokens)
  return ModelReport(
      model_name=model_name,
      epochs_run=len(history.history['loss']),
      prompt_exact=exact,
      slot_accuracy=slot_accuracy,
      mean_hamming=mean_hamming,
      benchmark_ms_1000=benchmark_model(model, benchmark_prompts),
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

  train_ids = prompts_to_ids([sample.prompt for sample in train_samples])
  eval_ids = prompts_to_ids([sample.prompt for sample in eval_samples])
  train_tokens = np.asarray(
      [sample.runtime_tokens for sample in train_samples], dtype=np.int32
  )
  eval_tokens = np.asarray(
      [sample.runtime_tokens for sample in eval_samples], dtype=np.int32
  )

  reports = []
  for embedding_dim in args.embedding_dim:
    for hidden_dim in args.hidden_dim:
      reports.append(
          run_one_model(
              model_name=f'phrase_mlp_e{embedding_dim}_h{hidden_dim}',
              train_ids=train_ids,
              train_tokens=train_tokens,
              eval_ids=eval_ids,
              eval_tokens=eval_tokens,
              benchmark_prompts=benchmark_prompts,
              embedding_dim=embedding_dim,
              hidden_dim=hidden_dim,
              runtime_style_token_depth=args.runtime_style_token_depth,
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
  parser.add_argument('--train-count', type=int, default=6000)
  parser.add_argument('--eval-count', type=int, default=1000)
  parser.add_argument('--benchmark-count', type=int, default=1000)
  parser.add_argument('--teacher-batch-size', type=int, default=128)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--runtime-style-token-depth', type=int, default=6)
  parser.add_argument('--embedding-dim', nargs='*', type=int, default=[16, 32])
  parser.add_argument('--hidden-dim', nargs='*', type=int, default=[256, 384])
  parser.add_argument('--batch-size', type=int, default=512)
  parser.add_argument('--epochs', type=int, default=25)
  parser.add_argument('--allow-suffix-drop', action='store_true')
  return parser.parse_args()


def main() -> int:
  tf.random.set_seed(7)
  result = run(parse_args())
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

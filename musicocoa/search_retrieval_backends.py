#!/usr/bin/env python3

"""Searches retrieval-based backends against the legacy teacher."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel
from transformers import AutoTokenizer

if __package__ in (None, ''):
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
  from musicocoa.backend import BackendConfig
  from musicocoa.backend import MusicCoCaBackend
  from musicocoa.backend import rvq_dequantize
  from musicocoa.backend import rvq_quantization
  from musicocoa.prompt_space import generate_unique_prompts
else:
  from .backend import BackendConfig
  from .backend import MusicCoCaBackend
  from .backend import rvq_dequantize
  from .backend import rvq_quantization
  from .prompt_space import generate_unique_prompts


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


@dataclasses.dataclass(frozen=True)
class EncoderBundle:
  tokenizer: object
  model: object
  device: torch.device


_ENCODER_CACHE: dict[str, EncoderBundle] = {}


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


def encode_prompts(
    *,
    model_name: str,
    prompts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
  bundle = _ENCODER_CACHE.get(model_name)
  if bundle is None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    bundle = EncoderBundle(tokenizer=tokenizer, model=model, device=device)
    _ENCODER_CACHE[model_name] = bundle
  tokenizer = bundle.tokenizer
  model = bundle.model
  device = bundle.device
  outputs = []
  autocast_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
  with torch.inference_mode():
    for start in range(0, len(prompts), batch_size):
      batch = list(prompts[start : start + batch_size])
      encoded = tokenizer(
          batch,
          padding=True,
          truncation=True,
          max_length=32,
          return_tensors='pt',
      )
      encoded = {key: value.to(device) for key, value in encoded.items()}
      with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        hidden = model(**encoded, return_dict=True).last_hidden_state
      mask = encoded['attention_mask'].unsqueeze(-1).to(hidden.dtype)
      pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
      pooled = F.normalize(pooled.float(), dim=1)
      outputs.append(pooled.cpu().numpy().astype(np.float32))
  return np.concatenate(outputs, axis=0)


def retrieval_topk(
    train_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
  scores = query_embeddings @ train_embeddings.T
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


def predict_vector_average(
    train_vectors: np.ndarray,
    train_tokens: np.ndarray,
    codebooks: np.ndarray,
    topk_idx: np.ndarray,
    topk_scores: np.ndarray,
    *,
    snap_to_neighbor: bool,
) -> np.ndarray:
  weights = np.exp(topk_scores - np.max(topk_scores, axis=1, keepdims=True))
  weights = weights / np.sum(weights, axis=1, keepdims=True)
  neighbor_vectors = train_vectors[topk_idx]
  predicted_vectors = np.sum(neighbor_vectors * weights[:, :, None], axis=1).astype(np.float32)
  if not snap_to_neighbor:
    return rvq_quantization(predicted_vectors, codebooks)[0]
  predicted = np.zeros((topk_idx.shape[0], train_tokens.shape[1]), dtype=np.int32)
  pred_unit = predicted_vectors / np.maximum(np.linalg.norm(predicted_vectors, axis=1, keepdims=True), 1e-8)
  neighbor_unit = neighbor_vectors / np.maximum(
      np.linalg.norm(neighbor_vectors, axis=2, keepdims=True),
      1e-8,
  )
  scores = np.sum(pred_unit[:, None, :] * neighbor_unit, axis=2)
  nearest = np.argmax(scores, axis=1)
  for row_index, neighbor_rank in enumerate(nearest.tolist()):
    predicted[row_index] = train_tokens[topk_idx[row_index, neighbor_rank]]
  return predicted


def benchmark_backend(
    predict_fn,
    benchmark_prompts: Sequence[str],
    model_name: str,
    repeats: int = 10,
) -> float:
  timings = []
  for _ in range(repeats):
    started = time.perf_counter()
    _ = predict_fn(list(benchmark_prompts), model_name=model_name)
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

  embed_start = time.perf_counter()
  train_embeddings = encode_prompts(
      model_name=args.model_name,
      prompts=prompts_train,
      batch_size=args.embed_batch_size,
  )
  eval_embeddings = encode_prompts(
      model_name=args.model_name,
      prompts=prompts_eval,
      batch_size=args.embed_batch_size,
  )
  embed_ms = (time.perf_counter() - embed_start) * 1000.0

  backend = MusicCoCaBackend(
      BackendConfig(
          backend_name=args.backend_name,
          device=args.device,
          runtime_style_token_depth=args.runtime_style_token_depth,
      )
  )
  codebooks = backend._rvq_codebooks[: args.runtime_style_token_depth].copy()
  train_vectors = rvq_dequantize(train_tokens, codebooks).astype(np.float32)

  reports: list[ModelReport] = []
  for k in args.knn_k:
    topk_idx, topk_scores = retrieval_topk(train_embeddings, eval_embeddings, k)

    predicted_vote = predict_sequence_vote(train_tokens, topk_idx, topk_scores)
    exact, slot_accuracy, mean_hamming = token_metrics(predicted_vote, eval_tokens)

    def vote_predict(prompts_batch: list[str], *, model_name: str) -> np.ndarray:
      query_embeddings = encode_prompts(
          model_name=model_name,
          prompts=prompts_batch,
          batch_size=args.embed_batch_size,
      )
      idx, scores = retrieval_topk(train_embeddings, query_embeddings, k)
      return predict_sequence_vote(train_tokens, idx, scores)

    reports.append(
        ModelReport(
            model_name=f'sequence_vote_k{k}',
            prompt_exact=exact,
            slot_accuracy=slot_accuracy,
            mean_hamming=mean_hamming,
            benchmark_ms_250=benchmark_backend(
                vote_predict,
                benchmark_prompts,
                args.model_name,
            ),
        )
    )

    predicted_neighbor_snap = predict_vector_average(
        train_vectors,
        train_tokens,
        codebooks,
        topk_idx,
        topk_scores,
        snap_to_neighbor=True,
    )
    exact, slot_accuracy, mean_hamming = token_metrics(predicted_neighbor_snap, eval_tokens)

    def neighbor_snap_predict(prompts_batch: list[str], *, model_name: str) -> np.ndarray:
      query_embeddings = encode_prompts(
          model_name=model_name,
          prompts=prompts_batch,
          batch_size=args.embed_batch_size,
      )
      idx, scores = retrieval_topk(train_embeddings, query_embeddings, k)
      return predict_vector_average(
          train_vectors,
          train_tokens,
          codebooks,
          idx,
          scores,
          snap_to_neighbor=True,
      )

    reports.append(
        ModelReport(
            model_name=f'neighbor_snap_k{k}',
            prompt_exact=exact,
            slot_accuracy=slot_accuracy,
            mean_hamming=mean_hamming,
            benchmark_ms_250=benchmark_backend(
                neighbor_snap_predict,
                benchmark_prompts,
                args.model_name,
            ),
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
      'teacher_label_ms': teacher_ms,
      'embedding_ms': embed_ms,
      'train_count': args.train_count,
      'eval_count': args.eval_count,
      'benchmark_count': args.benchmark_count,
      'reports': [dataclasses.asdict(report) for report in reports_sorted],
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--teacher-url', default='http://127.0.0.1:8770')
  parser.add_argument('--model-name', default='sentence-transformers/all-MiniLM-L6-v2')
  parser.add_argument('--train-count', type=int, default=10000)
  parser.add_argument('--eval-count', type=int, default=1500)
  parser.add_argument('--benchmark-count', type=int, default=250)
  parser.add_argument('--teacher-batch-size', type=int, default=128)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--backend-name', default='cpu_compat')
  parser.add_argument('--device', default='/CPU:0')
  parser.add_argument('--runtime-style-token-depth', type=int, default=6)
  parser.add_argument('--embed-batch-size', type=int, default=512)
  parser.add_argument('--knn-k', nargs='*', type=int, default=[4, 8, 16, 32])
  parser.add_argument('--allow-suffix-drop', action='store_true')
  return parser.parse_args()


def main() -> int:
  result = run(parse_args())
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

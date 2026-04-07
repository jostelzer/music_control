#!/usr/bin/env python3

"""Searches Hugging Face torch students against the legacy teacher."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from transformers import AutoModel
from transformers import AutoTokenizer

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
              runtime_tokens=np.asarray(item['runtime_tokens'], dtype=np.int64),
          )
      )
  return records


def token_metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
  exact = float(np.mean(np.all(predicted == target, axis=1)))
  slot_accuracy = float(np.mean(predicted == target))
  mean_hamming = float(np.mean(np.sum(predicted != target, axis=1)))
  return exact, slot_accuracy, mean_hamming


class PromptStudent(nn.Module):
  def __init__(
      self,
      *,
      model_name: str,
      runtime_style_token_depth: int,
      hidden_dim: int,
      freeze_encoder: bool,
  ):
    super().__init__()
    self.encoder = AutoModel.from_pretrained(model_name)
    hidden_size = int(self.encoder.config.hidden_size)
    self.proj = nn.Linear(hidden_size, hidden_dim)
    self.heads = nn.ModuleList(
        [nn.Linear(hidden_dim, 1024) for _ in range(runtime_style_token_depth)]
    )
    if freeze_encoder:
      for parameter in self.encoder.parameters():
        parameter.requires_grad = False

  def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
    encoded = self.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    ).last_hidden_state
    mask = attention_mask.unsqueeze(-1).to(encoded.dtype)
    pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    hidden = F.gelu(self.proj(pooled))
    return [head(hidden) for head in self.heads]


def collate_prompts(
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    target_tokens: np.ndarray,
    max_length: int,
) -> TensorDataset:
  encoded = tokenizer(
      list(prompts),
      padding=True,
      truncation=True,
      max_length=max_length,
      return_tensors='pt',
  )
  targets = torch.from_numpy(target_tokens.astype(np.int64))
  return TensorDataset(encoded['input_ids'], encoded['attention_mask'], targets)


def evaluate_model(
    model: PromptStudent,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
  model.eval()
  predicted_batches: list[np.ndarray] = []
  autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
  autocast_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
  with torch.inference_mode():
    for input_ids, attention_mask, _ in dataloader:
      input_ids = input_ids.to(device)
      attention_mask = attention_mask.to(device)
      with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
        logits = model(input_ids, attention_mask)
      predicted = torch.stack([head.argmax(dim=1) for head in logits], dim=1)
      predicted_batches.append(predicted.cpu().numpy())
  return np.concatenate(predicted_batches, axis=0)


def benchmark_model(
    model: PromptStudent,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    device: torch.device,
    max_length: int,
    repeats: int = 10,
) -> float:
  model.eval()
  autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
  autocast_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
  # Warm tokenizer + model.
  encoded = tokenizer(
      list(prompts),
      padding=True,
      truncation=True,
      max_length=max_length,
      return_tensors='pt',
  )
  encoded = {key: value.to(device) for key, value in encoded.items()}
  with torch.inference_mode():
    with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
      logits = model(encoded['input_ids'], encoded['attention_mask'])
    _ = torch.stack([head.argmax(dim=1) for head in logits], dim=1).cpu().numpy()

  timings_ms: list[float] = []
  for _ in range(repeats):
    started = time.perf_counter()
    encoded = tokenizer(
        list(prompts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
      with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
        logits = model(encoded['input_ids'], encoded['attention_mask'])
      _ = torch.stack([head.argmax(dim=1) for head in logits], dim=1).cpu().numpy()
    if device.type == 'cuda':
      torch.cuda.synchronize(device)
    timings_ms.append((time.perf_counter() - started) * 1000.0)
  return float(np.mean(timings_ms))


def run_one_model(
    *,
    model_name: str,
    prompts_train: Sequence[str],
    tokens_train: np.ndarray,
    prompts_eval: Sequence[str],
    tokens_eval: np.ndarray,
    benchmark_prompts: Sequence[str],
    runtime_style_token_depth: int,
    hidden_dim: int,
    freeze_encoder: bool,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    max_length: int,
) -> ModelReport:
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  train_ds = collate_prompts(tokenizer, prompts_train, tokens_train, max_length)
  eval_ds = collate_prompts(tokenizer, prompts_eval, tokens_eval, max_length)
  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

  model = PromptStudent(
      model_name=model_name,
      runtime_style_token_depth=runtime_style_token_depth,
      hidden_dim=hidden_dim,
      freeze_encoder=freeze_encoder,
  ).to(device)
  optimizer = torch.optim.AdamW(
      [parameter for parameter in model.parameters() if parameter.requires_grad],
      lr=learning_rate,
  )

  best_state = None
  best_eval_loss = float('inf')
  epochs_run = 0
  autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
  autocast_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
  for epoch in range(epochs):
    model.train()
    for input_ids, attention_mask, targets in train_loader:
      input_ids = input_ids.to(device)
      attention_mask = attention_mask.to(device)
      targets = targets.to(device)
      optimizer.zero_grad(set_to_none=True)
      with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
        logits = model(input_ids, attention_mask)
        loss = sum(
            F.cross_entropy(logits[depth], targets[:, depth])
            for depth in range(runtime_style_token_depth)
        )
      loss.backward()
      optimizer.step()

    model.eval()
    eval_losses = []
    with torch.inference_mode():
      for input_ids, attention_mask, targets in eval_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)
        with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
          logits = model(input_ids, attention_mask)
          eval_loss = sum(
              F.cross_entropy(logits[depth], targets[:, depth])
              for depth in range(runtime_style_token_depth)
          )
        eval_losses.append(float(eval_loss.detach().cpu()))
    mean_eval_loss = float(np.mean(eval_losses))
    epochs_run = epoch + 1
    if mean_eval_loss < best_eval_loss:
      best_eval_loss = mean_eval_loss
      best_state = {
          key: value.detach().cpu().clone() for key, value in model.state_dict().items()
      }
    elif epoch >= 2 and mean_eval_loss > best_eval_loss * 1.002:
      break

  assert best_state is not None
  model.load_state_dict(best_state)
  predicted_tokens = evaluate_model(model, eval_loader, device)
  exact, slot_accuracy, mean_hamming = token_metrics(predicted_tokens, tokens_eval)
  benchmark_ms = benchmark_model(
      model,
      tokenizer,
      benchmark_prompts,
      device,
      max_length=max_length,
  )
  return ModelReport(
      model_name=(
          f'{model_name}|freeze={int(freeze_encoder)}|hidden={hidden_dim}'
      ),
      epochs_run=epochs_run,
      prompt_exact=exact,
      slot_accuracy=slot_accuracy,
      mean_hamming=mean_hamming,
      benchmark_ms_1000=benchmark_ms,
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

  prompts_train = [sample.prompt for sample in train_samples]
  prompts_eval = [sample.prompt for sample in eval_samples]
  tokens_train = np.asarray(
      [sample.runtime_tokens for sample in train_samples], dtype=np.int64
  )
  tokens_eval = np.asarray(
      [sample.runtime_tokens for sample in eval_samples], dtype=np.int64
  )

  reports = []
  for freeze_encoder in args.freeze_encoder:
    for hidden_dim in args.hidden_dim:
      reports.append(
          run_one_model(
              model_name=args.model_name,
              prompts_train=prompts_train,
              tokens_train=tokens_train,
              prompts_eval=prompts_eval,
              tokens_eval=tokens_eval,
              benchmark_prompts=benchmark_prompts,
              runtime_style_token_depth=args.runtime_style_token_depth,
              hidden_dim=hidden_dim,
              freeze_encoder=bool(freeze_encoder),
              batch_size=args.batch_size,
              epochs=args.epochs,
              learning_rate=args.learning_rate,
              max_length=args.max_length,
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
  parser.add_argument('--model-name', default='google/bert_uncased_L-2_H-128_A-2')
  parser.add_argument('--train-count', type=int, default=3000)
  parser.add_argument('--eval-count', type=int, default=800)
  parser.add_argument('--benchmark-count', type=int, default=1000)
  parser.add_argument('--teacher-batch-size', type=int, default=128)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--runtime-style-token-depth', type=int, default=6)
  parser.add_argument('--hidden-dim', nargs='*', type=int, default=[256])
  parser.add_argument('--freeze-encoder', nargs='*', type=int, default=[1])
  parser.add_argument('--batch-size', type=int, default=128)
  parser.add_argument('--epochs', type=int, default=8)
  parser.add_argument('--learning-rate', type=float, default=1e-3)
  parser.add_argument('--max-length', type=int, default=32)
  parser.add_argument('--allow-suffix-drop', action='store_true')
  return parser.parse_args()


def main() -> int:
  result = run(parse_args())
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

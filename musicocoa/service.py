"""Repo-local MusicCoCa embedding cache and worker-pool service."""

from __future__ import annotations

import dataclasses
import hashlib
from concurrent import futures
from typing import Any

import numpy as np

from .backend import BackendConfig
from .backend import EmbeddedPrompt
from .backend import MusicCoCaBackend
from .backend import embed_prompt_worker
from .backend import init_worker


def _sha_prefix(prefix: str, payload: bytes) -> str:
  return f'{prefix}_{hashlib.sha256(payload).hexdigest()[:8]}'


@dataclasses.dataclass
class StoredPrompt:
  embedding_id: str
  prompt: str
  embedding: np.ndarray
  full_tokens: np.ndarray
  runtime_tokens: np.ndarray
  norm: float
  backend_name: str
  device: str


class MusicCoCaLab:
  """Prompt embedding cache with optional process-pool acceleration."""

  def __init__(
      self,
      *,
      backend_name: str = 'cpu_compat',
      device: str = '/CPU:0',
      runtime_style_token_depth: int = 6,
      workers: int = 0,
      disable_tf32: bool = False,
  ):
    requested_workers = max(0, int(workers))
    if backend_name == 'xla_exact':
      requested_workers = 0
    self._config = BackendConfig(
        backend_name=backend_name,
        device=device,
        runtime_style_token_depth=runtime_style_token_depth,
        disable_tf32=disable_tf32,
    )
    self._workers = requested_workers
    self._entries: dict[str, StoredPrompt] = {}
    self._prompt_cache: dict[str, str] = {}
    self._backend: MusicCoCaBackend | None = None
    self._executor: futures.ProcessPoolExecutor | None = None

    if self._workers > 0:
      self._executor = futures.ProcessPoolExecutor(
          max_workers=self._workers,
          initializer=init_worker,
          initargs=(dataclasses.asdict(self._config),),
      )

  def close(self) -> None:
    if self._executor is not None:
      self._executor.shutdown(wait=True, cancel_futures=False)
      self._executor = None

  @property
  def workers(self) -> int:
    return self._workers

  def _ensure_backend(self) -> MusicCoCaBackend:
    if self._backend is None:
      self._backend = MusicCoCaBackend(self._config)
    return self._backend

  def _store(self, embedded: EmbeddedPrompt) -> StoredPrompt:
    existing = self._entries.get(embedded.embedding_id)
    if existing is not None:
      self._prompt_cache[embedded.prompt] = existing.embedding_id
      return existing
    stored = StoredPrompt(
        embedding_id=embedded.embedding_id,
        prompt=embedded.prompt,
        embedding=embedded.embedding,
        full_tokens=embedded.full_tokens,
        runtime_tokens=embedded.runtime_tokens,
        norm=embedded.norm,
        backend_name=embedded.backend_name,
        device=embedded.device,
    )
    self._entries[stored.embedding_id] = stored
    self._prompt_cache[stored.prompt] = stored.embedding_id
    return stored

  def _record(
      self,
      entry: StoredPrompt,
      *,
      cached: bool,
      include_embedding: bool,
      include_full_tokens: bool,
  ) -> dict[str, Any]:
    record: dict[str, Any] = {
        'embedding_id': entry.embedding_id,
        'text': entry.prompt,
        'embedding_dim': int(entry.embedding.shape[0]),
        'norm': float(entry.norm),
        'runtime_tokens': [int(token) for token in entry.runtime_tokens.tolist()],
        'cached': cached,
        'backend_name': entry.backend_name,
        'device': entry.device,
    }
    if include_embedding:
      record['embedding'] = [float(value) for value in entry.embedding.tolist()]
    if include_full_tokens:
      record['full_tokens'] = [int(token) for token in entry.full_tokens.tolist()]
    return record

  def _embed_uncached_prompt(self, prompt: str) -> EmbeddedPrompt:
    if self._executor is None:
      return self._ensure_backend().embed_prompt(prompt)
    payload = next(self._executor.map(embed_prompt_worker, [prompt]))
    return EmbeddedPrompt.from_wire(payload)

  def embed_text(
      self,
      text: str,
      *,
      include_embedding: bool = False,
      include_full_tokens: bool = False,
  ) -> dict[str, Any]:
    prompt = text.strip()
    if not prompt:
      raise ValueError('Prompt text must be non-empty.')
    cached_embedding_id = self._prompt_cache.get(prompt)
    if cached_embedding_id is not None:
      return self._record(
          self._entries[cached_embedding_id],
          cached=True,
          include_embedding=include_embedding,
          include_full_tokens=include_full_tokens,
      )

    stored = self._store(self._embed_uncached_prompt(prompt))
    return self._record(
        stored,
        cached=False,
        include_embedding=include_embedding,
        include_full_tokens=include_full_tokens,
    )

  def embed_batch_text(
      self,
      texts: list[str],
      *,
      include_embedding: bool = False,
      include_full_tokens: bool = False,
  ) -> list[dict[str, Any]]:
    if not isinstance(texts, list):
      raise TypeError('`texts` must be a list of prompt strings.')
    if not texts:
      return []

    results: list[dict[str, Any] | None] = [None] * len(texts)
    pending_prompts: list[str] = []
    pending_indices: dict[str, list[int]] = {}

    for index, raw_text in enumerate(texts):
      prompt = str(raw_text).strip()
      if not prompt:
        raise ValueError('Batch prompts must be non-empty.')
      cached_embedding_id = self._prompt_cache.get(prompt)
      if cached_embedding_id is not None:
        results[index] = self._record(
            self._entries[cached_embedding_id],
            cached=True,
            include_embedding=include_embedding,
            include_full_tokens=include_full_tokens,
        )
        continue
      if prompt not in pending_indices:
        pending_prompts.append(prompt)
        pending_indices[prompt] = []
      pending_indices[prompt].append(index)

    if pending_prompts:
      if self._executor is None:
        embedded_items = self._ensure_backend().embed_prompts(pending_prompts)
      else:
        embedded_items = [
            EmbeddedPrompt.from_wire(payload)
            for payload in self._executor.map(embed_prompt_worker, pending_prompts)
        ]

      for embedded in embedded_items:
        stored = self._store(embedded)
        record = self._record(
            stored,
            cached=False,
            include_embedding=include_embedding,
            include_full_tokens=include_full_tokens,
        )
        for index in pending_indices[stored.prompt]:
          results[index] = dict(record)

    assert all(result is not None for result in results)
    return [result for result in results if result is not None]

  def health(self) -> dict[str, Any]:
    metadata: dict[str, Any]
    if self._backend is not None:
      metadata = self._backend.backend_metadata()
    else:
      supports_true_batching = self._config.backend_name == 'xla_exact'
      preloaded_notes = [
          'Backend not loaded yet; metadata is static until first embed.',
      ]
      if supports_true_batching:
        preloaded_notes.append(
            'xla_exact will load the legacy StableHLO graph and batch on the selected device.'
        )
        if self._config.disable_tf32:
          preloaded_notes.append('TF32 disabled for stricter float32 fidelity.')
        else:
          preloaded_notes.append('TF32 enabled for faster GPU matmul throughput.')
      else:
        preloaded_notes.append(
            'Use backend=xla_exact for direct batched XlaCallModule execution.'
        )
      metadata = {
          'backend_name': self._config.backend_name,
          'device': self._config.device,
          'runtime_style_token_depth': self._config.runtime_style_token_depth,
          'supports_true_batching': supports_true_batching,
          'notes': preloaded_notes,
      }
    metadata.update(
        {
            'status': 'ok',
            'workers': self._workers,
            'cached_prompts': len(self._prompt_cache),
            'cached_embeddings': len(self._entries),
        }
    )
    return metadata

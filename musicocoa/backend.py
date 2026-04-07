"""Repo-local MusicCoCa backend helpers.

This module keeps the prompt-embedding service logic in this repo while still
reusing the upstream Magenta RT asset cache and model files.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import os
import pathlib
import sys
from typing import Any

import numpy as np


MUSICCOCA_RVQ_VAR_ORDER = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3]
DEFAULT_RUNTIME_STYLE_TOKEN_DEPTH = 6
DEFAULT_EMBEDDING_DIM = 768
DEFAULT_RVQ_DEPTH = 12
DEFAULT_RVQ_CODEBOOK_SIZE = 1024

_MAX_TEXT_LENGTH = 128
_TARGET_SOS_ID = 1
_WORKER_BACKEND: 'MusicCoCaBackend | None' = None


def _candidate_magenta_rt_roots() -> list[pathlib.Path]:
  here = pathlib.Path(__file__).resolve()
  candidates = []
  env_path = os.environ.get('MAGENTA_REALTIME_REPO')
  if env_path:
    candidates.append(pathlib.Path(env_path).expanduser())
  candidates.extend(
      [
          here.parents[2] / 'magenta-realtime',
          pathlib.Path.home() / 'git' / 'magenta-realtime',
          pathlib.Path('/mnt/ai/git/magenta-realtime'),
      ]
  )
  unique = []
  seen = set()
  for candidate in candidates:
    resolved = candidate.resolve()
    if resolved in seen:
      continue
    seen.add(resolved)
    unique.append(resolved)
  return unique


def ensure_magenta_rt_importable() -> None:
  if importlib.util.find_spec('magenta_rt') is not None:
    return
  for candidate in _candidate_magenta_rt_roots():
    if (candidate / 'magenta_rt' / 'asset.py').exists():
      sys.path.insert(0, str(candidate))
      return
  raise RuntimeError(
      'Could not locate the sibling `magenta-realtime` repo. Set '
      '`MAGENTA_REALTIME_REPO` to a checkout that contains `magenta_rt`.'
  )


def _normalize_embedding(embedding: np.ndarray) -> tuple[np.ndarray, float]:
  vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
  norm = float(np.linalg.norm(vector))
  if norm <= 0.0:
    raise ValueError('Embedding norm must be > 0.')
  return vector / norm, norm


def _sha_prefix(prefix: str, payload: bytes) -> str:
  return f'{prefix}_{hashlib.sha256(payload).hexdigest()[:8]}'


def _device_to_xla_platform(device: str) -> str:
  normalized = device.strip().lower()
  if normalized.startswith('/gpu:'):
    return 'CUDA'
  if normalized.startswith('/cpu:'):
    return 'CPU'
  raise ValueError(f'Unsupported TensorFlow device for XLA backend: {device!r}')


def _make_ids_paddings(prompts: list[str], vocab) -> tuple[np.ndarray, np.ndarray]:
  ids_rows = []
  paddings_rows = []
  for prompt in prompts:
    labels = vocab.EncodeAsIds(prompt.lower())
    num_tokens = min(len(labels), _MAX_TEXT_LENGTH - 1)
    labels = labels[: _MAX_TEXT_LENGTH - 1]
    ids = [_TARGET_SOS_ID] + labels
    ids += [0] * (_MAX_TEXT_LENGTH - len(ids))
    paddings = np.ones((_MAX_TEXT_LENGTH,), dtype=np.float32)
    paddings[: num_tokens + 1] = 0.0
    ids_rows.append(ids)
    paddings_rows.append(paddings)
  return (
      np.asarray(ids_rows, dtype=np.int32),
      np.asarray(paddings_rows, dtype=np.float32),
  )


def _extract_embed_text_module(concrete_fn) -> dict[str, Any]:
  graph_def = concrete_fn.graph.as_graph_def()
  function_defs = {f.signature.name: f for f in graph_def.library.function}

  wrapper_node = None
  for node in graph_def.node:
    if node.op == 'StatefulPartitionedCall':
      wrapper_node = node
      break
  if wrapper_node is None:
    raise RuntimeError('Could not find top-level StatefulPartitionedCall.')

  wrapper_fn = function_defs[wrapper_node.attr['f'].func.name]
  xla_call_name = None
  preprocess_name = None
  for node in wrapper_fn.node_def:
    if node.name == 'StatefulPartitionedCall':
      preprocess_name = node.attr['f'].func.name
    elif node.name == 'StatefulPartitionedCall_1':
      xla_call_name = node.attr['f'].func.name
  if xla_call_name is None or preprocess_name is None:
    raise RuntimeError('Could not resolve embed_text wrapper subfunctions.')

  xla_fn = function_defs[xla_call_name]
  preprocess_fn = function_defs[preprocess_name]
  xla_node = None
  for node in xla_fn.node_def:
    if node.op == 'XlaCallModule':
      xla_node = node
      break
  if xla_node is None:
    raise RuntimeError('Could not find XlaCallModule node.')

  attrs = xla_node.attr
  return {
      'module_bytes': bytes(attrs['module'].s),
      'version': int(attrs['version'].i),
      'sout': [
          [dim.size for dim in shape.dim] for shape in attrs['Sout'].list.shape
      ],
      'tout': list(attrs['Tout'].list.type),
      'platforms': [
          value.decode('utf-8', errors='replace')
          for value in attrs['platforms'].list.s
      ],
      'preprocess_uses_random': any(
          node.op == 'RandomUniformInt' for node in preprocess_fn.node_def
      ),
  }


def rvq_quantization(
    embeddings: np.ndarray,
    codebooks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Performs RVQ quantization on a batch of embeddings."""
  if embeddings.ndim != 2:
    raise ValueError(f'embeddings must be 2D, got {embeddings.shape}')
  if codebooks.ndim != 3:
    raise ValueError(f'codebooks must be 3D, got {codebooks.shape}')
  if embeddings.shape[1] != codebooks.shape[2]:
    raise ValueError('embedding dim must match codebook dim')

  batch_size, embedding_dim = embeddings.shape
  tokens = np.zeros((batch_size, codebooks.shape[0]), dtype=np.int32)
  residual = embeddings.copy()
  codebook_norms = np.sum(codebooks * codebooks, axis=2)
  for depth in range(codebooks.shape[0]):
    scores = residual @ codebooks[depth].T
    distances = (
        np.sum(residual * residual, axis=1, keepdims=True)
        + codebook_norms[depth][np.newaxis, :]
        - 2.0 * scores
    )
    nearest = np.argmin(distances, axis=1)
    tokens[:, depth] = nearest
    residual = residual - codebooks[depth, nearest, :]
    assert residual.shape == (batch_size, embedding_dim)
  return tokens, residual


def rvq_dequantize(tokens: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
  """Reconstructs embeddings from RVQ token ids."""
  if tokens.ndim != 2:
    raise ValueError(f'tokens must be 2D, got {tokens.shape}')
  if codebooks.ndim != 3:
    raise ValueError(f'codebooks must be 3D, got {codebooks.shape}')
  if tokens.shape[1] > codebooks.shape[0]:
    raise ValueError('token depth cannot exceed codebook depth')

  batch_size = tokens.shape[0]
  embedding_dim = codebooks.shape[2]
  reconstructed = np.zeros((batch_size, embedding_dim), dtype=np.float32)
  for depth in range(tokens.shape[1]):
    reconstructed += codebooks[depth, tokens[:, depth]]
  return reconstructed


@dataclasses.dataclass(frozen=True)
class BackendConfig:
  backend_name: str = 'cpu_compat'
  device: str = '/CPU:0'
  runtime_style_token_depth: int = DEFAULT_RUNTIME_STYLE_TOKEN_DEPTH
  embedding_dim: int = DEFAULT_EMBEDDING_DIM
  rvq_depth: int = DEFAULT_RVQ_DEPTH
  rvq_codebook_size: int = DEFAULT_RVQ_CODEBOOK_SIZE
  disable_tf32: bool = False
  allow_cpu_fallback: bool = True


@dataclasses.dataclass
class EmbeddedPrompt:
  embedding_id: str
  prompt: str
  embedding: np.ndarray
  full_tokens: np.ndarray
  runtime_tokens: np.ndarray
  norm: float
  backend_name: str
  device: str

  def to_wire(self) -> dict[str, Any]:
    return {
        'embedding_id': self.embedding_id,
        'prompt': self.prompt,
        'embedding': [float(value) for value in self.embedding.tolist()],
        'full_tokens': [int(token) for token in self.full_tokens.tolist()],
        'runtime_tokens': [int(token) for token in self.runtime_tokens.tolist()],
        'norm': float(self.norm),
        'backend_name': self.backend_name,
        'device': self.device,
    }

  @classmethod
  def from_wire(cls, payload: dict[str, Any]) -> 'EmbeddedPrompt':
    return cls(
        embedding_id=str(payload['embedding_id']),
        prompt=str(payload['prompt']),
        embedding=np.asarray(payload['embedding'], dtype=np.float32),
        full_tokens=np.asarray(payload['full_tokens'], dtype=np.int32),
        runtime_tokens=np.asarray(payload['runtime_tokens'], dtype=np.int32),
        norm=float(payload['norm']),
        backend_name=str(payload['backend_name']),
        device=str(payload['device']),
    )


class MusicCoCaBackend:
  """Single-process prompt embedding backend."""

  def __init__(self, config: BackendConfig):
    ensure_magenta_rt_importable()
    self._requested_device = config.device
    self._device_fallback_reason: str | None = None

    from magenta_rt import asset
    import tensorflow as tf

    self._asset = asset
    self._tf = tf
    if config.disable_tf32:
      tf.config.experimental.enable_tensor_float_32_execution(False)

    actual_device = config.device
    if config.backend_name == 'xla_exact' and config.device.lower().startswith('/gpu:'):
      gpu_devices = tf.config.list_physical_devices('GPU')
      if not gpu_devices and config.allow_cpu_fallback:
        actual_device = '/CPU:0'
        self._device_fallback_reason = (
            'Requested GPU device was unavailable to TensorFlow; falling back to CPU.'
        )

    self.config = dataclasses.replace(config, device=actual_device)
    self._xla_call_module = None
    self._xla_platform = None
    self._xla_module_info = None
    self._xla_weights = None

    if config.backend_name == 'cpu_compat':
      import tensorflow_text  # noqa: F401

      self._input_mode = 'string'
      self._model_path = asset.fetch(
          'savedmodels/musiccoca_mv212f_cpu_compat',
          is_dir=True,
      )
      self._vocab = None
    elif config.backend_name == 'cpu_novocab':
      import sentencepiece as sentencepiece_processor

      self._input_mode = 'ids'
      self._model_path = asset.fetch(
          'savedmodels/musiccoca_mv212f_cpu_novocab',
          is_dir=True,
      )
      vocab_path = asset.fetch('vocabularies/musiccoca_mv212f_vocab.model')
      vocab = sentencepiece_processor.SentencePieceProcessor()
      vocab.Load(vocab_path)
      self._vocab = vocab
    elif config.backend_name == 'xla_exact':
      import sentencepiece as sentencepiece_processor
      from tensorflow.compiler.tf2xla.ops import gen_xla_ops

      self._input_mode = 'ids'
      self._xla_call_module = gen_xla_ops.xla_call_module
      self._model_path = asset.fetch(
          'savedmodels/musiccoca_mv212f_cpu_novocab',
          is_dir=True,
      )
      vocab_path = asset.fetch('vocabularies/musiccoca_mv212f_vocab.model')
      vocab = sentencepiece_processor.SentencePieceProcessor()
      vocab.Load(vocab_path)
      self._vocab = vocab
    else:
      raise ValueError(f'Unsupported backend: {config.backend_name}')

    model_load_device = '/CPU:0' if config.backend_name == 'xla_exact' else config.device
    with tf.device(model_load_device):
      self._model = tf.saved_model.load(self._model_path)
    self._embed_text = self._model.signatures['embed_text']
    if config.backend_name == 'xla_exact':
      self._xla_platform = _device_to_xla_platform(self.config.device)
      self._xla_module_info = _extract_embed_text_module(self._embed_text)
      self._xla_weights = self._load_xla_weights()
    self._rvq_codebooks = self._load_rvq_codebooks()

  def _load_xla_weights(self) -> list[Any]:
    if self._xla_call_module is None:
      raise RuntimeError('XLA weights requested for non-XLA backend.')
    with self._tf.device('/CPU:0'):
      weight_arrays = [
          np.asarray(
              self._tf.raw_ops.ReadVariableOp(resource=resource, dtype=self._tf.float32).numpy()
          )
          for resource in self._embed_text.captured_inputs
      ]
    with self._tf.device(self.config.device):
      return [self._tf.constant(array) for array in weight_arrays]

  def _load_rvq_codebooks(self) -> np.ndarray:
    path = self._asset.fetch('savedmodels/musiccoca_mv212_quant', is_dir=True)
    var_path = f'{path}/variables/variables'
    codebooks = np.zeros(
        (
            self.config.rvq_depth,
            self.config.rvq_codebook_size,
            self.config.embedding_dim,
        ),
        dtype=np.float32,
    )
    for depth, v_name in enumerate(MUSICCOCA_RVQ_VAR_ORDER):
      var = self._tf.train.load_variable(
          var_path, f'variables/{v_name}/.ATTRIBUTES/VARIABLE_VALUE'
      )
      codebooks[depth] = var.T
    return codebooks

  def _embed_prompt_raw(self, prompt: str) -> np.ndarray:
    prompt = prompt.strip()
    if not prompt:
      raise ValueError('Prompt text must be non-empty.')

    if self._input_mode == 'string':
      with self._tf.device(self.config.device):
        result = self._embed_text(inputs_0=self._tf.constant([prompt]))
      return np.asarray(result['contrastive_txt_embed'].numpy()[0], dtype=np.float32)

    assert self._vocab is not None
    labels = self._vocab.EncodeAsIds(prompt.lower())
    num_tokens = min(len(labels), _MAX_TEXT_LENGTH - 1)
    labels = labels[: _MAX_TEXT_LENGTH - 1]
    ids = [_TARGET_SOS_ID] + labels
    ids += [0] * (_MAX_TEXT_LENGTH - len(ids))
    ids_tensor = self._tf.constant(np.asarray([ids], dtype=np.int32))
    paddings = np.ones((_MAX_TEXT_LENGTH,), dtype=np.float32)
    paddings[: num_tokens + 1] = 0.0
    paddings_tensor = self._tf.constant(paddings.reshape(1, -1))
    with self._tf.device(self.config.device):
      result = self._embed_text(inputs_0=ids_tensor, inputs_0_1=paddings_tensor)
    return np.asarray(result['contrastive_txt_embed'].numpy()[0], dtype=np.float32)

  def _embed_batch_raw_exact_xla(self, prompts: list[str]) -> np.ndarray:
    if not prompts:
      return np.zeros((0, self.config.embedding_dim), dtype=np.float32)
    if self._xla_call_module is None or self._xla_weights is None:
      raise RuntimeError('Exact XLA backend was not initialized correctly.')
    if self._vocab is None or self._xla_module_info is None or self._xla_platform is None:
      raise RuntimeError('Exact XLA backend is missing vocab or module metadata.')

    ids_np, paddings_np = _make_ids_paddings(prompts, self._vocab)
    with self._tf.device(self.config.device):
      ids = self._tf.constant(ids_np, dtype=self._tf.int32)
      paddings = self._tf.constant(paddings_np, dtype=self._tf.float32)
      x2_zero = self._tf.constant(0, dtype=self._tf.int32)
      outputs = self._xla_call_module(
          args=[*self._xla_weights, ids, paddings, x2_zero],
          version=self._xla_module_info['version'],
          module=self._xla_module_info['module_bytes'],
          Sout=self._xla_module_info['sout'],
          Tout=self._xla_module_info['tout'],
          platforms=[self._xla_platform],
      )
    return np.asarray(outputs[0].numpy(), dtype=np.float32)

  def _build_embedded_prompt(
      self,
      *,
      prompt: str,
      normalized_embedding: np.ndarray,
      full_tokens: np.ndarray,
      norm: float,
  ) -> EmbeddedPrompt:
    runtime_tokens = full_tokens[: self.config.runtime_style_token_depth].copy()
    embedding_id = _sha_prefix(
        'emb', normalized_embedding.astype(np.float32).tobytes()
    )
    return EmbeddedPrompt(
        embedding_id=embedding_id,
        prompt=prompt,
        embedding=normalized_embedding,
        full_tokens=full_tokens,
        runtime_tokens=runtime_tokens,
        norm=norm,
        backend_name=self.config.backend_name,
        device=self.config.device,
    )

  def _quantize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
    return rvq_quantization(embeddings, self._rvq_codebooks)[0]

  def embed_prompt(self, prompt: str) -> EmbeddedPrompt:
    return self.embed_prompts([prompt])[0]

  def embed_prompts(self, prompts: list[str]) -> list[EmbeddedPrompt]:
    cleaned = []
    for prompt in prompts:
      value = str(prompt).strip()
      if not value:
        raise ValueError('Prompt text must be non-empty.')
      cleaned.append(value)
    if not cleaned:
      return []

    if self.config.backend_name == 'xla_exact':
      embeddings = self._embed_batch_raw_exact_xla(cleaned)
      norms = np.linalg.norm(embeddings, axis=1).astype(np.float32)
      if np.any(norms <= 0.0):
        raise ValueError('Embedding norm must be > 0.')
      normalized_embeddings = embeddings / norms[:, np.newaxis]
      full_tokens_batch = self._quantize_embeddings(normalized_embeddings)
      return [
          self._build_embedded_prompt(
              prompt=prompt,
              normalized_embedding=normalized_embedding,
              full_tokens=full_tokens,
              norm=float(norm),
          )
          for prompt, normalized_embedding, full_tokens, norm in zip(
              cleaned,
              normalized_embeddings,
              full_tokens_batch,
              norms,
              strict=True,
          )
      ]

    return [
        self._build_embedded_prompt_from_prompt(prompt)
        for prompt in cleaned
    ]

  def _build_embedded_prompt_from_prompt(self, prompt: str) -> EmbeddedPrompt:
    embedding = self._embed_prompt_raw(prompt)
    normalized, norm = _normalize_embedding(embedding)
    full_tokens = rvq_quantization(
        normalized.reshape(1, -1),
        self._rvq_codebooks,
    )[0][0]
    return self._build_embedded_prompt(
        prompt=prompt,
        normalized_embedding=normalized,
        full_tokens=full_tokens,
        norm=norm,
    )

  def backend_metadata(self) -> dict[str, Any]:
    supports_true_batching = self.config.backend_name == 'xla_exact'
    notes = []
    if self.config.backend_name == 'xla_exact':
      notes.extend(
          [
              'Direct XlaCallModule execution of the legacy MusicCoCa text graph.',
              f'Exact StableHLO module retargeted to {self._xla_platform}.',
              'Weights are materialized once and reused for batched prompt calls.',
          ]
      )
      if self.config.disable_tf32:
        notes.append('TF32 disabled for stricter float32 fidelity.')
      else:
        notes.append('TF32 enabled for faster GPU matmul throughput.')
      if self._xla_module_info is not None:
        notes.append(
            'Embedded module originally declared platforms='
            + ','.join(self._xla_module_info['platforms'])
        )
      if self._device_fallback_reason is not None:
        notes.append(self._device_fallback_reason)
    else:
      notes.extend(
          [
              'Current MusicCoCa exports are fixed to batch size 1.',
              'The shipped SavedModels contain CPU-only XLA modules, so GPU '
              'attachment alone does not accelerate prompt embedding.',
          ]
      )
    return {
        'backend_name': self.config.backend_name,
        'device': self.config.device,
        'requested_device': self._requested_device,
        'model_path': self._model_path,
        'runtime_style_token_depth': self.config.runtime_style_token_depth,
        'supports_true_batching': supports_true_batching,
        'notes': notes,
    }


def init_worker(config_dict: dict[str, Any]) -> None:
  global _WORKER_BACKEND
  _WORKER_BACKEND = MusicCoCaBackend(BackendConfig(**config_dict))


def embed_prompt_worker(prompt: str) -> dict[str, Any]:
  if _WORKER_BACKEND is None:
    raise RuntimeError('Worker backend has not been initialized.')
  return _WORKER_BACKEND.embed_prompt(prompt).to_wire()

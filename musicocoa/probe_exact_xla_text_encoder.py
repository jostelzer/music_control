#!/usr/bin/env python3

"""Probe exact MusicCoCa text StableHLO execution via xla_call_module."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections.abc import Sequence

import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.ops import gen_xla_ops

if __package__ in (None, ''):
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
  from musicocoa.prompt_space import generate_unique_prompts
else:
  from .prompt_space import generate_unique_prompts


MUSICCOCA_RVQ_VAR_ORDER = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3]


def load_saved_model(path: str):
  return tf.saved_model.load(path)


def extract_embed_text_module(concrete_fn) -> dict[str, object]:
  graph_def = concrete_fn.graph.as_graph_def()
  function_defs = {f.signature.name: f for f in graph_def.library.function}

  wrapper_node = None
  for node in graph_def.node:
    if node.op == 'StatefulPartitionedCall':
      wrapper_node = node
      break
  if wrapper_node is None:
    raise RuntimeError('Could not find top-level StatefulPartitionedCall.')

  wrapper_name = wrapper_node.attr['f'].func.name
  wrapper_fn = function_defs[wrapper_name]

  xla_call_name = None
  preprocess_name = None
  for node in wrapper_fn.node_def:
    if node.name == 'StatefulPartitionedCall':
      preprocess_name = node.attr['f'].func.name
    elif node.name == 'StatefulPartitionedCall_1':
      xla_call_name = node.attr['f'].func.name
  if xla_call_name is None or preprocess_name is None:
    raise RuntimeError('Could not resolve wrapper subfunctions.')

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
  sout = [[dim.size for dim in shape.dim] for shape in attrs['Sout'].list.shape]
  tout = list(attrs['Tout'].list.type)
  platforms = [value.decode('utf-8', errors='replace') for value in attrs['platforms'].list.s]

  preprocess_uses_random = any(node.op == 'RandomUniformInt' for node in preprocess_fn.node_def)
  return {
      'module_bytes': bytes(attrs['module'].s),
      'version': int(attrs['version'].i),
      'sout': sout,
      'tout': tout,
      'platforms': platforms,
      'preprocess_name': preprocess_name,
      'xla_function_name': xla_call_name,
      'preprocess_uses_random': preprocess_uses_random,
  }


def make_ids_paddings(
    *,
    prompts: Sequence[str],
    vocab,
    max_text_length: int = 128,
    target_sos_id: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
  ids_rows = []
  paddings_rows = []
  for prompt in prompts:
    labels = vocab.EncodeAsIds(prompt.lower())
    num_tokens = min(len(labels), max_text_length - 1)
    labels = labels[: max_text_length - 1]
    ids = [target_sos_id] + labels
    ids += [0] * (max_text_length - len(ids))
    paddings = np.ones((max_text_length,), dtype=np.float32)
    paddings[: num_tokens + 1] = 0.0
    ids_rows.append(ids)
    paddings_rows.append(paddings)
  return np.asarray(ids_rows, dtype=np.int32), np.asarray(paddings_rows, dtype=np.float32)


def call_xla_module(
    *,
    weights: Sequence[tf.Tensor],
    ids: tf.Tensor,
    paddings: tf.Tensor,
    module_bytes: bytes,
    version: int,
    sout: Sequence[Sequence[int]],
    tout: Sequence[int],
    platforms: Sequence[str],
    x2: tf.Tensor | None,
    device: str,
):
  args = list(weights) + [ids, paddings]
  if x2 is not None:
    args.append(x2)
  with tf.device(device):
    outputs = gen_xla_ops.xla_call_module(
        args=args,
        version=version,
        module=module_bytes,
        Sout=list(sout),
        Tout=list(tout),
        platforms=list(platforms),
    )
  return outputs


def make_constant_weights(weight_arrays: Sequence[np.ndarray], device: str) -> list[tf.Tensor]:
  with tf.device(device):
    return [tf.constant(array) for array in weight_arrays]


def make_constant_tensor(array: np.ndarray, device: str, dtype=None) -> tf.Tensor:
  with tf.device(device):
    return tf.constant(array, dtype=dtype)


def tensor_list_to_numpy(values) -> list[np.ndarray]:
  if isinstance(values, (tuple, list)):
    return [np.asarray(value.numpy()) for value in values]
  return [np.asarray(values.numpy())]


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
  return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def benchmark(fn, repeats: int) -> float:
  timings = []
  for _ in range(repeats):
    started = time.perf_counter()
    _ = fn()
    timings.append((time.perf_counter() - started) * 1000.0)
  return float(np.mean(timings))


def load_rvq_codebooks(path: str) -> np.ndarray:
  var_path = f'{path}/variables/variables'
  result = np.zeros((12, 1024, 768), dtype=np.float32)
  for depth, variable_name in enumerate(MUSICCOCA_RVQ_VAR_ORDER):
    variable = tf.train.load_variable(
        var_path, f'variables/{variable_name}/.ATTRIBUTES/VARIABLE_VALUE'
    )
    result[depth] = variable.T
  return result


def rvq_quantization(
    embeddings: np.ndarray,
    codebooks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--saved-model-dir', required=True)
  parser.add_argument('--vocab-model', required=True)
  parser.add_argument('--batch-size', type=int, default=32)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--cpu-device', default='/CPU:0')
  parser.add_argument('--gpu-device', default='/GPU:0')
  parser.add_argument('--repeats', type=int, default=20)
  parser.add_argument('--gpu-platform', default='CUDA')
  parser.add_argument('--skip-legacy-baseline', action='store_true')
  parser.add_argument('--skip-cpu-benchmark', action='store_true')
  parser.add_argument('--rvq-codebooks-dir')
  parser.add_argument('--disable-tf32', action='store_true')
  args = parser.parse_args()

  if args.disable_tf32:
    tf.config.experimental.enable_tensor_float_32_execution(False)

  from sentencepiece import SentencePieceProcessor

  saved_model = load_saved_model(args.saved_model_dir)
  concrete_fn = saved_model.signatures['embed_text']
  module_info = extract_embed_text_module(concrete_fn)

  vocab = SentencePieceProcessor()
  vocab.Load(args.vocab_model)
  prompts = generate_unique_prompts(args.batch_size, args.seed)
  ids_np, paddings_np = make_ids_paddings(prompts=prompts, vocab=vocab)

  with tf.device(args.cpu_device):
    weight_arrays = [
        np.asarray(tf.raw_ops.ReadVariableOp(resource=resource, dtype=tf.float32).numpy())
        for resource in concrete_fn.captured_inputs
    ]

  cpu_weights = make_constant_weights(weight_arrays, args.cpu_device)
  cpu_ids = make_constant_tensor(ids_np, args.cpu_device, dtype=tf.int32)
  cpu_paddings = make_constant_tensor(paddings_np, args.cpu_device, dtype=tf.float32)
  cpu_x2_zero = make_constant_tensor(np.asarray(0, dtype=np.int32), args.cpu_device)
  cpu_x2_one = make_constant_tensor(np.asarray(1, dtype=np.int32), args.cpu_device)

  baseline_np = None
  if not args.skip_legacy_baseline:
    def call_legacy_cpu():
      with tf.device(args.cpu_device):
        return concrete_fn(inputs_0=cpu_ids, inputs_0_1=cpu_paddings)

    baseline_outputs = call_legacy_cpu()
    baseline_np = {
        key: np.asarray(value.numpy()) for key, value in baseline_outputs.items()
    }

  cpu_outputs_zero = call_xla_module(
      weights=cpu_weights,
      ids=cpu_ids,
      paddings=cpu_paddings,
      x2=cpu_x2_zero,
      module_bytes=module_info['module_bytes'],
      version=module_info['version'],
      sout=module_info['sout'],
      tout=module_info['tout'],
      platforms=['CPU'],
      device=args.cpu_device,
  )
  cpu_outputs_one = call_xla_module(
      weights=cpu_weights,
      ids=cpu_ids,
      paddings=cpu_paddings,
      x2=cpu_x2_one,
      module_bytes=module_info['module_bytes'],
      version=module_info['version'],
      sout=module_info['sout'],
      tout=module_info['tout'],
      platforms=['CPU'],
      device=args.cpu_device,
  )
  cpu_zero_np = tensor_list_to_numpy(cpu_outputs_zero)
  cpu_one_np = tensor_list_to_numpy(cpu_outputs_one)

  result = {
      'module_platforms': module_info['platforms'],
      'preprocess_name': module_info['preprocess_name'],
      'xla_function_name': module_info['xla_function_name'],
      'preprocess_uses_random': module_info['preprocess_uses_random'],
      'batch_size': args.batch_size,
      'cpu_wrapper_zero_vs_one': {
          'contrastive_txt_embed_max_abs_diff': max_abs_diff(cpu_zero_np[0], cpu_one_np[0]),
          'contrastive_txt_embed_l2_normalized_max_abs_diff': max_abs_diff(cpu_zero_np[1], cpu_one_np[1]),
      },
  }
  if baseline_np is not None:
    result['cpu_wrapper_zero_vs_baseline'] = {
        'contrastive_txt_embed_max_abs_diff': max_abs_diff(
            cpu_zero_np[0], baseline_np['contrastive_txt_embed']
        ),
        'contrastive_txt_embed_l2_normalized_max_abs_diff': max_abs_diff(
            cpu_zero_np[1], baseline_np['contrastive_txt_embed_l2_normalized']
        ),
    }

  if not args.skip_cpu_benchmark:
    cpu_benchmark_ms = {
        'wrapped_xla_call_module': benchmark(
            lambda: call_xla_module(
                weights=cpu_weights,
                ids=cpu_ids,
                paddings=cpu_paddings,
                x2=cpu_x2_zero,
                module_bytes=module_info['module_bytes'],
                version=module_info['version'],
                sout=module_info['sout'],
                tout=module_info['tout'],
                platforms=['CPU'],
                device=args.cpu_device,
            ),
            args.repeats,
        ),
    }
    if baseline_np is not None:
      cpu_benchmark_ms['legacy_embed_text'] = benchmark(
          call_legacy_cpu,
          args.repeats,
      )
    result['cpu_benchmark_ms'] = cpu_benchmark_ms

  try:
    gpu_weights = make_constant_weights(weight_arrays, args.gpu_device)
    gpu_ids = make_constant_tensor(ids_np, args.gpu_device, dtype=tf.int32)
    gpu_paddings = make_constant_tensor(paddings_np, args.gpu_device, dtype=tf.float32)
    gpu_x2_zero = make_constant_tensor(np.asarray(0, dtype=np.int32), args.gpu_device)
    gpu_outputs = call_xla_module(
        weights=gpu_weights,
        ids=gpu_ids,
        paddings=gpu_paddings,
        x2=gpu_x2_zero,
        module_bytes=module_info['module_bytes'],
        version=module_info['version'],
        sout=module_info['sout'],
        tout=module_info['tout'],
        platforms=[args.gpu_platform],
        device=args.gpu_device,
    )
    gpu_np = tensor_list_to_numpy(gpu_outputs)
    result['gpu_status'] = {
        'ok': True,
        'contrastive_txt_embed_max_abs_diff_vs_cpu_wrapper': max_abs_diff(
            gpu_np[0], cpu_zero_np[0]
        ),
        'contrastive_txt_embed_l2_normalized_max_abs_diff_vs_cpu_wrapper': max_abs_diff(
            gpu_np[1], cpu_zero_np[1]
        ),
        'wrapped_xla_call_module_gpu_ms': benchmark(
            lambda: call_xla_module(
                weights=gpu_weights,
                ids=gpu_ids,
                paddings=gpu_paddings,
                x2=gpu_x2_zero,
                module_bytes=module_info['module_bytes'],
                version=module_info['version'],
                sout=module_info['sout'],
                tout=module_info['tout'],
                platforms=[args.gpu_platform],
                device=args.gpu_device,
            ),
            args.repeats,
        ),
    }
    if args.rvq_codebooks_dir:
      codebooks = load_rvq_codebooks(args.rvq_codebooks_dir)
      cpu_tokens, _ = rvq_quantization(cpu_zero_np[1].astype(np.float32), codebooks)
      gpu_tokens, _ = rvq_quantization(gpu_np[1].astype(np.float32), codebooks)
      full_equal = cpu_tokens == gpu_tokens
      runtime_equal = full_equal[:, :6]
      result['gpu_status']['token_agreement'] = {
          'full_slot_accuracy': float(np.mean(full_equal)),
          'runtime_slot_accuracy': float(np.mean(runtime_equal)),
          'full_exact_prompt_match_rate': float(np.mean(np.all(full_equal, axis=1))),
          'runtime_exact_prompt_match_rate': float(
              np.mean(np.all(runtime_equal, axis=1))
          ),
          'mismatched_prompts': int(np.sum(~np.all(runtime_equal, axis=1))),
      }
  except Exception as exc:  # pylint: disable=broad-except
    result['gpu_status'] = {
        'ok': False,
        'error_type': type(exc).__name__,
        'error': str(exc),
    }

  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

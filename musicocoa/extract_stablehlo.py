#!/usr/bin/env python3

"""Extracts embedded StableHLO bytecode from a SavedModel XlaCallModule op."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

import tensorflow as tf


def extract_xla_modules(saved_model_dir: str) -> list[dict[str, object]]:
  model = tf.saved_model.load(saved_model_dir)
  results: list[dict[str, object]] = []
  for signature_name, fn in model.signatures.items():
    graph_def = fn.graph.as_graph_def()
    for function_def in graph_def.library.function:
      for node_def in function_def.node_def:
        if node_def.op != 'XlaCallModule':
          continue
        attrs = node_def.attr
        platforms = list(attrs['platforms'].list.s) if 'platforms' in attrs else []
        tin = list(attrs['Tin'].list.type) if 'Tin' in attrs else []
        tout = list(attrs['Tout'].list.type) if 'Tout' in attrs else []
        sout = []
        if 'Sout' in attrs:
          for shape in attrs['Sout'].list.shape:
            sout.append([dim.size for dim in shape.dim])
        results.append(
            {
                'signature_name': signature_name,
                'function_name': function_def.signature.name,
                'node_name': node_def.name,
                'platforms': [value.decode('utf-8', errors='replace') for value in platforms],
                'version': int(attrs['version'].i) if 'version' in attrs else None,
                'module_bytes': bytes(attrs['module'].s) if 'module' in attrs else b'',
                'tin_count': len(tin),
                'tout_count': len(tout),
                'sout': sout,
            }
        )
  return results


def write_outputs(modules: list[dict[str, object]], output_dir: pathlib.Path) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  metadata = []
  for index, module_info in enumerate(modules):
    bytecode = module_info.pop('module_bytes')
    bytecode_path = output_dir / f'module_{index:02d}.stablehlo.bc'
    bytecode_path.write_bytes(bytecode)
    module_info['bytecode_path'] = str(bytecode_path)
    metadata.append(module_info)
  (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--saved-model-dir', required=True)
  parser.add_argument('--output-dir', required=True)
  args = parser.parse_args()

  modules = extract_xla_modules(args.saved_model_dir)
  if not modules:
    raise SystemExit('No XlaCallModule nodes found.')
  write_outputs(modules, pathlib.Path(args.output_dir))
  print(json.dumps({'module_count': len(modules), 'output_dir': args.output_dir}, indent=2))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  args=(
    --host "${MUSICOCOA_HOST:-0.0.0.0}"
    --port "${MUSICOCOA_PORT:-8773}"
    --backend "${MUSICOCOA_BACKEND:-xla_exact}"
    --device "${MUSICOCOA_DEVICE:-/GPU:0}"
    --workers "${MUSICOCOA_WORKERS:-0}"
    --runtime-style-token-depth "${MUSICOCOA_RUNTIME_STYLE_TOKEN_DEPTH:-6}"
  )
  if [ "${MUSICOCOA_DISABLE_TF32:-0}" = "1" ]; then
    args+=(--disable-tf32)
  fi
  set -- "${args[@]}"
fi

exec python /opt/music_control/musicocoa/server.py "$@"

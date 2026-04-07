#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec python "$SCRIPT_DIR/server.py" \
  --backend xla_exact \
  --device "${MUSICOCOA_DEVICE:-/GPU:0}" \
  --workers 0 \
  "$@"

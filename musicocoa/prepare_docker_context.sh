#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  musicocoa/prepare_docker_context.sh <output_dir>

Description:
  Creates a compact Docker build context containing:
    - music_control/musicocoa
    - magenta-realtime/magenta_rt

  The sibling `magenta-realtime` repo is resolved from:
    1. $MAGENTA_REALTIME_REPO
    2. <music_control parent>/magenta-realtime
EOF
}

if [ "$#" -ne 1 ]; then
  usage >&2
  exit 2
fi

OUT_DIR="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PARENT_ROOT="$(cd -- "$REPO_ROOT/.." && pwd)"
MAGENTA_RT_ROOT="${MAGENTA_REALTIME_REPO:-$PARENT_ROOT/magenta-realtime}"

if [ ! -d "$MAGENTA_RT_ROOT/magenta_rt" ]; then
  echo "error: could not find sibling magenta-realtime repo at $MAGENTA_RT_ROOT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR/music_control" "$OUT_DIR/magenta-realtime"

rsync -a --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "$REPO_ROOT/musicocoa/" "$OUT_DIR/music_control/musicocoa/"

rsync -a --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "$MAGENTA_RT_ROOT/magenta_rt/" "$OUT_DIR/magenta-realtime/magenta_rt/"

echo "$OUT_DIR"

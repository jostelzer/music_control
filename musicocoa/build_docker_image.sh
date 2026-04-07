#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="musicocoa:xla-exact"
BASE_IMAGE="nvcr.io/nvidia/tensorflow:25.02-tf2-py3"

usage() {
  cat <<'EOF'
Usage:
  musicocoa/build_docker_image.sh [--tag <image_tag>] [--base-image <image>]

Description:
  Builds the repo-local MusicCoCa Docker image from a compact temporary context.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag)
      IMAGE_TAG="${2:-}"
      shift 2
      ;;
    --base-image)
      BASE_IMAGE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

"$SCRIPT_DIR/prepare_docker_context.sh" "$TMP_DIR" >/dev/null

docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -f "$TMP_DIR/music_control/musicocoa/Dockerfile" \
  -t "$IMAGE_TAG" \
  "$TMP_DIR"

echo "Built image: $IMAGE_TAG"
echo "Base image: $BASE_IMAGE"

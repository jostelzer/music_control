#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="musicocoa:xla-exact"
BASE_IMAGE="nvcr.io/nvidia/tensorflow:25.02-tf2-py3"
CONTAINER_NAME="musicocoa-xla-exact"
HOST_PORT="8773"
GPU_SPEC="all"
REBUILD=0
STRICT=0
CPU_MODE=0

usage() {
  cat <<'EOF'
Usage:
  musicocoa/run_docker_server.sh [options]

Options:
  --rebuild                 Rebuild the image before running.
  --strict                  Disable TF32 for stricter fidelity.
  --cpu                     Run on CPU instead of GPU.
  --base-image <image>      Docker base image. Default: nvcr.io/nvidia/tensorflow:25.02-tf2-py3
  --tag <image_tag>         Docker image tag. Default: musicocoa:xla-exact
  --name <container_name>   Container name. Default: musicocoa-xla-exact
  --host-port <port>        Host port to bind to container port 8773. Default: 8773
  --gpu <spec>              Value for `--gpus`. Default: all
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --rebuild)
      REBUILD=1
      shift
      ;;
    --strict)
      STRICT=1
      shift
      ;;
    --cpu)
      CPU_MODE=1
      shift
      ;;
    --base-image)
      BASE_IMAGE="${2:-}"
      shift 2
      ;;
    --tag)
      IMAGE_TAG="${2:-}"
      shift 2
      ;;
    --name)
      CONTAINER_NAME="${2:-}"
      shift 2
      ;;
    --host-port)
      HOST_PORT="${2:-}"
      shift 2
      ;;
    --gpu)
      GPU_SPEC="${2:-}"
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

if [ "$REBUILD" -eq 1 ] || ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  "$SCRIPT_DIR/build_docker_image.sh" --tag "$IMAGE_TAG" --base-image "$BASE_IMAGE"
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker_args=(
  run -d --restart unless-stopped
  --name "$CONTAINER_NAME"
  -p "127.0.0.1:${HOST_PORT}:8773"
  -e MUSICOCOA_PORT=8773
  -e MUSICOCOA_BACKEND=xla_exact
  -e MUSICOCOA_WORKERS=0
)

if [ "$STRICT" -eq 1 ]; then
  docker_args+=(-e MUSICOCOA_DISABLE_TF32=1)
fi

if [ "$CPU_MODE" -eq 1 ]; then
  docker_args+=(-e MUSICOCOA_DEVICE=/CPU:0)
else
  docker_args+=(--gpus "$GPU_SPEC" -e MUSICOCOA_DEVICE=/GPU:0)
fi

docker_args+=("$IMAGE_TAG")

CONTAINER_ID="$(docker "${docker_args[@]}")"
echo "Started container: $CONTAINER_ID"

python - <<PY
import json
import time
import urllib.request

health_url = "http://127.0.0.1:${HOST_PORT}/health"
deadline = time.time() + 120
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(health_url, timeout=2) as response:
            payload = json.load(response)
        print(json.dumps(payload, indent=2, sort_keys=True))
        break
    except Exception as exc:  # pylint: disable=broad-except
        last_error = exc
        time.sleep(1)
else:
    raise SystemExit(f"health check failed for {health_url}: {last_error}")

embed_body = json.dumps(
    {
        "text": "minimal techno, rubbery bassline",
        "include_full_tokens": True,
    }
).encode("utf-8")
request = urllib.request.Request(
    "http://127.0.0.1:${HOST_PORT}/embed",
    data=embed_body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=120) as response:
    payload = json.load(response)

print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo
echo "Tunnel or call locally via: http://127.0.0.1:${HOST_PORT}"

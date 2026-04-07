#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  cat <<'EOF' >&2
Usage:
  musicocoa/spool_remote_docker.sh <ssh_host> [options]

Options:
  --strict                  Disable TF32 for stricter fidelity.
  --cpu                     Run on CPU instead of GPU.
  --base-image <image>      Docker base image. Default: nvcr.io/nvidia/tensorflow:25.02-tf2-py3
  --tag <image_tag>         Docker image tag. Default: musicocoa:xla-exact
  --name <container_name>   Container name. Default: musicocoa-xla-exact
  --host-port <port>        Remote host port to bind to container port 8773. Default: 8773
  --gpu <spec>              Value for `--gpus`. Default: all
EOF
  exit 2
fi

SSH_HOST="$1"
shift

IMAGE_TAG="musicocoa:xla-exact"
BASE_IMAGE="nvcr.io/nvidia/tensorflow:25.02-tf2-py3"
CONTAINER_NAME="musicocoa-xla-exact"
HOST_PORT="8773"
GPU_SPEC="all"
STRICT=0
CPU_MODE=0

while [ "$#" -gt 0 ]; do
  case "$1" in
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
    *)
      echo "error: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="$(mktemp -d)"
REMOTE_TMP="/tmp/musicocoa-docker-context-${USER}-$$"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

"$SCRIPT_DIR/prepare_docker_context.sh" "$TMP_DIR" >/dev/null

COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 tar -C "$TMP_DIR" -cf - . | ssh "$SSH_HOST" "rm -rf '$REMOTE_TMP' && mkdir -p '$REMOTE_TMP' && tar -C '$REMOTE_TMP' -xf -"

RUN_FLAGS=(
  docker run -d --restart unless-stopped
  --name "$CONTAINER_NAME"
  -p "127.0.0.1:${HOST_PORT}:8773"
  -e MUSICOCOA_PORT=8773
  -e MUSICOCOA_BACKEND=xla_exact
  -e MUSICOCOA_WORKERS=0
)

if [ "$STRICT" -eq 1 ]; then
  RUN_FLAGS+=(-e MUSICOCOA_DISABLE_TF32=1)
fi

if [ "$CPU_MODE" -eq 1 ]; then
  RUN_FLAGS+=(-e MUSICOCOA_DEVICE=/CPU:0)
else
  RUN_FLAGS+=(--gpus "$GPU_SPEC" -e MUSICOCOA_DEVICE=/GPU:0)
fi

RUN_FLAGS+=("$IMAGE_TAG")

remote_command=$(cat <<EOF
set -euo pipefail
docker build --build-arg 'BASE_IMAGE=$BASE_IMAGE' -f '$REMOTE_TMP/music_control/musicocoa/Dockerfile' -t '$IMAGE_TAG' '$REMOTE_TMP'
docker rm -f '$CONTAINER_NAME' >/dev/null 2>&1 || true
$(printf '%q ' "${RUN_FLAGS[@]}")
python3 - <<'PY'
import json
import time
import urllib.request

health_url = 'http://127.0.0.1:${HOST_PORT}/health'
deadline = time.time() + 120
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(health_url, timeout=2) as response:
            payload = json.load(response)
        print(json.dumps(payload, indent=2, sort_keys=True))
        break
    except Exception as exc:
        last_error = exc
        time.sleep(1)
else:
    raise SystemExit(f'health check failed for {health_url}: {last_error}')

embed_body = json.dumps(
    {
        'text': 'minimal techno, rubbery bassline',
        'include_full_tokens': True,
    }
).encode('utf-8')
request = urllib.request.Request(
    'http://127.0.0.1:${HOST_PORT}/embed',
    data=embed_body,
    headers={'Content-Type': 'application/json'},
    method='POST',
)
with urllib.request.urlopen(request, timeout=120) as response:
    payload = json.load(response)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
rm -rf '$REMOTE_TMP'
EOF
)

ssh "$SSH_HOST" "$remote_command"

echo
echo "Remote service ready on $SSH_HOST:127.0.0.1:${HOST_PORT}"
echo "Tunnel from this Mac with:"
echo "ssh -L ${HOST_PORT}:127.0.0.1:${HOST_PORT} ${SSH_HOST}"

#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="musicocoa-xla-exact"
SSH_HOST=""

usage() {
  cat <<'EOF'
Usage:
  musicocoa/stop_docker_server.sh [--name <container_name>] [--host <ssh_host>]

Examples:
  bash musicocoa/stop_docker_server.sh
  bash musicocoa/stop_docker_server.sh --host ias@iki --name musicocoa-xla-exact-vdd
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --name)
      CONTAINER_NAME="${2:-}"
      shift 2
      ;;
    --host)
      SSH_HOST="${2:-}"
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

COMMAND="docker rm -f '$CONTAINER_NAME' >/dev/null 2>&1 || true"

if [ -n "$SSH_HOST" ]; then
  ssh "$SSH_HOST" "$COMMAND"
  echo "Stopped remote container '$CONTAINER_NAME' on $SSH_HOST"
else
  eval "$COMMAND"
  echo "Stopped local container '$CONTAINER_NAME'"
fi

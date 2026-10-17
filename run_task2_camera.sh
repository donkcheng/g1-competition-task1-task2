#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/drgh/g1_official/GR00T-WholeBodyControl/.venv_teleop/bin/python
SERVER="$SCRIPT_DIR/src/videohub_camera_server.py"
UNIT=g1-task2-camera.service
WRIST_HOST="${WRIST_CAMERA_HOST:-192.168.123.164}"
WRIST_PORT="${WRIST_CAMERA_PORT:-5559}"

is_running() {
  systemctl --user is-active --quiet "$UNIT"
}

case "${1:-status}" in
  start)
    if is_running; then
      echo "Task2 camera bridge already running"
      exit 0
    fi
    systemd-run --user --unit="$UNIT" --collect \
      --property=Type=simple \
      "$PYTHON" -u "$SERVER" --interface enp8s0 --bind 192.168.123.222 --port 5555 \
      --wrist-host "$WRIST_HOST" --wrist-port "$WRIST_PORT" --require-wrist
    for _ in {1..20}; do
      is_running && break
      sleep 0.1
    done
    if ! is_running; then
      echo "Camera bridge exited during startup; run: $0 logs" >&2
      exit 1
    fi
    echo "Task2 camera bridge started at 192.168.123.222:5555"
    ;;
  foreground)
    exec "$PYTHON" -u "$SERVER" --interface enp8s0 --bind 192.168.123.222 --port 5555 \
      --wrist-host "$WRIST_HOST" --wrist-port "$WRIST_PORT" --require-wrist
    ;;
  status)
    if is_running; then
      systemctl --user --no-pager --full status "$UNIT"
    else
      echo "STOPPED"
      exit 1
    fi
    ;;
  stop)
    if is_running; then
      systemctl --user stop "$UNIT"
      echo "Task2 camera bridge stopped"
    else
      echo "Already stopped"
    fi
    ;;
  logs)
    journalctl --user -u "$UNIT" -n 100 --no-pager
    ;;
  *)
    echo "Usage: $0 {start|foreground|status|stop|logs}" >&2
    exit 2
    ;;
esac

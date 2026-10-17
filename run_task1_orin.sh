#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROUTE="${1:-}"
CONFIRM="${2:-}"

if [[ -z "$ROUTE" || "$CONFIRM" != "RUN_TASK1" ]]; then
  echo "用法：$0 routes/路线.json RUN_TASK1" >&2
  exit 2
fi

export TASK1_IFACE="eth0"
exec "$SCRIPT_DIR/run_task1_scene_replay.sh" "$ROUTE" RUN_TASK1

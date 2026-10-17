#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${1:-}"
CONFIRM="${2:-}"

if [[ -z "$OUTPUT" || "$CONFIRM" != "SET_START_REFERENCE" ]]; then
  echo "用法：$0 routes/新路线.json SET_START_REFERENCE" >&2
  exit 2
fi

export TASK1_IFACE="eth0"
exec "$SCRIPT_DIR/run_task1_record_scene_aligned.sh" "$OUTPUT" SET_START_REFERENCE

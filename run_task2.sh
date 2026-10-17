#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "collect-act" ]]; then
  shift
  exec bash "$SCRIPT_DIR/run_task2_collect_continuous.sh" "$@"
fi
exec python3 "$SCRIPT_DIR/src/pico_collection.py" --env "$SCRIPT_DIR/.env" "$@"

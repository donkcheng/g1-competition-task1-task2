#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TASK1_PYTHON="/home/drgh/g1_official/GR00T-WholeBodyControl/.venv_teleop/bin/python"
if [[ -n "${TASK1_PYTHON:-}" ]]; then
  TASK1_PYTHON_BIN="$TASK1_PYTHON"
elif [[ -x "$DEFAULT_TASK1_PYTHON" ]]; then
  TASK1_PYTHON_BIN="$DEFAULT_TASK1_PYTHON"
else
  TASK1_PYTHON_BIN="python3"
fi
exec "$TASK1_PYTHON_BIN" "$SCRIPT_DIR/src/task1_route.py" --config "$SCRIPT_DIR/config/task1.json" "$@"

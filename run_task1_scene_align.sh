#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON="/home/drgh/g1_official/GR00T-WholeBodyControl/.venv_teleop/bin/python"
if [[ -n "${TASK1_PYTHON:-}" ]]; then
  PYTHON_BIN="$TASK1_PYTHON"
elif [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON_BIN="$DEFAULT_PYTHON"
else
  PYTHON_BIN="python3"
fi
exec "$PYTHON_BIN" "$SCRIPT_DIR/src/task1_scene_alignment.py" \
  --config "$SCRIPT_DIR/config/task1_start_alignment.json" \
  --reference "$SCRIPT_DIR/config/task1_start_scene.jpg" \
  --debug-output "$SCRIPT_DIR/config/task1_start_scene_current.jpg" "$@"

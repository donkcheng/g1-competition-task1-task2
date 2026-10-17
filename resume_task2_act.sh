#!/usr/bin/env bash
# Resume an interrupted LeRobot ACT run from a checkpoint train_config.json.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ENV="$PROJECT_DIR/config/task2_act_train.env"
if [[ -f "$TRAIN_ENV" ]]; then
    # shellcheck disable=SC1090
    source "$TRAIN_ENV"
fi

CONFIG_PATH="${1:-}"
if [[ -z "$CONFIG_PATH" || ! -f "$CONFIG_PATH" ]]; then
    echo "Usage: $0 /path/to/checkpoint/pretrained_model/train_config.json" >&2
    exit 2
fi

WBC_ROOT="${TASK2_WBC_ROOT:-/home/drgh/g1_official/GR00T-WholeBodyControl}"
OUTPUT_ROOT="${TASK2_OUTPUT_ROOT:-$PROJECT_DIR/outputs/task2_act}"
CACHE_ROOT="${TASK2_CACHE_ROOT:-$OUTPUT_ROOT/.cache}"
PYTHON="${TASK2_PYTHON:-$WBC_ROOT/.venv_data_collection/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Training Python not found: $PYTHON" >&2; exit 1; }

mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" "$CACHE_ROOT/xdg"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

cd "$WBC_ROOT"
exec "$PYTHON" -m lerobot.scripts.train \
    --config_path="$(realpath "$CONFIG_PATH")" \
    --resume=true

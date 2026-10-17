#!/usr/bin/env bash
# Train one of the three independent task-2 ACT policies on a CUDA server.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ENV="$PROJECT_DIR/config/task2_act_train.env"
if [[ -f "$TRAIN_ENV" ]]; then
    # shellcheck disable=SC1090
    source "$TRAIN_ENV"
fi

skill="${1:-}"
if [[ "$skill" != button && "$skill" != knob && "$skill" != switch ]]; then
    echo "Usage: $0 {button|knob|switch}" >&2
    exit 2
fi

WBC_ROOT="${TASK2_WBC_ROOT:-/home/drgh/g1_official/GR00T-WholeBodyControl}"
DATA_ROOT="${TASK2_DATA_ROOT:-$PROJECT_DIR/data}"
OUTPUT_ROOT="${TASK2_OUTPUT_ROOT:-$PROJECT_DIR/outputs/task2_act}"
CACHE_ROOT="${TASK2_CACHE_ROOT:-$OUTPUT_ROOT/.cache}"
PYTHON="${TASK2_PYTHON:-$WBC_ROOT/.venv_data_collection/bin/python}"
DATASET_NAME="task2_${skill}_act"
DATASET_DIR="$DATA_ROOT/$DATASET_NAME"
STEPS="${TASK2_TRAIN_STEPS:-50000}"
SAVE_FREQ="${TASK2_SAVE_FREQ:-5000}"
BATCH_SIZE="${TASK2_BATCH_SIZE:-8}"
NUM_WORKERS="${TASK2_NUM_WORKERS:-8}"
LOG_FREQ="${TASK2_LOG_FREQ:-100}"
CHUNK_SIZE="${TASK2_CHUNK_SIZE:-100}"
# FP16 AMP produced non-finite ACT gradients with the pinned LeRobot/Torch
# stack on L40S.  Full precision is stable and still fits comfortably in the
# 48 GB device, so keep AMP opt-in rather than risking skipped optimizer steps.
USE_AMP="${TASK2_USE_AMP:-false}"
IMAGE_AUGMENTATION="${TASK2_IMAGE_AUGMENTATION:-true}"
RUN_ID="${TASK2_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="$OUTPUT_ROOT/${DATASET_NAME}_${RUN_ID}"

[[ -x "$PYTHON" ]] || { echo "Training Python not found: $PYTHON" >&2; exit 1; }
[[ -d "$DATASET_DIR" ]] || { echo "Dataset not found: $DATASET_DIR" >&2; exit 1; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Output already exists: $OUTPUT_DIR" >&2; exit 1; }

mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" "$CACHE_ROOT/xdg"
export TASK2_CACHE_ROOT="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

"$PYTHON" "$PROJECT_DIR/tools/verify_task2_act_dataset.py" "$DATASET_DIR"
"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the selected Python environment")
print("CUDA_OK", torch.cuda.get_device_name(0), "torch", torch.__version__)
PY

mkdir -p "$OUTPUT_ROOT"
echo "Training $skill -> $OUTPUT_DIR"
echo "steps=$STEPS batch=$BATCH_SIZE chunk=$CHUNK_SIZE amp=$USE_AMP augmentation=$IMAGE_AUGMENTATION"

cd "$WBC_ROOT"
exec "$PYTHON" -m lerobot.scripts.train \
    --dataset.repo_id=tmp/tmp_dataset \
    --dataset.root="$DATASET_DIR" \
    --dataset.video_backend=pyav \
    --dataset.use_imagenet_stats=true \
    --dataset.image_transforms.enable="$IMAGE_AUGMENTATION" \
    --policy.type=act \
    --policy.device=cuda \
    --policy.use_amp="$USE_AMP" \
    --policy.vision_backbone=resnet18 \
    --policy.chunk_size="$CHUNK_SIZE" \
    --policy.n_action_steps="$CHUNK_SIZE" \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$DATASET_NAME" \
    --batch_size="$BATCH_SIZE" \
    --num_workers="$NUM_WORKERS" \
    --steps="$STEPS" \
    --eval_freq=0 \
    --log_freq="$LOG_FREQ" \
    --save_checkpoint=true \
    --save_freq="$SAVE_FREQ" \
    --wandb.enable=false

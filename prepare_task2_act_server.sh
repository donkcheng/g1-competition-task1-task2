#!/usr/bin/env bash
# Bootstrap the pinned task-2 ACT environment on a fresh Ubuntu CUDA server.
set -euo pipefail

WBC_ROOT="${TASK2_WBC_ROOT:-/workspace/GR00T-WholeBodyControl}"
WBC_REVISION="${TASK2_WBC_REVISION:-b042411fae38ee4d1af9aac82a37a1f8d14d6dd0}"
CACHE_ROOT="${TASK2_CACHE_ROOT:-/workspace/task2_cache}"

command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is unavailable; choose a GPU image with the NVIDIA driver enabled" >&2
    exit 1
}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ ! -d "$WBC_ROOT/.git" ]]; then
    command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
    git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "$WBC_ROOT"
fi

git -C "$WBC_ROOT" fetch origin "$WBC_REVISION"
git -C "$WBC_ROOT" checkout --detach "$WBC_REVISION"
bash "$WBC_ROOT/install_scripts/install_data_collection.sh"

PYTHON="$WBC_ROOT/.venv_data_collection/bin/python"

# The pinned LeRobot revision assumes every camera already has an entry in
# dataset.meta.stats.  Task-2 deliberately stores only numeric per-episode
# statistics and asks LeRobot to apply ImageNet statistics to the videos, so
# initialize the camera mapping before writing mean/std.  This one-line
# compatibility fix is required by local v2.1 datasets without image stats.
LEROBOT_FACTORY="$("$PYTHON" -c 'from pathlib import Path; import lerobot.common.datasets.factory as f; print(Path(f.__file__).resolve())')"
if grep -Fq 'dataset.meta.stats[key][stats_type]' "$LEROBOT_FACTORY"; then
    sed -i \
        's/dataset\.meta\.stats\[key\]\[stats_type\]/dataset.meta.stats.setdefault(key, {})[stats_type]/' \
        "$LEROBOT_FACTORY"
fi
grep -Fq 'dataset.meta.stats.setdefault(key, {})[stats_type]' "$LEROBOT_FACTORY" || {
    echo "Unable to verify the LeRobot camera-stat compatibility fix in $LEROBOT_FACTORY" >&2
    exit 1
}

mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" "$CACHE_ROOT/xdg"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
"$PYTHON" - <<'PY'
import torch
from torchvision.models import ResNet18_Weights, resnet18

print("torch", torch.__version__, "cuda build", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the rented GPU")
print("GPU", torch.cuda.get_device_name(0))
# Cache the only pretrained file ACT needs before the timed training run.
resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print("TASK2_ACT_SERVER_READY")
PY

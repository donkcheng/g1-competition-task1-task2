#!/usr/bin/env bash
# Validate and rsync the task-2 training code and datasets to a cloud server.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WBC_ROOT="${TASK2_WBC_ROOT:-/home/drgh/g1_official/GR00T-WholeBodyControl}"
PYTHON="${TASK2_PYTHON:-$WBC_ROOT/.venv_data_collection/bin/python}"
LOCAL_DATA_ROOT="${TASK2_DATA_ROOT:-$PROJECT_DIR/data}"

DESTINATION="${1:-}"
REMOTE_PROJECT="${2:-/workspace/g1_competition}"
REMOTE_DATA_ROOT="${3:-/workspace/task2_data}"
if [[ -z "$DESTINATION" ]]; then
    echo "Usage: $0 user@server [/workspace/g1_competition] [/workspace/task2_data]" >&2
    exit 2
fi
command -v rsync >/dev/null || { echo "rsync is required" >&2; exit 1; }
command -v ssh >/dev/null || { echo "ssh is required" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Validation Python not found: $PYTHON" >&2; exit 1; }

for skill in button knob switch; do
    dataset="$LOCAL_DATA_ROOT/task2_${skill}_act"
    "$PYTHON" "$PROJECT_DIR/tools/verify_task2_act_dataset.py" "$dataset"
done

ssh "$DESTINATION" mkdir -p "$REMOTE_PROJECT" "$REMOTE_DATA_ROOT"

# Transfer code without local robot data, caches, outputs, or Python bytecode.
rsync -a --partial --info=progress2 \
    --exclude=data/ \
    --exclude=outputs/ \
    --exclude=__pycache__/ \
    --exclude='*.pyc' \
    "$PROJECT_DIR/" "$DESTINATION:$REMOTE_PROJECT/"

# Video files are already compressed. rsync's resumable transfer is preferable
# to wrapping them in another compressed archive.
for skill in button knob switch; do
    dataset_name="task2_${skill}_act"
    rsync -a --partial --info=progress2 \
        "$LOCAL_DATA_ROOT/$dataset_name/" \
        "$DESTINATION:$REMOTE_DATA_ROOT/$dataset_name/"
done

echo "TASK2_CLOUD_SYNC_OK destination=$DESTINATION"

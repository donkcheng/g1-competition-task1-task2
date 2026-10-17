#!/usr/bin/env bash
# Train the three independent task-2 ACT policies sequentially.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ENV="$PROJECT_DIR/config/task2_act_train.env"
[[ -f "$TRAIN_ENV" ]] || { echo "Missing $TRAIN_ENV" >&2; exit 1; }
# shellcheck disable=SC1090
source "$TRAIN_ENV"

OUTPUT_ROOT="${TASK2_OUTPUT_ROOT:-$PROJECT_DIR/outputs/task2_act}"
RUN_GROUP="${TASK2_RUN_GROUP:-full_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$OUTPUT_ROOT/logs/$RUN_GROUP"
mkdir -p "$LOG_DIR"

echo "TASK2_ACT_ALL_START run_group=$RUN_GROUP time=$(date --iso-8601=seconds)"
for skill in button knob switch; do
    echo "TASK2_ACT_SKILL_START skill=$skill time=$(date --iso-8601=seconds)"
    TASK2_RUN_ID="${RUN_GROUP}_${skill}" \
        bash "$PROJECT_DIR/train_task2_act.sh" "$skill" \
        2>&1 | tee "$LOG_DIR/${skill}.log"
    echo "TASK2_ACT_SKILL_DONE skill=$skill time=$(date --iso-8601=seconds)"
done
echo "TASK2_ACT_ALL_DONE run_group=$RUN_GROUP time=$(date --iso-8601=seconds)"

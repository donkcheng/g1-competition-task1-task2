#!/usr/bin/env bash
# Continuous ACT-oriented task-2 collection (head + right wrist + Dex1).
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_DIR=/home/drgh/g1_official/GR00T-WholeBodyControl
PYTHON="$OFFICIAL_DIR/.venv_data_collection/bin/python"

skill="${1:-}"
if [[ "$skill" != button && "$skill" != knob && "$skill" != switch ]]; then
    echo "Usage: $0 {button|knob|switch} [collector options]" >&2
    echo "Example: $0 button" >&2
    exit 2
fi
shift

case "$skill" in
    button) default_dataset=task2_button_act ;;
    knob) default_dataset=task2_knob_act ;;
    switch) default_dataset=task2_switch_act ;;
esac

has_dataset=0
for arg in "$@"; do
    [[ "$arg" == --dataset || "$arg" == --dataset=* ]] && has_dataset=1
done

args=(
    --repo "$OFFICIAL_DIR"
    --data-root "$PROJECT_DIR/data"
    --skill "$skill"
    --fps 20
)
if (( has_dataset == 0 )); then
    args+=(--dataset "$default_dataset")
fi

cd "$OFFICIAL_DIR"
exec "$PYTHON" -u "$PROJECT_DIR/src/task2_act_collector.py" "${args[@]}" "$@"

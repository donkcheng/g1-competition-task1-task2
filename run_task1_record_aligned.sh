#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IFACE="${TASK1_IFACE:-enp8s0}"
OUTPUT="${1:-$SCRIPT_DIR/routes/task1_aligned_$(date +%Y%m%d_%H%M%S).json}"
if [[ "$OUTPUT" != /* ]]; then
  OUTPUT="$SCRIPT_DIR/$OUTPUT"
fi

printf '%s\n' '第1步：检查头部相机起点位姿（只读，不控制机器人）'
"$SCRIPT_DIR/run_task1_align.sh" --iface "$IFACE" check

printf '%s\n' '第2步：起点合格。请用原厂遥控器驾驶；1/2/3标点，P保存，X急停保存。'
"$SCRIPT_DIR/run_task1.sh" record \
  --real --passive --iface "$IFACE" --output "$OUTPUT"

printf '%s\n' '第3步：校验新路线'
"$SCRIPT_DIR/run_task1.sh" validate --route "$OUTPUT" --require-markers 3
printf '新路线可用：%s\n' "$OUTPUT"

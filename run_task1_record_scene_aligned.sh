#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IFACE="${TASK1_IFACE:-enp8s0}"
OUTPUT="${1:-$SCRIPT_DIR/routes/task1_scene_aligned_$(date +%Y%m%d_%H%M%S).json}"
CONFIRM="${2:-}"
if [[ "$OUTPUT" != /* ]]; then
  OUTPUT="$SCRIPT_DIR/$OUTPUT"
fi
if [[ "$CONFIRM" != "SET_START_REFERENCE" ]]; then
  echo "用法：$0 routes/路线.json SET_START_REFERENCE" >&2
  echo "仅当机器人已位于本次录制采用的起点时执行。" >&2
  exit 2
fi

printf '%s\n' '第1步：把当前头部画面保存为本次路线的参考起点（只读，不控制机器人）'
"$SCRIPT_DIR/run_task1_scene_align.sh" --iface "$IFACE" capture-reference --overwrite

printf '%s\n' '第2步：请从当前姿态用原厂遥控器驾驶；1/2/3标点，P保存，X急停保存。'
"$SCRIPT_DIR/run_task1.sh" record \
  --real --passive --iface "$IFACE" --output "$OUTPUT"

printf '%s\n' '第3步：校验新路线'
"$SCRIPT_DIR/run_task1.sh" validate --route "$OUTPUT" --require-markers 3
printf '新路线可用：%s\n' "$OUTPUT"

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IFACE="${TASK1_IFACE:-enp8s0}"
ROUTE="${1:-}"
CONFIRM="${2:-}"
if [[ -z "$ROUTE" || "$CONFIRM" != "RUN_TASK1" ]]; then
  echo "用法：$0 routes/路线.json RUN_TASK1" >&2
  exit 2
fi
if [[ "$ROUTE" != /* ]]; then
  ROUTE="$SCRIPT_DIR/$ROUTE"
fi

MEASUREMENT="$(mktemp /tmp/task1_scene_offset.XXXXXX.json)"
trap 'rm -f -- "$MEASUREMENT"' EXIT

echo "正在检查起点是否落在允许偏差范围；检查通过后直接起步，并在前几步修正航向。"
"$SCRIPT_DIR/run_task1_scene_align.sh" \
  --iface "$IFACE" \
  measure \
  --output "$MEASUREMENT"
DEFAULT_PYTHON="/home/drgh/g1_official/GR00T-WholeBodyControl/.venv_teleop/bin/python"
if [[ -n "${TASK1_PYTHON:-}" ]]; then
  PYTHON_BIN="$TASK1_PYTHON"
elif [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON_BIN="$DEFAULT_PYTHON"
else
  PYTHON_BIN="python3"
fi
YAW_OFFSET="$($PYTHON_BIN -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
yaw = float(data["yaw_compensation_deg"])
dx = float(data["image_dx_px"])
dy = float(data["image_dy_px"])
max_yaw = float(sys.argv[2])
max_pixel = float(sys.argv[3])
if abs(yaw) > max_yaw:
    raise SystemExit(f"起点偏角 {yaw:.2f}° 超过允许范围 ±{max_yaw:.1f}°，只需人工粗略摆正")
if abs(dx) > max_pixel or abs(dy) > max_pixel:
    raise SystemExit(
        f"起点图像偏差 ({dx:+.1f}, {dy:+.1f})px 超过允许范围 ±{max_pixel:.0f}px，"
        "只需人工粗略移近参考起点"
    )
print(f"{yaw:.6f}")
' "$MEASUREMENT" "${TASK1_START_MAX_YAW_DEG:-12}" "${TASK1_START_MAX_PIXEL:-100}")"

echo "起点在允许范围内；不要求原地精调。开始行走后在前几步补偿航向 ${YAW_OFFSET}°。"
exec "$SCRIPT_DIR/run_task1.sh" replay \
  --real --confirm RUN_TASK1 \
  --iface "$IFACE" \
  --route "$ROUTE" \
  --require-markers 3 \
  --initial-yaw-offset-deg "$YAW_OFFSET"

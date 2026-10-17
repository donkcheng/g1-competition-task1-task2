#!/usr/bin/env bash
# Right-arm-only Task 2 teleoperation. Normal locomotion holds the body.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_DIR=/home/drgh/g1_official/GR00T-WholeBodyControl
PYTHON="$OFFICIAL_DIR/.venv_teleop/bin/python"
PROGRAM="$PROJECT_DIR/tools/task2_right_arm_teleop.py"

if [[ "${1:-}" != --real || $# -ne 1 ]]; then
    echo "Usage: bash $0 --real" >&2
    echo "Robot must be in normal locomotion mode with suspension and E-stop ready." >&2
    exit 2
fi

[[ -x "$PYTHON" ]] || { echo "Teleop Python missing: $PYTHON" >&2; exit 1; }
[[ -s "$PROGRAM" ]] || { echo "Arm-only program missing: $PROGRAM" >&2; exit 1; }

echo "RIGHT-ARM-ONLY mode: full 6D PICO pose drives all 7 right-arm joints; SONIC is not used."
echo "Hold right grip to move; release grip releases Arm SDK; right trigger=Dex1; ABXY exits."
echo "No artificial elbow/wrist lock or workspace freeze; physical joint limits remain active."

cd "$OFFICIAL_DIR"
exec "$PYTHON" -u "$PROGRAM" \
    --real \
    --interface enp8s0 \
    --translation-scale "${TASK2_ARM_TRANSLATION_SCALE:-0.75}" \
    --joint-speed "${TASK2_ARM_JOINT_SPEED:-0.50}" \
    --arm-weight "${TASK2_ARM_WEIGHT:-1.0}" \
    --weight-rate "${TASK2_ARM_WEIGHT_RATE:-0.50}" \
    --right-arm-kp "${TASK2_ARM_KP:-50}" \
    --right-arm-kd "${TASK2_ARM_KD:-1.5}" \
    --max-ik-position-error "${TASK2_ARM_MAX_IK_POSITION_ERROR:-0.05}" \
    --max-ik-rotation-error-deg "${TASK2_ARM_MAX_IK_ROTATION_ERROR_DEG:-55}" \
    --thermal-warn "${TASK2_THERMAL_WARN:-65}" \
    --thermal-stop "${TASK2_THERMAL_STOP:-70}" \
    --thermal-resume "${TASK2_THERMAL_RESUME:-62}"

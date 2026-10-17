#!/usr/bin/env bash
# Task 2: one XR client for SONIC body teleoperation and the right Dex1 gripper.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_DIR=/home/drgh/g1_official/GR00T-WholeBodyControl
PYTHON="$OFFICIAL_DIR/.venv_teleop/bin/python"
WRAPPER="$PROJECT_DIR/tools/start_pico_manager_in_planner.py"

if [[ "${1:-}" != --real || $# -ne 1 ]]; then
    echo "Usage: bash $0 --real" >&2
    echo "This auto-starts PLANNER, then enters VR_3PT after 5 seconds." >&2
    exit 2
fi

[[ -x "$PYTHON" ]] || { echo "Teleop Python missing: $PYTHON" >&2; exit 1; }
[[ -s "$WRAPPER" ]] || { echo "Combined PICO wrapper missing: $WRAPPER" >&2; exit 1; }

THERMAL_WARN="${TASK2_THERMAL_WARN:-70}"
THERMAL_CRITICAL="${TASK2_THERMAL_CRITICAL:-${TASK2_THERMAL_STOP:-75}}"
RIGHT_WRIST_SOURCE="${TASK2_RIGHT_WRIST_SOURCE:-controller}"
RIGHT_WRIST_ANGULAR_SPEED="${TASK2_RIGHT_WRIST_ANGULAR_SPEED:-240}"
RIGHT_WRIST_LINEAR_SPEED="${TASK2_RIGHT_WRIST_LINEAR_SPEED:-1.5}"
RIGHT_WRIST_FILTER_TAU="${TASK2_RIGHT_WRIST_FILTER_TAU:-0.04}"

echo "Task 2 combined PICO control: right trigger=right Dex1, body=SONIC VR_3PT"
echo "Temperature monitor: warn ${THERMAL_WARN}C; critical log ${THERMAL_CRITICAL}C; warning-only, no temperature-driven VR exit"
echo "Both wrists: direct controller 6DoF; angular rate ${RIGHT_WRIST_ANGULAR_SPEED}deg/s; filter tau ${RIGHT_WRIST_FILTER_TAU}s"
echo "No post-calibration wrist-angle clamp is applied."
echo "Passive SONIC wrist target/command/measurement logging is enabled."
echo "Automatic PLANNER/VR entry is enabled; keep the suspension and E-stop ready."

cd "$OFFICIAL_DIR"
exec "$PYTHON" -u "$WRAPPER" \
    --confirm-autostart \
    --auto-vr-delay 5 \
    --dex1-right \
    --thermal-guard \
    --thermal-warn "$THERMAL_WARN" \
    --thermal-critical "$THERMAL_CRITICAL" \
    --right-wrist-source "$RIGHT_WRIST_SOURCE" \
    --right-wrist-max-angular-speed "$RIGHT_WRIST_ANGULAR_SPEED" \
    --right-wrist-max-linear-speed "$RIGHT_WRIST_LINEAR_SPEED" \
    --right-wrist-filter-tau "$RIGHT_WRIST_FILTER_TAU" \
    --wrist-diagnostics \
    --interface enp8s0

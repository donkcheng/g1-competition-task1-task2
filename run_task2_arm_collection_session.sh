#!/usr/bin/env bash
# Task 2 arm-only collection: normal locomotion + rt/arm_sdk + Dex1 + cameras.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="$PROJECT_DIR/.task2_arm_collection.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another task-2 arm collection session is already running." >&2
    echo "Do not start a second copy; stop the existing terminal with Ctrl+C first." >&2
    exit 3
fi
WRIST_MANAGER="$PROJECT_DIR/tools/manage_orin_wrist_camera.py"
ORIN_PREPARE="$PROJECT_DIR/tools/prepare_orin_task2.py"
ARM_CONTROL="$PROJECT_DIR/run_task2_right_arm_teleop.sh"
COLLECTOR="$PROJECT_DIR/run_task2.sh"
LOG_DIR="$PROJECT_DIR/logs"
DATA_PYTHON=/home/drgh/g1_official/GR00T-WholeBodyControl/.venv_data_collection/bin/python
TELEOP_PYTHON=/home/drgh/g1_official/GR00T-WholeBodyControl/.venv_teleop/bin/python
CAMERA_CHECK="$PROJECT_DIR/tools/check_task2_cameras.py"
TEMPERATURE_CHECK="$PROJECT_DIR/tools/check_g1_temperature.py"
XR_CHECK="$PROJECT_DIR/tools/check_xr_stream.py"

usage() {
    cat <<'EOF'
Usage: bash run_task2_arm_collection_session.sh {button|knob|switch} --real [collector options]

Requirements:
  - robot is in NORMAL LOCOMOTION mode (do not start SONIC)
  - suspension has slack, area is clear, original E-stop is available
  - PICO XRoboToolkit is WORKING and Send is enabled

Controls:
  Hold right grip       right arm follows controller translation
  Release right grip    smoothly release Arm SDK to normal locomotion
  Right trigger         Dex1 close/open
  Right stick left/right rotate tool for the knob (tilt remains locked)
  Left grip+A           start / stop-and-save an episode
  Left grip+B           abort the current episode
  A+B+X+Y               release Arm SDK and stop
EOF
}

skill="${1:-}"
if [[ "$skill" == -h || "$skill" == --help || -z "$skill" ]]; then
    usage
    exit 0
fi
if [[ "$skill" != button && "$skill" != knob && "$skill" != switch ]]; then
    usage >&2
    exit 2
fi
shift
if [[ "${1:-}" != --real ]]; then
    echo "Refusing real arm control without --real." >&2
    usage >&2
    exit 2
fi
shift
collector_args=("$@")

control_pid=""
collector_pid=""
wrist_started=0
camera_started=0
cleanup_started=0

stop_child() {
    local pid="${1:-}"
    [[ -n "$pid" ]] || return 0
    kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return 0; }
    kill -INT "$pid" 2>/dev/null || true
    for _ in {1..20}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.1
        done
    fi
    wait "$pid" 2>/dev/null || true
}

cleanup() {
    local status=$?
    if (( cleanup_started )); then
        return "$status"
    fi
    cleanup_started=1
    trap - EXIT INT TERM HUP
    echo
    echo "Stopping task-2 arm-only collection..."
    local control_was_started=0
    [[ -n "$control_pid" ]] && control_was_started=1
    stop_child "$collector_pid"
    stop_child "$control_pid"
    if (( camera_started )); then
        "$PROJECT_DIR/run_task2_camera.sh" stop || true
    fi
    if (( wrist_started )); then
        ORIN_PASSWORD="$ORIN_PASSWORD" python3 "$WRIST_MANAGER" stop || true
    fi
    if (( control_was_started )); then
        echo "Arm SDK released; normal locomotion remains responsible for the robot body."
    else
        echo "Preflight stopped before Arm SDK control was started."
    fi
    return "$status"
}
trap cleanup EXIT INT TERM HUP

echo "[0/6] Checking G1 motor temperatures before starting peripherals..."
if ! "$TELEOP_PYTHON" "$TEMPERATURE_CHECK" \
    --ready-below "${TASK2_THERMAL_WARN:-65}"; then
    echo "Temperature is high; session will start but arm control will remain in COOLING until safe." >&2
fi

if [[ -z "${ORIN_PASSWORD:-}" ]]; then
    read -r -s -p "Password for unitree@192.168.123.164: " ORIN_PASSWORD
    echo
fi
export ORIN_PASSWORD
mkdir -p "$LOG_DIR"
control_log="$LOG_DIR/task2_arm_only_$(date +%Y%m%d_%H%M%S).log"

echo "[1/6] Ensuring XRoboToolkit PC service is running..."
"$PROJECT_DIR/run_xrobotoolkit.sh" start
echo "      Waiting for fresh PICO body tracking (WORKING + Tracking/Send)..."
if ! "$TELEOP_PYTHON" "$XR_CHECK" \
    --wait-seconds "${TASK2_XR_WAIT_SECONDS:-45}" \
    --seconds 3 \
    --min-rate 10 \
    --max-gap 0.5; then
    echo "PICO is connected at the network layer but is not continuously sending body frames." >&2
    echo "Wake/wear the headset, keep XRoboToolkit in front, enable Tracking/Send, then run this command again." >&2
    exit 4
fi

echo "[2/6] Preparing the Orin right Dex1 service..."
for attempt in 1 2 3; do
    if ORIN_PASSWORD="$ORIN_PASSWORD" python3 "$ORIN_PREPARE" prepare; then
        break
    else
        status=$?
    fi
    if (( status != 3 )); then
        exit "$status"
    fi
    if (( attempt == 3 )); then
        echo "Orin authentication failed three times." >&2
        exit 3
    fi
    read -r -s -p "Password rejected; enter Orin password again: " ORIN_PASSWORD
    echo
    export ORIN_PASSWORD
done

echo "[3/6] Restarting the Orin right-wrist camera..."
ORIN_PASSWORD="$ORIN_PASSWORD" python3 "$WRIST_MANAGER" restart
wrist_started=1

echo "[4/6] Starting and checking the PC dual-camera bridge..."
"$PROJECT_DIR/run_task2_camera.sh" start
camera_started=1
"$DATA_PYTHON" "$CAMERA_CHECK" --timeout 12

echo "[5/6] Starting right-arm-only PICO control (SONIC is NOT used)..."
bash "$ARM_CONTROL" --real > >(tee -a "$control_log") 2>&1 &
control_pid=$!
sleep 3
if ! kill -0 "$control_pid" 2>/dev/null; then
    set +e
    wait "$control_pid"
    status=$?
    set -e
    control_pid=""
    echo "Arm-only controller failed during startup; see $control_log" >&2
    exit "$status"
fi

echo "[6/6] Starting continuous ACT collection for: $skill"
echo "      Hold right grip only while moving/interacting; release it between attempts."
echo "      Left Grip+A=start/stop-save; Left Grip+B=abort; terminal q+Enter=quit."
bash "$COLLECTOR" collect-act "$skill" "${collector_args[@]}" < /dev/tty &
collector_pid=$!

while kill -0 "$control_pid" 2>/dev/null && kill -0 "$collector_pid" 2>/dev/null; do
    sleep 0.2
done

if ! kill -0 "$control_pid" 2>/dev/null; then
    set +e
    wait "$control_pid"
    control_status=$?
    set -e
    control_pid=""
    if (( control_status == 75 )); then
        echo "Arm temperature protection stopped control; ending collection." >&2
    else
        echo "Arm-only controller exited (status=$control_status); ending collection." >&2
    fi
    stop_child "$collector_pid"
    collector_pid=""
    exit "$control_status"
fi

set +e
wait "$collector_pid"
collector_status=$?
set -e
collector_pid=""
exit "$collector_status"

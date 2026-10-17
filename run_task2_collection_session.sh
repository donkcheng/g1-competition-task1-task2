#!/usr/bin/env bash
# Task 2 terminal-B launcher: XR + dual cameras + PICO body/Dex1 manager + ACT collector.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WRIST_MANAGER="$PROJECT_DIR/tools/manage_orin_wrist_camera.py"
ORIN_PREPARE="$PROJECT_DIR/tools/prepare_orin_task2.py"
COMBINED="$PROJECT_DIR/run_task2_pico_combined.sh"
COLLECTOR="$PROJECT_DIR/run_task2.sh"
LOG_DIR="$PROJECT_DIR/logs"
DATA_PYTHON=/home/drgh/g1_official/GR00T-WholeBodyControl/.venv_data_collection/bin/python
CAMERA_CHECK="$PROJECT_DIR/tools/check_task2_cameras.py"

usage() {
    cat <<'EOF'
Usage: bash run_task2_collection_session.sh {button|knob|switch} --real [collector options]

Run this only after terminal A shows SONIC "Init Done":
  bash run_task2_body_teleop.sh --real

This terminal B script starts/ensures:
  XRoboToolkit PC service
  Orin right Dex1 service (and pauses conflicting LoRa)
  Orin right-wrist camera service
  PC dual-camera bridge
  PICO body + right Dex1 manager
  continuous ACT collector

PICO collection controls:
  Left Grip+A  start / stop-and-save an episode
  Left Grip+B  abort the current episode

The script prompts once for the Orin password unless ORIN_PASSWORD is set.
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
    echo "Refusing to start real teleoperation without the explicit --real flag." >&2
    usage >&2
    exit 2
fi
shift
collector_args=("$@")

manager_pid=""
collector_pid=""
wrist_started=0
camera_started=0
cleanup_started=0

cleanup() {
    local status=$?
    if (( cleanup_started )); then
        return "$status"
    fi
    cleanup_started=1
    trap - EXIT INT TERM HUP

    echo
    echo "Stopping task-2 terminal-B services..."
    if [[ -n "$manager_pid" ]] && kill -0 "$manager_pid" 2>/dev/null; then
        kill -INT "$manager_pid" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "$manager_pid" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$manager_pid" 2>/dev/null; then
            kill -TERM "$manager_pid" 2>/dev/null || true
        fi
        wait "$manager_pid" 2>/dev/null || true
    fi
    if [[ -n "$collector_pid" ]] && kill -0 "$collector_pid" 2>/dev/null; then
        kill -INT "$collector_pid" 2>/dev/null || true
        wait "$collector_pid" 2>/dev/null || true
    fi
    if (( camera_started )); then
        "$PROJECT_DIR/run_task2_camera.sh" stop || true
    fi
    if (( wrist_started )); then
        ORIN_PASSWORD="$ORIN_PASSWORD" python3 "$WRIST_MANAGER" stop || true
    fi
    echo "Terminal B stopped. Stop SONIC safely in terminal A with uppercase O."
    return "$status"
}
trap cleanup EXIT INT TERM HUP

if [[ -z "${ORIN_PASSWORD:-}" ]]; then
    read -r -s -p "Password for unitree@192.168.123.164: " ORIN_PASSWORD
    echo
fi
export ORIN_PASSWORD

mkdir -p "$LOG_DIR"
manager_log="$LOG_DIR/task2_combined_$(date +%Y%m%d_%H%M%S).log"

echo "[1/6] Ensuring XRoboToolkit PC service is running..."
"$PROJECT_DIR/run_xrobotoolkit.sh" start

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
        echo "Orin authentication failed three times; nothing was started." >&2
        exit 3
    fi
    echo "The Orin password was rejected; please enter it again." >&2
    read -r -s -p "Password for unitree@192.168.123.164: " ORIN_PASSWORD
    echo
    export ORIN_PASSWORD
done

echo "[3/6] Restarting the Orin right-wrist camera..."
ORIN_PASSWORD="$ORIN_PASSWORD" python3 "$WRIST_MANAGER" restart
wrist_started=1

echo "[4/6] Starting the PC dual-camera bridge..."
"$PROJECT_DIR/run_task2_camera.sh" start
camera_started=1
echo "      Waiting for fresh head and right-wrist frames before robot takeover..."
"$DATA_PYTHON" "$CAMERA_CHECK" --timeout 12

echo "[5/6] Starting PICO body + right Dex1 control..."
echo "      Keep PICO still; this automatically enters PLANNER and VR_3PT."
bash "$COMBINED" --real > >(tee -a "$manager_log") 2>&1 &
manager_pid=$!
sleep 3
if ! kill -0 "$manager_pid" 2>/dev/null; then
    wait "$manager_pid"
    echo "PICO combined manager exited during startup; see $manager_log" >&2
    exit 1
fi

echo "[6/6] Starting continuous ACT collection for: $skill"
echo "      The collector waits until SONIC, PICO, Dex1 and both cameras are fresh."
echo "      Left Grip+A=start/stop-save; Left Grip+B=abort; terminal q+Enter=quit."
bash "$COLLECTOR" collect-act "$skill" "${collector_args[@]}" < /dev/tty &
collector_pid=$!

# Keep terminal input attached to the collector, but supervise both long-lived
# processes here.  Previously a thermal stop ended the manager while the
# collector remained alive with stale sources, which looked like an unexplained
# loss of teleoperation.
while kill -0 "$manager_pid" 2>/dev/null && kill -0 "$collector_pid" 2>/dev/null; do
    sleep 0.2
done

if ! kill -0 "$manager_pid" 2>/dev/null; then
    set +e
    wait "$manager_pid"
    manager_status=$?
    set -e
    manager_pid=""
    if (( manager_status == 75 )); then
        echo "Thermal protection stopped teleoperation; ending collection now." >&2
    else
        echo "PICO combined manager exited (status=$manager_status); ending collection now." >&2
    fi
    if kill -0 "$collector_pid" 2>/dev/null; then
        kill -INT "$collector_pid" 2>/dev/null || true
    fi
    wait "$collector_pid" 2>/dev/null || true
    collector_pid=""
    exit "$manager_status"
fi

set +e
wait "$collector_pid"
collector_status=$?
set -e
collector_pid=""
exit "$collector_status"

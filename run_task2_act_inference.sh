#!/usr/bin/env bash
# Task-2 ACT inference launcher: normal locomotion + right Arm SDK + right Dex1.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_PYTHON=/home/drgh/g1_official/GR00T-WholeBodyControl/.venv_data_collection/bin/python
PROGRAM="$PROJECT_DIR/src/task2_act_inference.py"
ORIN_PREPARE="$PROJECT_DIR/tools/prepare_orin_task2.py"
WRIST_MANAGER="$PROJECT_DIR/tools/manage_orin_wrist_camera.py"
CAMERA_CHECK="$PROJECT_DIR/tools/check_task2_cameras.py"
LOCK_FILE="$PROJECT_DIR/.task2_act_inference.lock"

usage() {
  cat <<'EOF'
Usage:
  bash run_task2_act_inference.sh {button|knob|switch} --dry-run
  bash run_task2_act_inference.sh {button|knob|switch} --real
  bash run_task2_act_inference.sh {button|knob|switch} --offline-smoke

Robot requirements for --real:
  - NORMAL LOCOMOTION mode; SONIC and PICO arm teleop are stopped
  - robot starts at the same cabinet-relative pose as demonstrations
  - suspension/clear area/E-stop are ready

Inside the inference terminal: s+Enter=start, x+Enter=release, q+Enter=quit.
EOF
}

skill="${1:-}"
mode="${2:-}"
if [[ "$skill" != button && "$skill" != knob && "$skill" != switch ]]; then
  usage >&2
  exit 2
fi
if [[ "$mode" != --dry-run && "$mode" != --real && "$mode" != --offline-smoke ]]; then
  usage >&2
  exit 2
fi
shift 2

[[ -x "$DATA_PYTHON" ]] || { echo "Missing data Python: $DATA_PYTHON" >&2; exit 1; }
[[ -s "$PROGRAM" ]] || { echo "Missing inference program: $PROGRAM" >&2; exit 1; }

mkdir -p "$PROJECT_DIR/logs"
SESSION_LOG="$PROJECT_DIR/logs/task2_act_${skill}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$SESSION_LOG") 2>&1
echo "Task2 ACT session log: $SESSION_LOG"

# Reject invalid options before changing any Orin or camera services.
"$DATA_PYTHON" "$PROGRAM" "$skill" "$mode" "$@" --validate-only

if [[ "$mode" == --offline-smoke ]]; then
  cd "$PROJECT_DIR"
  exec "$DATA_PYTHON" -u "$PROGRAM" "$skill" --offline-smoke "$@"
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another task-2 ACT inference session is already running." >&2
  exit 3
fi

if pgrep -f '[t]ask2_right_arm_teleop.py' >/dev/null; then
  echo "PICO right-arm teleop is still running. Stop it before ACT inference." >&2
  exit 4
fi

if [[ ! -d /sys/class/net/enp8s0 ]]; then
  echo "Robot Ethernet interface enp8s0 does not exist." >&2
  exit 5
fi
if [[ "$(cat /sys/class/net/enp8s0/carrier 2>/dev/null || echo 0)" != 1 ]]; then
  cat >&2 <<'EOF'
Robot Ethernet enp8s0 has NO-CARRIER: the PC cannot see a physical link.
Check/reseat the Ethernet cable and wait for the G1 Orin to finish booting.
No robot command was started.
EOF
  exit 5
fi
if ! ip -4 addr show dev enp8s0 | grep -q '192\.168\.123\.222/24'; then
  cat >&2 <<'EOF'
enp8s0 does not have 192.168.123.222/24. Configure it first:
  sudo ip link set dev enp8s0 up
  sudo ip addr replace 192.168.123.222/24 dev enp8s0
EOF
  exit 5
fi
if ! ping -c 1 -W 1 192.168.123.164 >/dev/null 2>&1; then
  cat >&2 <<'EOF'
G1 Orin 192.168.123.164 is not reachable. Wait for Orin boot, then check
the cable and wired IP before retrying. No robot command was started.
EOF
  exit 5
fi

if [[ -z "${ORIN_PASSWORD:-}" ]]; then
  read -r -s -p "Password for unitree@192.168.123.164: " ORIN_PASSWORD
  echo
fi
export ORIN_PASSWORD="${ORIN_PASSWORD:-}"

wrist_started=0
camera_started=0
cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  echo
  echo "Stopping task-2 ACT peripherals..."
  if (( camera_started )); then
    "$PROJECT_DIR/run_task2_camera.sh" stop || true
  fi
  if (( wrist_started )); then
    ORIN_PASSWORD="$ORIN_PASSWORD" python3 "$WRIST_MANAGER" stop || true
  fi
  return "$status"
}
trap cleanup EXIT INT TERM HUP

echo "[1/4] Preparing Orin Dex1 and right-wrist camera..."
ORIN_PASSWORD="$ORIN_PASSWORD" python3 "$ORIN_PREPARE" prepare
ORIN_PASSWORD="$ORIN_PASSWORD" python3 "$WRIST_MANAGER" restart
wrist_started=1

echo "[2/4] Starting the dual-camera bridge..."
"$PROJECT_DIR/run_task2_camera.sh" start
camera_started=1
"$DATA_PYTHON" "$CAMERA_CHECK" --timeout 12

echo "[3/4] Checking local model files..."
(cd "$PROJECT_DIR/models/task2_act" && sha256sum -c SHA256SUMS)

echo "[4/4] Starting ACT $mode for: $skill"
if [[ " $* " == *" --auto-start "* ]]; then
  echo "      自动开始，无需输入 s；x+Enter=release, q+Enter=quit, Ctrl+C=release"
else
  echo "      s+Enter=start, x+Enter=release, q+Enter=quit"
fi
cd "$PROJECT_DIR"
"$DATA_PYTHON" -u "$PROGRAM" "$skill" "$mode" "$@"

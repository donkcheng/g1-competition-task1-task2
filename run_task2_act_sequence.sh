#!/usr/bin/env bash
# Run or live-dry-run button -> knob -> switch ACT policies without switching.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_PYTHON=/home/drgh/g1_official/GR00T-WholeBodyControl/.venv_data_collection/bin/python
PROGRAM="$PROJECT_DIR/src/task2_act_inference.py"
ORIN_PREPARE="$PROJECT_DIR/tools/prepare_orin_task2.py"
WRIST_MANAGER="$PROJECT_DIR/tools/manage_orin_wrist_camera.py"
CAMERA_CHECK="$PROJECT_DIR/tools/check_task2_cameras.py"
LOCK_FILE="$PROJECT_DIR/.task2_act_sequence.lock"

mkdir -p "$PROJECT_DIR/logs"
SEQUENCE_LOG="$PROJECT_DIR/logs/task2_act_sequence_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$SEQUENCE_LOG") 2>&1
echo "Task2 ACT sequence log: $SEQUENCE_LOG"

mode="${1:-}"
if [[ "$mode" != --real && "$mode" != --dry-run ]]; then
  cat >&2 <<'EOF'
Usage:
  bash run_task2_act_sequence.sh --dry-run [common inference options]
  bash run_task2_act_sequence.sh --real [common inference options]

Runs all three policies automatically in this order:
  1. button  2. knob  3. switch

--dry-run uses live G1/Dex1/dual-camera data and CUDA inference, but creates no
arm or gripper command publishers.  It is the required no-motion rehearsal.

Before running: use NORMAL LOCOMOTION, stop SONIC/PICO arm teleop, place G1 at
the demonstrated cabinet-relative start pose, and keep the E-stop accessible.
Ctrl+C or x/q+Enter stops the sequence; an inference/sensor/control failure
also stops the sequence instead of advancing to the next model.
EOF
  exit 2
fi
shift

for argument in "$@"; do
  if [[ "$argument" == --preposition-only ]]; then
    echo "--preposition-only is a single-policy validation option; use run_task2_act_inference.sh." >&2
    exit 2
  fi
done

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another task-2 ACT sequence is already running." >&2
  exit 3
fi
if pgrep -f '[t]ask2_right_arm_teleop.py' >/dev/null; then
  echo "PICO right-arm teleop is still running. Stop it first." >&2
  exit 4
fi
if pgrep -f '[t]ask2_act_inference.py' >/dev/null; then
  echo "Another ACT inference process is still running. Stop it first." >&2
  exit 4
fi
if pgrep -f '[g]1_deploy_onnx_ref' >/dev/null; then
  echo "SONIC g1_deploy_onnx_ref is still running. Stop it before ACT inference." >&2
  exit 4
fi

if [[ ! -d /sys/class/net/enp8s0 ]]; then
  echo "Robot Ethernet interface enp8s0 does not exist." >&2
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
  echo "Cannot reach G1 Orin at 192.168.123.164; check robot boot and Ethernet." >&2
  exit 5
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "NVIDIA GPU/driver is unavailable; ACT real-time inference will not start." >&2
  exit 6
fi
if ! "$DATA_PYTHON" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "The task-2 Python environment cannot access CUDA." >&2
  exit 6
fi

if [[ -z "${ORIN_PASSWORD:-}" ]]; then
  read -r -s -p "Password for unitree@192.168.123.164: " ORIN_PASSWORD
  echo
fi
export ORIN_PASSWORD

wrist_started=0
camera_started=0
cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  echo
  echo "Stopping task-2 ACT sequence peripherals..."
  if (( camera_started )); then
    "$PROJECT_DIR/run_task2_camera.sh" stop || true
  fi
  if (( wrist_started )); then
    ORIN_PASSWORD="$ORIN_PASSWORD" python3 "$WRIST_MANAGER" stop || true
  fi
  return "$status"
}
trap cleanup EXIT INT TERM HUP

echo "[1/4] Preparing Orin Dex1 and right-wrist camera once for all models..."
ORIN_PASSWORD="$ORIN_PASSWORD" python3 "$ORIN_PREPARE" prepare
ORIN_PASSWORD="$ORIN_PASSWORD" python3 "$WRIST_MANAGER" restart
wrist_started=1

echo "[2/4] Starting and checking the dual-camera bridge..."
"$PROJECT_DIR/run_task2_camera.sh" start
camera_started=1
"$DATA_PYTHON" "$CAMERA_CHECK" --timeout 12

echo "[3/4] Verifying all three local policy checkpoints..."
(cd "$PROJECT_DIR/models/task2_act" && sha256sum -c SHA256SUMS)

button_seconds="${TASK2_BUTTON_TIMEOUT_SECONDS:-30}"
knob_seconds="${TASK2_KNOB_TIMEOUT_SECONDS:-30}"
switch_seconds="${TASK2_SWITCH_TIMEOUT_SECONDS:-30}"
pause_seconds="${TASK2_SEQUENCE_PAUSE_SECONDS:-3}"
timed_out_skills=""

if [[ "$mode" == --dry-run ]]; then
  dry_run_seconds="${TASK2_DRY_RUN_SECONDS:-8}"
  button_seconds="$dry_run_seconds"
  knob_seconds="$dry_run_seconds"
  switch_seconds="$dry_run_seconds"
fi

echo "[4/4] Automatic $mode sequence starts now: button -> knob -> switch"
if [[ "$mode" == --real ]]; then
  echo "      Each model advances only after a measured reach/return/settle cycle."
  echo "      Failure timeouts: ${button_seconds}s / ${knob_seconds}s / ${switch_seconds}s"
else
  echo "      LIVE DRY-RUN: no arm or gripper command publishers are created."
  echo "      Each model observes and infers for ${dry_run_seconds}s, then exits."
fi
echo "      Temperature is warning-only; onboard robot protection remains active."

run_skill() {
  local index="$1"
  local skill="$2"
  local duration="$3"
  shift 3
  echo
  echo "========== ACT STEP $index/3: $skill (${duration}s) =========="
  if [[ "$mode" == --real ]]; then
    "$DATA_PYTHON" -u "$PROGRAM" "$skill" \
      --real \
      --auto-start \
      --exit-after-attempt \
      --thermal-action warn \
      --completion-mode motion-cycle \
      --max-run-seconds "$duration" \
      "$@"
  else
    "$DATA_PYTHON" -u "$PROGRAM" "$skill" \
      --dry-run \
      --auto-start \
      --exit-after-attempt \
      --thermal-action warn \
      --completion-mode timeout \
      --max-run-seconds "$duration" \
      "$@"
  fi
}

run_skill_allow_timeout() {
  local index="$1"
  local skill="$2"
  local duration="$3"
  shift 3
  local status
  set +e
  run_skill "$index" "$skill" "$duration" "$@"
  status=$?
  set -e
  if [[ "$mode" == --real && "$status" -eq 6 ]]; then
    echo "ACT STEP $index/3 ($skill) reached ${duration}s without completion; arm released. Continuing to next skill."
    timed_out_skills="${timed_out_skills}${timed_out_skills:+,}$skill"
    return 0
  fi
  return "$status"
}

cd "$PROJECT_DIR"
run_skill_allow_timeout 1 button "$button_seconds" "$@"
if [[ "$mode" == --real ]]; then
  echo "Button motion cycle complete; allowing normal locomotion to settle ${pause_seconds}s..."
else
  echo "Button live dry-run passed; waiting ${pause_seconds}s before loading the next model..."
fi
sleep "$pause_seconds"
run_skill_allow_timeout 2 knob "$knob_seconds" "$@"
if [[ "$mode" == --real ]]; then
  echo "Knob motion cycle complete; allowing normal locomotion to settle ${pause_seconds}s..."
else
  echo "Knob live dry-run passed; waiting ${pause_seconds}s before loading the next model..."
fi
sleep "$pause_seconds"
run_skill_allow_timeout 3 switch "$switch_seconds" "$@"

echo
if [[ "$mode" == --real ]]; then
  if [[ -n "$timed_out_skills" ]]; then
    echo "TASK2 ACT SEQUENCE FINISHED; timed-out skills were released and skipped: $timed_out_skills"
  else
    echo "TASK2 ACT SEQUENCE COMPLETE: button -> knob -> switch"
  fi
else
  echo "TASK2 ACT LIVE DRY-RUN COMPLETE: button -> knob -> switch; ZERO COMMANDS"
fi

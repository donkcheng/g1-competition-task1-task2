#!/usr/bin/env bash
# One-command real ACT execution after all task-2 peripherals pass preflight.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
skill="${1:-}"

if [[ "$skill" != button && "$skill" != knob && "$skill" != switch ]]; then
  cat >&2 <<'EOF'
Usage: bash run_task2_act_auto.sh {button|knob|switch} --real [inference options]

This is real robot control. Before running it:
  - put G1 in NORMAL LOCOMOTION mode at the demonstrated cabinet-relative pose
  - stop SONIC and PICO arm teleop
  - keep the suspension, clear area and original E-stop ready

The policy starts automatically after model/sensor preflight.
Use x/q+Enter or Ctrl+C to release it.
EOF
  exit 2
fi
if [[ "${2:-}" != --real ]]; then
  echo "Refusing automatic real control without the literal --real argument." >&2
  exit 2
fi
shift 2

echo "AUTO ACT: skill=$skill"
echo "The robot will begin after model, G1, Dex1 and both cameras are ready."
echo "Temperature is warning-only; onboard robot protection remains active."
echo "Stop with x/q+Enter or Ctrl+C."

exec bash "$PROJECT_DIR/run_task2_act_inference.sh" \
  "$skill" --real --auto-start --thermal-action warn "$@"

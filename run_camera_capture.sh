#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORIN_HOST="${ORIN_HOST:-192.168.123.164}"
ORIN_USER="${ORIN_USER:-unitree}"
CAMERA_DEVICE="${CAMERA_DEVICE:-4}"

if [[ ! "$CAMERA_DEVICE" =~ ^[0-9]+$ ]]; then
  printf 'ERROR: CAMERA_DEVICE must be a non-negative integer\n' >&2
  exit 2
fi

remote="${ORIN_USER}@${ORIN_HOST}"
session_id="$(date +%Y%m%d_%H%M%S)"
remote_script="/tmp/g1_capture_head_camera.py"
remote_output="/home/${ORIN_USER}/g1_camera_captures/${session_id}"
local_output="${SCRIPT_DIR}/captures/${session_id}"
control_dir="$(mktemp -d /tmp/g1camera.XXXXXX)"
control_socket="${control_dir}/ssh"

cleanup() {
  ssh -S "$control_socket" -O exit "$remote" >/dev/null 2>&1 || true
  rm -f -- "$control_socket"
  rmdir -- "$control_dir" 2>/dev/null || true
}
trap cleanup EXIT

printf '连接 G1 Orin %s（请按提示输入现场配置的 SSH 密码）...\n' "$remote"
ssh -MNf \
  -o ControlMaster=yes \
  -o ControlPath="$control_socket" \
  -o ControlPersist=120 \
  -o ConnectTimeout=5 \
  "$remote"

scp -q \
  -o ControlPath="$control_socket" \
  "$SCRIPT_DIR/src/capture_head_camera.py" \
  "$remote:$remote_script"

set +e
ssh -tt \
  -o ControlPath="$control_socket" \
  "$remote" \
  "python3 '$remote_script' --device '$CAMERA_DEVICE' --output-dir '$remote_output'"
capture_status=$?
set -e

mkdir -p -- "$local_output"
remote_photo_count="$(ssh -o ControlPath="$control_socket" "$remote" \
  "find '$remote_output' -maxdepth 1 -type f -name '*.jpg' | wc -l")"
if (( remote_photo_count > 0 )); then
  scp -q \
    -o ControlPath="$control_socket" \
    "$remote:$remote_output/*.jpg" \
    "$local_output/"
fi

photo_count="$(find "$local_output" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
printf '已下载 %s 张照片到：%s\n' "$photo_count" "$local_output"

if (( capture_status != 0 )); then
  printf 'WARN: 远端拍摄程序退出码为 %s\n' "$capture_status" >&2
  exit "$capture_status"
fi

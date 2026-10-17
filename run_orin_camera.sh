#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    'Run this script ON THE G1 ORIN.' \
    'Usage:' \
    '  GROOT_WBC_ROOT=/path/to/GR00T-WholeBodyControl EGO_MXID=<OAK_MXID> ./run_orin_camera.sh check' \
    '  GROOT_WBC_ROOT=/path/to/GR00T-WholeBodyControl EGO_MXID=<OAK_MXID> ./run_orin_camera.sh start' \
    '  ./run_orin_camera.sh service-status|service-logs'
}

action="${1:-}"
case "$action" in
  service-status)
    exec systemctl status composed_camera_server.service
    ;;
  service-logs)
    exec journalctl -u composed_camera_server.service -f
    ;;
  check|start)
    ;;
  *)
    usage
    exit 2
    ;;
esac

repo="${GROOT_WBC_ROOT:-}"
mxid="${EGO_MXID:-}"
if [[ -z "$repo" || -z "$mxid" ]]; then
  usage
  exit 2
fi
python_bin="$repo/.venv_camera/bin/python"
if [[ ! -x "$python_bin" ]]; then
  printf 'ERROR: missing %s; run official install_scripts/install_camera_server.sh first\n' "$python_bin" >&2
  exit 2
fi

if [[ "$action" == check ]]; then
  "$python_bin" -c 'import depthai as dai; print(dai.Device.getAllAvailableDevices())'
  printf 'Configured ego MxID: %s\n' "$mxid"
  exit 0
fi

cd "$repo"
exec "$python_bin" -m gear_sonic.camera.composed_camera \
  --ego-view-camera oak \
  --ego-view-device-id "$mxid" \
  --port 5555

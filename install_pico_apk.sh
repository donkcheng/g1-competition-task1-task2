#!/usr/bin/env bash
set -euo pipefail

ADB="${ADB_BIN:-/home/drgh/g1_official/deps/platform-tools/adb}"
APK="${PICO_APK:-/home/drgh/g1_official/deps/XRoboToolkit-PICO-1.1.1.apk}"

require_files() {
  [[ -x "$ADB" ]] || {
    echo "Missing adb: $ADB" >&2
    exit 1
  }
  [[ -s "$APK" ]] || {
    echo "Missing APK: $APK" >&2
    exit 1
  }
}

require_device() {
  local state
  state="$($ADB get-state 2>&1)" || {
    echo "$state" >&2
    echo "Connect PICO by USB, enable Developer mode and USB debugging, then accept the authorization prompt in the headset." >&2
    exit 1
  }
  [[ "$state" == "device" ]] || {
    echo "PICO adb state is '$state', expected 'device'." >&2
    exit 1
  }
}

require_files

case "${1:-check}" in
  check)
    "$ADB" start-server >/dev/null
    "$ADB" devices -l
    require_device
    echo "PICO USB debugging is ready"
    ;;
  install)
    require_device
    "$ADB" install -r -g "$APK"
    ;;
  *)
    echo "Usage: $0 {check|install}" >&2
    exit 2
    ;;
esac

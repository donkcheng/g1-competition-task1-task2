#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="${XROBOT_SERVICE_DIR:-/home/drgh/g1_official/deps/xrobotoolkit_pc_service_root/opt/apps/roboticsservice}"
UNIT="g1-xrobotoolkit.service"

require_binary() {
  [[ -x "$SERVICE_DIR/RoboticsServiceProcess" ]] || {
    echo "Missing service binary: $SERVICE_DIR/RoboticsServiceProcess" >&2
    exit 1
  }
}

is_running() {
  systemctl --user is-active --quiet "$UNIT"
}

case "${1:-status}" in
  start)
    require_binary
    if is_running; then
      echo "XRoboToolkit PC Service already running"
      exit 0
    fi

    systemd-run --user \
      --unit="$UNIT" \
      --collect \
      --property=Type=simple \
      --property="WorkingDirectory=$SERVICE_DIR" \
      --setenv="LD_LIBRARY_PATH=$SERVICE_DIR:$SERVICE_DIR/lib:$SERVICE_DIR/SDK/x64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      --setenv="QT_PLUGIN_PATH=$SERVICE_DIR/plugins/${QT_PLUGIN_PATH:+:$QT_PLUGIN_PATH}" \
      --setenv="QT_QML_PATH=$SERVICE_DIR/qml/${QT_QML_PATH:+:$QT_QML_PATH}" \
      "$SERVICE_DIR/RoboticsServiceProcess"

    for _ in {1..20}; do
      is_running && break
      sleep 0.1
    done
    if ! is_running; then
      echo "Service exited during startup; run: $0 logs" >&2
      exit 1
    fi
    echo "XRoboToolkit PC Service started as user unit $UNIT"
    ;;
  status)
    if is_running; then
      systemctl --user --no-pager --full status "$UNIT"
      exit 0
    fi
    echo "STOPPED"
    exit 1
    ;;
  stop)
    if ! is_running; then
      echo "Already stopped"
      exit 0
    fi
    systemctl --user stop "$UNIT"
    echo "XRoboToolkit PC Service stopped"
    ;;
  logs)
    journalctl --user -u "$UNIT" -n 100 --no-pager
    ;;
  *)
    echo "Usage: $0 {start|status|stop|logs}" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail
OUTPUT="${1:-}"
CONFIRM="${2:-}"
# Competition task 1 always reaches the Orin over the dedicated robot cable.
# TASK1_ORIN_HOST remains available only for deliberate maintenance overrides.
ORIN_HOST="${TASK1_ORIN_HOST:-192.168.123.164}"

if [[ -z "$OUTPUT" || "$CONFIRM" != "SET_START_REFERENCE" ]]; then
  echo "用法：$0 routes/新路线.json SET_START_REFERENCE" >&2
  exit 2
fi
if [[ ! "$OUTPUT" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  echo "路线名称只能包含字母、数字、点、下划线、斜杠和连字符" >&2
  exit 2
fi

printf '正在从本机连接 Orin %s；标点按键会传给远端录制程序。\n' "$ORIN_HOST"
exec ssh -tt \
  -o StrictHostKeyChecking=accept-new \
  -o ServerAliveInterval=2 \
  -o ServerAliveCountMax=3 \
  "unitree@$ORIN_HOST" \
  "cd /home/unitree/g1_task1 && exec ./run_task1_orin_record.sh '$OUTPUT' SET_START_REFERENCE"

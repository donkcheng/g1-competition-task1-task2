#!/usr/bin/env bash
set -euo pipefail

ORIN_WIRED_HOST="192.168.123.164"

cat <<'EOF'
这一步只执行一次，并保持网线连接：
1. SSH/管理员密码使用现场为 Orin 配置的密码（输入时不会回显）
2. 出现 Wi-Fi password 提示时，输入 iQOO 13 的热点密码（输入不会显示）
3. 成功后会打印 Orin 的无线 IP
EOF

exec ssh -tt \
  -o StrictHostKeyChecking=accept-new \
  "unitree@$ORIN_WIRED_HOST" \
  'sudo nmcli connection modify G1_TASK1_AP connection.autoconnect no; \
   sudo nmcli connection down G1_TASK1_AP >/dev/null 2>&1 || true; \
   sudo rfkill unblock wifi; \
   sudo nmcli radio wifi on; \
   sudo nmcli device set wlan0 managed yes; \
   sleep 3; \
   sudo nmcli device wifi rescan ifname wlan0; \
   sleep 4; \
   if ! nmcli -t -f SSID device wifi list ifname wlan0 | grep -Fx "iQOO 13" >/dev/null; then \
     echo "ERROR: Orin 重新扫描后仍未发现 iQOO 13" >&2; \
     exit 3; \
   fi; \
   echo "已发现 iQOO 13，现在请输入热点密码"; \
   sudo nmcli --ask device wifi connect "iQOO 13" ifname wlan0 name ORIN_iQOO; \
   ip -4 -brief address show wlan0'

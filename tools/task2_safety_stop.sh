#!/usr/bin/env bash
# Independent fail-safe invoked by a systemd timer.  Safe to run repeatedly.
set -u

stamp="$(date --iso-8601=seconds)"
log=/home/drgh/shucai/g1_competition/logs/task2_safety_stop.log
mkdir -p /home/drgh/shucai/g1_competition/logs
echo "$stamp safety stop fired" >> "$log"

# Stop the real-time body policy first, then its command/collection producers.
tmux send-keys -t g1_task2_body:deploy O 2>/dev/null || true
sleep 1
tmux send-keys -t g1_task2_body:pico C-c 2>/dev/null || true
tmux send-keys -t g1_task2_body:collector C-c 2>/dev/null || true

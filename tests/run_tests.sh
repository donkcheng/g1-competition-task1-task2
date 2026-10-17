#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m unittest discover -s "$ROOT/tests" -p 'test_core.py' -v
TASK1_VISION_PY=/home/drgh/g1_official/GR00T-WholeBodyControl/.venv_teleop/bin/python
"$TASK1_VISION_PY" -m unittest discover -s "$ROOT/tests" -p 'test_task1_start_alignment.py' -v
"$TASK1_VISION_PY" -m unittest discover -s "$ROOT/tests" -p 'test_task1_scene_alignment.py' -v
TASK2_PY=/home/drgh/g1_official/GR00T-WholeBodyControl/.venv_data_collection/bin/python
"$TASK2_PY" -m unittest discover -s "$ROOT/tests" -p 'test_task2_*.py' -v
TASK2_TELEOP_PY=/home/drgh/g1_official/GR00T-WholeBodyControl/.venv_teleop/bin/python
"$TASK2_TELEOP_PY" -m unittest discover -s "$ROOT/tests" -p 'test_arm_only_teleop.py' -v
"$TASK2_TELEOP_PY" -m unittest discover -s "$ROOT/tests" -p 'test_camera_bridge.py' -v
bash -n "$ROOT/run_preflight.sh"
bash -n "$ROOT/run_task1.sh"
bash -n "$ROOT/run_task1_align.sh"
bash -n "$ROOT/run_task1_record_aligned.sh"
bash -n "$ROOT/run_task1_scene_align.sh"
bash -n "$ROOT/run_task1_record_scene_aligned.sh"
bash -n "$ROOT/run_task1_scene_replay.sh"
bash -n "$ROOT/run_task2.sh"
bash -n "$ROOT/run_task2_camera.sh"
bash -n "$ROOT/run_task2_collect_continuous.sh"
bash -n "$ROOT/run_task2_pico_combined.sh"
bash -n "$ROOT/run_task2_right_arm_teleop.sh"
bash -n "$ROOT/run_task2_arm_collection_session.sh"
bash -n "$ROOT/run_task2_act_inference.sh"
bash -n "$ROOT/run_task2_act_auto.sh"
bash -n "$ROOT/run_task2_act_sequence.sh"
bash -n "$ROOT/run_orin_camera.sh"

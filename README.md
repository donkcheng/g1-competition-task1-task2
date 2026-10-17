# G1 Competition — Task 1 Navigation and Task 2 Cabinet Interaction

This repository contains the code used for two Unitree G1 competition tasks:

- **Task 1:** replay a visually aligned walking route over the robot Ethernet link.
- **Task 2:** collect PICO demonstrations and run three independent ACT policies for pressing a button, turning a knob, and toggling a switch.

The deployment used one right-side Dex1-1 gripper, a head Intel RealSense D435i, a right-wrist D405, an Orin computer on the G1, and an RTX 4060 workstation. The project integrates NVIDIA's GR00T Whole-Body Control repository but does not redistribute that repository or its checkpoints.

> **Safety:** This software can command a full-size humanoid. Use a safety gantry, clear the workspace, keep the physical emergency stop accessible, and test with `--dry-run` before `--real`. Never rely on software temperature warnings or `Ctrl+C` as the only safety mechanism.

## Repository contents

- `src/`: task controllers, data collection, camera servers, and ACT inference.
- `tools/`: diagnostics, Orin service management, dataset validation, and teleoperation helpers.
- `routes/`: Task 1 route recordings used during the competition.
- `config/`: checked-in templates and non-secret deployment configuration.
- `docs/`: operating, training, inference, and end-to-end reproduction guides.
- `tests/`: offline regression tests that do not command the robot.

Raw demonstrations, videos, logs, local `.env`, trained weights, Python environments, and third-party repositories are intentionally excluded from Git. See [Reproduction guide](docs/REPRODUCTION_GUIDE.md#9-assets-not-stored-in-git).

## Quick start

Clone NVIDIA's upstream repository at the pinned revision, prepare its environments, then configure this project:

```bash
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git checkout b042411fae38ee4d1af9aac82a37a1f8d14d6dd0

cd /path/to/g1_competition
cp config/task2.env.example .env
# Edit paths, network interface, camera serial, and host addresses in .env.
./run_preflight.sh
bash tests/run_tests.sh
```

Task 1 competition route:

```bash
TASK1_ORIN_HOST=192.168.123.164 \
  ./run_task1_remote.sh routes/task1_scene_v2.json RUN_TASK1
```

Task 2 no-motion rehearsal and real three-policy sequence:

```bash
# Reads live sensors and runs CUDA inference, but creates no command publishers.
bash run_task2_act_sequence.sh --dry-run

# Real robot: stop SONIC/PICO arm teleoperation first; use normal locomotion.
bash run_task2_act_sequence.sh --real
```

The real sequence automatically runs `button -> knob -> switch`. A normal per-skill action timeout releases control and advances; sensor loss, excessive body motion, tracking failure, or another control fault stops the sequence.

## Documentation

- [End-to-end reproduction guide](docs/REPRODUCTION_GUIDE.md)
- [Task 1 → Task 2 competition runbook](docs/TASK1_THEN_TASK2_COMPETITION_RUNBOOK.md)
- [ACT cloud training](docs/TASK2_ACT_CLOUD_TRAINING.md)
- [ACT inference and safety behavior](docs/TASK2_ACT_INFERENCE.md)
- [Boot-to-run checklist](docs/TASK2_ACT_BOOT_TO_RUN.md)

## Tested upstream revision

- Repository: `NVlabs/GR00T-WholeBodyControl`
- Commit: `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`

Hardware addresses, serial numbers, and absolute paths in the examples reflect the original deployment and must be changed for another robot.

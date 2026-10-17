# End-to-end reproduction guide

This guide reconstructs the software chain used for both competition tasks. It separates source code from machine-specific configuration, collected demonstrations, and model weights so the repository can be shared without publishing credentials or large binary assets.

## 1. System architecture

The tested deployment used:

- Unitree G1 in normal locomotion mode;
- G1 Orin reachable at `192.168.123.164` over dedicated Ethernet;
- workstation Ethernet `enp8s0` at `192.168.123.222/24`;
- one right Dex1-1 gripper on the Orin serial bus;
- head RealSense D435i supplied through Unitree `videohub_pc4`;
- right-wrist RealSense D405 read by a small Orin camera server;
- PICO XRoboToolkit for demonstration collection;
- an NVIDIA RTX workstation for aggregation, collection, and inference.

The original camera serial numbers and paths are examples. Replace them for another robot.

## 2. Obtain the source

```bash
git clone git@github.com:donkcheng/g1-competition-task1-task2.git g1_competition
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git GR00T-WholeBodyControl
cd GR00T-WholeBodyControl
git checkout b042411fae38ee4d1af9aac82a37a1f8d14d6dd0
```

Use NVIDIA's install scripts to create the upstream data-collection and teleoperation environments. This project expects upstream environments named `.venv_data_collection` and `.venv_teleop`. If the install path differs, update `GROOT_WBC_ROOT` in `.env` and deployment-specific absolute paths.

For ACT training on a clean CUDA server, review and run:

```bash
bash prepare_task2_act_server.sh
```

## 3. Configure the workstation

```bash
cd g1_competition
cp config/task2.env.example .env
```

Edit at least:

- `GROOT_WBC_ROOT`: pinned upstream checkout;
- `ROBOT_IFACE`: Ethernet interface connected to G1;
- `CAMERA_HOST`: workstation address on the robot subnet;
- `WRIST_CAMERA_HOST`: Orin address;
- `WRIST_CAMERA_SERIAL`: right-wrist D405 serial number;
- `DATA_ROOT`: local demonstration directory.

Do not store SSH passwords in `.env` or scripts. Launchers request the Orin password interactively. Configure Ethernet and run non-motion checks:

```bash
sudo ip link set dev enp8s0 up
sudo ip addr replace 192.168.123.222/24 dev enp8s0
ping -c 1 192.168.123.164

./run_preflight.sh
bash tests/run_tests.sh
```

## 4. Orin services and Task 1/Task 2 conflict

The original robot exposed the right Dex1-1 and LoRa/MQTT through a conflicting serial-device setup. They were used at different times:

- Task 1: LoRa/MQTT active, right-Dex1 service inactive.
- Task 2: LoRa/MQTT inactive, right-Dex1 service active.

Task 2 launchers call `tools/prepare_orin_task2.py` for this switch. Restore LoRa before Task 1 according to the competition runbook. Never run both task controllers simultaneously.

The right-wrist camera is managed with:

```bash
python3 tools/manage_orin_wrist_camera.py start
./run_task2_camera.sh start
./run_task2_camera.sh status
```

The head D435i remains owned by Unitree's video service. Do not open its `/dev/video*` node directly while `videohub_pc4` is using it.

## 5. Reproduce Task 1

Task 1 is route replay, not a learned policy. The competition route is `routes/task1_scene_v2.json`; its start-view reference is under `config/`.

1. Put G1 in the expected normal locomotion state.
2. Restore Task 1 LoRa and ensure no Task 2 controller is running.
3. Place G1 near the demonstrated start pose with a matching camera view.
4. Keep the physical emergency stop accessible.
5. Run:

```bash
TASK1_ORIN_HOST=192.168.123.164 \
  ./run_task1_remote.sh routes/task1_scene_v2.json RUN_TASK1
```

The launcher checks visual start offset, applies heading correction, and replays the route. `Ctrl+C` requests a stop, but the physical emergency stop remains primary.

## 6. Reproduce Task 2 collection

Task 2 uses three independent skills and datasets: button press-and-hold, knob rotation of at least 15 degrees, and switch motion of at least 15 degrees with a hold.

PICO/XRoboToolkit is required for demonstration collection, not autonomous ACT inference. Confirm the headset, both controllers, and body trackers continuously stream before teleoperation.

```bash
# Terminal A: SONIC whole-body teleoperation
bash run_task2_body_teleop.sh --real

# Terminal B: Orin services, cameras, PICO manager, and ACT-format collection
bash run_task2_collection_session.sh button --real
# Replace button with knob or switch.
```

Controls:

- left grip + A: start/stop and save an episode;
- left grip + B: discard the current attempt;
- terminal `s`, `x`, `q` + Enter: record, discard, and quit fallbacks.

Episodes store synchronized G1 state, right-arm targets, right Dex1 state/target, head RGB, and right-wrist RGB in a LeRobot-style dataset. Stale camera, robot, or gripper telemetry causes the current attempt to be rejected.

The original final split contained 17 button, 13 knob, and 9 switch episodes. This is a small competition dataset, not a generally robust benchmark. Another cabinet should use more demonstrations and varied start offsets.

```bash
python3 tools/verify_task2_act_dataset.py --help
```

## 7. Train the ACT policies

Copy validated datasets to the training machine, configure `config/task2_act_train.env.example`, then train separately:

```bash
bash train_task2_act.sh button
bash train_task2_act.sh knob
bash train_task2_act.sh switch
```

See [TASK2_ACT_CLOUD_TRAINING.md](TASK2_ACT_CLOUD_TRAINING.md) for server setup and transfer. Deployment expects:

```text
models/task2_act/
├── SHA256SUMS
├── button/{config.json,train_config.json,model.safetensors}
├── knob/{config.json,train_config.json,model.safetensors}
└── switch/{config.json,train_config.json,model.safetensors}
```

Weights are not committed. Restore and verify them with:

```bash
(cd models/task2_act && sha256sum -c SHA256SUMS)
```

## 8. Validate and run ACT inference

Inference does not need PICO. Stop SONIC and PICO arm teleoperation, use normal locomotion to hold the body, place G1 at the demonstrated cabinet-relative pose, and keep the emergency stop accessible.

Required live no-command rehearsal:

```bash
bash run_task2_act_sequence.sh --dry-run
```

This reads G1, gripper, and both cameras and executes CUDA inference, but creates no arm or gripper command publishers.

Single staged-takeover check:

```bash
bash run_task2_act_auto.sh button --real \
  --preposition-only --exit-after-attempt
```

Competition sequence:

```bash
bash run_task2_act_sequence.sh --real
```

The order is `button -> knob -> switch`. Before policy output is applied, the controller holds measured pose and ramps authority. It monitors body travel/tilt, tracking, sensor freshness, and command speed. A normal 30-second action timeout releases and advances; sensor or control faults release and stop. `Ctrl+C`, `x` + Enter, or `q` + Enter requests software release.

Settings are in `config/task2_act_deploy.json`. Do not loosen body-motion or tracking guards merely to hide a deployment problem.

## 9. Assets not stored in Git

| Asset | Local location | Recovery |
|---|---|---|
| Machine config | `.env` | Copy `config/task2.env.example` and edit |
| Raw demonstrations/videos | `data/` | Restore archive or recollect |
| ACT weights | `models/task2_act/*/model.safetensors` | Restore checkpoints; verify checksums |
| Runtime logs | `logs/` | Regenerated on run |
| Camera captures | `captures/` | Regenerated by camera tools |
| NVIDIA/Unitree code | external checkout | Clone/install from official source |
| Python environments | upstream `.venv_*` | Recreate with pinned upstream scripts |

## 10. Troubleshooting

- **Cannot reach Orin:** verify boot, cable, interface, and `192.168.123.222/24`.
- **SSH authentication fails:** use the password configured on that Orin; none is embedded here.
- **Camera port in use:** stop the stale managed service, then restart it instead of launching a second server.
- **PICO connected but no frames:** wake/wear it, keep XRoboToolkit in front, and enable Tracking/Send.
- **ACT exits before motion:** inspect `logs/task2_act_*.log`; do not bypass stale-sensor, body-motion, or tracking guards.
- **Unexpected body motion:** stop, confirm SONIC/PICO controllers are gone, return to normal locomotion, and ensure only one process owns the arm SDK.
- **High temperature:** allow G1 to cool. Scripts report warnings while onboard firmware protection remains active.

Reproduction is complete only after offline tests, live dry-run, staged takeover, and a guarded real-robot test pass on the target hardware.

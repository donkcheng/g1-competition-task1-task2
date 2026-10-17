# Task 2 ACT cloud training

The cloud server only performs offline training. PICO, SONIC, the cameras, and
the robot remain on the competition computer.

## Recommended server

- Ubuntu 22.04, NVIDIA driver already working, at least 16 GB VRAM;
- 24 GB VRAM (RTX 4090/L40S/A5000-class) is preferred for batch size 8;
- at least 150 GB SSD and a stable SSH connection;
- rent three GPUs only if all three policies must train concurrently. One GPU
  can train them sequentially.

## One-time server setup

Copy this `g1_competition` directory to the server, then run:

```bash
cd /workspace/g1_competition
TASK2_WBC_ROOT=/workspace/GR00T-WholeBodyControl bash prepare_task2_act_server.sh
cp config/task2_act_train.env.example config/task2_act_train.env
```

Edit only the paths and batch size in `config/task2_act_train.env`. The setup is
pinned to WholeBodyControl commit `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`;
its data-collection extra pins the matching LeRobot revision.

## Upload and validate data

Copy all three complete dataset directories without renaming them:

```text
/workspace/task2_data/task2_button_act
/workspace/task2_data/task2_knob_act
/workspace/task2_data/task2_switch_act
```

Do not copy a dataset while its collector is saving an episode. After upload:

From the competition computer, the resumable helper validates all three
datasets first and then sends only the required code/data:

```bash
cd /home/drgh/shucai/g1_competition
./sync_task2_to_server.sh user@server-ip
```

Alternatively, copy the directories with the cloud provider's upload tool.
After upload, validate again on the server:

```bash
/workspace/GR00T-WholeBodyControl/.venv_data_collection/bin/python \
  tools/verify_task2_act_dataset.py /workspace/task2_data/task2_button_act
```

Repeat for `knob` and `switch`. Training refuses to start unless this check
passes, including real decoding of both camera videos.

## Train

One GPU, sequential:

```bash
./train_task2_act.sh button
./train_task2_act.sh knob
./train_task2_act.sh switch
```

Three GPUs, concurrent (three terminals):

```bash
CUDA_VISIBLE_DEVICES=0 ./train_task2_act.sh button
CUDA_VISIBLE_DEVICES=1 ./train_task2_act.sh knob
CUDA_VISIBLE_DEVICES=2 ./train_task2_act.sh switch
```

The default is 50,000 updates with a checkpoint every 5,000. Test an early
checkpoint on the robot before spending time on 100,000 updates. If a 16 GB GPU
runs out of memory, set `TASK2_BATCH_SIZE=4`; for 8-12 GB, start with 2.

To resume after a server interruption, locate the latest checkpoint's
`pretrained_model/train_config.json` and run:

```bash
./resume_task2_act.sh /path/to/checkpoint/pretrained_model/train_config.json
```

Image augmentation covers lighting/color changes. Small robot-to-panel position
offsets must still be present in the demonstrations because arbitrary geometric
image augmentation would not transform the corresponding joint actions.

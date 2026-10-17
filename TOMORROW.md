# 2026-10-12 执行清单

> 任务 2 的手型、相机方案已更新：实机为单只右侧 Dex1-1，头部 D435i 彩色画面经电脑端 videohub 桥接。下文关于双 Dex3、OAK 和 Orin `:5555` 的步骤是旧计划，**不要用于当前实机**。请以 [README.md](README.md) 和 `.env` 的当前配置为准。

## 0. 出发前必须确认

- G1 是 29DoF，且左右均安装 Dex3/ThreeFinger（每手 7 个关节）。否则 NVIDIA 当前官方 PICO exporter 不能直接用于任务 2。
- G1 Orin 上至少有一台 OAK 头部相机；记下 MxID、Orin 用户名和 IP。
- PICO 4/4 Pro、两个手柄、两个脚踝 tracker 均到位。
- 带网线；笔记本有原生网口，不需要 USB-C 转网口。
- 全程使用吊架/保护绳、实体急停和独立安全员。

## 1. 工作站一次性准备

官方仓库已经准备在 `/home/drgh/g1_official/GR00T-WholeBodyControl`；不要使用旧的 `/home/drgh/GR00T-WholeBodyControl`，也不要重新克隆。

运行环境由本机准备阶段完成。若环境目录不存在，只按下面顺序补装；PICO 环境跳过无关的 MuJoCo 仿真依赖，再单独安装任务 1 所需 SDK2：

```bash
export TensorRT_ROOT=/home/drgh/NVIDIA/TensorRT
cd /home/drgh/g1_official/GR00T-WholeBodyControl
SKIP_SIM_AND_UNITREE=1 bash install_scripts/install_pico.sh
source .venv_teleop/bin/activate
uv pip install -e external_dependencies/unitree_sdk2_python
deactivate
bash install_scripts/install_data_collection.sh
cd gear_sonic_deploy
./scripts/install_deps.sh
source scripts/setup_env.sh
just build
cd ..
python download_from_hf.py --sonic-v1-1
```

若要先跑仿真数采，再额外执行 `bash install_scripts/install_mujoco_sim.sh`。不要执行 `download_from_hf.py --training`，会额外下载约 30 GB SMPL 数据。安装会创建独立环境，先保证至少 30 GiB 空闲。

XRoboToolkit PC Service 已从官方 Ubuntu 22.04 包解压到用户目录，不需要 sudo。启动并检查：

```bash
cd /home/drgh/g1_competition
./run_xrobotoolkit.sh start
./run_xrobotoolkit.sh status
```

PICO 与工作站连同一 Wi-Fi，在 PICO 应用中填写工作站 Wi-Fi IPv4，确认状态为 `WORKING`。

编辑本工程配置：

```bash
cd /home/drgh/g1_competition
cp config/task2.env.example .env
nano .env
```

必须修改：`GROOT_WBC_ROOT=/home/drgh/g1_official/GR00T-WholeBodyControl`、`ROBOT_IFACE`、`HAND_TYPE=dex3`。确认 `TENSORRT_ROOT` 与实机一致。

给专用有线网口配置同网段地址（将网卡名替换成实际值，不要给这个接口设置默认网关）：

```bash
sudo ip link set <有线网卡> up
sudo ip addr add 192.168.123.222/24 dev <有线网卡>
```

## 2. Orin 相机

在 Orin 上安装同版官方仓库，然后执行官方安装：

```bash
cd /path/to/GR00T-WholeBodyControl
bash install_scripts/install_camera_server.sh
```

如果不用 systemd，手工启动：

```bash
GROOT_WBC_ROOT=/path/to/GR00T-WholeBodyControl \
EGO_MXID=<头部OAK的MxID> \
./run_orin_camera.sh start
```

若脚本在工作站，先用 `scp` 复制到 Orin。工作站验证：

```bash
./run_task2.sh check --real --network
```

必须看到 `192.168.123.164:5555` 可达；腕相机未安装时 `RECORD_WRIST_CAMERAS=0`。

## 3. 任务 1：录路线并闭环回放

任务 1 使用 Unitree SDK2 高层速度控制，不启动 SONIC。先确保任务 2 的 tmux 已停止。

使用官方 teleop 环境中的 SDK2 Python：

```bash
export TASK1_PYTHON=/home/drgh/g1_official/GR00T-WholeBodyControl/.venv_teleop/bin/python
cd /home/drgh/g1_competition
./run_task1.sh preflight --iface <有线网卡> --connect
```

确认能持续收到 `rt/odommodestate` 后，让人用原厂遥控器走一遍；程序只记录，不发运动命令。在机器人中心位于三个白贴片上方时分别按 `1`、`2`、`3`，按 `P` 保存：

```bash
./run_task1.sh record \
  --real --passive \
  --iface <有线网卡> \
  --output routes/task1_20261012.json

./run_task1.sh validate \
  --route routes/task1_20261012.json \
  --require-markers 3
```

先做不发真机命令的回放测试：

```bash
./run_task1.sh replay --route routes/task1_20261012.json
```

吊架、空旷区域和安全员就位后才运行真机：

```bash
./run_task1.sh replay \
  --real --confirm RUN_TASK1 \
  --iface <有线网卡> \
  --route routes/task1_20261012.json
```

`Ctrl+C` 停止并发送零速度。若里程数据过期，程序会拒绝继续。当前版本用相对里程到达三个记录点；白贴片视觉确认未接入，不能把它误写成已完成。

## 4. 任务 2：PICO 数采流程跑通

先只采每项 1 条约 10 秒的测试 episode，不采正式数据。

```bash
cd /home/drgh/shucai/g1_competition
./run_xrobotoolkit.sh start
python3 tools/manage_orin_wrist_camera.py start
./run_task2_camera.sh start
```

先在安全员可见的 `g1_task2_body:deploy` 窗口启动
`run_task2_body_teleop.sh --real`，等待 `Init Done` 且机器人稳定。
然后启动：

```bash
bash run_task2_pico_combined.sh --real
```

该脚本会自动注入一次 `A+B+X+Y` 进入 PLANNER，5 秒后自动注入
左摇杆按下（L3）进入 `VR_3PT`。这 5 秒内 PICO 操作员保持中立
标定姿势，不要再手动按 ABXY、L3 或其他模式组合键。

确认身体跟随与右扳机夹爪均正常后，另开终端：

```bash
./run_task2.sh collect-act button
```

只采一条按钮数据时：

1. `Left Grip+A` 开始录制，等采集终端显示 `RECORDING`。
2. 靠近按钮，按下并保持至少 2 秒，然后退回初始姿势。
3. 再按一次 `Left Grip+A` 停止并保存，必须看到 `SAVED episode`。
4. 任何动作错误、遮挡或温度告警都按 `Left Grip+B` 丢弃本次。
5. 右扳机只控制右 Dex1：按下闭合，松开张开。

保存后检查数据：

```bash
/home/drgh/g1_official/GR00T-WholeBodyControl/.venv_data_collection/bin/python \
  tools/verify_task2_act_dataset.py data/task2_button_act
```

## 5. 明天的完成标准

- 任务 1：SDK/里程预检通过；录得含 3 个标记的路线；吊架下完成一次闭环回放。
- 任务 2：按钮、旋钮、开关各得到 1 个非空 LeRobot v2.1 测试数据集；`verify` 检查通过。
- 不训练模型；不下载训练 SMPL 数据；不复制任何已有 checkpoint/conda 环境。

注意：NVIDIA 当前这条数据链输出的是 `UNITREE_G1_SONIC`/GR00T 所需数据，不能不做动作字段适配就直接当作 ACT 数据集。

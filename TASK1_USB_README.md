# G1 任务1 USB 包

本目录只包含任务1（三点巡航）所需的程序、配置、有效路线和 Unitree SDK2 Python 源码，不包含任务2和相机照片。

## 路线选择

- 主路线：`routes/task1_fsm501_take2.json`
- 备用路线：`routes/task1_fsm501_take1.json`
- 两条路线均使用三腰运控 `FSM=501` 录制，并包含 `point_1 -> point_2 -> point_3`。

## 在原笔记本运行

```bash
cd task1
./run_task1.sh preflight --iface enp8s0 --connect
./run_task1.sh replay \
  --real --confirm RUN_TASK1 \
  --iface enp8s0 \
  --route routes/task1_fsm501_take2.json \
  --require-markers 3
```

预检必须显示 `FSM=501`。运行前必须确认防跌倒保护、遥控器急停和安全员就位。

## 复制到 Orin 后运行

不要直接在 U 盘中运行。先把整个 `task1` 目录复制到 Orin 用户目录。

如果 Orin 已有能导入 `unitree_sdk2py` 的 Python 环境：

```bash
cd ~/task1
TASK1_PYTHON=/path/to/python ./run_task1.sh preflight --iface <机器人内部网卡> --connect
```

如果没有 Unitree SDK2 Python，可使用包内源码安装：

```bash
python3 -m pip install ./vendor/unitree_sdk2_python
```

该安装仍需要环境中已有或能够安装 `cyclonedds==0.10.2`、`numpy` 和 `opencv-python`。

## 当前限制

当前程序使用里程计闭环跟踪录制路线，不使用相机识别黄色圆片。机器人必须从与录制时一致的起点、朝向和三腰运控状态启动。

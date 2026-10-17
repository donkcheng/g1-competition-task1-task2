# G1 比赛现场速查：先任务1，再任务2

本文只描述比赛执行顺序。任务1和任务2不得同时运行，中间必须由人工确认机器人停稳并重新摆位。

## 0. 共同准备

1. 网线连接电脑与 G1，机器人进入常规运控站立模式。
2. 原厂遥控器和急停可立即触及，周围人员离开运动范围。
3. 电脑配置机器人网口：

```bash
sudo ip link set dev enp8s0 up
sudo ip addr replace 192.168.123.222/24 dev enp8s0
ping -c 2 192.168.123.164
```

4. 确认没有遗留控制器。下面命令应无实际控制进程输出：

```bash
pgrep -af 'task1_route.py|g1_deploy_onnx_ref|task2_right_arm_teleop.py|task2_act_inference.py'
```

若有输出，回到启动它的终端用 `Ctrl+C` 正常退出；不要启动 SONIC 或 PICO 遥操。

## 1. 任务1前切回 LoRa/MQTT 服务

任务1前停止右 Dex1 服务并恢复 LoRa/MQTT：

```bash
ssh unitree@192.168.123.164
sudo systemctl stop g1-dex1-right.service
sudo systemctl restart lora2mqtt.service
systemctl is-active g1-dex1-right.service
systemctl is-active lora2mqtt.service
exit
```

预期依次看到：

```text
inactive
active
```

## 2. 执行任务1

将机器人放到任务1保存的参考起点附近并大致对准。主路线使用昨天已经多次回放的 Orin 路线 `task1_scene_v2.json`：

```bash
cd /home/drgh/g1_competition
./run_task1_remote.sh routes/task1_scene_v2.json RUN_TASK1
```

该命令通过网线 SSH 到 Orin；Orin 使用自身 `eth0` 执行起点场景检查和路线回放。密码提示时输入 Orin 密码。

正常完成时终端应出现：

```text
路线完成，终点平面误差约 ... m
```

紧急停止优先使用原厂遥控器急停；程序正常停止使用 `Ctrl+C`。

如果 Orin 上的主路线无法启动，可使用电脑本地备用路线：

```bash
cd /home/drgh/g1_competition
./run_task1_scene_replay.sh routes/task1_scene_final.json RUN_TASK1
```

主路线与备用路线二选一，绝不能同时运行。

## 3. 任务1到任务2的人工转场

1. 确认任务1终端已经退出，机器人完全停稳。
2. 保持机器人处于常规运控站立模式；不要启动 SONIC。
3. 将机器人移动到任务2示教时相对于电箱的起始站位和朝向。
4. 确认头部相机和右腕 D405 没有遮挡，右夹爪开合路径清空。
5. 再次检查没有遗留程序：

```bash
pgrep -af 'task1_route.py|g1_deploy_onnx_ref|task2_right_arm_teleop.py|task2_act_inference.py'
```

不要把两个任务写成无人工停顿的一键连续执行：两项任务的站位不同，必须人工转场。

## 4. 执行任务2

任务2不需要 PICO。ACT 启动器会自动停止 `lora2mqtt`、启动右 Dex1、启动双相机并校验三个模型：

上场前的零动作全链路演练：

```bash
cd /home/drgh/g1_competition
bash run_task2_act_sequence.sh --dry-run
```

确认结尾出现 `ZERO COMMANDS` 后，先在正式任务2站位只验证渐进接管并进入训练起点分布：

```bash
cd /home/drgh/g1_competition
bash run_task2_act_inference.sh button --real --preposition-only --exit-after-attempt
```

看到 `REAL ready` 后输入 `s` 回车。这个模式不会执行 ACT 动作；它会缓慢接管、以最高 `0.12 rad/s` 靠近按钮示教中位姿态，并在实测右臂和夹爪进入训练第一帧的第 10～90 百分位范围、稳定 1 秒后自动释放。该步骤平稳通过后，再单独验证按钮模型：

```bash
bash run_task2_act_inference.sh button --real --max-run-seconds 15
```

单模型动作方向和稳定性均确认正常后，才执行完整真机序列：

```bash
cd /home/drgh/g1_competition
bash run_task2_act_sequence.sh --real
```

密码提示时输入 Orin 密码。程序自动依次运行：

```text
button -> knob -> switch
```

每个模型优先在检测到右臂完成“离开起点—动作—返回并稳定”的运动周期后进入下一项。单项达到动作时限时会安全释放、记录失败并继续下一项；传感器失联、身体异常运动、跟踪严重超差等故障仍会停止整个序列。

注意：当前完成检测确认的是机器人运动周期，不是电箱按钮、旋钮或拨杆的物理状态。现场仍需目视确认实际交互成功。

正常停止使用 `Ctrl+C`，程序会平滑释放 Arm SDK；异常运动立即使用原厂急停。

## 5. 全部结束后的服务状态

任务2退出后，相机桥和腕部相机进程会自动停止，但通常保留：

```text
g1-dex1-right.service: active
lora2mqtt.service: inactive
```

若之后还需要再次执行任务1，重新执行第1节的服务切换即可。

## 最短命令清单

```bash
# 任务1服务切换
ssh unitree@192.168.123.164
sudo systemctl stop g1-dex1-right.service
sudo systemctl restart lora2mqtt.service
exit

# 任务1（电脑终端）
cd /home/drgh/g1_competition
./run_task1_remote.sh routes/task1_scene_v2.json RUN_TASK1

# 人工确认停稳并重新摆到任务2站位后，任务2（电脑终端）
cd /home/drgh/g1_competition
bash run_task2_act_sequence.sh --real
```

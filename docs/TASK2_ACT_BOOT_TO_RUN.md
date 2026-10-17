# G1 任务2：开机到 ACT 模型运行的完整链路

## 1. 机器人现场准备

1. 确认右 Dex1-1、头部相机、右腕 D405 和网线连接牢固，线缆不会进入手臂或夹爪运动范围。
2. G1 挂在保护吊架上，吊带留有余量；周围清空，原厂遥控器急停可立即触及。
3. 给 G1 上电，等待 Orin 和运动控制系统完整启动。
4. 使用原厂遥控器让 G1 进入**常规运控站立模式**并确认稳定。运行 ACT 时不要退出常规运控。
5. 不启动 SONIC，不启动 PICO 手臂遥操。ACT 推理不需要 PICO。

## 2. 电脑网络与 GPU

电脑可以继续用 Wi-Fi/VPN 联网；机器人通信固定走有线网卡 `enp8s0`。

```bash
sudo ip link set dev enp8s0 up
sudo ip addr replace 192.168.123.222/24 dev enp8s0

ip -4 addr show dev enp8s0
ping -c 2 192.168.123.164
nvidia-smi
```

必须看到：

- `enp8s0` 具有 `192.168.123.222/24`
- `192.168.123.164` 可以 ping 通
- `nvidia-smi` 能看到 RTX 4060

## 3. 放置比赛起始位姿

把机器人放到与采集示教时相近的电柜相对位置和朝向。两个相机必须同时能看到相应工作区域。ACT 只控制右臂和右夹爪，不会自动移动机器人底座来修正很大的站位偏差。

确认没有遗留控制程序：

```bash
pgrep -af 'g1_deploy_onnx_ref|task2_right_arm_teleop.py|task2_act_inference.py'
```

没有输出才继续；如果有输出，回到对应终端用 `Ctrl+C` 正常停止。

## 4. 零动作链路检查

```bash
cd /home/drgh/shucai/g1_competition
bash run_task2_act_sequence.sh --dry-run
```

该模式读取真实 G1、Dex1 和双相机数据并运行三个 CUDA 模型，但不会创建手臂或夹爪命令发布器。

## 5. 首次真机只验证渐进接管

不要从常规运控直接运行完整三动作序列。先只验证按钮模型的接管和起点对齐：

```bash
bash run_task2_act_inference.sh button --real --preposition-only --exit-after-attempt
```

看到 `REAL ready` 后输入 `s` 回车。程序只会：

1. 保持当前实测姿态并缓慢取得 0.25 的 Arm SDK 权重；
2. `ARMING` 阶段固定当前右臂和 Dex1 目标，先将权重和增益平稳升至终值并连续稳定 1 秒；默认约 7.5 秒，不是卡住；
3. 权重与增益保持不变，以最高 `0.12 rad/s` 靠近按钮示教的中位起始姿态；
4. 实测右臂和 Dex1 进入按钮训练集第一帧的第 10～90 百分位范围（含 `0.05 rad` 实机阻抗跟踪裕量）并稳定 1 秒后，平滑释放并退出。

这个模式不会把 ACT 输出发给机器人。若此阶段仍出现突跳、抖动或失衡，立即急停，不得继续单模型或三模型动作。

## 6. 单模型动作验证

仅在第 5 节平稳完成后运行：

```bash
bash run_task2_act_inference.sh button --real --max-run-seconds 15
```

看到 `REAL ready` 后输入 `s` 回车。按钮单模型平稳且动作方向正确后，才进入完整序列。

## 7. 完整三动作序列

```bash
bash run_task2_act_sequence.sh --real
```

提示 Orin 密码时输入现场为机器人配置的 SSH 密码，终端不会回显密码。不要把密码写入脚本或提交到 Git。

启动器自动完成：

1. 检查有线 IP、Orin 连通性、RTX 4060/CUDA。
2. 确认 SONIC、PICO 手臂遥操和其他 ACT 推理进程未运行。
3. SSH 登录 `unitree@192.168.123.164`。
4. 临时停止 `lora2mqtt.service`，启动 `g1-dex1-right.service`，确认右 Dex1 电机 ID 0。
5. 在 Orin 重启右腕 D405 服务，端口为 `5559`。
6. 在电脑启动双相机桥 `192.168.123.222:5555`。
7. 验证 `ego_view` 与 `right_wrist` 均为 `480×640×3` 的新鲜画面。
8. 校验三个模型权重的 SHA256。
9. 加载并预热按钮模型，执行渐进接管并进入训练起点分布，然后才开始 ACT 动作。
10. 检测到完整“伸出 → 交互 → 返回 → 稳定”运动周期后释放右臂，等待 3 秒。
11. 自动切换旋钮模型，再自动切换拨杆模型；每个模型都会重新渐进接管并慢速进入自己的训练起点分布。
12. 任一步超过 30 秒仍未完成，或者相机、状态、推理、关节速度/跟随异常，立即终止整个序列，不进入下一模型。

## 8. 正常终端输出

启动阶段应依次看到类似内容：

```text
g1-dex1-right.service: active
lora2mqtt.service: inactive
Right Dex1 motor ID 0 detected and server started.
Right wrist D405 ... listening on 0.0.0.0:5559
Dual cameras ready: {'ego_view': (480, 640, 3), 'right_wrist': (480, 640, 3)}
button/model.safetensors: 成功
[ACT] Model ready: device=cuda ...
[ACT] Attempt 1: staged takeover ...
[ACT] TAKEOVER stable; slowly approaching ... demonstration start pose.
[ACT] Demonstration start pose stable ...; computing first action chunk.
[ACT] Attempt 1 RUNNING ...
```

运行状态中的：

```text
cycle=max.../now.../stable...s
```

表示右臂最大离开距离、当前距起始姿态的距离和返回后的稳定时间。

## 9. 停止与故障处理

随时停止：

```text
Ctrl+C
```

`Ctrl+C` 会立即发送 Arm SDK 权重 0 并退出，总序列不会继续下一动作；它不会再等待约 1 秒的渐退。输入 `x` 或 `q` 后回车则使用正常的平滑释放。

不要直接关闭终端或断开电脑电源。`Ctrl+C` 仍依赖电脑、网线和 DDS 正常，发生剧烈异常运动时优先使用原厂急停，同时按 `Ctrl+C`。

常见错误：

- `enp8s0 does not have ...`：重新执行第 2 节的两条 `sudo ip` 命令。
- `Cannot reach G1 Orin`：确认 G1 完成启动、网线正常、IP 未变化。
- `CUDA is unavailable`：先让 `nvidia-smi` 恢复正常，再运行模型。
- `Dual-camera readiness timed out`：检查头部相机、腕部 D405 和 Orin `5559` 服务。
- `completion was not detected before 30s timeout`：本动作未完成，控制会先安全释放；自动三动作入口会记录失败并进入下一项，单动作入口则退出。应检查站位、画面与训练分布后再重试。

## 10. 单独运行某个模型

需要单独测试时：

```bash
bash run_task2_act_auto.sh button --real
bash run_task2_act_auto.sh knob --real
bash run_task2_act_auto.sh switch --real
```

单模型启动器完成预检查后直接开始，不需要输入 `s`。

## 11. 切回任务1/LoRa

任务2会停止 `lora2mqtt`，因为它与右 Dex1 共用 `/dev/ttyUSB0`。比赛中准备切回任务1时：

```bash
ssh unitree@192.168.123.164
sudo systemctl stop g1-dex1-right.service
sudo systemctl restart lora2mqtt.service
systemctl is-active lora2mqtt.service
exit
```

`systemctl is-active` 应输出 `active`。切回任务2时无需手工反向操作，任务2启动器会自动完成。

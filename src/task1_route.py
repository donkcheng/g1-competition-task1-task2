#!/usr/bin/env python3
"""Safe route recorder/follower for a fixed-start G1 inspection task.

The real backend is loaded lazily, so validation and dry-runs have no external
dependencies. Real motion requires two explicit flags and every exit path sends
a short-lived zero velocity command.
"""

from __future__ import annotations

import argparse
import json
import math
import select
import shutil
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    value = getattr(obj, name, default)
    return value() if callable(value) else value


def motion_fsm_block_reason(fsm_id: int) -> str | None:
    if fsm_id == 501:
        return None
    if fsm_id == 4:
        return (
            "FSM=4（锁定站立）尚未进入可行走运控，禁止回放路线。"
            "请在机器人已做好防跌倒保护时，按本机铭牌操作进入 R1+X 单腰常规运控"
        )
    if fsm_id == 802:
        return (
            "FSM=802；该状态此前调用 SetVelocity 已超时，禁止直接回放整条路线。"
            "请先确认 R1+X（单腰常规运控）已被机器人接受，再进行受控的短距离点动测试"
        )
    return f"FSM={fsm_id} 尚未经过本机点动验证；整路回放仅允许已验证的 FSM=501 三腰运控"


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float
    stamp: float


class MockBackend:
    def __init__(self) -> None:
        self._pose = Pose2D(0.0, 0.0, 0.0, time.monotonic())
        self._cmd = (0.0, 0.0, 0.0)
        self._last = time.monotonic()

    def _integrate(self) -> None:
        now = time.monotonic()
        dt = min(0.2, now - self._last)
        vx, vy, wz = self._cmd
        c, s = math.cos(self._pose.yaw), math.sin(self._pose.yaw)
        self._pose = Pose2D(
            self._pose.x + (c * vx - s * vy) * dt,
            self._pose.y + (s * vx + c * vy) * dt,
            wrap_angle(self._pose.yaw + wz * dt),
            now,
        )
        self._last = now

    def set_velocity(self, vx: float, vy: float, wz: float) -> None:
        self._integrate()
        self._cmd = (vx, vy, wz)

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0, 0.0)

    def pose(self) -> Pose2D:
        self._integrate()
        return self._pose

    def state_age(self) -> float:
        return 0.0


class UnitreeBackend:
    """Thin SDK2 adapter. It deliberately uses expiring SetVelocity requests."""

    def __init__(
        self,
        interface: str,
        state_topic: str,
        command_duration: float = 0.5,
        *,
        enable_motion: bool = True,
        query_control_status: bool = False,
        allow_blocked_fsm_probe: bool = False,
        loco_service: str = "sport",
        initialize_channel: bool = True,
        velocity_no_reply: bool = False,
    ):
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        except ImportError as exc:
            raise RuntimeError(
                "unitree_sdk2py is not installed in this Python environment; "
                "install the official Unitree SDK2 Python package first"
            ) from exc

        if initialize_channel:
            ChannelFactoryInitialize(0, interface)
        self._client = None
        self._motion_enabled = enable_motion
        self._consecutive_velocity_timeouts = 0
        self.fsm_id: int | None = None
        self.fsm_mode: int | None = None
        self.motion_owner: dict[str, Any] | None = None
        if enable_motion or query_control_status:
            # Current G1 firmware uses the "sport" RPC service. Keep the
            # override for an explicitly selected legacy service.
            from unitree_sdk2py.g1.loco import g1_loco_client

            g1_loco_client.LOCO_SERVICE_NAME = loco_service
            self._client = g1_loco_client.LocoClient()
            self._client.SetTimeout(2.0)
            self._client.Init()
            if hasattr(self._client, "GetFsmId"):
                code, fsm_id = self._client.GetFsmId()
            else:
                # NVIDIA's pinned SDK registers API 7001 but omits its small
                # convenience wrapper. Calling the registered read-only RPC is
                # compatible with both that SDK and the current Unitree SDK.
                code, payload = self._client._Call(7001, "{}")
                fsm_id = None if code != 0 else json.loads(payload).get("data")
            if code != 0 or fsm_id is None:
                raise RuntimeError(f"G1 GetFsmId failed with code {code}")
            self.fsm_id = int(fsm_id)
            mode_code, mode_payload = self._client._Call(7002, "{}")
            if mode_code == 0:
                mode_data = json.loads(mode_payload).get("data")
                if mode_data is not None:
                    self.fsm_mode = int(mode_data)
            try:
                from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
                    MotionSwitcherClient,
                )

                switcher = MotionSwitcherClient()
                switcher.SetTimeout(2.0)
                switcher.Init()
                owner_code, owner = switcher.CheckMode()
                if owner_code == 0:
                    self.motion_owner = owner
            except Exception:
                # Some older G1 firmware does not expose motion_switcher.
                self.motion_owner = None
            if enable_motion:
                if self.motion_owner is not None and self.motion_owner.get("name") != "ai":
                    raise RuntimeError(
                        f"motion owner is {self.motion_owner!r}, not the verified ai controller"
                    )
                blocked = motion_fsm_block_reason(self.fsm_id)
                if blocked and not allow_blocked_fsm_probe:
                    raise RuntimeError(blocked)
        self._duration = command_duration
        self._velocity_no_reply = velocity_no_reply
        self._lock = threading.Lock()
        self._absolute: Pose2D | None = None
        self._origin: Pose2D | None = None
        self._source_topic: str | None = None

        def make_callback(topic: str):
            def callback(message: Any) -> None:
                position = get_field(message, "position")
                imu = get_field(message, "imu_state")
                rpy = get_field(imu, "rpy") if imu is not None else None
                if position is None or rpy is None or len(position) < 2 or len(rpy) < 3:
                    return
                values = (float(position[0]), float(position[1]), float(rpy[2]))
                if not all(math.isfinite(value) for value in values):
                    return
                pose = Pose2D(*values, time.monotonic())
                with self._lock:
                    if self._source_topic is not None and self._source_topic != topic:
                        return
                    self._source_topic = topic
                    self._absolute = pose
                    if self._origin is None:
                        self._origin = pose

            return callback

        topics = ["rt/odommodestate", "rt/lf/odommodestate"] if state_topic == "auto" else [state_topic]
        self._subscribers = []
        for topic in topics:
            subscriber = ChannelSubscriber(topic, SportModeState_)
            subscriber.Init(make_callback(topic), 10)
            self._subscribers.append(subscriber)

    def set_velocity(self, vx: float, vy: float, wz: float) -> None:
        if self._client is None or not self._motion_enabled:
            raise RuntimeError("motion client is disabled in read-only mode")
        if getattr(self, "_velocity_no_reply", False):
            parameter = json.dumps(
                {
                    "velocity": [float(vx), float(vy), float(wz)],
                    "duration": self._duration,
                }
            )
            # Streaming velocity does not need a response for every packet.
            # Unitree's RPC layer exposes this mode for motion commands; it
            # avoids blocking the follower on an occasionally lost ACK.  Each
            # command still expires after _duration, so loss of the process or
            # network cannot leave an unlimited command active.
            result = 3102
            for attempt in range(3):
                result = self._client._CallNoReply(7105, parameter)
                if result == 0:
                    self._consecutive_velocity_timeouts = 0
                    return
                if result == 3102 and attempt < 2:
                    time.sleep(0.03)
                    continue
                break
            raise RuntimeError(f"G1 velocity packet send failed with code {result}")
        result = self._client.SetVelocity(float(vx), float(vy), float(wz), self._duration)
        if result == 3104:
            self._consecutive_velocity_timeouts += 1
            print(
                f"[WARN] SetVelocity RPC timeout "
                f"({self._consecutive_velocity_timeouts}/3); closed-loop follower will retry",
                file=sys.stderr,
            )
            if self._consecutive_velocity_timeouts >= 3:
                raise RuntimeError("G1 SetVelocity timed out 3 consecutive times (code 3104)")
            return
        if result not in (None, 0):
            raise RuntimeError(f"G1 SetVelocity failed with code {result}")
        self._consecutive_velocity_timeouts = 0

    def stop(self) -> None:
        if self._client is None or not self._motion_enabled:
            return
        try:
            if getattr(self, "_velocity_no_reply", False):
                parameter = json.dumps({"velocity": [0.0, 0.0, 0.0], "duration": 0.2})
                # Repeat the finite stop packet to tolerate one dropped DDS
                # sample without waiting for an RPC acknowledgement.
                for _ in range(3):
                    self._client._CallNoReply(7105, parameter)
                    time.sleep(0.02)
            else:
                self._client.SetVelocity(0.0, 0.0, 0.0, 0.2)
        except Exception as exc:  # best effort during emergency cleanup
            print(f"[WARN] StopMove failed: {exc}", file=sys.stderr)

    def pose(self) -> Pose2D:
        with self._lock:
            current, origin = self._absolute, self._origin
        if current is None or origin is None:
            raise RuntimeError("no SportModeState pose received")
        dx, dy = current.x - origin.x, current.y - origin.y
        c, s = math.cos(origin.yaw), math.sin(origin.yaw)
        return Pose2D(
            c * dx + s * dy,
            -s * dx + c * dy,
            wrap_angle(current.yaw - origin.yaw),
            current.stamp,
        )

    def state_age(self) -> float:
        with self._lock:
            current = self._absolute
        return math.inf if current is None else time.monotonic() - current.stamp


class RawTerminal:
    def __enter__(self) -> "RawTerminal":
        if not sys.stdin.isatty():
            raise RuntimeError("recording requires an interactive terminal")
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def read(self, timeout: float) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1).lower() if ready else None

    def __exit__(self, *_: Any) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "control_hz",
        "max_linear_speed_mps",
        "max_yaw_speed_rps",
        "state_timeout_s",
        "max_route_duration_s",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"config is missing: {', '.join(missing)}")
    return config


def load_route(path: Path) -> dict[str, Any]:
    route = json.loads(path.read_text(encoding="utf-8"))
    if route.get("schema_version") != 1:
        raise ValueError("unsupported or missing route schema_version")
    if not isinstance(route.get("events"), list) or not route["events"]:
        raise ValueError("route must contain at least one event")
    return route


def validate_route(route: dict[str, Any], config: dict[str, Any], require_markers: int = 0) -> list[str]:
    errors: list[str] = []
    previous = -1.0
    markers: list[str] = []
    poses = route.get("poses", [])
    for index, event in enumerate(route.get("events", [])):
        try:
            stamp = float(event["t"])
            vx, vy, wz = float(event["vx"]), float(event["vy"]), float(event["wz"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"event {index}: malformed ({exc})")
            continue
        if not all(math.isfinite(value) for value in (stamp, vx, vy, wz)):
            errors.append(f"event {index}: values must be finite")
            continue
        if stamp < previous:
            errors.append(f"event {index}: timestamps are not monotonic")
        if stamp < 0:
            errors.append(f"event {index}: timestamp is negative")
        previous = stamp
        if math.hypot(vx, vy) > float(config["max_linear_speed_mps"]) + 1e-9:
            errors.append(f"event {index}: linear speed exceeds configured maximum")
        if abs(wz) > float(config["max_yaw_speed_rps"]) + 1e-9:
            errors.append(f"event {index}: yaw speed exceeds configured maximum")
        if event.get("marker"):
            markers.append(str(event["marker"]))
            if poses:
                pose_index = event.get("pose_index")
                if not isinstance(pose_index, int) or not 0 <= pose_index < len(poses):
                    errors.append(f"event {index}: marker has invalid pose_index")
    duration = float(route.get("duration_s", previous))
    if duration < 0 or duration > float(config["max_route_duration_s"]):
        errors.append("route duration is outside the configured safety limit")
    if previous > duration + 1e-9:
        errors.append("route duration is earlier than its final event")
    if require_markers:
        expected = [f"point_{index}" for index in range(1, require_markers + 1)]
        if markers != expected:
            errors.append(f"marker order must be {expected}; got {markers or 'none'}")
    previous_pose_stamp = -1.0
    for index, pose in enumerate(poses):
        if not all(key in pose for key in ("t", "x", "y", "yaw")):
            errors.append(f"pose {index}: missing t/x/y/yaw")
            continue
        try:
            values = tuple(float(pose[key]) for key in ("t", "x", "y", "yaw"))
        except (TypeError, ValueError):
            errors.append(f"pose {index}: t/x/y/yaw must be numeric")
            continue
        if not all(math.isfinite(value) for value in values):
            errors.append(f"pose {index}: values must be finite")
            continue
        if values[0] < previous_pose_stamp:
            errors.append(f"pose {index}: timestamps are not monotonic")
        if values[0] < 0:
            errors.append(f"pose {index}: timestamp is negative")
        previous_pose_stamp = values[0]
    return errors


def make_backend(args: argparse.Namespace, real_motion: bool = False):
    if args.real:
        if real_motion and args.confirm != "RUN_TASK1":
            raise RuntimeError("real motion refused: add --confirm RUN_TASK1")
        if real_motion and shutil.which("tmux"):
            active = []
            for session in ("g1_task2_collection", "sonic_data_collection"):
                result = subprocess.run(
                    ["tmux", "has-session", "-t", session], capture_output=True, check=False
                )
                if result.returncode == 0:
                    active.append(session)
            if active:
                raise RuntimeError(
                    "refusing task 1 motion while SONIC/PICO tmux is active: " + ", ".join(active)
                )
        return UnitreeBackend(
            args.iface,
            args.state_topic,
            command_duration=0.8,
            enable_motion=real_motion,
            loco_service=args.loco_service,
            velocity_no_reply=real_motion,
        )
    return MockBackend()


def wait_for_pose(backend: Any, timeout: float) -> Pose2D:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return backend.pose()
        except RuntimeError as exc:
            last_error = exc
            time.sleep(0.05)
    raise RuntimeError(f"pose stream unavailable after {timeout:.1f}s: {last_error}")


def record_route(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if args.real and not args.passive and args.confirm != "RECORD_TASK1":
        raise RuntimeError("real recording refused: add --confirm RECORD_TASK1")
    backend = (
        UnitreeBackend(
            args.iface,
            args.state_topic,
            enable_motion=not args.passive,
            loco_service=args.loco_service,
        )
        if args.real
        else MockBackend()
    )
    if args.real:
        wait_for_pose(backend, args.pose_wait)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    hz = float(config["control_hz"])
    period = 1.0 / hz
    vx = vy = wz = 0.0
    events: list[dict[str, Any]] = [{"t": 0.0, "vx": 0.0, "vy": 0.0, "wz": 0.0}]
    poses: list[dict[str, float]] = []
    start = time.monotonic()
    marker_counter = 0
    print("W/S前后 A/D横移 Q/E转向 空格停；1/2/3标记贴片；P保存退出；X急停退出")
    if args.passive:
        print("被动录制：程序不发运动命令，请用原厂遥控器驾驶。")
    try:
        with RawTerminal() as terminal:
            while True:
                now = time.monotonic()
                elapsed = now - start
                if elapsed > float(config["max_route_duration_s"]):
                    raise RuntimeError("recording exceeded max_route_duration_s")
                try:
                    pose = backend.pose()
                    poses.append({"t": round(elapsed, 4), "x": pose.x, "y": pose.y, "yaw": pose.yaw})
                except RuntimeError:
                    if args.real:
                        raise
                key = terminal.read(period)
                changed = False
                if key == "w":
                    vx, vy, wz, changed = float(config["forward_speed_mps"]), 0.0, 0.0, True
                elif key == "s":
                    vx, vy, wz, changed = -float(config["forward_speed_mps"]), 0.0, 0.0, True
                elif key == "a":
                    vx, vy, wz, changed = 0.0, float(config["lateral_speed_mps"]), 0.0, True
                elif key == "d":
                    vx, vy, wz, changed = 0.0, -float(config["lateral_speed_mps"]), 0.0, True
                elif key == "q":
                    vx, vy, wz, changed = 0.0, 0.0, float(config["yaw_speed_rps"]), True
                elif key == "e":
                    vx, vy, wz, changed = 0.0, 0.0, -float(config["yaw_speed_rps"]), True
                elif key == " ":
                    vx = vy = wz = 0.0
                    changed = True
                elif key in {"1", "2", "3"}:
                    marker_counter += 1
                    marker = f"point_{key}"
                    event = {"t": round(elapsed, 4), "vx": vx, "vy": vy, "wz": wz, "marker": marker}
                    if poses:
                        event["pose_index"] = len(poses) - 1
                    events.append(event)
                    print(f"\n已标记 {marker}")
                elif key in {"p", "x"}:
                    if key == "x":
                        print("\n急停并保存当前记录")
                    break
                if changed:
                    events.append({"t": round(elapsed, 4), "vx": vx, "vy": vy, "wz": wz})
                if not args.passive:
                    backend.set_velocity(vx, vy, wz)
    finally:
        if not args.passive:
            backend.stop()
    duration = time.monotonic() - start
    events.append({"t": round(duration, 4), "vx": 0.0, "vy": 0.0, "wz": 0.0})
    route = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "interface": args.iface,
        "state_topic": args.state_topic,
        "recording_mode": "passive" if args.passive else "keyboard",
        "duration_s": round(duration, 4),
        "events": events,
        "poses": poses,
    }
    output.write_text(json.dumps(route, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已保存 {output}：{len(poses)} 个位姿，{marker_counter} 个标记，{duration:.1f}s")
    return 0


def replay_timed(backend: Any, route: dict[str, Any], config: dict[str, Any], speedup: float) -> None:
    events = route["events"]
    hz = float(config["control_hz"])
    start = time.monotonic()
    index = 0
    while index < len(events):
        event = events[index]
        target_t = float(event["t"]) / speedup
        while time.monotonic() - start < target_t:
            previous = events[max(0, index - 1)]
            backend.set_velocity(float(previous["vx"]), float(previous["vy"]), float(previous["wz"]))
            time.sleep(1.0 / hz)
        backend.set_velocity(float(event["vx"]), float(event["vy"]), float(event["wz"]))
        if event.get("marker"):
            print(f"到达记录标记（仅时间回放）：{event['marker']}")
        index += 1


def sampled_path(route: dict[str, Any], spacing: float = 0.08) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    last: dict[str, Any] | None = None
    marker_by_pose = {
        int(event["pose_index"]): event["marker"]
        for event in route.get("events", [])
        if "pose_index" in event and event.get("marker")
    }
    for index, pose in enumerate(route.get("poses", [])):
        current = {"x": float(pose["x"]), "y": float(pose["y"]), "yaw": float(pose["yaw"])}
        if index in marker_by_pose:
            current["marker"] = marker_by_pose[index]
        distance = math.inf if last is None else math.hypot(current["x"] - last["x"], current["y"] - last["y"])
        angle = math.inf if last is None else abs(wrap_angle(current["yaw"] - last["yaw"]))
        if last is None or distance >= spacing or angle >= math.radians(8) or "marker" in current:
            result.append(current)
            last = current
    if route.get("poses"):
        final = route["poses"][-1]
        if not result or (result[-1]["x"], result[-1]["y"]) != (float(final["x"]), float(final["y"])):
            result.append({"x": float(final["x"]), "y": float(final["y"]), "yaw": float(final["yaw"])})
    return result


def apply_terminal_right_bias(
    path: list[dict[str, Any]], right_bias_m: float, start_before_marker_m: float
) -> tuple[list[dict[str, Any]], int]:
    """Ramp a small rightward offset into the final checkpoint approach.

    This is deliberately a terminal-only empirical correction.  Earlier
    checkpoints and their recorded geometry remain unchanged.
    """
    if right_bias_m <= 0.0 or start_before_marker_m <= 0.0:
        return path, len(path)
    marker_indices = [index for index, point in enumerate(path) if point.get("marker")]
    if not marker_indices:
        return path, len(path)
    marker_index = marker_indices[-1]
    if marker_index < 1:
        return path, len(path)

    cumulative = [0.0]
    for previous, current in zip(path, path[1:]):
        cumulative.append(
            cumulative[-1]
            + math.hypot(current["x"] - previous["x"], current["y"] - previous["y"])
        )
    start_distance = max(0.0, cumulative[marker_index] - start_before_marker_m)
    start_index = next(
        (index for index, distance in enumerate(cumulative) if distance >= start_distance),
        marker_index,
    )

    previous_index = marker_index - 1
    while previous_index >= 0:
        dx = path[marker_index]["x"] - path[previous_index]["x"]
        dy = path[marker_index]["y"] - path[previous_index]["y"]
        length = math.hypot(dx, dy)
        if length > 1e-4:
            ux, uy = dx / length, dy / length
            break
        previous_index -= 1
    else:
        return path, len(path)

    # For a forward tangent (1, 0), the robot's right side is (0, -1).
    right_x, right_y = uy, -ux
    ramp_length = max(1e-6, cumulative[marker_index] - start_distance)
    for index in range(start_index, len(path)):
        factor = min(1.0, max(0.0, (cumulative[index] - start_distance) / ramp_length))
        path[index]["x"] += right_x * right_bias_m * factor
        path[index]["y"] += right_y * right_bias_m * factor
    return path, start_index


def rotate_pose_route(route: dict[str, Any], yaw_offset_rad: float) -> dict[str, Any]:
    """Express a recorded pose route in a start frame with a yaw offset."""
    rotated = json.loads(json.dumps(route))
    c, s = math.cos(yaw_offset_rad), math.sin(yaw_offset_rad)
    for pose in rotated.get("poses", []):
        x, y = float(pose["x"]), float(pose["y"])
        pose["x"] = c * x - s * y
        pose["y"] = s * x + c * y
        pose["yaw"] = wrap_angle(float(pose["yaw"]) + yaw_offset_rad)
    return rotated


def advance_path_index(
    path: list[dict[str, Any]], index: int, pose: Pose2D, max_ahead: int = 12
) -> int:
    """Advance to the nearest forward waypoint without skipping a marker.

    The controller follows a lookahead target, so it can legitimately pass
    several intermediate samples between control ticks.  Keeping the old
    index in that situation makes the robot turn back toward an already
    passed point.  Marker waypoints remain hard barriers and are never
    skipped by this progress update.
    """
    if index >= len(path):
        return index
    if path[index].get("marker"):
        return index
    limit = min(len(path) - 1, index + max_ahead)
    for candidate in range(index, limit + 1):
        if candidate > index and path[candidate].get("marker"):
            limit = candidate
            break
    return min(
        range(index, limit + 1),
        key=lambda candidate: math.hypot(
            path[candidate]["x"] - pose.x,
            path[candidate]["y"] - pose.y,
        ),
    )


def marker_gate_error(
    path: list[dict[str, Any]], index: int, pose: Pose2D
) -> tuple[float, float, float, float]:
    """Return along/cross-track error and the marker's incoming unit vector.

    A humanoid cannot reliably execute the last few centimetres of an exact
    point target.  Checkpoints are therefore treated as gates perpendicular to
    the recorded incoming path.  The robot must physically cross the gate;
    merely getting close to it is not counted.
    """
    marker = path[index]
    previous_index = index - 1
    while previous_index >= 0:
        dx = marker["x"] - path[previous_index]["x"]
        dy = marker["y"] - path[previous_index]["y"]
        length = math.hypot(dx, dy)
        if length > 1e-4:
            ux, uy = dx / length, dy / length
            break
        previous_index -= 1
    else:
        ux, uy = math.cos(marker["yaw"]), math.sin(marker["yaw"])

    offset_x = pose.x - marker["x"]
    offset_y = pose.y - marker["y"]
    along = offset_x * ux + offset_y * uy
    cross = abs(offset_x * uy - offset_y * ux)
    return along, cross, ux, uy


def replay_pose(backend: Any, route: dict[str, Any], config: dict[str, Any]) -> None:
    path = sampled_path(route)
    if len(path) < 2:
        raise RuntimeError("route does not contain enough pose samples")
    terminal_right_bias = float(config.get("terminal_right_bias_m", 0.0))
    path, terminal_start_index = apply_terminal_right_bias(
        path,
        terminal_right_bias,
        float(config.get("terminal_correction_start_before_marker_m", 0.8)),
    )
    if terminal_start_index < len(path):
        print(
            f"终点段修正已启用：最后约 "
            f"{float(config.get('terminal_correction_start_before_marker_m', 0.8)):.2f}m "
            f"逐渐向右补偿 {terminal_right_bias * 100:.1f}cm"
        )
    hz = float(config["control_hz"])
    command_hz = float(config.get("command_hz", hz))
    if not (0.5 <= command_hz <= hz):
        raise RuntimeError("command_hz must be between 0.5 and control_hz")
    command_interval = 1.0 / command_hz
    state_timeout = float(config["state_timeout_s"])
    linear_max = min(0.20, float(config["max_linear_speed_mps"]))
    yaw_max = min(0.25, float(config["max_yaw_speed_rps"]))
    tolerance = float(config["waypoint_tolerance_m"])
    marker_cross_tolerance = float(config.get("marker_cross_track_tolerance_m", 0.10))
    marker_through_distance = float(config.get("marker_through_distance_m", 0.18))
    marker_miss_distance = float(config.get("marker_miss_distance_m", 0.30))
    lookahead = float(config["lookahead_m"])
    terminal_lookahead = min(lookahead, float(config.get("terminal_lookahead_m", 0.12)))
    terminal_linear_max = min(linear_max, float(config.get("terminal_max_linear_speed_mps", 0.12)))
    terminal_tolerance = min(tolerance, float(config.get("terminal_tolerance_m", 0.08)))
    index = 1
    paused_markers: set[str] = set()
    last_command_time = -math.inf
    started = time.monotonic()
    while index < len(path):
        if time.monotonic() - started > float(config["max_route_duration_s"]):
            raise RuntimeError("route follower timed out")
        if backend.state_age() > state_timeout:
            raise RuntimeError(f"pose state stale for > {state_timeout:.1f}s")
        pose = backend.pose()
        index = advance_path_index(path, index, pose)
        while index < len(path):
            point = path[index]
            point_error = math.hypot(point["x"] - pose.x, point["y"] - pose.y)
            marker = point.get("marker")
            if marker:
                along, cross, _, _ = marker_gate_error(path, index, pose)
                if along < 0.0:
                    break
                if cross > marker_cross_tolerance:
                    if along > marker_miss_distance:
                        backend.stop()
                        raise RuntimeError(
                            f"已越过 {marker} 但横向偏差 {cross * 100:.1f}cm，"
                            "超过安全阈值，已停止"
                        )
                    break
            elif point_error > (terminal_tolerance if index == len(path) - 1 else tolerance):
                break
            if index == len(path) - 1:
                yaw_error = abs(wrap_angle(point["yaw"] - pose.yaw))
                if yaw_error > float(config["yaw_tolerance_rad"]):
                    break
            if marker and marker not in paused_markers:
                backend.stop()
                print(
                    f"已穿过 {marker}，横向误差约 {cross * 100:.1f}cm，"
                    f"停留 {config['marker_pause_s']}s"
                )
                time.sleep(float(config["marker_pause_s"]))
                paused_markers.add(marker)
            index += 1
        if index >= len(path):
            break
        target_index = index
        accumulated = 0.0
        active_lookahead = terminal_lookahead if index >= terminal_start_index else lookahead
        while (
            target_index + 1 < len(path)
            and accumulated < active_lookahead
            and not path[target_index].get("marker")
        ):
            a, b = path[target_index], path[target_index + 1]
            accumulated += math.hypot(b["x"] - a["x"], b["y"] - a["y"])
            target_index += 1
            if path[target_index].get("marker"):
                break
        target = path[target_index]
        if target.get("marker"):
            _, _, ux, uy = marker_gate_error(path, target_index, pose)
            target = {
                "x": target["x"] + ux * marker_through_distance,
                "y": target["y"] + uy * marker_through_distance,
                "yaw": target["yaw"],
            }
        dx, dy = target["x"] - pose.x, target["y"] - pose.y
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        local_x, local_y = c * dx + s * dy, -s * dx + c * dy
        terminal_active = index >= terminal_start_index
        active_linear_max = terminal_linear_max if terminal_active else linear_max
        lateral_gain = 1.2 if terminal_active else 0.8
        vx = clamp(0.8 * local_x, active_linear_max)
        vy = clamp(lateral_gain * local_y, min(0.08, active_linear_max))
        wz = clamp(1.2 * wrap_angle(target["yaw"] - pose.yaw), yaw_max)
        now = time.monotonic()
        if now - last_command_time >= command_interval:
            backend.set_velocity(vx, vy, wz)
            last_command_time = time.monotonic()
        time.sleep(1.0 / hz)
    backend.stop()
    final = path[-1]
    pose = backend.pose()
    error = math.hypot(final["x"] - pose.x, final["y"] - pose.y)
    print(f"路线完成，终点平面误差约 {error:.3f} m")


def replay_route(args: argparse.Namespace, config: dict[str, Any]) -> int:
    route = load_route(args.route)
    errors = validate_route(route, config, args.require_markers)
    if errors:
        raise RuntimeError("route validation failed:\n  - " + "\n  - ".join(errors))
    has_pose = len(route.get("poses", [])) >= 2
    mode = args.mode
    if mode == "auto":
        mode = "pose" if has_pose else "timed"
    if args.real and mode == "timed" and not args.allow_timed_fallback:
        raise RuntimeError(
            "real timed replay is disabled by default because it has no drift correction; "
            "record valid pose data or explicitly add --allow-timed-fallback"
        )
    yaw_offset = math.radians(float(args.initial_yaw_offset_deg))
    if abs(yaw_offset) > 1e-9:
        if mode != "pose":
            raise RuntimeError("initial yaw compensation requires pose replay")
        if abs(float(args.initial_yaw_offset_deg)) > 12.0:
            raise RuntimeError("initial yaw compensation exceeds the 12 degree safety limit")
        route = rotate_pose_route(route, yaw_offset)
        print(f"应用视觉起点航向补偿：{float(args.initial_yaw_offset_deg):+.2f}°")
    backend = make_backend(args, real_motion=args.real)
    if args.real and mode == "pose":
        wait_for_pose(backend, args.pose_wait)
    print(f"模式={mode}，真机={args.real}，{config['countdown_s']} 秒后开始；Ctrl+C立即停止")
    for remaining in range(int(config["countdown_s"]), 0, -1):
        print(remaining, flush=True)
        time.sleep(1.0 if args.real else 0.05)
    try:
        if mode == "pose":
            replay_pose(backend, route, config)
        else:
            replay_timed(backend, route, config, args.speedup)
    finally:
        backend.stop()
    return 0


def nudge_robot(args: argparse.Namespace) -> int:
    """Issue one tightly bounded forward command to verify control authority."""
    if not args.real or args.confirm != "NUDGE_TASK1":
        raise RuntimeError(
            "real nudge refused: add --real --confirm NUDGE_TASK1 after physical safety checks"
        )
    profiles = {
        "zero": (0.0, 0.0, 0.0, 0.18),
        "tiny": (0.03, 0.0, 0.0, 0.2),
        "visible": (0.08, 0.0, 0.0, 0.8),
        "clear": (0.12, 0.0, 0.0, 1.5),
        "route-first": (0.20, -0.017, -0.005, 0.2),
    }
    vx, vy, wz, duration = profiles[args.profile]
    backend = UnitreeBackend(
        args.iface,
        args.state_topic,
        command_duration=duration,
        enable_motion=True,
        allow_blocked_fsm_probe=True,
        loco_service=args.loco_service,
    )
    before = wait_for_pose(backend, args.pose_wait)
    print(
        f"点动测试：FSM={backend.fsm_id}, FsmMode={backend.fsm_mode}; "
        f"3 秒后 vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f}，持续 {duration:.2f} 秒"
    )
    for remaining in range(3, 0, -1):
        print(remaining, flush=True)
        time.sleep(1.0)
    try:
        backend.set_velocity(vx, vy, wz)
        time.sleep(duration)
    finally:
        backend.stop()
    time.sleep(0.3)
    after = backend.pose()
    displacement = math.hypot(after.x - before.x, after.y - before.y)
    print(f"点动完成；里程计位移约 {displacement:.4f} m")
    return 0


def preflight(args: argparse.Namespace) -> int:
    failures = 0
    print("Task 1 preflight (read-only)")
    iface_path = Path("/sys/class/net") / args.iface
    if iface_path.exists():
        print(f"[OK] network interface: {args.iface}")
    else:
        print(f"[FAIL] network interface not found: {args.iface}")
        failures += 1
    try:
        __import__("unitree_sdk2py")
        print("[OK] unitree_sdk2py import")
    except Exception as exc:
        print(f"[FAIL] unitree_sdk2py import: {exc}")
        failures += 1
    if args.connect:
        try:
            backend = UnitreeBackend(
                args.iface,
                args.state_topic,
                enable_motion=False,
                query_control_status=True,
                loco_service=args.loco_service,
            )
            pose = wait_for_pose(backend, args.pose_wait)
            print(f"[OK] fresh pose: x={pose.x:.3f} y={pose.y:.3f} yaw={pose.yaw:.3f}")
            print(f"[OK] G1 control service: FSM={backend.fsm_id}, FsmMode={backend.fsm_mode}")
            print(f"[OK] motion owner: {backend.motion_owner}")
            blocked = motion_fsm_block_reason(backend.fsm_id) if backend.fsm_id is not None else None
            if blocked:
                print(f"[BLOCKED] {blocked}")
                failures += 1
        except Exception as exc:
            print(f"[FAIL] G1 read-only state connection: {exc}")
            failures += 1
    else:
        print("[SKIP] G1 state connection (add --connect when robot is wired)")
    return 1 if failures else 0


def inspect_route(args: argparse.Namespace, config: dict[str, Any]) -> int:
    route = load_route(args.route)
    errors = validate_route(route, config, args.require_markers)
    markers = [event["marker"] for event in route["events"] if event.get("marker")]
    print(f"route: {args.route.resolve()}")
    print(f"duration: {float(route.get('duration_s', 0)):.1f}s")
    print(f"events: {len(route['events'])}; poses: {len(route.get('poses', []))}")
    print(f"markers: {markers or 'none'}")
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] route validation passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G1 task 1 route recorder and follower")
    parser.add_argument("--config", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_robot_options(target: argparse.ArgumentParser) -> None:
        target.add_argument("--iface", default="eth0")
        target.add_argument(
            "--loco-service",
            choices=("sport", "loco"),
            default="sport",
            help="G1 high-level RPC service; current firmware uses sport",
        )
        target.add_argument(
            "--state-topic",
            default="auto",
            help="odometry DDS topic; auto listens on rt/odommodestate and rt/lf/odommodestate",
        )
        target.add_argument("--pose-wait", type=float, default=5.0)

    check = sub.add_parser("preflight", help="read-only SDK/network check")
    add_robot_options(check)
    check.add_argument("--connect", action="store_true", help="also subscribe to G1 state")

    record = sub.add_parser("record", help="record a route")
    add_robot_options(record)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--real", action="store_true")
    record.add_argument("--passive", action="store_true", help="record pose while original controller drives")
    record.add_argument("--confirm", default="")

    replay = sub.add_parser("replay", help="follow a recorded route")
    add_robot_options(replay)
    replay.add_argument("--route", type=Path, required=True)
    replay.add_argument("--real", action="store_true")
    replay.add_argument("--confirm", default="")
    replay.add_argument("--mode", choices=("auto", "pose", "timed"), default="auto")
    replay.add_argument("--allow-timed-fallback", action="store_true")
    replay.add_argument("--speedup", type=float, default=10.0, help="dry-run timed replay speed")
    replay.add_argument("--require-markers", type=int, default=3)
    replay.add_argument("--initial-yaw-offset-deg", type=float, default=0.0)

    nudge = sub.add_parser("nudge", help="one bounded real-robot control-authority test")
    add_robot_options(nudge)
    nudge.add_argument("--real", action="store_true")
    nudge.add_argument("--confirm", default="")
    nudge.add_argument(
        "--profile",
        choices=("zero", "tiny", "visible", "clear", "route-first"),
        default="tiny",
    )

    inspect_cmd = sub.add_parser("validate", help="validate and summarize a route")
    inspect_cmd.add_argument("--route", type=Path, required=True)
    inspect_cmd.add_argument("--require-markers", type=int, default=3)
    return parser


def main() -> int:
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "preflight":
            return preflight(args)
        if args.command == "record":
            return record_route(args, config)
        if args.command == "replay":
            if args.speedup <= 0:
                raise ValueError("--speedup must be positive")
            return replay_route(args, config)
        if args.command == "nudge":
            return nudge_robot(args)
        if args.command == "validate":
            return inspect_route(args, config)
        parser.error("unknown command")
    except KeyboardInterrupt:
        print("\n已停止。", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

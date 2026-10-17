#!/usr/bin/env python3
"""Stable bilateral wrist mapping for Task 2 PICO teleoperation.

SONIC's upstream PICO manager uses XR body joints 22/23 (inferred hand joints)
as wrist targets. Near a straight arm their inferred forearm twists can switch
branches even though the tracked controllers remain continuous. This module
replaces both joints with the directly tracked controller poses before the
upstream coordinate conversion and calibration.

The implementation deliberately lives in the Task 2 wrapper.  It does not
modify the official GR00T-WBC checkout and falls back to the original body
joint whenever valid paired controller poses are unavailable. It deliberately
does not clamp either wrist around its calibration angle: SONIC receives the
full relative controller orientation, with only temporal filtering and
quaternion-sign continuity applied.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import threading
import time
from typing import Any
import weakref

import numpy as np


LEFT_HAND_JOINT_INDEX = 22
RIGHT_HAND_JOINT_INDEX = 23


def _normalized_quaternion_xyzw(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion).all() or norm <= 1.0e-9:
        raise ValueError("controller quaternion is not finite and non-zero")
    return quaternion / norm


def _pose7(value: Any) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("controller pose must be finite [x,y,z,qx,qy,qz,qw]")
    pose = pose.copy()
    pose[3:] = _normalized_quaternion_xyzw(pose[3:])
    return pose


def quaternion_angle_rad(lhs_xyzw: np.ndarray, rhs_xyzw: np.ndarray) -> float:
    """Return the shortest unsigned angle between two rotations."""
    lhs = _normalized_quaternion_xyzw(lhs_xyzw)
    rhs = _normalized_quaternion_xyzw(rhs_xyzw)
    dot = float(np.clip(abs(np.dot(lhs, rhs)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def continuous_quaternion_xyzw(
    previous_xyzw: np.ndarray,
    current_xyzw: np.ndarray,
) -> np.ndarray:
    """Choose the current quaternion hemisphere nearest to the previous one."""
    previous = _normalized_quaternion_xyzw(previous_xyzw)
    current = _normalized_quaternion_xyzw(current_xyzw)
    if float(np.dot(previous, current)) < 0.0:
        current = -current
    return current


def _slerp_xyzw(lhs_xyzw: np.ndarray, rhs_xyzw: np.ndarray, fraction: float) -> np.ndarray:
    lhs = _normalized_quaternion_xyzw(lhs_xyzw)
    rhs = continuous_quaternion_xyzw(lhs, rhs_xyzw)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    dot = float(np.clip(np.dot(lhs, rhs), -1.0, 1.0))
    if dot > 0.9995:
        return _normalized_quaternion_xyzw(lhs + fraction * (rhs - lhs))
    angle = math.acos(dot)
    sin_angle = math.sin(angle)
    return _normalized_quaternion_xyzw(
        math.sin((1.0 - fraction) * angle) / sin_angle * lhs
        + math.sin(fraction * angle) / sin_angle * rhs
    )


@dataclass(frozen=True)
class WristMappingConfig:
    max_angular_speed_deg_s: float = 240.0
    max_linear_speed_m_s: float = 1.5
    smoothing_time_constant_s: float = 0.04
    max_controller_body_distance_m: float = 0.75
    paired_pose_cache_size: int = 32

    def validate(self) -> None:
        if self.max_angular_speed_deg_s <= 0.0:
            raise ValueError("max angular speed must be positive")
        if self.max_linear_speed_m_s <= 0.0:
            raise ValueError("max linear speed must be positive")
        if self.smoothing_time_constant_s < 0.0:
            raise ValueError("smoothing time constant cannot be negative")
        if self.max_controller_body_distance_m <= 0.0:
            raise ValueError("controller/body distance guard must be positive")
        if self.paired_pose_cache_size < 2:
            raise ValueError("paired pose cache must contain at least two samples")


class ControllerPoseFilter:
    """Continuity, low-pass and rate limits for a controller pose stream."""

    def __init__(self, config: WristMappingConfig):
        config.validate()
        self.config = config
        self._previous_pose: np.ndarray | None = None
        self._previous_timestamp: float | None = None

    def update(self, pose: Any, timestamp_monotonic: float) -> np.ndarray:
        current = _pose7(pose)
        timestamp = float(timestamp_monotonic)
        if not math.isfinite(timestamp):
            raise ValueError("controller timestamp is not finite")
        if self._previous_pose is None or self._previous_timestamp is None:
            self._previous_pose = current
            self._previous_timestamp = timestamp
            return current.copy()

        previous = self._previous_pose
        dt = timestamp - self._previous_timestamp
        if dt <= 0.0 or dt > 0.25:
            # A long interruption starts a new segment.  Preserve quaternion
            # hemisphere continuity but do not extrapolate across missing data.
            current[3:] = continuous_quaternion_xyzw(previous[3:], current[3:])
            self._previous_pose = current
            self._previous_timestamp = timestamp
            return current.copy()

        if self.config.smoothing_time_constant_s == 0.0:
            smoothing_fraction = 1.0
        else:
            smoothing_fraction = 1.0 - math.exp(
                -dt / self.config.smoothing_time_constant_s
            )

        delta_position = current[:3] - previous[:3]
        distance = float(np.linalg.norm(delta_position))
        if distance > 0.0:
            max_distance = self.config.max_linear_speed_m_s * dt
            position_fraction = min(smoothing_fraction, max_distance / distance)
        else:
            position_fraction = 1.0

        current_quaternion = continuous_quaternion_xyzw(previous[3:], current[3:])
        angle = quaternion_angle_rad(previous[3:], current_quaternion)
        if angle > 1.0e-9:
            max_angle = math.radians(self.config.max_angular_speed_deg_s) * dt
            rotation_fraction = min(smoothing_fraction, max_angle / angle)
        else:
            rotation_fraction = 1.0

        filtered = np.empty(7, dtype=np.float64)
        filtered[:3] = previous[:3] + position_fraction * delta_position
        filtered[3:] = _slerp_xyzw(
            previous[3:], current_quaternion, rotation_fraction
        )
        self._previous_pose = filtered
        self._previous_timestamp = timestamp
        return filtered.copy()


class RightControllerPoseBridge:
    """Pair filtered controller poses with the exact body array read beside them."""

    def __init__(self, config: WristMappingConfig):
        self.config = config
        self.filter = ControllerPoseFilter(config)
        self._lock = threading.Lock()
        self._paired: OrderedDict[
            int, tuple[weakref.ReferenceType[np.ndarray], np.ndarray]
        ] = OrderedDict()
        self._last_output_wxyz: np.ndarray | None = None
        self.controller_samples = 0
        self.fallback_samples = 0

    def register(
        self,
        body_poses_np: np.ndarray,
        right_controller_pose: Any,
        timestamp_monotonic: float,
    ) -> np.ndarray:
        body = np.asarray(body_poses_np, dtype=np.float64)
        if body.shape != (24, 7) or not np.isfinite(body).all():
            raise ValueError(f"body pose must be finite (24,7), got {body.shape}")
        raw_controller = _pose7(right_controller_pose)
        controller_body_distance = float(
            np.linalg.norm(raw_controller[:3] - body[RIGHT_HAND_JOINT_INDEX, :3])
        )
        if controller_body_distance > self.config.max_controller_body_distance_m:
            raise ValueError(
                "right controller is inconsistent with body hand "
                f"({controller_body_distance:.3f}m apart)"
            )
        filtered = self.filter.update(right_controller_pose, timestamp_monotonic)
        with self._lock:
            self._paired[id(body_poses_np)] = (
                weakref.ref(body_poses_np),
                filtered.copy(),
            )
            self._paired.move_to_end(id(body_poses_np))
            while len(self._paired) > self.config.paired_pose_cache_size:
                self._paired.popitem(last=False)
            self.controller_samples += 1
        return filtered

    def substitute(self, body_poses_np: np.ndarray) -> tuple[np.ndarray, bool]:
        body = np.asarray(body_poses_np, dtype=np.float64)
        if body.shape != (24, 7) or not np.isfinite(body).all():
            raise ValueError(f"body pose must be finite (24,7), got {body.shape}")
        with self._lock:
            paired = self._paired.get(id(body_poses_np))
            if paired is None or paired[0]() is not body_poses_np:
                self.fallback_samples += 1
                return body_poses_np, False
            controller = paired[1].copy()
        mapped = body.copy()
        mapped[RIGHT_HAND_JOINT_INDEX] = controller
        return mapped, True

    def stabilize_output_wxyz(self, current_wxyz: np.ndarray) -> np.ndarray:
        current = np.asarray(current_wxyz, dtype=np.float64).reshape(4)
        # Reuse xyzw helpers by moving the scalar component to the end.
        current_xyzw = current[[1, 2, 3, 0]]
        with self._lock:
            if self._last_output_wxyz is None:
                stable_xyzw = _normalized_quaternion_xyzw(current_xyzw)
            else:
                previous_xyzw = self._last_output_wxyz[[1, 2, 3, 0]]
                stable_xyzw = continuous_quaternion_xyzw(
                    previous_xyzw, current_xyzw
                )
            stable_wxyz = stable_xyzw[[3, 0, 1, 2]]
            self._last_output_wxyz = stable_wxyz.copy()
        return stable_wxyz


class BilateralControllerPoseBridge:
    """Pair both directly tracked controllers with one exact body frame."""

    def __init__(self, config: WristMappingConfig):
        self.config = config
        self.left_filter = ControllerPoseFilter(config)
        self.right_filter = ControllerPoseFilter(config)
        self._lock = threading.Lock()
        self._paired: OrderedDict[
            int, tuple[weakref.ReferenceType[np.ndarray], np.ndarray, np.ndarray]
        ] = OrderedDict()
        self._last_output_wxyz: list[np.ndarray | None] = [None, None]
        self.controller_samples = 0
        self.fallback_samples = 0

    def _validate_distance(
        self,
        body: np.ndarray,
        controller: np.ndarray,
        body_joint_index: int,
        side: str,
    ) -> None:
        distance = float(np.linalg.norm(
            controller[:3] - body[body_joint_index, :3]
        ))
        if distance > self.config.max_controller_body_distance_m:
            raise ValueError(
                f"{side} controller is inconsistent with body hand "
                f"({distance:.3f}m apart)"
            )

    def register(
        self,
        body_poses_np: np.ndarray,
        left_controller_pose: Any,
        right_controller_pose: Any,
        timestamp_monotonic: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        body = np.asarray(body_poses_np, dtype=np.float64)
        if body.shape != (24, 7) or not np.isfinite(body).all():
            raise ValueError(f"body pose must be finite (24,7), got {body.shape}")
        raw_left = _pose7(left_controller_pose)
        raw_right = _pose7(right_controller_pose)
        self._validate_distance(body, raw_left, LEFT_HAND_JOINT_INDEX, "left")
        self._validate_distance(body, raw_right, RIGHT_HAND_JOINT_INDEX, "right")
        filtered_left = self.left_filter.update(left_controller_pose, timestamp_monotonic)
        filtered_right = self.right_filter.update(right_controller_pose, timestamp_monotonic)
        with self._lock:
            self._paired[id(body_poses_np)] = (
                weakref.ref(body_poses_np),
                filtered_left.copy(),
                filtered_right.copy(),
            )
            self._paired.move_to_end(id(body_poses_np))
            while len(self._paired) > self.config.paired_pose_cache_size:
                self._paired.popitem(last=False)
            self.controller_samples += 1
        return filtered_left, filtered_right

    def substitute(self, body_poses_np: np.ndarray) -> tuple[np.ndarray, bool]:
        body = np.asarray(body_poses_np, dtype=np.float64)
        if body.shape != (24, 7) or not np.isfinite(body).all():
            raise ValueError(f"body pose must be finite (24,7), got {body.shape}")
        with self._lock:
            paired = self._paired.get(id(body_poses_np))
            if paired is None or paired[0]() is not body_poses_np:
                self.fallback_samples += 1
                return body_poses_np, False
            left_controller = paired[1].copy()
            right_controller = paired[2].copy()
        mapped = body.copy()
        mapped[LEFT_HAND_JOINT_INDEX] = left_controller
        mapped[RIGHT_HAND_JOINT_INDEX] = right_controller
        return mapped, True

    def stabilize_output_wxyz(self, side_index: int, current_wxyz: np.ndarray) -> np.ndarray:
        if side_index not in (0, 1):
            raise ValueError("side_index must be 0 (left) or 1 (right)")
        current = np.asarray(current_wxyz, dtype=np.float64).reshape(4)
        current_xyzw = current[[1, 2, 3, 0]]
        with self._lock:
            previous_wxyz = self._last_output_wxyz[side_index]
            if previous_wxyz is None:
                stable_xyzw = _normalized_quaternion_xyzw(current_xyzw)
            else:
                previous_xyzw = previous_wxyz[[1, 2, 3, 0]]
                stable_xyzw = continuous_quaternion_xyzw(
                    previous_xyzw, current_xyzw
                )
            stable_wxyz = stable_xyzw[[3, 0, 1, 2]]
            self._last_output_wxyz[side_index] = stable_wxyz.copy()
        return stable_wxyz


def install_bilateral_controller_wrist_mapping(
    manager: Any,
    config: WristMappingConfig,
) -> BilateralControllerPoseBridge:
    """Patch the manager to use both real controller poses as wrist targets."""
    bridge = BilateralControllerPoseBridge(config)
    original_process_3pt_pose = manager._process_3pt_pose

    def process_3pt_pose_with_controllers(body_poses_np: np.ndarray) -> np.ndarray:
        mapped_body, used_controller = bridge.substitute(body_poses_np)
        result = original_process_3pt_pose(mapped_body)
        # SONIC consumes quaternion components directly. Keep both wrists on a
        # continuous quaternion hemisphere; q and -q encode the same rotation
        # but a sign flip is a large numeric discontinuity to a neural policy.
        result[0, 3:] = bridge.stabilize_output_wxyz(0, result[0, 3:])
        result[1, 3:] = bridge.stabilize_output_wxyz(1, result[1, 3:])
        if not used_controller and bridge.fallback_samples in (1, 50, 500):
            print(
                "[WristMapping] valid paired controller poses unavailable; "
                "using upstream body-hand targets",
                flush=True,
            )
        return result

    def pico_reader_run_with_controllers(self) -> None:
        last_report = time.time()
        while not self._stop.is_set():
            if not manager.xrt.is_body_data_available():
                time.sleep(0.001)
                continue
            stamp_ns = manager.xrt.get_time_stamp_ns()
            previous_stamp_ns = self._last_stamp_ns
            if previous_stamp_ns is not None and stamp_ns == previous_stamp_ns:
                time.sleep(0.000001)
                continue
            device_dt = (
                (stamp_ns - previous_stamp_ns) * 1.0e-9
                if previous_stamp_ns is not None
                else 0.0
            )
            if device_dt > 0.0:
                instantaneous_fps = 1.0 / device_dt
                self._fps_ema = (
                    instantaneous_fps
                    if self._fps_ema == 0.0
                    else 0.9 * self._fps_ema + 0.1 * instantaneous_fps
                )
            self._last_stamp_ns = stamp_ns
            timestamp_realtime = time.time()
            timestamp_monotonic = time.monotonic()
            try:
                body_poses_np = np.asarray(
                    manager.xrt.get_body_joints_pose(), dtype=np.float64
                )
                left_controller_pose = manager.xrt.get_left_controller_pose()
                right_controller_pose = manager.xrt.get_right_controller_pose()
                controllers_valid = True
                try:
                    filtered_left, filtered_right = bridge.register(
                        body_poses_np,
                        left_controller_pose,
                        right_controller_pose,
                        timestamp_monotonic,
                    )
                except (TypeError, ValueError) as exc:
                    controllers_valid = False
                    filtered_left = None
                    filtered_right = None
                    if bridge.fallback_samples in (0, 49, 499):
                        print(
                            f"[WristMapping] invalid controller pose: {exc}",
                            flush=True,
                        )

                sample = {
                    "body_poses_np": body_poses_np,
                    "left_controller_pose_np": filtered_left,
                    "right_controller_pose_np": filtered_right,
                    "controllers_valid": controllers_valid,
                    "timestamp_realtime": timestamp_realtime,
                    "timestamp_monotonic": timestamp_monotonic,
                    "timestamp_ns": stamp_ns,
                    "dt": device_dt,
                    "fps": self._fps_ema,
                }
                with self._lock:
                    self._latest = sample
                now = time.time()
                if now - last_report >= 5.0:
                    source = "controllers" if controllers_valid else "body fallback"
                    print(
                        f"[PicoReader] dt_ts: {device_dt * 1000.0:.2f} ms, "
                        f"fps: {self._fps_ema:.2f}, both wrists: {source}",
                        flush=True,
                    )
                    last_report = now
            except Exception as exc:
                print(f"[PicoReader] read error: {exc}", flush=True)

    manager._process_3pt_pose = process_3pt_pose_with_controllers
    manager.PicoReader._run = pico_reader_run_with_controllers
    print(
        "[WristMapping] direct left/right controller 6DoF -> SONIC wrists enabled "
        f"(angular limit={config.max_angular_speed_deg_s:.0f}deg/s, "
        f"linear limit={config.max_linear_speed_m_s:.1f}m/s, "
        f"filter tau={config.smoothing_time_constant_s:.3f}s); "
        "no post-calibration wrist-angle clamp",
        flush=True,
    )
    return bridge

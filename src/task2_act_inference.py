#!/usr/bin/env python3
"""Run one task-2 ACT policy with a jitter-resistant right-arm controller.

The policy observes both RGB cameras plus the measured 29-DoF G1 state and
right Dex1 position.  It predicts a 100-step action chunk at 20 Hz.  Inference
runs asynchronously so the 50 Hz robot command loop never waits for the GPU.

Stability measures applied after the network:

* overlapping action chunks are aligned in time and temporally ensembled;
* the 20 Hz trajectory is linearly interpolated at the 50 Hz control rate;
* targets pass through a low-pass, velocity and acceleration limiter;
* commands stay inside the demonstrated range and close to measured joints;
* stale cameras/state/model output or excessive tracking error causes a gradual
  Arm-SDK release instead of holding a stale pose;
* motor temperature is reported continuously.  By default it is warning-only
  and the robot's onboard protection remains responsible for thermal shutdown.

Terminal controls (press Enter after the letter):

  s  start one autonomous attempt
  x  stop the attempt and smoothly release the right arm
  q  release and quit

Real actuation is impossible unless ``--real`` is supplied.  Use
``--dry-run`` first at a new cabinet pose; it executes the complete vision and
policy pipeline but creates no command publishers.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import select
import signal
import sys
import threading
import time
from typing import Any

import numpy as np


BODY_JOINT_COUNT = 29
ARM_SDK_WEIGHT_INDEX = 29
RIGHT_ARM_BODY_INDICES = np.arange(22, 29, dtype=np.int64)
SUPPORT_BODY_INDICES = np.arange(12, 22, dtype=np.int64)  # waist + left arm


class StationaryBodyGuard:
    """Snapshot upper-body support; reject movement instead of locking legs.

    Thresholds are conservative commissioning limits, not a guarantee that
    the locomotion controller will never take a balance-recovery step.
    """

    def __init__(
        self, body, rpy, position,
        waist_kp=40.0, waist_kd=2.0,
        left_arm_kp=28.0, left_arm_kd=1.8,
    ):
        self.support = np.asarray(body, dtype=float)[SUPPORT_BODY_INDICES].copy()
        self.rpy = np.asarray(rpy, dtype=float).copy()
        self.position = np.asarray(position, dtype=float)[:2].copy()
        self.waist_kp = float(waist_kp)
        self.waist_kd = float(waist_kd)
        self.left_arm_kp = float(left_arm_kp)
        self.left_arm_kd = float(left_arm_kd)
        if not all(np.isfinite(v).all() for v in (self.support, self.rpy, self.position)):
            raise ValueError("non-finite body reference")

    def metrics(self, body, rpy, position, velocity):
        angles = np.asarray(rpy, dtype=float)
        delta = np.arctan2(np.sin(angles - self.rpy), np.cos(angles - self.rpy))
        waist_error = np.abs(np.asarray(body)[12:15] - self.support[:3])
        return {
            "waist_yaw": float(waist_error[0]),
            "waist_roll": float(waist_error[1]),
            "waist_pitch": float(waist_error[2]),
            "left_arm": float(np.max(np.abs(np.asarray(body)[15:22] - self.support[3:]))),
            "tilt": float(np.max(np.abs(angles[:2]))),
            "tilt_delta": float(np.max(np.abs(delta[:2]))),
            "yaw_delta": float(abs(delta[2])),
            "travel": float(np.linalg.norm(np.asarray(position)[:2] - self.position)),
            "speed": float(np.linalg.norm(np.asarray(velocity)[:2])),
        }

    def reason(self, body, rpy, position, velocity):
        values = self.metrics(body, rpy, position, velocity)
        limits = dict(waist_yaw=0.10, waist_roll=0.10, waist_pitch=0.10,
                      left_arm=0.20, tilt=0.20,
                      tilt_delta=0.08, yaw_delta=0.10, travel=0.04, speed=0.12)
        for name, value in values.items():
            if not math.isfinite(value) or value > limits[name]:
                return f"body guard: {name}={value:.3f}, limit={limits[name]:.3f}"
        return None

    def apply_support(self, low_command):
        # A single Arm SDK weight also claims the waist and left arm.  They
        # need explicit support commands, not zero-filled gains.  Capture once
        # before takeover so a leaning pose is not chased on every tick.
        for local_index, (index, target) in enumerate(
            zip(SUPPORT_BODY_INDICES, self.support)
        ):
            command = low_command.motor_cmd[int(index)]
            command.q = float(target)
            command.dq = command.tau = 0.0
            if local_index < 3:
                command.kp = self.waist_kp
                command.kd = self.waist_kd
            else:
                command.kp = self.left_arm_kp
                command.kd = self.left_arm_kd
JOINT_NAMES = (
    "LeftHipPitch", "LeftHipRoll", "LeftHipYaw", "LeftKnee",
    "LeftAnklePitch", "LeftAnkleRoll", "RightHipPitch", "RightHipRoll",
    "RightHipYaw", "RightKnee", "RightAnklePitch", "RightAnkleRoll",
    "WaistYaw", "WaistRoll", "WaistPitch", "LeftShoulderPitch",
    "LeftShoulderRoll", "LeftShoulderYaw", "LeftElbow", "LeftWristRoll",
    "LeftWristPitch", "LeftWristYaw", "RightShoulderPitch",
    "RightShoulderRoll", "RightShoulderYaw", "RightElbow",
    "RightWristRoll", "RightWristPitch", "RightWristYaw",
)
RIGHT_ACTION_NAMES = JOINT_NAMES[22:29] + ("RightDex1",)

# These limits include the configurations that were actually observed during
# task-2 demonstrations.  The demonstrated-range limiter below is tighter for
# each skill and is therefore the primary policy-output guard.
RIGHT_ARM_COMMAND_LOWER = np.asarray(
    [-3.0892, -2.2515, -2.6180, -1.0472, -1.9722, -1.6144, -1.6144],
    dtype=np.float64,
)
RIGHT_ARM_COMMAND_UPPER = np.asarray(
    [2.6704, 0.15, 2.6180, 2.0944, 1.9722, 1.6144, 1.6144],
    dtype=np.float64,
)
GRIPPER_LOWER = 0.0
GRIPPER_UPPER = 5.2


@dataclass(frozen=True)
class Observation:
    episode_id: int
    policy_step: int
    captured_at: float
    state: np.ndarray
    ego_view: np.ndarray
    right_wrist: np.ndarray


@dataclass(frozen=True)
class InferenceResult:
    episode_id: int
    policy_step: int
    captured_at: float
    completed_at: float
    latency: float
    actions: np.ndarray | None = None
    error: str | None = None


@dataclass(frozen=True)
class ActionChunk:
    episode_id: int
    start_step: int
    actions: np.ndarray
    completed_at: float


def parse_action_bounds(path: Path, skill: str) -> tuple[np.ndarray, np.ndarray]:
    document = json.loads(path.read_text())
    entry = document["skills"][skill]
    lower = np.asarray(entry["action_min"], dtype=np.float64)
    upper = np.asarray(entry["action_max"], dtype=np.float64)
    if lower.shape != (8,) or upper.shape != (8,) or np.any(lower > upper):
        raise ValueError(f"invalid action bounds for {skill} in {path}")
    return lower, upper


def parse_start_action(path: Path, skill: str) -> np.ndarray:
    document = json.loads(path.read_text())
    value = np.asarray(
        document["skills"][skill]["start_action_median"], dtype=np.float64
    )
    if value.shape != (8,) or not np.isfinite(value).all():
        raise ValueError(f"invalid start action for {skill} in {path}")
    if np.any(value[:7] < RIGHT_ARM_COMMAND_LOWER) or np.any(
        value[:7] > RIGHT_ARM_COMMAND_UPPER
    ) or not GRIPPER_LOWER <= value[7] <= GRIPPER_UPPER:
        raise ValueError(f"start action outside physical limits for {skill}")
    return value


def parse_start_observation_envelope(
    path: Path, skill: str
) -> tuple[np.ndarray, np.ndarray]:
    entry = json.loads(path.read_text())["skills"][skill]
    lower = np.asarray(entry["start_observation_p10"], dtype=np.float64)
    upper = np.asarray(entry["start_observation_p90"], dtype=np.float64)
    if (
        lower.shape != (8,)
        or upper.shape != (8,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower > upper)
    ):
        raise ValueError(f"invalid start observation envelope for {skill} in {path}")
    return lower, upper


class TemporalChunkEnsembler:
    """Blend time-aligned predictions from overlapping ACT chunks."""

    def __init__(self, max_chunks: int = 8, newer_weight_decay: float = 0.08):
        if max_chunks < 1 or newer_weight_decay < 0:
            raise ValueError("invalid temporal ensemble settings")
        self.max_chunks = int(max_chunks)
        self.newer_weight_decay = float(newer_weight_decay)
        self._chunks: deque[ActionChunk] = deque(maxlen=self.max_chunks)

    def clear(self) -> None:
        self._chunks.clear()

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, chunk: ActionChunk) -> None:
        actions = np.asarray(chunk.actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 8 or len(actions) < 2:
            raise ValueError(f"invalid ACT chunk shape {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("ACT chunk contains NaN/Inf")
        if self._chunks and chunk.episode_id != self._chunks[-1].episode_id:
            self.clear()
        self._chunks.append(ActionChunk(
            chunk.episode_id,
            int(chunk.start_step),
            actions.copy(),
            float(chunk.completed_at),
        ))

    @staticmethod
    def _interpolate(actions: np.ndarray, local_step: float) -> np.ndarray | None:
        if local_step < 0.0 or local_step > len(actions) - 1:
            return None
        lo = int(math.floor(local_step))
        hi = min(lo + 1, len(actions) - 1)
        fraction = local_step - lo
        return actions[lo] * (1.0 - fraction) + actions[hi] * fraction

    def action_at(self, episode_id: int, policy_step: float) -> np.ndarray | None:
        candidates: list[np.ndarray] = []
        for chunk in self._chunks:
            if chunk.episode_id != episode_id:
                continue
            value = self._interpolate(chunk.actions, policy_step - chunk.start_step)
            if value is not None:
                candidates.append(value)
        if not candidates:
            return None
        # The original ACT temporal ensemble deliberately gives older
        # predictions slightly more weight.  That damps frame-to-frame vision
        # noise while the finite history still lets feedback correct the path.
        ranks = np.arange(len(candidates), dtype=np.float64)
        weights = np.exp(-self.newer_weight_decay * ranks)
        weights /= weights.sum()
        return np.sum(np.stack(candidates) * weights[:, None], axis=0)


class SmoothActionFilter:
    """Low-pass, velocity and acceleration limiting for eight ACT outputs."""

    def __init__(
        self,
        action_lower: np.ndarray,
        action_upper: np.ndarray,
        cutoff_hz: float,
        arm_speed: float,
        arm_acceleration: float,
        gripper_speed: float,
        gripper_acceleration: float,
        max_follow_error: float,
    ):
        demo_margin = np.asarray([0.08] * 7 + [0.05], dtype=np.float64)
        lower = np.asarray(action_lower, dtype=np.float64) - demo_margin
        upper = np.asarray(action_upper, dtype=np.float64) + demo_margin
        lower[:7] = np.maximum(lower[:7], RIGHT_ARM_COMMAND_LOWER)
        upper[:7] = np.minimum(upper[:7], RIGHT_ARM_COMMAND_UPPER)
        lower[7] = max(lower[7], GRIPPER_LOWER)
        upper[7] = min(upper[7], GRIPPER_UPPER)
        self.lower = lower
        self.upper = upper
        self.cutoff_hz = float(cutoff_hz)
        self.max_follow_error = float(max_follow_error)
        self.max_speed = np.asarray([arm_speed] * 7 + [gripper_speed], dtype=np.float64)
        self.max_acceleration = np.asarray(
            [arm_acceleration] * 7 + [gripper_acceleration], dtype=np.float64
        )
        self.command = np.zeros(8, dtype=np.float64)
        self.filtered_target = np.zeros(8, dtype=np.float64)
        self.velocity = np.zeros(8, dtype=np.float64)
        self.initialized = False

    def reset(self, measured: np.ndarray) -> None:
        value = np.asarray(measured, dtype=np.float64).reshape(8)
        self.command = value.copy()
        self.filtered_target = value.copy()
        self.velocity.fill(0.0)
        self.initialized = True

    def handoff_from(self, previous: "SmoothActionFilter") -> None:
        """Continue the last sent command, preserving its support effort.

        Resetting to measured position discards the position error that was
        generating holding torque, producing a torque drop at policy start.
        """
        if not previous.initialized:
            raise ValueError("handoff source is not initialized")
        self.reset(previous.command)
        self.velocity = np.clip(previous.velocity, -self.max_speed, self.max_speed).copy()

    def clamp_raw(self, raw: np.ndarray, measured: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64).reshape(8)
        measured = np.asarray(measured, dtype=np.float64).reshape(8)
        if not np.isfinite(raw).all() or not np.isfinite(measured).all():
            raise ValueError("non-finite raw/measured action")
        desired = np.clip(raw, self.lower, self.upper)
        desired[:7] = np.clip(
            desired[:7],
            measured[:7] - self.max_follow_error,
            measured[:7] + self.max_follow_error,
        )
        # Dex1 was intentionally allowed a larger planning horizon during
        # demonstrations, but each command still follows measured motion.
        desired[7] = np.clip(desired[7], measured[7] - 0.40, measured[7] + 0.40)
        return desired

    def step(self, raw: np.ndarray, measured: np.ndarray, dt: float) -> np.ndarray:
        dt = float(np.clip(dt, 0.001, 0.1))
        measured = np.asarray(measured, dtype=np.float64).reshape(8)
        if not self.initialized:
            self.reset(measured)
        desired = self.clamp_raw(raw, measured)
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * dt)
        self.filtered_target += alpha * (desired - self.filtered_target)
        requested_velocity = np.clip(
            (self.filtered_target - self.command) / dt,
            -self.max_speed,
            self.max_speed,
        )
        delta_velocity = np.clip(
            requested_velocity - self.velocity,
            -self.max_acceleration * dt,
            self.max_acceleration * dt,
        )
        self.velocity = np.clip(
            self.velocity + delta_velocity, -self.max_speed, self.max_speed
        )
        candidate = self.command + self.velocity * dt
        # Do not overshoot the filtered target because of accumulated velocity.
        crossed = (self.filtered_target - self.command) * (
            self.filtered_target - candidate
        ) <= 0.0
        candidate[crossed] = self.filtered_target[crossed]
        self.velocity[crossed] = 0.0
        # The measured joint may be pulled back by normal locomotion while
        # Arm SDK is blended in.  Re-apply the following window to the final
        # accumulated command so it cannot wind up beyond the configured
        # error guard even when the target itself was correctly clamped.
        bounded = candidate.copy()
        bounded[:7] = np.clip(
            bounded[:7],
            measured[:7] - self.max_follow_error,
            measured[:7] + self.max_follow_error,
        )
        bounded[7] = np.clip(bounded[7], measured[7] - 0.40, measured[7] + 0.40)
        limited = np.abs(bounded - candidate) > 1.0e-12
        self.velocity[limited] = 0.0
        candidate = bounded
        self.command = np.clip(candidate, self.lower, self.upper)
        return self.command.copy()


class MotionCycleCompletion:
    """Detect a demonstrated reach/interact/return cycle from measured joints.

    All accepted task-2 demonstrations leave the initial right-arm pose by at
    least 2.9 rad in joint-space norm, then return within 0.74 rad and settle.
    Conservative defaults sit well inside those observed margins.  This is an
    execution-completion signal, not proof that the external panel changed.
    """

    def __init__(
        self,
        excursion_threshold: float,
        return_threshold: float,
        stable_speed: float,
        hold_seconds: float,
        minimum_seconds: float,
    ):
        self.excursion_threshold = float(excursion_threshold)
        self.return_threshold = float(return_threshold)
        self.stable_speed = float(stable_speed)
        self.hold_seconds = float(hold_seconds)
        self.minimum_seconds = float(minimum_seconds)
        self.start = np.zeros(7, dtype=np.float64)
        self.previous = np.zeros(7, dtype=np.float64)
        self.max_excursion = 0.0
        self.current_excursion = 0.0
        self.current_speed = 0.0
        self.stable_for = 0.0
        self.departed = False
        self.initialized = False

    def reset(self, measured_arm: np.ndarray) -> None:
        value = np.asarray(measured_arm, dtype=np.float64).reshape(7)
        self.start = value.copy()
        self.previous = value.copy()
        self.max_excursion = 0.0
        self.current_excursion = 0.0
        self.current_speed = 0.0
        self.stable_for = 0.0
        self.departed = False
        self.initialized = True

    def update(self, measured_arm: np.ndarray, dt: float, elapsed: float) -> bool:
        value = np.asarray(measured_arm, dtype=np.float64).reshape(7)
        if not np.isfinite(value).all():
            self.stable_for = 0.0
            return False
        if not self.initialized:
            self.reset(value)
            return False
        dt = float(np.clip(dt, 0.001, 0.1))
        self.current_excursion = float(np.linalg.norm(value - self.start))
        self.current_speed = float(np.linalg.norm(value - self.previous) / dt)
        self.previous = value.copy()
        self.max_excursion = max(self.max_excursion, self.current_excursion)
        if self.max_excursion >= self.excursion_threshold:
            self.departed = True
        returned_and_stable = (
            self.departed
            and elapsed >= self.minimum_seconds
            and self.current_excursion <= self.return_threshold
            and self.current_speed <= self.stable_speed
        )
        self.stable_for = self.stable_for + dt if returned_and_stable else 0.0
        return self.stable_for + 1.0e-9 >= self.hold_seconds


class ACTEngine:
    """Local LeRobot ACT loader with explicit full-chunk inference."""

    def __init__(self, model_dir: Path, device: str):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.common.policies.act.modeling_act import ACTPolicy

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Real-time ACT is disabled because CPU inference "
                "cannot meet the robot control deadline. Check nvidia-smi/driver."
            )
        self.device = torch.device(device)
        config = PreTrainedConfig.from_pretrained(model_dir, local_files_only=True)
        config.device = str(self.device)
        # The trained checkpoint already contains the backbone.  Disabling the
        # constructor download makes deployment fully offline and deterministic.
        config.pretrained_backbone_weights = None
        self.policy = ACTPolicy.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True,
            strict=True,
        )
        self.policy.eval()
        self.torch = torch
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def predict_chunk(
        self, state: np.ndarray, ego_view: np.ndarray, right_wrist: np.ndarray
    ) -> np.ndarray:
        torch = self.torch

        def image_tensor(image: np.ndarray):
            value = np.asarray(image)
            if value.shape != (480, 640, 3):
                raise ValueError(f"unexpected camera image shape {value.shape}")
            tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1)
            return tensor.to(self.device, dtype=torch.float32).div_(255.0).unsqueeze(0)

        state_tensor = torch.from_numpy(
            np.ascontiguousarray(np.asarray(state, dtype=np.float32).reshape(30))
        ).to(self.device).unsqueeze(0)
        batch = {
            "observation.state": state_tensor,
            "observation.images.ego_view": image_tensor(ego_view),
            "observation.images.right_wrist": image_tensor(right_wrist),
        }
        with torch.inference_mode():
            normalized = self.policy.normalize_inputs(batch)
            normalized = dict(normalized)
            normalized["observation.images"] = [
                normalized[key] for key in self.policy.config.image_features
            ]
            actions = self.policy.model(normalized)[0]
            actions = actions[:, : self.policy.config.n_action_steps]
            actions = self.policy.unnormalize_outputs({"action": actions})["action"]
        result = actions[0].detach().to("cpu", dtype=torch.float32).numpy()
        if result.ndim != 2 or result.shape[1] != 8 or not np.isfinite(result).all():
            raise RuntimeError(f"invalid ACT output {result.shape}")
        return result

    def warmup(self) -> float:
        started = time.monotonic()
        self.predict_chunk(
            np.zeros(30, dtype=np.float32),
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        return time.monotonic() - started


class InferenceWorker:
    def __init__(self, engine: ACTEngine):
        self.engine = engine
        self.requests: queue.Queue[Observation | None] = queue.Queue(maxsize=1)
        self.results: queue.SimpleQueue[InferenceResult] = queue.SimpleQueue()
        self.thread = threading.Thread(target=self._run, name="act-inference", daemon=True)
        self.thread.start()

    def submit(self, observation: Observation) -> None:
        try:
            self.requests.put_nowait(observation)
            return
        except queue.Full:
            pass
        try:
            self.requests.get_nowait()
        except queue.Empty:
            pass
        self.requests.put_nowait(observation)

    def drain(self) -> list[InferenceResult]:
        values: list[InferenceResult] = []
        while True:
            try:
                values.append(self.results.get_nowait())
            except queue.Empty:
                return values

    def clear_requests(self) -> None:
        while True:
            try:
                self.requests.get_nowait()
            except queue.Empty:
                return

    def _run(self) -> None:
        while True:
            request = self.requests.get()
            if request is None:
                return
            started = time.monotonic()
            try:
                actions = self.engine.predict_chunk(
                    request.state, request.ego_view, request.right_wrist
                )
                completed = time.monotonic()
                result = InferenceResult(
                    request.episode_id,
                    request.policy_step,
                    request.captured_at,
                    completed,
                    completed - started,
                    actions=actions,
                )
            except Exception as exc:  # Keep control loop alive for smooth release.
                completed = time.monotonic()
                result = InferenceResult(
                    request.episode_id,
                    request.policy_step,
                    request.captured_at,
                    completed,
                    completed - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
            self.results.put(result)

    def close(self) -> None:
        self.clear_requests()
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=2.0)


@dataclass(frozen=True)
class RobotMeasurement:
    body: np.ndarray
    gripper: float
    temperatures: np.ndarray
    arm_dq: np.ndarray
    received_at: float
    source_tick: int
    rpy: np.ndarray

    @property
    def speed_index(self) -> int:
        invalid = np.flatnonzero(~np.isfinite(self.arm_dq))
        return int(invalid[0]) if len(invalid) else int(np.argmax(np.abs(self.arm_dq)))

    @property
    def arm_speed(self) -> float:
        if not np.isfinite(self.arm_dq).all():
            return math.inf  # Never interpret a bad velocity as a stationary arm.
        return float(np.max(np.abs(self.arm_dq)))

    def speed_detail(self, previous: "RobotMeasurement | None", now: float) -> str:
        index = self.speed_index
        sample_dt = self.received_at - previous.received_at if previous else 0.0
        # Receiver-time finite differences are diagnostic only: queue/network
        # jitter can distort them. Motor dq is the speed protection source.
        fd = "n/a"
        if previous and sample_dt > 0:
            fd = f"{(self.body[22 + index] - previous.body[22 + index]) / sample_dt:+.3f}"
        return (
            f"joint={RIGHT_ACTION_NAMES[index]} motor_dq={self.arm_dq[index]:+.3f}rad/s "
            f"q_fd={fd}rad/s sample_dt={sample_dt * 1000:.1f}ms "
            f"sample_age={max(0., now - self.received_at) * 1000:.1f}ms "
            f"source_tick={self.source_tick} "
            f"arm_q={np.round(self.body[22:29], 4).tolist()} "
            f"arm_dq={np.round(self.arm_dq, 4).tolist()}"
        )


class RobotSources:
    """Dual cameras, G1/Dex1 state, and optional real command publishers."""

    def __init__(
        self,
        interface: str,
        camera_host: str,
        camera_port: int,
        real: bool,
        waist_hold_kp: float = 40.0,
        waist_hold_kd: float = 2.0,
        left_arm_hold_kp: float = 28.0,
        left_arm_hold_kd: float = 1.8,
    ):
        from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
            MotorCmd_, MotorCmds_, MotorStates_, SportModeState_,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.real = bool(real)
        self._LowCmd = LowCmd_
        self._MotorCmd = MotorCmd_
        self._MotorCmds = MotorCmds_
        self._crc = CRC()
        # One assignment pairs message and reception time across DDS/main threads.
        self._low_snapshot: tuple[Any, float] | None = None
        self._dex_state: tuple[float, float] | None = None
        self._odometry = None
        self._odometry_topic = None
        self.body_guard = None
        self._support_gains = (
            float(waist_hold_kp), float(waist_hold_kd),
            float(left_arm_hold_kp), float(left_arm_hold_kd),
        )
        self.camera_message: dict[str, Any] | None = None
        self.camera_at = 0.0
        self.camera_signature: Any = None

        ChannelFactoryInitialize(0, interface)

        def on_low_state(message: Any) -> None:
            self._low_snapshot = (message, time.monotonic())

        def on_dex_state(message: Any) -> None:
            if message.states and math.isfinite(float(message.states[0].q)):
                self._dex_state = (float(message.states[0].q), time.monotonic())

        self.low_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_subscriber.Init(on_low_state, 1)
        self.dex_subscriber = ChannelSubscriber("rt/dex1/right/state", MotorStates_)
        self.dex_subscriber.Init(on_dex_state, 1)
        self.odom_subscribers = []
        if self.real:
            def make_odom_callback(topic):
                def callback(message):
                    position = np.asarray(message.position, dtype=float)
                    velocity = np.asarray(message.velocity, dtype=float)
                    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
                        return
                    if self._odometry_topic not in (None, topic):
                        return
                    self._odometry_topic = topic
                    self._odometry = (position.copy(), velocity.copy(), time.monotonic())
                return callback
            for topic in ("rt/odommodestate", "rt/lf/odommodestate"):
                subscriber = ChannelSubscriber(topic, SportModeState_)
                subscriber.Init(make_odom_callback(topic), 1)
                self.odom_subscribers.append(subscriber)
        self.arm_publisher = None
        self.dex_publisher = None
        if self.real:
            self.arm_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
            self.arm_publisher.Init()
            self.dex_publisher = ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)
            self.dex_publisher.Init()
        self.camera = ComposedCameraClientSensor(camera_host, camera_port)
        self.low_command = unitree_hg_msg_dds__LowCmd_()

    def poll_camera(self) -> None:
        message = self.camera.read(blocking=False)
        if message is None:
            return
        signature = tuple(sorted(message.get("timestamps", {}).items()))
        if signature != self.camera_signature:
            self.camera_signature = signature
            self.camera_message = message
            self.camera_at = time.monotonic()

    @property
    def _low_state(self):
        sample = self._low_snapshot
        return sample[0] if sample else None

    @property
    def _low_state_at(self):
        sample = self._low_snapshot
        return sample[1] if sample else 0.0

    def measured(self) -> RobotMeasurement | None:
        snapshot = self._low_snapshot
        dex = self._dex_state
        if snapshot is None or dex is None:
            return None
        state, received_at = snapshot
        body = np.asarray(
            [float(motor.q) for motor in state.motor_state[:BODY_JOINT_COUNT]],
            dtype=np.float64,
        )
        temperatures = np.asarray(
            [
                max(float(motor.temperature[0]), float(motor.temperature[1]))
                for motor in state.motor_state[:BODY_JOINT_COUNT]
            ],
            dtype=np.float64,
        )
        return RobotMeasurement(
            body, float(dex[0]), temperatures,
            np.asarray([float(state.motor_state[i].dq) for i in range(22, 29)]),
            received_at, int(state.tick),
            np.asarray(state.imu_state.rpy, dtype=float).copy(),
        )

    def readiness(self, max_age: float) -> tuple[bool, str]:
        self.poll_camera()
        now = time.monotonic()
        missing: list[str] = []
        if self._low_state is None or now - self._low_state_at > max_age:
            missing.append("G1 lowstate")
        if self._dex_state is None or now - self._dex_state[1] > max_age:
            missing.append("Dex1 state")
        if self.real and (self._odometry is None or now - self._odometry[2] > max_age):
            missing.append("body odometry (required for stationary ACT)")
        if self.real and self._low_state is not None:
            if not np.isfinite(self._low_state.imu_state.rpy).all():
                missing.append("finite body IMU")
        if self.camera_message is None or now - self.camera_at > max_age:
            missing.append("dual camera")
        else:
            images = self.camera_message.get("images", {})
            if not {"ego_view", "right_wrist"}.issubset(images):
                missing.append("ego_view/right_wrist")
            else:
                shapes = [np.asarray(images[key]).shape for key in ("ego_view", "right_wrist")]
                if shapes != [(480, 640, 3), (480, 640, 3)]:
                    missing.append(f"camera shapes {shapes}")
        return not missing, ", ".join(missing) if missing else "all sources fresh"

    def observation(self, episode_id: int, policy_step: int, max_age: float) -> Observation:
        ready, detail = self.readiness(max_age)
        measured = self.measured()
        if not ready or measured is None or self.camera_message is None:
            raise RuntimeError(detail)
        body, gripper = measured.body, measured.gripper
        state = np.concatenate((body, np.asarray([gripper]))).astype(np.float32)
        images = self.camera_message["images"]
        return Observation(
            episode_id=episode_id,
            policy_step=int(policy_step),
            captured_at=time.monotonic(),
            state=state,
            ego_view=np.ascontiguousarray(images["ego_view"]).copy(),
            right_wrist=np.ascontiguousarray(images["right_wrist"]).copy(),
        )

    @property
    def low_state_age(self) -> float:
        return time.monotonic() - self._low_state_at

    def capture_body_hold(self, measurement):
        if not self.real:
            return
        if self._low_state is None or self._odometry is None:
            raise ValueError("body IMU/odometry unavailable")
        position, velocity, _ = self._odometry
        rpy = measurement.rpy
        if not np.isfinite(measurement.body).all() or not math.isfinite(measurement.arm_speed):
            raise ValueError("invalid joint position/velocity feedback")
        if not np.isfinite(rpy).all() or np.max(np.abs(rpy[:2])) > 0.12:
            raise ValueError("body is tilted; restore stable upright stance before starting")
        if np.linalg.norm(velocity[:2]) > 0.03:
            raise ValueError("robot is moving; stop locomotion before starting")
        self.body_guard = StationaryBodyGuard(
            measurement.body, rpy, position, *self._support_gains
        )

    def body_motion(self, measurement):
        if self.body_guard is None or self._odometry is None or self._low_state is None:
            return None, {}
        position, velocity, _ = self._odometry
        values = (measurement.body, measurement.rpy, position, velocity)
        return self.body_guard.reason(*values), self.body_guard.metrics(*values)

    def publish(self, action: np.ndarray, arm_weight: float, kp: float, kd: float) -> None:
        if not self.real:
            return
        assert self.arm_publisher is not None and self.dex_publisher is not None
        if getattr(self, "emergency_stop_requested", lambda: False)():
            self.release_arm_ownership()
            return
        value = np.asarray(action, dtype=np.float64).reshape(8)
        if arm_weight > 0 and self.body_guard is None:
            raise RuntimeError("refusing Arm SDK takeover without a body support reference")
        if self.body_guard is not None:
            self.body_guard.apply_support(self.low_command)
        self.low_command.motor_cmd[ARM_SDK_WEIGHT_INDEX].q = float(arm_weight)
        for local_index, body_index in enumerate(RIGHT_ARM_BODY_INDICES):
            command = self.low_command.motor_cmd[int(body_index)]
            command.q = float(value[local_index])
            command.dq = 0.0
            command.tau = 0.0
            command.kp = float(kp)
            command.kd = float(kd)
        self.low_command.crc = self._crc.Crc(self.low_command)
        if getattr(self, "emergency_stop_requested", lambda: False)():
            self.release_arm_ownership()
            return
        self.arm_publisher.Write(self.low_command)
        if getattr(self, "emergency_stop_requested", lambda: False)():
            self.release_arm_ownership()
            return
        self.dex_publisher.Write(self._MotorCmds(cmds=[self._MotorCmd(
            mode=1,
            q=float(value[7]),
            dq=0.0,
            tau=0.0,
            kp=5.0,
            kd=0.05,
            reserve=[0, 0, 0],
        )]))

    def close(self) -> None:
        self.camera.stop_client()
        if self.dex_publisher is not None:
            self.dex_publisher.Close()
        if self.arm_publisher is not None:
            self.arm_publisher.Close()
        self.dex_subscriber.Close()
        self.low_subscriber.Close()
        for subscriber in self.odom_subscribers:
            subscriber.Close()

    def release_arm_ownership(self) -> None:
        """Best-effort zero blend even if parsing feedback raised an exception."""
        if self.arm_publisher is None:
            return
        self.low_command.motor_cmd[ARM_SDK_WEIGHT_INDEX].q = 0.0
        self.low_command.crc = self._crc.Crc(self.low_command)
        for _ in range(3):
            self.arm_publisher.Write(self.low_command)


def terminal_commands() -> list[str]:
    if not sys.stdin.isatty():
        return []
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return []
    value = sys.stdin.readline().strip().lower()
    return [value] if value in {"s", "x", "q"} else []


def add_runtime_dependencies(project_dir: Path) -> None:
    official = Path("/home/drgh/g1_official/GR00T-WholeBodyControl")
    teleop_site = official / ".venv_teleop/lib/python3.10/site-packages"
    vendor_sdk = project_dir / "vendor/unitree_sdk2_python"
    # Append teleop packages after the active data environment so its Torch,
    # NumPy and LeRobot stay authoritative; only missing DDS modules resolve
    # from the teleop environment.
    for path in (official, vendor_sdk, teleop_site):
        value = str(path)
        if value not in sys.path:
            sys.path.append(value)


def build_parser(project_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skill", choices=("button", "knob", "switch"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--real", action="store_true", help="publish real arm/Dex1 commands")
    mode.add_argument("--dry-run", action="store_true", help="live sensors and ACT, no commands")
    mode.add_argument("--offline-smoke", action="store_true", help="model-only synthetic test")
    parser.add_argument("--model-root", type=Path, default=project_dir / "models/task2_act")
    parser.add_argument(
        "--bounds", type=Path, default=project_dir / "config/task2_act_deploy.json"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--interface", default="enp8s0")
    parser.add_argument("--camera-host", default="192.168.123.222")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--policy-hz", type=float, default=20.0)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--ensemble-decay", type=float, default=0.08)
    parser.add_argument("--cutoff-hz", type=float, default=1.5)
    parser.add_argument("--arm-speed", type=float, default=0.35)
    parser.add_argument("--arm-acceleration", type=float, default=1.0)
    parser.add_argument("--gripper-speed", type=float, default=1.2)
    parser.add_argument("--gripper-acceleration", type=float, default=6.0)
    parser.add_argument("--max-follow-error", type=float, default=0.18)
    parser.add_argument("--arm-weight", type=float, default=0.90)
    parser.add_argument("--weight-rate", type=float, default=0.50)
    parser.add_argument("--right-arm-kp", type=float, default=28.0)
    parser.add_argument("--right-arm-kd", type=float, default=1.8)
    parser.add_argument("--waist-hold-kp", type=float, default=40.0)
    parser.add_argument("--waist-hold-kd", type=float, default=2.0)
    parser.add_argument("--left-arm-hold-kp", type=float, default=28.0)
    parser.add_argument("--left-arm-hold-kd", type=float, default=1.8)
    parser.add_argument("--takeover-hold-seconds", type=float, default=2.0)
    parser.add_argument("--takeover-weight", type=float, default=0.25)
    parser.add_argument("--takeover-weight-rate", type=float, default=0.12)
    parser.add_argument("--takeover-kp", type=float, default=16.0)
    parser.add_argument("--takeover-kd", type=float, default=1.5)
    parser.add_argument("--preposition-speed", type=float, default=0.12)
    parser.add_argument("--preposition-acceleration", type=float, default=0.35)
    parser.add_argument("--preposition-weight-rate", type=float, default=0.10)
    parser.add_argument("--preposition-kp", type=float, default=28.0)
    parser.add_argument("--preposition-kd", type=float, default=1.8)
    parser.add_argument("--preposition-gain-ramp-seconds", type=float, default=5.0)
    # P10-P90 deliberately excludes the edge demonstrations.  Keep a small
    # real-hardware allowance so normal-locomotion/Arm-SDK impedance does not
    # keep pushing a settled joint toward one exact recorded angle.
    parser.add_argument("--preposition-envelope-margin", type=float, default=0.05)
    parser.add_argument("--preposition-hold-seconds", type=float, default=1.0)
    parser.add_argument("--preposition-stable-speed", type=float, default=0.12)
    parser.add_argument("--preposition-timeout", type=float, default=20.0)
    parser.add_argument("--preposition-tracking-error", type=float, default=0.16)
    parser.add_argument("--takeover-max-speed", type=float, default=0.60)
    parser.add_argument("--max-preposition-joint-delta", type=float, default=0.90)
    parser.add_argument("--policy-gain-ramp-seconds", type=float, default=2.0)
    parser.add_argument("--max-measured-joint-speed", type=float, default=1.50)
    parser.add_argument("--max-source-age", type=float, default=0.40)
    parser.add_argument("--max-inference-age", type=float, default=1.0)
    parser.add_argument("--max-tracking-error", type=float, default=0.24)
    parser.add_argument("--max-run-seconds", type=float, default=20.0)
    parser.add_argument(
        "--completion-mode",
        choices=("timeout", "motion-cycle"),
        default="timeout",
        help="finish at max duration, or require measured reach/return/settle cycle",
    )
    parser.add_argument("--completion-excursion", type=float, default=1.50)
    parser.add_argument("--completion-return", type=float, default=0.85)
    parser.add_argument("--completion-stable-speed", type=float, default=1.0)
    parser.add_argument("--completion-hold-seconds", type=float, default=1.5)
    parser.add_argument("--completion-min-seconds", type=float, default=6.0)
    parser.add_argument("--thermal-warn", type=float, default=65.0)
    parser.add_argument("--thermal-stop", type=float, default=70.0)
    parser.add_argument("--thermal-resume", type=float, default=62.0)
    parser.add_argument(
        "--thermal-action",
        choices=("warn", "release"),
        default="warn",
        help="warn only (default), or release Arm SDK at --thermal-stop",
    )
    parser.add_argument("--preflight-timeout", type=float, default=15.0)
    parser.add_argument(
        "--preposition-only",
        action="store_true",
        help="real-mode staged takeover/start-pose test; never runs ACT actions",
    )
    parser.add_argument("--auto-start", action="store_true", help="start automatically after preflight, without typing s")
    parser.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--exit-after-attempt",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name, value in vars(args).items():
        if isinstance(value, float) and not np.isfinite(value):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if args.real and args.device == "cpu":
        parser.error("--real refuses CPU inference; use CUDA")
    if args.preposition_only and not args.real:
        parser.error("--preposition-only requires --real")
    if not 20.0 <= args.control_hz <= 100.0:
        parser.error("--control-hz must be 20..100")
    if not 5.0 <= args.policy_hz <= args.control_hz:
        parser.error("--policy-hz must be 5..control-hz")
    if not 1 <= args.replan_steps <= 20:
        parser.error("--replan-steps must be 1..20")
    if not 0.5 <= args.cutoff_hz <= 5.0:
        parser.error("--cutoff-hz must be 0.5..5")
    if not 0.1 <= args.arm_weight <= 1.0:
        parser.error("--arm-weight must be 0.1..1")
    if not 0 < args.waist_hold_kp <= 60 or not 0 < args.left_arm_hold_kp <= 60:
        parser.error("support kp must be positive and at most the official example value 60")
    if not 0 < args.waist_hold_kd <= 5 or not 0 < args.left_arm_hold_kd <= 5:
        parser.error("support kd must be positive and at most 5")
    if not 0.05 <= args.takeover_weight < args.arm_weight:
        parser.error("--takeover-weight must be below --arm-weight")
    if args.takeover_hold_seconds <= 0 or args.takeover_weight_rate <= 0:
        parser.error("takeover hold/rate must be positive")
    if args.takeover_kp <= 0 or args.takeover_kd <= 0:
        parser.error("takeover gains must be positive")
    if args.preposition_hold_seconds <= 0 or args.preposition_tracking_error <= 0:
        parser.error("preposition hold/tracking error must be positive")
    if not 0 < args.preposition_stable_speed <= args.takeover_max_speed:
        parser.error("preposition stable speed must be positive and at most takeover max speed")
    if args.preposition_speed <= 0 or args.preposition_acceleration <= 0:
        parser.error("preposition speed/acceleration must be positive")
    if args.preposition_weight_rate <= 0 or args.preposition_timeout <= 0:
        parser.error("preposition weight rate/timeout must be positive")
    if not args.takeover_kp <= args.preposition_kp <= 60.0:
        parser.error("preposition kp must be between takeover kp and 60")
    if args.preposition_kd <= 0 or args.preposition_gain_ramp_seconds <= 0:
        parser.error("preposition kd and gain ramp must be positive")
    minimum_arming_seconds = max(
        (args.arm_weight - args.takeover_weight) / args.preposition_weight_rate,
        args.preposition_gain_ramp_seconds,
    ) + args.preposition_hold_seconds
    if args.preposition_timeout <= minimum_arming_seconds:
        parser.error("preposition timeout must exceed the stationary arming ramp plus hold duration")
    if not 0 <= args.preposition_envelope_margin <= 0.10:
        parser.error("preposition envelope margin must be 0..0.10rad")
    if args.max_tracking_error <= args.max_follow_error:
        parser.error("--max-tracking-error must exceed --max-follow-error")
    if args.policy_gain_ramp_seconds <= 0 or args.max_measured_joint_speed <= 0:
        parser.error("policy gain ramp and measured-speed guard must be positive")
    if args.thermal_action == "release" and args.thermal_resume >= args.thermal_stop:
        parser.error("thermal resume must be below thermal stop")
    if args.max_run_seconds <= 0:
        parser.error("--max-run-seconds must be positive")
    if not 0.5 <= args.completion_excursion <= 4.0:
        parser.error("--completion-excursion must be 0.5..4.0rad")
    if not 0.1 <= args.completion_return < args.completion_excursion:
        parser.error("--completion-return must be positive and below excursion")
    if args.completion_stable_speed <= 0 or args.completion_hold_seconds <= 0:
        parser.error("completion speed/hold must be positive")
    if args.completion_min_seconds < 0:
        parser.error("completion minimum must be non-negative")
    if (
        args.completion_mode == "motion-cycle"
        and not args.preposition_only
        and args.completion_min_seconds >= args.max_run_seconds
    ):
        parser.error("completion minimum must be below max run time")


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]
    parser = build_parser(project_dir)
    args = parser.parse_args()
    validate_args(args, parser)
    if args.validate_only:
        print("[ACT] Arguments valid; no model or robot connection started.", flush=True)
        return 0
    print(
        "[ACT][CONTROL] stationary-dq-v3: stationary arming, waist/left support, continuous handoff, "
        "motor dq feedback, interrupt-safe release",
        flush=True,
    )
    add_runtime_dependencies(project_dir)

    model_dir = args.model_root / args.skill
    for required in (model_dir / "config.json", model_dir / "model.safetensors"):
        if not required.is_file():
            raise SystemExit(f"missing model file: {required}")
    action_lower, action_upper = parse_action_bounds(args.bounds, args.skill)
    start_action = parse_start_action(args.bounds, args.skill)
    start_observation_lower, start_observation_upper = (
        parse_start_observation_envelope(args.bounds, args.skill)
    )

    device = args.device
    if args.real and device == "auto":
        device = "cuda"
    print(f"[ACT] Loading {args.skill} policy from {model_dir} on {device}...", flush=True)
    engine = ACTEngine(model_dir, device)
    warmup_seconds = engine.warmup()
    print(
        f"[ACT] Model ready: device={engine.device}, warmup={warmup_seconds:.3f}s, "
        "chunk=100x8",
        flush=True,
    )
    if args.offline_smoke:
        print("OFFLINE_SMOKE_OK: finite 100-step action chunk produced", flush=True)
        return 0

    if args.real and not sys.stdin.isatty():
        raise SystemExit("--real requires an interactive terminal for s/x/q controls")

    worker = InferenceWorker(engine)
    sources: RobotSources | None = None
    stop_requested = False
    emergency_release_requested = False
    signal_exit_code = 0

    def request_stop(_signum=None, _frame=None):
        nonlocal stop_requested, emergency_release_requested, signal_exit_code
        stop_requested = True
        emergency_release_requested = True
        signal_exit_code = 130

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        sources = RobotSources(
            args.interface, args.camera_host, args.camera_port, args.real,
            args.waist_hold_kp, args.waist_hold_kd,
            args.left_arm_hold_kp, args.left_arm_hold_kd,
        )
        sources.emergency_stop_requested = lambda: emergency_release_requested
        print("[ACT] Waiting for fresh G1, Dex1 and dual-camera data...", flush=True)
        deadline = time.monotonic() + args.preflight_timeout
        detail = "not checked"
        while time.monotonic() < deadline and not stop_requested:
            ready, detail = sources.readiness(args.max_source_age)
            if ready:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(f"preflight timed out: {detail}")
        measured = sources.measured()
        if measured is None:
            raise RuntimeError("G1/Dex1 state disappeared after preflight")
        temperatures = measured.temperatures
        hottest_index = int(np.argmax(temperatures))
        if (
            args.thermal_action == "release"
            and temperatures[hottest_index] >= args.thermal_stop
        ):
            raise RuntimeError(
                f"{JOINT_NAMES[hottest_index]} is already "
                f"{temperatures[hottest_index]:.1f}C; cool below "
                f"{args.thermal_resume:.1f}C before inference"
            )

        ensembler = TemporalChunkEnsembler(args.max_chunks, args.ensemble_decay)
        smoother = SmoothActionFilter(
            action_lower,
            action_upper,
            args.cutoff_hz,
            args.arm_speed,
            args.arm_acceleration,
            args.gripper_speed,
            args.gripper_acceleration,
            args.max_follow_error,
        )
        # Prepositioning may legitimately begin just outside a skill's narrow
        # demonstrated envelope.  Using that envelope here would clip the
        # very first command to its boundary and could create a step.  The
        # staging filter therefore uses only the robot's physical limits; the
        # policy filter above still enforces the tighter demonstrated range.
        prepositioner = SmoothActionFilter(
            np.concatenate((RIGHT_ARM_COMMAND_LOWER, [GRIPPER_LOWER])),
            np.concatenate((RIGHT_ARM_COMMAND_UPPER, [GRIPPER_UPPER])),
            1.0,
            args.preposition_speed,
            args.preposition_acceleration,
            min(args.gripper_speed, 0.60),
            min(args.gripper_acceleration, 2.0),
            args.preposition_tracking_error,
        )
        completion = MotionCycleCompletion(
            args.completion_excursion,
            args.completion_return,
            args.completion_stable_speed,
            args.completion_hold_seconds,
            args.completion_min_seconds,
        )
        episode_id = 0
        state = "IDLE"
        episode_clock_started = 0.0
        prefill_started = 0.0
        takeover_started = 0.0
        arming_started = 0.0
        arming_action = None
        arming_stable_for = 0.0
        preposition_started = 0.0
        preposition_stable_for = 0.0
        last_result_at = 0.0
        last_latency = 0.0
        next_replan_step = 0
        last_submitted_step = -1
        arm_weight = 0.0
        last_tick = time.monotonic()
        last_report = 0.0
        thermal_warning_at = 0.0
        release_action: np.ndarray | None = None
        exit_when_released = False
        attempt_exit_code = 0
        previous_measurement: RobotMeasurement | None = None
        measured_joint_speed = 0.0
        start_arm_error = 0.0
        start_gripper_error = 0.0
        start_envelope_error = 0.0
        start_envelope_index = 0

        if args.preposition_only:
            print(
                "[ACT][预定位测试] 只执行接管和起点定位，不执行按钮/旋钮/拨杆动作。"
                "通过后自动释放右臂，回到常规运控姿态属于预期行为。",
                flush=True,
            )
        elif args.real:
            print(
                f"[ACT][模型执行] {args.skill}：预定位通过后继续执行 ACT；"
                f"策略运行时限 {args.max_run_seconds:g} 秒。",
                flush=True,
            )
        print(
            f"[ACT] {'REAL' if args.real else 'DRY-RUN'} ready for {args.skill}. "
            "Place the robot at the demonstrated start pose. "
            + ("自动开始，无需输入 s；x+Enter 释放，q+Enter 退出，Ctrl+C 软件紧急释放。"
               if args.auto_start else
               "Type s+Enter to start, x+Enter to release, q+Enter to quit."),
            flush=True,
        )

        def begin_attempt(now: float) -> bool:
            nonlocal episode_id, state, prefill_started, takeover_started
            nonlocal preposition_started, preposition_stable_for, next_replan_step
            nonlocal last_submitted_step, arm_weight, last_result_at, release_action
            nonlocal previous_measurement, start_arm_error, start_gripper_error
            nonlocal start_envelope_error, start_envelope_index
            ready, reason = sources.readiness(args.max_source_age)
            latest = sources.measured()
            if not ready or latest is None:
                print(f"[ACT] Cannot start: {reason}", flush=True)
                return False
            body, gripper, temps = latest.body, latest.gripper, latest.temperatures
            try:
                sources.capture_body_hold(latest)
            except ValueError as exc:
                print(f"[ACT] Cannot start: {exc}", flush=True)
                return False
            hot = int(np.argmax(temps))
            if args.thermal_action == "release" and temps[hot] >= args.thermal_stop:
                print(
                    f"[ACT] Cannot start: {JOINT_NAMES[hot]}={temps[hot]:.1f}C",
                    flush=True,
                )
                return False
            episode_id += 1
            next_replan_step = 0
            last_submitted_step = -1
            last_result_at = 0.0
            arm_weight = 0.0
            release_action = None
            preposition_stable_for = 0.0
            ensembler.clear()
            worker.clear_requests()
            measured_action = np.concatenate(
                (body[RIGHT_ARM_BODY_INDICES], [gripper])
            )
            smoother.reset(measured_action)
            prepositioner.reset(measured_action)
            completion.reset(body[RIGHT_ARM_BODY_INDICES])
            previous_measurement = latest
            start_arm_error = float(np.max(np.abs(
                start_action[:7] - measured_action[:7]
            )))
            start_gripper_error = float(abs(start_action[7] - measured_action[7]))
            below_envelope = np.maximum(
                start_observation_lower - measured_action, 0.0
            )
            above_envelope = np.maximum(
                measured_action - start_observation_upper, 0.0
            )
            envelope_error_vector = np.maximum(below_envelope, above_envelope)
            start_envelope_index = int(np.argmax(envelope_error_vector))
            start_envelope_error = float(
                envelope_error_vector[start_envelope_index]
            )
            if args.real and start_arm_error > args.max_preposition_joint_delta:
                print(
                    f"[ACT] Cannot start: demonstrated start pose differs by "
                    f"{start_arm_error:.2f}rad on one joint; limit is "
                    f"{args.max_preposition_joint_delta:.2f}rad",
                    flush=True,
                )
                return False
            if args.real:
                state = "TAKEOVER"
                takeover_started = now
                preposition_started = 0.0
                print(
                    f"[ACT] Attempt {episode_id}: staged takeover; start error "
                    f"arm={start_arm_error:.2f}rad gripper={start_gripper_error:.2f}rad; "
                    f"training-envelope error={start_envelope_error:.2f}rad at "
                    f"{RIGHT_ACTION_NAMES[start_envelope_index]}. "
                    "Holding measured pose before slow prepositioning.",
                    flush=True,
                )
            else:
                state = "PREFILL"
                prefill_started = now
                last_submitted_step = 0
                worker.submit(sources.observation(episode_id, 0, args.max_source_age))
                print(
                    f"[ACT] Attempt {episode_id}: computing first action chunk...",
                    flush=True,
                )
            return True

        def begin_release(reason: str, failure: bool = False) -> None:
            nonlocal state, release_action, exit_when_released
            nonlocal attempt_exit_code, stop_requested
            if failure:
                attempt_exit_code = max(attempt_exit_code, 5)
            if state in {"IDLE", "RELEASING"}:
                return
            latest = sources.measured()
            if latest is not None:
                body, gripper = latest.body, latest.gripper
                release_action = np.concatenate((body[RIGHT_ARM_BODY_INDICES], [gripper]))
            else:
                release_action = smoother.command.copy()
            state = "RELEASING" if arm_weight > 1.0e-4 else "IDLE"
            ensembler.clear()
            worker.clear_requests()
            print(f"[ACT] Releasing: {reason}", flush=True)
            if args.exit_after_attempt:
                exit_when_released = True
                if state == "IDLE":
                    stop_requested = True

        if args.auto_start:
            if not begin_attempt(time.monotonic()):
                raise RuntimeError("automatic attempt could not start")

        period = 1.0 / args.control_hz
        while not stop_requested:
            tick = time.monotonic()
            dt = min(max(tick - last_tick, 0.001), 0.1)
            last_tick = tick
            sources.poll_camera()
            if stop_requested:
                break

            for command in terminal_commands():
                if command == "q":
                    begin_release("operator quit", failure=args.exit_after_attempt)
                    stop_requested = True
                elif command == "x":
                    begin_release("operator stop", failure=args.exit_after_attempt)
                elif command == "s":
                    if state == "IDLE":
                        begin_attempt(tick)
                    else:
                        begin_release(
                            "operator toggled stop", failure=args.exit_after_attempt
                        )

            if stop_requested:
                break

            latest = sources.measured()
            if latest is None:
                print("[ACT] G1/Dex1 state missing; releasing Arm SDK and exiting", flush=True)
                # Fading needs feedback. Waiting here would skip the release
                # state forever if feedback never returns.
                emergency_release_requested = True
                attempt_exit_code = 5
                stop_requested = True
                break
            body, gripper, temperatures = latest.body, latest.gripper, latest.temperatures
            measured_action = np.concatenate((body[RIGHT_ARM_BODY_INDICES], [gripper]))
            if not np.isfinite(measured_action).all() or not math.isfinite(latest.arm_speed):
                print("[ACT] Invalid joint feedback; releasing Arm SDK", flush=True)
                # Do not put invalid feedback into the outgoing command.
                sources.release_arm_ownership()
                arm_weight = 0.0
                attempt_exit_code = 5
                stop_requested = True
                break
            measured_joint_speed = latest.arm_speed
            speed_detail = latest.speed_detail(previous_measurement, time.monotonic())
            previous_measurement = latest
            start_arm_error = float(np.max(np.abs(
                start_action[:7] - measured_action[:7]
            )))
            start_gripper_error = float(abs(start_action[7] - measured_action[7]))
            below_envelope = np.maximum(
                start_observation_lower - measured_action, 0.0
            )
            above_envelope = np.maximum(
                measured_action - start_observation_upper, 0.0
            )
            envelope_error_vector = np.maximum(below_envelope, above_envelope)
            start_envelope_index = int(np.argmax(envelope_error_vector))
            start_envelope_error = float(
                envelope_error_vector[start_envelope_index]
            )
            hottest_index = int(np.argmax(temperatures))
            hottest = float(temperatures[hottest_index])
            if hottest >= args.thermal_warn and tick - thermal_warning_at >= 1.0:
                thermal_detail = (
                    f"automatic release at {args.thermal_stop:.1f}C"
                    if args.thermal_action == "release"
                    else "warning only; onboard robot protection remains active"
                )
                print(
                    f"[ACT][TEMP] {JOINT_NAMES[hottest_index]}={hottest:.1f}C "
                    f"({thermal_detail})",
                    flush=True,
                )
                thermal_warning_at = tick
            if args.thermal_action == "release" and hottest >= args.thermal_stop:
                begin_release(
                    f"temperature {JOINT_NAMES[hottest_index]}={hottest:.1f}C",
                    failure=True,
                )

            ready, source_detail = sources.readiness(args.max_source_age)
            if stop_requested:
                break
            if not ready and state in {"TAKEOVER", "ARMING", "PREPOSITION", "PREFILL", "RUNNING"}:
                begin_release(f"stale source: {source_detail}", failure=True)

            body_reason, body_metrics = sources.body_motion(latest)
            if args.real and arm_weight > 0 and (
                body_reason or sources._odometry is None
                or tick - sources._odometry[2] > args.max_source_age
            ):
                print(
                    f"[ACT][BODY STOP] {body_reason or 'body odometry stale'}; "
                    "立即释放 Arm SDK，请现场确认站稳。", flush=True,
                )
                release_action = measured_action.copy()
                emergency_release_requested = True
                attempt_exit_code = 5
                stop_requested = True
                break

            for result in worker.drain():
                if result.episode_id != episode_id:
                    continue
                if result.error is not None:
                    begin_release(f"inference failed: {result.error}", failure=True)
                    continue
                assert result.actions is not None
                try:
                    ensembler.add(ActionChunk(
                        result.episode_id,
                        result.policy_step,
                        result.actions,
                        result.completed_at,
                    ))
                except ValueError as exc:
                    begin_release(str(exc), failure=True)
                    continue
                last_result_at = result.completed_at
                last_latency = result.latency
                if state == "PREFILL":
                    state = "RUNNING"
                    episode_clock_started = tick
                    next_replan_step = args.replan_steps
                    if args.real:
                        smoother.handoff_from(prepositioner)
                    else:
                        smoother.reset(measured_action)
                    completion.reset(measured_action[:7])
                    print(
                        f"[ACT] Attempt {episode_id} RUNNING; first inference "
                        f"{last_latency * 1000:.0f}ms",
                        flush=True,
                    )

            if state == "TAKEOVER":
                takeover_elapsed = tick - takeover_started
                if (
                    takeover_elapsed >= 0.25
                    and measured_joint_speed > args.takeover_max_speed
                ):
                    begin_release(
                        f"unexpected motion during takeover: "
                        f"{measured_joint_speed:.2f}rad/s; {speed_detail}",
                        failure=True,
                    )
                else:
                    arm_weight = min(
                        args.takeover_weight,
                        arm_weight + args.takeover_weight_rate * dt,
                    )
                    # Track the measured pose while ownership is introduced.
                    # This changes impedance gradually without requesting a
                    # joint displacement from normal locomotion.
                    sources.publish(
                        measured_action,
                        arm_weight,
                        args.takeover_kp,
                        args.takeover_kd,
                    )
                    if (
                        takeover_elapsed >= args.takeover_hold_seconds
                        and arm_weight >= args.takeover_weight - 1.0e-3
                    ):
                        state = "ARMING"
                        arming_started = tick
                        arming_stable_for = 0.0
                        arming_action = measured_action.copy()
                        prepositioner.reset(arming_action)
                        print(
                            "[ACT] TAKEOVER stable; ARMING holds the current arm/gripper "
                            "pose while weight and gains ramp. No preposition motion yet.",
                            flush=True,
                        )

            elif state == "ARMING":
                arming_elapsed = tick - arming_started
                hold_error = float(np.max(np.abs(arming_action[:7] - measured_action[:7])))
                if arming_elapsed > args.preposition_timeout:
                    begin_release("stationary arming timed out", failure=True)
                elif measured_joint_speed > args.takeover_max_speed:
                    begin_release(
                        f"unexpected motion during stationary arming: {measured_joint_speed:.2f}rad/s; "
                        f"{speed_detail}; weight={arm_weight:.3f}", failure=True,
                    )
                elif hold_error > args.preposition_tracking_error:
                    begin_release(f"stationary arming hold error {hold_error:.3f}rad", failure=True)
                else:
                    arm_weight = min(args.arm_weight, arm_weight + args.preposition_weight_rate * dt)
                    fraction = min(1., arming_elapsed / args.preposition_gain_ramp_seconds)
                    kp = args.takeover_kp + fraction * (args.preposition_kp - args.takeover_kp)
                    kd = args.takeover_kd + fraction * (args.preposition_kd - args.takeover_kd)
                    sources.publish(arming_action, arm_weight, kp, kd)
                    settled = (
                        arm_weight >= args.arm_weight - 1.e-3 and fraction >= 1.
                        and measured_joint_speed <= args.preposition_stable_speed
                    )
                    arming_stable_for = arming_stable_for + dt if settled else 0.
                    if settled and arming_stable_for >= args.preposition_hold_seconds:
                        state = "PREPOSITION"
                        preposition_started = tick
                        preposition_stable_for = 0.
                        # Preserve the holding error/effort at the handoff.
                        prepositioner.reset(arming_action)
                        print(
                            "[ACT] ARMING stable; ownership/gains fixed. "
                            f"Slowly approaching {args.skill} start pose.", flush=True,
                        )

            elif state == "PREPOSITION":
                preposition_elapsed = tick - preposition_started
                if preposition_elapsed > args.preposition_timeout:
                    begin_release(
                        f"preposition timed out: arm error "
                        f"{start_arm_error:.2f}rad; training-envelope error "
                        f"{start_envelope_error:.2f}rad at "
                        f"{RIGHT_ACTION_NAMES[start_envelope_index]}",
                        failure=True,
                    )
                elif measured_joint_speed > args.takeover_max_speed:
                    begin_release(
                        f"unexpected motion during preposition: "
                        f"{measured_joint_speed:.2f}rad/s; {speed_detail}; "
                        f"weight={arm_weight:.3f} kp={args.preposition_kp:.1f} "
                        f"last_target={np.round(prepositioner.command[:7], 4).tolist()}",
                        failure=True,
                    )
                else:
                    command = prepositioner.step(start_action, measured_action, dt)
                    tracking_vector = np.abs(command[:7] - measured_action[:7])
                    tracking_index = int(np.argmax(tracking_vector))
                    tracking_error = float(tracking_vector[tracking_index])
                    if tracking_error > args.preposition_tracking_error + 1.0e-6:
                        begin_release(
                            f"preposition tracking error "
                            f"{tracking_error:.2f}rad at "
                            f"{JOINT_NAMES[22 + tracking_index]}",
                            failure=True,
                        )
                    else:
                        sources.publish(
                            command,
                            arm_weight,
                            args.preposition_kp,
                            args.preposition_kd,
                        )
                        at_start = (
                            start_envelope_error
                            <= args.preposition_envelope_margin
                            and measured_joint_speed
                            <= args.preposition_stable_speed
                            and arm_weight >= args.arm_weight - 1.0e-3
                        )
                        preposition_stable_for = (
                            preposition_stable_for + dt if at_start else 0.0
                        )
                        if (
                            preposition_stable_for + 1.0e-9
                            >= args.preposition_hold_seconds
                        ):
                            if args.preposition_only:
                                print(
                                    "[ACT][预定位通过] 已稳定达到起点判据；"
                                    "本次未执行 ACT 动作，现在按测试流程释放右臂。",
                                    flush=True,
                                )
                                begin_release(
                                    "preposition-only test reached and held the "
                                    "demonstration start pose"
                                )
                                exit_when_released = True
                            else:
                                state = "PREFILL"
                                prefill_started = tick
                                last_submitted_step = 0
                                smoother.reset(measured_action)
                                completion.reset(measured_action[:7])
                                worker.submit(sources.observation(
                                    episode_id, 0, args.max_source_age
                                ))
                                print(
                                    f"[ACT] Demonstration start pose stable for "
                                    f"{preposition_stable_for:.1f}s; computing "
                                    "first action chunk.",
                                    flush=True,
                                )

            if stop_requested:
                break

            if state == "PREFILL" and tick - prefill_started > max(
                5.0, 3.0 * args.max_inference_age
            ):
                begin_release("first inference timed out", failure=True)
            elif state == "PREFILL" and args.real:
                # Continue holding the verified start pose while the first GPU
                # inference completes.  This normally lasts well under 0.2s.
                command = prepositioner.step(start_action, measured_action, dt)
                sources.publish(
                    command,
                    arm_weight,
                    args.preposition_kp,
                    args.preposition_kd,
                )

            if state == "RUNNING":
                elapsed = tick - episode_clock_started
                policy_step = elapsed * args.policy_hz
                step_integer = int(math.floor(policy_step))
                cycle_complete = completion.update(
                    measured_action[:7], dt, elapsed
                )
                if (
                    args.real
                    and measured_joint_speed > args.max_measured_joint_speed
                ):
                    begin_release(
                        f"measured joint speed {measured_joint_speed:.2f}rad/s "
                        f"exceeded safety guard; {speed_detail}",
                        failure=True,
                    )
                elif args.completion_mode == "motion-cycle" and cycle_complete:
                    begin_release(
                        "measured reach/interact/return cycle completed and settled"
                    )
                elif elapsed >= args.max_run_seconds:
                    if args.completion_mode == "motion-cycle":
                        begin_release(
                            f"completion was not detected before "
                            f"{args.max_run_seconds:.1f}s timeout",
                            failure=True,
                        )
                        # A sequence may elect to continue after an ordinary
                        # skill timeout, but must still stop for sensor, body,
                        # tracking, inference or operator failures (exit 5/130).
                        attempt_exit_code = max(attempt_exit_code, 6)
                    else:
                        begin_release(
                            f"{args.max_run_seconds:.1f}s attempt limit reached"
                        )
                elif last_result_at and tick - last_result_at > args.max_inference_age:
                    begin_release(
                        f"policy output stale for {tick - last_result_at:.2f}s",
                        failure=True,
                    )
                else:
                    if (
                        step_integer >= next_replan_step
                        and step_integer != last_submitted_step
                    ):
                        try:
                            observation = sources.observation(
                                episode_id, step_integer, args.max_source_age
                            )
                            worker.submit(observation)
                            last_submitted_step = step_integer
                            next_replan_step = step_integer + args.replan_steps
                        except RuntimeError as exc:
                            begin_release(f"cannot replan: {exc}", failure=True)
                    if stop_requested:
                        break
                    raw_action = ensembler.action_at(episode_id, policy_step)
                    if raw_action is None:
                        begin_release("no time-aligned action remains", failure=True)
                    else:
                        command = smoother.step(raw_action, measured_action, dt)
                        tracking_vector = np.abs(
                            command[:7] - measured_action[:7]
                        )
                        tracking_index = int(np.argmax(tracking_vector))
                        tracking_error = float(tracking_vector[tracking_index])
                        if tracking_error > args.max_tracking_error:
                            begin_release(
                                f"right-arm tracking error {tracking_error:.2f}rad at "
                                f"{JOINT_NAMES[22 + tracking_index]}",
                                failure=True,
                            )
                        else:
                            if args.real:
                                arm_weight = min(
                                    args.arm_weight,
                                    arm_weight + args.weight_rate * dt,
                                )
                                gain_fraction = min(
                                    1.0,
                                    elapsed / args.policy_gain_ramp_seconds,
                                )
                                kp = args.preposition_kp + gain_fraction * (
                                    args.right_arm_kp - args.preposition_kp
                                )
                                kd = args.preposition_kd + gain_fraction * (
                                    args.right_arm_kd - args.preposition_kd
                                )
                                sources.publish(command, arm_weight, kp, kd)
                            else:
                                arm_weight = 0.0

            if state == "RELEASING":
                arm_weight = max(0.0, arm_weight - args.weight_rate * 2.0 * dt)
                # Follow the measured pose while fading the shared Arm-SDK
                # blend.  Holding the first release pose would fight normal
                # locomotion if balance control moves the torso/arm meanwhile.
                release_action = measured_action.copy()
                if release_action is not None and arm_weight > 0.0:
                    sources.publish(
                        release_action,
                        arm_weight,
                        args.takeover_kp,
                        args.takeover_kd,
                    )
                else:
                    # Send the final zero weight, not merely stop publishing
                    # with the previous tick's small nonzero weight latched.
                    sources.publish(
                        measured_action, 0.0, args.takeover_kp, args.takeover_kd
                    )
                    state = "IDLE"
                    arm_weight = 0.0
                    if exit_when_released:
                        print("[ACT] Attempt released; exiting for next policy", flush=True)
                        stop_requested = True
                    else:
                        print("[ACT] Arm SDK released; ready for s+Enter", flush=True)

            if tick - last_report >= 1.0:
                active_filter = (
                    prepositioner
                    if state in {"TAKEOVER", "ARMING", "PREPOSITION", "PREFILL"}
                    else smoother
                )
                tracking_vector = (
                    np.abs(active_filter.command[:7] - measured_action[:7])
                    if active_filter.initialized
                    else np.zeros(7, dtype=np.float64)
                )
                tracking_index = int(np.argmax(tracking_vector))
                tracking = float(tracking_vector[tracking_index])
                start_vector = np.abs(start_action[:7] - measured_action[:7])
                start_index = int(np.argmax(start_vector))
                print(
                    f"[ACT][{state}] skill={args.skill} arm_w={arm_weight:.2f} "
                    f"chunks={ensembler.size} infer={last_latency * 1000:.0f}ms "
                    f"track={tracking:.2f}rad@{JOINT_NAMES[22 + tracking_index]} "
                    f"speed={measured_joint_speed:.2f}rad/s@{RIGHT_ACTION_NAMES[latest.speed_index]}(motor_dq) "
                    f"start={start_arm_error:.2f}rad@{JOINT_NAMES[22 + start_index]} "
                    f"envelope={start_envelope_error:.2f}rad@"
                    f"{RIGHT_ACTION_NAMES[start_envelope_index]} "
                    f"temp={hottest:.1f}C "
                    f"cycle=max{completion.max_excursion:.2f}/"
                    f"now{completion.current_excursion:.2f}/"
                    f"stable{completion.stable_for:.1f}s",
                    flush=True,
                )
                if body_metrics and state != "IDLE":
                    print(
                        "[ACT][BODY] " + " ".join(
                            f"{key}={value:.3f}" for key, value in body_metrics.items()
                        ) + " (angles=rad, travel=m, speed=m/s)",
                        flush=True,
                    )
                last_report = tick

            sleep_time = period - (time.monotonic() - tick)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Ctrl+C/SIGTERM is the software emergency-release path.  Do not keep
        # fading a potentially bad ACT command for another second: publish a
        # zero Arm-SDK blend immediately at the latest measured pose, then
        # close the publishers.  x/q still use the normal smooth release.
        if arm_weight > 1.0e-4 and release_action is None and not emergency_release_requested:
            latest = sources.measured()
            if latest is not None:
                body, gripper = latest.body, latest.gripper
                release_action = np.concatenate(
                    (body[RIGHT_ARM_BODY_INDICES], [gripper])
                )
            elif smoother.initialized:
                release_action = smoother.command.copy()
        if emergency_release_requested:
            # Relinquish ownership using the last valid command. Do not
            # copy possibly corrupt feedback into motor or gripper commands.
            sources.release_arm_ownership()
            arm_weight = 0.0
            print("[ACT] Software stop: Arm SDK zero weight published", flush=True)
        elif arm_weight > 1.0e-4 and release_action is not None:
            deadline = time.monotonic() + 1.0
            while arm_weight > 1.0e-4 and time.monotonic() < deadline:
                if emergency_release_requested:
                    sources.release_arm_ownership()
                    arm_weight = 0.0
                    break
                arm_weight = max(0.0, arm_weight - args.weight_rate * 2.0 * period)
                sources.publish(
                    release_action,
                    arm_weight,
                    args.takeover_kp,
                    args.takeover_kd,
                )
                time.sleep(period)
        return max(attempt_exit_code, signal_exit_code)
    finally:
        # Worker/camera shutdown may block; relinquish control before either.
        # This also covers unexpected exceptions outside normal state exits.
        if sources is not None:
            try:
                sources.release_arm_ownership()
            except Exception as exc:
                print(f"[ACT] Arm SDK release write failed: {exc}; use hardware stop", flush=True)
        worker.close()
        if sources is not None:
            sources.close()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Task-2 right-arm-only PICO teleoperation through Unitree ``rt/arm_sdk``.

The normal locomotion controller keeps the robot standing. While the right Grip
is held, this process moves the right seven-DoF arm. The tracked controller
drives a full 6D tool pose. All seven arm joints are solved together inside
their physical limits; no elbow or wrist angle is artificially locked.

Controls:
  right grip (deadman)   follow tracked right-hand translation
  right trigger         close/open right Dex1
  right stick left/right rotate tool about its forward axis (knob task)
  left grip + A         start/stop one collection episode
  left grip + B         abort the current episode
  A+B+X+Y               release Arm SDK and exit
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import signal
import time
from typing import Any

import msgpack
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import zmq

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
from pico_wrist_mapping import ControllerPoseFilter, WristMappingConfig


BODY_JOINT_COUNT = 29
ARM_SDK_WEIGHT_INDEX = 29
RIGHT_ARM_BODY_INDICES = np.arange(22, 29)
RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
WAIST_BODY_INDICES = np.asarray([12, 13, 14], dtype=np.int64)
TOOL_OFFSET_LOCAL = np.asarray([0.18, 0.025, 0.0], dtype=np.float64)
UNITY_TO_ROBOT = np.asarray(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
# This is the root-frame offset used by the official SONIC PICO processing
# path (pico_manager_thread_server.py, OFFSETS[0]).  Applying it as part of
# the pelvis frame is different from mixing controller-world and body-root
# poses and keeps the tracked SMPL avatar and robot translation consistent.
OFFICIAL_PICO_ROOT_OFFSET = Rotation.from_euler(
    "z", -90.0, degrees=True
).as_matrix()
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


def normalized_xyzw(value: Any) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion).all() or norm <= 1.0e-9:
        raise ValueError("invalid quaternion")
    return quaternion / norm


def controller_position_in_root(controller_pose: Any, root_pose: Any) -> np.ndarray:
    """Convert a Unity controller position to the robot root-local frame."""
    return tracked_position_in_root(controller_pose, root_pose)


def tracked_position_in_root(tracked_pose: Any, root_pose: Any) -> np.ndarray:
    """Convert any XR tracked pose position to the robot root-local frame."""
    controller = np.asarray(tracked_pose, dtype=np.float64).reshape(7)
    root = np.asarray(root_pose, dtype=np.float64).reshape(7)
    if not np.isfinite(controller).all() or not np.isfinite(root).all():
        raise ValueError("controller/root pose is not finite")
    root_rotation = Rotation.from_quat(normalized_xyzw(root[3:])).as_matrix()
    robot_root_rotation = (
        UNITY_TO_ROBOT
        @ root_rotation
        @ UNITY_TO_ROBOT.T
        @ OFFICIAL_PICO_ROOT_OFFSET
    )
    robot_delta = UNITY_TO_ROBOT @ (controller[:3] - root[:3])
    return robot_root_rotation.T @ robot_delta


def tracked_rotation_in_root(tracked_pose: Any, root_pose: Any) -> np.ndarray:
    """Convert an XR tracked orientation to the same robot-root frame."""
    tracked = np.asarray(tracked_pose, dtype=np.float64).reshape(7)
    root = np.asarray(root_pose, dtype=np.float64).reshape(7)
    if not np.isfinite(tracked).all() or not np.isfinite(root).all():
        raise ValueError("tracked/root pose is not finite")
    root_rotation = Rotation.from_quat(normalized_xyzw(root[3:])).as_matrix()
    tracked_rotation = Rotation.from_quat(
        normalized_xyzw(tracked[3:])
    ).as_matrix()
    robot_root_rotation = (
        UNITY_TO_ROBOT
        @ root_rotation
        @ UNITY_TO_ROBOT.T
        @ OFFICIAL_PICO_ROOT_OFFSET
    )
    robot_tracked_rotation = (
        UNITY_TO_ROBOT @ tracked_rotation @ UNITY_TO_ROBOT.T
    )
    return robot_root_rotation.T @ robot_tracked_rotation


def normalized_direction(value: Any) -> np.ndarray:
    direction = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction).all() or norm <= 1.0e-6:
        raise ValueError("invalid direction")
    return direction / norm


def rotation_between_vectors(source: Any, target: Any) -> np.ndarray:
    """Return the minimum rotation mapping ``source`` onto ``target``."""
    source_unit = normalized_direction(source)
    target_unit = normalized_direction(target)
    dot = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
    if dot >= 1.0 - 1.0e-9:
        return np.eye(3)
    if dot <= -1.0 + 1.0e-9:
        basis = np.asarray([1.0, 0.0, 0.0])
        if abs(float(np.dot(source_unit, basis))) > 0.8:
            basis = np.asarray([0.0, 1.0, 0.0])
        axis = normalized_direction(np.cross(source_unit, basis))
        return Rotation.from_rotvec(math.pi * axis).as_matrix()
    axis = normalized_direction(np.cross(source_unit, target_unit))
    return Rotation.from_rotvec(math.acos(dot) * axis).as_matrix()


def clamp_sphere(point: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    delta = point - center
    distance = float(np.linalg.norm(delta))
    if distance <= radius or distance <= 1.0e-9:
        return point.copy()
    return center + delta * (radius / distance)


def rate_limit_vector(
    previous: np.ndarray, desired: np.ndarray, max_rate: float, dt: float
) -> np.ndarray:
    previous = np.asarray(previous, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    maximum = max(0.0, float(max_rate) * max(0.0, float(dt)))
    return previous + np.clip(desired - previous, -maximum, maximum)


def apply_deadzone(value: float, deadzone: float = 0.15) -> float:
    value = float(np.clip(value, -1.0, 1.0))
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    return math.copysign((magnitude - deadzone) / (1.0 - deadzone), value)


def update_deadman(
    previous: bool,
    grip: float,
    xr_fresh: bool,
    thermal_paused: bool,
) -> bool:
    """Grip deadman with hysteresis, but no latching across stale XR data."""
    if not xr_fresh or thermal_paused:
        return False
    threshold = 0.20 if previous else 0.55
    return float(grip) >= threshold


def ik_solution_is_feasible(
    position_error: float,
    rotation_error: float,
    maximum_position_error: float = 0.10,
    maximum_rotation_error: float = math.radians(55.0),
) -> bool:
    """Reject saturated IK solutions before they create sustained arm load."""
    values = (position_error, rotation_error)
    return (
        all(math.isfinite(value) for value in values)
        and position_error <= maximum_position_error
        and rotation_error <= maximum_rotation_error
    )


class RightArmIK:
    """Full-pose right-arm IK bounded only by physical joint limits."""

    def __init__(
        self,
        orientation_weight: float = 0.12,
        posture_weight: float = 0.008,
    ):
        import pinocchio as pin

        self.pin = pin
        self.robot = instantiate_g1_robot_model()
        self.model = self.robot.pinocchio_wrapper.model
        self.data = self.robot.pinocchio_wrapper.data
        self.frame_id = self.model.getFrameId("right_wrist_yaw_link")
        self.shoulder_frame_id = self.model.getFrameId("right_shoulder_pitch_link")
        self.elbow_frame_id = self.model.getFrameId("right_elbow_link")
        self.body_indices = self.robot.get_body_actuated_joint_indices()
        self.active_indices = np.asarray(
            [self.robot.dof_index(name) for name in RIGHT_ARM_JOINT_NAMES],
            dtype=np.int64,
        )
        self.lower = np.asarray(
            [self.robot.lower_joint_limits[index] for index in self.active_indices],
            dtype=np.float64,
        )
        self.upper = np.asarray(
            [self.robot.upper_joint_limits[index] for index in self.active_indices],
            dtype=np.float64,
        )
        self.orientation_weight = float(orientation_weight)
        self.posture_weight = float(posture_weight)

    def full_configuration(self, body_q: np.ndarray) -> np.ndarray:
        body = np.asarray(body_q, dtype=np.float64).reshape(BODY_JOINT_COUNT)
        return self.robot.get_configuration_from_actuated_joints(body)

    def body_configuration(self, full_q: np.ndarray) -> np.ndarray:
        return np.asarray(full_q[self.body_indices], dtype=np.float64)

    def tool_pose(self, full_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.pin.framesForwardKinematics(self.model, self.data, full_q)
        placement = self.data.oMf[self.frame_id]
        position = placement.translation + placement.rotation @ TOOL_OFFSET_LOCAL
        return position.copy(), placement.rotation.copy()

    def shoulder_position(self, full_q: np.ndarray) -> np.ndarray:
        self.pin.framesForwardKinematics(self.model, self.data, full_q)
        return self.data.oMf[self.shoulder_frame_id].translation.copy()

    def elbow_position(self, full_q: np.ndarray) -> np.ndarray:
        self.pin.framesForwardKinematics(self.model, self.data, full_q)
        return self.data.oMf[self.elbow_frame_id].translation.copy()

    def neutral_orientation(self, body_q: np.ndarray) -> np.ndarray:
        q = self.full_configuration(body_q)
        for local_index in (4, 5, 6):
            q[self.active_indices[local_index]] = 0.0
        return self.tool_pose(q)[1]

    def _tool_jacobian(self, full_q: np.ndarray) -> np.ndarray:
        self.pin.computeJointJacobians(self.model, self.data, full_q)
        self.pin.updateFramePlacements(self.model, self.data)
        jacobian = self.pin.getFrameJacobian(
            self.model,
            self.data,
            self.frame_id,
            self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).copy()
        placement = self.data.oMf[self.frame_id]
        offset_world = placement.rotation @ TOOL_OFFSET_LOCAL
        jacobian[:3] -= self.pin.skew(offset_world) @ jacobian[3:]
        return jacobian

    def _elbow_jacobian(self, full_q: np.ndarray) -> np.ndarray:
        self.pin.computeJointJacobians(self.model, self.data, full_q)
        self.pin.updateFramePlacements(self.model, self.data)
        return self.pin.getFrameJacobian(
            self.model,
            self.data,
            self.elbow_frame_id,
            self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).copy()

    def solve(
        self,
        seed_body_q: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        q = self.full_configuration(seed_body_q)
        seed_active = np.clip(
            q[self.active_indices], self.lower + 1.0e-6, self.upper - 1.0e-6
        )
        q[self.active_indices] = seed_active
        target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
        target_rotation = np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)

        def residual(active_q: np.ndarray) -> np.ndarray:
            q[self.active_indices] = active_q
            current_position, current_rotation = self.tool_pose(q)
            rotation_error = Rotation.from_matrix(
                target_rotation.T @ current_rotation
            ).as_rotvec()
            return np.concatenate((
                current_position - target_position,
                self.orientation_weight * rotation_error,
                self.posture_weight * (active_q - seed_active),
            ))

        initial_residual = residual(seed_active)
        if (
            float(np.linalg.norm(initial_residual[:3])) >= 1.0e-5
            or float(np.linalg.norm(initial_residual[3:6])) >= 1.0e-4
        ):
            result = least_squares(
                residual,
                seed_active,
                bounds=(self.lower, self.upper),
                method="trf",
                jac="2-point",
                max_nfev=80,
                ftol=1.0e-7,
                xtol=1.0e-7,
                gtol=1.0e-7,
            )
            q[self.active_indices] = result.x
        final_position, final_rotation = self.tool_pose(q)
        position_error_value = float(np.linalg.norm(target_position - final_position))
        rotation_error_value = float(np.linalg.norm(
            Rotation.from_matrix(target_rotation.T @ final_rotation).as_rotvec()
        ))
        return self.body_configuration(q), position_error_value, rotation_error_value


class PublisherBundle:
    def __init__(self, state_port: int, manager_port: int, gripper_port: int):
        self.context = zmq.Context()
        self.state = self.context.socket(zmq.PUB)
        self.manager = self.context.socket(zmq.PUB)
        self.gripper = self.context.socket(zmq.PUB)
        for socket in (self.state, self.manager, self.gripper):
            socket.setsockopt(zmq.SNDHWM, 3)
            socket.setsockopt(zmq.LINGER, 0)
        self.state.bind(f"tcp://*:{state_port}")
        self.manager.bind(f"tcp://*:{manager_port}")
        self.gripper.bind(f"tcp://*:{gripper_port}")

    def close(self) -> None:
        self.state.close()
        self.manager.close()
        self.gripper.close()
        self.context.term()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--interface", default="enp8s0")
    parser.add_argument("--translation-scale", type=float, default=0.75)
    parser.add_argument("--joint-speed", type=float, default=0.60)
    parser.add_argument("--arm-weight", type=float, default=1.0)
    parser.add_argument("--weight-rate", type=float, default=0.50)
    parser.add_argument("--right-arm-kp", type=float, default=50.0)
    parser.add_argument("--right-arm-kd", type=float, default=1.5)
    parser.add_argument("--max-ik-position-error", type=float, default=0.10)
    parser.add_argument("--max-ik-rotation-error-deg", type=float, default=55.0)
    parser.add_argument("--thermal-warn", type=float, default=65.0)
    parser.add_argument("--thermal-stop", type=float, default=70.0)
    parser.add_argument("--thermal-resume", type=float, default=62.0)
    parser.add_argument("--state-port", type=int, default=5557)
    parser.add_argument("--manager-port", type=int, default=5556)
    parser.add_argument("--gripper-port", type=int, default=5558)
    parser.add_argument(
        "--thermal-log-dir",
        type=Path,
        default=Path("/home/drgh/shucai/g1_competition/logs/thermal"),
    )
    args = parser.parse_args()
    if not args.real:
        parser.error("--real is required")
    if not 0.1 <= args.translation_scale <= 1.0:
        parser.error("--translation-scale must be 0.1..1.0")
    if not 0.1 <= args.arm_weight <= 1.0:
        parser.error("--arm-weight must be 0.1..1.0")
    if not 0.05 <= args.weight_rate <= 3.0:
        parser.error("--weight-rate must be 0.05..3.0/s")
    if not 5.0 <= args.right_arm_kp <= 60.0:
        parser.error("--right-arm-kp must be 5..60")
    if not 0.1 <= args.right_arm_kd <= 3.0:
        parser.error("--right-arm-kd must be 0.1..3.0")
    if args.thermal_stop <= args.thermal_warn:
        parser.error("thermal stop must exceed warning")
    if args.thermal_resume >= args.thermal_stop:
        parser.error("thermal resume must be below thermal stop")
    if not 0.02 <= args.max_ik_position_error <= 0.20:
        parser.error("--max-ik-position-error must be 0.02..0.20m")
    if not 10.0 <= args.max_ik_rotation_error_deg <= 90.0:
        parser.error("--max-ik-rotation-error-deg must be 10..90deg")

    import xrobotoolkit_sdk as xrt
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmd_, MotorCmds_, MotorStates_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    stop_requested = False

    def request_stop(_signum=None, _frame=None):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    ChannelFactoryInitialize(0, args.interface)
    low_state: LowState_ | None = None
    low_state_at = 0.0
    dex_state: tuple[float, float] | None = None

    def on_low_state(message: LowState_) -> None:
        nonlocal low_state, low_state_at
        low_state = message
        low_state_at = time.monotonic()

    def on_dex_state(message: MotorStates_) -> None:
        nonlocal dex_state
        if message.states and math.isfinite(float(message.states[0].q)):
            dex_state = (float(message.states[0].q), time.monotonic())

    low_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    low_subscriber.Init(on_low_state, 1)
    dex_subscriber = ChannelSubscriber("rt/dex1/right/state", MotorStates_)
    dex_subscriber.Init(on_dex_state, 1)
    arm_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
    arm_publisher.Init()
    dex_publisher = ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)
    dex_publisher.Init()
    outputs = PublisherBundle(args.state_port, args.manager_port, args.gripper_port)

    print("[ArmOnly] Waiting for fresh G1 and Dex1 state...", flush=True)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and (low_state is None or dex_state is None):
        time.sleep(0.05)
    if low_state is None or dex_state is None:
        raise RuntimeError("fresh G1/Dex1 state not received")
    startup_temperatures = [
        max(float(state.temperature[0]), float(state.temperature[1]))
        for state in low_state.motor_state[:BODY_JOINT_COUNT]
    ]
    startup_hot_index = int(np.argmax(startup_temperatures))
    startup_hot_value = startup_temperatures[startup_hot_index]
    thermal_paused = startup_hot_value >= args.thermal_stop
    thermal_release_seen = False
    if thermal_paused:
        print(
            f"[ArmOnly][COOLING] {JOINT_NAMES[startup_hot_index]} is already "
            f"{startup_hot_value:.1f}C; process remains alive but Arm SDK will "
            f"stay released until <= {args.thermal_resume:.1f}C.",
            flush=True,
        )
    xrt.init()
    print("[ArmOnly] Waiting for PICO body/controller data...", flush=True)
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and not xrt.is_body_data_available():
        time.sleep(0.05)
    if not xrt.is_body_data_available():
        # XRoboToolkit starts background native threads in xrt.init().  Close
        # them explicitly on this startup failure; otherwise interpreter
        # teardown can end in "terminate called without an active exception".
        xrt.close()
        outputs.close()
        dex_publisher.Close()
        arm_publisher.Close()
        dex_subscriber.Close()
        low_subscriber.Close()
        raise RuntimeError("PICO body data not available")

    solver = RightArmIK()
    controller_filter = ControllerPoseFilter(WristMappingConfig())
    measured_body = np.asarray(
        [float(state.q) for state in low_state.motor_state[:BODY_JOINT_COUNT]],
        dtype=np.float64,
    )
    commanded_body = measured_body.copy()
    tool_roll = 0.0
    target_delta = np.zeros(3, dtype=np.float64)
    controller_anchor = None
    controller_rotation_anchor = None
    target_anchor = None
    target_rotation_anchor = None
    deadman_previous = False
    workspace_limited = False
    trigger_released = False
    dex_command = float(dex_state[0])
    last_collection_combo = False
    last_abort_combo = False
    last_tick = time.monotonic()
    last_report = 0.0
    last_xr_timestamp = None
    last_xr_update_at = 0.0
    body_poses = None
    right_hand_pose = None
    crc = CRC()
    low_command = unitree_hg_msg_dds__LowCmd_()
    arm_claimed = False
    arm_weight = 0.0
    args.thermal_log_dir.mkdir(parents=True, exist_ok=True)
    thermal_log = args.thermal_log_dir / time.strftime("arm_only_%Y%m%d_%H%M%S.csv")
    thermal_file = thermal_log.open("w", buffering=1)
    thermal_file.write("wall_time,max_joint,max_channel,max_temp_c\n")
    print(
        "[ArmOnly] Ready. Hold RIGHT GRIP to move; right trigger controls Dex1. "
        "ABXY releases control.",
        flush=True,
    )
    try:
        while not stop_requested:
            tick = time.monotonic()
            dt = min(max(tick - last_tick, 0.001), 0.05)
            last_tick = tick
            if low_state is None or tick - low_state_at > 0.3:
                raise RuntimeError("rt/lowstate stale; releasing Arm SDK")
            states = low_state.motor_state[:BODY_JOINT_COUNT]
            measured_body = np.asarray([float(state.q) for state in states], dtype=np.float64)
            temperatures = [(float(s.temperature[0]), float(s.temperature[1])) for s in states]
            max_index, max_channel = max(
                ((joint, channel) for joint in range(BODY_JOINT_COUNT) for channel in (0, 1)),
                key=lambda item: temperatures[item[0]][item[1]],
            )
            max_temp = temperatures[max_index][max_channel]
            if tick - last_report >= 1.0:
                thermal_file.write(
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')},"
                    f"{JOINT_NAMES[max_index]},{'case' if max_channel == 0 else 'winding'},"
                    f"{max_temp:.1f}\n"
                )
            if max_temp >= args.thermal_stop and not thermal_paused:
                print(
                    f"[ArmOnly][COOLING] {JOINT_NAMES[max_index]}={max_temp:.1f}C; "
                    "releasing Arm SDK but keeping the process alive",
                    flush=True,
                )
                thermal_paused = True
                thermal_release_seen = False
            if max_temp >= args.thermal_warn and tick - last_report >= 1.0:
                print(
                    f"[ArmOnly][TEMP WARNING] {JOINT_NAMES[max_index]}={max_temp:.1f}C "
                    f"(stop {args.thermal_stop:.1f}C)",
                    flush=True,
                )

            timestamp = xrt.get_time_stamp_ns() if xrt.is_body_data_available() else 0
            if timestamp not in (0, last_xr_timestamp):
                last_xr_timestamp = timestamp
                body_poses = np.asarray(xrt.get_body_joints_pose(), dtype=np.float64)
                try:
                    # Body joint 23 is an inferred hand pose and can flip its
                    # wrist branch near elbow extension. Use the directly
                    # tracked controller pose for both translation and rotation.
                    right_hand_pose = controller_filter.update(
                        xrt.get_right_controller_pose(), tick
                    )
                    last_xr_update_at = tick
                except (TypeError, ValueError):
                    # Do not silently fall back to the inferred hand joint: a
                    # stale deadman release is safer than reintroducing the
                    # discontinuous wrist source.
                    pass
            xr_fresh = (
                body_poses is not None
                and right_hand_pose is not None
                and tick - last_xr_update_at <= 0.5
            )
            right_grip = float(xrt.get_right_grip()) if xr_fresh else 0.0
            right_trigger = float(xrt.get_right_trigger()) if xr_fresh else 0.0
            try:
                right_axis = xrt.get_right_axis() if xr_fresh else [0.0, 0.0]
                right_axis_x = float(right_axis[0]) if len(right_axis) >= 1 else 0.0
            except Exception:
                right_axis_x = 0.0
            a = bool(xrt.get_A_button()) if xr_fresh else False
            b = bool(xrt.get_B_button()) if xr_fresh else False
            x = bool(xrt.get_X_button()) if xr_fresh else False
            y = bool(xrt.get_Y_button()) if xr_fresh else False
            left_grip = float(xrt.get_left_grip()) if xr_fresh else 0.0
            if thermal_paused:
                if right_grip <= 0.1:
                    thermal_release_seen = True
                if max_temp <= args.thermal_resume and thermal_release_seen:
                    thermal_paused = False
                    print(
                        f"[ArmOnly] cooled to {max_temp:.1f}C; arm control available again",
                        flush=True,
                    )
            if a and b and x and y:
                print("[ArmOnly] ABXY stop requested", flush=True)
                break

            deadman = update_deadman(
                deadman_previous, right_grip, xr_fresh, thermal_paused
            )
            if deadman_previous and not deadman:
                if thermal_paused:
                    release_reason = "thermal cooling"
                elif not xr_fresh:
                    release_reason = "XR stream stale"
                else:
                    release_reason = f"right grip released ({right_grip:.2f})"
                print(f"[ArmOnly] releasing deadman: {release_reason}", flush=True)
            if deadman and not deadman_previous:
                workspace_limited = False
                commanded_body[RIGHT_ARM_BODY_INDICES] = measured_body[
                    RIGHT_ARM_BODY_INDICES
                ]
                controller_anchor = tracked_position_in_root(
                    right_hand_pose, body_poses[0]
                )
                controller_rotation_anchor = tracked_rotation_in_root(
                    right_hand_pose, body_poses[0]
                )
                # Anchor the exact measured arm pose. Pressing Grip therefore
                # creates no jump and does not straighten or lock any joint.
                anchor_body = measured_body.copy()
                target_anchor, target_rotation_anchor = solver.tool_pose(
                    solver.full_configuration(anchor_body)
                )
                print("[ArmOnly] right-grip deadman armed", flush=True)
            if (
                deadman
                and controller_anchor is not None
                and controller_rotation_anchor is not None
                and target_anchor is not None
                and target_rotation_anchor is not None
            ):
                tool_roll = float(np.clip(
                    tool_roll + apply_deadzone(right_axis_x) * math.radians(45.0) * dt,
                    -math.radians(40.0),
                    math.radians(40.0),
                ))
                current_controller = tracked_position_in_root(right_hand_pose, body_poses[0])
                current_controller_rotation = tracked_rotation_in_root(
                    right_hand_pose, body_poses[0]
                )
                relative_controller_rotation = (
                    controller_rotation_anchor.T @ current_controller_rotation
                )
                target_rotation = (
                    target_rotation_anchor
                    @ relative_controller_rotation
                    @ Rotation.from_rotvec([tool_roll, 0.0, 0.0]).as_matrix()
                )
                # Drive the robot tool from the tracked hand's displacement
                # relative to the deadman anchor.  The former implementation
                # independently aligned the human upper-arm and forearm
                # directions.  Those two rotations did not preserve one
                # coherent elbow geometry, so an upward human motion could ask
                # IK for an impossible chain and drive the elbow straight/down.
                target_position = target_anchor + args.translation_scale * (
                    current_controller - controller_anchor
                )
                target_delta = target_position - target_anchor
                # FK must use the real torso pose. The normal motion firmware
                # can bend the waist for balance even while arm_sdk requests
                # its old value; solving with that old value made the hand sag
                # in world space despite correct right-arm joint tracking.
                ik_context_body = measured_body.copy()
                ik_context_body[RIGHT_ARM_BODY_INDICES] = commanded_body[
                    RIGHT_ARM_BODY_INDICES
                ]
                solved_body, position_error, rotation_error = solver.solve(
                    ik_context_body,
                    target_position,
                    target_rotation,
                )
                desired_right = solved_body[RIGHT_ARM_BODY_INDICES]
                residual_large = not ik_solution_is_feasible(
                    position_error,
                    rotation_error,
                    args.max_ik_position_error,
                    math.radians(args.max_ik_rotation_error_deg),
                )
                if residual_large and not workspace_limited:
                    print(
                        f"[ArmOnly][TRACKING RESIDUAL] "
                        f"{position_error * 1000:.0f}mm/"
                        f"{math.degrees(rotation_error):.0f}deg; "
                        "continuing with closest physical joint solution.",
                        flush=True,
                    )
                if not residual_large and workspace_limited:
                    print("[ArmOnly] full-pose tracking residual recovered", flush=True)
                workspace_limited = residual_large
            else:
                # Do not hold an unloaded standby pose with Arm SDK stiffness:
                # that continued heating RightShoulderPitch after Grip release.
                # Track the measured pose while fading the SDK weight to zero,
                # then normal locomotion owns the arm again.
                desired_right = measured_body[RIGHT_ARM_BODY_INDICES].copy()
                target_delta[:] = 0.0
                tool_roll = float(rate_limit_vector(
                    np.asarray([tool_roll]), np.zeros(1), math.radians(45.0), dt
                )[0])
                position_error = 0.0
                rotation_error = 0.0
                workspace_limited = False
            deadman_previous = deadman

            if deadman:
                commanded_body[RIGHT_ARM_BODY_INDICES] = rate_limit_vector(
                    commanded_body[RIGHT_ARM_BODY_INDICES],
                    desired_right,
                    args.joint_speed,
                    dt,
                )
            else:
                commanded_body[RIGHT_ARM_BODY_INDICES] = desired_right
            # rt/arm_sdk exposes one blend weight for waist and both arms; it
            # has no per-arm ownership mask.  Keep part of the normal
            # locomotion command blended in and acquire it gradually so a
            # right-arm deadman press cannot abruptly change whole-body load.
            target_arm_weight = args.arm_weight if deadman else 0.0
            arm_weight = float(rate_limit_vector(
                np.asarray([arm_weight]),
                np.asarray([target_arm_weight]),
                args.weight_rate,
                dt,
            )[0])
            low_command.motor_cmd[ARM_SDK_WEIGHT_INDEX].q = arm_weight
            arm_claimed = arm_claimed or arm_weight > 1.0e-3
            # Do not publish artificial waist/left-arm angle holds. Their
            # command fields remain zero-stiffness and normal locomotion keeps
            # responsibility for them; only the seven right-arm motors receive
            # position gains from this process.
            for body_index in RIGHT_ARM_BODY_INDICES:
                command = low_command.motor_cmd[int(body_index)]
                command.q = float(commanded_body[int(body_index)])
                command.dq = 0.0
                command.tau = 0.0
                command.kp = args.right_arm_kp
                command.kd = args.right_arm_kd
            low_command.crc = crc.Crc(low_command)
            arm_publisher.Write(low_command)

            if right_trigger <= 0.1:
                trigger_released = True
            state_fresh = dex_state is not None and tick - dex_state[1] <= 0.3
            if trigger_released and state_fresh:
                desired_dex = 5.2 * (1.0 - np.clip(right_trigger, 0.0, 1.0))
                dex_command = float(rate_limit_vector(
                    np.asarray([dex_command]), np.asarray([desired_dex]), 1.0, dt
                )[0])
                measured_dex = float(dex_state[0])
                dex_command = float(np.clip(dex_command, measured_dex - 0.35, measured_dex + 0.35))
                dex_command = float(np.clip(dex_command, 0.0, 5.2))
                dex_publisher.Write(MotorCmds_(cmds=[MotorCmd_(
                    mode=1, q=dex_command, dq=0.0, tau=0.0,
                    kp=5.0, kd=0.05, reserve=[0, 0, 0],
                )]))

            collection_combo = left_grip > 0.5 and a
            abort_combo = left_grip > 0.5 and b
            toggle_collection = collection_combo and not last_collection_combo
            toggle_abort = abort_combo and not last_abort_combo
            last_collection_combo = collection_combo
            last_abort_combo = abort_combo
            outputs.manager.send(pack_pose_message(
                {
                    "toggle_data_collection": np.asarray([toggle_collection], dtype=bool),
                    "toggle_data_abort": np.asarray([toggle_abort], dtype=bool),
                    "timestamp_monotonic": np.asarray([tick], dtype=np.float64),
                },
                topic="manager_state",
            ))
            outputs.state.send(b"g1_debug" + msgpack.packb(
                {
                    "body_q_measured": measured_body.tolist(),
                    "last_action": commanded_body.tolist(),
                }, use_bin_type=True,
            ))
            outputs.gripper.send(b"dex1" + msgpack.packb(
                {
                    "timestamp": time.time(),
                    "measured_q": float(dex_state[0]) if dex_state else None,
                    "commanded_q": dex_command,
                    "trigger": right_trigger,
                    "armed": trigger_released and state_fresh,
                    "xr_fresh": xr_fresh,
                    "state_fresh": state_fresh,
                }, use_bin_type=True,
            ))
            if tick - last_report >= 1.0:
                print(
                    f"[ArmOnly] "
                    f"{'COOLING' if thermal_paused else ('RESIDUAL' if workspace_limited else ('FOLLOW' if deadman else 'HOLD'))} "
                    f"grip={right_grip:.2f} xr={'fresh' if xr_fresh else 'stale'} "
                    f"target_d=({target_delta[0]:+.2f},{target_delta[1]:+.2f},"
                    f"{target_delta[2]:+.2f})m "
                    f"waist_meas=({measured_body[12]:+.2f},"
                    f"{measured_body[13]:+.2f},{measured_body[14]:+.2f}) "
                    f"arm_w={arm_weight:.2f} "
                    f"qR_cmd=({commanded_body[22]:+.2f},{commanded_body[23]:+.2f},"
                    f"{commanded_body[24]:+.2f},{commanded_body[25]:+.2f}) "
                    f"qR_meas=({measured_body[22]:+.2f},{measured_body[23]:+.2f},"
                    f"{measured_body[24]:+.2f},{measured_body[25]:+.2f}) "
                    f"wrist_cmd=({commanded_body[26]:+.2f},{commanded_body[27]:+.2f},"
                    f"{commanded_body[28]:+.2f}) wrist_meas=({measured_body[26]:+.2f},"
                    f"{measured_body[27]:+.2f},{measured_body[28]:+.2f}) "
                    f"tool_roll={math.degrees(tool_roll):.0f}deg "
                    f"IKerr={position_error * 1000:.0f}mm/"
                    f"{math.degrees(rotation_error):.0f}deg max_temp={max_temp:.1f}C",
                    flush=True,
                )
                last_report = tick
            time.sleep(max(0.0, 0.02 - (time.monotonic() - tick)))
    finally:
        if arm_claimed:
            print("[ArmOnly] Releasing Arm SDK to normal locomotion control...", flush=True)
            release_start_weight = arm_weight
            for step in range(25):
                low_command.motor_cmd[ARM_SDK_WEIGHT_INDEX].q = (
                    release_start_weight * (1.0 - (step + 1) / 25.0)
                )
                low_command.crc = crc.Crc(low_command)
                arm_publisher.Write(low_command)
                time.sleep(0.02)
        thermal_file.close()
        outputs.close()
        dex_publisher.Close()
        arm_publisher.Close()
        dex_subscriber.Close()
        low_subscriber.Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Start the SONIC PICO manager and inject one deliberate OFF->PLANNER edge.

This is a recovery launcher for cases where XR button polling misses the
operator's initial A+B+X+Y combination.  Only the first poll is overridden;
all subsequent controller input is read from XRoboToolkit normally.  Optional
right-gripper and body-temperature monitoring threads share this one process so
they do not create competing XRoboToolkit clients.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import subprocess
import threading
import time

from gear_sonic.scripts import pico_manager_thread_server as manager

from pico_wrist_mapping import (
    WristMappingConfig,
    install_bilateral_controller_wrist_mapping,
)


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

LEFT_ARM_INDICES = tuple(range(15, 22))
RIGHT_ARM_INDICES = tuple(range(22, 29))


def _quaternion_wxyz_to_rpy_deg(value) -> tuple[float, float, float]:
    """Convert a normalized wxyz quaternion to intrinsic roll/pitch/yaw."""
    quaternion = [float(item) for item in value]
    if len(quaternion) != 4 or not all(math.isfinite(item) for item in quaternion):
        raise ValueError("quaternion must contain four finite values")
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm <= 1.0e-9:
        raise ValueError("quaternion must be non-zero")
    w, x, y, z = (item / norm for item in quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(math.degrees(angle) for angle in (roll, pitch, yaw))


def run_wrist_diagnostics(
    stop_event: threading.Event,
    log_dir: Path,
    feedback_host: str = "localhost",
    feedback_port: int = 5557,
) -> None:
    """Record SONIC wrist inputs, policy commands, and measured joints together.

    This is a passive ZMQ subscriber. It never publishes a robot command and
    cannot alter the control loop. The shared timeline distinguishes a bad
    PICO target from a SONIC-policy response or actuator tracking error.
    """
    import msgpack
    import zmq

    topic = b"g1_debug"
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, topic)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://{feedback_host}:{feedback_port}")

    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"sonic_wrist_{stamp}.csv"
    arm_columns = []
    for source in ("command", "measured"):
        for index in LEFT_ARM_INDICES + RIGHT_ARM_INDICES:
            arm_columns.append(f"{source}_{JOINT_NAMES[index]}_rad")
    header = [
        "wall_time", "monotonic",
        *[f"left_target_{axis}_m" for axis in "xyz"],
        *[f"right_target_{axis}_m" for axis in "xyz"],
        *[f"left_target_q_{axis}" for axis in "wxyz"],
        *[f"right_target_q_{axis}" for axis in "wxyz"],
        *arm_columns,
    ]
    print(f"[WristDiag] passive input/command/measurement log: {log_path}", flush=True)
    last_write = 0.0
    last_report = 0.0
    waiting_reported = False
    try:
        with log_path.open("w", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(header)
            log_file.flush()
            while not stop_event.is_set():
                if not socket.poll(timeout=200):
                    if not waiting_reported:
                        print("[WristDiag] waiting for SONIC g1_debug", flush=True)
                        waiting_reported = True
                    continue
                raw = socket.recv()
                if not raw.startswith(topic):
                    continue
                try:
                    data = msgpack.unpackb(raw[len(topic):], raw=False)
                    positions = list(data["vr_3point_position"])
                    orientations = list(data["vr_3point_orientation"])
                    command = list(data["last_action"])
                    measured = list(data["body_q_measured"])
                    if not (
                        len(positions) == 9
                        and len(orientations) == 12
                        and len(command) >= 29
                        and len(measured) >= 29
                    ):
                        raise ValueError("unexpected g1_debug vector sizes")
                    numeric = positions + orientations + command[:29] + measured[:29]
                    if not all(math.isfinite(float(item)) for item in numeric):
                        raise ValueError("non-finite g1_debug value")
                except (KeyError, TypeError, ValueError, msgpack.ExtraData) as exc:
                    print(f"[WristDiag] ignored malformed feedback: {exc}", flush=True)
                    continue

                waiting_reported = False
                now = time.monotonic()
                if now - last_write >= 0.05:
                    indices = LEFT_ARM_INDICES + RIGHT_ARM_INDICES
                    writer.writerow([
                        time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        f"{now:.6f}",
                        *[f"{float(item):.8f}" for item in positions[:6]],
                        *[f"{float(item):.8f}" for item in orientations[:8]],
                        *[f"{float(command[index]):.8f}" for index in indices],
                        *[f"{float(measured[index]):.8f}" for index in indices],
                    ])
                    log_file.flush()
                    last_write = now

                if now - last_report >= 1.0:
                    right_rpy = _quaternion_wxyz_to_rpy_deg(orientations[4:8])
                    right_wrist_command = command[26:29]
                    right_wrist_measured = measured[26:29]
                    error_deg = [
                        math.degrees(float(cmd) - float(actual))
                        for cmd, actual in zip(right_wrist_command, right_wrist_measured)
                    ]
                    print(
                        "[WristDiag] R target xyz="
                        f"[{positions[3]:+.3f},{positions[4]:+.3f},{positions[5]:+.3f}] "
                        "rpy="
                        f"[{right_rpy[0]:+.1f},{right_rpy[1]:+.1f},{right_rpy[2]:+.1f}]deg "
                        "wrist cmd/meas/error="
                        f"[{','.join(f'{math.degrees(float(v)):+.1f}' for v in right_wrist_command)}]/"
                        f"[{','.join(f'{math.degrees(float(v)):+.1f}' for v in right_wrist_measured)}]/"
                        f"[{','.join(f'{v:+.1f}' for v in error_deg)}]deg",
                        flush=True,
                    )
                    last_report = now
    except Exception as exc:
        # Diagnostics must never take control down.
        print(f"[WristDiag] disabled after error: {exc}", flush=True)
    finally:
        socket.close()
        context.term()


def run_right_gripper(
    reader,
    stop_event: threading.Event,
    telemetry_port: int,
) -> None:
    """Drive the right Dex1 from the manager's existing in-process XRT client."""
    import msgpack
    import zmq

    from unitree_sdk2py.core.channel import (
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmd_, MotorCmds_, MotorStates_

    min_q = 0.0
    open_q = 5.2
    max_q = 5.5
    max_speed = 1.0
    max_lead = 0.35
    max_age = 0.3
    hz = 50.0

    state_lock = threading.Lock()
    last_state = None

    def on_state(msg: MotorStates_) -> None:
        nonlocal last_state
        if not msg.states:
            return
        q = float(msg.states[0].q)
        if math.isfinite(q):
            with state_lock:
                last_state = (q, time.monotonic())

    subscriber = ChannelSubscriber("rt/dex1/right/state", MotorStates_)
    subscriber.Init(on_state, 1)
    publisher = ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)
    publisher.Init()
    telemetry_context = zmq.Context()
    telemetry_socket = telemetry_context.socket(zmq.PUB)
    telemetry_socket.setsockopt(zmq.SNDHWM, 3)
    telemetry_socket.setsockopt(zmq.LINGER, 0)
    telemetry_socket.bind(f"tcp://*:{telemetry_port}")
    print(
        f"[CombinedGripper] telemetry on tcp://*:{telemetry_port} topic dex1",
        flush=True,
    )

    trigger_release_seen = False
    commanded_q = None
    last_tick = time.monotonic()
    last_motion_q = None
    last_motion_at = last_tick
    last_target_q = None
    last_status = ""
    last_values_at = 0.0
    print("[CombinedGripper] Waiting for fresh XR and right Dex1 state", flush=True)
    try:
        while not stop_event.is_set():
            tick = time.monotonic()
            dt = min(tick - last_tick, 0.1)
            last_tick = tick
            sample = reader.get_latest()
            xr_fresh = (
                sample is not None
                and tick - float(sample["timestamp_monotonic"]) <= max_age
            )
            with state_lock:
                state = last_state
            state_fresh = state is not None and tick - state[1] <= max_age
            trigger = float(manager.xrt.get_right_trigger()) if xr_fresh else 0.0
            emergency_combo = (
                bool(manager.xrt.get_A_button())
                and bool(manager.xrt.get_B_button())
                and bool(manager.xrt.get_X_button())
                and bool(manager.xrt.get_Y_button())
            ) if xr_fresh else False

            if not xr_fresh or not state_fresh or emergency_combo:
                trigger_release_seen = False
                commanded_q = None
                last_motion_q = None
                last_target_q = None
                status = (
                    "waiting for XR" if not xr_fresh else
                    "waiting for motor state" if not state_fresh else
                    "emergency buttons pressed"
                )
            else:
                measured_q = state[0]
                if not min_q <= measured_q <= max_q:
                    raise RuntimeError(f"Dex1 state outside calibrated range: q={measured_q:.3f}")
                if trigger <= 0.1:
                    trigger_release_seen = True
                if not trigger_release_seen:
                    status = "release trigger once to arm"
                else:
                    desired_q = open_q * (1.0 - min(1.0, max(0.0, trigger)))
                    if commanded_q is None:
                        commanded_q = measured_q
                        last_motion_q = measured_q
                        last_motion_at = tick
                    if last_target_q is None or abs(desired_q - last_target_q) >= 0.1:
                        last_motion_at = tick
                        last_target_q = desired_q
                    if last_motion_q is None or abs(measured_q - last_motion_q) >= 0.03:
                        last_motion_q = measured_q
                        last_motion_at = tick
                    if abs(desired_q - measured_q) > 0.45 and tick - last_motion_at > 2.5:
                        raise RuntimeError("right gripper stalled; stopped publishing")
                    delta = min(max_speed * max(dt, 0.0), abs(desired_q - commanded_q))
                    commanded_q += math.copysign(delta, desired_q - commanded_q)
                    commanded_q = min(measured_q + max_lead, max(measured_q - max_lead, commanded_q))
                    commanded_q = min(open_q, max(min_q, commanded_q))
                    publisher.Write(MotorCmds_(cmds=[MotorCmd_(
                        mode=1, q=commanded_q, dq=0.0, tau=0.0,
                        kp=5.0, kd=0.05, reserve=[0, 0, 0],
                    )]))
                    status = "armed"
                    if tick - last_values_at >= 1.0:
                        print(
                            f"[CombinedGripper] trigger={trigger:.2f} "
                            f"target={desired_q:.2f} actual={measured_q:.2f}",
                            flush=True,
                        )
                        last_values_at = tick
            if status != last_status:
                print(f"[CombinedGripper] {status}", flush=True)
                last_status = status
            measured_for_telemetry = state[0] if state_fresh else None
            payload = {
                "timestamp": time.time(),
                "measured_q": measured_for_telemetry,
                "commanded_q": commanded_q,
                "trigger": trigger,
                "armed": status == "armed",
                "xr_fresh": xr_fresh,
                "state_fresh": state_fresh,
            }
            try:
                telemetry_socket.send(
                    b"dex1" + msgpack.packb(payload, use_bin_type=True),
                    flags=zmq.NOBLOCK,
                )
            except zmq.Again:
                pass
            time.sleep(max(0.0, 1.0 / hz - (time.monotonic() - tick)))
    except Exception as exc:
        print(f"[CombinedGripper] stopped: {exc}", flush=True)
    finally:
        telemetry_socket.close()
        telemetry_context.term()
        publisher.Close()
        subscriber.Close()


def run_thermal_guard(
    stop_event: threading.Event,
    ready_event: threading.Event,
    fatal_trip_event: threading.Event,
    policy_started_event: threading.Event,
    status_lock: threading.Lock,
    status: dict,
    warn_temp: float,
    critical_temp: float,
    log_dir: Path,
) -> None:
    """Passively log temperatures; never change control mode from temperature."""
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    sample_lock = threading.Lock()
    latest_sample = None

    def on_state(msg: LowState_) -> None:
        nonlocal latest_sample
        states = msg.motor_state[: len(JOINT_NAMES)]
        if len(states) != len(JOINT_NAMES):
            return
        temperatures = []
        torques = []
        for state in states:
            pair = list(state.temperature)
            if len(pair) < 2:
                return
            temperatures.append((float(pair[0]), float(pair[1])))
            torques.append(float(state.tau_est))
        with sample_lock:
            latest_sample = (time.monotonic(), temperatures, torques)
        with status_lock:
            status["latest_monotonic"] = latest_sample[0]
        ready_event.set()

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(on_state, 1)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"thermal_{stamp}.csv"
    header = ["wall_time", "monotonic", "max_joint", "max_channel", "max_temp_c"]
    for name in JOINT_NAMES:
        header.extend((f"{name}_case_c", f"{name}_winding_c", f"{name}_tau_est"))

    last_logged = 0.0
    last_warning = 0.0
    stale_reported = False
    print(f"[ThermalGuard] logging to {log_path}", flush=True)
    try:
        with log_path.open("w", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(header)
            log_file.flush()
            while not stop_event.is_set():
                now = time.monotonic()
                with sample_lock:
                    sample = latest_sample
                if sample is None or now - sample[0] > 1.0:
                    ready_event.clear()
                    if not stale_reported:
                        print("[ThermalGuard] waiting for fresh rt/lowstate", flush=True)
                        stale_reported = True
                    # Loading the robot model and hand IK can hold Python's GIL
                    # for several seconds.  Before the policy start edge this is
                    # not a control hazard: wait for telemetry to become fresh
                    # again and refuse to inject ABXY meanwhile.  Once control
                    # has actually started, the same stale condition is a trip.
                    if policy_started_event.is_set() and sample is not None:
                        print(
                            f"[ThermalGuard] TRIP telemetry stale for "
                            f"{now - sample[0]:.2f}s; stopping",
                            flush=True,
                        )
                        fatal_trip_event.set()
                        break
                    time.sleep(0.05)
                    continue

                stale_reported = False
                _, temperatures, torques = sample
                max_index, max_channel_index = max(
                    (
                        (joint_index, channel_index)
                        for joint_index, pair in enumerate(temperatures)
                        for channel_index in range(2)
                    ),
                    key=lambda index: temperatures[index[0]][index[1]],
                )
                max_temp = temperatures[max_index][max_channel_index]
                max_joint = JOINT_NAMES[max_index]
                max_channel = "case" if max_channel_index == 0 else "winding"

                if now - last_logged >= 1.0:
                    row = [
                        time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        f"{now:.6f}", max_joint, max_channel, f"{max_temp:.1f}",
                    ]
                    for pair, torque in zip(temperatures, torques):
                        row.extend((f"{pair[0]:.1f}", f"{pair[1]:.1f}", f"{torque:.4f}"))
                    writer.writerow(row)
                    log_file.flush()
                    last_logged = now

                if max_temp >= warn_temp and now - last_warning >= 1.0:
                    level = "CRITICAL" if max_temp >= critical_temp else "WARNING"
                    print(
                        f"[ThermalMonitor] {level} {max_joint} {max_channel}="
                        f"{max_temp:.1f}C (warning-only; no VR exit)",
                        flush=True,
                    )
                    last_warning = now
                time.sleep(0.05)
    except Exception as exc:
        print(f"[ThermalGuard] failed: {exc}", flush=True)
        fatal_trip_event.set()
    finally:
        subscriber.Close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-autostart",
        action="store_true",
        help="required acknowledgement that the robot may enter PLANNER",
    )
    parser.add_argument(
        "--auto-vr-delay",
        type=float,
        default=0.0,
        help="inject one L3 edge this many seconds after the PLANNER start edge",
    )
    parser.add_argument(
        "--dex1-right",
        action="store_true",
        help="run right-trigger Dex1 control inside this same XRT process",
    )
    parser.add_argument("--interface", default="enp8s0")
    parser.add_argument("--dex1-telemetry-port", type=int, default=5558)
    parser.add_argument("--thermal-guard", action="store_true")
    parser.add_argument("--thermal-warn", type=float, default=70.0)
    parser.add_argument("--thermal-critical", type=float, default=75.0)
    parser.add_argument(
        "--right-wrist-source",
        choices=("controller", "body"),
        default="controller",
        help="use both direct controller 6DoF poses (default) or upstream body joints",
    )
    parser.add_argument(
        "--right-wrist-max-angular-speed",
        type=float,
        default=240.0,
        help="right-controller orientation rate limit in degrees/second",
    )
    parser.add_argument(
        "--right-wrist-max-linear-speed",
        type=float,
        default=1.5,
        help="right-controller position rate limit in metres/second",
    )
    parser.add_argument(
        "--right-wrist-filter-tau",
        type=float,
        default=0.04,
        help="right-controller pose low-pass time constant in seconds",
    )
    parser.add_argument(
        "--thermal-log-dir",
        type=Path,
        default=Path("/home/drgh/shucai/g1_competition/logs/thermal"),
    )
    parser.add_argument(
        "--wrist-diagnostics",
        action="store_true",
        help="passively log SONIC wrist targets, commands, and measurements",
    )
    parser.add_argument(
        "--wrist-diagnostic-log-dir",
        type=Path,
        default=Path("/home/drgh/shucai/g1_competition/logs/wrist"),
    )
    parser.add_argument(
        "--thermal-tmux-target",
        default="g1_task2_body:deploy",
        help="tmux target whose SONIC process receives O after a thermal trip",
    )
    args = parser.parse_args()
    if not args.confirm_autostart:
        parser.error("--confirm-autostart is required")
    if args.thermal_warn <= 0.0:
        parser.error("--thermal-warn must be positive")
    if args.thermal_critical <= args.thermal_warn:
        parser.error("--thermal-critical must be greater than --thermal-warn")

    if args.right_wrist_source == "controller":
        wrist_mapping_config = WristMappingConfig(
            max_angular_speed_deg_s=args.right_wrist_max_angular_speed,
            max_linear_speed_m_s=args.right_wrist_max_linear_speed,
            smoothing_time_constant_s=args.right_wrist_filter_tau,
        )
        try:
            wrist_mapping_config.validate()
        except ValueError as exc:
            parser.error(str(exc))
        install_bilateral_controller_wrist_mapping(manager, wrist_mapping_config)
    else:
        print(
            "[WristMapping] upstream body wrist joints selected; controller overrides disabled",
            flush=True,
        )

    original = manager.get_abxy_buttons
    original_axis_clicks = manager.get_axis_clicks
    inject_once = True
    planner_edge_at = None
    vr_injected = False
    thermal_fatal_event = threading.Event()
    thermal_ready_event = threading.Event()
    policy_started_event = threading.Event()
    thermal_status_lock = threading.Lock()
    thermal_status = {"latest_monotonic": 0.0}
    thermal_fatal_stop_injected = False
    thermal_defer_reported = False

    def get_abxy_buttons(reader=None):
        nonlocal inject_once, planner_edge_at, thermal_fatal_stop_injected, thermal_defer_reported
        if thermal_fatal_event.is_set():
            if inject_once:
                print("[ThermalGuard] startup aborted before policy start", flush=True)
                raise SystemExit(2)
            if not thermal_fatal_stop_injected:
                thermal_fatal_stop_injected = True
                print("[ThermalGuard] fatal telemetry fault; injecting stop edge", flush=True)
                return True, True, True, True
            return False, False, False, False
        if inject_once:
            if args.thermal_guard:
                with thermal_status_lock:
                    thermal_timestamp = float(thermal_status["latest_monotonic"])
                thermal_age = time.monotonic() - thermal_timestamp
                if thermal_timestamp <= 0.0 or thermal_age > 1.0:
                    if not thermal_defer_reported:
                        print(
                            "[ThermalGuard] deferring policy start until telemetry is fresh",
                            flush=True,
                        )
                        thermal_defer_reported = True
                    return False, False, False, False
            inject_once = False
            planner_edge_at = time.monotonic()
            policy_started_event.set()
            print("[Recovery] Injecting one A+B+X+Y start edge", flush=True)
            return True, True, True, True
        return original(reader)

    def get_axis_clicks(reader=None):
        nonlocal vr_injected
        left, right = original_axis_clicks(reader)
        if (
            not vr_injected
            and args.auto_vr_delay > 0.0
            and planner_edge_at is not None
            and time.monotonic() - planner_edge_at >= args.auto_vr_delay
        ):
            vr_injected = True
            print("[Recovery] Injecting one L3 edge for VR_3PT", flush=True)
            return True, right
        return left, right

    manager.get_abxy_buttons = get_abxy_buttons
    manager.get_axis_clicks = get_axis_clicks
    stop_event = threading.Event()
    gripper_thread = None
    thermal_thread = None
    wrist_diagnostic_thread = None
    original_init_input_source = manager._init_input_source

    def init_input_source(input_source, buffer_size):
        nonlocal gripper_thread, thermal_thread, wrist_diagnostic_thread
        reader = original_init_input_source(input_source, buffer_size)
        if args.dex1_right or args.thermal_guard:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize

            ChannelFactoryInitialize(0, args.interface)
        if args.thermal_guard:
            thermal_thread = threading.Thread(
                target=run_thermal_guard,
                args=(
                    stop_event,
                    thermal_ready_event,
                    thermal_fatal_event,
                    policy_started_event,
                    thermal_status_lock,
                    thermal_status,
                    args.thermal_warn,
                    args.thermal_critical,
                    args.thermal_log_dir,
                ),
                daemon=True,
            )
            thermal_thread.start()
            if not thermal_ready_event.wait(timeout=5.0):
                raise RuntimeError("no rt/lowstate received; refusing unguarded startup")
            if thermal_fatal_event.is_set():
                raise RuntimeError("robot temperature telemetry failed during startup")
            print("[ThermalGuard] fresh telemetry received; startup permitted", flush=True)
        if args.dex1_right:
            gripper_thread = threading.Thread(
                target=run_right_gripper,
                args=(reader, stop_event, args.dex1_telemetry_port),
                daemon=True,
            )
            gripper_thread.start()
        if args.wrist_diagnostics:
            wrist_diagnostic_thread = threading.Thread(
                target=run_wrist_diagnostics,
                args=(stop_event, args.wrist_diagnostic_log_dir),
                daemon=True,
            )
            wrist_diagnostic_thread.start()
        return reader

    manager._init_input_source = init_input_source
    try:
        manager.run_pico_manager()
    finally:
        stop_event.set()
        if gripper_thread is not None:
            gripper_thread.join(timeout=1.0)
        if thermal_thread is not None:
            thermal_thread.join(timeout=1.0)
        if wrist_diagnostic_thread is not None:
            wrist_diagnostic_thread.join(timeout=1.0)
        if thermal_fatal_event.is_set():
            result = subprocess.run(
                ["tmux", "send-keys", "-t", args.thermal_tmux_target, "O"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                print("[ThermalGuard] sent O to SONIC deploy terminal", flush=True)
            else:
                print(
                    "[ThermalGuard] deploy terminal not found; manager stop was still sent",
                    flush=True,
                )
            raise SystemExit(75)


if __name__ == "__main__":
    main()

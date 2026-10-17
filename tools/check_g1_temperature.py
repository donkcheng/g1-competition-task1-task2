#!/usr/bin/env python3
"""Read-only G1 motor temperature check; publishes no robot commands."""

from __future__ import annotations

import argparse
import time


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="enp8s0")
    parser.add_argument("--ready-below", type=float, default=65.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    latest = None

    def receive(message: LowState_) -> None:
        nonlocal latest
        latest = message

    ChannelFactoryInitialize(0, args.interface)
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(receive, 1)
    deadline = time.monotonic() + args.timeout
    while latest is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if latest is None:
        print("No rt/lowstate received")
        subscriber.Close()
        return 3

    readings = []
    for index, state in enumerate(latest.motor_state[: len(JOINT_NAMES)]):
        case = float(state.temperature[0])
        winding = float(state.temperature[1])
        readings.append((max(case, winding), JOINT_NAMES[index], case, winding))
    readings.sort(reverse=True)
    for maximum, name, case, winding in readings[:5]:
        print(f"{name}: max={maximum:.1f}C case={case:.1f}C winding={winding:.1f}C")
    hottest, name, _case, _winding = readings[0]
    if hottest >= args.ready_below:
        print(f"NOT READY: hottest joint is {name} at {hottest:.1f}C; wait below {args.ready_below:.1f}C")
        subscriber.Close()
        return 2
    print(f"READY: all joints are below {args.ready_below:.1f}C")
    subscriber.Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

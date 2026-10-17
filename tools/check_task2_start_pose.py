#!/usr/bin/env python3
"""Read-only comparison of the measured right arm/Dex1 to an ACT start pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


RIGHT_ARM_BODY_INDICES = tuple(range(22, 29))
RIGHT_ARM_NAMES = (
    "RightShoulderPitch",
    "RightShoulderRoll",
    "RightShoulderYaw",
    "RightElbow",
    "RightWristRoll",
    "RightWristPitch",
    "RightWristYaw",
)


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skill", choices=("button", "knob", "switch"))
    parser.add_argument("--interface", default="enp8s0")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--config", type=Path, default=project / "config/task2_act_deploy.json"
    )
    args = parser.parse_args()

    official = Path("/home/drgh/g1_official/GR00T-WholeBodyControl")
    for dependency in (
        project / "vendor/unitree_sdk2_python",
        official / ".venv_teleop/lib/python3.10/site-packages",
    ):
        value = str(dependency)
        if value not in sys.path:
            sys.path.append(value)
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    target = json.loads(args.config.read_text())["skills"][args.skill][
        "start_action_median"
    ]
    low_state = None
    dex_state = None

    def receive_low(message: LowState_) -> None:
        nonlocal low_state
        low_state = message

    def receive_dex(message: MotorStates_) -> None:
        nonlocal dex_state
        if message.states:
            dex_state = message

    ChannelFactoryInitialize(0, args.interface)
    low_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    dex_subscriber = ChannelSubscriber("rt/dex1/right/state", MotorStates_)
    low_subscriber.Init(receive_low, 1)
    dex_subscriber.Init(receive_dex, 1)
    deadline = time.monotonic() + args.timeout
    while (low_state is None or dex_state is None) and time.monotonic() < deadline:
        time.sleep(0.05)
    if low_state is None or dex_state is None:
        print(
            "Missing "
            + " and ".join(
                name
                for value, name in (
                    (low_state, "rt/lowstate"),
                    (dex_state, "rt/dex1/right/state"),
                )
                if value is None
            )
        )
        low_subscriber.Close()
        dex_subscriber.Close()
        return 3

    rows = []
    for local_index, (body_index, name) in enumerate(
        zip(RIGHT_ARM_BODY_INDICES, RIGHT_ARM_NAMES)
    ):
        measured = float(low_state.motor_state[body_index].q)
        desired = float(target[local_index])
        rows.append((abs(desired - measured), name, measured, desired))
    dex_measured = float(dex_state.states[0].q)
    dex_target = float(target[7])
    rows.append((abs(dex_target - dex_measured), "RightDex1", dex_measured, dex_target))

    print(f"skill={args.skill} (read-only; no publishers created)")
    for absolute, name, measured, desired in rows:
        print(
            f"{name:20s} measured={measured:+.4f} target={desired:+.4f} "
            f"error={desired - measured:+.4f} abs={absolute:.4f}rad"
        )
    worst = max(rows)
    print(f"max_abs_error={worst[0]:.4f}rad joint={worst[1]}")
    low_subscriber.Close()
    dex_subscriber.Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

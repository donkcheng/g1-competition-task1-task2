"""One small, bounded, right-Dex1 command-path check; no PICO or body control."""

from __future__ import annotations

import argparse
import math
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmd_, MotorCmds_, MotorStates_


def read_q(sub: ChannelSubscriber, timeout: float = 0.2) -> float:
    msg = sub.Read(timeout)
    if not msg or len(msg.states) != 1:
        raise RuntimeError("Dex1 right state absent or malformed")
    q = float(msg.states[0].q)
    if not math.isfinite(q) or not -0.2 <= q <= 6.0:
        raise RuntimeError(f"Dex1 right state out of expected range: {q}")
    return q


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="enp8s0")
    parser.add_argument("--confirm-small-motion", action="store_true")
    parser.add_argument("--direction", choices=("auto", "close", "open"), default="auto")
    parser.add_argument("--delta", type=float, default=0.05, help="bounded position change in rad (max 0.2)")
    args = parser.parse_args()
    if not args.confirm_small_motion:
        parser.error("explicit --confirm-small-motion is required")
    if not math.isfinite(args.delta) or not 0 < args.delta <= 0.2:
        parser.error("--delta must be in (0, 0.2]")

    ChannelFactoryInitialize(0, args.interface)
    sub = ChannelSubscriber("rt/dex1/right/state", MotorStates_)
    sub.Init()
    pub = None
    try:
        baseline = [read_q(sub) for _ in range(10)]
        if max(baseline) - min(baseline) > 0.03:
            raise RuntimeError("gripper is already moving; refusing test")
        start_q = baseline[-1]
        direction = {
            "auto": 1.0 if start_q <= 5.4 else -1.0,
            "close": -1.0,
            "open": 1.0,
        }[args.direction]
        target_q = min(5.5, max(0.0, start_q + direction * args.delta))
        if abs(target_q - start_q) < 0.01:
            raise RuntimeError("already at the requested calibrated limit")
        print(f"Right Dex1 start={start_q:.4f}, bounded target={target_q:.4f} rad", flush=True)
        pub = ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)
        pub.Init()
        observations = []
        for step in range(125):
            q = read_q(sub)
            if abs(q - start_q) > args.delta + 0.1:
                raise RuntimeError(f"unexpected gripper movement: {q:.4f}")
            observations.append(q)
            desired = start_q + (target_q - start_q) * min(1.0, (step + 1) / 50.0)
            msg = MotorCmds_(cmds=[MotorCmd_(mode=1, q=desired, dq=0.0,
                                            tau=0.0, kp=5.0, kd=0.05,
                                            reserve=[0, 0, 0])])
            pub.Write(msg)
            time.sleep(0.02)
        end_q = read_q(sub)
        observed = direction * (end_q - start_q)
        print(f"Right Dex1 end={end_q:.4f}, observed motion={observed:.4f} rad", flush=True)
        if observed < 0.01:
            raise RuntimeError("command was sent but expected motion was not confirmed")
        return 0
    finally:
        if pub is not None:
            pub.Close()  # Official service brakes when DDS commands stop.
        sub.Close()


if __name__ == "__main__":
    raise SystemExit(main())

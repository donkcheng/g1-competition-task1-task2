"""One supervised Dex1-1 right-gripper open/close cycle; never controls G1 body."""

from __future__ import annotations

import argparse
import math
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmd_, MotorCmds_, MotorStates_


INTERFACE = "enp8s0"
OPEN_Q = 5.5  # Maximum commanded by Unitree's example after closed-at-zero calibration.
MIN_Q = -0.05
MAX_Q = 5.6
SPEED = 1.0  # rad/s command ramp
MAX_GAP = 0.35  # Maximum commanded position lead over actual position.
HZ = 50
PHASE_TIMEOUT = 25.0
STALL_TIMEOUT = 2.5


def read_q(sub: ChannelSubscriber) -> float:
    state = sub.Read(0.2)
    if state is None or len(state.states) != 1:
        raise RuntimeError("Dex1 right motor state missing")
    q = float(state.states[0].q)
    if not math.isfinite(q) or not MIN_Q <= q <= MAX_Q:
        raise RuntimeError(f"Dex1 right motor state invalid: {q}")
    return q


def move(sub: ChannelSubscriber, pub: ChannelPublisher, goal: float, label: str) -> float:
    start = time.monotonic()
    q = read_q(sub)
    commanded = q
    last_progress_q = q
    last_progress_time = start
    previous_q = q
    at_goal_count = 0
    last_report = start
    print(f"{label}: start={q:.3f}, target={goal:.3f}", flush=True)

    while time.monotonic() - start < PHASE_TIMEOUT:
        tick = time.monotonic()
        q = read_q(sub)
        if abs(q - previous_q) > 0.4:
            raise RuntimeError(f"implausible single-step position jump: {previous_q:.3f} -> {q:.3f}")
        previous_q = q
        error = goal - q
        if abs(error) <= 0.05:
            at_goal_count += 1
            if at_goal_count >= 10:
                print(f"{label}: reached near target, q={q:.3f}", flush=True)
                return q
        else:
            at_goal_count = 0

        if abs(q - last_progress_q) >= 0.015:
            last_progress_q = q
            last_progress_time = tick
        elif abs(error) > 0.15 and tick - last_progress_time > STALL_TIMEOUT:
            raise RuntimeError(f"{label} stalled at q={q:.3f}, target={goal:.3f}")

        step = min(SPEED / HZ, abs(goal - commanded))
        commanded += math.copysign(step, goal - commanded)
        commanded = min(q + MAX_GAP, max(q - MAX_GAP, commanded))
        commanded = min(OPEN_Q, max(0.0, commanded))
        pub.Write(MotorCmds_(cmds=[MotorCmd_(mode=1, q=commanded, dq=0.0,
                                              tau=0.0, kp=5.0, kd=0.05,
                                              reserve=[0, 0, 0])]))
        if tick - last_report >= 1.0:
            print(f"{label}: actual={q:.3f}, cmd={commanded:.3f}", flush=True)
            last_report = tick
        time.sleep(max(0.0, 1 / HZ - (time.monotonic() - tick)))
    raise RuntimeError(f"{label} timed out at q={q:.3f}, target={goal:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--close-only", action="store_true", help="finish an interrupted open/close cycle")
    args = parser.parse_args()
    ChannelFactoryInitialize(0, INTERFACE)
    sub = ChannelSubscriber("rt/dex1/right/state", MotorStates_)
    sub.Init()
    pub = None
    try:
        baseline = [read_q(sub) for _ in range(10)]
        if max(baseline) - min(baseline) > 0.03:
            raise RuntimeError("gripper is already moving")
        if args.close_only:
            if not 4.5 <= baseline[-1] <= 5.6:
                raise RuntimeError(f"expected open gripper for close-only, q={baseline[-1]:.3f}")
        elif not -0.05 <= baseline[-1] <= 0.5:
            raise RuntimeError(f"expected initially closed gripper, q={baseline[-1]:.3f}")
        pub = ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)
        pub.Init()
        if args.close_only:
            open_q = baseline[-1]
        else:
            open_q = move(sub, pub, OPEN_Q, "OPEN")
            if open_q < 5.0:
                raise RuntimeError(f"did not reach expected open range: q={open_q:.3f}")
        close_q = move(sub, pub, 0.0, "CLOSE")
        print(f"CYCLE COMPLETE: open={open_q:.3f}, closed={close_q:.3f}; verify jaws visually", flush=True)
        return 0
    finally:
        if pub is not None:
            pub.Close()  # Dex1 service brakes after its 1 s command timeout.
        sub.Close()


if __name__ == "__main__":
    raise SystemExit(main())

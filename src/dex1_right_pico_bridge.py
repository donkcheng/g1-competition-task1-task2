"""PICO right trigger -> calibrated right Dex1-1 (release=open, press=close).

Only the right gripper DDS topic is published. Commands require --arm, fresh
XRoboToolkit and motor streams, and a released trigger observed after startup.
Losing either stream stops commands; the Dex1-1 service brakes on timeout.
"""

from __future__ import annotations

import argparse
import math
import threading
import time

import xrobotoolkit_sdk as xrt
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmd_, MotorCmds_, MotorStates_


MIN_Q = 0.0  # Official calibration defines the mechanically closed pose as q=0.
OPEN_Q = 5.2  # Verified open sweep reached 5.279 rad; leave endpoint margin.
MAX_Q = 5.5  # Software state guard, beyond the verified open position.
MAX_SPEED = 1.0  # rad/s; full sweep takes about five seconds.
MAX_LEAD = 0.35  # Never command farther than this from measured position.
MAX_AGE = 0.3  # seconds
HZ = 50


def target_q(trigger: float) -> float:
    if not math.isfinite(trigger):
        raise ValueError("non-finite trigger input")
    return OPEN_Q * (1.0 - min(1.0, max(0.0, trigger)))


def step_q(current: float, target: float, dt: float) -> float:
    delta = min(MAX_SPEED * max(dt, 0.0), abs(target - current))
    return current + math.copysign(delta, target - current)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="enp8s0", help="PC wired robot interface")
    parser.add_argument("--arm", action="store_true", help="allow right-gripper commands")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="maximum armed run time in seconds (default: 30)")
    args = parser.parse_args()
    if args.duration <= 0 or not math.isfinite(args.duration):
        parser.error("--duration must be a positive finite number")

    xrt.init()
    ChannelFactoryInitialize(0, args.interface)
    lock = threading.Lock()
    last_state: tuple[float, float] | None = None

    def on_state(msg: MotorStates_) -> None:
        nonlocal last_state
        if not msg.states:
            return
        q = float(msg.states[0].q)
        if math.isfinite(q):
            with lock:
                last_state = (q, time.monotonic())

    sub = ChannelSubscriber("rt/dex1/right/state", MotorStates_)
    sub.Init(on_state, 1)
    pub = ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)
    pub.Init()

    print("Waiting for fresh Dex1-1 right motor state and PICO stream.", flush=True)
    print("Right trigger: released=open 5.2 rad; pressed=closed 0 rad.", flush=True)
    if not args.arm:
        print("Monitor-only mode. Restart with --arm to permit commands.", flush=True)

    last_xr_stamp = 0
    last_xr_update = 0.0
    last_tick = time.monotonic()
    commanded_q: float | None = None
    trigger_release_seen = False
    armed_at: float | None = None
    last_motion_q: float | None = None
    last_motion_at = time.monotonic()
    last_target_q: float | None = None
    last_status = ""
    last_values_at = 0.0
    try:
        while True:
            tick = time.monotonic()
            dt = min(tick - last_tick, 0.1)
            last_tick = tick

            stamp = int(xrt.get_time_stamp_ns())
            if stamp != last_xr_stamp and stamp > 0:
                last_xr_stamp = stamp
                last_xr_update = tick
            xr_fresh = (stamp > 0 and tick - last_xr_update <= MAX_AGE
                        and bool(xrt.is_body_data_available()))
            with lock:
                state = last_state
            state_fresh = state is not None and tick - state[1] <= MAX_AGE
            trigger = float(xrt.get_right_trigger()) if xr_fresh else 0.0
            emergency_combo = (bool(xrt.get_A_button()) and bool(xrt.get_B_button())
                               and bool(xrt.get_X_button()) and bool(xrt.get_Y_button())) if xr_fresh else False

            if not xr_fresh or not state_fresh or emergency_combo:
                trigger_release_seen = False
                commanded_q = None
                last_motion_q = None
                last_target_q = None
                status = ("waiting for XR" if not xr_fresh else
                          "waiting for motor state" if not state_fresh else "emergency buttons pressed")
            else:
                if not math.isfinite(trigger):
                    raise RuntimeError("non-finite right trigger value")
                measured_q = state[0]
                if not MIN_Q <= measured_q <= MAX_Q:
                    raise RuntimeError(f"Dex1-1 state outside calibrated range: q={measured_q:.3f}")
                if trigger <= 0.1:
                    trigger_release_seen = True
                if not args.arm:
                    status = "monitor-only"
                elif not trigger_release_seen:
                    status = "release trigger once to arm"
                else:
                    if armed_at is None:
                        armed_at = tick
                    if tick - armed_at >= args.duration:
                        print("Armed time limit reached; stopping commands.", flush=True)
                        break
                    desired_q = target_q(trigger)
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
                        raise RuntimeError("right gripper stalled; stopped publishing commands")
                    commanded_q = step_q(commanded_q, desired_q, dt)
                    commanded_q = min(measured_q + MAX_LEAD,
                                      max(measured_q - MAX_LEAD, commanded_q))
                    commanded_q = min(OPEN_Q, max(MIN_Q, commanded_q))
                    msg = MotorCmds_(cmds=[MotorCmd_(mode=1, q=commanded_q, dq=0.0,
                                                    tau=0.0, kp=5.0, kd=0.05,
                                                    reserve=[0, 0, 0])])
                    pub.Write(msg)
                    status = "armed"
                if tick - last_values_at >= 1.0:
                    print(f"trigger={trigger:.2f} target={target_q(trigger):.2f} "
                          f"actual={measured_q:.2f}", flush=True)
                    last_values_at = tick
            if status != last_status:
                print(status, flush=True)
                last_status = status
            if emergency_combo:
                print("PICO emergency button combination; bridge exiting.", flush=True)
                break
            time.sleep(max(0.0, 1.0 / HZ - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print("Stopped; Dex1-1 service will brake on command timeout.", flush=True)
    finally:
        pub.Close()
        sub.Close()
        xrt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

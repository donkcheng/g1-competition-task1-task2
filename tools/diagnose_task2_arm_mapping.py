#!/usr/bin/env python3
"""Read-only synchronized PICO/right-arm mapping diagnostic.

This process never publishes robot commands.  It samples the PICO controller
from XRoboToolkit and pairs it with the measured/commanded body state already
published by ``task2_right_arm_teleop.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import msgpack
import numpy as np
import zmq

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "tools"))

from task2_right_arm_teleop import (  # noqa: E402
    RightArmIK,
    controller_position_in_root,
)


def unpack_debug(message: bytes) -> dict:
    topic = b"g1_debug"
    if not message.startswith(topic):
        raise ValueError("unexpected debug topic")
    return msgpack.unpackb(message[len(topic):], raw=False)


def estimate_lag_ms(commanded: np.ndarray, measured: np.ndarray, period: float) -> float | None:
    if len(commanded) < 12:
        return None
    command_velocity = np.diff(commanded, axis=0)
    measured_velocity = np.diff(measured, axis=0)
    command_velocity -= command_velocity.mean(axis=0, keepdims=True)
    measured_velocity -= measured_velocity.mean(axis=0, keepdims=True)
    best_lag = 0
    best_score = -np.inf
    for lag in range(0, min(25, len(command_velocity) // 3) + 1):
        left = command_velocity[: len(command_velocity) - lag or None]
        right = measured_velocity[lag:]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 1.0e-9:
            continue
        score = float(np.sum(left * right) / denominator)
        if score > best_score:
            best_score = score
            best_lag = lag
    return 1000.0 * best_lag * period


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument("--state", default="tcp://127.0.0.1:5557")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "logs" / f"arm_mapping_{time.strftime('%Y%m%d_%H%M%S')}.csv",
    )
    args = parser.parse_args()

    import xrobotoolkit_sdk as xrt

    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.setsockopt(zmq.RCVTIMEO, 250)
    subscriber.connect(args.state)
    solver = RightArmIK()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    active_before = False
    pico_anchor = None
    command_anchor = None
    measured_anchor = None
    last_xr_timestamp = None
    started = time.monotonic()
    last_progress = started
    xrt.init()
    print(f"[MappingDiag] Recording read-only data for {args.seconds:.0f}s -> {args.output}", flush=True)
    try:
        while time.monotonic() - started < args.seconds:
            try:
                debug = unpack_debug(subscriber.recv())
            except zmq.Again:
                continue
            now = time.monotonic()
            measured_q = np.asarray(debug["body_q_measured"], dtype=np.float64)
            commanded_q = np.asarray(debug["last_action"], dtype=np.float64)
            measured_ee, _ = solver.tool_pose(solver.full_configuration(measured_q))
            commanded_ee, _ = solver.tool_pose(solver.full_configuration(commanded_q))
            xr_timestamp = xrt.get_time_stamp_ns() if xrt.is_body_data_available() else 0
            xr_fresh = xr_timestamp not in (0, last_xr_timestamp)
            if xr_fresh:
                last_xr_timestamp = xr_timestamp
            grip = float(xrt.get_right_grip()) if xr_timestamp else 0.0
            active = xr_timestamp != 0 and grip >= 0.55
            if active:
                body = np.asarray(xrt.get_body_joints_pose(), dtype=np.float64)
                controller = np.asarray(xrt.get_right_controller_pose(), dtype=np.float64)
                pico_position = controller_position_in_root(controller, body[0])
                if not active_before or pico_anchor is None:
                    pico_anchor = pico_position.copy()
                    command_anchor = commanded_ee.copy()
                    measured_anchor = measured_ee.copy()
                pico_delta = pico_position - pico_anchor
                command_delta = commanded_ee - command_anchor
                measured_delta = measured_ee - measured_anchor
                joint_error = commanded_q[22:29] - measured_q[22:29]
                rows.append({
                    "t": now - started,
                    "grip": grip,
                    "pico_x": pico_delta[0], "pico_y": pico_delta[1], "pico_z": pico_delta[2],
                    "expected_x": args.scale * pico_delta[0],
                    "expected_y": args.scale * pico_delta[1],
                    "expected_z": args.scale * pico_delta[2],
                    "command_x": command_delta[0], "command_y": command_delta[1], "command_z": command_delta[2],
                    "measured_x": measured_delta[0], "measured_y": measured_delta[1], "measured_z": measured_delta[2],
                    "joint_rmse_rad": float(np.sqrt(np.mean(joint_error * joint_error))),
                })
            active_before = active
            if now - last_progress >= 2.0:
                print(f"[MappingDiag] samples={len(rows)} grip={grip:.2f} xr={'fresh' if xr_fresh else 'unchanged'}", flush=True)
                last_progress = now
    finally:
        xrt.close()
        subscriber.close()
        context.term()

    if not rows:
        print("[MappingDiag] No active samples: right Grip was not detected.", flush=True)
        return 2
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pico = np.asarray([[r[f"pico_{axis}"] for axis in "xyz"] for r in rows])
    expected = args.scale * pico
    commanded = np.asarray([[r[f"command_{axis}"] for axis in "xyz"] for r in rows])
    measured = np.asarray([[r[f"measured_{axis}"] for axis in "xyz"] for r in rows])
    mapping = np.linalg.lstsq(pico, measured, rcond=None)[0] if len(rows) >= 6 else np.zeros((3, 3))
    period = args.seconds / max(1, len(rows))
    summary = {
        "samples": len(rows),
        "pico_range_cm": (np.ptp(pico, axis=0) * 100).round(1).tolist(),
        "command_range_cm": (np.ptp(commanded, axis=0) * 100).round(1).tolist(),
        "measured_range_cm": (np.ptp(measured, axis=0) * 100).round(1).tolist(),
        "expected_to_command_rmse_cm": float(np.sqrt(np.mean((expected - commanded) ** 2)) * 100),
        "command_to_measured_rmse_cm": float(np.sqrt(np.mean((commanded - measured) ** 2)) * 100),
        "joint_tracking_rmse_deg": float(np.degrees(np.mean([r["joint_rmse_rad"] for r in rows]))),
        "estimated_robot_lag_ms": estimate_lag_ms(commanded, measured, period),
        "fitted_pico_to_measured_matrix": np.round(mapping, 3).tolist(),
        "csv": str(args.output),
    }
    print("[MappingDiag] SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

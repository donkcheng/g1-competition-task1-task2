#!/usr/bin/env python3
"""Wait until the task-2 camera bridge provides fresh head and right-wrist frames."""

from __future__ import annotations

import argparse
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.123.222")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout", type=float, default=12.0)
    args = parser.parse_args()

    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

    client = ComposedCameraClientSensor(args.host, args.port)
    deadline = time.monotonic() + args.timeout
    last_detail = "no camera message"
    try:
        while time.monotonic() < deadline:
            message = client.read(blocking=False)
            if message is None:
                time.sleep(0.05)
                continue
            images = message.get("images", {})
            missing = {"ego_view", "right_wrist"} - set(images)
            if missing:
                last_detail = f"missing streams: {sorted(missing)}"
                time.sleep(0.05)
                continue
            shapes = {name: images[name].shape for name in ("ego_view", "right_wrist")}
            if any(shape != (480, 640, 3) for shape in shapes.values()):
                last_detail = f"unexpected image shapes: {shapes}"
                time.sleep(0.05)
                continue
            print(f"Dual cameras ready: {shapes}")
            return 0
        print(f"Dual-camera readiness timed out: {last_detail}")
        return 1
    finally:
        client.stop_client()


if __name__ == "__main__":
    raise SystemExit(main())

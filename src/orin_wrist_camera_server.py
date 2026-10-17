#!/usr/bin/env python3
"""Publish the G1 right-wrist RealSense D405 as a SONIC camera stream."""

from __future__ import annotations

import argparse
import signal
import socket
import struct
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def find_d405_serial(requested: str | None) -> str:
    devices = list(rs.context().query_devices())
    if requested:
        for device in devices:
            serial = device.get_info(rs.camera_info.serial_number)
            if serial == requested:
                return serial
        raise RuntimeError(f"RealSense serial {requested!r} is not connected")

    matches = []
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        if "D405" in name.upper():
            matches.append(device.get_info(rs.camera_info.serial_number))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one D405, found {matches}")
    return matches[0]


def encode_jpeg(image_bgr: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise RuntimeError("failed to encode D405 RGB frame")
    return encoded.tobytes()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5559)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--capture-fps", type=int, default=30)
    parser.add_argument("--publish-fps", type=float, default=20.0)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    args = parser.parse_args()

    if not 0 < args.publish_fps <= args.capture_fps:
        parser.error("publish-fps must be positive and no greater than capture-fps")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("jpeg-quality must be in [1, 100]")

    serial = find_d405_serial(args.serial)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.capture_fps,
    )

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.bind, args.port))
    server.listen(1)
    server.setblocking(False)
    client = None

    stop = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    sent = 0
    failures = 0
    last_report = time.monotonic()
    next_publish = time.monotonic()
    print(
        f"Right wrist D405 {serial} listening on {args.bind}:{args.port}",
        flush=True,
    )
    try:
        pipeline.start(config)
        while not stop:
            try:
                try:
                    new_client, address = server.accept()
                    new_client.settimeout(0.2)
                    if client is not None:
                        client.close()
                    client = new_client
                    print(f"Wrist client connected: {address}", flush=True)
                except BlockingIOError:
                    pass
                frames = pipeline.wait_for_frames(timeout_ms=2000)
                frame = frames.get_color_frame()
                if not frame:
                    raise RuntimeError("D405 returned no RGB frame")
                now = time.monotonic()
                if now < next_publish:
                    continue
                image = np.asanyarray(frame.get_data())
                jpeg = encode_jpeg(image, args.jpeg_quality)
                if client is not None:
                    try:
                        client.sendall(struct.pack("!dI", time.time(), len(jpeg)) + jpeg)
                        sent += 1
                    except OSError as exc:
                        print(f"Wrist client disconnected: {exc}", flush=True)
                        client.close()
                        client = None
                next_publish = now + 1.0 / args.publish_fps
            except RuntimeError as exc:
                failures += 1
                if failures <= 3 or failures % 20 == 0:
                    print(f"Wrist frame error ({failures}): {exc}", flush=True)
            now = time.monotonic()
            if now - last_report >= 5.0:
                print(f"Wrist camera: sent={sent}, failures={failures}", flush=True)
                last_report = now
    finally:
        try:
            pipeline.stop()
        except RuntimeError:
            pass
        if client is not None:
            client.close()
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

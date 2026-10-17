"""Publish the G1 head and optional right-wrist stream in SONIC format.

This intentionally does not open /dev/video4 or stop Unitree's videohub. The
head image is read through Unitree's VideoClient.  A right-wrist stream can be
merged from the D405 service running on Orin.
"""

from __future__ import annotations

import argparse
import socket as tcp_socket
import struct
import threading
import time

import cv2
import msgpack
import numpy as np
import zmq

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.rpc.client import Client


G1_CHEST_VIDEO_SERVICE = "videohub_chest"
G1_CHEST_VIDEO_API_VERSION = "1.0.0.1"
G1_CHEST_VIDEO_GET_IMAGE_API = 1001


class G1ChestVideoClient(Client):
    """Video client matching the service exported by G1's chest camera hub."""

    def __init__(self):
        # unitree_sdk2py's VideoClient targets the Go2 service named
        # ``videohub``.  G1 exports the same API as ``videohub_chest``.
        super().__init__(G1_CHEST_VIDEO_SERVICE, False)

    def Init(self) -> None:
        self._SetApiVerson(G1_CHEST_VIDEO_API_VERSION)
        self._RegistApi(G1_CHEST_VIDEO_GET_IMAGE_API, 0)

    def GetImageSample(self):
        return self._CallBinary(G1_CHEST_VIDEO_GET_IMAGE_API, [])


def prepare_image(jpeg: bytes, width: int = 640, height: int = 480) -> bytes:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("videohub did not return a decodable color image")
    source_h, source_w = image.shape[:2]
    target_aspect = width / height
    source_aspect = source_w / source_h
    if source_aspect > target_aspect:
        crop_w = round(source_h * target_aspect)
        left = (source_w - crop_w) // 2
        image = image[:, left : left + crop_w]
    elif source_aspect < target_aspect:
        crop_h = round(source_w / target_aspect)
        top = (source_h - crop_h) // 2
        image = image[top : top + crop_h]
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("failed to encode camera frame")
    return encoded.tobytes()


def pack_message(jpeg: bytes, timestamp: float) -> bytes:
    if not timestamp > 0:
        raise ValueError("timestamp must be positive")
    return msgpack.packb(
        {"timestamps": {"ego_view": timestamp},
         "images": {"ego_view": prepare_image(jpeg)}},
        use_bin_type=True,
    )


class WristCameraClient:
    """Reconnectable client for the lightweight Orin D405 TCP stream."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest: tuple[bytes, float, float] | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _receive_exact(connection: tcp_socket.socket, length: int) -> bytes:
        chunks = []
        remaining = length
        while remaining:
            chunk = connection.recv(remaining)
            if not chunk:
                raise ConnectionError("right_wrist connection closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                with tcp_socket.create_connection(
                    (self.host, self.port), timeout=2.0
                ) as connection:
                    connection.settimeout(2.0)
                    print(
                        f"Connected to right wrist at {self.host}:{self.port}",
                        flush=True,
                    )
                    while not self.stop_event.is_set():
                        header = self._receive_exact(connection, 12)
                        source_timestamp, length = struct.unpack("!dI", header)
                        if source_timestamp <= 0 or not 0 < length <= 5_000_000:
                            raise ValueError("invalid right_wrist frame header")
                        jpeg = self._receive_exact(connection, length)
                        with self.lock:
                            # Orin and PC clocks are not guaranteed to be synchronized.
                            # Use the PC receipt time for cross-camera pairing/staleness.
                            self.latest = (jpeg, time.time(), time.monotonic())
            except (OSError, ValueError) as exc:
                if not self.stop_event.is_set():
                    print(f"Right wrist reconnecting after: {exc}", flush=True)
                    self.stop_event.wait(1.0)

    def start(self) -> None:
        self.thread.start()

    def read(self, max_age: float) -> tuple[bytes, float] | None:
        with self.lock:
            latest = self.latest
        if latest is None or time.monotonic() - latest[2] > max_age:
            return None
        return latest[0], latest[1]

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3.0)


def pack_composed_message(
    head_jpeg: bytes,
    head_timestamp: float,
    wrist: tuple[bytes, float] | None,
) -> bytes:
    timestamps = {"ego_view": head_timestamp}
    images = {"ego_view": prepare_image(head_jpeg)}
    if wrist is not None:
        wrist_jpeg, wrist_timestamp = wrist
        timestamps["right_wrist"] = wrist_timestamp
        images["right_wrist"] = wrist_jpeg
    return msgpack.packb(
        {"timestamps": timestamps, "images": images}, use_bin_type=True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="enp8s0")
    parser.add_argument("--bind", default="192.168.123.222")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--wrist-host", default=None)
    parser.add_argument("--wrist-port", type=int, default=5559)
    parser.add_argument("--wrist-max-age", type=float, default=0.35)
    parser.add_argument("--require-wrist", action="store_true")
    args = parser.parse_args()
    if not 0 < args.fps <= 30:
        parser.error("fps must be in (0, 30]")

    ChannelFactoryInitialize(0, args.interface)
    client = G1ChestVideoClient()
    client.SetTimeout(2.0)
    client.Init()
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 3)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://{args.bind}:{args.port}")

    wrist_client = None
    if args.wrist_host:
        wrist_client = WristCameraClient(args.wrist_host, args.wrist_port)
        wrist_client.start()
        print(
            f"Subscribing to right wrist at "
            f"tcp://{args.wrist_host}:{args.wrist_port}",
            flush=True,
        )
    elif args.require_wrist:
        parser.error("--require-wrist requires --wrist-host")

    print(f"Camera bridge listening on tcp://{args.bind}:{args.port}", flush=True)

    sent = 0
    failures = 0
    last_report = time.monotonic()
    try:
        while True:
            start = time.monotonic()
            try:
                latest_wrist = (
                    wrist_client.read(args.wrist_max_age)
                    if wrist_client is not None
                    else None
                )
                wrist_fresh = latest_wrist is not None
                if args.require_wrist and not wrist_fresh:
                    raise RuntimeError("right_wrist stream is missing or stale")
                code, data = client.GetImageSample()
                if code != 0 or not data:
                    raise RuntimeError(f"videohub error code {code}")
                packet = pack_composed_message(
                    bytes(data),
                    time.time(),
                    latest_wrist if wrist_fresh else None,
                )
                socket.send(packet, flags=zmq.NOBLOCK)
                sent += 1
            except (ValueError, RuntimeError, zmq.Again) as exc:
                failures += 1
                if failures <= 3 or failures % 20 == 0:
                    print(f"Camera frame error ({failures}): {exc}", flush=True)
            now = time.monotonic()
            if now - last_report >= 5.0:
                print(f"Camera bridge: sent={sent}, failures={failures}", flush=True)
                last_report = now
            time.sleep(max(0.0, 1.0 / args.fps - (time.monotonic() - start)))
    except KeyboardInterrupt:
        print("Camera bridge stopped", flush=True)
    finally:
        if wrist_client is not None:
            wrist_client.close()
        socket.close()
        context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

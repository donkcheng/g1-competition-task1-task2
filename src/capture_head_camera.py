#!/usr/bin/env python3
"""Interactive still capture for the G1 onboard RealSense RGB camera."""

import argparse
import select
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--warmup-frames", type=int, default=90)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def save_frame(frame, output_dir, sequence, label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = output_dir / f"{sequence:03d}_{label}_{timestamp}.jpg"
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"failed to save {path}")
    return path


def main():
    args = parse_args()
    if not sys.stdin.isatty():
        raise SystemExit("ERROR: an interactive terminal is required (use ssh -t)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        raise SystemExit(f"ERROR: cannot open /dev/video{args.device}")

    frame = None
    try:
        for _ in range(args.warmup_frames):
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("camera frame read failed during warmup")

        fd = sys.stdin.fileno()
        original_terminal = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        sequence = 0
        print("相机已就绪：空格/回车=拍照，1/2/3=按点位拍照，q=退出并下载", flush=True)

        try:
            while True:
                ok, latest = camera.read()
                if ok:
                    frame = latest
                readable, _, _ = select.select([sys.stdin], [], [], 0)
                if not readable:
                    continue

                key = sys.stdin.read(1)
                if key.lower() == "q":
                    break
                if key not in (" ", "\r", "\n", "1", "2", "3", "s", "S"):
                    continue
                if frame is None:
                    print("WARN: 暂无有效画面", flush=True)
                    continue

                sequence += 1
                label = f"point_{key}" if key in ("1", "2", "3") else "capture"
                path = save_frame(frame, args.output_dir, sequence, label)
                print(f"已拍摄 {sequence}: {path.name}", flush=True)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, original_terminal)
    finally:
        camera.release()

    print(f"拍摄结束，共 {sequence} 张", flush=True)


if __name__ == "__main__":
    main()

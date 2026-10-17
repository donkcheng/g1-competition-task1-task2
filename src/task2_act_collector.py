#!/usr/bin/env python3
"""Continuous dual-camera ACT data collection for task 2.

The collector records successful demonstrations only.  An aborted attempt is
kept in RAM and discarded without consuming a LeRobot episode index.  Existing
datasets are resumed automatically.

PICO controls (from the running SONIC manager):
  left grip + A  start recording / stop and save
  left grip + B  abort current attempt

Terminal fallbacks require Enter: ``s`` toggle, ``x`` abort, ``q`` quit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import sys
import time
from typing import Any

import msgpack
import numpy as np
import zmq


BODY_JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch",
    "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch",
    "right_ankle_roll", "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
RIGHT_ARM_INDICES = np.arange(22, 29)
RIGHT_ARM_ACTION_NAMES = [BODY_JOINT_NAMES[index] for index in RIGHT_ARM_INDICES]
TASK_DESCRIPTIONS = {
    "button": "Press the button and hold it for at least two seconds.",
    "knob": "Rotate the knob by at least fifteen degrees.",
    "switch": "Toggle the switch by at least fifteen degrees and hold for two seconds.",
}


def dataset_features() -> dict[str, dict[str, Any]]:
    return {
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
        },
        "observation.images.right_wrist": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (30,),
            "names": BODY_JOINT_NAMES + ["right_gripper_q"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": RIGHT_ARM_ACTION_NAMES + ["right_gripper_target_q"],
        },
    }


def modality_config() -> dict[str, dict[str, Any]]:
    return {
        "state": {
            "g1_body_and_right_gripper": {
                "start": 0,
                "end": 30,
                "original_key": "observation.state",
            }
        },
        "action": {
            "right_arm_and_gripper": {
                "start": 0,
                "end": 8,
                "original_key": "action",
            }
        },
        "video": {
            "ego_view": {"original_key": "observation.images.ego_view"},
            "right_wrist": {"original_key": "observation.images.right_wrist"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"}
        },
    }


def encode_rgb(image: np.ndarray, quality: int = 88) -> bytes:
    import cv2

    if image.shape != (480, 640, 3):
        raise ValueError(f"expected 640x480 RGB image, got {image.shape}")
    ok, encoded = cv2.imencode(
        ".jpg", image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise RuntimeError("failed to compress buffered camera frame")
    return encoded.tobytes()


def decode_rgb(jpeg: bytes) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (480, 640, 3):
        raise ValueError("failed to decode buffered 640x480 camera frame")
    return np.ascontiguousarray(image[..., ::-1])


@dataclass
class BufferedFrame:
    ego_jpeg: bytes
    wrist_jpeg: bytes
    state: np.ndarray
    action: np.ndarray


@dataclass
class PendingEpisode:
    started_monotonic: float = 0.0
    frames: list[BufferedFrame] = field(default_factory=list)

    def start(self, now: float) -> None:
        self.started_monotonic = now
        self.frames.clear()

    def abort(self) -> int:
        count = len(self.frames)
        self.frames.clear()
        self.started_monotonic = 0.0
        return count

    @property
    def active(self) -> bool:
        return self.started_monotonic > 0.0


class TopicSubscriber:
    def __init__(self, endpoint: str, topic: bytes):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, topic)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.RCVHWM, 3)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(endpoint)
        self.topic = topic
        self.latest = None
        self.received_monotonic = 0.0

    def poll_msgpack(self) -> dict[str, Any] | None:
        try:
            raw = self.socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return self.latest
        if not raw.startswith(self.topic):
            return self.latest
        self.latest = msgpack.unpackb(raw[len(self.topic):], raw=False)
        self.received_monotonic = time.monotonic()
        return self.latest

    def poll_raw(self) -> bytes | None:
        try:
            raw = self.socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        self.received_monotonic = time.monotonic()
        return raw

    def close(self) -> None:
        self.socket.close()
        self.context.term()


class LiveSources:
    def __init__(
        self,
        camera_host: str,
        camera_port: int,
        state_host: str,
        state_port: int,
        sonic_host: str,
        sonic_port: int,
        gripper_host: str,
        gripper_port: int,
    ):
        from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
        from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

        self.camera = ComposedCameraClientSensor(camera_host, camera_port)
        self.robot = ZMQStateSubscriber(state_host, state_port)
        self.gripper = TopicSubscriber(f"tcp://{gripper_host}:{gripper_port}", b"dex1")
        self.manager = TopicSubscriber(f"tcp://{sonic_host}:{sonic_port}", b"manager_state")
        self.camera_msg = None
        self.camera_received_monotonic = 0.0
        self.camera_signature = None
        self.robot_msg = None
        self.robot_received_monotonic = 0.0
        self.manager_seen = False

    def poll(self, unpack_pose_message) -> list[str]:
        commands = []
        camera_msg = self.camera.read(blocking=False)
        if camera_msg is not None:
            timestamps = camera_msg.get("timestamps", {})
            signature = tuple(sorted(timestamps.items()))
            if signature != self.camera_signature:
                self.camera_signature = signature
                self.camera_msg = camera_msg
                self.camera_received_monotonic = time.monotonic()

        robot_msg = self.robot.get_msg(clear=True)
        if robot_msg is not None:
            self.robot_msg = robot_msg
            self.robot_received_monotonic = time.monotonic()

        self.gripper.poll_msgpack()
        while True:
            raw = self.manager.poll_raw()
            if raw is None:
                break
            self.manager_seen = True
            try:
                data = unpack_pose_message(raw, topic="manager_state")
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
            if bool(np.asarray(data.get("toggle_data_abort", [False])).flat[0]):
                commands.append("abort")
            elif bool(np.asarray(data.get("toggle_data_collection", [False])).flat[0]):
                commands.append("toggle")
        return commands

    def readiness(self, max_age: float) -> tuple[bool, str]:
        now = time.monotonic()
        missing = []
        if self.camera_msg is None or now - self.camera_received_monotonic > max_age:
            missing.append("dual camera")
        else:
            images = self.camera_msg.get("images", {})
            if not {"ego_view", "right_wrist"}.issubset(images):
                missing.append("ego_view/right_wrist")
        if self.robot_msg is None or now - self.robot_received_monotonic > max_age:
            missing.append("g1_debug")
        if (
            self.gripper.latest is None
            or now - self.gripper.received_monotonic > max_age
            or not self.gripper.latest.get("armed", False)
        ):
            missing.append("armed Dex1 telemetry")
        if not self.manager_seen or now - self.manager.received_monotonic > max_age:
            missing.append("PICO manager_state")
        return not missing, ", ".join(missing) if missing else "all streams fresh"

    def make_frame(self, max_age: float) -> BufferedFrame:
        ready, detail = self.readiness(max_age)
        if not ready:
            raise RuntimeError(detail)
        assert self.camera_msg is not None
        assert self.robot_msg is not None
        assert self.gripper.latest is not None

        body_q = self.robot_msg.get("body_q_measured")
        if body_q is None:
            body_q = self.robot_msg.get("body_q")
        last_action = self.robot_msg.get("last_action")
        measured_gripper = self.gripper.latest.get("measured_q")
        commanded_gripper = self.gripper.latest.get("commanded_q")
        body_q = np.asarray(body_q, dtype=np.float32).reshape(-1)
        last_action = np.asarray(last_action, dtype=np.float32).reshape(-1)
        if body_q.size < 29 or last_action.size < 29:
            raise RuntimeError(
                f"invalid G1 vectors body_q={body_q.size}, last_action={last_action.size}"
            )
        if measured_gripper is None or commanded_gripper is None:
            raise RuntimeError("Dex1 command/state unavailable")
        if not math.isfinite(float(measured_gripper)) or not math.isfinite(
            float(commanded_gripper)
        ):
            raise RuntimeError("non-finite Dex1 command/state")

        state = np.concatenate(
            (body_q[:29], np.array([measured_gripper], dtype=np.float32))
        ).astype(np.float32)
        action = np.concatenate(
            (
                last_action[RIGHT_ARM_INDICES],
                np.array([commanded_gripper], dtype=np.float32),
            )
        ).astype(np.float32)
        images = self.camera_msg["images"]
        return BufferedFrame(
            ego_jpeg=encode_rgb(np.asarray(images["ego_view"])),
            wrist_jpeg=encode_rgb(np.asarray(images["right_wrist"])),
            state=state,
            action=action,
        )

    def close(self) -> None:
        self.camera.stop_client()
        self.robot.close()
        self.gripper.close()
        self.manager.close()


def terminal_commands() -> list[str]:
    if not sys.stdin.isatty():
        return []
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return []
    value = sys.stdin.readline().strip().lower()
    return {"s": ["toggle"], "x": ["abort"], "q": ["quit"]}.get(value, [])


def append_manifest(root: Path, event: dict[str, Any]) -> None:
    path = root / "collection_manifest.json"
    manifest = {"version": 1, "events": []}
    if path.exists():
        manifest = json.loads(path.read_text())
    manifest.setdefault("events", []).append(event)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def close_empty_video_writers(exporter) -> None:
    if exporter.episode_buffer.get("size", 0) != 0:
        return
    for writer in exporter.video_writers.values():
        if writer.queue.empty():
            writer.cancel()


def save_pending(exporter, pending: PendingEpisode, task: str) -> int:
    count = len(pending.frames)
    for frame in pending.frames:
        exporter.add_frame(
            {
                "observation.images.ego_view": decode_rgb(frame.ego_jpeg),
                "observation.images.right_wrist": decode_rgb(frame.wrist_jpeg),
                "observation.state": frame.state,
                "action": frame.action,
                "task": TASK_DESCRIPTIONS[task],
            }
        )
    exporter.save_episode()
    pending.abort()
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--skill", choices=tuple(TASK_DESCRIPTIONS), required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--max-source-age", type=float, default=0.5)
    parser.add_argument("--preflight-timeout", type=float, default=15.0)
    parser.add_argument("--camera-host", default="192.168.123.222")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--state-host", default="127.0.0.1")
    parser.add_argument("--state-port", type=int, default=5557)
    parser.add_argument("--sonic-host", default="127.0.0.1")
    parser.add_argument("--sonic-port", type=int, default=5556)
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--gripper-port", type=int, default=5558)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.dataset) is None:
        raise SystemExit("--dataset must be a simple name without spaces or path separators")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if not 0 < args.min_duration < args.max_duration:
        raise SystemExit("duration must satisfy 0 < min < max")
    sys.path.insert(0, str(args.repo))

    from gear_sonic.data.exporter import Gr00tDataExporter
    from gear_sonic.scripts.run_data_exporter import unpack_pose_message

    sources = LiveSources(
        args.camera_host, args.camera_port,
        args.state_host, args.state_port,
        args.sonic_host, args.sonic_port,
        args.gripper_host, args.gripper_port,
    )
    exporter = None
    pending = PendingEpisode()
    dataset_path = args.data_root / args.dataset
    period = 1.0 / args.fps
    stale_since = None
    last_status = 0.0
    stop_requested = False

    def request_stop(_signum=None, _frame=None):
        nonlocal stop_requested
        stop_requested = True

    # This program runs as an asynchronous shell child. Explicitly replace
    # the inherited signal dispositions so session cleanup cannot hang.
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        print("Waiting for dual camera, G1 state, Dex1 telemetry and PICO manager...", flush=True)
        deadline = time.monotonic() + args.preflight_timeout
        while time.monotonic() < deadline:
            sources.poll(unpack_pose_message)
            ready, detail = sources.readiness(args.max_source_age)
            if ready:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(f"collection preflight failed: {detail}")
        print(f"Collection sources ready: {detail}", flush=True)
        if args.check_only:
            return 0

        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        exporter = Gr00tDataExporter.create(
            save_root=dataset_path,
            fps=args.fps,
            features=dataset_features(),
            modality_config=modality_config(),
            task=TASK_DESCRIPTIONS[args.skill],
            script_config={
                "collector": "task2_act_collector_v1",
                "skill": args.skill,
                "body_joint_count": 29,
                "right_arm_indices": RIGHT_ARM_INDICES.tolist(),
                "right_gripper": "dex1_1",
                "cameras": ["ego_view", "right_wrist"],
            },
            robot_type="unitree_g1_dex1_right",
        )
        print(
            f"Ready: {dataset_path} (next episode {exporter.meta.total_episodes})\n"
            "PICO left grip+A=start/stop-save, left grip+B=abort. "
            "Terminal: s/x/q + Enter.",
            flush=True,
        )

        next_tick = time.monotonic()
        while not stop_requested:
            commands = sources.poll(unpack_pose_message) + terminal_commands()
            for command in commands:
                if command == "quit":
                    if pending.active:
                        discarded = pending.abort()
                        print(f"Quit: discarded {discarded} pending frames", flush=True)
                    return 0
                if command == "abort":
                    if pending.active:
                        discarded = pending.abort()
                        append_manifest(dataset_path, {
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "event": "aborted",
                            "frames": discarded,
                        })
                        print(
                            f"Attempt aborted ({discarded} frames); episode number reused",
                            flush=True,
                        )
                    continue
                if command != "toggle":
                    continue
                if not pending.active:
                    ready, detail = sources.readiness(args.max_source_age)
                    if not ready:
                        print(f"Cannot start: {detail}", flush=True)
                        continue
                    pending.start(time.monotonic())
                    stale_since = None
                    print(
                        f"RECORDING episode {exporter.meta.total_episodes:04d}",
                        flush=True,
                    )
                else:
                    min_frames = math.ceil(args.min_duration * args.fps)
                    if len(pending.frames) < min_frames:
                        discarded = pending.abort()
                        print(
                            f"Too short ({discarded} < {min_frames} frames); discarded",
                            flush=True,
                        )
                    else:
                        episode_index = exporter.meta.total_episodes
                        frame_count = save_pending(
                            exporter, pending, args.skill
                        )
                        append_manifest(dataset_path, {
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "event": "saved",
                            "episode_index": episode_index,
                            "frames": frame_count,
                            "skill": args.skill,
                        })
                        print(
                            f"SAVED episode {episode_index:04d}: {frame_count} frames; "
                            "ready for next attempt",
                            flush=True,
                        )

            now = time.monotonic()
            if pending.active and now >= next_tick:
                try:
                    pending.frames.append(sources.make_frame(args.max_source_age))
                    stale_since = None
                except (RuntimeError, ValueError) as exc:
                    if stale_since is None:
                        stale_since = now
                    elif now - stale_since >= args.max_source_age:
                        discarded = pending.abort()
                        append_manifest(dataset_path, {
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "event": "auto_aborted_stale_source",
                            "frames": discarded,
                            "reason": str(exc),
                        })
                        print(
                            f"AUTO-ABORT: {exc}; discarded {discarded} frames",
                            flush=True,
                        )
                if pending.active and now - pending.started_monotonic >= args.max_duration:
                    episode_index = exporter.meta.total_episodes
                    frame_count = save_pending(exporter, pending, args.skill)
                    append_manifest(dataset_path, {
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "event": "saved_at_max_duration",
                        "episode_index": episode_index,
                        "frames": frame_count,
                        "skill": args.skill,
                    })
                    print(
                        f"SAVED episode {episode_index:04d} at max duration; ready",
                        flush=True,
                    )
                next_tick += period
                if next_tick < now - period:
                    next_tick = now + period

            if now - last_status >= 2.0:
                state = "RECORDING" if pending.active else "ARMED"
                ready, detail = sources.readiness(args.max_source_age)
                print(
                    f"[{state}] frames={len(pending.frames)} sources="
                    f"{'OK' if ready else detail}",
                    flush=True,
                )
                last_status = now
            time.sleep(0.005)
        if pending.active:
            discarded = pending.abort()
            print(f"Stopped: discarded {discarded} pending frames", flush=True)
        return 0
    finally:
        if pending.active:
            pending.abort()
        if exporter is not None:
            close_empty_video_writers(exporter)
        sources.close()


if __name__ == "__main__":
    raise SystemExit(main())

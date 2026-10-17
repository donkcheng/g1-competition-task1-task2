#!/usr/bin/env python3
"""Start-pose alignment for Task 1 using a fixed ArUco reference marker.

The tool never publishes robot commands.  It measures the head-camera pose
relative to a marker that stays fixed in the venue and tells the operator how
to correct the robot with the original remote before recording/replaying a
route.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class MarkerPose:
    camera_from_marker: np.ndarray
    reprojection_error_px: float
    corners: np.ndarray
    frame: np.ndarray


@dataclass(frozen=True)
class AlignmentError:
    forward_m: float
    left_m: float
    yaw_left_deg: float


def center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Match the 4:3 frame used when the stored camera intrinsics were measured."""
    source_h, source_w = image.shape[:2]
    target_aspect = width / height
    source_aspect = source_w / source_h
    if source_aspect > target_aspect:
        crop_w = round(source_h * target_aspect)
        left = (source_w - crop_w) // 2
        image = image[:, left:left + crop_w]
    elif source_aspect < target_aspect:
        crop_h = round(source_w / target_aspect)
        top = (source_h - crop_h) // 2
        image = image[top:top + crop_h]
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


class VideoHubSource:
    def __init__(self, interface: str, timeout_s: float, width: int, height: int):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from videohub_camera_server import G1ChestVideoClient

        ChannelFactoryInitialize(0, interface)
        # G1 publishes the head/chest RealSense through ``videohub_chest``.
        # The SDK's generic VideoClient targets the Go2-only ``videohub``
        # service and times out with code 3102 on this robot.
        self.client = G1ChestVideoClient()
        self.client.SetTimeout(timeout_s)
        self.client.Init()
        self.width = width
        self.height = height

    def read(self) -> np.ndarray:
        code, data = self.client.GetImageSample()
        if code != 0 or not data:
            raise RuntimeError(f"head videohub GetImageSample failed with code {code}")
        image = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("head videohub returned an invalid JPEG")
        return center_crop_resize(image, self.width, self.height)

    def close(self) -> None:
        return None


class ZmqSource:
    def __init__(self, host: str, port: int, timeout_s: float):
        import msgpack
        import zmq

        self.msgpack = msgpack
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.RCVTIMEO, round(timeout_s * 1000))
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{host}:{port}")

    def read(self) -> np.ndarray:
        try:
            packet = self.msgpack.unpackb(self.socket.recv(), raw=False)
        except Exception as exc:
            raise RuntimeError(f"camera stream receive failed: {exc}") from exc
        jpeg = packet.get("images", {}).get("ego_view")
        if not jpeg:
            raise RuntimeError("camera packet has no ego_view")
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("camera stream returned an invalid ego_view JPEG")
        return image

    def close(self) -> None:
        self.socket.close()
        self.context.term()


class ImageSource:
    def __init__(self, path: Path):
        self.image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if self.image is None:
            raise RuntimeError(f"cannot read test image: {path}")

    def read(self) -> np.ndarray:
        return self.image.copy()

    def close(self) -> None:
        return None


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dictionary", "marker_id", "marker_size_m", "camera_width",
        "camera_height", "camera_matrix", "dist_coeffs", "reference_file",
        "capture_samples", "guide_samples", "max_reprojection_error_px",
        "max_translation_spread_m", "forward_tolerance_m",
        "lateral_tolerance_m", "yaw_tolerance_deg", "stable_cycles",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError("alignment config is missing: " + ", ".join(missing))
    return config


def camera_parameters(config: dict[str, Any], shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape[:2]
    base_width = int(config["camera_width"])
    base_height = int(config["camera_height"])
    if abs(width / height - base_width / base_height) > 0.01:
        raise RuntimeError(
            f"camera aspect ratio {width}x{height} differs from calibration "
            f"{base_width}x{base_height}"
        )
    matrix = np.asarray(config["camera_matrix"], dtype=np.float64).copy()
    matrix[0, :] *= width / base_width
    matrix[1, :] *= height / base_height
    matrix[2, 2] = 1.0
    distortion = np.asarray(config["dist_coeffs"], dtype=np.float64)
    return matrix, distortion


def aruco_dictionary(name: str):
    dictionary_id = getattr(cv2.aruco, name, None)
    if dictionary_id is None:
        raise ValueError(f"unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def detect_marker(frame: np.ndarray, config: dict[str, Any]) -> MarkerPose | None:
    dictionary = aruco_dictionary(str(config["dictionary"]))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners_list, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return None
    matches = np.flatnonzero(ids.reshape(-1) == int(config["marker_id"]))
    if len(matches) != 1:
        return None
    corners = np.asarray(corners_list[int(matches[0])], dtype=np.float64).reshape(4, 2)
    side = float(config["marker_size_m"])
    half = side / 2.0
    object_points = np.array(
        [[-half, half, 0.0], [half, half, 0.0],
         [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )
    matrix, distortion = camera_parameters(config, frame.shape)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, corners, matrix, distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok or float(tvec.reshape(3)[2]) <= 0:
        return None
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, matrix, distortion)
    error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - corners) ** 2, axis=1))))
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.reshape(3)
    return MarkerPose(transform, error, corners, frame)


def rotation_mean(rotations: list[np.ndarray]) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.mean(rotations, axis=0))
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def average_poses(poses: list[MarkerPose], config: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    if not poses:
        raise RuntimeError("no valid marker observations")
    translations = np.asarray([pose.camera_from_marker[:3, 3] for pose in poses])
    median = np.median(translations, axis=0)
    distances = np.linalg.norm(translations - median, axis=1)
    mad = float(np.median(distances))
    limit = max(0.005, 3.0 * mad)
    keep = [pose for pose, distance in zip(poses, distances) if distance <= limit]
    if len(keep) < max(5, len(poses) // 2):
        raise RuntimeError("marker observations are too unstable; hold robot and marker still")
    translations = np.asarray([pose.camera_from_marker[:3, 3] for pose in keep])
    spread = float(np.max(np.linalg.norm(translations - np.median(translations, axis=0), axis=1)))
    if spread > float(config["max_translation_spread_m"]):
        raise RuntimeError(
            f"marker translation spread {spread:.3f}m exceeds "
            f"{float(config['max_translation_spread_m']):.3f}m"
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_mean([pose.camera_from_marker[:3, :3] for pose in keep])
    transform[:3, 3] = np.median(translations, axis=0)
    quality = {
        "samples": float(len(keep)),
        "translation_spread_m": spread,
        "median_reprojection_error_px": float(
            np.median([pose.reprojection_error_px for pose in keep])
        ),
    }
    return transform, quality


def collect_pose(source: Any, config: dict[str, Any], samples: int, timeout_s: float) -> tuple[np.ndarray, dict[str, float], MarkerPose]:
    deadline = time.monotonic() + timeout_s
    poses: list[MarkerPose] = []
    last_pose: MarkerPose | None = None
    failures = 0
    while len(poses) < samples and time.monotonic() < deadline:
        try:
            frame = source.read()
        except RuntimeError:
            failures += 1
            if failures >= 3:
                raise
            continue
        pose = detect_marker(frame, config)
        if pose is None:
            continue
        if pose.reprojection_error_px > float(config["max_reprojection_error_px"]):
            continue
        poses.append(pose)
        last_pose = pose
    if len(poses) < samples:
        raise RuntimeError(
            f"only {len(poses)}/{samples} valid marker frames in {timeout_s:.1f}s; "
            "make marker larger/closer and keep the full white border visible"
        )
    transform, quality = average_poses(poses, config)
    assert last_pose is not None
    return transform, quality, last_pose


def _planar_camera_pose(camera_from_marker: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera position/heading on the plane of a marker fixed flat on the floor."""
    marker_from_camera = np.linalg.inv(camera_from_marker)
    position = marker_from_camera[:2, 3]
    rotation = marker_from_camera[:3, :3]
    forward = rotation[:2, 2]
    right = rotation[:2, 0]
    forward_norm = float(np.linalg.norm(forward))
    right_norm = float(np.linalg.norm(right))
    if forward_norm < 0.1 or right_norm < 0.1:
        raise RuntimeError("camera/marker geometry is unsuitable; place the marker flat ahead of the robot")
    forward = forward / forward_norm
    right = right / right_norm
    left = np.array([-forward[1], forward[0]])
    if float(np.dot(left, -right)) < 0.0:
        left = -left
    return position, forward, left


def alignment_error(reference_camera_from_marker: np.ndarray, current_camera_from_marker: np.ndarray) -> AlignmentError:
    reference_position, reference_forward, reference_left = _planar_camera_pose(reference_camera_from_marker)
    current_position, current_forward, _ = _planar_camera_pose(current_camera_from_marker)
    displacement = current_position - reference_position
    cross = reference_forward[0] * current_forward[1] - reference_forward[1] * current_forward[0]
    dot = float(np.dot(reference_forward, current_forward))
    yaw_left = math.atan2(float(cross), dot)
    return AlignmentError(
        forward_m=float(np.dot(displacement, reference_forward)),
        left_m=float(np.dot(displacement, reference_left)),
        yaw_left_deg=math.degrees(yaw_left),
    )


def is_aligned(error: AlignmentError, config: dict[str, Any]) -> bool:
    return (
        abs(error.forward_m) <= float(config["forward_tolerance_m"])
        and abs(error.left_m) <= float(config["lateral_tolerance_m"])
        and abs(error.yaw_left_deg) <= float(config["yaw_tolerance_deg"])
    )


def guidance(error: AlignmentError, config: dict[str, Any]) -> list[str]:
    result: list[str] = []
    yaw_tol = float(config["yaw_tolerance_deg"])
    if error.yaw_left_deg > yaw_tol:
        result.append(f"先向右转 {abs(error.yaw_left_deg):.1f}°")
    elif error.yaw_left_deg < -yaw_tol:
        result.append(f"先向左转 {abs(error.yaw_left_deg):.1f}°")
    if abs(error.yaw_left_deg) <= max(5.0, yaw_tol):
        forward_tol = float(config["forward_tolerance_m"])
        lateral_tol = float(config["lateral_tolerance_m"])
        if error.forward_m > forward_tol:
            result.append(f"向后移动 {abs(error.forward_m):.2f}m")
        elif error.forward_m < -forward_tol:
            result.append(f"向前移动 {abs(error.forward_m):.2f}m")
        if error.left_m > lateral_tol:
            result.append(f"向右横移 {abs(error.left_m):.2f}m")
        elif error.left_m < -lateral_tol:
            result.append(f"向左横移 {abs(error.left_m):.2f}m")
    return result or ["位置和朝向已在允许误差内"]


def annotate(pose: MarkerPose, text: str) -> np.ndarray:
    image = pose.frame.copy()
    points = np.round(pose.corners).astype(int)
    cv2.polylines(image, [points], True, (0, 255, 0), 2)
    cv2.putText(image, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return image


def resolve_reference(config_path: Path, config: dict[str, Any]) -> Path:
    path = Path(str(config["reference_file"]))
    if path.is_absolute():
        return path
    # Config lives in <project>/config; reference paths are project-relative.
    return config_path.parent.parent / path


def make_source(args: argparse.Namespace, config: dict[str, Any]):
    if args.source == "videohub":
        return VideoHubSource(
            args.iface, args.camera_timeout,
            int(config["camera_width"]), int(config["camera_height"]),
        )
    if args.source == "zmq":
        return ZmqSource(args.camera_host, args.camera_port, args.camera_timeout)
    if args.image is None:
        raise ValueError("--source image requires --image PATH")
    return ImageSource(args.image)


def save_reference(args: argparse.Namespace, config: dict[str, Any], reference_path: Path) -> int:
    if reference_path.exists() and not args.overwrite:
        raise RuntimeError(
            f"start reference already exists: {reference_path}; "
            "add --overwrite only after placing the robot at the known-correct start"
        )
    source = make_source(args, config)
    try:
        transform, quality, last_pose = collect_pose(
            source, config, int(config["capture_samples"]), args.capture_timeout
        )
    finally:
        source.close()
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dictionary": config["dictionary"],
        "marker_id": int(config["marker_id"]),
        "marker_size_m": float(config["marker_size_m"]),
        "camera_from_marker": transform.tolist(),
        "quality": quality,
    }
    reference_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    debug_path = reference_path.with_suffix(".jpg")
    if not cv2.imwrite(str(debug_path), annotate(last_pose, "TASK1 START REFERENCE")):
        raise RuntimeError(f"failed to save {debug_path}")
    print(f"[OK] 起点参考已保存：{reference_path}")
    print(f"[OK] 检测图已保存：{debug_path}")
    print(
        f"距离标记约 {transform[2, 3]:.2f}m；"
        f"采样离散 {quality['translation_spread_m']:.3f}m；"
        f"重投影误差 {quality['median_reprojection_error_px']:.2f}px"
    )
    print("从现在起不要移动参考牌。")
    return 0


def load_reference(reference_path: Path, config: dict[str, Any]) -> np.ndarray:
    if not reference_path.exists():
        raise RuntimeError(f"start reference does not exist: {reference_path}")
    data = json.loads(reference_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise RuntimeError("unsupported start reference schema")
    if data.get("dictionary") != config["dictionary"] or int(data.get("marker_id", -1)) != int(config["marker_id"]):
        raise RuntimeError("reference marker does not match alignment config")
    transform = np.asarray(data.get("camera_from_marker"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError("invalid camera_from_marker in start reference")
    return transform


def measure_once(args: argparse.Namespace, config: dict[str, Any], reference: np.ndarray) -> tuple[AlignmentError, MarkerPose]:
    source = make_source(args, config)
    try:
        current, _, last_pose = collect_pose(
            source, config, int(config["guide_samples"]), args.capture_timeout
        )
    finally:
        source.close()
    return alignment_error(reference, current), last_pose


def report_error(error: AlignmentError, config: dict[str, Any]) -> bool:
    ok = is_aligned(error, config)
    status = "OK" if ok else "ADJUST"
    print(
        f"[{status}] 当前相对正确起点：前方 {error.forward_m:+.3f}m，"
        f"左侧 {error.left_m:+.3f}m，左转 {error.yaw_left_deg:+.2f}°"
    )
    print("；".join(guidance(error, config)))
    return ok


def check_alignment(args: argparse.Namespace, config: dict[str, Any], reference_path: Path) -> int:
    reference = load_reference(reference_path, config)
    error, last_pose = measure_once(args, config, reference)
    ok = report_error(error, config)
    debug_path = reference_path.with_name("task1_start_current.jpg")
    cv2.imwrite(str(debug_path), annotate(last_pose, "ALIGNED" if ok else "ADJUST START"))
    return 0 if ok else 3


def guide_alignment(args: argparse.Namespace, config: dict[str, Any], reference_path: Path) -> int:
    reference = load_reference(reference_path, config)
    stable = 0
    required = int(config["stable_cycles"])
    print("只读引导已启动：请用原厂遥控器低速调整；程序不会发送运动命令。Ctrl+C退出。")
    source = make_source(args, config)
    try:
        while True:
            current, _, _ = collect_pose(
                source, config, int(config["guide_samples"]), args.capture_timeout
            )
            error = alignment_error(reference, current)
            ok = report_error(error, config)
            stable = stable + 1 if ok else 0
            if stable >= required:
                print(f"[OK] 连续 {required} 次合格，起点校准完成。")
                return 0
            time.sleep(0.3)
    finally:
        source.close()


def generate_marker(args: argparse.Namespace, config: dict[str, Any]) -> int:
    dictionary = aruco_dictionary(str(config["dictionary"]))
    marker = cv2.aruco.generateImageMarker(
        dictionary, int(config["marker_id"]), args.marker_pixels
    )
    margin = max(40, args.marker_pixels // 8)
    canvas = np.full((args.marker_pixels + 2 * margin, args.marker_pixels + 2 * margin), 255, np.uint8)
    canvas[margin:-margin, margin:-margin] = marker
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise RuntimeError(f"failed to save marker: {args.output}")
    print(f"[OK] ArUco 标记已生成：{args.output}")
    print(
        f"打印后测量黑色外框边长，并将 marker_size_m 改为实际米数；"
        f"当前配置是 {float(config['marker_size_m']):.3f}m。"
    )
    return 0


def save_snapshot(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Save one raw head-camera frame without requiring an ArUco detection."""
    source = make_source(args, config)
    try:
        frame = source.read()
    finally:
        source.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), frame):
        raise RuntimeError(f"failed to save camera snapshot: {args.output}")
    print(f"[OK] 头部相机截图已保存：{args.output}")
    print(f"[OK] 图像尺寸：{frame.shape[1]}x{frame.shape[0]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", choices=("videohub", "zmq", "image"), default="videohub")
    parser.add_argument("--iface", default="enp8s0")
    parser.add_argument("--camera-host", default="192.168.123.222")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--camera-timeout", type=float, default=3.0)
    parser.add_argument("--capture-timeout", type=float, default=20.0)
    parser.add_argument("--image", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    marker = sub.add_parser("generate-marker", help="generate printable ArUco PNG")
    marker.add_argument("--output", type=Path, required=True)
    marker.add_argument("--marker-pixels", type=int, default=1600)
    snapshot = sub.add_parser("snapshot", help="save one raw head-camera frame")
    snapshot.add_argument("--output", type=Path, required=True)
    capture = sub.add_parser("capture-reference", help="save the known-correct start pose")
    capture.add_argument(
        "--overwrite", action="store_true",
        help="replace an existing reference after physically re-establishing the correct start",
    )
    sub.add_parser("check", help="one-shot start alignment check")
    sub.add_parser("guide", help="continuous read-only manual alignment guide")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config)
        reference_path = resolve_reference(args.config.resolve(), config)
        if args.command == "generate-marker":
            return generate_marker(args, config)
        if args.command == "snapshot":
            return save_snapshot(args, config)
        if args.command == "capture-reference":
            return save_reference(args, config, reference_path)
        if args.command == "check":
            return check_alignment(args, config, reference_path)
        if args.command == "guide":
            return guide_alignment(args, config, reference_path)
        raise ValueError(f"unknown command: {args.command}")
    except KeyboardInterrupt:
        print("\n已停止视觉引导。")
        return 130
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

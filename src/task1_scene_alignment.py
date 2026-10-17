#!/usr/bin/env python3
"""Read-only Task 1 start alignment against a saved natural-scene image."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from task1_start_alignment import load_config, make_source


@dataclass(frozen=True)
class SceneError:
    image_dx_px: float
    image_dy_px: float
    image_rotation_deg: float
    local_scale: float
    inliers: int
    inlier_ratio: float
    corner_rms_px: float


def enhanced_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def make_feature_detector():
    """Use SIFT when available and a compact ORB fallback on older Orin images."""
    if hasattr(cv2, "SIFT_create"):
        return (
            cv2.SIFT_create(nfeatures=1800, contrastThreshold=0.02, edgeThreshold=12),
            cv2.NORM_L2,
            0.72,
        )
    return (
        cv2.ORB_create(nfeatures=2600, scaleFactor=1.15, nlevels=10, fastThreshold=10),
        cv2.NORM_HAMMING,
        0.78,
    )


def estimate_scene_error(
    reference: np.ndarray,
    current: np.ndarray,
    ignore_left_fraction: float = 0.50,
) -> tuple[SceneError, np.ndarray]:
    if reference.shape != current.shape:
        raise RuntimeError(f"reference/current size mismatch: {reference.shape} vs {current.shape}")
    if not (0.0 <= ignore_left_fraction <= 0.70):
        raise ValueError("ignore_left_fraction must be between 0.0 and 0.70")
    detector, norm_type, ratio_threshold = make_feature_detector()
    height, width = reference.shape[:2]
    feature_mask = np.full((height, width), 255, dtype=np.uint8)
    ignored_width = round(width * ignore_left_fraction)
    feature_mask[:, :ignored_width] = 0
    ref_keypoints, ref_descriptors = detector.detectAndCompute(enhanced_gray(reference), feature_mask)
    cur_keypoints, cur_descriptors = detector.detectAndCompute(enhanced_gray(current), feature_mask)
    if ref_descriptors is None or cur_descriptors is None:
        raise RuntimeError("场景纹理不足，无法提取视觉特征")
    matches = cv2.BFMatcher(norm_type).knnMatch(cur_descriptors, ref_descriptors, k=2)
    good = [
        first
        for pair in matches
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < ratio_threshold * second.distance
    ]
    if len(good) < 20:
        raise RuntimeError(f"场景匹配点不足：{len(good)}/20；保持光照并清除镜头前遮挡")
    current_points = np.float32([cur_keypoints[item.queryIdx].pt for item in good])
    reference_points = np.float32([ref_keypoints[item.trainIdx].pt for item in good])
    homography, mask = cv2.findHomography(current_points, reference_points, cv2.RANSAC, 3.0)
    if homography is None or mask is None:
        raise RuntimeError("无法计算当前画面到参考画面的变换")
    inlier_mask = mask.reshape(-1).astype(bool)
    inliers = int(np.count_nonzero(inlier_mask))
    ratio = inliers / len(good)
    if inliers < 15 or ratio < 0.30:
        raise RuntimeError(f"场景匹配不可靠：内点 {inliers}，比例 {ratio:.2f}")

    cx, cy = width / 2.0, height / 2.0
    probe = np.float32([[[cx, cy], [cx + 100.0, cy]]])
    mapped = cv2.perspectiveTransform(probe, homography)[0]
    delta = mapped[1] - mapped[0]
    angle = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    scale = float(np.linalg.norm(delta) / 100.0)
    dx, dy = mapped[0] - np.array([cx, cy], dtype=np.float32)

    corners = np.float32([[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]])
    warped_corners = cv2.perspectiveTransform(corners, homography)[0]
    corner_rms = float(np.sqrt(np.mean(np.sum((warped_corners - corners[0]) ** 2, axis=1))))

    debug = current.copy()
    if ignored_width:
        overlay = debug.copy()
        overlay[:, :ignored_width] = (0, 0, 0)
        debug = cv2.addWeighted(debug, 0.35, overlay, 0.65, 0.0)
        cv2.line(debug, (ignored_width, 0), (ignored_width, height - 1), (0, 165, 255), 2)
    for item, keep in zip(good, inlier_mask):
        if keep:
            x, y = cur_keypoints[item.queryIdx].pt
            cv2.circle(debug, (round(x), round(y)), 2, (0, 255, 0), -1)
    cv2.putText(
        debug, f"dx={dx:+.1f} dy={dy:+.1f} yaw={angle:+.2f}",
        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
    )
    return SceneError(float(dx), float(dy), angle, scale, inliers, ratio, corner_rms), debug


def is_aligned(error: SceneError, pixel_tolerance: float, yaw_tolerance: float) -> bool:
    return (
        abs(error.image_dx_px) <= pixel_tolerance
        and abs(error.image_dy_px) <= pixel_tolerance
        and abs(error.image_rotation_deg) <= yaw_tolerance
    )


def guidance(error: SceneError, pixel_tolerance: float, yaw_tolerance: float) -> list[str]:
    result: list[str] = []
    if error.image_rotation_deg > yaw_tolerance:
        result.append(f"先向左转约 {abs(error.image_rotation_deg):.1f}°")
    elif error.image_rotation_deg < -yaw_tolerance:
        result.append(f"先向右转约 {abs(error.image_rotation_deg):.1f}°")
    if abs(error.image_rotation_deg) <= max(3.0, yaw_tolerance):
        if error.image_dx_px > pixel_tolerance:
            result.append("向左小幅横移")
        elif error.image_dx_px < -pixel_tolerance:
            result.append("向右小幅横移")
        if error.image_dy_px > pixel_tolerance:
            result.append("向前小幅移动")
        elif error.image_dy_px < -pixel_tolerance:
            result.append("向后小幅移动")
    return result or ["视觉起点已对齐"]


def choose_motion(
    error: SceneError,
    pixel_tolerance: float,
    yaw_tolerance: float,
    linear_speed: float,
    lateral_speed: float,
    yaw_speed: float,
) -> tuple[float, float, float, str]:
    """Choose one low-speed body-axis correction; yaw always has priority."""
    if error.image_rotation_deg > yaw_tolerance:
        return 0.0, 0.0, yaw_speed, "左转微调"
    if error.image_rotation_deg < -yaw_tolerance:
        return 0.0, 0.0, -yaw_speed, "右转微调"
    horizontal = abs(error.image_dx_px) / pixel_tolerance
    vertical = abs(error.image_dy_px) / pixel_tolerance
    if horizontal >= vertical and horizontal > 1.0:
        if error.image_dx_px > 0:
            return 0.0, lateral_speed, 0.0, "向左横移微调"
        return 0.0, -lateral_speed, 0.0, "向右横移微调"
    if vertical > 1.0:
        if error.image_dy_px > 0:
            return linear_speed, 0.0, 0.0, "向前微调"
        return -linear_speed, 0.0, 0.0, "向后微调"
    return 0.0, 0.0, 0.0, "无需动作"


def execute_odometry_bounded_motion(
    backend,
    vx: float,
    vy: float,
    wz: float,
    *,
    yaw_target_rad: float,
    translation_target_m: float,
    max_action_time: float,
) -> tuple[float, str]:
    """Repeat an expiring velocity command until odometry observes a small motion."""
    from task1_route import wait_for_pose, wrap_angle

    before = wait_for_pose(backend, 3.0)
    deadline = time.monotonic() + max_action_time
    moved = 0.0
    mode = "转角" if abs(wz) > 0.0 else "位移"
    try:
        while time.monotonic() < deadline:
            backend.set_velocity(vx, vy, wz)
            time.sleep(0.10)
            after = backend.pose()
            if abs(wz) > 0.0:
                moved = abs(wrap_angle(after.yaw - before.yaw))
                if moved >= yaw_target_rad:
                    break
            else:
                moved = math.hypot(after.x - before.x, after.y - before.y)
                if moved >= translation_target_m:
                    break
    finally:
        backend.stop()
    minimum = math.radians(0.25) if abs(wz) > 0.0 else 0.003
    if moved < minimum:
        unit = "°" if abs(wz) > 0.0 else "m"
        shown = math.degrees(moved) if abs(wz) > 0.0 else moved
        raise RuntimeError(f"运控命令未产生有效{mode}（仅 {shown:.3f}{unit}），已停车")
    return moved, mode


def read_frame(args: argparse.Namespace, config: dict) -> np.ndarray:
    source = make_source(args, config)
    try:
        return source.read()
    finally:
        source.close()


def save_reference(args: argparse.Namespace, config: dict) -> int:
    if args.reference.exists() and not args.overwrite:
        raise RuntimeError(f"场景参考已存在：{args.reference}；确认新起点后使用 --overwrite")
    image = read_frame(args, config)
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.reference), image):
        raise RuntimeError(f"无法保存场景参考：{args.reference}")
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "width": image.shape[1],
        "height": image.shape[0],
        "note": "Natural-scene reference; fixed venue and approximately unchanged lighting required.",
    }
    args.reference.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] 当前头部画面已保存为起点参考：{args.reference}")
    print("从现在起不要移动场地黄色胶带、柜体或起点附近固定物体。")
    return 0


def load_reference(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"场景参考不存在或无法读取：{path}")
    return image


def report(error: SceneError, args: argparse.Namespace) -> bool:
    ok = is_aligned(error, args.pixel_tolerance, args.yaw_tolerance)
    print(
        f"[{'OK' if ok else 'ADJUST'}] 图像偏差：水平 {error.image_dx_px:+.1f}px，"
        f"纵向 {error.image_dy_px:+.1f}px，转角 {error.image_rotation_deg:+.2f}°；"
        f"内点 {error.inliers}，匹配率 {error.inlier_ratio:.2f}"
    )
    print("；".join(guidance(error, args.pixel_tolerance, args.yaw_tolerance)))
    return ok


def check(args: argparse.Namespace, config: dict) -> int:
    reference = load_reference(args.reference)
    current = read_frame(args, config)
    error, debug = estimate_scene_error(reference, current, args.ignore_left_fraction)
    ok = report(error, args)
    cv2.imwrite(str(args.debug_output), debug)
    return 0 if ok else 3


def measure(args: argparse.Namespace, config: dict) -> int:
    """Measure the start-frame offset without requiring physical alignment."""
    reference = load_reference(args.reference)
    current = read_frame(args, config)
    error, debug = estimate_scene_error(reference, current, args.ignore_left_fraction)
    report(error, args)
    cv2.imwrite(str(args.debug_output), debug)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "image_dx_px": error.image_dx_px,
                "image_dy_px": error.image_dy_px,
                "yaw_compensation_deg": error.image_rotation_deg,
                "inliers": error.inliers,
                "inlier_ratio": error.inlier_ratio,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] 起点视觉偏差已保存：{args.output}")
    return 0


def guide(args: argparse.Namespace, config: dict) -> int:
    reference = load_reference(args.reference)
    stable = 0
    print("自然场景对齐已启动：只读，不控制机器人；请用遥控器低速调整。Ctrl+C退出。")
    source = make_source(args, config)
    try:
        while True:
            current = source.read()
            error, debug = estimate_scene_error(reference, current, args.ignore_left_fraction)
            ok = report(error, args)
            cv2.imwrite(str(args.debug_output), debug)
            stable = stable + 1 if ok else 0
            if stable >= args.stable_cycles:
                print(f"[OK] 连续 {args.stable_cycles} 次合格，视觉起点校准完成。")
                return 0
            time.sleep(0.35)
    finally:
        source.close()


def auto_align(args: argparse.Namespace, config: dict) -> int:
    raise RuntimeError(
        "auto-align 已停用：任务1采用允许范围检查，并在开始行走后进行航向补偿"
    )
    if not args.real or args.confirm != "AUTO_ALIGN_START":
        raise RuntimeError("真机自动校准需要同时添加 --real --confirm AUTO_ALIGN_START")
    if not (0.50 <= args.pulse_duration <= 1.00):
        raise RuntimeError("--pulse-duration 必须在 0.50 至 1.00 秒之间")
    if not (0.08 <= args.linear_speed <= 0.15):
        raise RuntimeError("--linear-speed 必须在 0.08 至 0.15m/s 之间")
    if not (0.08 <= args.lateral_speed <= 0.12):
        raise RuntimeError("--lateral-speed 必须在 0.08 至 0.12m/s 之间")
    if not (0.15 <= args.yaw_speed <= 0.30):
        raise RuntimeError("--yaw-speed 必须在 0.15 至 0.30rad/s 之间")
    if not (1 <= args.max_pulses <= 6):
        raise RuntimeError("--max-pulses 必须在 1 至 6 之间")
    if not (2.0 <= args.max_yaw_step_deg <= 6.0):
        raise RuntimeError("--max-yaw-step-deg 必须在 2.0 至 6.0度之间")
    if not (0.05 <= args.translation_step <= 0.12):
        raise RuntimeError("--translation-step 必须在 0.05 至 0.12m 之间")
    if not (1.0 <= args.max_action_time <= 3.0):
        raise RuntimeError("--max-action-time 必须在 1.0 至 3.0秒之间")
    from task1_route import UnitreeBackend

    reference = load_reference(args.reference)
    source = make_source(args, config)
    backend = None
    try:
        # The camera source has already initialized the DDS domain.
        backend = UnitreeBackend(
            args.iface,
            args.state_topic,
            command_duration=args.pulse_duration,
            enable_motion=True,
            loco_service=args.loco_service,
            initialize_channel=False,
            velocity_no_reply=True,
        )
        print(
            f"起点视觉粗对齐准备：FSM={backend.fsm_id}，最多走 {args.max_pulses} 个正常小步；"
            "请确保机器人四周至少留出 0.5m，并握住遥控器准备急停。"
        )
        for remaining in range(5, 0, -1):
            print(remaining, flush=True)
            time.sleep(1.0)

        stable = 0
        stagnant_actions = 0
        previous_metric: float | None = None
        for pulse_index in range(args.max_pulses + 1):
            current = source.read()
            error, debug = estimate_scene_error(reference, current, args.ignore_left_fraction)
            ok = report(error, args)
            cv2.imwrite(str(args.debug_output), debug)
            if abs(error.image_rotation_deg) > 12.0:
                raise RuntimeError(
                    f"起点偏角 {error.image_rotation_deg:+.1f}° 过大；只需人工粗略摆正到 ±12° 内"
                )
            if abs(error.image_dx_px) > 120.0 or abs(error.image_dy_px) > 120.0:
                raise RuntimeError(
                    "起点离参考位置过远；只需人工放到画面水平/纵向偏差 ±120px 内"
                )
            stable = stable + 1 if ok else 0
            if stable >= args.stable_cycles:
                backend.stop()
                print(f"[OK] 连续 {args.stable_cycles} 次合格，自动起点校准完成。")
                return 0

            metric = math.hypot(error.image_dx_px, error.image_dy_px) / args.pixel_tolerance
            metric += abs(error.image_rotation_deg) / args.yaw_tolerance
            if previous_metric is not None:
                if metric > previous_metric * 1.15 + 0.25:
                    backend.stop()
                    raise RuntimeError("上一步后视觉误差反而增大，已停车；请检查起点和行走方向")
                improvement = previous_metric - metric
                stagnant_actions = stagnant_actions + 1 if improvement < max(0.15, previous_metric * 0.03) else 0
                if stagnant_actions >= 2:
                    backend.stop()
                    raise RuntimeError("连续两步后视觉误差没有明显减小，已停车")
            if pulse_index >= args.max_pulses:
                raise RuntimeError(f"达到最大修正步数 {args.max_pulses}，已停车")

            vx, vy, wz, description = choose_motion(
                error,
                args.pixel_tolerance,
                args.yaw_tolerance,
                args.linear_speed,
                args.lateral_speed,
                args.yaw_speed,
            )
            print(
                f"[ALIGN {pulse_index + 1}/{args.max_pulses}] {description}: "
                f"vx={vx:+.3f}, vy={vy:+.3f}, wz={wz:+.3f}"
            )
            previous_metric = metric
            yaw_target = math.radians(
                min(args.max_yaw_step_deg, max(2.0, abs(error.image_rotation_deg) - args.yaw_tolerance * 0.7))
            )
            moved, mode = execute_odometry_bounded_motion(
                backend,
                vx,
                vy,
                wz,
                yaw_target_rad=yaw_target,
                translation_target_m=args.translation_step,
                max_action_time=args.max_action_time,
            )
            shown = math.degrees(moved) if mode == "转角" else moved
            unit = "°" if mode == "转角" else "m"
            print(f"[ALIGN] 里程计实测{mode}：{shown:.3f}{unit}；重新拍照确认")
            time.sleep(args.settle_time)
        raise RuntimeError("起点粗对齐未完成")
    finally:
        if backend is not None:
            backend.stop()
        source.close()


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
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--debug-output", type=Path, required=True)
    parser.add_argument("--pixel-tolerance", type=float, default=8.0)
    parser.add_argument("--yaw-tolerance", type=float, default=1.2)
    parser.add_argument("--stable-cycles", type=int, default=3)
    parser.add_argument(
        "--ignore-left-fraction", type=float, default=0.50,
        help="ignore this fraction of the image from the left (competition cabinets may change)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture-reference")
    capture.add_argument("--overwrite", action="store_true")
    sub.add_parser("check")
    sub.add_parser("guide")
    measurement = sub.add_parser("measure")
    measurement.add_argument("--output", type=Path, required=True)
    automatic = sub.add_parser("auto-align")
    automatic.add_argument("--real", action="store_true")
    automatic.add_argument("--confirm")
    automatic.add_argument("--state-topic", default="auto")
    automatic.add_argument("--loco-service", default="sport")
    automatic.add_argument("--pulse-duration", type=float, default=0.80)
    automatic.add_argument("--settle-time", type=float, default=1.20)
    automatic.add_argument("--linear-speed", type=float, default=0.12)
    automatic.add_argument("--lateral-speed", type=float, default=0.10)
    automatic.add_argument("--yaw-speed", type=float, default=0.22)
    automatic.add_argument("--max-yaw-step-deg", type=float, default=5.0)
    automatic.add_argument("--translation-step", type=float, default=0.08)
    automatic.add_argument("--max-action-time", type=float, default=2.0)
    automatic.add_argument("--max-pulses", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config)
        if args.command == "capture-reference":
            return save_reference(args, config)
        if args.command == "check":
            return check(args, config)
        if args.command == "guide":
            return guide(args, config)
        if args.command == "measure":
            return measure(args, config)
        if args.command == "auto-align":
            return auto_align(args, config)
        raise ValueError(f"unknown command: {args.command}")
    except KeyboardInterrupt:
        print("\n已停止视觉引导。")
        return 130
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

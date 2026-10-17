#!/usr/bin/env python3
"""Validate a task-2 LeRobot v2.1 dataset before ACT training.

The checks deliberately cover both the on-disk contract and the installed
LeRobot decoder.  This catches incomplete rsync uploads, wrong camera keys,
dimension drift, NaN joint values, timestamp gaps, and broken video files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


EXPECTED_FEATURES = {
    "observation.images.ego_view": ("video", [480, 640, 3]),
    "observation.images.right_wrist": ("video", [480, 640, 3]),
    "observation.state": ("float32", [30]),
    "action": ("float32", [8]),
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument(
        "--decode-samples",
        type=int,
        default=3,
        help="number of evenly spaced frames whose two videos are decoded",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        fail(f"missing {info_path}")

    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v2.1":
        fail(f"expected LeRobot v2.1, got {info.get('codebase_version')!r}")
    if int(info.get("fps", 0)) != 20:
        fail(f"expected 20 fps, got {info.get('fps')!r}")
    if int(info.get("total_episodes", 0)) <= 0:
        fail("dataset has no episodes")
    if int(info.get("total_frames", 0)) <= 0:
        fail("dataset has no frames")

    features = info.get("features", {})
    for key, (dtype, shape) in EXPECTED_FEATURES.items():
        feature = features.get(key)
        if feature is None:
            fail(f"missing feature {key}")
        if feature.get("dtype") != dtype or list(feature.get("shape", [])) != shape:
            fail(
                f"bad {key}: dtype={feature.get('dtype')!r}, "
                f"shape={feature.get('shape')!r}; expected {dtype}, {shape}"
            )
    if "observation.images.left_wrist" in features:
        fail("unexpected left wrist stream in right-only task-2 dataset")

    parquet_files = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    video_files = sorted(root.glob("videos/chunk-*/*/episode_*.mp4"))
    episodes = int(info["total_episodes"])
    if len(parquet_files) != episodes:
        fail(f"expected {episodes} parquet files, found {len(parquet_files)}")
    if len(video_files) != episodes * 2:
        fail(f"expected {episodes * 2} videos, found {len(video_files)}")
    if any(path.stat().st_size == 0 for path in parquet_files + video_files):
        fail("found an empty parquet or video file")

    # Import only after inexpensive structural checks so setup errors are clear.
    cache_root = Path(
        os.environ.get("TASK2_CACHE_ROOT", str(root.parent / ".task2_cache"))
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("tmp/tmp_dataset", root=root, video_backend="pyav")
    if len(dataset) != int(info["total_frames"]):
        fail(f"loader length {len(dataset)} != metadata {info['total_frames']}")

    table = dataset.hf_dataset
    state = np.asarray(table["observation.state"], dtype=np.float32)
    action = np.asarray(table["action"], dtype=np.float32)
    if state.shape != (len(dataset), 30):
        fail(f"loaded state shape is {state.shape}, expected {(len(dataset), 30)}")
    if action.shape != (len(dataset), 8):
        fail(f"loaded action shape is {action.shape}, expected {(len(dataset), 8)}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        fail("state/action contains NaN or infinity")

    episode_index = np.asarray(table["episode_index"], dtype=np.int64)
    frame_index = np.asarray(table["frame_index"], dtype=np.int64)
    timestamps = np.asarray(table["timestamp"], dtype=np.float64)
    minimum_frames = math.ceil(args.min_duration * int(info["fps"]))
    for episode in range(episodes):
        mask = episode_index == episode
        count = int(mask.sum())
        if count < minimum_frames:
            fail(
                f"episode {episode} has {count} frames; minimum is {minimum_frames} "
                f"({args.min_duration:.1f} s)"
            )
        expected_frames = np.arange(count, dtype=np.int64)
        if not np.array_equal(frame_index[mask], expected_frames):
            fail(f"episode {episode} frame_index is not contiguous from zero")
        delta = np.diff(timestamps[mask])
        if delta.size and not np.allclose(delta, 1.0 / int(info["fps"]), atol=1e-4):
            fail(f"episode {episode} timestamps are not uniformly spaced at 20 Hz")

    sample_count = min(max(args.decode_samples, 1), len(dataset))
    sample_indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=int)
    for index in np.unique(sample_indices):
        sample = dataset[int(index)]
        for key in (
            "observation.images.ego_view",
            "observation.images.right_wrist",
        ):
            image = sample[key]
            if tuple(image.shape) != (3, 480, 640):
                fail(f"decoded {key} frame {index} has shape {tuple(image.shape)}")
            if not bool(image.isfinite().all()):
                fail(f"decoded {key} frame {index} contains non-finite pixels")

    print(
        "TASK2_ACT_DATASET_OK "
        f"root={root} episodes={episodes} frames={len(dataset)} fps={info['fps']} "
        "state=30 action=8 cameras=ego_view,right_wrist"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"TASK2_ACT_DATASET_INVALID: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

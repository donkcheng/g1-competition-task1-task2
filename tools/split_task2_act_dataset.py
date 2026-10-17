#!/usr/bin/env python3
"""Split the mixed task-2 capture into three self-contained LeRobot v2.1 datasets.

The output episode indices are intentionally rebuilt from zero because LeRobot
expects contiguous indices.  ``source_episode_map.json`` preserves the original
capture index for review and traceability.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.common.datasets.compute_stats import compute_episode_stats


CAMERAS = ("observation.images.ego_view", "observation.images.right_wrist")
SPLITS = {
    "button": {
        "source_ids": list(range(3, 20)),
        "task": "Press the button and hold it for at least two seconds.",
    },
    "knob": {
        "source_ids": list(range(20, 33)),
        "task": "Rotate the knob by at least 15 degrees.",
    },
    "switch": {
        "source_ids": [*range(33, 40), 43, 44],
        "task": "Toggle the switch by at least 15 degrees and hold it for at least two seconds.",
    },
}


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=4, default=json_default) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=json_default) + "\n")


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise KeyError(f"missing parquet column: {name}")
    return table.set_column(column_index, name, pa.array(values, type=table.schema.field(name).type))


def numeric_episode_data(table: pa.Table, info: dict) -> dict[str, np.ndarray]:
    result = {}
    for name in table.column_names:
        feature = info["features"][name]
        if feature["dtype"] in ("video", "image", "string"):
            continue
        column = table[name]
        if pa.types.is_fixed_size_list(column.type):
            result[name] = np.asarray(column.to_pylist())
        else:
            result[name] = np.asarray(column.to_numpy())
    return result


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def build_one(source: Path, destination: Path, skill: str, source_ids: list[int], task: str) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    source_info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    info = copy.deepcopy(source_info)
    info["total_episodes"] = len(source_ids)
    info["total_frames"] = 0
    info["total_tasks"] = 1
    info["total_videos"] = len(source_ids) * len(CAMERAS)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(source_ids)}"}
    info["discarded_episode_indices"] = []
    info.setdefault("script_config", {})["skill"] = skill
    info["script_config"]["derived_from"] = str(source.resolve())

    episodes = []
    episode_stats = []
    mapping = []
    global_index = 0

    for new_index, source_index in enumerate(source_ids):
        source_parquet = source / "data/chunk-000" / f"episode_{source_index:06d}.parquet"
        if not source_parquet.is_file():
            raise FileNotFoundError(source_parquet)
        table = pq.read_table(source_parquet)
        length = table.num_rows
        table = replace_column(table, "episode_index", np.full(length, new_index, dtype=np.int64))
        table = replace_column(table, "index", np.arange(global_index, global_index + length, dtype=np.int64))
        table = replace_column(table, "task_index", np.zeros(length, dtype=np.int64))

        destination_parquet = destination / "data/chunk-000" / f"episode_{new_index:06d}.parquet"
        destination_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination_parquet)

        for camera in CAMERAS:
            source_video = source / "videos/chunk-000" / camera / f"episode_{source_index:06d}.mp4"
            destination_video = destination / "videos/chunk-000" / camera / f"episode_{new_index:06d}.mp4"
            if not source_video.is_file():
                raise FileNotFoundError(source_video)
            hardlink(source_video, destination_video)

        stats = compute_episode_stats(numeric_episode_data(table, info), info["features"])
        episodes.append({"episode_index": new_index, "tasks": [task], "length": length})
        episode_stats.append({"episode_index": new_index, "stats": stats})
        mapping.append({"episode_index": new_index, "source_episode_index": source_index})
        info["total_frames"] += length
        global_index += length

    write_json(destination / "meta/info.json", info)
    write_json(destination / "meta/modality.json", json.loads((source / "meta/modality.json").read_text()))
    write_jsonl(destination / "meta/tasks.jsonl", [{"task_index": 0, "task": task}])
    write_jsonl(destination / "meta/episodes.jsonl", episodes)
    write_jsonl(destination / "meta/episodes_stats.jsonl", episode_stats)
    write_json(
        destination / "source_episode_map.json",
        {"skill": skill, "source_dataset": str(source.resolve()), "episodes": mapping},
    )
    print(f"BUILT {skill}: episodes={len(source_ids)} frames={info['total_frames']} -> {destination}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    for skill, config in SPLITS.items():
        build_one(
            args.source,
            args.output_root / f"task2_{skill}_act",
            skill,
            config["source_ids"],
            config["task"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

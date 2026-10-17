from __future__ import annotations

import os
import re
import shutil
import socket
from pathlib import Path
from typing import Iterable


TRUTHY = {"1", "true", "yes", "y", "on"}


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_no}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"{path}:{line_no}: invalid key {key!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def env_bool(values: dict[str, str], key: str, default: bool = False) -> bool:
    raw = values.get(key, "1" if default else "0").strip().lower()
    if raw in TRUTHY:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{key} must be 0/1 or true/false, got {raw!r}")


def free_gib(path: Path) -> float:
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    return shutil.disk_usage(existing).free / (1024**3)


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def human_bytes(count: int) -> str:
    size = float(count)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or suffix == "TiB":
            return f"{size:.1f} {suffix}"
        size /= 1024
    raise AssertionError("unreachable")


def tcp_open(host: str, port: int, timeout: float = 1.0) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "connected"
    except OSError as exc:
        return False, str(exc)


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def all_exist(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from common import dir_size, free_gib, human_bytes


ROOT = Path(__file__).resolve().parent.parent


def command(command: list[str]) -> int:
    print("\n$ " + " ".join(command))
    return subprocess.run(command, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only preflight for tomorrow's G1 work")
    parser.add_argument("--network", action="store_true", help="probe Orin camera TCP/ping")
    args = parser.parse_args()
    failures = 0
    print("=== Host ===")
    print(f"OS: {platform.platform()}")
    print(f"arch: {platform.machine()}")
    free = free_gib(ROOT)
    print(f"disk free: {free:.1f} GiB")
    print(f"project size: {human_bytes(dir_size(ROOT))}")
    if free < 30:
        print("[FAIL] less than 30 GiB free")
        failures += 1
    else:
        print("[OK] disk guard")
    for tool in ("python3", "tmux", "git", "ping"):
        if shutil.which(tool):
            print(f"[OK] {tool}: {shutil.which(tool)}")
        else:
            print(f"[FAIL] missing command: {tool}")
            failures += 1

    env_path = ROOT / ".env"
    if not env_path.exists():
        print("\n[FAIL] .env missing")
        print("Run: cp config/task2.env.example .env && edit .env")
        failures += 1
    else:
        task2 = [str(ROOT / "run_task2.sh"), "check", "--real"]
        if args.network:
            task2.append("--network")
        failures += int(command(task2) != 0)

    iface = "eth0"
    task1_python = __import__("os").environ.get("TASK1_PYTHON", "").strip()
    if env_path.exists():
        try:
            from common import load_env

            env = load_env(env_path)
            iface = env.get("ROBOT_IFACE", iface)
            if not task1_python and env.get("GROOT_WBC_ROOT"):
                candidate = Path(env["GROOT_WBC_ROOT"]).expanduser() / ".venv_teleop/bin/python"
                if candidate.is_file():
                    task1_python = str(candidate)
        except ValueError as exc:
            print(f"[FAIL] cannot parse .env: {exc}")
            failures += 1
    if not task1_python:
        task1_python = "python3"
    failures += int(
        command(
            [
                task1_python,
                str(ROOT / "src/task1_route.py"),
                "--config",
                str(ROOT / "config/task1.json"),
                "preflight",
                "--iface",
                iface,
            ]
        )
        != 0
    )

    print("\n=== Result ===")
    if failures:
        print(f"NOT READY: {failures} check group(s) failed. No robot command was sent.")
        return 1
    print("READY FOR READ-ONLY CONNECTION CHECK. No robot command was sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

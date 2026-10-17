#!/usr/bin/env python3
"""Disk-aware tmux launcher around NVIDIA's official G1 PICO stack."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from common import command_exists, dir_size, env_bool, free_gib, human_bytes, load_env, tcp_open


SESSION = "g1_task2_collection"
SKILL_PROMPTS = {
    "button": "press the target button and hold it for at least 2 seconds",
    "knob": "rotate the target knob at least 15 degrees and hold it for 2 seconds",
    "switch": "toggle the target switch at least 15 degrees and hold it for 2 seconds",
}
SUPPORTED_HAND_NAMES = {"dex3", "dex3-1", "threefinger", "three_finger", "three-finger"}


@dataclass(frozen=True)
class Settings:
    repo: Path
    data_root: Path
    robot_iface: str
    camera_host: str
    camera_port: int
    hand_type: str
    camera_type: str
    min_free_gib: float
    frequency: int
    wrist_cameras: bool
    right_wrist_camera: bool
    camera_viewer: bool
    text_to_speech: bool
    dataset_name: str
    checkpoint: str
    obs_config: str
    kp_scale: str
    kd_scale: str
    tensorrt_root: Path | None


def settings_from(path: Path) -> Settings:
    if not path.exists():
        raise ValueError(f"configuration not found: {path}; run: cp config/task2.env.example .env")
    env = load_env(path)
    required = ["GROOT_WBC_ROOT", "DATA_ROOT", "ROBOT_IFACE", "CAMERA_HOST", "HAND_TYPE"]
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise ValueError("missing configuration: " + ", ".join(missing))
    trt = env.get("TENSORRT_ROOT", "").strip()
    return Settings(
        repo=Path(env["GROOT_WBC_ROOT"]).expanduser().resolve(),
        data_root=Path(env["DATA_ROOT"]).expanduser().resolve(),
        robot_iface=env["ROBOT_IFACE"],
        camera_host=env["CAMERA_HOST"],
        camera_port=int(env.get("CAMERA_PORT", "5555")),
        hand_type=env["HAND_TYPE"].lower(),
        camera_type=env.get("CAMERA_TYPE", "oak").lower(),
        min_free_gib=float(env.get("MIN_FREE_GB", "30")),
        frequency=int(env.get("DATA_FREQUENCY", "50")),
        wrist_cameras=env_bool(env, "RECORD_WRIST_CAMERAS"),
        right_wrist_camera=env_bool(env, "RECORD_RIGHT_WRIST_CAMERA"),
        camera_viewer=env_bool(env, "CAMERA_VIEWER", True),
        text_to_speech=env_bool(env, "TEXT_TO_SPEECH"),
        dataset_name=env.get("DATASET_NAME", "").strip(),
        checkpoint=env.get("DEPLOY_CHECKPOINT", "").strip(),
        obs_config=env.get("DEPLOY_OBS_CONFIG", "").strip(),
        kp_scale=env.get("MOTOR_KP_SCALE", "").strip(),
        kd_scale=env.get("MOTOR_KD_SCALE", "").strip(),
        tensorrt_root=Path(trt).expanduser().resolve() if trt else None,
    )


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, text: str) -> None:
        print(f"[OK]   {text}")

    def fail(self, text: str) -> None:
        self.failures.append(text)
        print(f"[FAIL] {text}")

    def warn(self, text: str) -> None:
        self.warnings.append(text)
        print(f"[WARN] {text}")


def repo_files(settings: Settings, sim: bool) -> list[Path]:
    paths = [
        settings.repo / "gear_sonic/scripts/pico_manager_thread_server.py",
        settings.repo / "gear_sonic/scripts/run_data_exporter.py",
        settings.repo / "gear_sonic/scripts/run_camera_viewer.py",
        settings.repo / "gear_sonic_deploy/deploy.sh",
        settings.repo / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref",
        settings.repo / "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
        settings.repo / ".venv_teleop/bin/python",
        settings.repo / ".venv_data_collection/bin/python",
    ]
    if sim:
        paths.extend(
            [
                settings.repo / "gear_sonic/scripts/run_sim_loop.py",
                settings.repo / ".venv_sim/bin/python",
            ]
        )
    return paths


def run_quiet(command: list[str], timeout: float = 5.0) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def tensorrt_version(root: Path) -> tuple[int, int, int] | None:
    header = root / "include/NvInferVersion.h"
    if not header.exists():
        return None
    text = header.read_text(encoding="utf-8", errors="replace")
    definitions = dict(
        re.findall(r"^#define\s+(\w+)\s+([A-Za-z_]\w*|\d+)\s*(?:/[/\*].*)?$", text, re.MULTILINE)
    )

    def resolve(name: str) -> int | None:
        seen: set[str] = set()
        value = definitions.get(name)
        while value is not None and not value.isdigit():
            if value in seen:
                return None
            seen.add(value)
            value = definitions.get(value)
        return int(value) if value and value.isdigit() else None

    values: list[int] = []
    for key in ("NV_TENSORRT_MAJOR", "NV_TENSORRT_MINOR", "NV_TENSORRT_PATCH"):
        value = resolve(key)
        if value is None:
            return None
        values.append(value)
    return values[0], values[1], values[2]


def check(settings: Settings, real: bool, sim: bool, network: bool) -> Report:
    report = Report()
    print("G1 PICO data collection preflight (no robot commands)\n")
    print(f"repo:      {settings.repo}")
    print(f"data root: {settings.data_root}")
    print(f"mode:      {'real' if real else 'sim' if sim else 'generic'}\n")

    if command_exists("tmux"):
        report.ok("tmux installed")
    else:
        report.fail("tmux missing: sudo apt install tmux")

    if platform.machine() == "x86_64":
        report.ok("workstation architecture x86_64")
    else:
        report.fail(f"this launcher is for the offboard x86_64 workstation, got {platform.machine()}")

    missing = [path for path in repo_files(settings, sim) if not path.exists()]
    if missing:
        report.fail("official repository/environment is incomplete")
        for path in missing:
            print(f"       missing: {path}")
    else:
        report.ok("official scripts and required virtual environments present")

    if settings.checkpoint:
        checkpoint_prefix = settings.repo / "gear_sonic_deploy" / settings.checkpoint
        checkpoint_files = [
            Path(str(checkpoint_prefix) + "_encoder.onnx"),
            Path(str(checkpoint_prefix) + "_decoder.onnx"),
        ]
        if all(path.exists() for path in checkpoint_files):
            report.ok(f"SONIC checkpoint pair: {checkpoint_prefix}_{{encoder,decoder}}.onnx")
        else:
            report.fail(f"SONIC checkpoint pair missing under: {checkpoint_prefix.parent}")
    if settings.obs_config:
        obs = settings.repo / "gear_sonic_deploy" / settings.obs_config
        (report.ok if obs.exists() else report.fail)(f"observation config: {obs}")

    available = free_gib(settings.data_root)
    if available >= settings.min_free_gib:
        report.ok(f"disk free {available:.1f} GiB (minimum {settings.min_free_gib:.1f} GiB)")
    else:
        report.fail(f"disk free only {available:.1f} GiB; need {settings.min_free_gib:.1f} GiB")
    existing_size = dir_size(settings.data_root)
    report.ok(f"current task data size {human_bytes(existing_size)}; no automatic deletion enabled")

    if settings.frequency < 1 or settings.frequency > 100:
        report.fail("DATA_FREQUENCY must be in 1..100 Hz")
    else:
        report.ok(f"data frequency {settings.frequency} Hz")

    if real:
        if settings.robot_iface == "CHANGE_ME":
            report.fail("ROBOT_IFACE is still CHANGE_ME")
        elif (Path("/sys/class/net") / settings.robot_iface).exists():
            report.ok(f"robot wired interface exists: {settings.robot_iface}")
            ip_ok, ip_text = run_quiet(
                ["ip", "-4", "-o", "addr", "show", "dev", settings.robot_iface], timeout=2
            )
            addresses = re.findall(r"\binet\s+([0-9.]+)/", ip_text) if ip_ok else []
            try:
                camera_ip = ipaddress.ip_address(settings.camera_host)
            except ValueError:
                camera_ip = None
            if camera_ip and camera_ip.version == 4:
                same_subnet = any(
                    ipaddress.ip_address(address) in ipaddress.ip_network(f"{camera_ip}/24", strict=False)
                    for address in addresses
                )
                if same_subnet:
                    report.ok(f"wired IPv4 shares camera /24 subnet: {', '.join(addresses)}")
                else:
                    report.fail(
                        f"{settings.robot_iface} has no IPv4 in {camera_ip}/24; current: {addresses or 'none'}"
                    )
            elif addresses:
                report.ok(f"wired IPv4: {', '.join(addresses)}")
            else:
                report.fail(f"{settings.robot_iface} has no IPv4 address")
        else:
            report.fail(f"robot wired interface not found: {settings.robot_iface}")

        if settings.hand_type in SUPPORTED_HAND_NAMES:
            report.ok(f"configured hand is compatible with official 7-joint-per-hand exporter: {settings.hand_type}")
        else:
            report.fail(
                f"HAND_TYPE={settings.hand_type!r}; current official exporter expects G1 29DoF + bilateral Dex3/ThreeFinger"
            )

        if settings.camera_type == "oak":
            report.ok("camera type OAK (officially tested path)")
        else:
            report.warn(f"CAMERA_TYPE={settings.camera_type}; official data collection currently tests OAK cameras")

        if settings.tensorrt_root and settings.tensorrt_root.exists():
            version = tensorrt_version(settings.tensorrt_root)
            if version and version[:2] == (10, 13):
                report.ok(f"TensorRT {'.'.join(map(str, version))}: {settings.tensorrt_root}")
            else:
                report.fail(
                    f"x86 deployment requires TensorRT 10.13; detected {version or 'unknown'} at "
                    f"{settings.tensorrt_root}"
                )
        else:
            report.fail("TENSORRT_ROOT is missing or does not exist")

        gpu_ok, gpu_text = run_quiet(["nvidia-smi", "-L"])
        if gpu_ok:
            report.ok("NVIDIA driver/GPU visible: " + gpu_text.splitlines()[0])
        else:
            report.fail("nvidia-smi failed; SONIC TensorRT deployment cannot be trusted")

        if network:
            camera_ok, detail = tcp_open(settings.camera_host, settings.camera_port)
            if camera_ok:
                report.ok(f"camera server TCP reachable: {settings.camera_host}:{settings.camera_port}")
            else:
                report.fail(f"camera server unreachable: {settings.camera_host}:{settings.camera_port} ({detail})")
            ping_ok, _ = run_quiet(["ping", "-c", "1", "-W", "1", settings.camera_host], timeout=3)
            (report.ok if ping_ok else report.warn)(f"robot/Orin ping {settings.camera_host}")
        else:
            report.warn("network probes skipped; rerun `check --real --network` after wiring robot")

        xrt_ok, xrt_text = run_quiet(["pgrep", "-fa", "runService.sh|roboticsservice|RoboticsService"])
        if xrt_ok and xrt_text:
            report.ok("XRoboToolkit PC service process detected")
        else:
            report.warn("XRoboToolkit PC service process not detected; start it before connecting PICO")

        if settings.wrist_cameras:
            report.warn("wrist camera recording enabled; exporter will fail if either wrist stream is absent")
        elif settings.right_wrist_camera:
            report.ok("custom ACT collector: ego + right_wrist enabled (official bilateral flag off)")
        else:
            report.ok("ego camera only; wrist streams disabled")

    if report.failures:
        print(f"\nRESULT: FAIL ({len(report.failures)} blocking issue(s), {len(report.warnings)} warning(s))")
    else:
        print(f"\nRESULT: PASS ({len(report.warnings)} warning(s))")
    return report


def shell_command(argv: list[str], cwd: Path, settings: Settings) -> str:
    body = f"cd {shlex.quote(str(cwd))} && exec {shlex.join(argv)}"
    if settings.tensorrt_root:
        body = f"export TENSORRT_ROOT={shlex.quote(str(settings.tensorrt_root))}; {body}"
    return "bash -lc " + shlex.quote(body)


def add_window(name: str, command: str, first: bool = False) -> None:
    if first:
        cmd = ["tmux", "new-session", "-d", "-s", SESSION, "-n", name, command]
    else:
        cmd = ["tmux", "new-window", "-t", SESSION, "-n", name, command]
    subprocess.run(cmd, check=True)


def safe_dataset_name(settings: Settings, skill: str, override: str | None) -> str:
    value = override or settings.dataset_name or f"{skill}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", value):
        raise ValueError("dataset name must contain only letters, digits, dot, underscore or dash")
    return value


def session_exists() -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", SESSION], capture_output=True, check=False
    ).returncode == 0


def start(settings: Settings, args: argparse.Namespace, sim: bool) -> int:
    real = not sim
    if real and not args.confirm_real:
        raise RuntimeError("real launch refused: add --confirm-real after completing the safety checklist")
    report = check(settings, real=real, sim=sim, network=real)
    if report.failures:
        raise RuntimeError("preflight failed; nothing was started")
    if session_exists():
        raise RuntimeError(f"tmux session {SESSION!r} already exists; attach or stop it first")
    settings.data_root.mkdir(parents=True, exist_ok=True)
    dataset = safe_dataset_name(settings, args.skill, args.dataset)
    prompt = SKILL_PROMPTS[args.skill]
    repo = settings.repo
    mode = "sim" if sim else "real"

    deploy_argv = ["bash", "./deploy.sh", "--input-type", "zmq_manager"]
    if settings.checkpoint:
        deploy_argv += ["--cp", settings.checkpoint]
    if settings.obs_config:
        deploy_argv += ["--obs-config", settings.obs_config]
    if settings.kp_scale:
        deploy_argv += ["--motor-kp-scale", settings.kp_scale]
    if settings.kd_scale:
        deploy_argv += ["--motor-kd-scale", settings.kd_scale]
    deploy_argv.append(mode)

    teleop_argv = [
        str(repo / ".venv_teleop/bin/python"),
        "gear_sonic/scripts/pico_manager_thread_server.py",
        "--manager",
    ]
    exporter_argv = [
        str(repo / ".venv_data_collection/bin/python"),
        "gear_sonic/scripts/run_data_exporter.py",
        "--task-prompt",
        prompt,
        "--dataset-name",
        dataset,
        "--root-output-dir",
        str(settings.data_root),
        "--data-collection-frequency",
        str(settings.frequency),
        "--camera-host",
        "localhost" if sim else settings.camera_host,
        "--camera-port",
        str(settings.camera_port),
    ]
    if settings.wrist_cameras:
        exporter_argv.append("--record-wrist-cameras")
    if not settings.text_to_speech:
        exporter_argv.append("--no-text-to-speech")

    try:
        if sim:
            sim_argv = [
                str(repo / ".venv_sim/bin/python"),
                "gear_sonic/scripts/run_sim_loop.py",
                "--enable-image-publish",
                "--enable-offscreen",
                "--camera-port",
                str(settings.camera_port),
            ]
            add_window("sim", shell_command(sim_argv, repo, settings), first=True)
            add_window("deploy", shell_command(deploy_argv, repo / "gear_sonic_deploy", settings))
        else:
            add_window("deploy", shell_command(deploy_argv, repo / "gear_sonic_deploy", settings), first=True)
        subprocess.run(["tmux", "set-option", "-t", SESSION, "remain-on-exit", "on"], check=True)
        add_window("pico", shell_command(teleop_argv, repo, settings))
        add_window("exporter", shell_command(exporter_argv, repo, settings))
        if settings.camera_viewer:
            viewer_argv = [
                str(repo / ".venv_data_collection/bin/python"),
                "gear_sonic/scripts/run_camera_viewer.py",
                "--camera-host",
                "localhost" if sim else settings.camera_host,
                "--camera-port",
                str(settings.camera_port),
            ]
            add_window("camera", shell_command(viewer_argv, repo, settings))
        subprocess.run(["tmux", "select-window", "-t", f"{SESSION}:deploy"], check=False)
    except Exception:
        subprocess.run(["tmux", "kill-session", "-t", SESSION], capture_output=True)
        raise

    print("\nStarted tmux session:", SESSION)
    print("Dataset:", settings.data_root / dataset)
    print("1. Attach and select deploy window; read the warning and press Enter.")
    print("2. Wait for `Init done`, then calibrate PICO.")
    print("3. Left Grip+A starts/stops an episode; Left Grip+B discards it.")
    print("4. Emergency stop: PICO A+B+X+Y or `O` in deploy window.")
    if not args.detach:
        os.execvp("tmux", ["tmux", "attach", "-t", SESSION])
    return 0


def status() -> int:
    if not session_exists():
        print(f"tmux session {SESSION!r} is not running")
        return 1
    subprocess.run(
        [
            "tmux",
            "list-windows",
            "-t",
            SESSION,
            "-F",
            "#{window_index} #{window_name} active=#{window_active} panes=#{window_panes}",
        ],
        check=False,
    )
    return 0


def verify(settings: Settings, dataset_arg: str | None) -> int:
    if not settings.data_root.exists():
        raise RuntimeError(f"data root does not exist: {settings.data_root}")
    if dataset_arg:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", dataset_arg):
            raise ValueError("dataset name must contain only letters, digits, dot, underscore or dash")
        dataset = settings.data_root / dataset_arg
    else:
        candidates = [p for p in settings.data_root.iterdir() if p.is_dir()]
        if not candidates:
            raise RuntimeError("no dataset directories found")
        dataset = max(candidates, key=lambda p: p.stat().st_mtime)
    required = [dataset / "meta/info.json", dataset / "meta/episodes.jsonl", dataset / "meta/tasks.jsonl"]
    failures = 0
    print(f"dataset: {dataset}")
    for path in required:
        if path.is_file() and path.stat().st_size > 0:
            print(f"[OK] {path.relative_to(dataset)} ({human_bytes(path.stat().st_size)})")
        else:
            print(f"[FAIL] missing/empty {path.relative_to(dataset)}")
            failures += 1
    parquet = list((dataset / "data").rglob("*.parquet")) if (dataset / "data").exists() else []
    videos = list((dataset / "videos").rglob("*.mp4")) if (dataset / "videos").exists() else []
    nonempty_parquet = [p for p in parquet if p.stat().st_size > 0]
    nonempty_videos = [p for p in videos if p.stat().st_size > 0]
    print(f"[{'OK' if nonempty_parquet else 'FAIL'}] parquet files: {len(nonempty_parquet)}")
    print(f"[{'OK' if nonempty_videos else 'FAIL'}] video files: {len(nonempty_videos)}")
    failures += int(not nonempty_parquet) + int(not nonempty_videos)
    episodes_path = dataset / "meta/episodes.jsonl"
    if episodes_path.exists():
        episodes = sum(1 for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"episodes metadata rows: {episodes}")
        failures += int(episodes == 0)
    info_path = dataset / "meta/info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            print("fps:", info.get("fps", "unknown"))
        except json.JSONDecodeError:
            print("[FAIL] meta/info.json is invalid JSON")
            failures += 1
    print("size:", human_bytes(dir_size(dataset)))
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official G1 SONIC/PICO data collection wrapper")
    parser.add_argument("--env", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    check_parser = sub.add_parser("check")
    check_parser.add_argument("--real", action="store_true")
    check_parser.add_argument("--sim", action="store_true")
    check_parser.add_argument("--network", action="store_true")
    for name in ("start-sim", "start-real"):
        start_parser = sub.add_parser(name)
        start_parser.add_argument("skill", choices=tuple(SKILL_PROMPTS))
        start_parser.add_argument("--dataset")
        start_parser.add_argument("--detach", action="store_true")
        start_parser.add_argument("--confirm-real", action="store_true")
    sub.add_parser("status")
    sub.add_parser("attach")
    stop_parser = sub.add_parser("stop")
    stop_parser.add_argument(
        "--confirm-off",
        action="store_true",
        help="confirm policy is OFF and current episode has been saved/discarded",
    )
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--dataset")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "status":
            return status()
        if args.command == "attach":
            if not session_exists():
                raise RuntimeError(f"tmux session {SESSION!r} is not running")
            os.execvp("tmux", ["tmux", "attach", "-t", SESSION])
        if args.command == "stop":
            if not session_exists():
                print("nothing to stop")
                return 0
            if not args.confirm_off:
                raise RuntimeError(
                    "refusing to kill the controller: first save/discard the episode, put policy in OFF, "
                    "then add --confirm-off"
                )
            subprocess.run(["tmux", "kill-session", "-t", SESSION], check=True)
            print(f"stopped tmux session {SESSION}; saved datasets were not deleted")
            return 0

        settings = settings_from(args.env)
        if args.command == "check":
            report = check(settings, real=args.real, sim=args.sim, network=args.network)
            return 1 if report.failures else 0
        if args.command == "start-sim":
            return start(settings, args, sim=True)
        if args.command == "start-real":
            return start(settings, args, sim=False)
        if args.command == "verify":
            return verify(settings, args.dataset)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

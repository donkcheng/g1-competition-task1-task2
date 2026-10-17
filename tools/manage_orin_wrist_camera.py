#!/usr/bin/env python3
"""Deploy and manage the task-2 D405 wrist-camera service on G1 Orin."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

import paramiko


REMOTE_DIR = "/home/unitree/g1_task2"
REMOTE_SCRIPT = f"{REMOTE_DIR}/orin_wrist_camera_server.py"
REMOTE_PID = f"{REMOTE_DIR}/wrist_camera.pid"
REMOTE_LOG = f"{REMOTE_DIR}/wrist_camera.log"
# Bracket syntax prevents pgrep from matching the pgrep command itself.
REMOTE_PROCESS_PATTERN = f"{REMOTE_DIR}/[o]rin_wrist_camera_server.py"


def run(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    _stdin, stdout, stderr = client.exec_command(command)
    code = stdout.channel.recv_exit_status()
    return code, stdout.read().decode(), stderr.read().decode()


def print_result(result: tuple[int, str, str]) -> int:
    code, stdout, stderr = result
    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, end="", file=os.sys.stderr)
    return code


def running_pids(client: paramiko.SSHClient) -> tuple[int, str, str]:
    return run(client, f"pgrep -f '{REMOTE_PROCESS_PATTERN}' || true")


def stop_service(client: paramiko.SSHClient) -> tuple[int, str, str]:
    # Find every instance by its deployed script path. This still works when
    # a failed launch has overwritten or removed the PID file.
    command = (
        f"pids=$(pgrep -f '{REMOTE_PROCESS_PATTERN}' || true); "
        f"if test -n \"$pids\"; then "
        f"kill $pids 2>/dev/null || true; "
        f"i=0; while test $i -lt 30; do "
        f"remaining=$(pgrep -f '{REMOTE_PROCESS_PATTERN}' || true); "
        f"test -z \"$remaining\" && break; "
        f"sleep 0.1; i=$((i + 1)); done; "
        f"remaining=$(pgrep -f '{REMOTE_PROCESS_PATTERN}' || true); "
        f"if test -n \"$remaining\"; then kill -KILL $remaining; fi; fi; "
        f"rm -f {REMOTE_PID} {REMOTE_PID}.tmp"
    )
    return run(client, command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "restart", "status", "logs"))
    parser.add_argument("--host", default="192.168.123.164")
    parser.add_argument("--user", default="unitree")
    parser.add_argument("--password", default=os.environ.get("ORIN_PASSWORD"))
    parser.add_argument("--serial", default="261022277224")
    parser.add_argument("--port", type=int, default=5559)
    args = parser.parse_args()
    password = args.password or getpass.getpass(f"Password for {args.user}@{args.host}: ")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(args.host, username=args.user, password=password, timeout=5)
    except paramiko.AuthenticationException:
        print(
            f"Authentication failed for {args.user}@{args.host}.",
            file=os.sys.stderr,
        )
        return 3
    except (OSError, paramiko.SSHException) as exc:
        print(f"Cannot connect to {args.host}: {exc}", file=os.sys.stderr)
        return 4

    try:
        if args.action in ("stop", "restart"):
            code = print_result(stop_service(client))
            if code or args.action == "stop":
                return code

        if args.action in ("start", "restart"):
            if args.action == "start":
                existing = running_pids(client)[1].strip()
                if existing:
                    print(f"Right wrist camera already running (PID {existing}).")
                    return 0

            code = print_result(run(client, f"mkdir -p {REMOTE_DIR}"))
            if code:
                return code
            source = (
                Path(__file__).resolve().parents[1]
                / "src"
                / "orin_wrist_camera_server.py"
            )
            with client.open_sftp() as sftp:
                sftp.put(str(source), REMOTE_SCRIPT)

            # Use a temporary PID file until the process survives startup so
            # a bind/camera failure cannot destroy the last valid PID record.
            command = (
                f"nohup python3 -u {REMOTE_SCRIPT} --serial {args.serial} "
                f"--bind 0.0.0.0 --port {args.port} > {REMOTE_LOG} 2>&1 "
                f"< /dev/null & p=$!; echo $p > {REMOTE_PID}.tmp; sleep 2; "
                f"if kill -0 $p 2>/dev/null; then "
                f"mv {REMOTE_PID}.tmp {REMOTE_PID}; cat {REMOTE_LOG}; "
                f"else cat {REMOTE_LOG}; rm -f {REMOTE_PID}.tmp; exit 1; fi"
            )
            return print_result(run(client, command))

        if args.action == "status":
            command = (
                f"pids=$(pgrep -f '{REMOTE_PROCESS_PATTERN}' || true); "
                f"if test -n \"$pids\"; then "
                f"for p in $pids; do ps -p $p -o pid,etime,args; done; "
                f"else echo STOPPED; exit 1; fi"
            )
            return print_result(run(client, command))

        return print_result(run(client, f"tail -n 100 {REMOTE_LOG}"))
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())

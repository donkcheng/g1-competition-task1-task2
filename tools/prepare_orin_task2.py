#!/usr/bin/env python3
"""Prepare or inspect the G1 Orin services required by task 2."""

from __future__ import annotations

import argparse
import getpass
import os
import time

import paramiko


def run(
    client: paramiko.SSHClient,
    command: str,
    *,
    password: str | None = None,
) -> tuple[int, str, str]:
    if password is None:
        _stdin, stdout, stderr = client.exec_command(command)
    else:
        stdin, stdout, stderr = client.exec_command(f"sudo -S -p '' {command}")
        stdin.write(password + "\n")
        stdin.flush()
        stdin.channel.shutdown_write()
    code = stdout.channel.recv_exit_status()
    return code, stdout.read().decode(), stderr.read().decode()


def print_result(result: tuple[int, str, str]) -> int:
    code, stdout, stderr = result
    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, end="", file=os.sys.stderr)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "status"))
    parser.add_argument("--host", default="192.168.123.164")
    parser.add_argument("--user", default="unitree")
    parser.add_argument("--password", default=os.environ.get("ORIN_PASSWORD"))
    args = parser.parse_args()
    password = args.password or getpass.getpass(
        f"Password for {args.user}@{args.host}: "
    )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            args.host,
            username=args.user,
            password=password,
            timeout=5,
        )
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
        if args.action == "prepare":
            if print_result(
                run(
                    client,
                    "systemctl stop lora2mqtt.service",
                    password=password,
                )
            ):
                return 1
            if print_result(
                run(
                    client,
                    "systemctl restart g1-dex1-right.service",
                    password=password,
                )
            ):
                return 1
            time.sleep(2.0)

        dex = run(client, "systemctl is-active g1-dex1-right.service")
        lora = run(client, "systemctl is-active lora2mqtt.service || true")
        print(f"g1-dex1-right.service: {dex[1].strip() or 'unknown'}")
        print(f"lora2mqtt.service: {lora[1].strip() or 'unknown'}")
        if dex[0] != 0 or dex[1].strip() != "active":
            print_result(
                run(
                    client,
                    "journalctl -u g1-dex1-right.service -n 40 --no-pager",
                )
            )
            return 1

        logs = run(
            client,
            "journalctl -u g1-dex1-right.service -n 80 --no-pager",
        )[1]
        if "Motor ID: 0" not in logs or "Gripper Server started" not in logs:
            print(logs, file=os.sys.stderr)
            print(
                "Dex1 service is active but the right motor was not detected.",
                file=os.sys.stderr,
            )
            return 1
        print("Right Dex1 motor ID 0 detected and server started.")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download and verify Git LFS SONIC example references into this project.

The official checkout is read-only here. Every downloaded object is checked
against the SHA-256 and byte count in its Git LFS pointer before installation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import tempfile
import time
import urllib.request
from pathlib import Path


POINTER = re.compile(
    rb"version https://git-lfs.github.com/spec/v1\n"
    rb"oid sha256:([0-9a-f]{64})\n"
    rb"size ([0-9]+)\n?\Z"
)
BATCH_URL = "https://github.com/NVlabs/GR00T-WholeBodyControl.git/info/lfs/objects/batch"


def download_verified(oid: str, size: int, entry: dict, scratch: Path) -> Path:
    if entry.get("size") != size or "error" in entry:
        raise RuntimeError(f"LFS object unavailable: {oid[:12]}")
    action = entry.get("actions", {}).get("download", {})
    url = action.get("href", "")
    if not url.startswith("https://"):
        raise RuntimeError(f"invalid download URL for {oid[:12]}")
    destination = scratch / oid
    for attempt in range(1, 6):
        try:
            download_request = urllib.request.Request(url, headers=action.get("header", {}))
            digest = hashlib.sha256()
            count = 0
            with urllib.request.urlopen(download_request, timeout=30) as response:
                with destination.open("wb") as stream:
                    while chunk := response.read(65536):
                        stream.write(chunk)
                        digest.update(chunk)
                        count += len(chunk)
                        if count > size:
                            raise RuntimeError(f"oversized LFS object: {oid[:12]}")
            if count != size or digest.hexdigest() != oid:
                raise RuntimeError(f"checksum/size mismatch: {oid[:12]}")
            return destination
        except (OSError, TimeoutError, RuntimeError):
            if attempt == 5:
                raise
            time.sleep(attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.repo / "gear_sonic_deploy/reference/example"
    files = sorted(source.glob("*/*.csv"))
    if len(files) != 78:
        raise RuntimeError(f"expected 78 official example CSV files, found {len(files)}")
    pointers: dict[Path, tuple[str, int]] = {}
    for path in files:
        match = POINTER.fullmatch(path.read_bytes())
        if not match:
            raise RuntimeError(f"not a Git LFS pointer: {path}")
        pointers[path] = (match.group(1).decode("ascii"), int(match.group(2)))
    if sum(size for _, size in pointers.values()) > 30_000_000:
        raise RuntimeError("unexpected aggregate download size")

    objects = sorted(set(pointers.values()))
    payload = json.dumps({
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": oid, "size": size} for oid, size in objects],
    }).encode("utf-8")
    request = urllib.request.Request(
        BATCH_URL, payload,
        headers={"Accept": "application/vnd.git-lfs+json",
                 "Content-Type": "application/vnd.git-lfs+json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        manifest = json.load(response)
    available = {entry["oid"]: entry for entry in manifest["objects"]}

    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sonic-lfs-", dir=args.output) as scratch:
        staged: dict[str, Path] = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(download_verified, oid, size, available.get(oid, {}), Path(scratch)): oid
                for oid, size in objects
            }
            for index, future in enumerate(as_completed(futures), 1):
                staged[futures[future]] = future.result()
                print(f"verified {index}/{len(objects)}", flush=True)

        for path, (oid, _) in pointers.items():
            relative = path.relative_to(source)
            output_file = args.output / relative
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_bytes(staged[oid].read_bytes())
    print(f"Ready: {len(files)} verified CSV files in {args.output}")


if __name__ == "__main__":
    main()

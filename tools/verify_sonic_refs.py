#!/usr/bin/env python3
"""Verify local SONIC CSVs against official Git LFS pointer SHA-256 values."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from download_sonic_refs import POINTER


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.repo / "gear_sonic_deploy/reference/example"
    pointers = sorted(source.glob("*/*.csv"))
    actual = sorted(args.reference_dir.glob("*/*.csv"))
    if len(pointers) != 78 or len(actual) != 78:
        raise RuntimeError(f"expected 78 source and local CSVs; found {len(pointers)} and {len(actual)}")
    for pointer in pointers:
        match = POINTER.fullmatch(pointer.read_bytes())
        if not match:
            raise RuntimeError(f"official pointer changed unexpectedly: {pointer}")
        target = args.reference_dir / pointer.relative_to(source)
        if not target.is_file():
            raise RuntimeError(f"missing local CSV: {target}")
        data = target.read_bytes()
        if len(data) != int(match.group(2)) or hashlib.sha256(data).hexdigest() != match.group(1).decode():
            raise RuntimeError(f"hash/size mismatch: {target}")
    print(f"Verified {len(pointers)}/{len(pointers)} SONIC CSVs against official LFS SHA-256 and size")


if __name__ == "__main__":
    main()

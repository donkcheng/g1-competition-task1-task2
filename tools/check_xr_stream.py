#!/usr/bin/env python3
"""Read-only freshness check for XRoboToolkit body tracking."""

from __future__ import annotations

import argparse
import time

import xrobotoolkit_sdk as xrt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument(
        "--wait-seconds", type=float, default=0.0,
        help="wait this long for the first genuinely updating body frame",
    )
    parser.add_argument(
        "--min-rate", type=float, default=0.0,
        help="fail unless the measured body-frame rate is at least this value",
    )
    parser.add_argument(
        "--max-gap", type=float, default=0.0,
        help="fail if the longest body-frame gap exceeds this many seconds",
    )
    args = parser.parse_args()
    if not 1 <= args.seconds <= 120:
        parser.error("--seconds must be between 1 and 120")
    if not 0 <= args.wait_seconds <= 300:
        parser.error("--wait-seconds must be between 0 and 300")
    if args.min_rate < 0:
        parser.error("--min-rate must be non-negative")
    if args.max_gap < 0:
        parser.error("--max-gap must be non-negative")

    xrt.init()
    start = time.monotonic()
    last_stamp = int(xrt.get_time_stamp_ns())
    if args.wait_seconds > 0:
        deadline = start + args.wait_seconds
        while time.monotonic() < deadline:
            stamp = int(xrt.get_time_stamp_ns())
            if bool(xrt.is_body_data_available()) and stamp > 0 and stamp != last_stamp:
                break
            last_stamp = stamp
            time.sleep(0.02)
        else:
            xrt.close()
            print(
                "ERROR: no updating PICO body frames. XRoboToolkit may show a "
                "TCP connection while Tracking/Send is still off.",
                flush=True,
            )
            return 2

    start = time.monotonic()
    last_stamp = 0
    last_change = start
    frames = 0
    stale_ticks = 0
    longest_gap = 0.0
    saw_frame = False
    stale_now = False
    dropout_events = 0
    sample_end = start
    try:
        while time.monotonic() - start < args.seconds:
            now = time.monotonic()
            stamp = int(xrt.get_time_stamp_ns())
            available = bool(xrt.is_body_data_available())
            if stamp > 0 and stamp != last_stamp and available:
                if saw_frame:
                    longest_gap = max(longest_gap, now - last_change)
                saw_frame = True
                last_change = now
                last_stamp = stamp
                frames += 1
            stale = saw_frame and (not available or now - last_change > 0.3)
            if stale:
                stale_ticks += 1
                if not stale_now:
                    dropout_events += 1
            stale_now = stale
            time.sleep(0.02)
    finally:
        # Snapshot the sampling end before xrt.close(): native XR teardown can
        # take roughly half a second and must not be counted as a stream gap.
        sample_end = time.monotonic()
        xrt.close()
    longest_gap = max(longest_gap, sample_end - last_change if saw_frame else 0.0)
    print(f"frames={frames} rate={frames / args.seconds:.1f}/s "
          f"dropouts_over_0.3s={dropout_events} stale_ticks={stale_ticks} "
          f"longest_gap={longest_gap:.3f}s")
    rate = frames / args.seconds
    if frames == 0:
        print("ERROR: PICO body stream produced no frames.", flush=True)
        return 2
    if args.min_rate > 0 and rate < args.min_rate:
        print(
            f"ERROR: PICO body rate {rate:.1f}/s is below {args.min_rate:.1f}/s.",
            flush=True,
        )
        return 3
    if args.max_gap > 0 and longest_gap > args.max_gap:
        print(
            f"ERROR: PICO body gap {longest_gap:.3f}s exceeds {args.max_gap:.3f}s.",
            flush=True,
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

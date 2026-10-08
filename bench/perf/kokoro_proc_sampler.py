#!/usr/bin/env python3
"""Bounded target-side RSS/disk sampler; run beside service in same PID namespace.

--pid is read-only; this process never signals that PID. JSONL unix_s timestamps
can be consumed by kokoro_service_soak.py --telemetry-jsonl. Synchronize clocks
when the HTTP runner is on another host. Stop only this sampler using SIGTERM.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import signal
import threading
import time


def sample(pid, disk_path, *, proc_root=Path("/proc")):
    row = {"unix_s": time.time(), "pid": pid, "disk_path": str(disk_path)}
    try:
        status = (proc_root / str(pid) / "status").read_text()
        row["rss_bytes"] = next(int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith("VmRSS:"))
        # comm (field 2) can contain spaces/parentheses. starttime is field 22.
        stat = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        row["process_start_ticks"] = int(stat[19])
        row["free_disk_bytes"] = shutil.disk_usage(disk_path).free
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--disk-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=1860)
    parser.add_argument("--interval", type=float, default=1)
    args = parser.parse_args(argv)
    if args.pid <= 0 or any(not math.isfinite(v) or v <= 0 for v in (args.seconds, args.interval)):
        parser.error("pid/seconds/interval must be positive and finite")
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    deadline = time.monotonic() + args.seconds
    failed = False
    try:
        with Path(args.out).open("x") as stream:
            while not stop.is_set() and time.monotonic() < deadline:
                row = sample(args.pid, args.disk_path)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                failed |= "error" in row
                stop.wait(min(args.interval, max(0, deadline - time.monotonic())))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

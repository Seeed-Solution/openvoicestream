#!/usr/bin/env python3
"""Optional resource sampler to run alongside bench.py on the DEVICE (not the
Mac driving the benchmark). Samples CPU/mem every --interval seconds and,
when available, NPU/GPU load, writing one CSV row per sample.

Accelerator probes (best-effort; missing ones are left blank, not guessed):
  - RK3576/RK3588 NPU: /sys/kernel/debug/rknpu/load (needs root/CAP_SYS_ADMIN)
  - Jetson GPU/NPU:    tegrastats --interval <ms> (parsed if present)
  - Hailo-8 NPU:       `hailortcli monitor` is interactive; not sampled here —
    fall back to /sys/devices/.../hailo0 utilization if the driver exposes one
    on this build (unverified — leave blank if absent rather than guess).

Usage (on the device, while bench.py runs from elsewhere):
    python3 resource_sampler.py --out /tmp/res.csv --interval 1 --duration 120
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import select
import signal
import subprocess
import time
from pathlib import Path

_STOP_REQUESTED = False

def _request_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def read_top_cpu_mem(deadline_mono: float | None = None) -> tuple[float | None, float | None]:
    """Return (cpu_pct_used, mem_pct_used) via /proc, portable across ARM boards."""
    try:
        with open("/proc/stat") as f:
            line1 = f.readline()
        delay = 0.15 if deadline_mono is None else max(0.0, min(0.15, deadline_mono - time.monotonic()))
        time.sleep(delay)
        with open("/proc/stat") as f:
            line2 = f.readline()
        def parts(line):
            vals = [int(x) for x in line.split()[1:]]
            idle = vals[3] + vals[4]
            total = sum(vals)
            return idle, total
        idle1, total1 = parts(line1)
        idle2, total2 = parts(line2)
        dt, di = total2 - total1, idle2 - idle1
        cpu_pct = 100.0 * (dt - di) / dt if dt > 0 else None
    except Exception:
        cpu_pct = None
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                k, v = line.split(":")
                mem[k.strip()] = int(v.strip().split()[0])
        total = mem.get("MemTotal", 0)
        avail = mem.get("MemAvailable", 0)
        mem_pct = 100.0 * (total - avail) / total if total > 0 else None
    except Exception:
        mem_pct = None
    return cpu_pct, mem_pct


def read_rknpu_load() -> str | None:
    path = Path("/sys/kernel/debug/rknpu/load")
    if not path.exists():
        return None
    try:
        return path.read_text().strip()
    except PermissionError:
        return "permission_denied"
    except Exception:
        return None


def _read_tegrastats_sample(deadline_mono: float, cleanup_deadline_mono: float,
                            command: list[str] | None = None) -> dict:
    """Read one tegrastats line.

    tegrastats has no `--count`: it prints forever until stopped. Start it,
    take the first line, then terminate and reap it, so neither the sample nor
    the process is lost. stderr is kept out of the CSV so an error message
    cannot be recorded as utilization data.
    """
    proc = None
    chunks: list[bytes] = []
    status = "UNPROVEN_NO_SAMPLE"
    result = {"raw": None, "status": status, "child_pid": None,
              "term_count": 0, "child_status": "NOT_STARTED"}
    command = command or ["tegrastats", "--interval", "200"]
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        result["child_pid"] = proc.pid
        result["child_status"] = "RUNNING"
        fd = proc.stdout.fileno() if proc.stdout else None
        if fd is None:
            result["status"] = "UNPROVEN_NO_PIPE"
            return result
        while time.monotonic() < deadline_mono:
            ready, _, _ = select.select([fd], [], [], max(0.0, deadline_mono - time.monotonic()))
            if not ready:
                status = "UNPROVEN_READ_DEADLINE"
                break
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                status = "OBSERVED_RAW"
                break
        else:
            status = "UNPROVEN_READ_DEADLINE"
        raw = b"".join(chunks).split(b"\n", 1)[0].decode("utf-8", "replace").strip() or None
        if raw and status != "UNPROVEN_READ_DEADLINE":
            status = "OBSERVED_RAW"
        result["raw"] = raw
        result["status"] = status
        return result
    except Exception:
        result["status"] = "UNPROVEN_READ_ERROR"
        return result
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            result["term_count"] = 1
            while proc.poll() is None and time.monotonic() < cleanup_deadline_mono:
                try:
                    proc.wait(timeout=max(0.0, cleanup_deadline_mono - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
            if proc.poll() is None:
                result["status"] = "UNPROVEN_CHILD_REMAINS"
                result["child_status"] = "UNPROVEN_CHILD_REMAINS"
            else:
                result["child_status"] = "REAPED_AFTER_TERM"
        elif proc is not None:
            result["child_status"] = "ALREADY_EXITED"
        if proc is not None and proc.stdout is not None:
            proc.stdout.close()


def read_tegrastats_once() -> str | None:
    """Compatibility wrapper with a short bounded local deadline."""
    now = time.monotonic()
    return _read_tegrastats_sample(now + 1.0, now + 2.0).get("raw")


_RAM_RE = re.compile(r"\bRAM\s+(\d+)/(\d+)MB\b")
# Jetson tegrastats emits both `GR3D_FREQ 27%@[1172]` and
# `GR3D_FREQ 0%` (with no clock suffix). Stop at `%` so either form parses;
# requiring digits immediately after whitespace keeps `-1%` invalid.
_GR3D_RE = re.compile(r"\bGR3D_FREQ\s+(\d+)%")
_TEMP_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)@(-?\d+(?:\.\d+)?)C\b")


def parse_tegrastats_line(line: str | None) -> dict:
    """Parse fields used by the resource gate; absent fields stay UNPROVEN."""
    if not line:
        return {"parse_status": "UNPROVEN_NO_SAMPLE"}
    row: dict = {"parse_status": "OBSERVED_RAW"}
    ram = _RAM_RE.search(line)
    if ram:
        row["ram_used_mib"], row["ram_total_mib"] = map(int, ram.groups())
    gr3d = _GR3D_RE.search(line)
    if gr3d:
        gr3d_pct = int(gr3d.group(1))
        if 0 <= gr3d_pct <= 100:
            row["gr3d_util_pct"] = gr3d_pct
    temps = {name: float(value) for name, value in _TEMP_RE.findall(line)}
    if temps:
        row["temperatures_c"] = temps
        row["thermal_max_c"] = max(temps.values())
        if "cpu" in temps:
            row["cpu_temp_c"] = temps["cpu"]
        if "gpu" in temps:
            row["gpu_temp_c"] = temps["gpu"]
    required = ("ram_used_mib", "ram_total_mib", "gr3d_util_pct", "thermal_max_c")
    if not all(key in row for key in required):
        row["parse_status"] = "UNPROVEN_MISSING_FIELDS"
        row["missing_fields"] = [key for key in required if key not in row]
    return row


def read_process_info(pid: int, *, proc_root: Path = Path("/proc")) -> dict:
    try:
        base = proc_root / str(pid)
        fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
        first_ticks = int(fields[19])
        status = (base / "status").read_text()
        second = (base / "stat").read_text().rsplit(")", 1)[1].split()
        second_ticks = int(second[19])
        if first_ticks != second_ticks:
            return {"start_ticks": None, "rss_bytes": None,
                    "identity_status": "UNPROVEN_PID_CHANGED"}
        try:
            rss = next(int(x.split()[1]) * 1024 for x in status.splitlines() if x.startswith("VmRSS:"))
        except StopIteration:
            return {"start_ticks": first_ticks, "rss_bytes": None,
                    "identity_status": "UNPROVEN_RSS_MISSING"}
        return {"start_ticks": first_ticks, "rss_bytes": rss,
                "identity_status": "STABLE"}
    except (OSError, ValueError, IndexError):
        return {"start_ticks": None, "rss_bytes": None,
                "identity_status": "UNPROVEN_PID_MISSING"}


def read_process_start_ticks(pid: int, *, proc_root: Path = Path("/proc")) -> int | None:
    return read_process_info(pid, proc_root=proc_root)["start_ticks"]


def main(argv=None) -> int:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--accel", choices=["none", "rknpu", "tegrastats"], default="none")
    p.add_argument("--pid", type=int, default=None,
                   help="optional target PID to bind samples to")
    p.add_argument("--start-ticks", type=int, default=None,
                   help="expected /proc/<pid>/stat starttime; requires --pid")
    p.add_argument("--reserve", type=float, default=1.0,
                   help="seconds retained for child TERM/reap")
    args = p.parse_args(argv)
    if args.pid is not None and args.pid <= 0:
        p.error("--pid must be positive")
    if args.start_ticks is not None and args.pid is None:
        p.error("--start-ticks requires --pid")
    if (not math.isfinite(args.interval) or not math.isfinite(args.duration)
            or not math.isfinite(args.reserve) or args.interval <= 0
            or args.duration <= 0 or args.reserve < 0 or args.reserve >= args.duration):
        p.error("interval/duration must be positive finite; reserve must be finite and in [0,duration)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["t", "sample_start_mono", "sample_end_mono", "cpu_pct", "mem_pct", "accel_raw", "ram_used_mib",
              "ram_total_mib", "gr3d_util_pct", "cpu_temp_c", "gpu_temp_c",
              "thermal_max_c", "temperatures_c", "parse_status", "missing_fields", "pid",
              "start_ticks", "rss_bytes", "process_status", "sample_interval_s",
              "accel_child_pid", "accel_term_count", "accel_child_status"]
    expected_ticks = args.start_ticks
    process_status = "UNBOUND"
    if args.pid is not None:
        observed = read_process_info(args.pid)
        observed_ticks = observed["start_ticks"]
        if observed_ticks is None:
            process_status = "UNPROVEN_PID_MISSING"
        elif expected_ticks is None:
            expected_ticks = observed_ticks
            process_status = "BOUND"
        elif observed_ticks != expected_ticks:
            process_status = "UNPROVEN_PID_CHANGED"
        else:
            process_status = "BOUND"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        t0 = time.monotonic()
        sample_deadline = t0 + args.duration - args.reserve
        previous_sample = t0
        while time.monotonic() < sample_deadline and not _STOP_REQUESTED:
            sample_start_mono = time.monotonic()
            if args.pid is not None:
                current = read_process_info(args.pid)
                current_ticks = current["start_ticks"]
                if expected_ticks is None or current_ticks != expected_ticks:
                    process_status = "UNPROVEN_PID_CHANGED"
                    sample_end_mono = time.monotonic()
                    w.writerow({"t": round(sample_end_mono - t0, 2),
                                "sample_start_mono": sample_start_mono,
                                "sample_end_mono": sample_end_mono,
                                "pid": args.pid, "start_ticks": current_ticks,
                                "rss_bytes": current["rss_bytes"],
                                "process_status": process_status})
                    f.flush()
                    break
            cpu_pct, mem_pct = read_top_cpu_mem(sample_deadline)
            accel_raw = None
            parsed = {}
            sample_status = None
            if args.accel == "rknpu":
                accel_raw = read_rknpu_load()
            elif args.accel == "tegrastats":
                sample = _read_tegrastats_sample(sample_deadline, t0 + args.duration)
                accel_raw = sample["raw"]
                sample_status = sample["status"]
                parsed = parse_tegrastats_line(accel_raw)
                if sample_status != "OBSERVED_RAW":
                    parsed["parse_status"] = sample_status
            current = read_process_info(args.pid) if args.pid is not None else {}
            now = time.monotonic()
            if args.pid is not None and current.get("start_ticks") != expected_ticks:
                process_status = "UNPROVEN_PID_CHANGED"
            sample_end_mono = time.monotonic()
            row = {"t": round(sample_end_mono - t0, 2),
                   "sample_start_mono": sample_start_mono,
                   "sample_end_mono": sample_end_mono,
                   "cpu_pct": cpu_pct, "mem_pct": mem_pct,
                   "accel_raw": accel_raw, "pid": args.pid,
                   "start_ticks": expected_ticks,
                   "rss_bytes": current.get("rss_bytes"),
                   "process_status": process_status,
                   "sample_interval_s": round(now - previous_sample, 6),
                   "accel_child_pid": sample.get("child_pid") if args.accel == "tegrastats" else None,
                   "accel_term_count": sample.get("term_count", 0) if args.accel == "tegrastats" else 0,
                   "accel_child_status": sample.get("child_status") if args.accel == "tegrastats" else None}
            if args.pid is not None and process_status != "BOUND":
                row["rss_bytes"] = None
            row.update(parsed)
            if "temperatures_c" in row:
                row["temperatures_c"] = str(row["temperatures_c"])
            if "missing_fields" in row:
                row["missing_fields"] = ",".join(row["missing_fields"])
            w.writerow(row)
            f.flush()
            previous_sample = now
            if args.pid is not None and process_status != "BOUND":
                break
            if args.accel == "tegrastats" and sample.get("status") == "UNPROVEN_CHILD_REMAINS":
                break
            time.sleep(min(args.interval, max(0.0, sample_deadline - time.monotonic())))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

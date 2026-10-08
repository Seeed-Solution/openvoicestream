import csv
import importlib.util
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from unittest import mock
import unittest
from pathlib import Path


MODULE = Path(__file__).with_name("resource_sampler.py")
spec = importlib.util.spec_from_file_location("resource_sampler_under_test", MODULE)
sampler = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sampler)


REAL_LINE = (
    "10-04-2026 10:02:18 RAM 5997/15656MB (lfb 50x1MB) "
    "CPU [0%@1984] EMC_FREQ 2%@3199 GR3D_FREQ 27%@[1172] "
    "gpu@59.625C cpu@62.593C tj@62.593C"
)
CURRENT_NO_CLOCK_LINE = (
    "RAM 4895/15656MB CPU [52%@1984] GR3D_FREQ 0% "
    "cv0@58.781C cpu@62.5C gpu@59.468C"
)


class ResourceSamplerTests(unittest.TestCase):
    def test_real_tegrastats_fields_are_parsed_without_zero_defaults(self):
        row = sampler.parse_tegrastats_line(REAL_LINE)
        self.assertEqual(row["parse_status"], "OBSERVED_RAW")
        self.assertEqual(row["ram_used_mib"], 5997)
        self.assertEqual(row["ram_total_mib"], 15656)
        self.assertEqual(row["gr3d_util_pct"], 27)
        self.assertEqual(row["gpu_temp_c"], 59.625)
        self.assertEqual(row["cpu_temp_c"], 62.593)

    def test_current_no_clock_gr3d_form_is_parsed(self):
        row = sampler.parse_tegrastats_line(CURRENT_NO_CLOCK_LINE)
        self.assertEqual(row["gr3d_util_pct"], 0)
        self.assertNotIn("gr3d_util_pct", row.get("missing_fields", []))

    def test_gr3d_boundary_values_are_valid(self):
        for value in (0, 100):
            row = sampler.parse_tegrastats_line(
                f"RAM 1/2MB GR3D_FREQ {value}% cv0@58C cpu@62C"
            )
            self.assertEqual(row["gr3d_util_pct"], value)
            self.assertNotIn("gr3d_util_pct", row.get("missing_fields", []))

    def test_negative_gr3d_utilization_is_not_accepted(self):
        row = sampler.parse_tegrastats_line(
            "RAM 1/2MB GR3D_FREQ -1% cpu@40C"
        )
        self.assertNotIn("gr3d_util_pct", row)
        self.assertIn("gr3d_util_pct", row["missing_fields"])

    def test_over_100_gr3d_utilization_is_not_accepted(self):
        for value in (101, 999):
            row = sampler.parse_tegrastats_line(
                f"RAM 1/2MB GR3D_FREQ {value}% cv0@58C cpu@62C"
            )
            self.assertNotIn("gr3d_util_pct", row)
            self.assertIn("gr3d_util_pct", row["missing_fields"])

    def test_missing_and_truncated_lines_remain_unproven(self):
        for line in (None, "RAM 1/2MB", "GR3D_FREQ malformed"):
            row = sampler.parse_tegrastats_line(line)
            self.assertTrue(row["parse_status"].startswith("UNPROVEN"))
            self.assertNotIn("gr3d_util_pct", row)

    def test_pid_gone_and_starttime_reuse_are_detected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            stat = root / "7" / "stat"
            stat.parent.mkdir()
            (stat.parent / "status").write_text("Name: worker\nVmRSS: 12 kB\n")
            # After the closing comm field, token index 19 is /proc stat starttime.
            stat.write_text("7 (worker) " + " ".join(["S"] + ["0"] * 18 + ["111"]))
            self.assertEqual(sampler.read_process_start_ticks(7, proc_root=root), 111)
            self.assertEqual(sampler.read_process_info(7, proc_root=root)["rss_bytes"], 12 * 1024)
            stat.write_text("7 (worker) " + " ".join(["S"] + ["0"] * 18 + ["222"]))
            self.assertEqual(sampler.read_process_start_ticks(7, proc_root=root), 222)
            stat.unlink()
            self.assertIsNone(sampler.read_process_start_ticks(7, proc_root=root))

    def test_stable_pid_without_vmrss_is_unproven_rss_not_missing_pid(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            stat = root / "8" / "stat"
            stat.parent.mkdir()
            stat.write_text("8 (worker) " + " ".join(["S"] + ["0"] * 18 + ["333"]))
            (stat.parent / "status").write_text("Name: worker\nState: Z (zombie)\n")
            info = sampler.read_process_info(8, proc_root=root)
            self.assertEqual(info["start_ticks"], 333)
            self.assertIsNone(info["rss_bytes"])
            self.assertEqual(info["identity_status"], "UNPROVEN_RSS_MISSING")

    def test_pipe_deadline_keeps_partial_line_and_reaps_with_term(self):
        cmd = [sys.executable, "-c", "import time; time.sleep(1)"]
        now = time.monotonic()
        result = sampler._read_tegrastats_sample(now + 0.03, now + 0.3, cmd)
        self.assertIsNone(result["raw"])
        self.assertIn(result["status"], {"UNPROVEN_READ_DEADLINE", "UNPROVEN_CHILD_REMAINS"})

        cmd = [sys.executable, "-c", "import sys,time; sys.stdout.write('RAM 1/2MB'); sys.stdout.flush(); time.sleep(1)"]
        now = time.monotonic()
        result = sampler._read_tegrastats_sample(now + 0.03, now + 0.3, cmd)
        self.assertEqual(result["raw"], "RAM 1/2MB")
        self.assertEqual(result["status"], "UNPROVEN_READ_DEADLINE")

    def test_child_handoff_has_pid_and_single_term_when_reap_is_late(self):
        class StableUnreapedProcess:
            pid = 424242

            def __init__(self, read_fd):
                self.stdout = os.fdopen(read_fd, "rb")
                self._terminated = 0

            def poll(self):
                return None

            def terminate(self):
                self._terminated += 1

            def wait(self, timeout=None):
                raise sampler.subprocess.TimeoutExpired(["stable-fixture"], timeout)

        read_fd, write_fd = os.pipe()
        fake = StableUnreapedProcess(read_fd)
        cmd = ["stable-fixture"]
        now = time.monotonic()
        started = time.monotonic()
        try:
            with mock.patch.object(sampler.subprocess, "Popen", return_value=fake):
                result = sampler._read_tegrastats_sample(now + 0.01, now + 0.02, cmd)
        finally:
            os.close(write_fd)
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertIsInstance(result["child_pid"], int)
        self.assertEqual(result["term_count"], 1)
        self.assertEqual(result["child_status"], "UNPROVEN_CHILD_REMAINS")
        self.assertEqual(fake._terminated, 1)

    def test_pid_reuse_during_sample_drops_rss_binding(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "samples.csv"
            stable = {"start_ticks": 111, "rss_bytes": 1000, "identity_status": "STABLE"}
            reused = {"start_ticks": 222, "rss_bytes": 9999, "identity_status": "STABLE"}
            with mock.patch.object(sampler, "read_process_info", side_effect=[stable, stable, reused]), \
                 mock.patch.object(sampler, "read_top_cpu_mem", return_value=(1.0, 2.0)), \
                 mock.patch.object(sampler, "_read_tegrastats_sample", return_value={"raw": None, "status": "UNPROVEN_NO_SAMPLE", "child_pid": None, "term_count": 0, "child_status": "NOT_STARTED"}):
                self.assertEqual(sampler.main(["--out", str(out), "--duration", "0.03", "--reserve", "0.005", "--interval", "0.001", "--pid", "7", "--start-ticks", "111"]), 0)
            with out.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[-1]["process_status"], "UNPROVEN_PID_CHANGED")
            self.assertEqual(rows[-1]["rss_bytes"], "")
            self.assertLessEqual(float(rows[-1]["sample_start_mono"]),
                                 float(rows[-1]["sample_end_mono"]))

    def test_absolute_sample_window_contains_sampling_and_joins_host_interval(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "samples.csv"
            observed = []

            def fake_cpu_mem(deadline):
                observed.append(time.monotonic())
                return 1.0, 2.0

            request_start = time.monotonic()
            with mock.patch.object(sampler, "read_top_cpu_mem", side_effect=fake_cpu_mem):
                self.assertEqual(sampler.main([
                    "--out", str(out), "--duration", "0.03", "--reserve", "0.005",
                    "--interval", "0.001",
                ]), 0)
            request_end = time.monotonic()
            with out.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertGreaterEqual(len(rows), 1)
            self.assertEqual(len(observed), len(rows))
            for row, sample_mono in zip(rows, observed):
                start = float(row["sample_start_mono"])
                end = float(row["sample_end_mono"])
                self.assertTrue(math.isfinite(start))
                self.assertTrue(math.isfinite(end))
                self.assertTrue(time.monotonic() >= end)
                self.assertLessEqual(start, sample_mono)
                self.assertLessEqual(sample_mono, end)
                self.assertLessEqual(request_start, end)
                self.assertLessEqual(start, request_end)
                self.assertIn("t", row)
                self.assertIn("cpu_pct", row)
                self.assertIn("mem_pct", row)

    def test_sigterm_exits_zero_after_writing_boundable_cpu_rows(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "samples.csv"
            proc = subprocess.Popen([sys.executable, str(MODULE), "--out", str(out), "--duration", "10", "--reserve", "0.5", "--interval", "0.01", "--accel", "none"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            deadline = time.monotonic() + 3
            while not out.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(out.exists())
            proc.send_signal(signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=3)
            self.assertEqual(proc.returncode, 0, (stdout, stderr))
            with out.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertGreaterEqual(len(rows), 1)
            self.assertTrue(all(math.isfinite(float(row["sample_start_mono"])) and math.isfinite(float(row["sample_end_mono"])) for row in rows))

    def test_monotonic_bounded_default_sampler_output(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "samples.csv"
            self.assertEqual(sampler.main(["--out", str(out), "--duration", "0.03", "--reserve", "0.005", "--interval", "0.001"]), 0)
            with out.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertGreaterEqual(len(rows), 1)
            times = [float(row["t"]) for row in rows]
            self.assertEqual(times, sorted(times))
            self.assertTrue(all(row["process_status"] == "UNBOUND" for row in rows))


if __name__ == "__main__":
    unittest.main()

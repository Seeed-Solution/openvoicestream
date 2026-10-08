"""CPU contract tests for the reusable MOSS JSONL benchmark entry."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench/perf/moss_worker_fresh_v091.py"
MOSS_PATCH = ROOT / "third_party/jetson-voice-engine/engine-overlay-v011/patches/v011-0008-moss-tts-nano-port.patch"


def _worker_patch_payload(text: str) -> tuple[int, list[str]]:
    marker = "+++ b/examples/omni/moss_tts_nano_worker.cpp"
    start = text.index(marker)
    match = re.search(r"^@@ -0,0 \+1,(\d+) @@$", text[start:], re.MULTILINE)
    assert match is not None
    body_start = start + match.end()
    body = []
    for line in text[body_start:].splitlines():
        if line.startswith("diff --git "):
            break
        if line.startswith("+"):
            body.append(line[1:])
    return int(match.group(1)), body


def test_v011_worker_patch_hunk_covers_complete_body(tmp_path: Path) -> None:
    text = MOSS_PATCH.read_text()
    declared, body = _worker_patch_payload(text)
    assert declared == len(body) == 722
    extracted = "\n".join(body) + "\n"
    assert extracted.endswith("    }\n}\n")
    assert "return 0;" in extracted

    # The historical 713-line header must fail this payload-integrity check.
    old_text = text.replace("@@ -0,0 +1,722 @@", "@@ -0,0 +1,713 @@")
    old_declared, old_body = _worker_patch_payload(old_text)
    assert old_declared != len(old_body)


def _fake_worker(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import base64, json, sys
print('[TensorRT] diagnostic before JSON', flush=True)
print(json.dumps({'event':'worker_ready','ok':True,'sample_rate':48000,'channels':2}), flush=True)
for line in sys.stdin:
    q=json.loads(line); cap=q.get('max_new_frames')
    print('non-json runtime log', flush=True)
    pcm=base64.b64encode(b'\\x00\\x20\\x00\\x20').decode()
    print(json.dumps({'event':'chunk','id':q['id'],'ok':True,'audio_b64':pcm,'frame_index':0,'samples':2}), flush=True)
    stop = cap is None or cap >= 3
    print(json.dumps({'event':'done','id':q['id'],'ok':True,'total_samples':2,
      'generated_frames':3 if stop else 1,'eos_seen':stop,
      'accepted_audio_frames':3 if stop else 1,'sampled_frames':4 if stop else 1,
      'sampler_stop':stop,'stop_signal_scope':'model_sampler_stop',
      'finish_reason':'model_stop' if stop else 'length',
      'effective_frame_cap':cap or 1000,'requested_max_new_frames':cap or 1000,
      'prefill_seq_len':2,'decode_budget':10,
      'codec_max_frames_per_batch':8,'ttfa_ms':4,'wall_ms':8}), flush=True)
"""
    )
    path.chmod(0o755)


def _run(tmp_path: Path, cap: int | None) -> dict:
    fake = tmp_path / "fake_worker.py"
    _fake_worker(fake)
    out = tmp_path / ("cap" if cap else "stop")
    cmd = [sys.executable, str(BENCH), "--worker", str(fake), "--engine-dir", str(tmp_path),
           "--codec-onnx-dir", str(tmp_path), "--output-dir", str(out), "--text", "Hello."]
    if cap is not None:
        cmd += ["--max-new-frames", str(cap)]
    cp = subprocess.run(cmd, text=True, capture_output=True, check=False)
    assert cp.returncode == 0, cp.stderr + cp.stdout
    report = json.loads((out / "moss-v091-smoke.json").read_text())
    return report


def test_model_stop_and_raw_logs_are_persisted(tmp_path: Path) -> None:
    report = _run(tmp_path, None)
    result = report["results"][0]
    assert result["finish_reason"] == "model_stop"
    assert result["eos_seen"] is True
    assert result["sampler_stop"] is True
    assert result["stop_signal_scope"] == "model_sampler_stop"
    assert result["accepted_audio_frames"] == 3
    assert result["sampled_frames"] == 4
    assert Path(report["raw_stdout"]).read_bytes().startswith(b"[TensorRT]")


def test_frame_cap_is_distinguished_from_model_stop(tmp_path: Path) -> None:
    result = _run(tmp_path, 1)["results"][0]
    assert result["finish_reason"] == "length"
    assert result["eos_seen"] is False
    assert result["sampler_stop"] is False
    assert result["accepted_audio_frames"] == 1
    assert result["sampled_frames"] == 1
    assert result["effective_frame_cap"] == 1


def test_old_done_event_remains_compatible_with_nullable_new_fields(tmp_path: Path) -> None:
    cp, out = _invoke_worker(tmp_path, """import base64,json,sys
print(json.dumps({'event':'worker_ready','ok':True,'sample_rate':48000,'channels':2}), flush=True)
for line in sys.stdin:
 q=json.loads(line); print(json.dumps({'event':'chunk','id':q['id'],'ok':True,'audio_b64':base64.b64encode(b'\\x00\\x20\\x00\\x20').decode()}), flush=True)
 print(json.dumps({'event':'done','id':q['id'],'ok':True,'eos_seen':True,'finish_reason':'model_stop'}), flush=True); break
""")
    assert cp.returncode == 0, cp.stderr + cp.stdout
    result = json.loads((out / "moss-v091-smoke.json").read_text())["results"][0]
    assert result["eos_seen"] is True
    assert result["sampler_stop"] is None
    assert result["accepted_audio_frames"] is None


def _invoke_worker(tmp_path: Path, body: str, *extra: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    worker = tmp_path / "special_worker.py"
    worker.write_text("#!/usr/bin/env python3\n" + body)
    worker.chmod(0o755)
    out = tmp_path / "special-out"
    cmd = [sys.executable, str(BENCH), "--worker", str(worker), "--engine-dir", str(tmp_path),
           "--codec-onnx-dir", str(tmp_path), "--output-dir", str(out), "--text", "Hello.",
           "--timeout", "0.4", "--reserve", "0.1"]
    cmd.extend(extra)
    return subprocess.run(cmd, text=True, capture_output=True, check=False), out


def test_invalid_budget_is_rejected_before_launch(tmp_path: Path) -> None:
    cp, out = _invoke_worker(tmp_path, "raise SystemExit(9)\n", "--max-new-frames", "0")
    assert cp.returncode != 0
    assert not out.exists()


def test_wrong_request_id_is_failure_with_persisted_result(tmp_path: Path) -> None:
    cp, out = _invoke_worker(tmp_path, """import json,sys
print(json.dumps({'event':'worker_ready','ok':True,'sample_rate':48000,'channels':2}), flush=True)
for line in sys.stdin:
 q=json.loads(line)
 print(json.dumps({'event':'chunk','id':'wrong','ok':True,'audio_b64':'ACAAIA=='}), flush=True)
 break
""")
    assert cp.returncode != 0
    result = json.loads((out / "moss-v091-smoke.json").read_text())
    assert result["status"] == "FAIL"
    assert "id mismatch" in result["error"]
    assert (out / "worker.stdout.raw").is_file()


def test_nonzero_child_is_not_observed_pass(tmp_path: Path) -> None:
    cp, out = _invoke_worker(tmp_path, """import json,sys
print(json.dumps({'event':'worker_ready','ok':True,'sample_rate':48000,'channels':2}), flush=True)
sys.exit(7)
""")
    assert cp.returncode != 0
    result = json.loads((out / "moss-v091-smoke.json").read_text())
    assert result["status"] == "FAIL"
    assert result["child_rc"] == 7


def test_timeout_terminates_owned_child_once(tmp_path: Path) -> None:
    cp, out = _invoke_worker(tmp_path, """import json,time
print(json.dumps({'event':'worker_ready','ok':True,'sample_rate':48000,'channels':2}), flush=True)
time.sleep(10)
""")
    assert cp.returncode != 0
    result = json.loads((out / "moss-v091-smoke.json").read_text())
    assert result["status"] in {"FAIL", "SURVIVOR_HANDOFF"}
    assert result["term_attempted"] <= 1


def test_done_then_noisy_delayed_exit_is_drained(tmp_path: Path) -> None:
    cp, out = _invoke_worker(tmp_path, """import base64,json,sys,time
print(json.dumps({'event':'worker_ready','ok':True,'sample_rate':48000,'channels':2}), flush=True)
for line in sys.stdin:
 q=json.loads(line); rid=q['id']; pcm=base64.b64encode(b'\\x00\\x20\\x00\\x20').decode()
 print(json.dumps({'event':'chunk','id':rid,'ok':True,'audio_b64':pcm}), flush=True)
 print(json.dumps({'event':'done','id':rid,'ok':True,'eos_seen':True,'finish_reason':'model_stop'}), flush=True)
 sys.stdout.write('diagnostic-after-done\\n' * 5000); sys.stdout.flush(); time.sleep(0.05); break
""")
    assert cp.returncode == 0, cp.stderr + cp.stdout
    result = json.loads((out / "moss-v091-smoke.json").read_text())
    assert result["status"] == "OBSERVED"
    assert result["child_rc"] == 0
    assert b"diagnostic-after-done" in (out / "worker.stdout.raw").read_bytes()

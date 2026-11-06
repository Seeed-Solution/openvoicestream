#!/usr/bin/env python3
"""N=2 concurrency + correctness verification for qwen3_tts_streaming_worker (slot-pool).

This variant: NO preload, role-prefix-fix worker, every request carries
`"language":"chinese"`, speaker `vivian`.

Phase 1+2: submit A and B back-to-back (no wait), demux chunks by id, record
           per-chunk emit timestamps -> WAVs + interleave evidence + wall-clock.
Phase 4:   single-turn C wall-clock baseline for ratio.
Phase 3:   saturate (submit 3 concurrent while 2 slots busy) -> expect pool_saturated/4429.

WAVs written to /tmp/tts_n2_{A,B}.wav for downstream RK3588 SenseVoice ASR loopback.
"""
import subprocess, json, base64, sys, struct, time, os, threading

WORKER = "/home/harvest/project/v071-build/TensorRT-Edge-LLM/build/examples/omni/qwen3_tts_streaming_worker"
ENG = "/home/harvest/qwen3-tts-export-workspace/Qwen3-TTS-12Hz-0.6B-CustomVoice/engines-nx"
CWD = "/home/harvest/project/v071-build/TensorRT-Edge-LLM"
PLUGIN = "/tmp/cv071-526/libNvInfer_edgellm_plugin.so.1.0"

TEXTS = {"A": "今天天气真不错", "B": "人工智能正在改变世界"}
LANG = "chinese"
SPK = "vivian"

cmd = [
    WORKER,
    f"--talkerEngineDir={ENG}/talker",
    f"--code2wavEngineDir={ENG}/code2wav",
    f"--codePredictorEngineDir={ENG}/code_predictor",
    f"--tokenizerDir={ENG}/talker",
    "--max_slots=2",
]

env = dict(os.environ)
env["EDGELLM_PLUGIN_PATH"] = PLUGIN
env.pop("QWEN3_TTS_PRELOAD_TALKER_EMBEDS", None)  # NO preload per spec
env["QWEN3_TTS_SEED"] = "42"
print("PRELOAD=<none>", flush=True)
print("PLUGIN=" + PLUGIN, flush=True)
print("CMD:", " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, cwd=CWD, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, bufsize=1, env=env)

pcm = {"A": bytearray(), "B": bytearray()}
chunk_log = []
done_events = {}
errors = []
sat_events = []
sample_rate = None
ready = threading.Event()
lock = threading.Lock()
t0 = None
stop = threading.Event()

def reader():
    global sample_rate
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            print("NONJSON:", line[:200], flush=True)
            continue
        et = ev.get("event") or ev.get("type")
        now = time.time()
        with lock:
            if et == "ready":
                print("READY:", json.dumps(ev, ensure_ascii=False), flush=True)
                sample_rate = ev.get("sample_rate", sample_rate)
                ready.set()
            elif et == "chunk":
                cid = ev.get("id")
                if sample_rate is None and "sample_rate" in ev:
                    sample_rate = ev["sample_rate"]
                b64 = ev.get("audio_b64") or ev.get("audio")
                if b64 and cid in pcm:
                    pcm[cid] += base64.b64decode(b64)
                tr = (now - t0) if t0 else 0.0
                chunk_log.append((tr, cid, ev.get("chunk_index"), ev.get("frames"),
                                  ev.get("is_final"), ev.get("elapsed_ms")))
            elif et == "done":
                cid = ev.get("id")
                done_events[cid] = (now, ev)
                print(f"DONE id={cid}:", json.dumps(ev, ensure_ascii=False)[:300], flush=True)
            elif et == "error":
                if ev.get("status") == 4429 or ev.get("error") == "pool_saturated" or "saturat" in json.dumps(ev):
                    sat_events.append((now, ev))
                    print("SATURATED:", json.dumps(ev, ensure_ascii=False), flush=True)
                else:
                    errors.append(ev)
                    print("ERROR:", json.dumps(ev, ensure_ascii=False), flush=True)
            else:
                print("EVENT:", json.dumps(ev, ensure_ascii=False)[:200], flush=True)
        if stop.is_set():
            break

rt = threading.Thread(target=reader, daemon=True)
rt.start()

def mkreq(rid, txt):
    return {"id": rid, "text": txt, "speaker": SPK, "language": LANG, "stream": True,
            "chunk_format": "pcm_s16le", "chunk_transport": "base64"}

def send(req):
    proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
    proc.stdin.flush()

if not ready.wait(timeout=120):
    print("FATAL: worker never emitted ready", flush=True)
    err = proc.stderr.read()
    print(err[-2000:], flush=True)
    proc.kill(); sys.exit(1)
time.sleep(0.3)

# ===== PHASE 1+2: back-to-back A then B, no wait =====
print("\n=== PHASE 1+2: concurrent A + B (back-to-back) ===", flush=True)
t0 = time.time()
send(mkreq("A", TEXTS["A"]))
send(mkreq("B", TEXTS["B"]))

deadline = time.time() + 180
while time.time() < deadline:
    with lock:
        if "A" in done_events and "B" in done_events:
            break
    if proc.poll() is not None:
        print("WORKER EXITED code", proc.returncode, flush=True)
        break
    time.sleep(0.05)
ab_wall = time.time() - t0
with lock:
    a_done = done_events.get("A", (0, {}))[0]
    b_done = done_events.get("B", (0, {}))[0]
print(f"AB_WALL_CLOCK={ab_wall:.3f}s  A_done@{a_done-t0:.3f} B_done@{b_done-t0:.3f}", flush=True)

# ===== single-turn C baseline for ratio =====
print("\n=== single-turn C (ratio baseline) ===", flush=True)
ts = time.time()
send(mkreq("C", TEXTS["A"]))
deadline = time.time() + 120
while time.time() < deadline:
    with lock:
        if "C" in done_events:
            break
    if proc.poll() is not None:
        break
    time.sleep(0.05)
single_wall = time.time() - ts
print(f"SINGLE_WALL_CLOCK(C)={single_wall:.3f}s", flush=True)

# ===== PHASE 3: saturation — fire 3 at once =====
print("\n=== PHASE 3: saturation (3 concurrent) ===", flush=True)
for rid in ("S1", "S2", "S3"):
    send(mkreq(rid, TEXTS["B"]))
deadline = time.time() + 120
while time.time() < deadline:
    with lock:
        resolved = sum(1 for r in ("S1", "S2", "S3") if r in done_events)
        sat = len(sat_events)
    if sat >= 1 and resolved >= 2:
        time.sleep(1)
        break
    if resolved == 3:
        break
    if proc.poll() is not None:
        break
    time.sleep(0.05)

time.sleep(1)
stop.set()
try:
    proc.stdin.close()
    proc.terminate()
    proc.wait(timeout=10)
except Exception:
    proc.kill()

err = proc.stderr.read()

sr = sample_rate or 24000
ch = 1
def write_wav(path, data, sr, ch):
    bits = 16
    byte_rate = sr * ch * bits // 8
    block_align = ch * bits // 8
    with open(path, "wb") as f:
        f.write(b"RIFF"); f.write(struct.pack("<I", 36 + len(data))); f.write(b"WAVE")
        f.write(b"fmt "); f.write(struct.pack("<IHHIIHH", 16, 1, ch, sr, byte_rate, block_align, bits))
        f.write(b"data"); f.write(struct.pack("<I", len(data))); f.write(data)

PFX = os.environ.get("OUT_PFX", "/tmp/tts_n2")
for rid in ("A", "B"):
    n = len(pcm[rid])
    dur = n / (sr * ch * 2) if n else 0
    write_wav(f"{PFX}_{rid}.wav", bytes(pcm[rid]), sr, ch)
    print(f"WROTE {PFX}_{rid}.wav pcm_bytes={n} dur={dur:.2f}s sr={sr}", flush=True)

print("\n=== CHUNK TIMELINE (t_rel, id, idx, frames, is_final) ===", flush=True)
for tr, cid, idx, fr, fin, el in sorted(chunk_log):
    if cid in ("A", "B"):
        print(f"  t={tr:.3f}s id={cid} idx={idx} frames={fr} final={fin}", flush=True)

a_times = [tr for tr, cid, *_ in chunk_log if cid == "A"]
b_times = [tr for tr, cid, *_ in chunk_log if cid == "B"]
interleaved = False
if a_times and b_times:
    a_first, a_last = min(a_times), max(a_times)
    b_first, b_last = min(b_times), max(b_times)
    interleaved = (b_first < a_last) and (a_first < b_last)
    print(f"\nA chunks: first={a_first:.3f} last={a_last:.3f}  count={len(a_times)}", flush=True)
    print(f"B chunks: first={b_first:.3f} last={b_last:.3f}  count={len(b_times)}", flush=True)
    print(f"INTERLEAVED={interleaved}", flush=True)

print(f"\nWALL: AB_concurrent={ab_wall:.3f}s  single_C={single_wall:.3f}s  ratio={ab_wall/single_wall if single_wall else 0:.2f}x", flush=True)
print(f"SATURATION_EVENTS={len(sat_events)}", flush=True)
for t, ev in sat_events:
    print("  RAW:", json.dumps(ev, ensure_ascii=False), flush=True)
print(f"NON_SAT_ERRORS={len(errors)}", flush=True)
for ev in errors:
    print("  ERR_RAW:", json.dumps(ev, ensure_ascii=False), flush=True)
print(f"CRASHED={proc.returncode != 0 and proc.returncode is not None}  returncode={proc.returncode}", flush=True)

if err:
    print("\n=== STDERR TAIL ===", flush=True)
    print(err[-2000:], flush=True)

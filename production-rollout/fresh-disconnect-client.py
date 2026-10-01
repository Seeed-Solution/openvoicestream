import datetime
import hashlib
import json
import pathlib
import time
import wave

import websocket


PORT = 8621
CONTAINER = "spark-voice-stack"
ROOT = pathlib.Path("/work/seeed/agent/tests/e2e/fixtures/wav")


def meta():
    return {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "container": CONTAINER,
        "port": PORT,
    }


def workers():
    found = {}
    for proc in pathlib.Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            exe = (proc / "exe").resolve()
            if exe.name.startswith("qwen3_asr_worker"):
                found[proc.name] = {
                    "exe": str(exe),
                    "sha": hashlib.sha256((proc / "exe").read_bytes()).hexdigest(),
                }
        except OSError:
            pass
    return found


def connect(mode):
    path = "v2v/stream" if mode == "v2v" else "asr/stream?language=English&vad=none"
    ws = websocket.create_connection(f"ws://127.0.0.1:{PORT}/{path}", timeout=10)
    if mode == "v2v":
        ws.send(json.dumps({
            "type": "config",
            "asr_language": "English",
            "sample_rate": 16000,
            "vad": "none",
            "multi_utterance": False,
        }))
    return ws


with wave.open(str(ROOT / "cmd_en_go_home.wav"), "rb") as wav:
    pcm = wav.readframes(wav.getnframes())

before = workers()
print(json.dumps({**meta(), "worker_before": before}), flush=True)

for mode in ["v2v", "asr"]:
    ws = connect(mode)
    ws.send_binary(pcm[:3200])
    ws.close()
    started = time.monotonic()
    time.sleep(0.1)
    ws = connect(mode)
    print(json.dumps({**meta(), "mode": mode, "reconnect_after_close_ms": (time.monotonic() - started) * 1000}), flush=True)
    for i in range(0, len(pcm), 3200):
        ws.send_binary(pcm[i:i + 3200])
    if mode == "v2v":
        ws.send(json.dumps({"type": "asr_eos"}))
    else:
        ws.send_binary(b"")
    events = []
    while True:
        raw = ws.recv()
        if not raw:
            break
        event = json.loads(raw)
        events.append(event)
        if event.get("type") in ["asr_final", "final"]:
            break
    ws.close()
    print(json.dumps({**meta(), "mode": mode, "events": events}), flush=True)
    assert any(
        e.get("text") == "Go home." and e.get("type") in ["asr_final", "final"]
        for e in events
    )

after = workers()
print(json.dumps({**meta(), "worker_after": after, "unchanged": before == after}), flush=True)
assert before and before == after

import datetime
import hashlib
import json
import pathlib
import wave

import requests
import websocket


BASE = "http://127.0.0.1:8621"
PORT = 8621
CONTAINER = "spark-voice-stack"
ROOT = pathlib.Path("/work/seeed/agent/tests/e2e/fixtures/wav")


def meta():
    return {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "container": CONTAINER,
        "port": PORT,
    }


print(json.dumps(meta()), flush=True)

for file, lang in [
    ("weather.wav", "Chinese"),
    ("weather.wav", None),
    ("cmd_en_go_home.wav", "English"),
    ("cmd_en_go_home.wav", None),
]:
    path = ROOT / file
    with path.open("rb") as audio:
        response = requests.post(
            BASE + "/asr",
            params={"language": lang} if lang else {},
            files={"file": (file, audio, "audio/wav")},
            timeout=30,
        )
    record = {
        **meta(),
        "http": file,
        "language": lang or "auto",
        "sha": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": response.status_code,
        "response": response.json(),
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)
    response.raise_for_status()

for file, lang in [("weather.wav", "Chinese"), ("cmd_en_go_home.wav", "English")]:
    with wave.open(str(ROOT / file), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
    ws = websocket.create_connection(BASE.replace("http", "ws") + "/v2v/stream", timeout=15)
    ws.send(json.dumps({
        "type": "config",
        "asr_language": lang,
        "sample_rate": 16000,
        "vad": "none",
        "multi_utterance": False,
    }))
    for i in range(0, len(pcm), 3200):
        ws.send_binary(pcm[i:i + 3200])
    ws.send(json.dumps({"type": "asr_eos"}))
    events = []
    while True:
        raw = ws.recv()
        if not raw:
            break
        event = json.loads(raw)
        events.append(event)
        if event.get("type") == "asr_final":
            break
    ws.close()
    print(json.dumps({**meta(), "v2v": file, "language": lang, "events": events}, ensure_ascii=False), flush=True)
    assert any(e.get("type") == "asr_final" and e.get("text") for e in events)

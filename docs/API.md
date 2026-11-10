# OpenVoiceStream API Reference

OpenVoiceStream exposes a unified HTTP and WebSocket API across all hardware backends (NVIDIA Jetson, Rockchip RKNN, Raspberry Pi CPU/Hailo).

All endpoints default to port `8621` on the host (internal container port `8000`).

---

## 1. OpenAI-Compatible Endpoints

OpenVoiceStream implements the standard OpenAI Audio API specification alongside model discovery endpoints.

### 1.1 Model & Capability Discovery

Before calling synthesis or transcription, clients can discover active models and supported runtime features:

```bash
# List active models and readiness metadata
curl http://device:8621/v1/models

# Discover runtime capabilities (voices, speed control, streaming, voice cloning)
curl http://device:8621/v1/capabilities
```

`GET /v1/capabilities` returns supported feature flags for the currently loaded model:
- `streaming`: Whether chunked streaming audio is supported.
- `voice_clone`: Whether zero-shot voice cloning or embedding registration is available.
- `speed_control`: Whether dynamic playback rate adjustment is supported.
- `voices`: Array of preset speaker voices and IDs.

---

### 1.2 Speech Synthesis (TTS) — `POST /v1/audio/speech`

Converts input text into spoken audio.

```bash
curl -X POST http://device:8621/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro-multi-lang-v1_0",
    "input": "Hello world, this is OpenVoiceStream running on edge hardware.",
    "voice": "af_heart",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output speech.wav
```

**Parameters:**
- `model` *(string, required)*: Model ID returned from `GET /v1/models`.
- `input` *(string, required)*: Text to synthesize.
- `voice` *(string, optional)*: Voice ID or preset name (e.g. `af_heart`, `zf_xiaobei`).
- `response_format` *(string, optional)*: Audio format (`wav` default, or `pcm`).
- `speed` *(float, optional)*: Playback speed (0.5 to 2.0). Supported on Sherpa/Matcha/RKNN backends; ignored where fixed duration is enforced.

> **Streaming**: When the backend supports streaming, `POST /v1/audio/speech` automatically responds with HTTP chunked transfer encoding containing progressive audio segments.

---

### 1.3 Audio Transcription (ASR) — `POST /v1/audio/transcriptions`

Transcribes speech from an uploaded audio file.

```bash
curl -X POST http://device:8621/v1/audio/transcriptions \
  -F "model=qwen3-asr-0.6b" \
  -F "file=@recording.wav" \
  -F "language=auto"
```

**Response:**
```json
{
  "text": "transcribed speech content"
}
```

---

## 2. Native Streaming ASR (WebSocket)

For low-latency interactive applications, OpenVoiceStream provides a bi-directional streaming WebSocket endpoint.

### Endpoint
```
WS /asr/stream?sample_rate=16000&language=auto
```

### Protocol Specification
1. **Client sends**: Raw binary chunks of **int16 PCM audio** (16kHz mono recommended, e.g. 100ms per frame).
2. **Server responds**: Progressive JSON messages with intermediate partial hypotheses:
   ```json
   {
     "text": "partial transcription",
     "is_final": false,
     "is_stable": false
   }
   ```
3. **Client commits / ends**: Send an empty binary frame `b""` to signal End-of-Speech (EOS).
4. **Server finalizes**: Returns final validated transcription:
   ```json
   {
     "text": "complete verified sentence.",
     "is_final": true,
     "is_stable": true
   }
   ```

### Python Client Example

```python
import asyncio
import numpy as np
import websockets
import json

async def stream_audio(audio_chunks):
    uri = "ws://device:8621/asr/stream?sample_rate=16000&language=auto"
    async with websockets.connect(uri) as ws:
        async def send_audio():
            for chunk in audio_chunks:
                # chunk is int16 numpy array or bytes
                await ws.send(chunk.tobytes() if hasattr(chunk, "tobytes") else chunk)
                await asyncio.sleep(0.02)
            await ws.send(b"")  # Signal end-of-speech

        async def receive_results():
            async for message in ws:
                data = json.loads(message)
                if data.get("is_final"):
                    print(f"[Final] {data.get('text')}")
                    break
                print(f"[Partial] {data.get('text')}")

        await asyncio.gather(send_audio(), receive_results())
```

---

## 3. Native Streaming & Batch TTS (HTTP)

### 3.1 Streaming TTS (`POST /tts/stream`)
Returns raw audio chunks with minimal TTFT (Time-To-First-Token).

- **URL**: `POST http://device:8621/tts/stream`
- **Payload**:
  ```json
  {
    "text": "Hello world, streaming out immediately.",
    "sid": 52
  }
  ```
- **Stream Format**: First 4 bytes are sample rate (`uint32` little-endian), followed by continuous `int16` mono PCM samples.

### 3.2 Batch TTS (`POST /tts`)
Returns a complete WAV file.

- **URL**: `POST http://device:8621/tts`
- **Payload**:
  ```json
  {
    "text": "Hello world",
    "sid": 52,
    "speed": 1.0
  }
  ```

---

## 4. Speaker & Voice-Clone Management

Endpoints for querying preset voices and registering zero-shot voice embeddings.

```bash
# List all preset speakers for the active model
curl http://device:8621/tts/speakers

# Register a custom voice-clone embedding (requires VOICE_CLONE capability)
curl -X POST http://device:8621/tts/speakers/register \
  -H "Content-Type: application/json" \
  -d '{"speaker_embedding_b64": "<base64_tensor>", "label": "custom-voice-01"}'

# Delete a registered custom speaker
curl -X DELETE http://device:8621/tts/speakers/custom-voice-01
```

- **Kokoro**: Exposes 53 preset voices (IDs `0` to `52`, e.g., `af_heart`, `bm_george`, `zf_xiaobei`).
- **Qwen3-TTS**: Exposes 52-language zero-shot cloning and 9 built-in CustomVoice instruction profiles (`vivian`, `ryan`, `serena`, etc.).

---

## 5. System Health Check

```bash
curl http://device:8621/health
```

**Response:**
```json
{
  "asr": true,
  "tts": true,
  "streaming_asr": true,
  "active_profile": "jetson-qwen3asr-matcha",
  "device": "jetson-orin-nano"
}
```

---

## 6. Backend Capabilities Matrix

| Backend | Speed Control | Pitch Shift | Voice Clone | Languages | Streaming ASR | Streaming TTS |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Matcha-TTS (TRT / RKNN)** | ✅ | ✅ | ❌ | 2 (zh+en) | — | ✅ (<10ms TTFT) |
| **Kokoro-TRT (Jetson)** | ❌ | ❌ | ❌ (53 presets) | 1 (en) | — | ✅ (~130ms TTFT) |
| **Kokoro-RKNN (RK3588/RK3576)** | ❌ | ❌ | ❌ (53 presets) | multi | — | ✅ (~380ms TTFT) |
| **Qwen3-TTS (TRT)** | ❌ | ❌ | ✅ (x-vector) | 52 | — | ✅ (~200ms TTFT) |
| **Qwen3-ASR (TRT / RKNN)** | — | — | — | 52 | ✅ | — |
| **Paraformer (TRT / RKNN)** | — | — | — | 2 (zh+en) | ✅ | — |
| **Whisper (TRT / RKNN / Hailo)** | — | — | — | multi | ❌ (Batch) | — |
| **Sherpa-ONNX (CPU / RPi)** | ✅ | ✅ | ❌ | 2 (zh+en) | ✅ | ✅ |

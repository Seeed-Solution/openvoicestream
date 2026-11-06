# Kokoro Dual-Bucket Deployment Runbook (RK3588)

Production operation runbook for deploying Kokoro Dual-Bucket TTS (CPU ONNXRuntime text frontend + Rockchip NPU RKNN T50 sliding-window streaming generator) on Rockchip RK3588 hardware (Seeed reComputer (RK3588)).

---

## 1. Architecture & Overview

Kokoro Dual-Bucket decouples TTS synthesis into two specialized stages:
1. **CPU ONNXRuntime Frontend** (`frontend_stem.onnx`, `duration_head.onnx`, `target_tail.onnx`):
   - Computes text phonemization (Misaki G2P), prosody embeddings, and duration prediction.
   - Pinned to 4 intra-op threads (`KOKORO_FRONTEND_INTRA_OP_THREADS=4`) and 1 inter-op thread to prevent core contention on Cortex-A76 cores.
2. **NPU RKNN T50 Sliding-Window Vocoder / Generator** (`generator_T50.rknn`):
   - Generates high-fidelity audio chunks on the Rockchip NPU (3-core NPU, 6 TOPS peak).
   - Window size: 50 frames; Stride: 25 frames; Overlap discard: 3001 samples.
   - Eliminates silence-gap artifacts and provides predictable sub-450ms time-to-first-token.

---

## 2. Hardware & Host Prerequisites

- **Host Board**: Seeed reComputer (RK3588) (SoC: Rockchip RK3588, 4x Cortex-A76 + 4x Cortex-A55, NPU 6 TOPS).
- **Kernel & Drivers**:
  - `/dev/rknpu` driver present with read/write access.
  - Userspace runtime: RKNPU 2.3.2 / RKNN-Toolkit2 Lite 2.3.2.
- **Docker & Compose**: Docker Engine >= 24.0, Docker Compose v2.
- **Disk Space**: At least 3 GB free disk space under `/opt` and `/var/lib/docker`.

---

## 3. Offline Model Bundle Structure

The model package is mounted read-only from the host into `/opt/kokoro-dualbucket/rk3588`.

### Directory Layout

```text
/opt/kokoro-dualbucket/rk3588/
├── kokoro_dualbucket.manifest.json  (SHA256: cfc6aa8d485415884e4cf5eac47b0a786889ed02d9556a1051ed66685927083d)
├── generate_manifest.py
├── generator.onnx
├── frontend/
│   ├── af_heart.bin
│   ├── bf_emma.bin
│   ├── config.json
│   ├── duration_head.onnx
│   ├── ef_dora.bin
│   ├── ff_siwis.bin
│   ├── frontend.manifest.json
│   ├── frontend_stem.onnx
│   ├── generator-export.manifest.json
│   ├── hf_alpha.bin
│   ├── if_sara.bin
│   ├── jf_alpha.bin
│   ├── pf_dora.bin
│   ├── target_tail.onnx
│   └── zf_xiaobei.bin
└── models/
    ├── generator_T50.rknn
    ├── generator_T240.rknn
    └── generator_T480.rknn
```

### Unpack & Verify Command

```bash
# 1. Ensure target directory exists
sudo mkdir -p /opt/kokoro-dualbucket/rk3588

# 2. Extract offline package (e.g. from release tarball)
sudo tar -zxvf kokoro_dualbucket_rk3588.tar.gz -C /opt/kokoro-dualbucket/rk3588/

# 3. Verify manifest checksum
sha256sum /opt/kokoro-dualbucket/rk3588/kokoro_dualbucket.manifest.json
# Expected: cfc6aa8d485415884e4cf5eac47b0a786889ed02d9556a1051ed66685927083d

# 4. Verify all 20 payload files against manifest (21 files total on disk)
python3 -c '
import json, hashlib, sys, pathlib
root = pathlib.Path("/opt/kokoro-dualbucket/rk3588")
manifest = json.loads((root / "kokoro_dualbucket.manifest.json").read_text())
rows = manifest.get("files", [])
if len(rows) != 20:
    sys.exit(f"Expected 20 payload files in manifest, got {len(rows)}")
for row in rows:
    p = root / row["path"]
    if not p.is_file():
        sys.exit(f"Missing file: {p}")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != row["sha256"]:
        sys.exit(f"SHA mismatch for {row['path']}")
print(f"All {len(rows)} payload bundle files verified successfully (21 files total on disk).")
'
```

---

## 4. One-Click Deployment (`deploy/install.sh`)

Deploy using the unified installation script with `--profile`:

```bash
# Production one-click deploy
deploy/install.sh --target rk3588 --profile rk3588-kokoro-dualbucket --pull --verify
```

The script automatically:
1. Validates host `/dev/rknpu` device availability.
2. Checks that `/opt/kokoro-dualbucket/rk3588/kokoro_dualbucket.manifest.json` exists.
3. Automatically appends the overlay `deploy/docker-compose.kokoro-dualbucket-rk3588.yml` to the base RK3588 configuration.
4. Starts containers and runs health smoke tests on port 8621.

---

## 5. Multi-File Compose Launch (Manual)

If running Docker Compose manually:

```bash
export OVS_PROFILE=rk3588-kokoro-dualbucket
docker compose \
  -f deploy/docker-compose.rk3588.yml \
  -f deploy/docker-compose.kokoro-dualbucket-rk3588.yml \
  up -d
```

### Docker Compose Overlay Configuration

File: `deploy/docker-compose.kokoro-dualbucket-rk3588.yml`

```yaml
services:
  speech:
    image: ${OVS_SPEECH_IMAGE:-sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rk-20261104-kokoro-dualbucket}
    environment:
      OVS_PROFILE: rk3588-kokoro-dualbucket
      ASR_BACKEND: disabled
      TTS_BACKEND: kokoro_dualbucket
      RK_PLATFORM: rk3588
      KOKORO_DUALBUCKET_ROOT: /opt/kokoro-dualbucket/rk3588
      KOKORO_CONVONLY_ROOT: /opt/kokoro-dualbucket/rk3588
      KOKORO_DUALBUCKET_MANIFEST_SHA256: cfc6aa8d485415884e4cf5eac47b0a786889ed02d9556a1051ed66685927083d
      KOKORO_CONVONLY_MANIFEST_SHA256: cfc6aa8d485415884e4cf5eac47b0a786889ed02d9556a1051ed66685927083d
      KOKORO_FRONTEND_INTRA_OP_THREADS: "4"
      KOKORO_FRONTEND_INTER_OP_THREADS: "1"
      KOKORO_DEFAULT_VOICE: af_heart
      KOKORO_VOICE: af_heart
      KOKORO_JA_DICDIR: /opt/resources/ja/unidic-lite-1.0.8
      OVS_TTS_MODEL_ID: kokoro-convonly-v1_0-rk3588
      RK_ARTIFACT_AUTO_DOWNLOAD: "0"
    volumes:
      - ${KOKORO_DUALBUCKET_HOST_ROOT:-/opt/kokoro-dualbucket}:/opt/kokoro-dualbucket:ro
      - ${KOKORO_JA_HOST_ROOT:-/opt/resources/ja/unidic-lite-1.0.8}:/opt/resources/ja/unidic-lite-1.0.8:ro
```

---

## 6. Runtime Configuration & Tuning

| Environment Variable | Recommended Value | Description |
|----------------------|-------------------|-------------|
| `TTS_BACKEND` | `kokoro_dualbucket` | Selects the dual-bucket streaming execution backend |
| `KOKORO_FRONTEND_INTRA_OP_THREADS` | `4` | Number of threads for ONNXRuntime text frontend |
| `KOKORO_FRONTEND_INTER_OP_THREADS` | `1` | Inter-operator thread count |
| `KOKORO_DEFAULT_VOICE` | `af_heart` | Default voice preset (53 voices available) |
| `KOKORO_DUALBUCKET_ROOT` | `/opt/kokoro-dualbucket/rk3588` | Mount path inside container |

---

## 7. Performance Benchmarks (Seeed reComputer RK3588)

Measured on Seeed reComputer (RK3588) (4x A76 @ 2.4GHz + 4x A55 @ 1.8GHz, NPU 6 TOPS):

| Metric | Measured Value | Standard / Condition |
|--------|----------------|----------------------|
| **TTFT (Time-To-First-Token)** | **383 ~ 414 ms** | First audio packet delivered to client |
| **Streaming RTF** | **0.380 ~ 0.424** | 50-token streaming chunks |
| **Acoustic Vocoder Engine** | `generator_T50.rknn` | NPU FP16 inference |
| **Peak Memory Consumption** | ~680 MB RSS | Container RSS after full model init |

---

## 8. Verification & Smoke Testing

### Health Probe

```bash
curl -f http://127.0.0.1:8621/readyz
# Expected: HTTP 200 {"status":"ok","profile":"rk3588-kokoro-dualbucket"}
```

### Batch TTS Verification

```bash
curl -X POST http://127.0.0.1:8621/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, welcome to OpenVoiceStream with Kokoro Dual-Bucket on RK3588.", "voice": "af_heart"}' \
  --output test_tts.wav

# Verify audio integrity
file test_tts.wav
# Expected: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 24000 Hz
```

### Streaming TTS Verification

```bash
curl -X POST http://127.0.0.1:8621/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Streaming synthesis delivers audio in real-time sliding windows.", "voice": "af_heart"}' \
  --output - | wc -c
# Expected: > 48000 bytes raw PCM stream
```

---

## 9. Rollback & Troubleshooting

- **Manifest Check Failure**:
  If container logs report `Manifest mismatch` or `FileNotFound`:
  ```bash
  sha256sum /opt/kokoro-dualbucket/rk3588/kokoro_dualbucket.manifest.json
  ```
  Ensure SHA matches `cfc6aa8d485415884e4cf5eac47b0a786889ed02d9556a1051ed66685927083d`.
- **NPU Device Not Found**:
  Verify driver on host:
  ```bash
  dmesg | grep -i rknpu
  ls -l /dev/rknpu
  ```
- **Rollback to Default RK3588 Profile**:
  ```bash
  docker compose -f deploy/docker-compose.rk3588.yml down
  docker compose -f deploy/docker-compose.rk3588.yml up -d
  ```

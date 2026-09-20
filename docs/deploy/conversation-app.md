# Deploy: Interruptible Conversational Voice AI

End-to-end bring-up for the conversation app on the two supported device
targets. Self-contained: everything referenced lives in this repo.

| Target | Compose | Speech | LLM |
|--------|---------|--------|-----|
| RK3588 + RK1828 PCIe NPU card | `deploy/docker-compose.conversation-rk3588-rk1828.yml` | Qwen3-ASR + matcha TTS on the RK3588 NPU (`:8621`) | Qwen3-4B on the RK1828 card (`:1828`) |
| Jetson Orin NX (JP 6.2 / TRT 10.3) | `deploy/docker-compose.conversation-orin-nx.yml` | Qwen3-ASR + matcha TTS on CUDA (`:8621`) | edge-llm Qwen3.5-4B GDN-MTP (`:8000`) |

Both composes run four services: `profile-init` (one-shot), `speech`, `llm`,
`agent` (`ovs-agent run conversation`).

## Prerequisites

* **Audio**: a reSpeaker XVF3800 mic array (USB) plus a speaker on the device's
  USB/3.5 mm output. The agent mounts `/dev/snd` and uses a wildcard ALSA
  cgroup rule so USB audio stays hot-pluggable. `audio_input_device: auto`
  prefers hardware-AEC reSpeaker devices and excludes HDMI/virtual nodes; the
  XVF3800 channel layouts are covered by
  `deploy/conversation/audio_profiles.yaml` (6-channel Flex and 4-mic 2-channel
  variants, with per-layout makeup gain).
* **RK1828 card (RK target only)**: the driver, the EP firmware and the
  userspace runtime live on the HOST, not in any image. One script checks all of
  it, and installs or persists what it can:

  ```bash
  sudo deploy/scripts/rk1828-host-bringup.sh --check   # read-only verification
  sudo deploy/scripts/rk1828-host-bringup.sh          # also load/persist/install
  sudo deploy/scripts/rk1828-host-bringup.sh --sdk ~/RK182X_RM182XMC0
  ```

  It covers the card on the bus, the `pcie_rkep` module (which does **not**
  persist across reboot), the EP firmware, `rknn3.service`, the `/dev/pcie-rkep-*`
  char device, and the userspace runtime — `librknn3_api{,_rkcp}.so` plus
  `rknn3_transfer_proxy`, printing their version. When the runtime is missing it
  installs `rknn3-rk182x-m2` from apt, or runs the vendor installer given
  `--sdk`.

  The version matters: the LLM container mounts the host's `/usr/lib` read-only
  and copies that client lib over its bundled one, because a client from a
  different generation than the host's proxy makes model init **block with no
  error** until the READY timeout. See *Runtime alignment* in
  `services/rk1828-llm/BUILD.md` for the signature and the 2026-09-17 case.
* **Orin NX**: JetPack 6.2 with the NVIDIA container runtime; the compose
  bind-mounts the host CUDA/TensorRT libraries read-only.
* Docker Engine + the compose plugin on the device.

## Bring up

RK3588 + RK1828:

```bash
docker compose -f deploy/docker-compose.conversation-rk3588-rk1828.yml up -d
```

Jetson Orin NX:

```bash
docker compose -f deploy/docker-compose.conversation-orin-nx.yml up -d
```

First start pulls models into named volumes (`rk-asr-models`,
`rk-tts-models`, `rk1828-llm-models` / `speech-models-v091`,
`edge-llm-models-v091`); the LLM healthchecks allow a 900 s start period for
the artifact download. `HF_ENDPOINT` defaults to `https://hf-mirror.com`.

### What `LANGUAGE` does

`LANGUAGE` (default `zh`) is the single operator input. The one-shot
`profile-init` service runs `tools/resolve_profile.py` against
`configs/matrix/language_device.yaml` and resolves `(LANGUAGE, device)` to
exactly one `OVS_PROFILE`, written to the `resolved-profile` volume and
sourced by `speech` before uvicorn starts. The agent independently receives
`ASR_LANGUAGE`/`TTS_LANGUAGE` from the same `LANGUAGE` value and sends them
as the per-session v2v config.

The resolver **refuses unsupported pairs**: exit 2 (e.g. a language the board
has no cell for) happens before any service starts, and
`depends_on: service_completed_successfully` propagates the failure so the
whole `up` stops. That is deliberate — a language the board cannot serve must
fail visibly instead of being transcribed by a model that cannot read it.
`untested` cells pass with a warning on stderr; `measured` cells pass
silently. Check with
`docker compose -f <file> logs profile-init` /
`docker inspect conversational-voice-profile-init --format '{{.State.ExitCode}}'`.

## Wake word

Set in `.env` next to the compose file (or the environment):

```
PIPELINE_MODE=wake_word
WAKEWORD_BACKEND=sherpa_onnx        # default in both composes
WAKEWORD_PHRASE=你好小智             # default; change to taste
WAKEWORD_THRESHOLD=0.25
WAKEWORD_MIC_SKIP_MS=120
```

The sherpa-onnx open-vocabulary KWS model
(`sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20`) is **bundled in the agent
image** under `/opt/ovs/models/` — selecting `wake_word` mode needs no image
rebuild and no model download. With `PIPELINE_MODE=always_on` (the compose
default) no wake word is required.

Confirm the runtime source registered:

```bash
docker logs conversational-voice-agent 2>&1 | grep "RuntimeKwsSource ready"
# expect: RuntimeKwsSource ready: 1 phrase(s)   (N = number of phrases)
```

## Debug dashboard (`:18000`)

The agent's debug dashboard listens on port 18000. The composes default
`OVS_DEBUG_DASHBOARD_BIND` to `0.0.0.0` so a headless device is reachable from
the operator's browser at `http://<device-ip>:18000`. **The dashboard is
unauthenticated and can trigger control actions (abort, send-text)** — treat
it as trusted-networks-only. On an untrusted network set
`OVS_DEBUG_DASHBOARD_BIND=127.0.0.1` and use SSH port forwarding instead.

## Endpoint behaviour

There is exactly **one endpoint detector** per path (the image carries a
single-endpoint-detector guard; `OVS_V2V_SINGLE_ENDPOINT_STRICT=1` turns
violations into a hard reject).

* The RK image profile raised the backend's `VAD_ENDPOINT_SILENCE_MS` from
  400 ms to **1500 ms** because 400 ms split a natural 1.5 s mid-sentence
  pause into two utterances (measured on RK3588; 5000 ms did not split).
  The baked profile's own key is re-stamped at startup and cannot be set from
  compose, so the deploy-time override is the separate
  **`OVS_V2V_VAD_ENDPOINT_SILENCE_MS`** variable, consumed **per session** via
  the ASR stream options. Both composes pass it through with default `0`
  (= keep the image profile value). Set e.g.
  `OVS_V2V_VAD_ENDPOINT_SILENCE_MS=2000` in `.env` to override per session.
* Client-VAD relationship: the agent runs its own Silero client VAD
  (`client_vad_silence_ms: 600`, `client_vad_drive_eos: true`), so the client
  normally wins end-of-speech at 600 ms of silence. The backend threshold
  must stay **above** the client value (1500 > 600 ✓) or the backend wins the
  race and cuts utterances in half.

## Failure recovery

If the `speech` container restarts, the agent's link watchdog **reconnects on
its own** (observed ≈15 s on device). Do **not** press reconnect in the
dashboard and do not restart the agent — just wait. Note that while the agent
holds the single `/v2v` session, `/readyz` returns 503
`{"reasons":["sessions_full"]}`; that is expected, not a fault — use `/health`
to judge the speech service.

## Verification checklist

1. `docker compose -f <file> ps` — `profile-init` exited 0; `speech`, `llm`,
   `agent` running; `speech` and `llm` `(healthy)`.
2. Speech health: `curl -fsS http://127.0.0.1:8621/health` on the device —
   expect `asr` and `tts` reported ready/true and
   `runtime_profile.verified: true` (no contract drift). LLM health:
   `curl -fsS http://127.0.0.1:1828/health` (RK) or `:8000/health` (Orin).
3. Dashboard: open `http://<device-ip>:18000` — session state visible, agent
   connected to `ws://127.0.0.1:8621/v2v/stream`.
4. Wake word (if enabled): the `RuntimeKwsSource ready: N phrase(s)` log line
   above, then say the phrase and listen for the wake tone.
5. Audible round trip: speak a full sentence with a natural mid-sentence
   pause — expect ONE coherent spoken reply (no mid-sentence cut, no two LLM
   turns), and barge-in: start talking over the reply and expect it to abort.

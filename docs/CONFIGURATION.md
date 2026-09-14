# Configuration

> **TL;DR — the one path you should use:** pick a **profile** with `OVS_PROFILE`.
> A profile fully describes a device's ASR/TTS backends, model paths, and
> execution policy. You almost never set individual model env vars by hand —
> the profile sets them for you. Everything below is detail for when you need it.

## The model: profiles set env, env can override

Configuration is plain environment variables, but you rarely write them
directly. A **profile** (`configs/profiles/<name>.json`) is a bundle of env
defaults plus the backend selection. At startup `server/core/profile_loader.py`
applies the chosen profile's `env` block, then individual env vars you set
yourself take precedence.

```
 OVS_PROFILE=jetson-zh-en
        │
        ▼  profile_loader applies configs/profiles/jetson-zh-en.json
 ┌──────────────────────────────────────────────┐
 │ asr_backend / tts_backend  →  which voxedge   │
 │ env: { LANGUAGE_MODE, MODEL_DIR, … }          │
 │ execution_policy           →  coordinator mode │
 │ required_engines           →  artifact check   │
 └──────────────────────────────────────────────┘
        │
        ▼  operator-set env vars win (snapshotted at import)
 effective configuration
```

### Precedence (highest first)

1. **Operator/shell/`.env` env vars** — snapshotted at import as "operator-owned"
   and never overwritten by a profile (`_snapshot_operator_keys`).
2. **Profile `env` block** — applied for any key the operator did not set.
3. **Hardcoded defaults** in the backend modules.

> **Guardrail:** a handful of *steering* keys (the ones that pick a backend or
> model family) are treated as **hard-mismatch**: if your env disagrees with the
> selected profile, the server fails loudly at boot rather than silently letting
> "operator wins" pick a broken combo. This is intentional — it prevents
> "profile says Qwen3 ASR but env forced Paraformer paths" foot-guns.

## Choosing a profile

```bash
OVS_PROFILE=jetson-zh-en        # selection: file stem under configs/profiles/
# or OVS_PROFILE_JSON=/abs/path/to/custom.json for an out-of-tree profile
```

The compose files in `deploy/` set `OVS_PROFILE` per device. The full list of
shipped profiles is in the README; a profile looks like:

```json
{
  "name": "jetson-zh-en",
  "asr_backend": "jetson.paraformer_trt",
  "tts_backend": "jetson.matcha_trt",
  "execution_policy": { "mode": "serialized", "shared_resource": "gpu" },
  "env": { "LANGUAGE_MODE": "zh_en", "MODEL_DIR": "/opt/models", … },
  "required_engines": [ … ]   // checked / auto-downloaded at boot
}
```

To add a device variant, copy the closest profile and edit `env` +
`{asr,tts}_backend`. You do not touch Python.

## Operator-facing env vars

These are the knobs you might actually set. **Everything else** (the
`PARAFORMER_*`, `MATCHA_*`, `MOSS_*`, `RK_ARTIFACT_*`, `SENSEVOICE_*`, … model
paths) is **profile-managed — leave it alone** unless you are authoring a
profile.

| Var | Purpose | Typical |
|---|---|---|
| `OVS_PROFILE` | Select a device profile (file stem) | `jetson-zh-en` |
| `OVS_PROFILE_JSON` | Select an out-of-tree profile by path | _(unset)_ |
| `LANGUAGE_MODE` | Language family | `zh_en`, `en`, `multilanguage` |
| `MODEL_DIR` | Root for downloaded model artifacts | `/opt/models` |
| `OVS_MAX_CONCURRENT_SESSIONS` | Session admission ceiling | `1`–`2` |
| `OVS_V2V_ENGINE` | `/v2v` orchestrator: `voxedge` enables the library engine | `voxedge` |
| `OVS_V2V_SERVER_LOOP` | Run the LLM+tool loop server-side (needs `OVS_V2V_ENGINE=voxedge`) | `1` |
| `OVS_AUTO_DOWNLOAD_ARTIFACTS` | Fetch missing engines/models at boot | `1` |
| `HF_ENDPOINT` | HuggingFace mirror (devices behind GFW) | `https://hf-mirror.com` |
| `LAZY_TTS` | Defer TTS model load until first request | `0`/`1` |
| `EDGE_LLM_BASE_URL` | External LLM service for the conversation loop | host:port |

> **The two flags that trip everyone up:** the server-loop voice path needs
> **both** `OVS_V2V_ENGINE=voxedge` **and** `OVS_V2V_SERVER_LOOP=1`. One without
> the other silently falls back to the legacy path.

## Streaming ASR endpointing: pick exactly one detector

`/asr/stream` supports two operating modes. **They are mutually exclusive.**
Running both at once is the single most expensive misconfiguration in this API —
it does not degrade gracefully, it silently truncates transcripts.

### Mode A — open-mic (server decides when the utterance ended)

```
ws://host:8621/asr/stream?vad_silence_ms=900
```

The server runs its own VAD. On `SPEECH_END` it sends `{"type":"vad_endpoint"}`,
then a `{"type":"final", "endpoint":"vad", ...}` for that segment, then
**replaces the ASR stream with a fresh one and keeps the socket open** for the
next utterance (`server/main.py:4899-4947`). One connection carries many
utterances; the client never sends the EOS frame.

Use this when the client is a dumb audio pipe — a browser page, a live-caption
demo, anything with no VAD of its own.

### Mode B — client-driven endpointing (client decides)

```
ws://host:8621/asr/stream?vad=none
```

No server VAD is created (`main.py:4686`, `core/vad.py:250-252`). The server
finalizes **only** when the client sends the empty `b""` frame.

Use this when the client already runs a VAD — because it usually knows things
the server cannot: whether a wake word just fired, whether the device is playing
TTS, what conversational state the session is in.

### Why running both breaks, and what it looks like

The two detectors race. Whichever fires first wins, and when the server wins it
**starts a new segment on its side** while the client still believes one
utterance is in progress. From then on the two disagree about which segment is
current, and text goes missing. Which end goes missing depends on how the client
treats the server's mid-utterance `final`:

| Client behaviour on server-issued `final` | Symptom |
|---|---|
| Honours it as "utterance over" | Cut short — head kept, tail dropped |
| Ignores it but **overwrites** its buffer | Head dropped, only the last segment survives |
| Ignores it and **accumulates** it | Correct |

Measured on real hardware (2026-08-06, Jetson Orin NX, Qwen3-ASR streaming),
with a 1.5 s pause inserted mid-sentence:

```
server VAD on   → '帮我查一下M6。'                 (cut at the pause)
?vad=none       → '帮我查一下M6螺母的库存。'        (complete)
```

A worse variant from the same session: 4.32 s of audio arrived byte-for-byte
intact (verified by dumping exactly what was written to the socket and diffing
against the client's own buffer), replaying those same bytes offline yielded
`'帮我查一下 SKU002 的库存。'` — yet the live session returned `'嗯。'`, because
every earlier segment had been overwritten. Nothing logged an error on either
side.

Raising `vad_silence_ms` does **not** fix this. It only makes the server slower
to fire, so the race is lost less often. Turn the server VAD off instead.

### If you must run Mode A with a VAD-capable client

Accumulate every `final` the server sends; do not overwrite. Each one is a
complete segment, and the server has already moved on. `{"type":"vad_endpoint"}`
arrives *before* the finalize compute, so it is also the right signal for
flipping client UI state without waiting for ASR.

### The RK Qwen3 backend always endpoints — so its threshold must lose the race

`vad:"none"` only disables the **engine's** VAD. The RK Qwen3-ASR backend runs
its own webrtcvad endpoint in **both** `true_streaming` and `chunk_confirm`
(measured 2026-09-13), and it re-opens a fresh segment when it fires. There is no
per-session override plumbed through the engine, so the profile's
`VAD_ENDPOINT_SILENCE_MS` is the only lever.

Measured on RK3588 with `bench/parity/v2v_wav_inject.py` — one 6.33 s clip
containing a 1.5 s mid-sentence pause (TTS halves spliced with silence):

| `VAD_ENDPOINT_SILENCE_MS` | Result |
|---|---|
| 400 ms (old default) | finalize at 2.00 s of audio — **first half only** (`帮我查一下 N64。`), rest arrives as a new segment |
| 1200 ms | finalize at 2.80 s — still cut at the pause (threshold below the pause length) |
| 5000 ms | one 6.80 s segment, a single final for the whole clip |

With the RK profiles at 1500 ms the shipped agent (client VAD
`client_vad_silence_ms: 600`, `client_vad_drive_eos: true`) wins deterministically:
the client decides the utterance boundary and the backend endpoint is only a
fallback. Clients can also pass `vad_endpoint_silence_ms` in the v2v config to
override the profile value for one session (`docs/api/v2v-stream.md`), which is
the right lever when only some clients need a longer threshold.
**Invariant:** `VAD_ENDPOINT_SILENCE_MS` must exceed the client VAD
silence plus the EOS→finalize budget; a client that relies on the server's
endpoint instead (dumb audio pipe, no VAD of its own) pays the full threshold as
trailing dead air before the reply starts — give such a client its own VAD.

Client-side, honouring a backend-issued `final` as "utterance over" (or
overwriting the buffer with it) truncates: the production agent used to overwrite
`_last_user_utterance_text` per final, which is exactly the "head kept, tail
dropped" row of the table above. Accumulate instead.

## Leaf composition (optional, opt-in — ignore unless you need it)

There is a second, newer config layer under `configs/leaves/` ("leaf
composition": `devices.yaml`, `models.yaml`, per-device ASR/TTS memory budgets).
It composes a backend stack from `capability × backend × device × concurrency`
and validates the result fits the device memory budget.

**It is OFF by default.** Flat profiles are the supported path. Leaf composition
only activates when a profile opts in (a `composition` key) and is gated by
`server/core/composition_boot.py`. If you are not deliberately working on it,
treat `configs/leaves/` as not-present. It exists to eventually replace
hand-maintained per-device memory tuning; until then, **do not mix the two** —
pick a flat profile.

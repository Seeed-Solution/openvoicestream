# External platform API v1 and voice capability contract

Status: independently reviewed; implementation is phased
Target branch: `codex/edgellm-v091-upstream-audit`

## Goal

Make the active ASR/TTS service self-describing and safe to integrate from an
external platform while preserving every existing native route and wire
format. Add a deliberately small OpenAI-compatible audio adapter on top of the
same internal execution paths.

## Scope

This change includes:

1. Correct Qwen3-TTS Base and CustomVoice default-speaker resolution.
2. Describe and enforce the distinct clone contracts used by Qwen Base,
   MOSS-TTS and SparkTTS.
3. Persist registered speakers and voice profiles across container replacement.
4. Add a versioned, machine-readable ASR/TTS capability schema.
5. Add native `/v1` aliases and OpenAI-compatible audio/model routes.
6. Add profile-level contract and device-level regression tests.

This change explicitly excludes gateway/TLS, API-key policy changes, CORS and
new rate-limiting behavior. Existing authentication and admission controls are
reused without changing their defaults.

Implementation is split so the existing clone-contract ambiguity cannot block
the basic platform API:

- Phase A: model/default-speaker correctness, structured capabilities,
  persistent voice data and profile/package truthfulness.
- Phase B: strict native HTTP v1 plus OpenAI-compatible speech,
  transcriptions and model listing.
- Phase C: modality-correct clone/enrollment endpoints and clone-stream
  cancellation. Legacy clone routes remain available throughout.

## Compatibility rules

- Existing `/tts`, `/tts/stream`, `/tts/clone*`, `/asr`, `/asr/stream` and
  `/v2v/stream` routes retain their current request and response contracts.
- Existing flat keys in `/tts/capabilities` and `/asr/capabilities` remain;
  structured fields are additive.
- `/tts/stream` keeps its four-byte sample-rate prefix. OpenAI-compatible
  endpoints never expose that private framing.
- Unsupported controls are rejected explicitly in `/v1`; they are not silently
  ignored and are never advertised as supported.
- The adapter lists and accepts only the currently configured ASR/TTS models.
  A profile present on disk is not advertised as live merely because it exists.

## 1. Model and voice contract

### Files

- `server/core/tts_speakers.py:33-185,269-395`
- `server/core/tts_runtime.py:154-190`
- `server/main.py:251-336,1449-1497`
- `configs/leaves/models.yaml:8-67`
- `server/core/leaf_composition.py:116-123,299-307`

### Resolution order

The effective selector remains:

`request > runtime override > model default > backend intrinsic default`.

`SpeakerType` becomes `preset | embedding | intrinsic`. `intrinsic` always
translates to `{}`. The resolved speaker is allowed to be `None`. Model aliases and defaults are
resolved before a selector is translated into backend kwargs. Explicit `/v1`
selectors are strict; unknown ids/names return an error. Legacy native routes
retain the existing permissive escape hatch for unmapped numeric ids.

The strict v1 resolver is a separate entry point and never consults
`OVS_TTS_ALLOW_UNMAPPED_SPEAKER_ID`. It accepts a numeric JSON id, a numeric
string, a canonical name or a declared alias. Omitted, `null` and an empty
string select the model default. Registered embeddings and Spark profiles must
match the active canonical model/backend or return 404.

### Required model definitions

- `qwen3-tts-0.6b-base`: expose public speaker id `0`, type `intrinsic`, label
  `Default reference voice`, and translate it to `{}`. This keeps discovery
  useful while allowing `EDGE_LLM_TTS_BASE_SPK_EMBED_PATH` to supply the actual
  fixed embedding. It must never become `speaker_id=0,speaker="0"` at the
  backend boundary.
- `qwen3-tts-customvoice`: expose all nine existing built-ins and use a real
  canonical default. The initial product default is `3065` (`vivian`); profile
  or runtime configuration may override it. Legacy selector `0` is a deprecated
  alias for that canonical id rather than a value sent to the worker.
- `moss-tts-nano-v1`: no preset speaker; omitted selector remains `None`.
- SparkTTS id `0` continues to mean `female_moderate_moderate`.

The sparse model catalog gains `qwen3-tts-0.6b-base` and optional metadata,
while old precision-only entries remain valid. Static catalog data may label a
model and its selectors, but runtime backend properties remain authoritative
for readiness, sample rate, clone availability and concurrency.

## 2. Clone contract

### Files

- `server/core/tts_backend.py:21-29,43-123`
- `server/core/tts_service.py:60-90`
- `server/core/sparktts_voices.py:1-288`
- `server/main.py:188-201,1575-1680,2594-2877`
- optional new `server/core/tts_contracts.py`

The public capability schema describes clone modes independently:

- Qwen3 Base: consumes `embedding`; enrollment from reference audio is exposed
  only when the active backend reports a usable on-device extractor.
- MOSS-TTS: consumes `reference_audio`; it does not expose reusable speaker
  embeddings.
- SparkTTS: consumes a registered `voice_profile`; profile upload is an
  enrollment method, while host reference-audio enrollment is advertised only
  when its analyzer is available.
- CustomVoice, Matcha, Kokoro, Sherpa and RK adapters advertise only modes their
  active runtime really implements.

Phase C uses modality-specific routes rather than putting large reference audio
inside an ambiguous JSON union:

- `POST /v1/tts/clone/embedding` and `/stream`: JSON float32 little-endian
  embedding for Qwen Base; optional `dim` must match decoded bytes.
- `POST /v1/tts/clone/reference` and `/stream`: multipart PCM16 WAV for MOSS.
  The server validates WAV, strips its header and forwards real
  `reference_audio` plus sample rate. The accepted channel count and exact
  sample rate come from the active codec contract; mismatches return 400 rather
  than being reinterpreted. No automatic resampling/downmix is claimed until it
  is implemented and device-validated.
- Spark profile synthesis uses strict `voice=<voice_id>` through `/v1/tts` or
  `/v1/audio/speech`. Profile upload/enrollment remains a separate operation.

The legacy `speaker_embedding_b64` field remains for compatibility, but
capabilities mark its model-specific deprecated meaning. Spark's raw
`/tts/clone` path must return an honest capability error instead of reaching an
unimplemented backend method. MOSS may continue accepting the legacy field as
a deprecated reference-PCM alias.

The Phase C clone streaming implementation must use the same disconnect watcher,
cancel token, executor drain and lease-release helpers as `/tts/stream`.
Cancellation tests cover pre-first-chunk, post-first-chunk and queued-not-started
cases in both manager and legacy paths; they assert the worker terminal cancel
event/counter and executor completion, not only HTTP slot release.

## 3. Persistent data

### Files

- `deploy/docker-compose.yml`
- `deploy/docker-compose.edgellm-v091-voice.yml`
- `deploy/docker-compose.rpi.yml`
- `deploy/docker-compose.rk.yml`
- `deploy/docker-compose.rk3588.yml`
- `deploy/spark/docker-compose.spark.yml`
- `server/core/tts_speakers.py:1-12`
- `server/core/sparktts_voices.py`

Mount a dedicated named volume with external name `seeed-local-voice-data` at
`/opt/seeed-local-voice/data`. Both
`speakers.json` and `sparktts_voices/` live below this one root. Model artifacts
remain in `speech-models`; user-created voice data is not mixed with models.

Container replacement must preserve registered data. First startup creates the
directory when the volume is empty; an existing bind-mounted data directory is
not migrated or overwritten. Image/compose paths are explicit and writable by
the runtime user. Tests use temporary directories and a two-process/reload
round trip, including atomic JSON/NPZ pair replacement, half-written pairs,
safe-id collision rejection and delete failures. Deployment verification uses
a cryptographically random marker after confirming it does not exist and
deletes both files after the check. It never removes the shared volume.

The unused v0.9.1 `EDGE_LLM_TTS_SPK_ENCODER_*` runtime requirement is removed
from v0.9.1 profiles and image invariants. There is no replacement enrollment
artifact in the formal v0.9.1 bundle: its honest production capability remains
`supports_voice_enrollment=false`. The active backend enables optional CPU-ONNX
enrollment only when `QWEN3_SPEAKER_ENCODER` points to an existing
`speaker_encoder.onnx` (or the documented
`$QWEN3_ARTIFACT_ROOT/tts/speaker_encoder/speaker_encoder.onnx` probe exists);
that path sets the backend's `supports_voice_enrollment` property. It is an
operator-supplied optional sidecar, not the removed TensorRT engine.

The v0.9.0 rollback profiles are unchanged. The TensorRT speaker-encoder engine
may remain in the artifact bundle for external tools, but is not required by or
advertised as an active service enrollment capability. Removal is accepted only
after three separate checks: the Base fixed embedding file at
`EDGE_LLM_TTS_BASE_SPK_EMBED_PATH` remains present and is actually selected for
default synthesis; a cold formal-image start reaches ready and synthesizes;
and optional enrollment flips false to true in a test only when a valid ONNX
sidecar path exists.

## 4. Capability schema

### Files

- new `server/core/api_capabilities.py`
- `server/main.py:1433-1497`
- `server/core/concurrency_capability.py`

`GET /v1/capabilities` is a versioned Pydantic response and returns HTTP 200 for
ASR-only, TTS-only, lazy, not-ready and failed states. A configured component is
present with `ready=false` and a non-secret failure class; the legacy component
capability routes retain their existing 503 behavior. It returns:

```json
{
  "object": "capabilities",
  "schema_version": "1.0",
  "api_versions": ["legacy", "v1"],
  "tts": {
    "model_id": "qwen3-tts-0.6b-base",
    "ready": true,
    "audio": {
      "sample_rate": 24000,
      "channels": 1,
      "sample_format": "pcm_s16le",
      "response_formats": ["wav", "pcm"]
    },
    "languages": {"mode": "multi_language", "values": null, "default": "auto"},
    "voices": {"default": {"speaker_id": 0, "source": "backend_intrinsic"}, "items": [{"id": 0, "type": "intrinsic", "label": "Default reference voice"}]},
    "controls": {
      "speed": {"supported": true, "min": 0.25, "max": 4.0, "implementation": "dsp"},
      "pitch": {"supported": true, "min": -24, "max": 24, "unit": "semitone", "implementation": "dsp"}
    },
    "cloning": {"supported": true, "modes": ["embedding"], "enrollment": {"supported": false, "methods": []}},
    "streaming": {"supported": true, "native_wire_format": "u32le_sample_rate+pcm_s16le"},
    "concurrency": {"backend_max_concurrent": 1, "admission_limit": 1, "active": 0, "available": 1}
  },
  "asr": {}
}
```

Unknown language lists are `null`, not invented. Backend-native versus DSP
controls are explicit. Spark adds its discrete style dimensions and selector
format. Clone booleans in legacy responses are derived from the structured
contract to avoid multiple sources of truth.

`api_versions` is generated from routes that are actually registered. Phase A
returns `legacy` and `v1` only. `openai-audio` is added only after all Phase B
routes and their contract tests pass; Phase C clone routes do not retroactively
change this field.

`backend_max_concurrent` comes from the active backend capability;
`admission_limit`, `active` and `available` come from the effective session
limiter after profile/env clamping. Schema fixtures freeze required, nullable
and enum fields for Base, CustomVoice, MOSS and Spark.

## 5. Versioned native and OpenAI-compatible APIs

### Files

- new `server/api/openai_compat.py` if callbacks can be injected cleanly;
  otherwise a tightly scoped router section in `server/main.py`
- `server/main.py:1685-1759,2882-2953`
- new `server/tests/test_openai_compat.py`

Native aliases in the first batch:

- `GET /v1/capabilities`
- `GET /v1/tts/capabilities`
- `GET /v1/tts/speakers`
- `POST /v1/tts`
- `POST /v1/asr`

Phase C adds the four modality-specific clone routes defined above and the
versioned voice profile operations. It is not implied by the Phase B aliases.

Operations/admin endpoints and probes are not versioned. WebSocket aliasing is
deferred until the HTTP contract is stable; the existing realtime endpoint is
already versioned by the `seeed.realtime.v2` subprotocol.

OpenAI-compatible routes:

- `POST /v1/audio/speech`: requires `model` and `input`; `voice` omitted/null/
  empty selects the model default. It accepts `speed` in `[0.25,4.0]` and
  `response_format`, whose service default is `wav`. Voice may be a numeric id,
  numeric string, canonical name, declared alias, Spark style or compatible
  registered profile. `wav` and headerless `pcm` are supported. Unsupported
  encodings such as mp3/opus/aac/flac return a structured 400 rather than WAV
  bytes under a false content type. The request must name the active model or a
  declared alias. Voice resolution is strict. When the active backend exposes
  streaming, this same route returns an HTTP chunked `StreamingResponse` as
  soon as the first PCM chunk is ready; there is no private `/stream` route.
  `pcm` is headerless PCM16. `wav` starts with a legal PCM WAV header whose
  RIFF/data lengths use the conventional unknown-length sentinel, so clients
  can begin playback before synthesis completes. Errors before the first
  header use the OpenAI JSON envelope; a backend error after bytes have started
  safely terminates the stream because HTTP status is already 200.
- `POST /v1/audio/transcriptions`: requires multipart `file` and `model`;
  omitted/empty language maps to `auto`. `response_format` defaults to `json`.
  JSON is exactly `{"text":"..."}`; text is UTF-8 `text/plain`. Empty prompt
  and temperature `0` are accepted as no-ops; non-empty prompt, nonzero
  temperature, `timestamp_granularities[]`, `verbose_json`, `srt` and `vtt`
  return 400 until implemented.
- `GET /v1/models`: returns OpenAI list shape and only the configured ASR/TTS
  canonical model ids. Each item contains `id`, `object="model"`, `created=0`,
  `owned_by="seeed-studio"` plus modality/backend/readiness metadata. Aliases
  are metadata, not duplicate rows; one id with two modalities is one row whose
  modality is an array.

Before adding adapters, split transport-neutral `_execute_tts` and
`_execute_asr` cores from the existing handlers. They return typed audio/
transcription results and raise typed domain errors. Legacy, native v1 and
OpenAI serializers call those cores through one admission-aware wrapper. This
prevents duplicate coordinator/session ownership. OpenAI streaming reuses the
native streaming generator, disconnect watcher and cleanup path, translating
only the private four-byte sample-rate prefix into headerless PCM or a
streamable WAV container.

Errors use the OpenAI envelope:

```json
{"error":{"message":"...","type":"invalid_request_error","param":"voice","code":"unsupported_voice"}}
```

Use 400 for unsupported controls/format, 404 for unknown model or voice, 413
for configured payload limits, 429 for existing admission saturation, and 503
for a backend that is not ready. Existing `X-Request-ID` behavior is retained.
422 request/multipart validation, authentication failures and unexpected domain
errors are normalized to the same envelope on `/v1/audio/*`; `Retry-After`,
`WWW-Authenticate` and `X-Request-ID` survive normalization.

New v1 uploads use a bounded reader with `OVS_API_MAX_AUDIO_BYTES` (default
32 MiB) and `OVS_API_MAX_PROFILE_BYTES` (default 16 MiB); text defaults to
64 KiB UTF-8. Limits count decoded bytes, do not trust `Content-Length`, and
return 413 once the streaming reader crosses the boundary. Strict base64 decode
uses validation and applies the decoded-byte limit. Legacy limits are unchanged
in Phase B to preserve compatibility; Phase C applies the same helper to every
new clone/profile upload.

## 6. Validation

### Unit and contract tests

- `server/tests/test_tts_speakers.py`: Base intrinsic selector produces `{}`;
  CustomVoice omitted/legacy/default selectors resolve to the canonical voice;
  Spark/Kokoro/Matcha/Sherpa mappings do not regress.
- `server/tests/test_tts_runtime.py` and `test_main_hot_swap.py`: four-level
  precedence, nullable speaker, strict v1 selectors and both manager paths.
- new `server/tests/test_clone_contract.py`: Qwen embedding, MOSS reference
  audio, Spark voice profile, invalid/multiple inputs and clone stream cancel.
- new `server/tests/test_capability_schema_v1.py`: legacy additive compatibility
  and Base/CustomVoice/MOSS/Spark structured truth.
- new `server/tests/test_openai_compat.py`: speech/transcriptions/models success,
  WAV/PCM bytes, JSON/text responses, error envelopes, 429/503, and unchanged
  legacy route responses.
- profile/packaging tests: v0.9.1 contains no unconsumed speaker-encoder runtime
  requirement; v0.9.0 rollback retains it; every speech compose mounts data.
- frozen legacy fixtures: capability/speaker bodies, clone capability errors,
  WAV headers and the four-byte native stream prefix.
- auth parity: disabled/enabled/missing/wrong/correct keys behave identically
  between native and v1 routes; no new CORS middleware exists and admission
  clamp/limits are unchanged.

### Device tests

Required gates are explicit: Base runs unit, profile-contract and Orin NX real
device gates; CustomVoice, MOSS and Spark run unit/profile-contract gates and a
real-device gate when their artifact profile is staged. A release claiming
those profiles as deployable is blocked until its real-device row passes; an
unstaged optional profile is reported `not_run`, never silently skipped.

For each required production profile:

1. Read capabilities and enumerate every declared built-in voice/style.
2. Synthesize a short deterministic text with the default and each selector.
3. Verify speed 0.5/1/2 duration ordering and pitch -6/0/+6 distinct output.
4. Exercise only the clone mode advertised by that profile; MOSS does not run
   embedding extraction/registration and Spark does not run raw embedding clone.
5. Disconnect native and clone streams after first PCM and verify slot release,
   cancellation count and immediate recovery.
6. Run declared N-way overlap plus one over-capacity request.
7. Recreate the container and verify a disposable registered profile survives.
8. Confirm legacy clients and `/v1/audio/*` produce valid audio/transcripts.

The scripted voice-agent regression keeps three layers of assertions: state,
protocol event and captured audio/data. It covers single turn, multi-turn,
barge-in, empty final, reconnect, idle stability and error recovery without a
physical microphone or speaker.

## Acceptance criteria

- No legacy response field or wire format is removed or renamed.
- Base default synthesis does not send `speaker_id=0` to the backend and uses
  the configured fixed embedding path.
- CustomVoice default synthesis sends a valid canonical built-in selector.
- Capability output fully describes voice selectors, speed/pitch behavior,
  audio format, clone mode and active concurrency.
- Spark raw embedding clone cannot reach an unimplemented method; MOSS is never
  described as embedding clone.
- Registered speaker/voice data survives container recreation.
- OpenAI-compatible WAV speech, PCM speech, JSON transcription, text
  transcription and models-list tests pass; unsupported formats fail honestly.
- Existing unit suite and the required backend/profile matrix pass; every
  executed device row has zero service restarts or CUDA/TensorRT errors.
- A changed-file allowlist review confirms no gateway/TLS/auth-policy/CORS or
  limiter-policy files changed. New routes all reuse `_require_api_key`; session
  limits and clamps are byte-for-byte unchanged apart from new route labels.

## Guardrails

- Do not change authentication defaults, CORS, TLS, gateway configuration or
  admission-limit policy.
- Do not change TensorRT engines, worker binaries or upstream patch series.
- Do not edit v0.8/v0.9.0 rollback behavior except tests that prove it remains.
- Do not add media encoders or claim formats/timestamps/languages that are not
  actually implemented.
- Do not stop or replace the formal Orin NX service until local tests pass and
  the replacement image has a verified rollback tag.

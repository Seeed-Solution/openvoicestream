# Legacy V1 `WS /v2v/stream` protocol

> This document describes the migration-only V1 dialect used when the client
> does not request a WebSocket subprotocol. New applications use
> `seeed.realtime.v2`; see [Realtime V2](realtime-v2.md). Provider switching
> and new application work must not depend on the events documented below.

Single bi-directional WebSocket. The first JSON frame the client sends
is a `config` that decides which features light up:

| Config key | Effect |
|---|---|
| `asr_language` set | ASR is enabled. Client may send PCM binary frames; server emits `asr_partial` / `asr_endpoint` / `asr_final` JSON. |
| `tts_language` set | TTS is enabled. Client may send `text` JSON frames; server emits PCM binary chunks + `tts_started` / `tts_sentence_done` / `tts_done` JSON. Use `"auto"` to enable TTS while delegating language selection to the backend (qwen3 inspects the text; backends without auto-detect fall back to their default). |
| `tts_speaker_id` | Numeric speaker ID from `/tts/speakers`. Resolved through the model-scoped speaker registry. Prefer over `tts_voice`. Omit/`null` to use the model default. |
| `tts_voice` | (Deprecated) Backend-specific opaque speaker string. Replaced by `tts_speaker_id`. Accepted only when `tts_speaker_id` is absent. |
| Both set | Full V2V duplex. Binary in both directions: client → ASR input, server → TTS output. |
| `vad` | Server-side VAD backend (default `silero` if ASR enabled, `none` otherwise). |
| `vad_silence_ms` | How long silence to trigger auto `asr_endpoint`. Default is `OVS_VAD_SILENCE_MS` or 400 ms. This is the **engine** VAD (`vad` above), separate from a backend's own endpointing. |
| `vad_endpoint_silence_ms` | Optional per-session override of a backend's **own** endpoint threshold — for ASR backends that endpoint internally regardless of `vad` (RK Qwen3: both `true_streaming` and `chunk_confirm`). Milliseconds; absent or `0` keeps the profile value; out-of-range/unparseable values are ignored (the session still runs). Keep it above your client-side VAD silence, or the backend wins the race and cuts the utterance in half — see `docs/CONFIGURATION.md` "Streaming ASR endpointing". |
| `multi_utterance` | If `true`, the session stays open across utterances; each VAD/backend endpoint emits a mid-session `asr_final` with `session_complete: false` and the loop keeps listening. Default `false` (single-utterance, current behaviour). |

Existing `/asr/stream` and `/tts/stream` endpoints stay unchanged for
backward compatibility. The new endpoint adds capability without
breaking anything.

Deployment defaults:

```bash
OVS_VAD_BACKEND=silero
OVS_VAD_SILENCE_MS=400
```

Clients can still override per connection with `vad` and `vad_silence_ms`.

## Protocol

### Client → Server JSON message types

```
{"type":"config",
 "asr_language":"zh",          // omit to disable ASR
 "tts_language":"zh",          // omit to disable TTS
 "tts_speaker_id":2301,        // speaker ID from /tts/speakers (preferred)
 "tts_voice":"default",        // deprecated; prefer tts_speaker_id
 "tts_speed":1.0,              // optional; some backends only
 "sample_rate":16000,          // PCM sample rate
 "vad":"silero",               // "silero" | "webrtcvad" | "none"
 "vad_silence_ms":400,
 "vad_endpoint_silence_ms":1500, // optional: backend-owned endpoint VAD
 "multi_utterance":false}      // see "End-of-utterance semantics" below

{"type":"text", "text":"<incremental text chunk>"}
{"type":"asr_eos"}             // manually finalize ASR (overrides VAD)
{"type":"tts_flush"}           // flush remaining TTS buffer
{"type":"abort"}               // barge-in: cancel current TTS, drop queue

// Server-side tool loop only (see "Remote tools" below). Ignored — with a
// warning in the service log, no error frame — unless the deployment runs the
// server-side LLM loop (OVS_V2V_ENGINE=voxedge + OVS_V2V_SERVER_LOOP=1).
{"type":"tool_advertise",
 "tools":[...],                // OpenAI-style function schemas this client can run
 "system_prompt":"...",        // optional, overrides the session system prompt
 "llm_params":{...},           // optional, merged into the session LLM params
 "warm_prefix":true}           // optional, false to skip LLM prefix warm-up
{"type":"tool_result",         // reply to a server tool_call
 "call_id":"...", "ok":true,  "result":{...}}
{"type":"tool_result",
 "call_id":"...", "ok":false, "error":"..."}
```

Plus: **binary frames** = int16 PCM at `sample_rate` (mono), feeding ASR.

### Server → Client JSON message types

```
{"type":"asr_partial",       "text":"...", "is_stable":false}
{"type":"asr_endpoint"}                              // VAD detected end of speech
{"type":"asr_final",         "text":"..."}           // single-utterance mode: one per session
                                                     // multi_utterance mode adds:
                                                     //   "session_complete": false   // mid-session boundary
                                                     //   "session_complete": true,   // session-end final
                                                     //   "duplicate_of_streamed": bool
{"type":"tts_started",       "sentence":"..."}       // about to ship audio
{"type":"tts_sentence_done", "sentence":"..."}       // one sentence finished
{"type":"tts_done"}                                  // tts_flush honored, no more audio
{"type":"vad_event",         "event":"speech_start"} // server-side VAD: user started speaking
{"type":"vad_event",         "event":"speech_end"}   // server-side VAD: user stopped speaking
{"type":"error",             "error":"..."}

{"type":"tool_call",                                 // server-side tool loop only
 "call_id":"<uuid4 hex>",                            // echo back in tool_result
 "name":"...", "arguments":{...}}
```

`vad_event` lets the client update its UI / playback state machine in
sync with server-side VAD. `speech_start` is emitted **before** the
server cancels any in-flight TTS for barge-in, so a client buffering
playback can drop pending audio immediately. `speech_end` is emitted
right after the endpoint is latched and precedes the resulting
`asr_final`. Clients that don't care about these events can ignore
unknown frame types — emission is unconditional whenever VAD is active.

Plus: **binary frames** = int16 little-endian PCM TTS output.

**Sample-rate header** (exactly once per WebSocket session): the very
first binary frame the server sends after the WS handshake is a 4-byte
little-endian uint32 carrying the TTS sample rate (matches the format
used by `/tts/stream`). All subsequent binary frames on the same WS
connection are raw int16 LE PCM at that fixed sample rate — there is
**no** header on subsequent frames.

The header is **not** re-emitted on utterance boundaries, sentence
boundaries, `tts_done`, barge-in, `abort`, ASR endpointing, or any
other state transition. Only a fresh WebSocket connection resets the
latch and produces a new header. Clients should parse the header once
on connect and treat every later binary frame as pure PCM payload.

(Implementation reference: `server/main.py:1214,1246-1253` — the
`sr_header_sent` flag lives in `tts_out_task` scope, is initialized
`False` on WS connect, set `True` on the first successful synth, and
never reset for the lifetime of the connection.)

## TTS chunk boundaries

By default, `WS /v2v/stream` uses low-latency TTS chunking rather than
waiting for a full sentence. For CJK text, short clauses can be emitted
on comma-like punctuation once they are useful for synthesis, and
punctuation-free spans are emitted around 12 characters. Sentence-ending
punctuation (`。！？；\n`, plus ASCII `.!?;`) still flushes immediately.

`tts_flush` always flushes the remainder, regardless of punctuation.

Set `OVS_TTS_LOW_LATENCY_CHUNKING=0` to use the older conservative
sentence splitter.

The low-latency thresholds can be tuned with
`OVS_TTS_LOW_LATENCY_CJK_MIN_CHARS`,
`OVS_TTS_LOW_LATENCY_CJK_TARGET_CHARS`, and
`OVS_TTS_LOW_LATENCY_CJK_MAX_CHARS` (defaults: 8 / 12 / 16).

In fallback mode, when `tts_language` maps to one of pysbd's supported
languages (zh/en/ja/ko/es/fr/de/it/pt/ru/ar/hi/...), the splitter uses
pysbd's rule-based segmenter. This is linguistically cleaner but can
wait for lookahead before emitting the current sentence.

## Barge-in

Two triggers, same effect (cancel the in-flight TTS synth task + drop
queued sentences):

1. **Client-initiated**: send `{"type":"abort"}` when the local
   audio playback indicates the user is interrupting.
2. **Server-initiated**: if VAD detects `speech_start` while a TTS
   synth is in flight, the synth is cancelled automatically.

After barge-in, ASR continues to receive audio normally and emit
partials. The next text frame from the client starts a new TTS
sequence (the sentence buffer is per-connection, persistent across
barge-ins).

### The client owns playback cutoff — the server cannot

**TTS output is not paced to real time.** The server pushes PCM as fast as the
backend synthesizes it, which on a fast backend is far quicker than playback:
measured on RK3588 (matcha, RTF ~0.09), a 3-second reply is fully delivered to
the client in roughly 300 ms. Cancelling the synth therefore usually cancels
nothing — the bytes are already on the client.

So barge-in latency, as the user perceives it, is **how fast the client stops
playing**, not how fast the server stops synthesizing. A client that only
cancels server-side and lets its own buffer drain will keep talking over the
user for the full remaining duration of the reply. Both shipped clients handle
this and are worth copying:

* Browser (`demos/common/frontend/v2v-client.js`): `stopPlayback()` stops every
  scheduled WebAudio source and clears the end timer; `demos/v2v-chat` calls it
  from its `vad_event` `speech_start` handler, i.e. server-detected barge-in
  drives local cutoff.
* Device agent (`agent/ovs_agent/audio_io.py`): `stop_playback()` drains the
  queue **and** sets a flag that makes `play()` drop subsequently-arriving PCM
  for the rest of the in-flight utterance — because the server may still be
  streaming the tail of a reply it has already been told to cancel.

Corollary for `abort`: send it *and* stop local playback. `abort` alone is not
a cutoff.

## Remote tools (server-side LLM loop)

Only relevant when the deployment runs the LLM itself
(`OVS_V2V_ENGINE=voxedge` **and** `OVS_V2V_SERVER_LOOP=1` — both, either one
missing and no LLM is called at all). In that mode the client can hand the
server a set of function schemas it is able to execute; the server-side LLM may
then pick one and proxy the call back to that same client.

The tool *runs on the client*. The server only holds the schema — it is not
validated and not authenticated, so a deployment that exposes `/v2v/stream` to
untrusted clients is letting them shape the LLM's tool surface.

### Handshake

Send `config` first, then `tool_advertise`:

```jsonc
{"type":"tool_advertise",
 "tools":[
   {"type":"function",
    "function":{"name":"get_weather",
                "description":"Look up current weather for a city",
                "parameters":{"type":"object",
                              "properties":{"city":{"type":"string"}},
                              "required":["city"]}},
    // optional siblings, all default-able:
    "timeout_s":15.0,
    "preamble_text":"好的，我查一下。",   // spoken as the tool starts
    "completion_text":"",                  // spoken on success if response_mode=template
    "response_mode":"await"}               // "await" | "parallel" | "template"
 ],
 "system_prompt":"你是语音助手……",
 "llm_params":{"model":"Qwen3-4B"}}
```

* A bare `{"name","description","parameters"}` entry (no `function` wrapper) is
  also accepted.
* Missing `parameters` defaults to `{"type":"object","properties":{}}`.
* **Re-advertising is an idempotent upsert**: a name already registered is
  overwritten, and the advertised set is what the LLM sees from the next turn
  on. Clients re-advertise after reconnect for exactly this reason.
* `system_prompt` and `llm_params` mutate **session** state, not just this turn.

### Call / result

```
server → {"type":"tool_call","call_id":"<uuid4 hex>","name":"...","arguments":{...}}
client → {"type":"tool_result","call_id":"<same>","ok":true,"result":{...}}
client → {"type":"tool_result","call_id":"<same>","ok":false,"error":"..."}
```

Every failure — timeout, transport error, client-reported error, barge-in
cancellation — is turned into an error *result* for the LLM rather than an
exception, so a broken tool degrades the turn instead of killing the
conversation.

### Things that bite

| | |
|---|---|
| **`ok`, not `success`** | The result flag is `ok`. A `result` body of `{"success":false}` is also treated as failure by the reference client, but the frame-level field is `ok`. |
| **Read `call_id` before `id`** | Both keys are accepted for correlation and the reference client sends both. Reading `id` first has caused a 15 s stall — prefer `call_id`. |
| **`timeout_s` is not actually advertised** | The reference client omits it, so the server falls back to **15 s** per tool. The client's own dispatch deadline defaults to 30 s, i.e. a client can legally outlive the server's patience. Keep external API calls well under 15 s. |
| **No retry, no ack** | A `tool_call` is sent once. If the client never answers, the server just times out. There is no liveness ping. |
| **Late results are dropped** | An unknown or already-resolved `call_id` is ignored, so a reply arriving after the timeout — or after a reconnect — cannot corrupt a newer turn. |
| **Silently inert without the server loop** | `tool_advertise` without `OVS_V2V_SERVER_LOOP=1` logs a warning server-side and returns. The client gets no error frame. |
| **Realtime V2 uses different frames** | Under the `seeed.realtime.v2` subprotocol this is `session.update` with tools under `session.tools` and the policy fields under a per-tool `x_v2v` object. See [Realtime V2](realtime-v2.md). |

### Runnable example

`agent/proof_driver.py` is a self-contained driver: raw WebSocket, `config` →
`tool_advertise` → WAV → `asr_eos`, answers `tool_call`, prints PASS/FAIL.

```bash
uv run python agent/proof_driver.py --wav /tmp/utt.wav --url ws://HOST:8621/v2v/stream
```

For the tool bodies themselves, `agent/ovs_agent/apps/companion_robot/demo_tools.py`
is the smallest pair of `@tool` handlers. Note that tools registered that way
run **in the agent process** (trusted, no sandbox, no MCP) — advertising them
over the wire is what makes them remote.

Full wire spec: `docs/specs/tool-calling-engine-migration.md` §4 Mode B.

## End-of-utterance semantics

**Single-utterance (default, `multi_utterance: false`)**

- VAD-driven: server emits `asr_endpoint`, then `asr_final`. The audio
  buffer is closed; further binary frames are ignored until the
  client opens a new WebSocket.
- Client-driven: send `{"type":"asr_eos"}` to override VAD and force
  finalize immediately.

If the client sends nothing (binary or eos) and VAD is disabled, the
ASR stream stays open indefinitely (or until WS disconnect).

**Multi-utterance (`multi_utterance: true`)**

Same WS session carries an unbounded sequence of utterances. On each
VAD or backend-detected end-of-speech the server emits:

```
{"type":"asr_endpoint"}
{"type":"asr_final", "text":"<this utterance>", "session_complete": false}
```

The audio buffer stays open; the next utterance is recognized in the
same stream (backends reset per-utterance state automatically on the
next chunk of speech). The session terminates on:

- **Client `asr_eos`** — server runs `finalize()`, emits a final
  `asr_final` with `session_complete: true`. If the final text matches
  the last mid-session final the client already received, the frame
  also carries `duplicate_of_streamed: true` so the client can dedupe.
- **WS disconnect** — same behaviour, but no closing frame is delivered.

Clients should treat mid-session `asr_final` frames as the canonical
transcript of each utterance; the closing frame only delivers any
trailing audio that arrived after the last VAD endpoint and is usually
a duplicate.

## Customer example (Python, asyncio)

Demo: take a Chinese sentence from a streaming LLM, speak it
synchronously to the user while the user can talk back and barge in.

```python
import asyncio, json, struct, websockets

async def v2v_session(transcribe_partial_cb, llm_token_stream):
    uri = "ws://device:8621/v2v/stream"
    async with websockets.connect(uri) as ws:
        # 1. config
        await ws.send(json.dumps({
            "type": "config",
            "asr_language": "Chinese",
            "tts_language": "zh",
            "vad": "silero",
            "vad_silence_ms": 400,
        }))

        sample_rate = None        # parsed once from the first binary
                                  # frame of the session; all later
                                  # binary frames are raw PCM

        async def upstream():
            # Stream mic PCM up to the server
            async for pcm_chunk in mic.read_chunks(16000):
                await ws.send(pcm_chunk)         # binary

        async def llm_to_tts():
            # When ASR final arrives we'll start pulling LLM tokens
            # and forwarding them as text frames to TTS.
            pass   # set up below in main loop

        async def downstream():
            nonlocal sample_rate
            llm_task = None
            async for msg in ws:
                if isinstance(msg, bytes):
                    if sample_rate is None:
                        sample_rate = struct.unpack("<I", msg[:4])[0]
                        speaker.open(sample_rate)
                        continue
                    speaker.write(msg)
                    continue
                evt = json.loads(msg)
                t = evt["type"]
                if t == "asr_partial":
                    transcribe_partial_cb(evt["text"])
                elif t == "asr_final":
                    # Got the user's full utterance. Kick off LLM
                    # → TTS forwarding.
                    user_text = evt["text"]
                    llm_task = asyncio.create_task(
                        forward_tokens_to_tts(ws, user_text, llm_token_stream))
                elif t == "tts_done":
                    break

        async def forward_tokens_to_tts(ws, prompt, stream_fn):
            async for token in stream_fn(prompt):
                await ws.send(json.dumps({"type":"text", "text": token}))
            await ws.send(json.dumps({"type":"tts_flush"}))

        await asyncio.gather(upstream(), downstream())
```

Customer integrates their LLM of choice in `stream_fn(prompt)`.

## Minimal modes

**TTS-only** (feed an LLM stream into TTS, no microphone):

```python
await ws.send(json.dumps({"type":"config", "tts_language":"zh"}))
for token in llm_stream():
    await ws.send(json.dumps({"type":"text", "text": token}))
await ws.send(json.dumps({"type":"tts_flush"}))
async for msg in ws:
    if isinstance(msg, bytes): play(msg)
    elif json.loads(msg)["type"] == "tts_done": break
```

**ASR-only with VAD** (drop-in replacement for `/asr/stream?vad=silero`):

```python
await ws.send(json.dumps({"type":"config",
                          "asr_language":"Chinese",
                          "vad":"silero"}))
async for pcm in mic.read_chunks(16000):
    await ws.send(pcm)         # never need to send {} or b""
async for msg in ws:
    evt = json.loads(msg)
    if evt["type"] == "asr_final":
        return evt["text"]
```

## Operational notes

- **VAD model load**: silero ONNX is loaded once per process, shared
  across every WS connection. First connection pays ~500 ms init.
- **Concurrent connections**: each connection gets its own
  `SileroVADSession` state, its own `SentenceBuffer`, and its own
  ASR/TTS streams. Coordinator policy (concurrent / serialized) still
  applies per device profile.
- **No persistent dialogue state**: each WS is one utterance round
  (audio → text → text → audio). Customer holds LLM context.
- **HTTPS / auth not included** — see operational hardening section in
  `docs/perf-test-runbook.md`.

## Connection lifetime

- **ASR-only mode**: server emits `asr_final` and closes the WS after
  the VAD or client `asr_eos` triggers finalization.
- **TTS-only mode**: server processes `text` frames as long as the
  client sends them; ships audio until `tts_flush` empties the queue,
  then emits `tts_done` and closes.
- **V2V mode**: server waits for *both* ASR work (final emitted) and
  TTS work (flush received + queue drained) to complete before
  closing. The client should always send `tts_flush` after the LLM
  stream ends, otherwise the TTS task lives forever (waiting for
  more text) even though no more audio will come.
- WS disconnect from the client side immediately cancels all in-flight
  tasks (synth threads are signaled to break out of their generators).

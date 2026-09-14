# Seeed Realtime V2 WebSocket Protocol

Status: implementation target (2026-07-16)

Endpoint: `WS /v2v/stream`

WebSocket subprotocol: `seeed.realtime.v2`

Realtime V2 is the canonical client-facing protocol for local cascade,
OpenAI Realtime, and Qwen Realtime providers. JSON frames carry control and
lifecycle events. Binary frames carry mono signed 16-bit little-endian PCM.

## 1. Version negotiation

Clients request the `seeed.realtime.v2` WebSocket subprotocol. The server must
echo the selected subprotocol and send `session.created` as its first JSON
frame.

During the migration window only, a connection without this subprotocol uses
the legacy V1 dialect documented in `v2v-stream.md`. V2 applications must not
parse or emit V1 events. The compatibility path exists for external clients
and diagnostic scripts, not for provider selection.

## 2. Required identifiers

Every server JSON event has a unique `event_id`. Events concerning generated
output also carry `response_id`; content events carry `item_id`,
`output_index`, and `content_index` where applicable. Function calls use a
stable `call_id` across request and result.

Identifiers are opaque strings. Clients compare them for equality and must
not parse their prefixes.

## 3. Session handshake

Server:

```json
{
  "type": "session.created",
  "event_id": "evt_1",
  "session": {
    "id": "sess_1",
    "object": "realtime.session",
    "protocol_version": 2,
    "provider": "local-cascade",
    "model": "local-cascade",
    "type": "realtime",
    "output_modalities": ["audio"],
    "audio": {
      "input": {
        "format": {
          "type": "audio/pcm",
          "sample_rate": 16000,
          "channels": 1,
          "endianness": "little"
        }
      },
      "output": {
        "format": {
          "type": "audio/pcm",
          "sample_rate": 24000,
          "channels": 1,
          "endianness": "little"
        }
      }
    },
    "capabilities": {
      "binary_audio": true,
      "function_calling": true,
      "conversation_truncate": true,
      "input_transcription": true,
      "direct_speak": true
    }
  }
}
```

Client:

```json
{
  "type": "session.update",
  "event_id": "evt_client_1",
  "session": {
    "type": "realtime",
    "output_modalities": ["audio"],
    "instructions": "You are a helpful robot.",
    "audio": {
      "input": {
        "format": {"type": "audio/pcm", "rate": 16000,
                   "channels": 1, "endianness": "little"},
        "transcription": {"language": "auto"},
        "turn_detection": {
          "type": "server_vad",
          "silence_duration_ms": 500,
          "create_response": true,
          "interrupt_response": true,
          "vad_endpoint_silence_ms": 1500
        }
      },
      "output": {
        "format": {"type": "audio/pcm", "rate": 24000,
                   "channels": 1, "endianness": "little"},
        "voice": "default",
        "speed": 1.0
      }
    },
    "tools": []
  }
}
```

Server replies with `session.updated`, containing the effective normalized
session. The client must not start its microphone pump until this ack arrives.
Unsupported required configuration produces a structured `error`; the server
must not silently report an unsupported option as active.

For the current local provider, `create_response=true` is available only when
the Gateway runs with `OVS_V2V_ENGINE=voxedge` and
`OVS_V2V_SERVER_LOOP=1`. Otherwise the handshake returns
`unsupported_create_response`. Cloud adapters will report their effective
mode through the same `session.updated` field rather than exposing provider
flags to applications.

## 4. Audio data plane

- Client binary frames are implicit `input_audio_buffer.append` operations.
- Server binary frames are implicit `response.output_audio.delta` operations.
- The negotiated session format applies to every binary frame.
- V2 has no in-band sample-rate header.
- A session has at most one active audio response. Binary multiplexing is not
  part of V2.0.

If concurrent audio responses are added later, they require a new subprotocol
with an explicit binary envelope.

## 5. Input and turn control

Client events:

```json
{"type":"input_audio_buffer.commit","event_id":"evt_client_2"}
{"type":"input_audio_buffer.clear","event_id":"evt_client_3"}
{"type":"response.create","event_id":"evt_client_4","response":{}}
{"type":"response.cancel","event_id":"evt_client_5","response_id":"resp_1"}
```

Server VAD/transcription events:

```text
input_audio_buffer.speech_started
input_audio_buffer.speech_stopped
input_audio_buffer.committed
conversation.item.input_audio_transcription.delta
conversation.item.input_audio_transcription.completed
conversation.item.input_audio_transcription.failed
```

With `turn_detection.create_response=true`, committing a VAD turn creates a
response automatically. With it false, the client sends `response.create`.

`turn_detection.vad_endpoint_silence_ms` (optional, ms) overrides the endpoint
threshold of an ASR backend that endpoints on its own — the RK Qwen3 backend
does so in both streaming modes even when `type` is `none`. Absent/`0` keeps the
profile value; unparseable or out-of-range values are ignored. Keep it above the
client-side VAD silence, otherwise the backend wins the race and cuts the
utterance in half (`docs/CONFIGURATION.md`, "Streaming ASR endpointing").

Manual mode is a generation barrier, not only a client-side convention. After
the transcription completes, the server retains the committed input but must
not invoke the language model or emit `response.created` until it receives
`response.create`. Events on one connection are applied in order, so a
`session.update` sent after transcription and before `response.create` must
affect that response. This is the supported flow for injecting per-turn visual
context without racing response generation.

V2.0 retains at most one pending manual input. `response.create` without a
pending input, or while another response is active, produces a structured
`error`. `response.cancel` and connection teardown discard pending manual
state as well as stopping an active response.

Input transcription is display/observability data. A native audio model may
interpret speech differently; clients must not treat the transcript as the
model's authoritative internal input.

## 6. Response lifecycle

The minimum audio response sequence is:

```text
response.created
response.output_audio_transcript.delta  (zero or more)
binary PCM                               (zero or more)
response.output_audio_transcript.done
response.output_audio.done
response.done
```

`response.output_audio.done` means the server will send no more audio bytes for
that response. `response.done` is the single terminal response event and has
one of these statuses:

```text
completed | cancelled | failed | incomplete
```

Example:

```json
{
  "type": "response.done",
  "event_id": "evt_9",
  "response": {
    "id": "resp_1",
    "object": "realtime.response",
    "status": "cancelled",
    "status_details": {
      "type": "cancelled",
      "reason": "turn_detected"
    },
    "output": [],
    "usage": null
  }
}
```

Terminal invariants:

1. Each `response.created` has exactly one `response.done`.
2. No binary audio for a response is sent after its
   `response.output_audio.done`.
3. No response events are sent after its `response.done`.
4. Cancellation is a normal terminal status, not an `error`.
5. Provider terminal events are normalized by the Gateway; applications never
   branch on provider names.

## 7. Playback completion

Provider output completion and device playback completion are separate:

```text
response.output_audio.done  -> remote generation ended
response.done               -> remote response ended
playback drained            -> local speaker queue is silent
```

`ovs_agent` exposes the stable application hook `on_assistant_done` only after
`response.done` and, when playback draining is enabled, after the local queue
is silent. Application plugins must use this hook rather than wire events.

## 8. Interruption and truncation

When the user interrupts:

1. Client sends `response.cancel(response_id)` unless server VAD already did.
2. Client immediately rejects queued/late PCM belonging to the old response.
3. Client stops local playback and records the played duration.
4. Client sends:

```json
{
  "type": "conversation.item.truncate",
  "event_id": "evt_client_6",
  "item_id": "item_assistant_1",
  "content_index": 0,
  "audio_end_ms": 1230
}
```

5. Server emits one `response.done` with `status=cancelled`.

If a provider lacks native truncation, the Gateway must emulate conversation
state or advertise `conversation_truncate=false`. It must never claim support
and silently retain unheard output in conversation history.

## 9. Tools

Tools are configured in `session.update.session.tools`. Provider-neutral fields
remain at the tool root. Seeed-only execution policy belongs under `x_v2v`.

Function call output is streamed through response output item/function argument
events. The device returns results as a conversation item:

```json
{
  "type": "response.function_call_arguments.done",
  "response_id": "resp_1",
  "item_id": "item_call_1",
  "call_id": "call_1",
  "name": "wave",
  "arguments": "{\"side\":\"left\"}",
  "x_v2v": {"timeout_s": 12.0}
}
```

```json
{
  "type": "conversation.item.create",
  "event_id": "evt_client_7",
  "item": {
    "type": "function_call_output",
    "call_id": "call_1",
    "output": "{\"ok\":true}"
  }
}
```

Tool list changes require another `session.update` and `session.updated` ack.
Reconnect creates a new session and therefore requires replaying instructions,
tools, and mutable session configuration.

## 10. Direct speech extension

Robot safety and failure announcements require deterministic text-to-speech
without asking a generative model to paraphrase. The application API is:

```python
await app.speak(text, conversation="none")
```

The corresponding Gateway capability is `direct_speak`. Provider adapters may
use a local/cloud TTS channel or a provider-specific exact-speech mechanism.
If exact speech cannot be guaranteed, the Gateway returns a structured error.

This extension must not add the announcement to normal conversation history.

Wire request:

```json
{
  "type": "x_v2v.response.speak",
  "speech": {
    "text": "机械臂即将复位，请注意安全。",
    "conversation": "none"
  }
}
```

It produces the normal `response.created → response.output_item.added → audio
→ response.output_audio.done → response.done` lifecycle. The response carries
`metadata.x_v2v.direct_speak=true`; applications must not wait for a separate
TTS-only completion event.

## 11. Conversation reset extension

Applications that implement single-turn or privacy-sensitive modes reset both
local and provider state with:

```json
{"type":"x_v2v.conversation.reset","event_id":"evt_client_8"}
```

The Gateway cancels an active response, clears provider conversation state,
and replies with `x_v2v.conversation.reset.done`. For the local cascade this is
an acknowledged no-op after cancellation because it does not retain remote
assistant audio history between turns.

## 12. Error shape

```json
{
  "type": "error",
  "event_id": "evt_10",
  "error": {
    "type": "invalid_request_error",
    "code": "unsupported_audio_format",
    "message": "Only mono PCM16 input is supported",
    "param": "session.audio.input.format",
    "event_id": "evt_client_1"
  }
}
```

Errors do not substitute for response terminal events. If an error terminates
an active response, the server emits the error for diagnostics and still emits
`response.done(status=failed)`.

## 13. Application-level compatibility contract

The following hooks stay stable across V1-to-V2 migration and across providers:

```text
on_user_speech_start
on_user_partial
on_user_utterance
on_assistant_sentence_start   (optional/provider-derived)
on_assistant_sentence         (optional/provider-derived)
on_tts_audio_frame
on_assistant_done
on_error
```

Sentence hooks are optional enhancements. Core FSM transitions may depend only
on session, input-buffer, response, binary-audio, playback, and error events.

## 14. Gateway provider selection

Applications always connect to the same `/v2v/stream` endpoint. Deployment
selects the upstream at the Gateway:

```bash
OVS_REALTIME_PROVIDER=local

OVS_REALTIME_PROVIDER=openai
OPENAI_API_KEY=...
OVS_REALTIME_OPENAI_MODEL=gpt-realtime-2.1

OVS_REALTIME_PROVIDER=qwen
DASHSCOPE_API_KEY=...
OVS_REALTIME_QWEN_URL=wss://WORKSPACE_ID.REGION.maas.aliyuncs.com/api-ws/v1/realtime
OVS_REALTIME_QWEN_MODEL=qwen-audio-3.0-realtime-flash
```

The cloud relay converts downstream binary PCM to provider Base64 audio events,
normalizes provider audio deltas to downstream binary PCM, and keeps provider
credentials out of robot applications. OpenAI input is resampled to 24 kHz at
the Gateway; Qwen input remains 16 kHz.

Capabilities are authoritative. OpenAI currently advertises native truncation;
Qwen advertises `conversation_truncate=false`. Both cloud adapters advertise
`direct_speak=false` until a deterministic TTS side channel is configured.
Safety announcements must not be replaced with a best-effort generative prompt.

The control plane follows the current OpenAI Realtime GA layout
(`output_modalities`, `audio.*.format.rate`, flat function tools). The Qwen
adapter maps its flat session fields and `response.audio.*` events internally.

"""V2V WebSocket protocol + sentence buffering.

Used by:
- WS /v2v/stream — unified ASR + TTS + VAD + barge-in endpoint
- (optionally exposed as TTS-only / ASR-only by which config keys the
  client supplies)

Protocol spec: docs/api/v2v-stream.md
"""
from __future__ import annotations

import re
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

REALTIME_V2_SUBPROTOCOL = "seeed.realtime.v2"

# pysbd — Python Sentence Boundary Disambiguation. Rule-based, no model
# files, 22 languages, handles abbreviations ("Dr. Smith", "U.S.A."),
# numbers ("3.14"), URLs ("example.com"). ~100 KB pure Python. If it's
# missing (older image, dev env), we fall back to a simple regex that
# over-splits abbreviations but still works.
try:
    import pysbd
    _PYSBD_AVAILABLE = True
except ImportError:
    pysbd = None  # type: ignore
    _PYSBD_AVAILABLE = False


# ────────────────────────────────────────────────────────────────────────
# Client → Server JSON message types
# ────────────────────────────────────────────────────────────────────────
CLIENT_CONFIG     = "config"        # initial setup, must be first message
CLIENT_TEXT       = "text"          # streaming text input for TTS
CLIENT_ASR_PREPARE = "asr_prepare"  # precompute ASR final before EOS
CLIENT_ASR_EOS    = "asr_eos"       # manually finalize ASR (overrides VAD)
CLIENT_TTS_FLUSH  = "tts_flush"     # flush remaining TTS buffer
CLIENT_ABORT      = "abort"         # barge-in: cancel current TTS
# Idle keepalive. Carries no state and is a no-op for the session — its only
# job is to be a frame, so the dispatcher's idle watchdog
# (OVS_V2V_IDLE_TIMEOUT_S) sees traffic and does not reap a live-but-silent
# client. A client that stops pinging is still reaped on schedule, so the
# half-open protection this watchdog exists for is unchanged.
CLIENT_PING       = "ping"
# Remote server-side tool loop (spec §4 Mode B). Additive; legacy clients that
# never enable the server tool loop never see/send these.
CLIENT_TOOL_RESULT = "tool_result"  # device client returns a remote-tool result
# Tool advertise handshake (spec §4/§6). The device client, right after opening
# the session, uploads the OpenAI-style tool schemas it can execute locally
# (payload: {"tools": [...], "system_prompt"?: str, "llm_params"?: {...}}). The
# server registers them as remote-dispatch tools so the server-side LLM loop can
# select one and proxy execution back via SERVER_TOOL_CALL. Additive — a legacy
# client that never enables the server loop never sends this.
CLIENT_TOOL_ADVERTISE = "tool_advertise"

# Realtime V2 canonical client events. The legacy names above remain available
# only while non-V2 clients are migrated; V2 applications must use these.
CLIENT_SESSION_UPDATE = "session.update"
CLIENT_INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
CLIENT_INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"
CLIENT_RESPONSE_CREATE = "response.create"
CLIENT_RESPONSE_CANCEL = "response.cancel"
CLIENT_CONVERSATION_ITEM_CREATE = "conversation.item.create"
CLIENT_CONVERSATION_ITEM_TRUNCATE = "conversation.item.truncate"
CLIENT_DIRECT_SPEAK = "x_v2v.response.speak"
CLIENT_CONVERSATION_RESET = "x_v2v.conversation.reset"

# ────────────────────────────────────────────────────────────────────────
# Server → Client JSON message types
# ────────────────────────────────────────────────────────────────────────
SERVER_ASR_PARTIAL        = "asr_partial"
SERVER_ASR_ENDPOINT       = "asr_endpoint"       # VAD detected end of speech
SERVER_ASR_FINAL          = "asr_final"
SERVER_TTS_STARTED        = "tts_started"        # first audio frame about to ship
SERVER_TTS_SENTENCE_DONE  = "tts_sentence_done"  # one sentence finished
SERVER_TTS_DONE           = "tts_done"           # flush complete, no more audio
SERVER_VAD_EVENT          = "vad_event"          # server-side VAD speech_start/speech_end
SERVER_ERROR              = "error"
# Remote server-side tool loop (spec §4 Mode B). Server asks a device client to
# run a tool and report back via CLIENT_TOOL_RESULT. Additive; only emitted when
# the server-side tool loop is enabled (OVS_V2V_SERVER_LOOP) with remote tools.
SERVER_TOOL_CALL          = "tool_call"

# Realtime V2 canonical server events.
SERVER_SESSION_CREATED = "session.created"
SERVER_SESSION_UPDATED = "session.updated"
SERVER_INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
SERVER_INPUT_AUDIO_BUFFER_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
SERVER_INPUT_AUDIO_BUFFER_COMMITTED = "input_audio_buffer.committed"
SERVER_INPUT_AUDIO_TRANSCRIPTION_DELTA = (
    "conversation.item.input_audio_transcription.delta"
)
SERVER_INPUT_AUDIO_TRANSCRIPTION_COMPLETED = (
    "conversation.item.input_audio_transcription.completed"
)
SERVER_RESPONSE_CREATED = "response.created"
SERVER_RESPONSE_OUTPUT_AUDIO_DONE = "response.output_audio.done"
SERVER_RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA = (
    "response.output_audio_transcript.delta"
)
SERVER_RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE = (
    "response.output_audio_transcript.done"
)
SERVER_RESPONSE_DONE = "response.done"
SERVER_RESPONSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
SERVER_RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA = (
    "response.function_call_arguments.delta"
)
SERVER_RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = (
    "response.function_call_arguments.done"
)
SERVER_CONVERSATION_ITEM_TRUNCATED = "conversation.item.truncated"
SERVER_CONVERSATION_RESET_DONE = "x_v2v.conversation.reset.done"

# vad_event "event" field values
VAD_EVENT_SPEECH_START    = "speech_start"
VAD_EVENT_SPEECH_END      = "speech_end"


def _new_realtime_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def session_update_to_legacy_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Realtime V2 ``session.update`` into the current engine cfg.

    The legacy config shape remains an internal adapter contract while the
    ASR/TTS engine is migrated. Provider and application code must not depend
    on it. Unknown V2 fields are intentionally ignored here and remain present
    in the canonical session object returned by ``session.updated``.
    """
    session = payload.get("session")
    if not isinstance(session, dict):
        raise ValueError("session.update.session must be an object")
    audio = session.get("audio") if isinstance(session.get("audio"), dict) else {}
    audio_in = audio.get("input") if isinstance(audio.get("input"), dict) else {}
    audio_out = audio.get("output") if isinstance(audio.get("output"), dict) else {}
    input_format = (
        audio_in.get("format") if isinstance(audio_in.get("format"), dict) else {}
    )
    transcription = (
        audio_in.get("transcription")
        if isinstance(audio_in.get("transcription"), dict)
        else {}
    )
    turn_detection = (
        audio_in.get("turn_detection")
        if isinstance(audio_in.get("turn_detection"), dict)
        else {}
    )

    sample_rate = int(input_format.get("rate", input_format.get("sample_rate", 16000)))
    channels = int(input_format.get("channels", 1))
    if channels != 1:
        raise ValueError("Realtime V2 currently supports mono input only")

    turn_type = turn_detection.get("type", "server_vad")
    vad = "none" if turn_type in (None, "none") else turn_detection.get("backend", "silero")
    cfg: dict[str, Any] = {
        "type": CLIENT_CONFIG,
        "sample_rate": sample_rate,
        "asr_language": transcription.get("language"),
        "tts_language": audio_out.get("language"),
        "tts_voice": audio_out.get("voice"),
        "tts_speaker_id": audio_out.get("speaker_id"),
        "tts_speed": audio_out.get("speed"),
        "vad": vad,
        "vad_silence_ms": int(turn_detection.get("silence_duration_ms", 400)),
        # Backend-owned endpoint threshold override (RK Qwen3 endpoints
        # internally even with vad:"none"). Additive: clients that don't send
        # it keep the profile value; unknown V2 fields are ignored elsewhere,
        # so older servers simply drop it.
        "vad_endpoint_silence_ms": turn_detection.get("vad_endpoint_silence_ms"),
        # Realtime sessions are persistent by definition.
        "multi_utterance": True,
        "_realtime_v2": True,
        # A disabled turn detector cannot autonomously create a response.
        # For server_vad, retain the Realtime-style default of true unless the
        # client explicitly selects client-loop operation.
        "_create_response": bool(
            turn_detection.get(
                "create_response", turn_type not in (None, "none")
            )
        ),
        "_interrupt_response": bool(turn_detection.get("interrupt_response", True)),
        "_canonical_session": session,
    }
    # A modality may intentionally disable one side of the local cascade.
    modalities = session.get("output_modalities", session.get("modalities"))
    if isinstance(modalities, list) and "audio" not in modalities:
        cfg["tts_language"] = None
    return cfg


@dataclass
class RealtimeV2EventAdapter:
    """Translate the current local-engine events to canonical V2 events.

    This is deliberately provider-neutral and stateful: it owns the active
    response/item IDs and guarantees a single terminal ``response.done``.
    It is also usable by the voxedge transport adapter so both orchestration
    paths expose the same wire lifecycle.
    """

    provider: str = "local-cascade"
    model: str = "local-cascade"
    input_sample_rate: int = 16000
    output_sample_rate: int = 16000
    id_factory: Callable[[str], str] = _new_realtime_id
    capabilities_override: dict[str, bool] | None = None
    session_id: str = field(init=False)
    active_response_id: str | None = field(default=None, init=False)
    active_item_id: str | None = field(default=None, init=False)
    active_status: str = field(default="completed", init=False)
    active_reason: str | None = field(default=None, init=False)
    input_item_id: str | None = field(default=None, init=False)
    audio_item_announced: bool = field(default=False, init=False)
    direct_speak: bool = field(default=False, init=False)
    output_item_count: int = field(default=0, init=False)
    active_audio_output_index: int = field(default=0, init=False)
    active_transcript: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.session_id = self.id_factory("sess")

    def _event_id(self) -> str:
        return self.id_factory("evt")

    def capabilities(self) -> dict[str, bool]:
        defaults = {
            "binary_audio": True,
            "function_calling": True,
            # Local cascade doesn't retain unheard assistant audio in
            # provider history, so truncate is an acknowledged no-op.
            "conversation_truncate": True,
            "input_transcription": True,
            "direct_speak": True,
            "conversation_reset": True,
        }
        if self.capabilities_override:
            defaults.update(self.capabilities_override)
        return defaults

    def session_created(self) -> dict[str, Any]:
        return {
            "type": SERVER_SESSION_CREATED,
            "event_id": self._event_id(),
            "session": {
                "id": self.session_id,
                "object": "realtime.session",
                "protocol_version": 2,
                "provider": self.provider,
                "model": self.model,
                "type": "realtime",
                "output_modalities": ["audio"],
                "audio": {
                    "input": {"format": {
                        "type": "audio/pcm",
                        "rate": self.input_sample_rate,
                        "channels": 1,
                        "endianness": "little",
                    }},
                    "output": {"format": {
                        "type": "audio/pcm",
                        "rate": self.output_sample_rate,
                        "channels": 1,
                        "endianness": "little",
                    }},
                },
                "capabilities": self.capabilities(),
            },
        }

    def session_updated(
        self,
        session: dict[str, Any],
        *,
        create_response: bool | None = None,
        interrupt_response: bool | None = None,
    ) -> dict[str, Any]:
        effective = dict(session)
        audio = dict(effective.get("audio") or {})
        audio_input = dict(audio.get("input") or {})
        audio_output = dict(audio.get("output") or {})
        input_format = dict(audio_input.get("format") or {})
        output_format = dict(audio_output.get("format") or {})
        input_format.update({
            "type": "audio/pcm",
            "rate": self.input_sample_rate,
            "channels": 1,
            "endianness": "little",
        })
        input_format.pop("sample_rate", None)
        output_format.update({
            "type": "audio/pcm",
            "rate": self.output_sample_rate,
            "channels": 1,
            "endianness": "little",
        })
        output_format.pop("sample_rate", None)
        audio_input["format"] = input_format
        turn_detection = dict(audio_input.get("turn_detection") or {})
        if create_response is not None:
            turn_detection["create_response"] = create_response
        if interrupt_response is not None:
            turn_detection["interrupt_response"] = interrupt_response
        if turn_detection:
            audio_input["turn_detection"] = turn_detection
        audio_output["format"] = output_format
        audio["input"] = audio_input
        audio["output"] = audio_output
        effective["audio"] = audio
        effective.update({
            "id": self.session_id,
            "object": "realtime.session",
            "protocol_version": 2,
            "provider": self.provider,
            "model": self.model,
            "capabilities": self.capabilities(),
        })
        return {
            "type": SERVER_SESSION_UPDATED,
            "event_id": self._event_id(),
            "session": effective,
        }

    def mark_cancelled(self, reason: str = "client_cancelled") -> None:
        if self.active_response_id is not None:
            self.active_status = "cancelled"
            self.active_reason = reason

    def mark_direct_speak(self) -> None:
        """Tag the next response as deterministic, history-free speech."""
        self.direct_speak = True

    def _ensure_response_created(self) -> list[dict[str, Any]]:
        if self.active_response_id is not None:
            return []
        self.active_response_id = self.id_factory("resp")
        self.active_item_id = self.id_factory("item")
        self.active_status = "completed"
        self.active_reason = None
        self.active_transcript = []
        return [{
            "type": SERVER_RESPONSE_CREATED,
            "event_id": self._event_id(),
            "response": {
                "id": self.active_response_id,
                "object": "realtime.response",
                "status": "in_progress",
                "output": [],
                "metadata": {"x_v2v.direct_speak": self.direct_speak},
            },
        }]

    def _ensure_audio_item_added(self) -> list[dict[str, Any]]:
        events = self._ensure_response_created()
        if self.audio_item_announced:
            return events
        self.audio_item_announced = True
        self.active_audio_output_index = self.output_item_count
        self.output_item_count += 1
        events.append({
            "type": SERVER_RESPONSE_OUTPUT_ITEM_ADDED,
            "event_id": self._event_id(),
            "response_id": self.active_response_id,
            "output_index": self.active_audio_output_index,
            "item": {
                "id": self.active_item_id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [{"type": "output_audio"}],
            },
        })
        return events

    def _finish_response(self) -> list[dict[str, Any]]:
        events = self._ensure_response_created()
        response_id = self.active_response_id or ""
        status = self.active_status
        reason = self.active_reason
        if self.audio_item_announced:
            events.append({
                "type": SERVER_RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE,
                "event_id": self._event_id(),
                "response_id": response_id,
                "item_id": self.active_item_id,
                "output_index": self.active_audio_output_index,
                "content_index": 0,
                "transcript": "".join(self.active_transcript),
            })
            events.append({
                "type": SERVER_RESPONSE_OUTPUT_AUDIO_DONE,
                "event_id": self._event_id(),
                "response_id": response_id,
                "item_id": self.active_item_id,
                "output_index": self.active_audio_output_index,
                "content_index": 0,
            })
        details = None
        if status != "completed":
            details = {"type": status, "reason": reason or status}
        events.append({
            "type": SERVER_RESPONSE_DONE,
            "event_id": self._event_id(),
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "status": status,
                "status_details": details,
                "output": [],
                "usage": None,
            },
        })
        self.active_response_id = None
        self.active_item_id = None
        self.active_status = "completed"
        self.active_reason = None
        self.audio_item_announced = False
        self.direct_speak = False
        self.output_item_count = 0
        self.active_audio_output_index = 0
        self.active_transcript = []
        return events

    def translate(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Return zero or more canonical events for one engine JSON event."""
        typ = payload.get("type")
        if isinstance(typ, str) and (
            typ.startswith("session.")
            or typ.startswith("response.")
            or typ.startswith("input_audio_buffer.")
            or typ.startswith("conversation.item.")
        ):
            event = dict(payload)
            event.setdefault("event_id", self._event_id())
            return [event]
        if typ == SERVER_TTS_STARTED:
            events = self._ensure_audio_item_added()
            events.append({
                "type": "x_v2v.tts_sentence.started",
                "event_id": self._event_id(),
                "response_id": self.active_response_id,
                "sentence": payload.get("sentence", ""),
            })
            return events
        if typ == SERVER_TTS_SENTENCE_DONE:
            sentence = str(payload.get("sentence") or "")
            self.active_transcript.append(sentence)
            return [{
                "type": SERVER_RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA,
                "event_id": self._event_id(),
                "response_id": self.active_response_id,
                "item_id": self.active_item_id,
                "output_index": self.active_audio_output_index,
                "content_index": 0,
                "delta": sentence,
            }, {
                "type": "x_v2v.tts_sentence.done",
                "event_id": self._event_id(),
                "response_id": self.active_response_id,
                "sentence": sentence,
            }]
        if typ == SERVER_TTS_DONE:
            # V1 emits an extra session-final tts_done after the last per-turn
            # done in persistent sessions. V2 has no response to terminate at
            # that point; session and response lifecycles are separate.
            if self.active_response_id is None and payload.get("session_complete") is True:
                return []
            return self._finish_response()
        if typ == SERVER_VAD_EVENT:
            mapped = (
                SERVER_INPUT_AUDIO_BUFFER_SPEECH_STARTED
                if payload.get("event") == VAD_EVENT_SPEECH_START
                else SERVER_INPUT_AUDIO_BUFFER_SPEECH_STOPPED
            )
            if mapped == SERVER_INPUT_AUDIO_BUFFER_SPEECH_STARTED:
                self.input_item_id = self.id_factory("item")
            return [{
                "type": mapped,
                "event_id": self._event_id(),
                "item_id": self.input_item_id,
            }]
        if typ == SERVER_ASR_ENDPOINT:
            self.input_item_id = self.input_item_id or self.id_factory("item")
            return [{
                "type": SERVER_INPUT_AUDIO_BUFFER_COMMITTED,
                "event_id": self._event_id(),
                "item_id": self.input_item_id,
            }]
        if typ == SERVER_ASR_PARTIAL:
            self.input_item_id = self.input_item_id or self.id_factory("item")
            return [{
                "type": SERVER_INPUT_AUDIO_TRANSCRIPTION_DELTA,
                "event_id": self._event_id(),
                "item_id": self.input_item_id,
                "content_index": 0,
                "delta": payload.get("text", ""),
                "is_stable": bool(payload.get("is_stable", False)),
            }]
        if typ == SERVER_ASR_FINAL:
            self.input_item_id = self.input_item_id or self.id_factory("item")
            event = {
                "type": SERVER_INPUT_AUDIO_TRANSCRIPTION_COMPLETED,
                "event_id": self._event_id(),
                "item_id": self.input_item_id,
                "content_index": 0,
                "transcript": payload.get("text", ""),
            }
            if payload.get("language"):
                event["language"] = payload["language"]
            self.input_item_id = None
            return [event]
        if typ == SERVER_TOOL_CALL:
            events = self._ensure_response_created()
            call_id = str(payload.get("call_id") or payload.get("id") or "")
            item_id = self.id_factory("item")
            output_index = self.output_item_count
            self.output_item_count += 1
            arguments = payload.get("arguments")
            if isinstance(arguments, str):
                arguments_json = arguments
            else:
                import json
                arguments_json = json.dumps(
                    arguments if isinstance(arguments, dict) else {},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            events.extend([
                {
                    "type": SERVER_RESPONSE_OUTPUT_ITEM_ADDED,
                    "event_id": self._event_id(),
                    "response_id": self.active_response_id,
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": call_id,
                        "name": payload.get("name", ""),
                        "arguments": "",
                    },
                },
                {
                    "type": SERVER_RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE,
                    "event_id": self._event_id(),
                    "response_id": self.active_response_id,
                    "item_id": item_id,
                    "output_index": output_index,
                    "call_id": call_id,
                    "name": payload.get("name", ""),
                    "arguments": arguments_json,
                    "x_v2v": {"timeout_s": payload.get("timeout_s", 15.0)},
                },
            ])
            return events
        if typ == SERVER_ERROR:
            raw = payload.get("error")
            if isinstance(raw, dict):
                error = dict(raw)
            else:
                error = {
                    "type": "server_error",
                    "code": payload.get("code") or str(raw or "unknown_error"),
                    "message": str(raw or "unknown server error"),
                    "param": payload.get("param"),
                }
            return [{
                "type": SERVER_ERROR,
                "event_id": self._event_id(),
                "error": error,
            }]
        # Keep non-core extension/tool events observable during the first
        # migration slice; subsequent phases normalize tool events fully.
        event = dict(payload)
        event.setdefault("event_id", self._event_id())
        return [event]


# ────────────────────────────────────────────────────────────────────────
# Sentence buffering for streaming TTS input
# ────────────────────────────────────────────────────────────────────────

# Languages pysbd 0.3.4 supports out-of-the-box (ISO-639-1).
_PYSBD_LANGS = {
    "am", "ar", "bg", "da", "de", "el", "en", "es", "fa", "fr",
    "hi", "hy", "it", "ja", "kk", "mr", "my", "nl", "pl", "ru",
    "ur", "zh",
}

# Verbose names → ISO codes. Customer configs sometimes pass these.
_LANG_ALIASES = {
    "english": "en",    "chinese": "zh",    "japanese": "ja",
    "korean": "ko",     "spanish": "es",    "french": "fr",
    "german": "de",     "italian": "it",    "portuguese": "pt",
    "russian": "ru",    "arabic": "ar",     "hindi": "hi",
    "dutch": "nl",      "polish": "pl",     "greek": "el",
    "burmese": "my",    "marathi": "mr",
}


def _normalize_lang(lang: Optional[str]) -> Optional[str]:
    """Return ISO 639-1 code if pysbd supports it, else None (caller
    falls back to the regex splitter)."""
    if not lang:
        return None
    lc = str(lang).strip().lower()
    code = _LANG_ALIASES.get(lc, lc)
    return code if code in _PYSBD_LANGS else None


# Regex-fallback sentence boundary: CJK terminators always count; ASCII
# `.!?` only count when followed by whitespace or buffer-end (avoids
# "3.14" but still over-splits "Dr. Smith" — that's why we prefer pysbd
# when available).
_SENTENCE_END_RE = re.compile(r"[。！？；\n]+|[!?.](?=\s|$)")

DEFAULT_MIN_SENTENCE_CHARS = 2
DEFAULT_MAX_BUFFER_CHARS   = 200


def _contains_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF:
            return True
    return False


def _has_speakable_text(text: str) -> bool:
    """Return whether *text* contains something a TTS backend should voice.

    Streaming chunkers can emit a bounded text span immediately and receive
    its sentence terminator in the next LLM token.  On final flush that leaves
    a punctuation-only remainder.  Sending that remainder to fixed-shape TTS
    backends is not harmless: Matcha renders a lone ``。`` as roughly one
    second of non-speech tail audio.  Letters and numbers (including CJK via
    ``isalnum``) are speakable; whitespace/punctuation-only fragments are not.
    """
    return any(ch.isalnum() for ch in text)


@dataclass
class SentenceBuffer:
    """Accumulates streaming text and emits complete sentences.

    Used to bridge a token-streaming source (LLM) to a sentence-batched
    sink (TTS engine). Two implementations:

    1. pysbd-backed (default when language is recognized & pysbd is
       installed) — correctly handles abbreviations, numbers, URLs.
    2. regex-backed fallback — splits on punctuation; over-splits
       abbreviations like "Dr. Smith" but works everywhere.

    Usage::

        buf = SentenceBuffer(language="en")     # or "zh"/"ja"/...
        for token in llm_tokens:
            for sentence in buf.add(token):
                tts.synthesize(sentence)
        for sentence in buf.flush():            # at end-of-stream
            tts.synthesize(sentence)

    Note on streaming latency: when the pysbd path is active, a sentence
    is only emitted once the buffer contains the NEXT sentence's first
    characters (pysbd needs lookahead to confidently split). For typical
    LLM streams with sub-50 ms inter-token gaps this is invisible. If
    you have a one-shot final sentence, call `flush()` to force it out.
    """

    language:   Optional[str] = None
    min_chars:  int = DEFAULT_MIN_SENTENCE_CHARS
    max_buffer: int = DEFAULT_MAX_BUFFER_CHARS
    _buf:       str = field(default="", init=False, repr=False)
    _seg:       object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        code = _normalize_lang(self.language)
        if _PYSBD_AVAILABLE and code is not None:
            try:
                self._seg = pysbd.Segmenter(language=code, clean=False)
            except Exception:
                self._seg = None

    # ─── public API ────────────────────────────────────────────────

    def add(self, chunk: str) -> Iterator[str]:
        """Append text, yield any sentences now complete."""
        if not chunk:
            return
        self._buf += chunk
        if self._seg is not None:
            yield from self._emit_pysbd()
        else:
            yield from self._emit_regex()

    def flush(self) -> Iterator[str]:
        """Yield remaining text as a final sentence (no min-length check)."""
        leftover = self._buf.strip()
        self._buf = ""
        if leftover and _has_speakable_text(leftover):
            yield leftover

    def is_empty(self) -> bool:
        return not self._buf.strip()

    @property
    def using_pysbd(self) -> bool:
        """For tests / observability — confirms which splitter is active."""
        return self._seg is not None

    # ─── pysbd path ────────────────────────────────────────────────

    def _emit_pysbd(self) -> Iterator[str]:
        # pysbd.segment returns *all* sentences in the input. The LAST
        # element might be incomplete (still buffering); the prefix
        # elements are confirmed sentence boundaries.
        sentences = self._seg.segment(self._buf)   # type: ignore[union-attr]
        if len(sentences) > 1:
            for s in sentences[:-1]:
                stripped = s.strip()
                if len(stripped) >= self.min_chars:
                    yield stripped
                # else: too short, swallow it (rare edge case — pysbd
                # rarely emits sub-min sentences; merging back would
                # confuse pysbd state in the next call)
            self._buf = sentences[-1]
            return
        # Single sentence so far — wait for more text. But guard against
        # runaway buffer (e.g. an LLM with no punctuation).
        if len(self._buf) >= self.max_buffer:
            out = self._buf.strip()
            self._buf = ""
            if out:
                yield out

    # ─── regex fallback path ───────────────────────────────────────

    def _emit_regex(self) -> Iterator[str]:
        while True:
            sentence = self._extract_next_sentence_regex()
            if sentence is None:
                return
            yield sentence

    def _extract_next_sentence_regex(self) -> Optional[str]:
        pos = 0
        while True:
            m = _SENTENCE_END_RE.search(self._buf, pos)
            if m is None:
                if len(self._buf) >= self.max_buffer:
                    out = self._buf.strip()
                    self._buf = ""
                    return out or None
                return None
            end = m.end()
            prefix = self._buf[:end]
            if len(prefix.strip()) >= self.min_chars:
                self._buf = self._buf[end:]
                return prefix.strip()
            pos = end


@dataclass
class LowLatencyTTSBuffer:
    """Emit short TTS-ready chunks without waiting for full sentences.

    This is intentionally separate from ``SentenceBuffer``. SentenceBuffer is
    conservative and linguistically cleaner; this buffer optimizes voice-agent
    TTFA by emitting CJK clauses and bounded no-punctuation spans early.
    """

    language: Optional[str] = None
    min_chars: Optional[int] = None
    target_chars: Optional[int] = None
    max_chars: Optional[int] = None
    _buf: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        lang = (self.language or "").strip().lower()
        cjk = lang in ("zh", "chinese", "ja", "japanese", "ko", "korean")
        prefix = "OVS_TTS_LOW_LATENCY_CJK" if cjk else "OVS_TTS_LOW_LATENCY_LATIN"
        default_min = int(os.environ.get(f"{prefix}_MIN_CHARS", "15" if cjk else "24"))
        default_target = int(os.environ.get(f"{prefix}_TARGET_CHARS", "24" if cjk else "48"))
        default_max = int(os.environ.get(f"{prefix}_MAX_CHARS", "40" if cjk else "80"))
        if self.min_chars is None:
            self.min_chars = default_min
        if self.target_chars is None:
            self.target_chars = default_target
        if self.max_chars is None:
            self.max_chars = default_max
        self.min_chars = max(2, int(self.min_chars))
        self.target_chars = max(self.min_chars, int(self.target_chars))
        self.max_chars = max(self.target_chars, int(self.max_chars))

    def add(self, chunk: str) -> Iterator[str]:
        if not chunk:
            return
        self._buf += chunk
        yield from self._emit_ready(final=False)

    def flush(self) -> Iterator[str]:
        yield from self._emit_ready(final=True)

    def is_empty(self) -> bool:
        return not self._buf.strip()

    def _emit_ready(self, *, final: bool) -> Iterator[str]:
        while True:
            part = self._next_chunk(final=final)
            if part is None:
                return
            yield part

    def _next_chunk(self, *, final: bool) -> Optional[str]:
        text = self._buf.lstrip()
        if text != self._buf:
            self._buf = text
        if not self._buf:
            return None

        if final:
            out = self._buf.strip()
            self._buf = ""
            return out if out and _has_speakable_text(out) else None

        is_cjk = _contains_cjk(self._buf) or (self.language or "").lower() in (
            "zh",
            "chinese",
            "ja",
            "japanese",
            "ko",
            "korean",
        )
        hard_breaks = "。！？!?；;\n"
        soft_breaks = "，,、：:" if is_cjk else ",;:"

        hard_idx = self._first_break_index(self._buf, hard_breaks)
        if hard_idx >= 0:
            end = hard_idx + 1
            if len(self._buf[:end].strip()) >= DEFAULT_MIN_SENTENCE_CHARS:
                return self._take(end)

        soft_idx = self._last_break_index(self._buf, soft_breaks, limit=len(self._buf))
        if soft_idx >= 0 and len(self._buf[: soft_idx + 1].strip()) >= self.min_chars:
            return self._take(soft_idx + 1)
        if soft_idx >= 0 and len(self._buf.strip()) >= self.target_chars:
            return self._take(len(self._buf))

        length_cut_threshold = self.max_chars if is_cjk else self.target_chars
        if len(self._buf.strip()) < length_cut_threshold:
            return None

        end = self._choose_length_cut(is_cjk=is_cjk)
        if end <= 0:
            return None
        return self._take(end)

    def _take(self, end: int) -> Optional[str]:
        out = self._buf[:end].strip()
        self._buf = self._buf[end:].lstrip()
        return out or None

    @staticmethod
    def _first_break_index(text: str, chars: str) -> int:
        found = [text.find(ch) for ch in chars if text.find(ch) >= 0]
        return min(found) if found else -1

    @staticmethod
    def _last_break_index(text: str, chars: str, *, limit: int) -> int:
        window = text[:limit]
        found = [window.rfind(ch) for ch in chars if window.rfind(ch) >= 0]
        return max(found) if found else -1

    def _choose_length_cut(self, *, is_cjk: bool) -> int:
        limit = min(len(self._buf), self.max_chars)
        if is_cjk:
            soft_idx = self._last_break_index(self._buf, "，,、：:", limit=limit)
            if soft_idx >= self.min_chars - 1:
                return soft_idx + 1
            return limit

        window = self._buf[:limit]
        for idx in range(len(window) - 1, self.min_chars - 2, -1):
            if window[idx].isspace():
                return idx + 1
        return min(len(self._buf), self.target_chars)

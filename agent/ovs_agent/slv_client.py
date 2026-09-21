"""Async client for SLV's /v2v/stream WebSocket.

Single persistent connection per App lifetime (invariant 1). All public
send methods are serialized through an internal lock so frames never
interleave. Reader task decodes JSON vs binary frames and pushes typed
V2VEvent values onto a queue exposed via `events()`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import struct
from dataclasses import dataclass
from typing import Any, AsyncIterator

import websockets
from websockets.asyncio.client import connect as ws_connect

# Protocol constants, vendored from the server's app/core/v2v.py into a local
# module (invariant 6: single source of truth — ovs_agent/protocol.py MUST
# stay in sync with the server). Vendored so the agent package has NO import
# dependency on the SLV server tree (survives the app/→server/ rename).
from .protocol import (
    CLIENT_ABORT,
    CLIENT_ASR_EOS,
    CLIENT_CONFIG,
    CLIENT_PING,
    CLIENT_TEXT,
    CLIENT_TOOL_ADVERTISE,
    CLIENT_TOOL_RESULT,
    CLIENT_TTS_FLUSH,
    SERVER_ASR_ENDPOINT,
    SERVER_ASR_FINAL,
    SERVER_ASR_PARTIAL,
    SERVER_ERROR,
    SERVER_INPUT_AUDIO_BUFFER_COMMITTED,
    SERVER_INPUT_AUDIO_BUFFER_SPEECH_STARTED,
    SERVER_INPUT_AUDIO_BUFFER_SPEECH_STOPPED,
    SERVER_INPUT_AUDIO_TRANSCRIPTION_COMPLETED,
    SERVER_INPUT_AUDIO_TRANSCRIPTION_DELTA,
    SERVER_RESPONSE_CREATED,
    SERVER_RESPONSE_DONE,
    SERVER_RESPONSE_OUTPUT_AUDIO_DONE,
    SERVER_RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA,
    SERVER_RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE,
    SERVER_SESSION_CREATED,
    SERVER_SESSION_UPDATED,
    SERVER_TOOL_CALL,
    SERVER_TTS_DONE,
    SERVER_TTS_SENTENCE_DONE,
    SERVER_TTS_STARTED,
    CLIENT_SESSION_UPDATE,
    CLIENT_CONVERSATION_ITEM_CREATE,
    CLIENT_CONVERSATION_ITEM_TRUNCATE,
    CLIENT_CONVERSATION_RESET,
    CLIENT_DIRECT_SPEAK,
    CLIENT_RESPONSE_CREATE,
    REALTIME_V2_SUBPROTOCOL,
    SERVER_CONVERSATION_ITEM_TRUNCATED,
    SERVER_CONVERSATION_RESET_DONE,
    SERVER_RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE,
    SERVER_RESPONSE_OUTPUT_ITEM_ADDED,
)

logger = logging.getLogger(__name__)


# ── Typed events ──────────────────────────────────────────────────────


class V2VEvent:
    """Base class for events emitted by SLVClient."""


@dataclass
class SessionCreated(V2VEvent):
    session_id: str
    session: dict[str, Any]


@dataclass
class SessionUpdated(V2VEvent):
    session_id: str
    session: dict[str, Any]


@dataclass
class ResponseCreated(V2VEvent):
    response_id: str
    response: dict[str, Any]


@dataclass
class ResponseOutputAudioDone(V2VEvent):
    response_id: str


@dataclass
class AssistantTranscriptDelta(V2VEvent):
    response_id: str
    delta: str


@dataclass
class AssistantTranscriptDone(V2VEvent):
    response_id: str
    transcript: str


@dataclass
class ResponseOutputItemAdded(V2VEvent):
    response_id: str
    item: dict[str, Any]


@dataclass
class ResponseDone(V2VEvent):
    response_id: str
    status: str
    response: dict[str, Any]


@dataclass
class InputAudioSpeechStarted(V2VEvent):
    item_id: str


@dataclass
class InputAudioSpeechStopped(V2VEvent):
    item_id: str


@dataclass
class InputAudioCommitted(V2VEvent):
    item_id: str


@dataclass
class ConversationItemTruncated(V2VEvent):
    item_id: str
    audio_end_ms: int


@dataclass
class ConversationResetDone(V2VEvent):
    pass


@dataclass
class ASRPartial(V2VEvent):
    text: str
    is_stable: bool = False


@dataclass
class ASREndpoint(V2VEvent):
    pass


@dataclass
class ASRFinal(V2VEvent):
    text: str
    session_complete: bool = True
    duplicate_of_streamed: bool = False
    # Per-utterance detected language as reported by the ASR backend
    # (e.g. "Chinese", "English"). None when the backend doesn't perform
    # language ID. Modes (e.g. InterpreterMode) consume this to pick a
    # translator src language at runtime.
    language: "str | None" = None


@dataclass
class TTSStarted(V2VEvent):
    sentence: str


@dataclass
class TTSSentenceDone(V2VEvent):
    sentence: str


@dataclass
class TTSDone(V2VEvent):
    # SLV signals session_complete=True when the TTS turn ended at a
    # session boundary (slot will release, WS may close), False when
    # the turn ended cleanly but the slot is still held (multi-utterance
    # continuation expected). Race #4: previously this field was dropped
    # by the dataclass and the app treated every TTSDone as session-end,
    # producing spurious reconnects.
    session_complete: bool = True


@dataclass
class TTSAudio(V2VEvent):
    pcm: bytes
    sample_rate: int


@dataclass
class SLVError(V2VEvent):
    message: str


@dataclass
class ServerToolCall(V2VEvent):
    """Server-loop mode (#37 Phase 2-product): SLV's server-side LLM loop
    selected one of the tools this client advertised and asks us to run it
    locally (the resource — e.g. the arm — lives here). The dispatch loop
    executes it against the local registry and replies with a
    CLIENT_TOOL_RESULT frame (see ``SLVClient.send_tool_result``)."""

    id: str
    name: str
    arguments: dict
    timeout_s: float = 15.0


class SLVReconnectError(Exception):
    """Raised when ``SLVClient.reconnect()`` cannot establish a working WS.

    Callers (e.g. ``App.wake()``) catch this to decide policy — most often,
    refuse the wake and stay SLEEPING so the user notices something is
    wrong rather than experiencing a silent mute.
    """


# ── Client ───────────────────────────────────────────────────────────


class SLVClient:
    """One persistent WS to /v2v/stream for the entire App lifetime."""

    def __init__(
        self,
        url: str,
        config: dict[str, Any],
        *,
        protocol_version: int = 1,
    ) -> None:
        self.url = url
        self.config = dict(config)
        if protocol_version not in (1, 2):
            raise ValueError("protocol_version must be 1 or 2")
        self.protocol_version = protocol_version
        # Make sure multi_utterance is on (invariant 1).
        self.config["multi_utterance"] = True

        self._ws: Any | None = None
        self._reader_task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()
        self._queue: asyncio.Queue[V2VEvent] = asyncio.Queue()
        self._tts_sample_rate: int | None = None
        self._closed = False
        # Set when reader exits for any reason; events() uses this to
        # break out of `await queue.get()` instead of hanging forever.
        self._reader_done: asyncio.Event = asyncio.Event()
        # Track wall-clock of last WS activity (recv from server OR send
        # audio/json from us). is_healthy() only checks TCP layer; the
        # server-side ASR session may be GC'd after long idle while the
        # TCP socket stays open. wake() consults seconds_since_activity()
        # to decide whether to force a reconnect (refresh ASR session)
        # even when is_healthy() reports True. See app_base.wake().
        self._last_activity_ts: float = 0.0
        # Race #3: set True while reconnect() is tearing down + reopening.
        # mic_pump consults is_reconnecting() to skip forwarding audio
        # during the outage (otherwise chunks queue up on _send_lock,
        # 0.5s send_audio timeout starves the mic pump, dropped-chunk
        # logs flood, and post-reconnect first utterance still carries
        # pre-reconnect preroll).
        self._reconnecting: bool = False
        # Real reconnect dedup. The _reconnecting flag above is observational
        # only — it does NOT serialize concurrent reconnect() callers. Both the
        # dispatch reader-exit path (app_base._slv_dispatch) and wake() call
        # reconnect(); without this lock they each open a WS, and the server's
        # admission eviction (limit=1) makes each new WS evict the prior one,
        # whose reader-exit triggers yet another reconnect → a self-eviction
        # storm that repeatedly tears the WS down mid-utterance and wedges the
        # ASR worker (cancel-on-ws_close timeout → forced worker restart).
        self._reconnect_lock: asyncio.Lock = asyncio.Lock()
        # Last observed WS close code/reason (set in _reader_loop, cleared on a
        # fresh successful open) — lets the dispatch loop tell an admission
        # eviction (1012 "superseded by new session") apart from a real death.
        self._last_close_code: "int | None" = None
        self._last_close_reason: str = ""
        # Monotonic session generation — bumped each time a fresh WS clears
        # the limiter grace window and becomes healthy. Lets the app detect a
        # session revived by the send-path connect() (mic pump) that never had
        # tools (re)advertised, and lazily re-advertise before the turn (#3).
        self._session_gen: int = 0
        self._active_response_id: str | None = None
        self._active_output_item_id: str | None = None
        self._session_capabilities: dict[str, Any] = {}
        # ── idle keepalive ──────────────────────────────────────────
        # The server's v2v dispatcher reaps a client that sends no frame
        # for OVS_V2V_IDLE_TIMEOUT_S (default 90s) as half-open. In
        # wake-word mode we legitimately send nothing while nobody is
        # talking, so a quiet room guarantees a reap + reconnect every
        # 90s — and an utterance that lands in the reconnect window
        # loses its first syllables. Sending a no-op ping keeps the
        # session alive without weakening the watchdog: a client that
        # actually dies stops pinging and is still reaped on schedule.
        # _last_send_ts tracks outgoing traffic (what the server's
        # watchdog observes), which is NOT the same as _last_activity_ts
        # (incoming, deliberately not touched by outgoing audio).
        self._last_send_ts: float = 0.0
        self._keepalive_task: asyncio.Task | None = None

    def _session_update_payload(self) -> dict[str, Any]:
        """Translate the existing app config into canonical session.update.

        Keeping this conversion in the shared client lets Reachy/reBot/SO-ARM
        migrate together while their human-facing YAML remains stable.
        """
        cfg = self.config
        sample_rate = int(cfg.get("sample_rate", 16000))
        asr_language = cfg.get("asr_language")
        tts_language = cfg.get("tts_language")
        vad = cfg.get("vad", "silero" if asr_language else "none")
        turn_type = "none" if vad in (None, "none") else "server_vad"
        audio_input: dict[str, Any] = {
            "format": {
                "type": "audio/pcm",
                "rate": sample_rate,
                "channels": 1,
                "endianness": "little",
            },
            "turn_detection": {
                "type": turn_type,
                "backend": vad,
                "silence_duration_ms": int(cfg.get("vad_silence_ms", 400)),
                "create_response": bool(cfg.get("create_response", False)),
                "interrupt_response": bool(cfg.get("interrupt_response", True)),
            },
        }
        if asr_language:
            audio_input["transcription"] = {"language": asr_language}
        audio_output: dict[str, Any] = {
            "format": {
                "type": "audio/pcm",
                # Requested output rate is advisory; session.updated carries
                # the effective backend rate used by binary PCM.
                "rate": int(cfg.get("output_sample_rate", sample_rate)),
                "channels": 1,
                "endianness": "little",
            },
        }
        if tts_language:
            audio_output["language"] = tts_language
        if cfg.get("tts_voice") is not None:
            audio_output["voice"] = cfg["tts_voice"]
        if cfg.get("tts_speaker_id") is not None:
            audio_output["speaker_id"] = cfg["tts_speaker_id"]
        if cfg.get("tts_speed") is not None:
            audio_output["speed"] = cfg["tts_speed"]
        return {
            "type": CLIENT_SESSION_UPDATE,
            "session": {
                "type": "realtime",
                "output_modalities": ["audio"],
                "audio": {"input": audio_input, "output": audio_output},
            },
        }

    async def _perform_v2_handshake(self) -> None:
        """Receive session.created, update the session, and await its ack."""
        try:
            created_raw = await asyncio.wait_for(self._ws.recv(), timeout=3.0)
        except asyncio.TimeoutError as exc:
            raise SLVReconnectError("Realtime V2: session.created timeout") from exc
        if not isinstance(created_raw, str):
            raise SLVReconnectError("Realtime V2: first server frame must be JSON")
        try:
            created = json.loads(created_raw)
        except json.JSONDecodeError as exc:
            raise SLVReconnectError("Realtime V2: invalid session.created JSON") from exc
        if created.get("type") != SERVER_SESSION_CREATED:
            raise SLVReconnectError(
                f"Realtime V2: expected session.created, got {created.get('type')!r}"
            )
        await self._handle_json(created_raw)
        await self._ws.send(json.dumps(self._session_update_payload()))
        try:
            updated_raw = await asyncio.wait_for(self._ws.recv(), timeout=3.0)
        except asyncio.TimeoutError as exc:
            raise SLVReconnectError("Realtime V2: session.updated timeout") from exc
        if not isinstance(updated_raw, str):
            raise SLVReconnectError("Realtime V2: session.updated must be JSON")
        try:
            updated = json.loads(updated_raw)
        except json.JSONDecodeError as exc:
            raise SLVReconnectError("Realtime V2: invalid session.updated JSON") from exc
        if updated.get("type") == SERVER_ERROR:
            error = updated.get("error") or {}
            if isinstance(error, dict):
                detail = error.get("message") or error.get("code") or str(error)
            else:
                detail = str(error)
            raise SLVReconnectError(f"Realtime V2 session rejected: {detail}")
        if updated.get("type") != SERVER_SESSION_UPDATED:
            raise SLVReconnectError(
                f"Realtime V2: expected session.updated, got {updated.get('type')!r}"
            )
        await self._handle_json(updated_raw)
        session = updated.get("session") or {}
        capabilities = session.get("capabilities")
        self._session_capabilities = (
            dict(capabilities) if isinstance(capabilities, dict) else {}
        )
        try:
            self._tts_sample_rate = int(
                session["audio"]["output"]["format"].get(
                    "rate",
                    session["audio"]["output"]["format"].get("sample_rate"),
                )
            )
        except (KeyError, TypeError, ValueError):
            # session.created is allowed to omit output when TTS is disabled.
            self._tts_sample_rate = None

    def last_close_code(self) -> "int | None":
        return self._last_close_code

    def last_close_reason(self) -> str:
        return self._last_close_reason

    def session_gen(self) -> int:
        """Monotonic counter of healthy WS sessions opened so far.

        Bumped in ``_open_with_retry`` once a fresh WS survives the limiter
        grace window. The app compares it against the generation it last
        advertised tools to, so a send-path-revived session (opened by the
        mic pump's ``connect()``, which never advertises) gets its tools
        re-advertised before the server-loop LLM turn fires (#3)."""
        return self._session_gen

    def _touch_activity(self) -> None:
        try:
            self._last_activity_ts = asyncio.get_event_loop().time()
        except RuntimeError:
            # No running loop (e.g. unit test using time module). Fall back
            # gracefully — treat as if we just had activity.
            import time as _time
            self._last_activity_ts = _time.monotonic()

    def _touch_send(self) -> None:
        """Record an outgoing frame — what the server's idle watchdog sees."""
        try:
            self._last_send_ts = asyncio.get_event_loop().time()
        except RuntimeError:
            import time as _time
            self._last_send_ts = _time.monotonic()

    def _seconds_since_send(self) -> float:
        if self._last_send_ts == 0:
            return float("inf")
        try:
            now = asyncio.get_event_loop().time()
        except RuntimeError:
            import time as _time
            now = _time.monotonic()
        return now - self._last_send_ts

    def seconds_since_activity(self) -> float:
        """Wall-clock seconds since last observed WS activity.

        Returns inf if no activity has ever been recorded (fresh client
        with no traffic yet — caller should treat as "stale" / reconnect
        candidate after the first turn).
        """
        if self._last_activity_ts == 0:
            return float("inf")
        try:
            now = asyncio.get_event_loop().time()
        except RuntimeError:
            import time as _time
            now = _time.monotonic()
        return now - self._last_activity_ts

    # ── lifecycle ───────────────────────────────────────────────────

    # Time we wait after sending CLIENT_CONFIG before declaring a fresh
    # WS healthy. If the server closes within this window the limiter
    # almost certainly rejected us (SLV's per-client WS slot is still
    # busy with the previous session's teardown); back off and retry.
    _RECONNECT_GRACE_S = 0.05
    # Backoff schedule for the limiter race. Empirically the slot
    # releases inside ~40ms, but Jetson under thermal throttle can be
    # slower; cover up to ~1.75s of contention before giving up.
    _RECONNECT_BACKOFFS = (0.25, 0.5, 1.0)

    # Idle keepalive cadence. Sized against the TIGHTEST deployed
    # OVS_V2V_IDLE_TIMEOUT_S, not the server default: the server defaults to
    # 90s but deploy/docker-compose.jetson-rebot.yml pins 45s (half-open-wedge
    # mitigation on the arm stack). A cadence chosen against 90s leaves only
    # 1.5x margin there, so one late ping reaps a live session. 15s keeps 3x
    # under 45s. The cost of pinging more often is one small JSON frame.
    # <= 0 disables.
    _KEEPALIVE_DEFAULT_S = 15.0
    # Tightest OVS_V2V_IDLE_TIMEOUT_S configured anywhere in this repo. Pinned
    # by tests so lowering that env below this value (or raising the cadence)
    # can't silently erase the margin.
    _TIGHTEST_DEPLOYED_IDLE_TIMEOUT_S = 45.0

    @property
    def _keepalive_interval_s(self) -> float:
        raw = os.getenv("OVS_SLV_KEEPALIVE_S")
        if raw is None or not raw.strip():
            return self._KEEPALIVE_DEFAULT_S
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "OVS_SLV_KEEPALIVE_S=%r is not a number; using %.0fs",
                raw, self._KEEPALIVE_DEFAULT_S,
            )
            return self._KEEPALIVE_DEFAULT_S
        # float() accepts "nan"/"inf". Neither is a usable sleep interval:
        # inf never pings (session reaped as if there were no keepalive at
        # all), and nan makes asyncio.sleep() raise, killing the task with no
        # diagnostic. Both are worse than the default, so reject them.
        if not math.isfinite(value):
            logger.warning(
                "OVS_SLV_KEEPALIVE_S=%r is not finite; using %.0fs",
                raw, self._KEEPALIVE_DEFAULT_S,
            )
            return self._KEEPALIVE_DEFAULT_S
        return value

    async def _keepalive_loop(self, interval_s: float) -> None:
        """Send a no-op ping whenever the send path has been idle too long.

        Only fires on genuine idleness: any real frame (audio or control)
        touches ``_last_send_ts``, so an active turn adds no wire traffic.
        Never revives a dead WS — ``connect_if_dead=False`` — because
        reconnect ownership belongs to ``reconnect()`` / the dispatch loop;
        a keepalive opening a fresh session behind their back would race
        the server's single-session limiter.
        """
        while not self._closed:
            idle = self._seconds_since_send()
            if idle < interval_s:
                await asyncio.sleep(max(0.1, interval_s - idle))
                continue
            if self._ws is None or self._reconnecting:
                # Dead or mid-reconnect: stay quiet, re-check shortly.
                await asyncio.sleep(min(1.0, interval_s))
                continue
            try:
                await self._send_json(
                    {"type": CLIENT_PING}, connect_if_dead=False
                )
            except Exception as e:  # never let keepalive kill the session
                logger.debug("keepalive ping failed (ignored): %s", e)
            await asyncio.sleep(interval_s)

    def _start_keepalive(self) -> None:
        self._stop_keepalive()
        interval = self._keepalive_interval_s
        if interval <= 0 or self._closed:
            return
        self._keepalive_task = asyncio.ensure_future(
            self._keepalive_loop(interval)
        )

    def _stop_keepalive(self) -> None:
        task = self._keepalive_task
        self._keepalive_task = None
        if task is not None and not task.done():
            task.cancel()

    async def connect(self) -> None:
        if self._ws is not None:
            return  # idempotent
        await self._open_with_retry()

    def is_healthy(self) -> bool:
        """Best-effort liveness check for the SLV WebSocket.

        Returns False when:
        - the client is closed
        - the WS handle is missing (last send observed ``ConnectionClosed``
          and nulled it)
        - the reader task is missing or has already exited (server closed
          the stream, or ``_open_with_retry`` aborted before launching it)

        Cheap to call — no I/O. Used by ``App.wake()`` to decide whether
        to skip a wake when the previous turn's reconnect fail-silently
        left us with a dead stream.
        """
        if self._closed:
            return False
        if self._ws is None:
            return False
        if self._reader_task is None or self._reader_task.done():
            return False
        return True

    async def reconnect(self) -> None:
        """Tear down current WS and open a fresh one, replaying config.

        Self-healing against SLV's session-limiter race (server `app/main.py`
        ``try_acquire_ws`` admission): the previous session's WS slot is
        held until a teardown chain (dispatcher cancel → ASR cancel →
        ws.close → manager unregister → token release) completes. A
        reconnect that arrives inside that 3–40 ms window is closed with
        WS code 4429. We detect the immediate close by waiting on the
        reader-done signal for ``_RECONNECT_GRACE_S`` after sending the
        config frame; if reader fires inside the grace, we back off and
        retry.
        """
        if self._closed:
            return
        # Serialize reconnects: the dispatch reader-exit path and wake() both
        # call reconnect(); the lock makes each reconnect close-then-open
        # without overlapping another's open, so the server's admission
        # eviction (limit=1) never fires (no two live WS at once) → no
        # self-eviction storm. We deliberately do NOT short-circuit on
        # is_healthy() here: wake()'s idle>30s call must force a fresh ASR
        # session even when the WS is TCP-healthy (else the long-idle "silent
        # mute" bug returns). Redundant-reconnect suppression lives in the
        # dispatch guard (is_healthy()/is_reconnecting() → resume, don't
        # reconnect), not here.
        async with self._reconnect_lock:
            if self._closed:
                return
            self._reconnecting = True
            try:
                async with self._send_lock:
                    old_reader = self._reader_task
                    old_ws = self._ws
                    self._ws = None
                    self._reader_task = None
                    if old_reader is not None and not old_reader.done():
                        old_reader.cancel()
                        try:
                            await old_reader
                        except (asyncio.CancelledError, Exception):
                            pass
                    if old_ws is not None:
                        try:
                            await old_ws.close()
                        except Exception:
                            pass
                    # Grace: let server SessionLimiter (limit=1, Qwen3-ASR
                    # worker is single-concurrent) observe the close + release
                    # the slot before we open a fresh WS. With proactive
                    # reconnects on tts_done / wake reverted (SLV server
                    # v1.15+ ASR turn timeout handles stuck workers),
                    # reconnect now only fires on genuine WS death — close-
                    # before-open races are no longer densely triggered, so
                    # 50ms of defensive grace is sufficient (was 150ms when
                    # proactive reconnect was active).
                    await asyncio.sleep(0.05)
                    self._reader_done.clear()
                    self._tts_sample_rate = None
                    await self._open_with_retry()
            finally:
                self._reconnecting = False

    def is_reconnecting(self) -> bool:
        """True while reconnect() is mid-flight (race #3 gate)."""
        return self._reconnecting

    async def _open_with_retry(self) -> None:
        """Open a WS and verify it survived the limiter grace window.

        Must be called with ``self._send_lock`` already held by the
        caller (or in a single-threaded init path like ``connect``).
        """
        attempts = list(self._RECONNECT_BACKOFFS) + [None]
        for attempt_idx, backoff in enumerate(attempts):
            self._reader_done.clear()
            self._tts_sample_rate = None
            connect_kwargs = {"max_size": None}
            if self.protocol_version == 2:
                connect_kwargs["subprotocols"] = [REALTIME_V2_SUBPROTOCOL]
            self._ws = await ws_connect(self.url, **connect_kwargs)
            try:
                if self.protocol_version == 2:
                    if self._ws.subprotocol != REALTIME_V2_SUBPROTOCOL:
                        raise SLVReconnectError(
                            "Realtime V2 subprotocol was not accepted by server"
                        )
                    await self._perform_v2_handshake()
                else:
                    await self._ws.send(json.dumps({"type": CLIENT_CONFIG, **self.config}))
            except websockets.ConnectionClosed as e:
                # Server slammed the door before we could send config
                # (4429 limiter race, etc.). Treat as failed attempt and
                # fall through to backoff. Without this catch the
                # exception escapes wake() as "unexpected error".
                logger.info(
                    "SLV reconnect attempt %d: send(config) closed by server (%s)",
                    attempt_idx + 1, e,
                )
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None
                if backoff is None:
                    raise SLVReconnectError(
                        f"SLV reconnect: server closed CLIENT_CONFIG send on all "
                        f"{len(attempts)} attempts ({e})"
                    )
                await asyncio.sleep(backoff)
                continue
            except SLVReconnectError:
                # Negotiation/configuration failures are deterministic. Close
                # the half-open socket and surface the reason instead of
                # leaking it or retrying the same incompatible handshake.
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None
                raise
            self._reader_task = asyncio.create_task(
                self._reader_loop(), name="slv-reader"
            )
            try:
                # If the server rejects this connection (limiter race,
                # bad config, etc.) the reader sees ConnectionClosed
                # almost immediately and sets _reader_done. Survive the
                # grace window → connection is healthy, return.
                await asyncio.wait_for(
                    self._reader_done.wait(), timeout=self._RECONNECT_GRACE_S
                )
            except asyncio.TimeoutError:
                # Healthy: reader is still running after grace window.
                # Mark fresh WS as just-active so the next wake doesn't
                # see "infinite idle" and immediately re-reconnect on top
                # of our brand new session (and trip the limiter).
                self._touch_activity()
                # Fresh healthy WS — clear any stale close code/reason so the
                # dispatch loop doesn't misread a past eviction.
                self._last_close_code = None
                self._last_close_reason = ""
                # New session is live: advance the generation so the app can
                # tell this session apart from the one it last advertised
                # tools to (send-path revival → re-advertise, #3).
                self._session_gen += 1
                # Fresh healthy session: (re)arm the idle keepalive so a
                # quiet wake-word room never trips the server's half-open
                # watchdog. Counts from now — the config frame we just sent
                # is itself the first piece of outgoing traffic.
                self._touch_send()
                self._start_keepalive()
                return
            # Reader fired → connection died inside grace window.
            # Tear down what we just built and back off. Capture the close
            # code first: the reader set _last_close_code before signalling
            # _reader_done, and the teardown below clears nothing, but draining
            # the queue / a later attempt can overwrite it — snapshot now so we
            # can name a 4429 (session-limit / slot-busy) rejection explicitly.
            rejected_code = self._last_close_code
            dead_reader = self._reader_task
            dead_ws = self._ws
            self._ws = None
            self._reader_task = None
            if dead_reader is not None and not dead_reader.done():
                dead_reader.cancel()
                try:
                    await dead_reader
                except (asyncio.CancelledError, Exception):
                    pass
            if dead_ws is not None:
                try:
                    await dead_ws.close()
                except Exception:
                    pass
            # Drain any SLVError the reader may have queued so subsequent
            # events() consumers don't see this rejection.
            try:
                while not self._queue.empty():
                    _ = self._queue.get_nowait()
            except Exception:
                pass
            is_4429 = rejected_code == 4429
            if backoff is None:
                raise SLVReconnectError(
                    f"SLV reconnect: server closed within {self._RECONNECT_GRACE_S}s "
                    f"on all {len(attempts)} attempts "
                    f"({'4429 session-limit/slot-busy' if is_4429 else 'limiter race?'})"
                )
            logger.warning(
                "SLV reconnect attempt %d rejected%s within %.0fms; retrying in %.2fs",
                attempt_idx + 1,
                " (4429 session-limit/slot-busy)" if is_4429 else "",
                self._RECONNECT_GRACE_S * 1000, backoff,
            )
            await asyncio.sleep(backoff)

    async def close(self) -> None:
        self._closed = True
        self._stop_keepalive()
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._ws = None

    # ── send helpers ────────────────────────────────────────────────

    async def _send_json(
        self, payload: dict[str, Any], *, connect_if_dead: bool = True
    ) -> None:
        async with self._send_lock:
            if self._ws is None:
                if self._closed:
                    return
                if not connect_if_dead:
                    # The caller's payload is bound to the session that just
                    # died (e.g. a tool_result carrying a call_id the server
                    # issued on the OLD session). Auto-connecting would deliver
                    # it to a FRESH session that never made that call → it's
                    # silently ignored and the server turn stalls (#6). Drop it
                    # instead; the new session will run its own clean turn.
                    logger.info(
                        "_send_json: WS dead — dropping %s (not auto-connecting; "
                        "would misdeliver to a fresh session)",
                        payload.get("type"),
                    )
                    return
                await self.connect()
            try:
                await self._ws.send(json.dumps(payload))
                self._touch_send()
            except websockets.ConnectionClosed:
                logger.info("send_json: WS closed mid-send, dropping %s", payload.get("type"))
                # Null the dead handle so the next caller triggers
                # connect() instead of replaying onto the closed WS.
                self._ws = None

    async def send_audio(self, pcm: bytes) -> None:
        async with self._send_lock:
            if self._ws is None:
                if self._closed:
                    return
                await self.connect()
            try:
                await self._ws.send(pcm)
                # Outgoing audio DOES count for the server's idle watchdog
                # (it sees frames, not their direction), so it suppresses
                # keepalives during a live turn. Distinct from _activity_ts
                # below, which is incoming-only on purpose.
                self._touch_send()
                # Do NOT touch activity on outgoing audio — the mute bug
                # is exactly "we keep sending into a dead session". Only
                # server-originated frames (handled in _handle_json /
                # _handle_binary) signal that the SLV session is alive.
            except websockets.ConnectionClosed:
                # Reader will notice and signal _reader_done; dispatch can
                # decide to reconnect. Audio chunks during reconnect are
                # naturally dropped — first ASR utterance after reconnect
                # picks up from the new chunks.
                self._ws = None

    async def send_text(self, text: str) -> None:
        if text:
            logger.info("SLV send text chunk len=%d", len(text))
        await self._send_json({"type": CLIENT_TEXT, "text": text})

    async def flush_tts(self) -> None:
        logger.info("SLV send tts_flush")
        await self._send_json({"type": CLIENT_TTS_FLUSH})

    async def speak(self, text: str, *, conversation: str = "none") -> None:
        """Speak deterministic text without adding it to model history."""
        if not text or not text.strip():
            return
        if getattr(self, "protocol_version", 1) == 2:
            await self._send_json({
                "type": CLIENT_DIRECT_SPEAK,
                "speech": {
                    "text": text,
                    "conversation": conversation,
                },
            })
            return
        await self.send_text(text)
        await self.flush_tts()

    async def update_session(self, session: dict[str, Any]) -> None:
        """Apply a provider-neutral partial Realtime session update.

        Device applications use this for dynamic, turn-scoped context such as
        the people currently visible to a robot.  Provider-specific field
        translation remains in the gateway; applications only send canonical
        Realtime V2 session fields.
        """
        if getattr(self, "protocol_version", 1) != 2:
            raise RuntimeError("session.update requires Realtime V2")
        if not isinstance(session, dict) or not session:
            raise ValueError("session update must be a non-empty object")
        await self._send_json({
            "type": CLIENT_SESSION_UPDATE,
            "session": dict(session),
        })

    async def create_response(self, response: dict[str, Any] | None = None) -> None:
        """Start a response after an application-controlled context update."""
        if getattr(self, "protocol_version", 1) != 2:
            raise RuntimeError("response.create requires Realtime V2")
        payload: dict[str, Any] = {"type": CLIENT_RESPONSE_CREATE}
        if response:
            payload["response"] = dict(response)
        await self._send_json(payload)

    async def abort(self, *, keep_asr: bool = False) -> None:
        """Cancel the assistant turn in flight.

        ``keep_asr=True`` asks the server to cancel TTS only and leave the
        ASR utterance running — what a *speech-driven* barge-in wants, since
        the in-flight utterance is the one the user is interrupting with.
        Cancelling it drops everything said before the barge-in threshold
        fired. Servers that predate the field ignore it (and still cancel
        ASR), so this degrades to the old behaviour rather than failing.
        """
        if self.protocol_version == 2:
            payload: dict[str, object] = {
                "type": "response.cancel",
                "response_id": self._active_response_id,
            }
        else:
            payload = {"type": CLIENT_ABORT}
        if keep_asr:
            payload["keep_asr"] = True
        await self._send_json(payload)

    async def asr_eos(self) -> None:
        if self.protocol_version == 2:
            await self._send_json({"type": "input_audio_buffer.commit"})
        else:
            await self._send_json({"type": CLIENT_ASR_EOS})

    async def truncate_active_response(self, audio_end_ms: int) -> None:
        """Trim unheard assistant audio from provider conversation state."""
        item_id = self._active_output_item_id
        if getattr(self, "protocol_version", 1) != 2 or not item_id:
            return
        if self._session_capabilities.get("conversation_truncate") is False:
            return
        await self._send_json({
            "type": CLIENT_CONVERSATION_ITEM_TRUNCATE,
            "item_id": item_id,
            "content_index": 0,
            "audio_end_ms": max(0, int(audio_end_ms)),
        })

    async def reset_conversation(self) -> None:
        if getattr(self, "protocol_version", 1) == 2:
            await self._send_json({"type": CLIENT_CONVERSATION_RESET})

    # ── server-loop mode (#37 Phase 2-product) ───────────────────────

    async def advertise_tools(
        self,
        tools: list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
        llm_params: dict[str, Any] | None = None,
    ) -> None:
        """Upload local tool schemas to SLV so the server-side LLM loop can
        select them (and proxy execution back via SERVER_TOOL_CALL).

        ``tools`` is the OpenAI-style ``tools[]`` list from
        ``ToolRegistry.list_openai_tools(...)``. ``system_prompt`` and
        ``llm_params`` are optional and only included when provided so the
        server can run the loop with the same prompt / sampling the client
        would have used locally. Only sent in server-loop mode — a legacy
        client never calls this.
        """
        if getattr(self, "protocol_version", 1) == 2:
            canonical_tools: list[dict[str, Any]] = []
            for entry in tools or []:
                if not isinstance(entry, dict):
                    continue
                fn = entry.get("function")
                fn = fn if isinstance(fn, dict) else entry
                canonical = {
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters") or {
                        "type": "object", "properties": {},
                    },
                }
                policy = {
                    key: entry[key]
                    for key in (
                        "timeout_s", "preamble_text", "completion_text",
                        "response_mode",
                    )
                    if key in entry
                }
                if policy:
                    canonical["x_v2v"] = policy
                canonical_tools.append(canonical)
            session: dict[str, Any] = {"tools": canonical_tools}
            if system_prompt is not None:
                session["instructions"] = system_prompt
            if llm_params:
                session["x_v2v"] = {"llm_params": dict(llm_params)}
            payload = {"type": CLIENT_SESSION_UPDATE, "session": session}
        else:
            payload = {
                "type": CLIENT_TOOL_ADVERTISE,
                "tools": list(tools or []),
            }
            if system_prompt is not None:
                payload["system_prompt"] = system_prompt
            if llm_params:
                payload["llm_params"] = dict(llm_params)
        logger.info(
            "SLV advertise %d tool(s) (sp_len=%d, llm_params=%s)",
            len(tools or []),
            len(system_prompt or ""),
            bool(llm_params),
        )
        await self._send_json(payload)

    async def send_tool_result(
        self,
        call_id: str,
        name: str,
        *,
        ok: bool,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Return the outcome of a SERVER_TOOL_CALL to SLV.

        On success pass ``ok=True`` + ``result`` (the handler's dict). On
        failure pass ``ok=False`` + ``error`` (a human-readable string).
        Mirrors the wire shape in spec §4.
        """
        if getattr(self, "protocol_version", 1) == 2:
            output: dict[str, Any] = {"ok": bool(ok), "name": name}
            if ok:
                output["result"] = result if isinstance(result, dict) else {}
            else:
                output["error"] = error or "tool execution failed"
            payload = {
                "type": CLIENT_CONVERSATION_ITEM_CREATE,
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(output, ensure_ascii=False),
                },
            }
        else:
            payload = {
                "type": CLIENT_TOOL_RESULT,
                "call_id": call_id,
                "id": call_id,
                "name": name,
                "ok": bool(ok),
            }
            if ok:
                payload["result"] = result if isinstance(result, dict) else {}
            else:
                payload["error"] = error or "tool execution failed"
        # %r (not %s) so an empty correlation id shows as '' and a None as
        # None — they diagnose different upstream bugs (see #B1 round2-stall).
        logger.info("SLV tool_result call_id=%r name=%r ok=%s", call_id, name, ok)
        # A tool_result is bound to the session that issued the SERVER_TOOL_CALL.
        # If that WS died mid-dispatch, do NOT auto-connect — the call_id is
        # meaningless to a fresh session (#6). Drop it; the live session (if any)
        # will drive its own turn.
        await self._send_json(payload, connect_if_dead=False)

    # ── reader ──────────────────────────────────────────────────────

    async def events(self) -> AsyncIterator[V2VEvent]:
        while True:
            # Drain anything already queued first so SLVError emitted on
            # reader exit is still surfaced to the consumer.
            if not self._queue.empty():
                yield self._queue.get_nowait()
                continue
            if self._reader_done.is_set() or self._closed:
                if (
                    not self._closed
                    and self._reader_task is not None
                    and not self._reader_task.done()
                ):
                    # A manual reconnect can cancel an old reader and start a
                    # new one while this iterator is still alive. The old
                    # reader's finally may set the shared event after the new
                    # reader is already running; don't make dispatch exit and
                    # reconnect a second time in that case.
                    self._reader_done.clear()
                    continue
                return
            get_task = asyncio.create_task(self._queue.get())
            done_task = asyncio.create_task(self._reader_done.wait())
            try:
                done, pending = await asyncio.wait(
                    {get_task, done_task}, return_when=asyncio.FIRST_COMPLETED
                )
            except asyncio.CancelledError:
                get_task.cancel()
                done_task.cancel()
                raise
            for t in pending:
                t.cancel()
            if get_task in done:
                yield get_task.result()
            else:
                if (
                    not self._closed
                    and self._reader_task is not None
                    and not self._reader_task.done()
                ):
                    self._reader_done.clear()
                    continue
                # Reader finished; flush any final items it pushed.
                while not self._queue.empty():
                    yield self._queue.get_nowait()
                return

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                if isinstance(msg, (bytes, bytearray)):
                    await self._handle_binary(bytes(msg))
                else:
                    await self._handle_json(msg)
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed as e:
            # Capture close code/reason so the dispatch loop can tell an
            # admission eviction (1012 "superseded by new session") apart from
            # a genuine connection loss and avoid the self-eviction reconnect
            # storm. Defensive across websockets versions (rcvd Close frame vs
            # direct .code/.reason).
            try:
                _rcvd = getattr(e, "rcvd", None)
                self._last_close_code = (
                    getattr(_rcvd, "code", None) if _rcvd is not None
                    else getattr(e, "code", None)
                )
                self._last_close_reason = (
                    (getattr(_rcvd, "reason", None) if _rcvd is not None
                     else getattr(e, "reason", None)) or ""
                )
            except Exception:  # pragma: no cover - defensive
                self._last_close_code, self._last_close_reason = None, ""
            logger.info(
                "SLV WS closed: code=%s reason=%r (%s)",
                self._last_close_code, self._last_close_reason, e,
            )
            await self._queue.put(SLVError(f"connection closed: {e}"))
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("SLV reader crashed")
            await self._queue.put(SLVError(str(e)))
        finally:
            # Wake any consumer blocked on events().
            self._reader_done.set()

    async def _handle_binary(self, data: bytes) -> None:
        self._touch_activity()
        if self._tts_sample_rate is None:
            if len(data) < 4:
                await self._queue.put(SLVError("first binary frame < 4 bytes"))
                return
            (sr,) = struct.unpack("<I", data[:4])
            self._tts_sample_rate = sr
            pcm = data[4:]
            logger.info("SLV tts sample_rate=%d first_pcm=%d", sr, len(pcm))
            if pcm:
                await self._queue.put(TTSAudio(pcm=pcm, sample_rate=sr))
            return
        logger.info("SLV tts audio bytes=%d", len(data))
        await self._queue.put(TTSAudio(pcm=data, sample_rate=self._tts_sample_rate))

    async def _handle_json(self, raw: str) -> None:
        self._touch_activity()
        try:
            evt = json.loads(raw)
        except json.JSONDecodeError as e:
            await self._queue.put(SLVError(f"bad json: {e}"))
            return
        t = evt.get("type")
        # Per-event trace (every partial/final/tts/tool frame) — debug only;
        # too chatty for production (fires several times per second on an
        # open mic). Enable with OVS log level DEBUG when investigating.
        logger.debug("SLV evt: %s", {k: v for k, v in evt.items() if k != "text" or len(str(v)) < 100})
        if t == SERVER_ASR_PARTIAL:
            await self._queue.put(
                ASRPartial(text=evt.get("text", ""), is_stable=bool(evt.get("is_stable", False)))
            )
        elif t == SERVER_ASR_ENDPOINT:
            await self._queue.put(ASREndpoint())
        elif t == SERVER_ASR_FINAL:
            await self._queue.put(
                ASRFinal(
                    text=evt.get("text", ""),
                    session_complete=bool(evt.get("session_complete", True)),
                    duplicate_of_streamed=bool(evt.get("duplicate_of_streamed", False)),
                    language=evt.get("language"),
                )
            )
        elif t == SERVER_TTS_STARTED:
            logger.info("SLV tts_started sentence=%r", evt.get("sentence", "")[:80])
            await self._queue.put(TTSStarted(sentence=evt.get("sentence", "")))
        elif t == SERVER_TTS_SENTENCE_DONE:
            await self._queue.put(TTSSentenceDone(sentence=evt.get("sentence", "")))
        elif t == SERVER_TTS_DONE:
            session_complete = bool(evt.get("session_complete", True))
            logger.info("SLV tts_done session_complete=%s", session_complete)
            await self._queue.put(TTSDone(session_complete=session_complete))
        elif t == SERVER_INPUT_AUDIO_BUFFER_SPEECH_STARTED:
            await self._queue.put(
                InputAudioSpeechStarted(item_id=str(evt.get("item_id") or ""))
            )
        elif t == SERVER_INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
            await self._queue.put(
                InputAudioSpeechStopped(item_id=str(evt.get("item_id") or ""))
            )
        elif t == SERVER_INPUT_AUDIO_BUFFER_COMMITTED:
            await self._queue.put(
                InputAudioCommitted(item_id=str(evt.get("item_id") or ""))
            )
        elif t == SERVER_INPUT_AUDIO_TRANSCRIPTION_DELTA:
            await self._queue.put(
                ASRPartial(
                    text=str(evt.get("delta") or ""),
                    is_stable=bool(evt.get("is_stable", False)),
                )
            )
        elif t == SERVER_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
            await self._queue.put(
                ASRFinal(
                    text=str(evt.get("transcript") or ""),
                    session_complete=False,
                    duplicate_of_streamed=False,
                    language=evt.get("language"),
                )
            )
        elif t == SERVER_SESSION_CREATED:
            session = evt.get("session")
            session = session if isinstance(session, dict) else {}
            await self._queue.put(
                SessionCreated(
                    session_id=str(session.get("id") or ""),
                    session=session,
                )
            )
        elif t == SERVER_SESSION_UPDATED:
            session = evt.get("session")
            session = session if isinstance(session, dict) else {}
            await self._queue.put(
                SessionUpdated(
                    session_id=str(session.get("id") or ""),
                    session=session,
                )
            )
        elif t == SERVER_RESPONSE_CREATED:
            response = evt.get("response")
            response = response if isinstance(response, dict) else {}
            self._active_response_id = str(
                response.get("id") or evt.get("response_id") or ""
            )
            await self._queue.put(
                ResponseCreated(
                    response_id=self._active_response_id,
                    response=response,
                )
            )
        elif t == SERVER_RESPONSE_OUTPUT_AUDIO_DONE:
            await self._queue.put(
                ResponseOutputAudioDone(
                    response_id=str(evt.get("response_id") or ""),
                )
            )
        elif t == SERVER_RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA:
            await self._queue.put(AssistantTranscriptDelta(
                response_id=str(evt.get("response_id") or ""),
                delta=str(evt.get("delta") or ""),
            ))
        elif t == SERVER_RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE:
            await self._queue.put(AssistantTranscriptDone(
                response_id=str(evt.get("response_id") or ""),
                transcript=str(evt.get("transcript") or ""),
            ))
        elif t == SERVER_RESPONSE_DONE:
            response = evt.get("response")
            response = response if isinstance(response, dict) else {}
            response_id = str(response.get("id") or evt.get("response_id") or "")
            if response_id == self._active_response_id:
                self._active_response_id = None
                self._active_output_item_id = None
            await self._queue.put(
                ResponseDone(
                    response_id=response_id,
                    status=str(response.get("status") or "completed"),
                    response=response,
                )
            )
        elif t == SERVER_RESPONSE_OUTPUT_ITEM_ADDED:
            item = evt.get("item")
            item = item if isinstance(item, dict) else {}
            if item.get("type") == "message" and item.get("role") == "assistant":
                self._active_output_item_id = str(item.get("id") or "") or None
            await self._queue.put(ResponseOutputItemAdded(
                response_id=str(evt.get("response_id") or ""),
                item=item,
            ))
        elif t == SERVER_RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
            raw_arguments = evt.get("arguments") or "{}"
            try:
                arguments = (
                    json.loads(raw_arguments)
                    if isinstance(raw_arguments, str)
                    else raw_arguments
                )
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            extension = evt.get("x_v2v")
            extension = extension if isinstance(extension, dict) else {}
            await self._queue.put(ServerToolCall(
                id=str(evt.get("call_id") or ""),
                name=str(evt.get("name") or ""),
                arguments=arguments if isinstance(arguments, dict) else {},
                timeout_s=float(extension.get("timeout_s", 15.0)),
            ))
        elif t == SERVER_CONVERSATION_ITEM_TRUNCATED:
            await self._queue.put(ConversationItemTruncated(
                item_id=str(evt.get("item_id") or ""),
                audio_end_ms=int(evt.get("audio_end_ms") or 0),
            ))
        elif t == SERVER_CONVERSATION_RESET_DONE:
            await self._queue.put(ConversationResetDone())
        elif t == "x_v2v.tts_sentence.started":
            await self._queue.put(TTSStarted(sentence=str(evt.get("sentence") or "")))
        elif t == "x_v2v.tts_sentence.done":
            await self._queue.put(
                TTSSentenceDone(sentence=str(evt.get("sentence") or ""))
            )
        elif t == SERVER_TOOL_CALL:
            # Server-loop mode: SLV asks us to run a local tool. Additive —
            # only emitted when the server-side loop is enabled, so a legacy
            # always-client-loop session never reaches this branch.
            args = evt.get("arguments")
            await self._queue.put(
                ServerToolCall(
                    # The server (voxedge tool_registry) emits the correlation
                    # id in field "call_id"; older/other frames may use "id".
                    # Read call_id FIRST — reading only "id" yielded an empty
                    # string, so send_tool_result echoed back no call_id, the
                    # server's resolve_remote never matched, and every remote
                    # tool dispatch blocked the full 15s timeout (the "挥手 →
                    # 已挥手 several seconds" round2 stall).
                    id=str(evt.get("call_id") or evt.get("id", "")),
                    name=str(evt.get("name", "")),
                    arguments=args if isinstance(args, dict) else {},
                    timeout_s=float(evt.get("timeout_s", 15.0)),
                )
            )
        elif t == SERVER_ERROR:
            error = evt.get("error", "unknown")
            if isinstance(error, dict):
                message = str(error.get("message") or error.get("code") or "unknown")
            else:
                message = str(error)
            await self._queue.put(SLVError(message))
        else:
            logger.debug("Unknown SLV message type: %r", t)

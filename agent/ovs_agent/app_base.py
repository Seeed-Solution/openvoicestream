"""BaseApp orchestrator -- wires SLV + LLM + Audio + Plugins.

Lifecycle:
  1. `await slv.connect()` (one persistent WS).
  2. Spawn `_mic_pump_task` (mic -> WS binary) and `_slv_dispatch_task`
     (WS events -> hooks / on_user_utterance routing).
  3. Call each registered plugin's `start()`.
  4. Wait on shutdown event.
  5. `shutdown()` reverses everything.

Plugin hook dispatch is parallel via `asyncio.gather(return_exceptions=True)`
so observers don't block one another or the dispatch loop.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import signal
import time
from typing import TYPE_CHECKING


class TypedLLMError(RuntimeError):
    """RuntimeError subclass carrying a structured payload for the dashboard.

    Keeps full backward compatibility with the old on_error contract
    (``isinstance(exc, RuntimeError)`` + ``str(exc)`` continue to work),
    but exposes ``.payload`` so plugins like debug_dashboard can render
    typed/coloured errors instead of opaque strings.
    """

    def __init__(
        self,
        type_: str,
        message: str,
        *,
        exc_class: str = "",
        **extra,
    ) -> None:
        super().__init__(message)
        self.payload: dict = {
            "type": type_,
            "message": message,
            "exc_class": exc_class,
            "timestamp": time.time(),
            **extra,
        }

from .app_mode import LLMTimeoutError
from .audio.devices import _is_auto_selector as _is_auto_audio_selector
from .audio.profiles import resolve_mic_setup
from .audio_io import AudioIO
from .config import Config
from .event_bus import EventBus
from .llm import EdgeLLMBackend, LLMBackend, LLMStreamError, OpenAICompatBackend, NoopLLM
from .plugins.llm_availability import LLMUnavailable
from .session import Session
from .state import ConvState
# Importing .tools also imports .tools.builtin which @-decorates the
# built-in tools onto default_registry as an import side-effect. Cheap
# (no IO, no model load) so we pay it unconditionally — actual use is
# gated by config.tools_enabled per turn in app_mode.
from .tools import default_registry as _default_tool_registry
from .translator import CTranslate2Translator, NoopTranslator, TranslatorBackend
from .vad import create_vad
from .slv_client import (
    AssistantTranscriptDelta,
    AssistantTranscriptDone,
    ASREndpoint,
    ASRFinal,
    ASRPartial,
    InputAudioSpeechStarted,
    InputAudioSpeechStopped,
    InputAudioCommitted,
    ConversationItemTruncated,
    ConversationResetDone,
    ResponseCreated,
    ResponseDone,
    ResponseOutputAudioDone,
    ServerToolCall,
    SLVClient,
    SLVError,
    SLVReconnectError,
    TTSAudio,
    TTSDone,
    TTSSentenceDone,
    TTSStarted,
)

if TYPE_CHECKING:
    from .plugin import Plugin

logger = logging.getLogger(__name__)


# ── Low-signal ASR final filter ─────────────────────────────────────
# An open-mic always-on pipeline will, fairly often, emit ASR finals
# that are just one Chinese character or one English letter — these
# are almost always noise / ambient speech / breath the silero VAD
# happened to clip out, NOT real intent. Routing them to the LLM
# triggers a "safe fallback" reply ("我在这里呢…") that, after a few
# repeats, locks a small quantised model into an echo loop where it
# emits the same fallback forever no matter what you say next.
_INTERJECTIONS: frozenset[str] = frozenset(
    {
        # Chinese: noncommittal acknowledgements / filler.
        "嗯", "啊", "哦", "呃", "唉", "诶", "哎", "噢", "唔", "呀", "哈",
        "哇", "呢", "吧", "吗", "呐", "嘛", "诶呀", "啊啊", "嗯嗯",
        # English: same idea — too short to convey intent on a voice mic.
        "uh", "um", "ah", "oh", "ok", "okay", "hmm", "huh", "yeah", "yep",
        "you", "the", "and", "a", "i",
    }
)


def _strip_for_signal(text: str) -> str:
    """Return the input with whitespace + common punctuation removed,
    lowercased, for low-signal comparison against ``_INTERJECTIONS``.
    Keeps Chinese chars and ASCII alphanumerics as-is.
    """
    import unicodedata
    out: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        # Drop separators / punctuation; keep letters and digits.
        if cat[0] in {"L", "N"}:
            out.append(ch)
    return "".join(out).lower()


def _is_ascii_word_char(ch: str) -> bool:
    return ch.isascii() and (ch.isalnum() or ch == "_")


def _join_asr_segment_text(left: str, right: str) -> str:
    """Join two ASR segment texts for one utterance.

    Each side is stripped; an empty side yields the other unchanged.
    A space is inserted at the seam ONLY when both sides end/start with
    an ASCII word character — Chinese segments must stay unspaced.
    """
    left = (left or "").strip()
    right = (right or "").strip()
    if not left:
        return right
    if not right:
        return left
    if _is_ascii_word_char(left[-1]) and _is_ascii_word_char(right[0]):
        return left + " " + right
    return left + right


def _normalize_tool_trigger_text(text: str) -> str:
    """Normalize text for simple phrase-containment tool guards."""
    import unicodedata
    out: list[str] = []
    for ch in text or "":
        if unicodedata.category(ch)[0] in {"L", "N"}:
            out.append(ch.lower())
    return "".join(out)


def _extract_tool_trigger_phrases(description: str) -> list[str]:
    """Extract quoted trigger phrases from a tool description."""
    import re
    match = re.search(r"(?:Triggers?|Trigger words?)\s*:\s*([^.。]*)", description or "", re.I | re.S)
    if not match:
        return []
    return [p.strip() for p in re.findall(r"""["']([^"']+)["']""", match.group(1)) if p.strip()]


def _build_llm(config: Config) -> LLMBackend:
    backend = config.llm_backend.lower()
    if backend == "noop":
        return NoopLLM()
    if backend == "edge_llm":
        return EdgeLLMBackend(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
            retry_on_transient=config.llm_retry_on_transient,
            retry_backoff_s=config.llm_retry_backoff_s,
        )
    if backend in ("openai_compat", "openai"):
        return OpenAICompatBackend(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
            retry_on_transient=config.llm_retry_on_transient,
            retry_backoff_s=config.llm_retry_backoff_s,
        )
    raise ValueError(f"Unknown llm_backend: {config.llm_backend!r}")


def _build_translator(config: Config) -> TranslatorBackend:
    backend = config.translator_backend.lower()
    if backend == "noop":
        return NoopTranslator()
    if backend == "ctranslate2":
        return CTranslate2Translator(
            base_url=config.translator_url,
            timeout=config.translator_timeout_s,
        )
    raise ValueError(f"Unknown translator_backend: {config.translator_backend!r}")


class BaseApp:
    """Subclass and implement `on_user_utterance` to define an App."""

    # AudioIO implementation this app opens the mic/speaker with. Apps that
    # need capture taps (local wake word) or the TTS mic gate override this
    # with ``TappedAudioIO`` instead of re-assigning ``self.audio`` after
    # ``super().__init__()`` — re-assigning would drop the resolved mic
    # channel layout below.
    AUDIO_IO_CLASS = AudioIO

    def _apply_mic_profile(self, setup) -> None:
        """Apply a freshly resolved automatic mic profile.

        AudioIO calls this after a deferred connect or a USB replug. Keep the
        gain override tied to the original config intent: a profile may update
        the default 1.0 value, but an explicit user gain must remain fixed.
        """
        self.mic_setup = setup
        if not getattr(self, "_mic_makeup_gain_auto", True):
            return
        gain = setup.makeup_gain
        if gain is None:
            gain = 1.0
        if abs(float(getattr(self.config, "mic_makeup_gain", 1.0)) - float(gain)) > 1e-9:
            logger.info(
                "mic makeup gain from profile '%s': %.1f",
                setup.profile_name or "native",
                gain,
            )
        self.config.mic_makeup_gain = float(gain)

    def __init__(self, config: Config) -> None:
        self.config = config
        self.events = EventBus()
        slv_config = dict(config.slv_config)
        # Make server-loop an explicit session semantic. This removes the old
        # failure mode where the speech service had its server loop enabled
        # but the device still behaved as a client-loop application.
        slv_config.setdefault("create_response", config.server_loop_enabled())
        slv_config.setdefault("interrupt_response", True)
        self.slv = SLVClient(
            config.slv_url,
            slv_config,
            protocol_version=config.realtime_protocol_version,
        )
        self.slv.on_reply_text = self._on_reply_text_sent
        # Set when a new user turn cut a reply that was still playing; TTS
        # events are that reply's tail until the new reply's text goes out.
        self._awaiting_new_reply = False
        # ``auto`` is deliberately kept as a selector rather than resolved to
        # a PortAudio index once. The same BaseApp is used by multi_mode,
        # translator, caption and robot apps; keeping the selector alive lets
        # AudioIO re-resolve a stable physical name after USB replug/card-index
        # drift. Explicit user overrides remain untouched.
        if _is_auto_audio_selector(config.audio_input_device):
            config.audio_input_device = "auto"
        if _is_auto_audio_selector(config.audio_output_device):
            config.audio_output_device = "auto"

        raw_mic_channels = getattr(config, "mic_channels", "auto")
        raw_mic_channel_select = getattr(config, "mic_channel_select", "auto")

        # Mic channel layout. Resolved here (not per-app) so EVERY app gets
        # reSpeaker firmware auto-detection: the 6ch Flex and 2ch 4-Mic
        # variants need different channel counts, and opening the wrong one
        # crash-loops the mic pump with PaErrorCode -9998. An explicit numeric
        # ``mic_channels`` in the YAML/env still pins the old behaviour.
        self.mic_setup = resolve_mic_setup(
            config.audio_input_device,
            channels_cfg=raw_mic_channels,
            channel_select_cfg=raw_mic_channel_select,
        )
        self._mic_makeup_gain_auto = (
            abs(float(getattr(config, "mic_makeup_gain", 1.0)) - 1.0) < 1e-9
        )
        # Makeup gain follows the firmware profile too — the 6ch variant's ch0
        # is quiet (~12x) while the 2ch variant clips at that gain (~2x). Only
        # fill it in when the config left makeup at the 1.0 no-op default; an
        # explicit value in the YAML still wins.
        self._apply_mic_profile(self.mic_setup)
        self.audio = self.AUDIO_IO_CLASS(
            input_device=config.audio_input_device,
            output_device=config.audio_output_device,
            input_sr=config.audio_input_sample_rate,
            output_sr=config.audio_output_sample_rate,
            mic_channels=self.mic_setup.channels,
            mic_channel_select=self.mic_setup.channel_select,
            mic_channels_cfg=raw_mic_channels,
            mic_channel_select_cfg=raw_mic_channel_select,
            on_input_profile_resolved=self._apply_mic_profile,
        )
        self.llm: LLMBackend = _build_llm(config)
        self.translator: TranslatorBackend = _build_translator(config)
        # Tool registry — single global default. Tests may construct
        # dedicated `ToolRegistry()` instances and inject via mode_ctx.
        self.tool_registry = _default_tool_registry
        self.session = Session(
            locale=str(config.slv_config.get("asr_language", "zh")).lower()[:2],
            max_input_tokens=getattr(config, "session_max_input_tokens", None),
            tokenizer_model=getattr(
                config, "session_tokenizer_model", "Qwen/Qwen3-4B-AWQ"
            ),
            event_bus=self.events,
        )
        self.plugins: list["Plugin"] = []
        # Set by LLMAvailabilityPlugin.start(); read by app_mode to fail-fast
        # when the LLM is DOWN instead of hitting the 15s first-token timeout.
        self.llm_availability = None
        self._shutdown_evt: asyncio.Event | None = None
        self._mic_task: asyncio.Task | None = None
        self._mic_watchdog_task: asyncio.Task | None = None
        self._mic_restart_lock: asyncio.Lock | None = None
        self._last_mic_chunk_ts: float | None = None
        # Wake-loop watchdog: catches a SILENTLY stalled wake source (its
        # listen loop blocked on tap.get() — no exception for the source's own
        # supervisor to see) while the mic capture is still healthy. Separate
        # from the mic watchdog (different subject + recovery). _mic_restart_ts
        # gives the wake loop a grace window to re-acquire after a mic restart.
        self._wake_watchdog_task: asyncio.Task | None = None
        # SLV link watchdog: owns liveness of the conversation WS *and* of the
        # dispatch task. A closed speech-service session is not something an
        # interactive retry budget can resolve, so this one retries forever.
        self._link_watchdog_task: asyncio.Task | None = None
        self._slv_unhealthy_since: float | None = None
        self._wake_restart_lock: asyncio.Lock | None = None
        self._mic_restart_ts: float = 0.0
        self._boot_ts: float = 0.0
        self._dispatch_task: asyncio.Task | None = None
        self._llm_turn_task: asyncio.Task | None = None
        self._first_tts_seen = False
        self._last_tts_started_sentence: str = ""
        self._last_tts_started_ts: float = 0.0
        self._drop_current_tts_sentence: bool = False
        # Client-side VAD state machine. Drives manual asr_eos to SLV when
        # server-side VAD is disabled (slv_config.vad == "none"), so the
        # ASR model gets a chance to accumulate enough audio before being
        # asked to finalize.
        self._client_vad = None
        if getattr(config, "client_vad_backend", "off") != "off":
            try:
                self._client_vad = create_vad(
                    config.client_vad_backend,
                    sample_rate=config.audio_input_sample_rate,
                    threshold=getattr(config, "client_vad_threshold", None),
                )
                logger.info(
                    "client VAD: %s (threshold=%s)",
                    self._client_vad.name,
                    self._client_vad.threshold,
                )
            except Exception as e:
                logger.warning("client VAD init failed (%s); disabled", e)
                self._client_vad = None
        self._vad_state = "idle"  # "idle" | "speech"
        self._vad_speech_ms = 0
        self._vad_silence_ms = 0
        self._vad_eos_sent = False
        # True while the current client-VAD segment began during playback and
        # was rejected as a barge-in by the echo guard. Its end must not close
        # a user turn — see the speech-end branch of _update_vad.
        self._vad_echo_segment = False
        # ── v2: conversation state machine + observability ──
        # Initial state depends on pipeline_mode: always_on boots IDLE
        # (legacy), wake_word / push_to_talk boot SLEEPING.
        if getattr(config, "pipeline_mode", "always_on") == "always_on":
            self._state: ConvState = ConvState.IDLE
        else:
            self._state = ConvState.SLEEPING
        self._slv_reconnect_count: int = 0
        # Monotonic deadline: drop (don't forward) mic audio until this time
        # after a wake-word fires, so the wake-word tail doesn't leak into the
        # command ASR. Set in wake() for audio wake sources; checked in _mic_pump.
        self._wake_mic_skip_until: float = 0.0
        # Monotonic deadline: drop mic audio during short local notification
        # playback, such as the wake tone, so speaker echo does not start an
        # ASR turn while the conversation state is still IDLE.
        self._local_output_mic_suppress_until: float = 0.0
        # #F4: in-flight SERVER_TOOL_CALL handler tasks. Dispatching a remote
        # tool in a background task keeps the dispatch loop draining events
        # (TTS audio / partials / control frames) instead of blocking on the
        # tool's execution. The set holds a strong reference (a bare
        # create_task can be GC'd mid-flight); the done callback prunes it.
        self._pending_tool_tasks: set[asyncio.Task] = set()
        # Auto-sleep timer (only armed when pipeline_mode != always_on).
        self._sleep_task: asyncio.Task | None = None
        # Wake-command timer: when enabled by a robot/control app, a wake word
        # opens one bounded command window and then returns to SLEEPING if no
        # usable ASR final arrives. This is separate from the general
        # auto-sleep timer because it closes the post-wake hot-mic window.
        self._wake_command_timeout_task: asyncio.Task | None = None
        self._wake_command_retry_after_no_final: bool = False
        # Push-to-talk: when True, the next asr_final is the explicit
        # close of a PTT turn — used to short-circuit empty-final guards
        # that would otherwise drop a clipped PTT utterance.
        self._ptt_explicit_eos_pending: bool = False
        self._last_user_utterance_text: str = ""
        # Mid-utterance ASR segment accumulator: with client-owned
        # endpointing (client_vad_drive_eos) the ASR backend may still
        # emit finals on its OWN internal endpoint (RK Qwen3-ASR: 400 ms
        # of silence) BEFORE the agent's asr_eos. Those segments belong
        # to the still-open utterance and are concatenated here; the
        # post-EOS final is the authoritative end and dispatches the
        # joined text to the LLM. Cleared on dispatch, barge-in, sleep,
        # wake-command single-turn completion, and session close.
        self._pending_asr_utterance_text: str = ""
        # Per-turn EOS dedupe: VAD silence and PTT/end can both want to
        # send asr_eos. Send at most one per turn. Cleared on every
        # ASRFinal, on PTT/start (next turn), and on reconnect.
        self._eos_sent_this_turn: bool = False
        # Watchdog: SLV in `always_on` pipeline mode does NOT emit
        # asr_final when ASR yields empty text (it filters server-side
        # to avoid noise turns). Without this watchdog the state machine
        # would stay THINKING forever after the very first VAD trigger
        # on mic noise. Started in send_asr_eos_once, cancelled in the
        # ASRFinal / SLVError / reconnect paths.
        self._asr_watchdog_task: asyncio.Task | None = None
        # Watchdog: SLV in jetson-qwen3asr-matcha-nx (and possibly others)
        # can silently fail to produce TTS after a successful asr_final →
        # tool_call → text-stream → tts_flush sequence. State stays
        # THINKING forever waiting for tts_started that never arrives.
        # Armed when we transition into THINKING; cancelled by the first
        # TTSStarted / TTSDone / ASRFinal (real activity) or SLVError.
        # On fire, force state back to IDLE so the next user turn isn't
        # blocked. Configurable via ``thinking_timeout_s`` (default 20s).
        self._thinking_watchdog_task: asyncio.Task | None = None
        self._playback_drain_task: asyncio.Task | None = None
        # silero-primary stall fallback (config.vad_stall_eos_ms): reset on
        # every real asr_partial; forces a single asr_eos if silero goes quiet
        # without finalizing, so the turn can't hang. See _stall_eos_watchdog.
        self._stall_watchdog_task: asyncio.Task | None = None
        # Dashboard mic RMS is a best-effort visualization signal. Never let
        # a slow browser/plugin backpressure the hot mic pump; at most one
        # RMS hook fanout may be in flight and newer samples are dropped.
        self._mic_rms_broadcast_task: asyncio.Task | None = None
        # Rate-limited stop-word matcher cache (compiled per Config update).
        self._stop_words_cache: tuple[list[str], list[str]] | None = None

    # ── startup budget validation ───────────────────────────────────

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """Cheap upper-bound token estimate without loading a tokenizer.

        ``len(text) // 3`` is a conservative bound: Chinese chars ≈ 0.4
        token, English chars ≈ 0.25 token; 1/3 sits above both. Used
        only for startup sanity-check logging, not for trim accounting.
        """
        if not text:
            return 0
        return max(1, len(text) // 3)

    @classmethod
    def _approx_tokens_for_tools(cls, tools: list[dict] | None) -> int:
        if not tools:
            return 0
        try:
            import json as _json
            return cls._approx_tokens(_json.dumps(tools, ensure_ascii=False))
        except Exception:
            return 0

    # Minimum history headroom we want to leave after the fixed prefix
    # before we ERROR. Anything below this means even a single user turn
    # may overflow ``session_max_input_tokens``.
    _MIN_HISTORY_HEADROOM = 1000

    def _validate_session_budget(
        self, system_prompt: str, tools: list[dict] | None
    ) -> None:
        """Sanity-check session_max_input_tokens vs the fixed prefix.

        A1-step2 semantics: ``Session._trim_to_budget`` only counts
        dynamic turns (user/assistant/tool) against
        ``session_max_input_tokens * 0.75``. The fixed prefix
        (system_prompt + tools schema) is NOT in the trim budget but
        IS still charged against the engine ``max_seq_len`` on every
        request. This validator therefore checks whether the fixed
        prefix leaves enough room inside ``session_max_input_tokens``
        for a reasonable history window.
        """
        max_input = getattr(self.config, "session_max_input_tokens", None)
        if not max_input:
            logger.info(
                "Session trim disabled (session_max_input_tokens=None); "
                "skipping budget validation"
            )
            return
        sys_tokens = self._approx_tokens(system_prompt or "")
        tools_tokens = self._approx_tokens_for_tools(tools)
        fixed_tokens = sys_tokens + tools_tokens
        history_headroom = max_input - fixed_tokens
        pct = (fixed_tokens / max_input * 100) if max_input else 0.0
        if history_headroom < self._MIN_HISTORY_HEADROOM:
            recommended = fixed_tokens + 2 * self._MIN_HISTORY_HEADROOM
            logger.error(
                "FIXED PREFIX (system_prompt %d + tools %d = %d tokens) leaves "
                "only %d tokens of history headroom inside "
                "session_max_input_tokens=%d (need >=%d). Even a single user "
                "turn may overflow. Raise session_max_input_tokens to at "
                "least %d.",
                sys_tokens, tools_tokens, fixed_tokens, history_headroom,
                max_input, self._MIN_HISTORY_HEADROOM, recommended,
            )
        elif fixed_tokens > max_input * 0.6:
            logger.warning(
                "Fixed prefix uses %.0f%% of session_max_input_tokens "
                "(%d / %d tokens, system=%d tools=%d). Trim only counts "
                "dynamic turns so behaviour is OK, but history headroom "
                "(%d tokens) is tight. Consider raising "
                "session_max_input_tokens.",
                pct, fixed_tokens, max_input, sys_tokens, tools_tokens,
                history_headroom,
            )
        else:
            logger.info(
                "Session budget OK: fixed prefix %d tokens (system=%d "
                "tools=%d), history headroom %d, max_input=%d. Trim budget "
                "(history only) = %d.",
                fixed_tokens, sys_tokens, tools_tokens, history_headroom,
                max_input, int(max_input * 0.75),
            )

    # ── v2: state machine + stop intent ─────────────────────────────

    def _set_state(self, new: ConvState) -> None:
        """Transition the conversation state. Logs + emits hook/event on change.

        Safe to call from any coroutine in the same event loop. Tests build
        BaseApp via __new__ without invoking __init__, so default missing
        attributes to IDLE rather than crashing.
        """
        old = getattr(self, "_state", ConvState.IDLE)
        if new == old:
            return
        self._state = new
        logger.info("ConvState: %s → %s", old.value, new.value)
        bus = getattr(self, "events", None)
        if bus is not None:
            try:
                bus.emit("state_change", {"state": new.value, "prev": old.value})
            except Exception:  # pragma: no cover - defensive
                logger.debug("EventBus state_change emit failed", exc_info=True)
        try:
            asyncio.get_running_loop().create_task(
                self._broadcast(
                    "on_state_change", {"state": new.value, "prev": old.value}
                )
            )
        except RuntimeError:
            pass

    def _normalise_for_stop(self, text: str) -> str:
        """Lowercase + strip whitespace and trailing punctuation."""
        if not text:
            return ""
        s = text.strip()
        # Strip trailing CJK + ASCII sentence punctuation.
        while s and s[-1] in "。，！？.!?,;:":
            s = s[:-1]
        return s.strip().lower()

    # Separators that can sit between a leaked wake phrase and the command.
    _WAKE_SEPARATORS = " ,，.。!！?？:：;、"

    def _strip_wake_prefix(self, text: str) -> str | None:
        """Suppress / strip a leaked wake phrase from an ASR final.

        The local wake-word detector fires only after hearing the full
        phrase, so that audio is already in the server ASR stream — the
        wake word comes back transcribed as (or prefixing) the utterance.

        Returns:
          * ``None``  — the whole utterance IS the wake phrase → drop it
            (no LLM turn; saying the wake word while awake isn't a command).
          * stripped remainder — the utterance STARTS with the wake phrase
            (e.g. "Hey Jarvis 挥手") → dispatch only the command part.
          * ``text`` unchanged — no wake phrase at the front.

        Only matches CLEAN transcriptions (config ``wake_phrases``); a
        mis-heard wake word ("só", "乔治") is an acoustic issue, not caught here.
        """
        phrases = getattr(self.config, "wake_phrases", None) or []
        if not text or not phrases:
            return text
        stripped = text.strip()
        low = stripped.lower()
        norm = self._normalise_for_stop(stripped)  # trailing punct removed
        for raw in phrases:
            p = (raw or "").strip().lower()
            if not p:
                continue
            if norm == p:
                return None  # bare wake word
            if low.startswith(p):
                rest = stripped[len(p):]  # lower() preserves length
                if rest[:1] in self._WAKE_SEPARATORS:
                    remainder = rest.lstrip(self._WAKE_SEPARATORS).strip()
                    return remainder or None
        return stripped

    # A prefix match ("stop ...") counts as a stop only while what follows is
    # a short tail like "please" / "it now" / "talking". A longer tail is a new
    # instruction that happens to open with "stop": "Stop, please answer in one
    # sentence." must reach the LLM, not end the turn in silence. Chinese stop
    # words already match the whole utterance only, so this brings English in
    # line. Measured on rk3588 (2026-09-21) once barge-in stopped dropping the
    # head of the utterance: that exact sentence went thinking → idle.
    _STOP_PREFIX_MAX_TAIL_WORDS = 2

    def _is_stop_intent(self, text: str) -> bool:
        """Match per spec: Chinese -> exact full-string; English -> case-
        insensitive whole-utterance OR word-boundary prefix (>= 2 chars)
        followed by at most ``_STOP_PREFIX_MAX_TAIL_WORDS`` words.
        """
        norm = self._normalise_for_stop(text)
        if not norm:
            return False
        # Partition stop_words by ASCII-ness.
        cfg = getattr(self, "config", None)
        words = (getattr(cfg, "stop_words", []) if cfg is not None else []) or []
        for w in words:
            if not w:
                continue
            wn = w.strip().lower()
            if not wn:
                continue
            is_ascii = wn.isascii()
            if not is_ascii:
                # CJK / unicode: full-string equality only.
                if norm == wn:
                    return True
            else:
                # English: whole-utterance equality OR
                # word-boundary prefix (matched word is at least 2 chars).
                if norm == wn:
                    return True
                if len(wn) >= 2 and (
                    norm.startswith(wn + " ")
                    or norm.startswith(wn + ",")
                    or norm.startswith(wn + "!")
                    or norm.startswith(wn + "?")
                    or norm.startswith(wn + ".")
                ):
                    tail = norm[len(wn):].replace(",", " ").replace(".", " ")
                    tail = tail.replace("!", " ").replace("?", " ").split()
                    if len(tail) <= self._STOP_PREFIX_MAX_TAIL_WORDS:
                        return True
        return False

    # ── pipeline_mode: wake / sleep / sleep-timer ──────────────────

    async def wake(self, source: str = "external") -> None:
        """Transition SLEEPING → IDLE and (re-)arm the sleep timer.

        No-op if not currently SLEEPING. Broadcasts on_wake with source.

        SLV stream health is verified before the state transition. The
        previous turn's proactive ``slv.reconnect()`` (on tts_done) can
        fail-silently: ``_open_with_retry`` may exhaust its backoff
        budget while ``send_audio`` / dispatch loop never observe the
        failure because the timeout is shorter than the retry window
        (``send_audio`` waits 0.5s, retry budget is up to ~1.75s + 3
        round-trips). Result was a dead stream with ``_ws=None`` that
        silently swallowed every post-wake mic chunk — the "mute bug".

        We now health-check on wake, attempt a single reconnect if dead,
        and *refuse the wake* (stay SLEEPING) if even that fails. Better
        to make the silence visible (log + on_wake_failed broadcast)
        than to pretend everything is fine and have the user repeat
        themselves into the void.
        """
        # A local wake-word just fired — arm the mic-skip so the wake-word tail
        # (and its reverb) isn't forwarded into the command ASR. Set BEFORE the
        # SLEEPING gate so it also applies to a re-wake spoken mid-conversation
        # (state already IDLE): otherwise that "Hey Jarvis" leaks into the next
        # command and the server decodes them as one garbled segment. External
        # (non-audio) wake sources have no wake word to skip.
        if any(
            getattr(plugin, "name", None) == source
            and bool(getattr(plugin, "local_audio", False))
            for plugin in getattr(self, "plugins", ())
        ):
            skip_ms = float(getattr(self.config, "wake_mic_skip_ms", 0.0) or 0.0)
            if skip_ms > 0:
                self._wake_mic_skip_until = time.monotonic() + skip_ms / 1000.0
        if getattr(self, "_state", ConvState.IDLE) != ConvState.SLEEPING:
            return
        logger.info("wake from %s", source)
        # Health-gated reconnect on wake. SLV server v1.15+ added an ASR
        # turn wall-clock timeout (SessionLimiter slot is force-released
        # even when the Qwen3-ASR worker wedges), so the previous
        # "always reconnect on wake" workaround is no longer needed —
        # and was actively causing 4429 too_many_sessions because the
        # new WS races the just-released slot. Only reconnect when
        # is_healthy() reports the current WS is actually dead
        # (ws missing or reader task exited). Healthy WS persists
        # across multiple turns (correct per SLV multi_utterance
        # protocol where session_complete=False).
        # Idle-based reconnect: TCP-alive WS doesn't guarantee SLV's ASR
        # session is still alive — after long idle (>30s) the server may
        # have internally recycled the session, in which case is_healthy()
        # returns True but mic data flows into a dead ASR worker (silent
        # mute bug, observed 2026-05-26 after tts_done→wake at +4min).
        # Force a reconnect on long idle to refresh the ASR session.
        # Hot-turn case (continuous dialogue) stays cheap: activity is
        # touched on every WS recv/send so seconds_since_activity stays
        # small and no reconnect fires — avoiding the 4429 limiter race.
        _idle_s = self.slv.seconds_since_activity()
        _healthy = self.slv.is_healthy()
        # reconnect_on_wake: a single streaming-ASR worker can degrade after
        # several utterances on one persistent session (empty finals); force
        # a fresh worker on every wake so re-saying the wake word recovers.
        _force_wake_reconnect = bool(getattr(self.config, "reconnect_on_wake", False))
        should_reconnect = _force_wake_reconnect or (not _healthy) or _idle_s > 30.0
        if should_reconnect:
            try:
                await asyncio.wait_for(self.slv.reconnect(), timeout=6.0)
                self._slv_reconnect_count = getattr(self, "_slv_reconnect_count", 0) + 1
                logger.info(
                    "wake: SLV reconnect (healthy=%s idle=%.1fs count=%d)",
                    _healthy, _idle_s, self._slv_reconnect_count,
                )
                try:
                    await self._broadcast(
                        "on_slv_reconnect",
                        {"count": self._slv_reconnect_count},
                    )
                except Exception:
                    logger.exception("on_slv_reconnect broadcast failed (wake)")
                # (#38) Re-advertise tools after a successful reconnect so
                # server-loop parity survives WS churn. Idempotent upsert on
                # SLV; no-op in client-loop mode.
                await self._readvertise_after_reconnect()
            except (SLVReconnectError, asyncio.TimeoutError, ConnectionError) as e:
                logger.error(
                    "wake: SLV reconnect failed (%s); staying SLEEPING to avoid silent mute",
                    e,
                )
                try:
                    await self._broadcast(
                        "on_wake_failed",
                        {"source": source, "reason": "slv_unhealthy"},
                    )
                except Exception:
                    logger.exception("on_wake_failed broadcast failed")
                # Do NOT transition to IDLE — caller (wake_source / dashboard)
                # observes that we are still SLEEPING and can surface the
                # failure to the user.
                return
            except Exception:
                logger.exception(
                    "wake: SLV reconnect raised unexpected error; staying SLEEPING"
                )
                return
        try:
            await self._broadcast("on_wake", {"source": source})
        except Exception:
            logger.exception("on_wake broadcast failed")
        # Clear the playback discard latch sleep() armed — otherwise the
        # first post-wake TTS (especially typed-text path with no ASRFinal)
        # would be silently dropped.
        try:
            arm = getattr(self.audio, "arm_for_next_turn", None)
            if callable(arm):
                arm()
        except Exception:  # pragma: no cover - defensive
            pass
        self._set_state(ConvState.IDLE)
        self._reset_sleep_timer()
        self._arm_wake_command_timeout()
        self._play_wake_tone()

    def _play_wake_tone(self) -> None:
        """Play a short notification tone so the user knows the wake word landed."""
        meta = getattr(self.config, "metadata", {}) or {}
        tone_cfg = meta.get("wake_tone", {}) or {}
        hz = float(tone_cfg.get("hz", 0))
        ms = float(tone_cfg.get("ms", 0))
        suppress_tail_ms = float(tone_cfg.get("mic_suppress_tail_ms", 600))
        if hz <= 0 or ms <= 0:
            return
        import math
        sr = self.audio.output_sr or 16000
        n = int(sr * ms / 1000)
        amp = 20000
        fade = min(n // 4, int(sr * 0.01))
        pcm = bytearray(n * 2)
        import struct
        for i in range(n):
            s = amp * math.sin(2 * math.pi * hz * i / sr)
            if i < fade:
                s *= i / fade
            elif i >= n - fade:
                s *= (n - 1 - i) / fade
            struct.pack_into("<h", pcm, i * 2, max(-32768, min(32767, int(s))))
        notify = getattr(self.audio, "play_notification", None)
        if callable(notify):
            notify(bytes(pcm))
            self._local_output_mic_suppress_until = max(
                getattr(self, "_local_output_mic_suppress_until", 0.0),
                time.monotonic() + (ms + suppress_tail_ms) / 1000.0,
            )
            logger.info("wake tone: %dHz %dms", int(hz), int(ms))

    def _play_sleep_tone(self) -> None:
        """Play a descending tone so the user knows the listening window closed."""
        meta = getattr(self.config, "metadata", {}) or {}
        tone_cfg = meta.get("sleep_tone", {}) or {}
        hz_start = float(tone_cfg.get("hz_start", 0))
        hz_end = float(tone_cfg.get("hz_end", 0))
        ms = float(tone_cfg.get("ms", 0))
        suppress_tail_ms = float(tone_cfg.get("mic_suppress_tail_ms", 400))
        if hz_start <= 0 or hz_end <= 0 or ms <= 0:
            return
        import math, struct
        sr = self.audio.output_sr or 16000
        n = int(sr * ms / 1000)
        amp = 16000
        fade = min(n // 4, int(sr * 0.01))
        pcm = bytearray(n * 2)
        for i in range(n):
            t = i / max(n - 1, 1)
            hz = hz_start + (hz_end - hz_start) * t
            s = amp * math.sin(2 * math.pi * hz * i / sr)
            if i < fade:
                s *= i / fade
            elif i >= n - fade:
                s *= (n - 1 - i) / fade
            struct.pack_into("<h", pcm, i * 2, max(-32768, min(32767, int(s))))
        notify = getattr(self.audio, "play_notification", None)
        if callable(notify):
            notify(bytes(pcm))
            self._local_output_mic_suppress_until = max(
                getattr(self, "_local_output_mic_suppress_until", 0.0),
                time.monotonic() + (ms + suppress_tail_ms) / 1000.0,
            )
            logger.info("sleep tone: %d→%dHz %dms", int(hz_start), int(hz_end), int(ms))

    async def sleep(self) -> None:
        """Forcibly transition to SLEEPING — cancel LLM turn, abort SLV,
        drop playback. Idempotent if already SLEEPING."""
        if getattr(self, "_state", ConvState.IDLE) == ConvState.SLEEPING:
            return
        logger.info("sleep")
        try:
            await self._broadcast("on_sleep", None)
        except Exception:
            logger.exception("on_sleep broadcast failed")
        if self._llm_turn_task is not None and not self._llm_turn_task.done():
            self._llm_turn_task.cancel()
            try:
                await self._llm_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await self.slv.abort()
        except Exception:
            pass
        try:
            await self.audio.stop_playback()
        except Exception:
            pass
        self._cancel_wake_command_timeout()
        self._clear_pending_asr_utterance()
        drain_task = getattr(self, "_playback_drain_task", None)
        if drain_task is not None and not drain_task.done():
            drain_task.cancel()
        self._first_tts_seen = False
        self._set_state(ConvState.SLEEPING)
        if self._sleep_task is not None and not self._sleep_task.done():
            self._sleep_task.cancel()
        self._sleep_task = None

    def _reset_sleep_timer(self) -> None:
        """(Re-)start the auto-sleep countdown. No-op for always_on."""
        if getattr(self.config, "pipeline_mode", "always_on") == "always_on":
            return
        if self._sleep_task is not None and not self._sleep_task.done():
            self._sleep_task.cancel()
        timeout = float(getattr(self.config, "sleep_timeout_s", 30.0))
        try:
            self._sleep_task = asyncio.create_task(
                self._sleep_after(timeout), name="sleep-timer"
            )
        except RuntimeError:
            # No running loop (called from sync context like tests).
            self._sleep_task = None

    def _ensure_sleep_timer(self) -> None:
        """Arm the auto-sleep countdown only if one is NOT already running —
        does NOT restart a running one.

        Used after a NOISE / empty asr_final recovers the FSM to IDLE: ambient
        room speech produces a stream of empty finals, and restarting the
        countdown on each one keeps the device hot-mic FOREVER (operator
        2026-06-18: "触发太灵敏，设备一直不会到待命状态"). A noise final is not user
        activity, so let the countdown started at the last REAL activity (wake /
        a command with content) keep running — the device returns to standby
        ``sleep_timeout_s`` after the last real command regardless of background
        noise. If no timer is running (already slept / never armed), arm one."""
        if getattr(self.config, "pipeline_mode", "always_on") == "always_on":
            return
        if self._sleep_task is not None and not self._sleep_task.done():
            return  # already counting down — a noise final must not extend it
        self._reset_sleep_timer()

    def _wake_command_single_turn_enabled(self) -> bool:
        return (
            getattr(self.config, "pipeline_mode", "always_on") == "wake_word"
            and bool(getattr(self.config, "wake_command_single_turn", False))
        )

    def _accumulate_asr_segment(self, text: str) -> None:
        """Append one backend-endpointed ASR segment to the pending
        utterance buffer (empty-after-strip segments are ignored)."""
        seg = (text or "").strip()
        if not seg:
            return
        self._pending_asr_utterance_text = _join_asr_segment_text(
            getattr(self, "_pending_asr_utterance_text", ""), seg
        )

    def _take_pending_asr_utterance(self) -> str:
        """Return and clear the accumulated mid-utterance segments."""
        pending = getattr(self, "_pending_asr_utterance_text", "")
        self._pending_asr_utterance_text = ""
        return pending

    def _clear_pending_asr_utterance(self) -> None:
        """Drop the pending buffer without dispatching (barge-in, sleep,
        session close): a half-accumulated utterance must never glue onto
        the NEXT turn's text."""
        self._pending_asr_utterance_text = ""

    def _cancel_wake_command_timeout(self) -> None:
        task = getattr(self, "_wake_command_timeout_task", None)
        if task is not None and not task.done():
            task.cancel()
        self._wake_command_timeout_task = None

    def _arm_wake_command_timeout(self) -> None:
        self._cancel_wake_command_timeout()
        if not self._wake_command_single_turn_enabled():
            return
        timeout = float(getattr(self.config, "wake_command_timeout_s", 0.0) or 0.0)
        if timeout <= 0:
            return
        try:
            self._wake_command_timeout_task = asyncio.create_task(
                self._wake_command_timeout(timeout),
                name="wake-command-timeout",
            )
        except RuntimeError:
            self._wake_command_timeout_task = None

    async def _wake_command_timeout(self, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        if not self._wake_command_single_turn_enabled():
            return
        if getattr(self, "_state", ConvState.IDLE) in (
            ConvState.IDLE,
            ConvState.LISTENING,
        ):
            logger.info(
                "wake command timeout %.1fs with no valid final; forcing asr_eos",
                timeout,
            )
            self._set_state(ConvState.THINKING)
            try:
                await self.send_asr_eos_once()
            except Exception:
                logger.exception("wake command timeout asr_eos failed")
                await self._return_to_sleep_after_command_turn()

    async def _return_to_sleep_after_command_turn(self) -> None:
        """Close a wake-command listening window without cancelling tools.

        This is intentionally NOT ``sleep()``. User/admin sleep is a cancel
        action and broadcasts ``on_sleep``; GraspPlugin listens to that hook to
        abort an in-flight grasp. Command-mode post-turn sleep only stops the
        mic from accepting another command until the next wake word.
        """
        self._cancel_wake_command_timeout()
        if getattr(self, "_state", ConvState.IDLE) == ConvState.SLEEPING:
            return
        if self._sleep_task is not None and not self._sleep_task.done():
            self._sleep_task.cancel()
        self._sleep_task = None
        self._first_tts_seen = False
        self._eos_sent_this_turn = False
        self._wake_command_retry_after_no_final = False
        self._clear_pending_asr_utterance()
        self._play_sleep_tone()
        self._set_state(ConvState.SLEEPING)

    async def _open_wake_command_retry_window(self) -> None:
        """Keep the post-wake command window open after a no-final prompt."""
        self._cancel_wake_command_timeout()
        if self._sleep_task is not None and not self._sleep_task.done():
            self._sleep_task.cancel()
        self._sleep_task = None
        self._first_tts_seen = False
        self._eos_sent_this_turn = False
        self._wake_command_retry_after_no_final = False
        if getattr(self, "_state", ConvState.IDLE) != ConvState.BARGED_IN:
            self._set_state(ConvState.IDLE)
            self._arm_wake_command_timeout()

    async def _complete_assistant_turn(self) -> None:
        if (
            self._state != ConvState.BARGED_IN
            and self._wake_command_single_turn_enabled()
            and getattr(self, "_wake_command_retry_after_no_final", False)
        ):
            await self._open_wake_command_retry_window()
            await self._broadcast("on_assistant_done")
            return
        if (
            self._state != ConvState.BARGED_IN
            and self._wake_command_single_turn_enabled()
        ):
            await self._return_to_sleep_after_command_turn()
        elif self._state != ConvState.BARGED_IN:
            self._set_state(ConvState.IDLE)
            self._reset_sleep_timer()
        await self._broadcast("on_assistant_done")

    async def _finish_assistant_turn_after_playback(self) -> None:
        """Complete a TTS turn after locally buffered PCM has drained."""
        timeout_s = float(getattr(self.config, "playback_drain_timeout_s", 10.0))
        deadline = time.monotonic() + max(0.1, timeout_s)
        try:
            while (
                self._state != ConvState.BARGED_IN
                and getattr(self.audio, "is_playing", False)
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            await self._complete_assistant_turn()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("playback drain completion failed")

    async def _speak_wake_command_no_final_feedback(self) -> bool:
        """Tell the user a wake-command turn heard no usable ASR final."""
        text = str(
            getattr(
                self.config,
                "wake_command_no_final_text",
                "没听清，请再说一遍。",
            )
            or ""
        ).strip()
        if not text:
            return False
        try:
            self._wake_command_retry_after_no_final = True
            logger.info("wake command no-final feedback: %s", text)
            await self._broadcast("on_assistant_sentence_start", text)
            await self.slv.send_text(text)
            await self.slv.flush_tts()
            return True
        except Exception:
            self._wake_command_retry_after_no_final = False
            logger.exception("wake command no-final feedback failed")
            return False

    async def _sleep_after(self, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        # Only sleep if still IDLE — an in-flight turn delays.
        if getattr(self, "_state", ConvState.IDLE) == ConvState.IDLE:
            await self.sleep()

    # ── public API ──────────────────────────────────────────────────

    async def speak(self, text: str, *, conversation: str = "none") -> None:
        """Speak exact text through the provider-neutral direct-speech API."""
        await self.slv.speak(text, conversation=conversation)

    async def reset_conversation(self) -> None:
        """Clear local and remote conversational context when available."""
        reset_local = getattr(self.session, "reset", None)
        if callable(reset_local):
            reset_local()
        reset = getattr(self.slv, "reset_conversation", None)
        if callable(reset):
            await reset()

    def register(self, plugin: "Plugin") -> bool:
        if not plugin.setup():
            logger.info("plugin %s setup() returned False -- skipped", plugin.name)
            return False
        self.plugins.append(plugin)
        return True

    async def on_user_utterance(
        self, text: str, detected_language: str | None = None
    ) -> None:
        """Subclasses MUST override. Default raises.

        ``detected_language`` is the ASR-reported language for this turn
        (e.g. ``"Chinese"``) or ``None`` if the backend doesn't do LID.
        Passed per-call so mode lifecycle hooks (enter/exit) never see a
        stale value.
        """
        raise NotImplementedError("Subclass BaseApp and implement on_user_utterance")

    async def on_user_partial(
        self, text: str, detected_language: str | None = None
    ) -> None:
        """ASR partial transcript update. Default: no-op.

        Apps that consume the streaming partial flow (live-caption,
        simultaneous-interpret) override this to drive a
        ``SegmentCommitter``. Plain apps ignore it. Called from the
        ASRPartial dispatch alongside the ``on_user_partial`` plugin
        broadcast (see ``_dispatch_one``).
        """
        return None

    def _active_mode_barge_in_override(self) -> bool | None:
        """Per-mode barge-in override, or ``None`` if not mode-driven.

        BaseApp has no mode concept; returns ``None`` so resolution falls
        back to config. ``MultiModeApp`` may override to surface the active
        mode's ``barge_in_enabled`` (mode override → mode class default).
        """
        return None

    def _barge_in_enabled(self) -> bool:
        """Resolve whether barge-in may interrupt the assistant.

        Order: active-mode override → ``config.barge_in_enabled`` →
        default ``True`` (unconfigured keeps the legacy always-on
        behaviour, so existing apps are unaffected).
        """
        override = self._active_mode_barge_in_override()
        if override is not None:
            return override
        cfg = getattr(self.config, "barge_in_enabled", None)
        return True if cfg is None else bool(cfg)

    def _barge_in_delay_elapsed(self) -> tuple[bool, float, int]:
        """Apply the same post-playback echo guard to every barge-in signal."""
        minimum_ms = int(
            getattr(self.config, "barge_in_min_speaking_ms", 500) or 0
        )
        speaking_since = float(getattr(self, "_speaking_since_ts", 0.0) or 0.0)
        elapsed_ms = (
            (time.monotonic() - speaking_since) * 1000
            if speaking_since
            else 99999.0
        )
        return elapsed_ms >= minimum_ms, elapsed_ms, minimum_ms

    # ── boot-time connect retry budget (#38) ─────────────────────────
    # The very first connect() races the SLV session-limiter (limit=1):
    # after an abrupt kill the previous slot can take ~60s to release.
    # The runtime ``reconnect()`` budget (``_RECONNECT_BACKOFFS`` ≈ 1.75s)
    # is deliberately short so it never wedges the mic/send loops — but
    # that short budget aborts boot before tools are ever advertised,
    # making server-loop parity unreachable. Boot uses a SEPARATE, much
    # longer budget: capped exponential backoff over ≥75s wall-clock.
    # This budget is ONLY used at boot; runtime reconnect is untouched.
    #
    # 75s was sized against the session-limiter release window (~60s). It is
    # NOT the binding constraint on a cold boot: when agent and speech service
    # start together, the agent is waiting on SLV model load, not on a slot.
    # Measured on rk3576 (2026-08-25): speech service took 102s from container
    # start to "Speech service ready", and the agent connected with 16s of
    # budget left — it survived only because its own init consumed part of the
    # wall-clock before the deadline started running. A slower cold cache would
    # abort boot. 180s covers model load with margin; the only cost of a larger
    # budget is a longer wait in the case where boot was going to fail anyway.
    # Override with OVS_AGENT_BOOT_CONNECT_DEADLINE_S.
    _BOOT_CONNECT_DEADLINE_S: float = 180.0
    _BOOT_CONNECT_BACKOFFS = (0.5, 1.0, 2.0, 5.0)  # then 5.0 until deadline

    def _boot_connect_deadline_s(self) -> float:
        raw = os.getenv("OVS_AGENT_BOOT_CONNECT_DEADLINE_S")
        if raw is None or not raw.strip():
            return self._BOOT_CONNECT_DEADLINE_S
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "OVS_AGENT_BOOT_CONNECT_DEADLINE_S=%r is not a number; "
                "using %.0fs", raw, self._BOOT_CONNECT_DEADLINE_S,
            )
            return self._BOOT_CONNECT_DEADLINE_S
        # float() accepts "nan"/"inf", and `value <= 0` is False for both.
        # nan would make the `remaining <= 0` deadline check never fire —
        # unbounded retries at zero backoff; inf retries forever. Reject
        # before the positivity check so neither reaches the retry loop.
        if not math.isfinite(value):
            logger.warning(
                "OVS_AGENT_BOOT_CONNECT_DEADLINE_S=%r is not finite; "
                "using %.0fs", raw, self._BOOT_CONNECT_DEADLINE_S,
            )
            return self._BOOT_CONNECT_DEADLINE_S
        if value <= 0:
            logger.warning(
                "OVS_AGENT_BOOT_CONNECT_DEADLINE_S=%.1f is not positive; "
                "using %.0fs", value, self._BOOT_CONNECT_DEADLINE_S,
            )
            return self._BOOT_CONNECT_DEADLINE_S
        return value

    async def _connect_with_boot_retry(self) -> None:
        """Open the first SLV WS, retrying past the session-limiter window.

        Unlike runtime ``reconnect()`` (short budget so it can't block the
        mic/send loops), boot can afford to wait for the prior session's
        slot to release (~60s after an abrupt kill). Retries with capped
        exponential backoff until ``_BOOT_CONNECT_DEADLINE_S`` of
        wall-clock has elapsed; only then does the final failure escape.
        """
        budget_s = self._boot_connect_deadline_s()
        deadline = time.monotonic() + budget_s
        attempt = 0
        while True:
            try:
                await self.slv.connect()
                if attempt:
                    logger.info(
                        "boot: SLV connect succeeded after %d retr%s",
                        attempt, "y" if attempt == 1 else "ies",
                    )
                return
            except (SLVReconnectError, ConnectionError, OSError) as e:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error(
                        "boot: SLV connect failed after %.0fs budget (%s); giving up",
                        budget_s, e,
                    )
                    raise
                idx = min(attempt, len(self._BOOT_CONNECT_BACKOFFS) - 1)
                backoff = min(self._BOOT_CONNECT_BACKOFFS[idx], max(0.0, remaining))
                logger.warning(
                    "boot: SLV connect attempt %d failed (%s); retrying in %.2fs "
                    "(%.0fs budget left)",
                    attempt + 1, e, backoff, remaining,
                )
                attempt += 1
                await asyncio.sleep(backoff)

    async def run(self) -> None:
        self._shutdown_evt = asyncio.Event()
        # Gate (#38 race fix, option (a)): the mic pump must not forward
        # audio until boot-time tool advertise has completed, otherwise the
        # first server-loop ASRFinal could reach SLV before it knows our
        # tool schemas. Cleared until advertise finishes (or is a no-op in
        # client-loop mode, in which case it is set immediately).
        self._advertise_ready = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown_evt.set)
            except (NotImplementedError, RuntimeError):
                # Windows / non-main thread -- caller is responsible.
                pass

        self._log_boot_diagnostic()

        await self._connect_with_boot_retry()
        self._mic_restart_lock = asyncio.Lock()
        self._wake_restart_lock = asyncio.Lock()
        self._boot_ts = time.monotonic()
        self._mic_task = asyncio.create_task(self._mic_pump(), name="mic-pump")
        self._mic_watchdog_task = asyncio.create_task(
            self._mic_watchdog(), name="mic-watchdog"
        )
        self._wake_watchdog_task = asyncio.create_task(
            self._wake_watchdog(), name="wake-watchdog"
        )
        self._dispatch_task = asyncio.create_task(self._slv_dispatch(), name="slv-dispatch")
        # SLV link supervisor: the dispatch loop's own retry is bounded in
        # practice (measured 2026-09-13: a speech-service restart left the
        # assistant mute until an operator hit reconnect), so a watchdog owns
        # liveness of BOTH the WS and the dispatch task.
        self._slv_unhealthy_since: float | None = None
        self._link_watchdog_task = asyncio.create_task(
            self._slv_link_watchdog(), name="slv-link-watchdog"
        )

        # LLM backend warmup — runs after all plugins have registered
        # (so tool_registry is fully populated) and BEFORE any plugin
        # start() so the very first user turn never pays cold-start cost.
        await self._maybe_run_llm_warmup()

        for p in self.plugins:
            try:
                await p.start()
            except Exception:
                logger.exception("plugin %s start() failed", p.name)

        # Server-loop mode (#37 Phase 2-product): advertise local tool
        # schemas to SLV AFTER plugin start() so plugin-registered tools
        # (e.g. ArmPlugin's arm actions) are included. No-op when the flag
        # is off — zero behaviour change for the legacy client-loop path.
        # (#38) Release the mic-forward gate only AFTER advertise completes
        # so the first server-loop ASRFinal never beats the tool schemas to
        # the server.
        try:
            await self._advertise_tools_if_server_loop()
        finally:
            self._advertise_ready.set()

        try:
            await self._shutdown_evt.wait()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        # 0. cancel any in-flight LLM turn
        if self._llm_turn_task is not None and not self._llm_turn_task.done():
            self._llm_turn_task.cancel()
            try:
                await self._llm_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        # 0a'. cancel any in-flight server tool-call tasks (#F4). The arm motion
        # itself runs on its own detached worker (parallel mode), so cancelling
        # the handler only stops a pending CLIENT_TOOL_RESULT send — fine during
        # shutdown (the server is going away too).
        for _t in list(getattr(self, "_pending_tool_tasks", ()) or ()):
            if not _t.done():
                _t.cancel()
                try:
                    await _t
                except (asyncio.CancelledError, Exception):
                    pass
        if getattr(self, "_pending_tool_tasks", None) is not None:
            self._pending_tool_tasks.clear()
        # 0a. cancel auto-sleep timer too — otherwise a pending
        # _sleep_after coroutine can fire mid-shutdown, racing with
        # the rest of the cleanup (and emitting on_sleep after plugins
        # have already stopped).
        if self._sleep_task is not None and not self._sleep_task.done():
            self._sleep_task.cancel()
            try:
                await self._sleep_task
            except (asyncio.CancelledError, Exception):
                pass
        self._sleep_task = None
        self._cancel_wake_command_timeout()
        mic_watchdog_task = getattr(self, "_mic_watchdog_task", None)
        if mic_watchdog_task is not None and not mic_watchdog_task.done():
            mic_watchdog_task.cancel()
            try:
                await mic_watchdog_task
            except (asyncio.CancelledError, Exception):
                pass
        self._mic_watchdog_task = None
        wake_watchdog_task = getattr(self, "_wake_watchdog_task", None)
        if wake_watchdog_task is not None and not wake_watchdog_task.done():
            wake_watchdog_task.cancel()
            try:
                await wake_watchdog_task
            except (asyncio.CancelledError, Exception):
                pass
        self._wake_watchdog_task = None
        link_watchdog_task = getattr(self, "_link_watchdog_task", None)
        if link_watchdog_task is not None and not link_watchdog_task.done():
            link_watchdog_task.cancel()
            try:
                await link_watchdog_task
            except (asyncio.CancelledError, Exception):
                pass
        self._link_watchdog_task = None
        # 1. stop mic capture
        if self._mic_task is not None:
            self._mic_task.cancel()
        # 2. cancel TTS if any
        if self.audio.is_playing:
            try:
                await self.slv.abort()
            except Exception:  # pragma: no cover
                pass
        # 3. stop plugins in reverse registration order
        for p in reversed(self.plugins):
            try:
                await p.stop()
            except Exception:
                logger.exception("plugin %s stop() failed", p.name)
        # 4. cancel dispatch
        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
        for t in (self._mic_task, self._dispatch_task):
            if t is None:
                continue
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        # 5. close transport
        try:
            await self.slv.close()
        except Exception:  # pragma: no cover
            pass
        # 6. drain playback
        try:
            await self.audio.stop_playback()
            await self.audio.close()
        except Exception:  # pragma: no cover
            pass
        # 7. release LLM client resources (HTTP connection pool, etc.)
        try:
            await self.llm.aclose()
        except Exception:  # pragma: no cover
            pass
        # 8. release translator client resources
        try:
            await self.translator.aclose()
        except Exception:  # pragma: no cover
            pass

    def request_shutdown(self) -> None:
        if self._shutdown_evt is not None:
            self._shutdown_evt.set()

    # ── internal pumps ──────────────────────────────────────────────

    async def restart_mic_capture(self, reason: str = "manual") -> None:
        """Restart only the local sounddevice input stream + mic pump.

        This is cheaper than restarting the whole agent and is useful after
        CoreAudio device changes / PaMacCore errors leave the input stream
        alive but no longer delivering useful chunks.
        """
        lock = getattr(self, "_mic_restart_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._mic_restart_lock = lock
        async with lock:
            logger.warning("restarting mic capture (%s)", reason)
            task = getattr(self, "_mic_task", None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                stop_input = getattr(self.audio, "_stop_input_stream", None)
                if callable(stop_input):
                    stop_input()
            except Exception:
                logger.debug("stop input stream failed during mic restart", exc_info=True)
            self._vad_state = "idle"
            self._vad_speech_ms = 0
            self._vad_silence_ms = 0
            self._vad_eos_sent = False
            try:
                reset = getattr(self._client_vad, "reset", None)
                if callable(reset):
                    reset()
            except Exception:
                logger.debug("client VAD reset failed during mic restart", exc_info=True)
            self._last_mic_chunk_ts = time.monotonic()
            # Grace window for the wake watchdog: a mic restart briefly stops
            # the capture callback, so the wake loop won't get chunks for a
            # moment — don't judge it stalled during/just after a mic restart.
            self._mic_restart_ts = time.monotonic()
            self._mic_task = asyncio.create_task(self._mic_pump(), name="mic-pump")

    async def _slv_link_watchdog(self) -> None:
        """Keep the speech-service link alive without human intervention.

        Two failure modes are covered, both observed on 2026-09-13 after
        recreating the `speech` container (which is exactly what an image
        update does):

          1. the dispatch loop stops retrying (its reader-exit path goes quiet),
             so the agent sits disconnected forever;
          2. the dispatch task itself ends, leaving nothing that consumes SLV
             events (every later `asr_eos` then hits a dead WS).

        Bounded reconnect budgets (`SLVClient._RECONNECT_BACKOFFS`, 4 attempts)
        are right for an interactive call like `wake()`; for the long-lived
        conversation link they must not be the last word, so this watchdog
        retries forever with a grace window between attempts and restarts the
        dispatch task whenever it finishes.
        """
        retry_every_s = 2.0
        # NB: `or 10.0` would swallow a deliberate 0 (instant retry).
        _grace = getattr(self.config, "slv_link_grace_s", None)
        grace_s = 10.0 if _grace is None else max(0.0, float(_grace))
        while True:
            await asyncio.sleep(retry_every_s)
            try:
                shutdown = getattr(self, "_shutdown_evt", None)
                if shutdown is not None and shutdown.is_set():
                    return
                slv = getattr(self, "slv", None)
                if slv is None or getattr(slv, "_closed", False):
                    return

                task = getattr(self, "_dispatch_task", None)
                if task is not None and task.done():
                    # Nothing consumes SLV events any more: revive the loop.
                    exc = None
                    if not task.cancelled():
                        exc = task.exception()
                    logger.error(
                        "SLV dispatch task ended (exc=%r); restarting it", exc
                    )
                    self._dispatch_task = asyncio.create_task(
                        self._slv_dispatch(), name="slv-dispatch"
                    )
                    self._slv_unhealthy_since = None
                    continue

                if slv.is_healthy():
                    self._slv_unhealthy_since = None
                    continue

                now = time.monotonic()
                if self._slv_unhealthy_since is None:
                    self._slv_unhealthy_since = now
                if now - self._slv_unhealthy_since < grace_s:
                    continue
                if slv.is_reconnecting():
                    continue
                logger.warning(
                    "SLV link unhealthy for %.0fs; forcing a reconnect",
                    now - self._slv_unhealthy_since,
                )
                await asyncio.wait_for(slv.reconnect(), timeout=10.0)
                # Success clears the clock on the next tick; leave it set so a
                # reconnect that returns without a healthy WS keeps retrying.
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - retry forever
                logger.warning("SLV link watchdog reconnect failed: %s", exc)

    async def _mic_watchdog(self) -> None:
        """Recover from dead CoreAudio/sounddevice capture streams."""
        stale_s = 5.0
        try:
            while True:
                await asyncio.sleep(2.0)
                if getattr(self, "_shutdown_evt", None) is not None and self._shutdown_evt.is_set():
                    return
                task = getattr(self, "_mic_task", None)
                if task is None:
                    await self.restart_mic_capture("watchdog:no-task")
                    continue
                if task.done():
                    exc = None
                    try:
                        exc = task.exception()
                    except (asyncio.CancelledError, Exception):
                        exc = None
                    logger.warning("mic pump stopped; restarting (exc=%r)", exc)
                    await self.restart_mic_capture("watchdog:task-done")
                    continue
                last = getattr(self, "_last_mic_chunk_ts", None)
                if last is not None and (time.monotonic() - last) > stale_s:
                    await self.restart_mic_capture("watchdog:stale")
        except asyncio.CancelledError:
            raise

    def _find_wake_source(self):
        """Locate the wake source exposing the heartbeat/restart hooks
        (duck-typed so app_base needn't import OpenWakeWordSource)."""
        for p in self.plugins:
            if callable(getattr(p, "last_chunk_ts", None)) and callable(
                getattr(p, "request_restart", None)
            ):
                return p
        return None

    async def _wake_watchdog(self) -> None:
        """Recover a silently-stalled wake source (its listen loop wedged on
        ``tap.get()`` — no exception, so the source's own supervisor can't see
        it) while the mic capture is still healthy. This is the ONLY recovery
        for that case; crashes are handled by the supervisor itself."""
        WAKE_STALE_S = 10.0
        MIC_FRESH_S = 5.0
        GRACE_S = 15.0
        try:
            while True:
                await asyncio.sleep(2.0)
                if getattr(self, "_shutdown_evt", None) is not None and self._shutdown_evt.is_set():
                    return
                ws = self._find_wake_source()
                if ws is None:
                    continue
                now = time.monotonic()
                # Boot + post-mic-restart grace: the loop needs time to
                # (re)acquire its tap and start pulling chunks.
                if now - self._boot_ts < GRACE_S:
                    continue
                if self._mic_restart_ts and (now - self._mic_restart_ts) < GRACE_S:
                    continue
                # Only act when the MIC is fresh. If the mic itself is stale
                # that's the mic watchdog's job — restarting the wake loop
                # wouldn't help and would thrash.
                mic_last = getattr(self, "_last_mic_chunk_ts", None)
                if mic_last is None or (now - mic_last) > MIC_FRESH_S:
                    continue
                wake_last = ws.last_chunk_ts()
                if wake_last is None:
                    continue  # no chunk processed yet → grace, not a stall
                if (now - wake_last) > WAKE_STALE_S:
                    logger.warning(
                        "wake loop stale (%.0fs since last chunk; mic fresh %.1fs "
                        "ago) — restarting wake source",
                        now - wake_last, now - mic_last,
                    )
                    lock = self._wake_restart_lock or asyncio.Lock()
                    async with lock:
                        ws.request_restart()
        except asyncio.CancelledError:
            raise

    async def _send_audio_nonblocking(self, pcm: bytes) -> None:
        """Send a mic chunk to SLV with a short ceiling on how long the
        send may block.

        Why: ``SLVClient._send_lock`` serialises the send half of the WS,
        and the dispatch loop's auto-reconnect (``slv.reconnect()``) holds
        the same lock for the duration of a fresh ``ws_connect`` — which
        can stall for several seconds on a network blip / DNS hiccup.
        Without a ceiling here, every mic chunk during reconnect parks on
        the lock, the mic_pump coroutine stops draining its input queue,
        sounddevice's callback thread floods ``call_soon_threadsafe`` with
        un-consumed PCM, and the log starts hemorrhaging
        ``mic queue full -- dropping chunk`` for the entire outage —
        which is exactly the "agent feels dead" symptom.

        Bounded wait + drop is the right trade for a mic stream: the
        chunks we drop while SLV is briefly unreachable would have been
        useless anyway (the WS that would have carried them is closed),
        and the post-reconnect first ASR utterance starts from fresh
        chunks. Pre-roll is still preserved for the *current* speech
        segment whose onset already won the VAD race.
        """
        # Debug inject (dashboard /api/control/inject_wav) feeds a clip STRAIGHT
        # to the SLV WS; while it does, suppress real-mic forwarding so the live
        # mic stream doesn't interleave with / drown the injected audio on the
        # SAME WS (observed real-machine: SLV transcribed ambient room audio
        # instead of the injected command). The inject sends via slv.send_audio
        # directly, so it bypasses this gate.
        if getattr(self, "_injecting", False):
            return
        # Race #3 fast path: while SLV is actively reconnecting, skip the
        # send_lock entirely — otherwise every mic chunk queues behind the
        # reconnect's grace+_open_with_retry chain (~50-2000ms), the 0.5s
        # ceiling was too tight (false-positive drop log floods) and 2.0s
        # is still too long if a hundred chunks pile up.
        is_reconn = getattr(self.slv, "is_reconnecting", None)
        if callable(is_reconn) and is_reconn():
            logger.debug("send_audio skipped (slv reconnecting)")
            return
        try:
            await asyncio.wait_for(self.slv.send_audio(pcm), timeout=2.0)
        except asyncio.TimeoutError:
            # SLV is mid-reconnect / unreachable; don't wedge the mic pump.
            # Logged at debug to avoid floods during normal reconnect blips.
            logger.debug("send_audio timed out (slv reconnecting?); dropping chunk")
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            logger.debug("send_audio failed; dropping chunk", exc_info=True)

    def _schedule_mic_rms_broadcast(self, data: dict) -> bool:
        task = getattr(self, "_mic_rms_broadcast_task", None)
        if task is not None and not task.done():
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        self._mic_rms_broadcast_task = loop.create_task(
            self._broadcast("on_mic_rms", data),
            name="mic-rms-broadcast",
        )
        return True

    async def _mic_pump(self) -> None:
        """Mic capture loop. When client VAD is enabled, only forwards audio
        to SLV during (and just before) actual speech — pre-roll buffer
        ensures the first ~300ms of an utterance isn't lost while the VAD
        is still confirming speech-start. Idle silence is never sent.

        Why: streaming background noise for minutes at a time saturates the
        WS write pipeline and starves websockets' keepalive ping coroutine,
        triggering 1011 keepalive ping timeout. Dropping idle chunks keeps
        the connection mostly quiet between turns.
        """
        from ovs_agent.audio.vad_gate import PrerollRing

        try:
            chunk_ms = getattr(self.audio, "chunk_ms", 100)
            preroll_max = max(1, 400 // max(chunk_ms, 1))  # ~400ms
            # Pre-roll ring + speech-onset drain — extracted to the shared,
            # robot-agnostic helper (ovs_agent.audio.vad_gate.PrerollRing).
            # _update_vad still owns VAD state / EOS / barge-in; this ring
            # only buffers idle chunks and replays them once at onset, which
            # is exactly the prior inline `deque` behaviour.
            preroll = PrerollRing(preroll_max)
            import numpy as _np
            # Rate-limit on_mic_rms broadcast: 10Hz is overkill for a
            # dashboard sparkline, and awaiting every plugin every 100ms
            # was starving the mic queue during TTS playback — VAD never
            # saw the burst of audio when the user spoke, so barge-in
            # never fired. Broadcast at most every ~200ms (every 2nd
            # chunk at chunk_ms=100, every 4th at chunk_ms=50).
            rms_broadcast_every = max(1, 200 // max(chunk_ms, 1))
            rms_chunk_counter = 0
            # ── continuous-dialogue mic-pump enhancements (opt-in via config) ──
            # All default OFF so existing deployments are unchanged; a solution
            # tunes these in its agent.yaml for its specific mic / acoustics.
            gate_enabled = bool(getattr(self.config, "energy_gate_enabled", False))
            gate_open = float(getattr(self.config, "energy_gate_open_rms", 0.08))
            gate_close = float(getattr(self.config, "energy_gate_close_rms", 0.05))
            gate_hangover_ms = float(getattr(self.config, "energy_gate_hangover_ms", 250.0))
            makeup_gain = float(getattr(self.config, "mic_makeup_gain", 1.0))
            drive_eos = bool(getattr(self.config, "gate_drive_eos", False))
            eos_min_speech_ms = float(getattr(self.config, "gate_eos_min_speech_ms", 250.0))
            gate_eos_delay_ms = float(getattr(self.config, "gate_eos_delay_ms", 0.0) or 0.0)
            drop_while_speaking = bool(getattr(self.config, "mic_drop_while_speaking", False))
            need_rms = gate_enabled or makeup_gain != 1.0
            _gate_open_state = False
            _gate_opened_at = 0.0
            _gate_last_loud_ts = 0.0
            _gate_zeros: bytes | None = None
            _diag_fwd_real = 0  # DIAG: real (non-zero) chunks forwarded to SLV

            def _reset_energy_gate(reason: str) -> None:
                nonlocal _gate_open_state, _gate_opened_at
                nonlocal _gate_last_loud_ts, _diag_fwd_real
                if _gate_open_state:
                    logger.info("DIAG mic-gate RESET reason=%s", reason)
                _gate_open_state = False
                _gate_opened_at = 0.0
                _gate_last_loud_ts = 0.0
                _diag_fwd_real = 0

            if gate_enabled or makeup_gain != 1.0 or drop_while_speaking:
                logger.info(
                    "mic-pump enhancements: gate=%s open=%.4f close=%.4f hangover=%.0fms "
                    "makeup_gain=%.1f drive_eos=%s eos_delay=%.0fms drop_while_speaking=%s",
                    gate_enabled, gate_open, gate_close, gate_hangover_ms,
                    makeup_gain, drive_eos, gate_eos_delay_ms, drop_while_speaking,
                )
            async for chunk in self.audio.start_capture():
                self._last_mic_chunk_ts = time.monotonic()
                # pipeline_mode gating: drop audio entirely while SLEEPING.
                # WS stays connected so wake-time reconnect cost is zero.
                # When mic_drop_while_speaking is set, also drop during
                # SPEAKING/THINKING (echo gate): the mic captures our own TTS
                # echo, and forwarding it opens a server-VAD segment that never
                # cleanly ends → wedges the continuous-dialogue loop.
                _st = getattr(self, "_state", ConvState.IDLE)
                if _st == ConvState.SLEEPING or (
                    drop_while_speaking
                    and _st in (ConvState.SPEAKING, ConvState.THINKING)
                ):
                    # Also clear pre-roll so we don't leak pre-sleep / TTS-echo
                    # audio into the next user utterance.
                    preroll.clear()
                    _reset_energy_gate(_st.value)
                    continue
                # Race #3: while SLV is reconnecting, drop chunks and
                # clear pre-roll. Carrying pre-reconnect audio into the
                # new WS produces partial garbled finals on the next turn
                # (audio fragment from before + speech after the gap →
                # ASR sees one mashed utterance).
                is_reconn = getattr(self.slv, "is_reconnecting", None)
                if callable(is_reconn) and is_reconn():
                    preroll.clear()
                    _reset_energy_gate("slv_reconnecting")
                    continue
                # Skip the wake-word tail: for a brief window after a wake-word
                # fires, drop mic audio so the trailing "Hey Jarvis" + reverb
                # isn't forwarded into the command utterance (else the server
                # ASR decodes wake-word+command as one garbled segment). Clear
                # preroll too so none of it is carried over.
                if time.monotonic() < getattr(self, "_wake_mic_skip_until", 0.0):
                    preroll.clear()
                    _reset_energy_gate("wake_mic_skip")
                    continue
                if time.monotonic() < getattr(self, "_local_output_mic_suppress_until", 0.0):
                    preroll.clear()
                    _reset_energy_gate("local_output")
                    continue
                # (#38) Hold audio until boot-time tool advertise finished.
                # Set immediately in client-loop mode (advertise is a no-op),
                # so this never adds latency outside the first server-loop
                # boot. Drop (don't queue) chunks during the brief window so
                # we don't carry pre-advertise audio into the first turn.
                _adv_ready = getattr(self, "_advertise_ready", None)
                if _adv_ready is not None and not _adv_ready.is_set():
                    preroll.clear()
                    _reset_energy_gate("advertise_not_ready")
                    continue
                # Per-chunk mic RMS. The energy gate / makeup gain need it
                # EVERY chunk; otherwise it's only needed for the rate-limited
                # dashboard broadcast (a slow WS client must not backpressure
                # the mic queue and starve VAD → barge-in during TTS).
                rms = 0.0
                if need_rms:
                    try:
                        _arr = _np.frombuffer(chunk, dtype=_np.int16)
                        rms = float(_np.sqrt(_np.mean((_arr.astype(_np.float32) / 32768.0) ** 2))) if _arr.size else 0.0
                    except Exception:  # pragma: no cover - defensive
                        rms = 0.0
                rms_chunk_counter = (rms_chunk_counter + 1) % rms_broadcast_every
                if rms_chunk_counter == 0:
                    try:
                        if not need_rms:
                            _arr = _np.frombuffer(chunk, dtype=_np.int16)
                            rms = float(_np.sqrt(_np.mean((_arr.astype(_np.float32) / 32768.0) ** 2))) if _arr.size else 0.0
                        thr = float(getattr(self.config, "client_vad_threshold", None) or 0.012)
                        self._schedule_mic_rms_broadcast(
                            {"rms": rms, "threshold": thr, "state": self._vad_state}
                        )
                        # Diagnostic floor must sit BELOW the energy gate's
                        # open threshold, or failed quiet-speech attempts leave
                        # zero log trace (this hid the 2026-08 shouting-only-mic
                        # diagnosis for weeks). Derive it from the configured
                        # gate instead of a constant so it tracks per-app tuning;
                        # apps without the gate keep the historical 0.03 floor.
                        loud_floor = gate_open * 0.9 if gate_enabled else 0.03
                        if rms > loud_floor:
                            logger.info(
                                "mic chunk loud: rms=%.4f state=%s convstate=%s",
                                rms, self._vad_state,
                                getattr(self, "_state", ConvState.IDLE).value,
                            )
                    except Exception:  # pragma: no cover - defensive
                        pass

                if self._client_vad is None:
                    # No client VAD: forward raw audio so the SERVER VAD
                    # segments + finalizes (preserves Qwen3 unbroken framing).
                    # Optional, all opt-in: energy gate (zero-fill gaps → clean
                    # silence so the server endpoints), makeup gain (quiet mic →
                    # the VAD/ASR trained range), and client-driven asr_eos on
                    # the gate's open→close edge (server finalizes immediately
                    # instead of relying on its own VAD endpoint, which wedges).
                    out_chunk = chunk
                    pending_gate_eos = False
                    now_mono = time.monotonic()
                    if gate_enabled:
                        if rms >= gate_open:
                            if not _gate_open_state:
                                _gate_opened_at = now_mono
                                _diag_fwd_real = 0
                                # DIAG: gate edge — closed→open (speech onset)
                                logger.info(
                                    "DIAG mic-gate OPEN rms=%.4f convstate=%s",
                                    rms, getattr(self, "_state", ConvState.IDLE).value,
                                )
                                # Pre-roll replay: the gate was CLOSED until this
                                # chunk, so the low-energy ONSET of the command
                                # (e.g. the unvoiced '抓' in '抓盒子') was sent to
                                # SLV as zero-fill and the server ASR only heard
                                # the loud tail ('盒子'). Replay the real audio
                                # buffered just before the gate opened so ASR sees
                                # the whole word. Apply makeup gain so it matches
                                # the gated stream's level.
                                for _buf in preroll.drain():
                                    _ob = _buf
                                    if makeup_gain != 1.0 and len(_ob):
                                        try:
                                            _gg = (
                                                _np.frombuffer(_ob, dtype=_np.int16).astype(_np.float32)
                                                * makeup_gain
                                            )
                                            _ob = _np.clip(_gg, -32768.0, 32767.0).astype(_np.int16).tobytes()
                                        except Exception:  # pragma: no cover - defensive
                                            pass
                                    _diag_fwd_real += 1
                                    await self._send_audio_nonblocking(_ob)
                            _gate_open_state = True
                            _gate_last_loud_ts = now_mono
                        elif rms < gate_close:
                            if (now_mono - _gate_last_loud_ts) * 1000.0 >= gate_hangover_ms:
                                if _gate_open_state:
                                    # DIAG: gate edge — open→closed (speech end)
                                    logger.info(
                                        "DIAG mic-gate CLOSE speech_ms=%.0f fwd_real=%d convstate=%s",
                                        (_gate_last_loud_ts - _gate_opened_at) * 1000.0,
                                        _diag_fwd_real,
                                        getattr(self, "_state", ConvState.IDLE).value,
                                    )
                                if _gate_open_state and drive_eos and (
                                    (_gate_last_loud_ts - _gate_opened_at) * 1000.0 >= eos_min_speech_ms
                                ):
                                    # Defer EOS until after this close/silence
                                    # chunk has been written to SLV. Sending
                                    # asr_eos before the boundary chunk can make
                                    # the server finalize a short command before
                                    # it has consumed the tail audio.
                                    pending_gate_eos = True
                                _gate_open_state = False
                                _diag_fwd_real = 0
                        if not _gate_open_state:
                            # Gate closed: buffer the REAL chunk for pre-roll
                            # (replayed at the next gate-open so the command
                            # onset survives), but send zeros to SLV now to keep
                            # the inter-utterance silence clean. The ring keeps
                            # only the last ~preroll_max chunks, so a long
                            # silence still replays just the moments before
                            # speech, not stale audio.
                            preroll.append(chunk)
                            if _gate_zeros is None or len(_gate_zeros) != len(chunk):
                                _gate_zeros = b"\x00" * len(chunk)
                            out_chunk = _gate_zeros
                    if makeup_gain != 1.0 and out_chunk is not _gate_zeros and len(out_chunk):
                        try:
                            _g = _np.frombuffer(out_chunk, dtype=_np.int16).astype(_np.float32) * makeup_gain
                            out_chunk = _np.clip(_g, -32768.0, 32767.0).astype(_np.int16).tobytes()
                        except Exception:  # pragma: no cover - defensive
                            pass
                    if out_chunk is not _gate_zeros:
                        _diag_fwd_real += 1
                    await self._send_audio_nonblocking(out_chunk)
                    if pending_gate_eos:
                        if gate_eos_delay_ms > 0:
                            await asyncio.sleep(gate_eos_delay_ms / 1000.0)
                        try:
                            sent_eos = await self.send_asr_eos_once()
                            if sent_eos:
                                self._cancel_wake_command_timeout()
                                self._set_state(ConvState.THINKING)
                                logger.info(
                                    "DIAG mic-gate EOS sent; state=thinking delay_ms=%.0f",
                                    gate_eos_delay_ms,
                                )
                        except Exception:
                            logger.exception("gate-edge asr_eos failed")
                    continue

                # Update VAD first; it may transition idle→speech this chunk.
                try:
                    await self._update_vad(chunk, chunk_ms)
                except Exception:
                    logger.exception("client VAD update failed")

                if self._vad_state == "speech":
                    # Drain the pre-roll buffer at speech onset (no-op on
                    # subsequent chunks once drained), then stream this chunk
                    # plus subsequent ones in real time.
                    for buffered in preroll.drain():
                        await self._send_audio_nonblocking(buffered)
                    await self._send_audio_nonblocking(chunk)
                else:
                    # Idle: keep a short rolling buffer but don't transmit.
                    preroll.append(chunk)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("mic pump crashed")

    async def send_asr_eos_once(self) -> bool:
        """Send asr_eos to SLV at most once per turn.

        Returns True if this call actually sent the EOS, False if it
        was a duplicate (already sent this turn). The flag is reset on
        ASRFinal / PTT-start / SLV reconnect.

        Arms an `_asr_final_watchdog` so the state machine self-recovers
        if SLV doesn't echo an ASR final back (always_on pipeline mode
        silently drops empty finals — without this, the FSM stays
        THINKING forever after a noise-triggered turn).
        """
        if getattr(self, "_eos_sent_this_turn", False):
            return False
        self._eos_sent_this_turn = True
        try:
            await self.slv.asr_eos()
        except Exception:
            logger.exception("asr_eos send failed")
            # Don't clear the flag — even on failure we don't want to
            # retry a second time within the same turn and risk the SLV
            # state machine getting into an inconsistent state.
        # Arm watchdog (cancels any stale one from a prior failed turn).
        self._cancel_asr_watchdog()
        self._asr_watchdog_task = asyncio.create_task(
            self._asr_final_watchdog(),
            name="asr-final-watchdog",
        )
        return True

    def _cancel_asr_watchdog(self) -> None:
        """Cancel any pending asr_final watchdog (idempotent)."""
        task = getattr(self, "_asr_watchdog_task", None)
        if task is not None and not task.done():
            task.cancel()
        self._asr_watchdog_task = None

    # ── silero-primary stall fallback (config.vad_stall_eos_ms) ──────
    def _arm_stall_watchdog(self) -> None:
        """(Re)arm the no-partial stall timer. Called on every real
        asr_partial so it acts as an inactivity timeout: as long as silero
        keeps emitting partials (incl. through a long sentence) the timer is
        reset and never fires; it only fires after silero goes quiet without
        a final. No-op when disabled (vad_stall_eos_ms<=0)."""
        if float(getattr(self.config, "vad_stall_eos_ms", 0.0) or 0.0) <= 0:
            return
        self._cancel_stall_watchdog()
        self._stall_watchdog_task = asyncio.create_task(
            self._stall_eos_watchdog(), name="stall-eos-watchdog",
        )

    def _cancel_stall_watchdog(self) -> None:
        """Cancel any pending stall watchdog (idempotent)."""
        task = getattr(self, "_stall_watchdog_task", None)
        if task is not None and not task.done():
            task.cancel()
        self._stall_watchdog_task = None

    async def _stall_eos_watchdog(self) -> None:
        """Force one asr_eos if silero stalls (no asr_partial for
        vad_stall_eos_ms) while we're still awaiting a command final. Rechecks
        state before firing so a barge-in / sleep that landed in the meantime
        doesn't trigger a spurious EOS. send_asr_eos_once is idempotent, so
        this never double-finalizes."""
        timeout = float(getattr(self.config, "vad_stall_eos_ms", 0.0) or 0.0) / 1000.0
        if timeout <= 0:
            return
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        if getattr(self, "_state", ConvState.IDLE) not in (
            ConvState.IDLE, ConvState.LISTENING
        ):
            return  # moved on (speaking/thinking/sleeping) — not our turn to end
        if getattr(self, "_eos_sent_this_turn", False):
            return  # already finalized this turn
        logger.info(
            "VAD stall: no asr_partial in %.1fs while %s — forcing asr_eos "
            "(silero-primary fallback)", timeout,
            getattr(self, "_state", ConvState.IDLE).name,
        )
        try:
            await self.send_asr_eos_once()
        except Exception:
            logger.exception("stall-eos asr_eos send failed")

    # ── server-loop mode (#37 Phase 2-product) ──────────────────────

    def _log_boot_diagnostic(self) -> None:
        """Log a one-line boot diagnostic for the server-loop flag.

        Boot diagnostic (2026-05-31 server-loop activation bug): surface the
        *raw* env value via ``repr()`` so quotes/whitespace are visible,
        alongside the resolved decision. Production ``--env-file`` injection
        delivered the flag as the literal 3-char string ``'"1"'`` (with
        quotes), which the old parser failed to recognise → server-loop
        silently off. This single line makes "what did the process actually
        read?" obvious in prod logs.
        """
        logger.info(
            "boot: pid=%d OVS_AGENT_SERVER_LOOP(raw)=%r config.server_loop=%r "
            "resolved server_loop_enabled=%s",
            os.getpid(),
            os.environ.get("OVS_AGENT_SERVER_LOOP"),
            getattr(self.config, "server_loop", None),
            self._server_loop_enabled(),
        )

    def _default_mode_tool_allow(self) -> "set[str] | None | bool":
        """Tool set the first turn will send, resolved exactly like
        ``ModeContext`` does per turn (app_mode.py): the mode's config
        override, else the mode object's attribute; ``None`` there means
        "use the global ``tools_enabled`` / ``tools_default_allowlist``".

        The mode is the one the ModeManager actually started (it falls back
        to the first registered mode when ``default_mode`` is missing), or
        ``config.default_mode`` for apps without one.

        Returns False (no tools), None (every registered tool) or the
        allow-listed names.
        """
        cfg = self.config
        manager = getattr(self, "modes", None)
        mode = getattr(manager, "_current", None)
        if mode is None and manager is not None:
            registered = getattr(manager, "_modes", None) or {}
            mode = registered.get(getattr(cfg, "default_mode", None) or "chat")
            if mode is None and registered:
                mode = next(iter(registered.values()))
        mode_name = getattr(mode, "name", None) or getattr(cfg, "default_mode", None) or "chat"
        overrides = getattr(cfg, "mode_overrides", None) or {}
        mode_cfg = overrides.get(mode_name) if isinstance(overrides, dict) else None
        mode_cfg = mode_cfg if isinstance(mode_cfg, dict) else {}

        def _resolve(key: str):
            if key in mode_cfg:
                return mode_cfg[key]
            if mode is not None:
                return getattr(mode, key, None)
            return None

        enabled = _resolve("tools_enabled")
        if enabled is None:
            enabled = getattr(cfg, "tools_enabled", False)
        if not bool(enabled):
            return False
        allow = _resolve("tools_allowlist")
        if allow is None:
            allow = getattr(cfg, "tools_default_allowlist", []) or []
        return set(allow) if allow else None

    async def _maybe_run_llm_warmup(self) -> None:
        """Warm the local LLM backend's prefix KV cache + CUDA graph.

        Server-loop mode runs the LLM + tool loop on SLV, not locally, so a
        local warmup is pure waste here — worse, it emits misleading
        "Session budget OK" lines and drives edge-llm traffic that (during the
        2026-05-31 server-loop activation triage) actively misled diagnosis into
        thinking the local LLM path was live. So skip it entirely in server-loop
        mode. The client-loop (legacy) warmup path is byte-for-byte unchanged.
        """
        if self._server_loop_enabled():
            logger.info(
                "server-loop mode: skipping local LLM warmup "
                "(LLM + tool loop runs on SLV, not on the agent)"
            )
            return
        # EdgeLLMBackend warms both the prefix KV cache and the TRT-LLM CUDA
        # graph; other backends inherit the default no-op.
        try:
            # Warm with the tool set the turns will actually send. Warming
            # with every registered tool when turns send none is not just a
            # cache miss: a runtime that keeps tools registered across
            # requests (RK1828 worker) then renders the tool preamble into
            # every tool-less turn, and the model "calls" set_mode as plain
            # text that reaches TTS (rk3588 devkit, 2026-09-21: prefill 220 vs
            # 28 tokens for the same tool-less request).
            tools_payload = None
            allow = self._default_mode_tool_allow()
            registry = getattr(self, "tool_registry", None)
            if (
                allow is not False
                and registry is not None
                and hasattr(registry, "list_openai_tools")
            ):
                try:
                    tools_payload = registry.list_openai_tools(allow=allow) or None
                except Exception:
                    logger.debug("warmup: tool_registry lookup failed", exc_info=True)
                    tools_payload = None
            sys_prompt = ""
            try:
                overrides = getattr(self.config, "mode_overrides", {}) or {}
                mode_cfg = overrides.get("chat") if isinstance(overrides, dict) else None
                if isinstance(mode_cfg, dict):
                    sys_prompt = mode_cfg.get("system_prompt") or ""
            except Exception:
                pass
            if not sys_prompt:
                sys_prompt = getattr(self.config, "system_prompt", "") or ""
            # Validate session budget BEFORE warmup. If the fixed prefix
            # (system_prompt + tools schema) is too close to the trim
            # threshold, every turn will trim → cache_warmed cleared →
            # KV-cache hot path defeated. Logs ERROR if misconfigured,
            # WARNING if tight, INFO if healthy.
            self._validate_session_budget(sys_prompt, tools_payload)
            warmup_result = await self.llm.warmup(
                system_prompt=sys_prompt,
                tools=tools_payload,
                enable_thinking=False,
            )
            if warmup_result:
                logger.info("LLM backend warmup result: %s", warmup_result)
                if self.session is not None:
                    if warmup_result.get("cache_warmed"):
                        self.session.prefix_cache_warmed = True
                        logger.info(
                            "session.prefix_cache_warmed=True after backend "
                            "warmup; first turn will use prefix_cache"
                        )
                    if warmup_result.get("graph_warmed"):
                        self.session.graph_warmed = True
                # Partial warmup hint (Plan D item 7): prefix is hot but
                # graph isn't — first tool_call decode may still pay JIT
                # / CUDA-graph capture cost. Surface so operators can
                # diagnose unexpected first-turn latency.
                if (
                    warmup_result.get("cache_warmed")
                    and not warmup_result.get("graph_warmed")
                ):
                    logger.info(
                        "partial warmup: prefix cached but engine graph "
                        "not warmed; first tool_call decode may be slow"
                    )
                # Plan D item 4: validate session_max_input_tokens vs
                # observed engine max_seq_len (when available).
                engine_max = warmup_result.get("engine_max_seq_len")
                cfg_max = getattr(self.config, "session_max_input_tokens", None)
                if isinstance(engine_max, int) and isinstance(cfg_max, int):
                    if cfg_max > engine_max - 1000:
                        logger.warning(
                            "session budget tight: session_max_input_tokens=%d "
                            "is within 1000 of engine max_seq_len=%d. After "
                            "trim+output the request may still hit the engine "
                            "ceiling. Consider lowering to ~%d.",
                            cfg_max, engine_max, max(1024, engine_max - 1500),
                        )
                    elif cfg_max < engine_max // 2:
                        logger.info(
                            "session budget conservative: "
                            "session_max_input_tokens=%d is <50%% of engine "
                            "max_seq_len=%d. You could raise it for longer "
                            "conversation history (user config takes priority "
                            "— not auto-adjusting).",
                            cfg_max, engine_max,
                        )
                    else:
                        logger.info(
                            "session budget OK vs engine: "
                            "session_max_input_tokens=%d, engine_max_seq_len=%d",
                            cfg_max, engine_max,
                        )
        except Exception:
            logger.warning("LLM warmup failed; first turn may be cold", exc_info=True)

    def _server_loop_enabled(self) -> bool:
        """True when the agent should run in server-loop client mode.

        Resolves via ``Config.server_loop_enabled()`` (OVS_AGENT_SERVER_LOOP
        env wins, else the ``server_loop`` config field). Defaults to False
        so the legacy client-loop path is the no-op default.
        """
        cfg = getattr(self, "config", None)
        if cfg is None:
            return False
        fn = getattr(cfg, "server_loop_enabled", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:  # pragma: no cover - defensive
                return False
        return bool(getattr(cfg, "server_loop", False))

    def _resolve_chat_system_prompt(self) -> str:
        """Resolve the system prompt the way the LLM warmup path does:
        mode_overrides['chat'].system_prompt → config.system_prompt."""
        sys_prompt = ""
        try:
            overrides = getattr(self.config, "mode_overrides", {}) or {}
            mode_cfg = overrides.get("chat") if isinstance(overrides, dict) else None
            if isinstance(mode_cfg, dict):
                sys_prompt = mode_cfg.get("system_prompt") or ""
        except Exception:  # pragma: no cover - defensive
            sys_prompt = ""
        if not sys_prompt:
            sys_prompt = getattr(self.config, "system_prompt", "") or ""
        return sys_prompt

    async def _advertise_tools_if_server_loop(self) -> None:
        """Advertise local tool schemas to SLV when in server-loop mode.

        Sends CLIENT_TOOL_ADVERTISE with the OpenAI-style tool schemas from
        the registry, the resolved system prompt, and a small bundle of LLM
        sampling params so the server can drive the loop the way the client
        would have. No-op (and never touches the WS) when the flag is off.
        """
        if not self._server_loop_enabled():
            logger.info(
                "server-loop mode OFF (OVS_AGENT_SERVER_LOOP raw=%r, "
                "config.server_loop=%r) — skipping tool advertise; running "
                "legacy client-loop",
                os.environ.get("OVS_AGENT_SERVER_LOOP"),
                getattr(self.config, "server_loop", None),
            )
            return
        registry = getattr(self, "tool_registry", None)
        tools_payload: list[dict] = []
        # Prefer list_advertise_tools: it carries per-tool preamble_text /
        # completion_text / response_mode so voxedge's server-loop engine can
        # fire the spoken "好的。" preamble + skip LLM round 2 for template
        # tools. Plain list_openai_tools strips those (server-loop then loses
        # the preamble + tight reply). Fall back for older registries.
        if registry is not None and hasattr(registry, "list_advertise_tools"):
            try:
                tools_payload = registry.list_advertise_tools(allow=None) or []
            except Exception:
                logger.warning("server-loop advertise: tool list failed", exc_info=True)
                tools_payload = []
        elif registry is not None and hasattr(registry, "list_openai_tools"):
            try:
                tools_payload = registry.list_openai_tools(allow=None) or []
            except Exception:
                logger.warning("server-loop advertise: tool list failed", exc_info=True)
                tools_payload = []
        system_prompt = self._resolve_chat_system_prompt()
        llm_params: dict = {}
        try:
            model = getattr(self.config, "llm_model", None)
            if model:
                llm_params["model"] = model
            max_iters = getattr(self.config, "tools_max_iterations", None)
            if max_iters is not None:
                llm_params["max_tool_iterations"] = int(max_iters)
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            await self.slv.advertise_tools(
                tools_payload,
                system_prompt=system_prompt,
                llm_params=llm_params or None,
            )
            # Record which session generation now carries our tools so the
            # dispatch path can detect a later send-path-revived session that
            # hasn't been advertised to yet and lazily re-advertise (#3).
            try:
                self._advertised_gen = self.slv.session_gen()
            except AttributeError:  # pragma: no cover - older SLV double
                pass
            logger.info(
                "server-loop mode: advertised %d tool(s) to SLV", len(tools_payload)
            )
        except Exception:
            logger.exception("server-loop advertise_tools failed")

    async def _readvertise_after_reconnect(self) -> None:
        """Re-advertise tool schemas after a successful SLV reconnect (#38).

        SLV treats CLIENT_TOOL_ADVERTISE as an idempotent upsert, so calling
        this after every reconnect is safe: server-loop parity survives WS
        churn (the fresh WS otherwise has zero tools registered). No-op in
        client-loop mode. Best-effort — an advertise failure must never abort
        the reconnect-recovery path, so errors are logged and swallowed.
        """
        if not self._server_loop_enabled():
            return
        try:
            await self._advertise_tools_if_server_loop()
            logger.info("server-loop: re-advertised tools after reconnect")
        except Exception:  # pragma: no cover - defensive
            logger.exception("server-loop re-advertise after reconnect failed")

    async def _readvertise_if_session_advanced(self) -> None:
        """Re-advertise tools if the live SLV session is one we haven't
        advertised to yet (#3).

        The mic pump's ``send_audio`` calls ``slv.connect()`` to revive a dead
        WS, which opens the socket + sends config but does NOT advertise tools.
        ``events()`` then keeps streaming on the new reader without the dispatch
        loop ever reaching its reconnect/readvertise guard — so the server-loop
        LLM on the fresh session has zero tools and the first tool_call after
        revival silently fails. Called at the top of ``_dispatch_one`` so the
        first observed event of a new generation (usually an early ASRPartial,
        well before the server's ASR-final → LLM turn) triggers a re-advertise.
        Cheap no-op on the steady-state path (generation unchanged).
        """
        if not self._server_loop_enabled():
            return
        try:
            gen = self.slv.session_gen()
        except AttributeError:  # pragma: no cover - older SLV double
            return
        if gen != getattr(self, "_advertised_gen", -1):
            logger.info(
                "slv dispatch: session gen advanced (%s→%s) without a "
                "readvertise (send-path revival) — re-advertising tools",
                getattr(self, "_advertised_gen", -1), gen,
            )
            await self._readvertise_after_reconnect()

    def _spawn_tool_task(self, evt: "ServerToolCall") -> None:
        """Schedule a SERVER_TOOL_CALL handler as a tracked background task
        (#F4). Keeps a strong reference in ``_pending_tool_tasks`` so the task
        can't be garbage-collected mid-flight, and prunes it on completion."""
        # Lazy-init so this is robust when the app is built via
        # ``BaseApp.__new__`` (tests / hot-swap paths that bypass __init__).
        tasks = getattr(self, "_pending_tool_tasks", None)
        if tasks is None:
            tasks = set()
            self._pending_tool_tasks = tasks
        task = asyncio.create_task(self._run_tool_task(evt))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def _run_tool_task(self, evt: "ServerToolCall") -> None:
        """Background wrapper around :meth:`_handle_server_tool_call`.

        ``_handle_server_tool_call`` already converts handler errors into an
        ``ok=False`` CLIENT_TOOL_RESULT, so the server loop self-recovers; this
        wrapper only guards the send path itself so a background task never dies
        with an unretrieved exception (which asyncio would log noisily)."""
        try:
            await self._handle_server_tool_call(evt)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            logger.exception("server tool_call background task crashed (id=%s)", evt.id)

    async def _handle_server_tool_call(self, evt: "ServerToolCall") -> None:
        """Execute a remote SERVER_TOOL_CALL against the local registry and
        reply with CLIENT_TOOL_RESULT.

        Only reached in server-loop mode (the server never emits
        SERVER_TOOL_CALL otherwise). The handler runs where the resource
        lives (e.g. the arm), so dispatch is identical to the local-loop
        registry dispatch — only the trigger is different. Errors are
        converted to ``ok=False`` results so the server-side LLM loop can
        self-recover instead of stalling.
        """
        # Correlation-id trace (debug only). Kept from the round2-stall
        # investigation: pairs with the info-level "SLV tool_result" line so a
        # DEBUG run shows recv-id vs sent-id. The tool_result line at info is
        # enough for production (it shows the id + ok status per tool call).
        logger.debug(
            "server tool_call recv: id=%r name=%r", evt.id, evt.name
        )
        registry = getattr(self, "tool_registry", None)
        if registry is None:
            from .tools import default_registry as registry  # type: ignore
        guard_error = self._server_tool_trigger_guard_error(evt, registry)
        if guard_error is not None and bool(
            getattr(self.config, "tool_trigger_guard_log_only", False)
        ) and not bool(getattr(self.config, "tool_trigger_guard", False)):
            # Monitor mode: record the suspicion, let the call through.
            logger.warning(
                "trigger-guard MONITOR: would flag tool_call %r: %s",
                evt.name, guard_error,
            )
            guard_error = None
        if guard_error is not None:
            logger.warning(
                "server tool_call %r blocked by trigger guard: %s",
                evt.name, guard_error,
            )
            await self.slv.send_tool_result(
                evt.id, evt.name, ok=False, error=guard_error
            )
            return
        # Build a tool ctx mirroring the local-loop path (app_mode builds the
        # same shape). session/event_bus/config let arm handlers reach state.
        try:
            from .tools import ToolCallCtx
            tool_ctx = ToolCallCtx(
                session=getattr(self, "session", None),
                mode_manager=getattr(self, "modes", None),
                event_bus=getattr(self, "events", None),
                config=getattr(self, "config", None),
            )
        except Exception:  # pragma: no cover - defensive
            tool_ctx = None
        # Per-call deadline. A hung tool handler (e.g. camera get_frame blocking
        # on a USB glitch) would otherwise stall the whole turn forever, and a
        # never-returning dispatch means we never send a tool_result — the
        # server-loop sits waiting, the /v2v session never completes, and the
        # slot leaks (soft-deadlock). Bound the await with the server-provided
        # timeout when present, else a sane default. This does NOT change the
        # fire-and-forget grasp path's own semantics: it returns {"started":...}
        # quickly and the real work runs in its own background task; we only
        # bound how long we wait for the dispatch *ack* itself.
        dispatch_timeout = getattr(evt, "timeout_s", None)
        if not isinstance(dispatch_timeout, (int, float)) or dispatch_timeout <= 0:
            dispatch_timeout = 30.0
        try:
            result = await asyncio.wait_for(
                registry.dispatch(evt.name, evt.arguments, tool_ctx),
                timeout=float(dispatch_timeout),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "server tool_call %r dispatch timed out after %.1fs; "
                "returning error result so the turn proceeds",
                evt.name, float(dispatch_timeout),
            )
            await self.slv.send_tool_result(
                evt.id, evt.name, ok=False,
                error=f"tool '{evt.name}' timed out after {float(dispatch_timeout):.1f}s",
            )
            return
        except Exception as e:  # noqa: BLE001 - never let a handler kill dispatch
            logger.exception("server tool_call %r dispatch crashed", evt.name)
            await self.slv.send_tool_result(
                evt.id, evt.name, ok=False, error=str(e)
            )
            return
        # A refusal/dispatch-failure is NOT ok: parallel-mode tools ack with
        # {"started": True}; guards refuse with {"started": False, "error":
        # ...} and NO "success" key. Reporting those ok=True made the
        # server-loop LLM retry the same motion tool every round until the
        # iteration cap — the arm visibly "looped" (real machine 2026-06-12).
        ok = not (
            isinstance(result, dict)
            and (result.get("success") is False or result.get("started") is False)
        )
        if ok:
            await self.slv.send_tool_result(
                evt.id, evt.name, ok=True, result=result
            )
        else:
            err = ""
            if isinstance(result, dict):
                err = str(result.get("error", ""))
            await self.slv.send_tool_result(
                evt.id, evt.name, ok=False, error=err or "tool execution failed"
            )

    def _server_tool_trigger_guard_error(self, evt: "ServerToolCall", registry) -> str | None:  # noqa: ANN001
        if not (
            bool(getattr(self.config, "tool_trigger_guard", False))
            or bool(getattr(self.config, "tool_trigger_guard_log_only", False))
        ):
            return None
        # Scoped guard: only tools with a fixed literal-trigger vocabulary
        # (the arm motions) are guarded. Semantic tools — grasp_object maps a
        # free-form spoken object to a catalog label, so its intent legitimately
        # has no fixed trigger phrase — are EXEMPT. The exempt set is explicit
        # (config-overridable) rather than relying on the tool's description
        # happening to omit a "Triggers:" list, so a later description edit can't
        # silently re-guard a semantic tool.
        exempt = getattr(self.config, "tool_trigger_guard_exempt", None)
        if exempt is None:
            exempt = ("grasp_object",)
        if evt.name in set(exempt):
            return None
        user_text = str(getattr(self, "_last_user_utterance_text", "") or "")
        if not user_text.strip():
            return None
        tools = getattr(registry, "_tools", {}) or {}
        tool_meta = tools.get(evt.name)
        if tool_meta is None:
            return None
        phrases = _extract_tool_trigger_phrases(getattr(tool_meta, "description", ""))
        if not phrases:
            return None
        normalized_text = _normalize_tool_trigger_text(user_text)
        for phrase in phrases:
            normalized_phrase = _normalize_tool_trigger_text(phrase)
            if normalized_phrase and normalized_phrase in normalized_text:
                return None
        return (
            f"no trigger phrase for tool {evt.name!r} in current user text "
            f"{user_text!r}"
        )

    async def _interrupt_current_turn_for_barge_in(self) -> None:
        """Stop the audible assistant turn before accepting barge-in audio.

        Barge-in semantics are intentionally ordered:
          1. cancel the local LLM streaming task so no more text is sent;
          2. stop local speaker playback immediately;
          3. send SLV's in-band abort control to cancel the already queued /
             in-flight TTS synthesis;
          4. keep the SLV WebSocket alive **and** the in-flight ASR utterance
             alive (``abort(keep_asr=True)``) so the user's current speech
             keeps flowing to ASR without a gap.

        The current SLV protocol multiplexes ASR input and TTS output on one
        connection. Closing/reconnecting it here also drops exactly the audio
        we need for the barge-in utterance, which turns an immediate interrupt
        into a multi-second delayed response. The right control is the in-band
        `abort` frame: SLV cancels current TTS and drains queued sentences
        without tearing down the WebSocket.

        The same reasoning applies one level down. A plain `abort` also makes
        the server cancel the in-flight ASR utterance, which is exactly the
        audio we are trying to keep — measured on rk3588 (2026-09-21): the
        server restarted its ASR stream 0.3 s after barge-in and "Stop,
        please answer in one sentence." arrived as "One sentence.". So this
        path sends `keep_asr=True` and, for the same reason, keeps the
        locally accumulated mid-utterance segments instead of dropping them.
        """
        if self._llm_turn_task is not None and not self._llm_turn_task.done():
            self._llm_turn_task.cancel()
            try:
                await self._llm_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        playback_position_ms = 0
        position = getattr(self.audio, "playback_position_ms", None)
        if callable(position):
            try:
                playback_position_ms = int(
                    position(getattr(self, "_active_response_id", None))
                )
            except Exception:
                logger.exception("failed to read playback position during barge-in")
        try:
            await self.audio.stop_playback()
        except Exception:
            logger.exception("stop_playback failed during barge-in")
        try:
            await asyncio.wait_for(self.slv.abort(keep_asr=True), timeout=0.5)
            logger.info("SLV abort sent during barge-in (keep_asr=1)")
        except asyncio.TimeoutError:
            logger.warning("SLV abort timed out during barge-in")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("SLV abort failed during barge-in")
        truncate = getattr(self.slv, "truncate_active_response", None)
        if callable(truncate):
            try:
                await asyncio.wait_for(
                    truncate(playback_position_ms), timeout=0.5
                )
            except asyncio.TimeoutError:
                logger.warning("SLV conversation truncate timed out during barge-in")
            except Exception:
                logger.exception("SLV conversation truncate failed during barge-in")
        self._eos_sent_this_turn = False
        self._cancel_asr_watchdog()
        self._first_tts_seen = False
        # NOT cleared here: the pending buffer holds mid-utterance segments of
        # the barge-in utterance itself (the RK Qwen3-ASR backend emits an
        # internal final after ~400 ms of silence, well before our asr_eos).
        # Dropping them loses the head of what the user just said — the very
        # thing the barge-in is meant to capture. It is still cleared on
        # dispatch, sleep, wake-command completion and session close, so it
        # cannot glue onto an unrelated next turn.

    def _arm_thinking_watchdog(self) -> None:
        """Re-arm the THINKING-state watchdog.

        Idempotent: a prior task is cancelled. Called every time we
        transition INTO THINKING (real asr_final, dashboard text inject).
        """
        if self._thinking_watchdog_task is not None and not self._thinking_watchdog_task.done():
            self._thinking_watchdog_task.cancel()
        self._thinking_watchdog_task = asyncio.create_task(
            self._thinking_watchdog(), name="thinking-watchdog",
        )

    def _cancel_thinking_watchdog(self) -> None:
        """Cancel the THINKING watchdog if armed. Idempotent."""
        if self._thinking_watchdog_task is not None and not self._thinking_watchdog_task.done():
            self._thinking_watchdog_task.cancel()
        self._thinking_watchdog_task = None

    async def _thinking_watchdog(self) -> None:
        """Force state back to IDLE if THINKING never resolves into
        SPEAKING (no ``tts_started`` event from SLV).

        Observed failure on jetson-qwen3asr-matcha-nx: after a successful
        ASR → LLM → tool_call → text-stream → flush_tts cycle, the SLV
        server occasionally fails to start TTS playback (no further
        events from the WS even though it stays connected). Without this
        watchdog the FSM stays THINKING forever and every subsequent
        user utterance is ignored. On fire we reconnect SLV (which gives
        a fresh worker) and reset state to IDLE so the next turn can
        proceed.
        """
        timeout = float(getattr(self.config, "thinking_timeout_s", 20.0))
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        if getattr(self, "_state", ConvState.IDLE) != ConvState.THINKING:
            return  # state moved on naturally; nothing to do
        logger.warning(
            "thinking watchdog fired (no tts_started in %.1fs after asr_final); "
            "resetting state to IDLE and reconnecting SLV",
            timeout,
        )
        # Drop any LLM turn task that's still believed to be in flight.
        if self._llm_turn_task is not None and not self._llm_turn_task.done():
            self._llm_turn_task.cancel()
            try:
                await self._llm_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        # Tell the server to tear down the wedged session BEFORE we drop the
        # WS. The thinking-watchdog fires precisely when SLV's /v2v session is
        # half-open (no tts_started); reconnecting without an in-band teardown
        # hint leaves the old server session holding its slot (soft-deadlock:
        # the next wake's fresh WS is 4429-rejected because the limiter slot is
        # still busy). abort() is the same in-band frame barge-in uses to nudge
        # the server to release the current session. Best-effort + guarded for
        # the no-SLV / not-connected case (unit tests, pre-connect watchdog).
        slv = getattr(self, "slv", None)
        if slv is not None:
            try:
                await asyncio.wait_for(slv.abort(), timeout=0.5)
                logger.info("SLV abort sent before thinking-watchdog reconnect")
            except asyncio.TimeoutError:
                logger.warning("SLV abort timed out during thinking-watchdog recovery")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("SLV abort failed during thinking-watchdog recovery")
        # Force a fresh WS — the server-side TTS pipeline is likely
        # wedged on the current session. Best-effort.
        try:
            await asyncio.wait_for(self.slv.reconnect(), timeout=3.0)
            self._slv_reconnect_count = getattr(self, "_slv_reconnect_count", 0) + 1
            await self._broadcast(
                "on_slv_reconnect", {"count": self._slv_reconnect_count}
            )
            await self._readvertise_after_reconnect()  # (#38)
        except Exception:
            logger.exception("SLV reconnect failed during thinking-watchdog recovery")
        self._first_tts_seen = False
        # Symmetric latch reset with the SLVError / dispatch-reconnect
        # paths. Without clearing this, the next utterance could
        # short-circuit ``send_asr_eos_once`` and never receive a final.
        self._eos_sent_this_turn = False
        self._set_state(ConvState.IDLE)
        self._reset_sleep_timer()

    async def _asr_final_watchdog(self) -> None:
        """Force state back to IDLE if asr_final never arrives after asr_eos.

        SLV's always_on pipeline filters empty-text finals server-side, so
        an EOS triggered by mic noise produces no client-visible final and
        the FSM would stay in THINKING forever. Real finals cancel this
        task before it fires.
        """
        timeout = float(getattr(self.config, "asr_final_timeout_s", 3.0))
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        # Only act if (a) we still believe an EOS is outstanding and
        # (b) the FSM hasn't moved on (e.g. via SLVError, a late final
        # that arrived just before us, or a barge-in).
        if not getattr(self, "_eos_sent_this_turn", False):
            return
        if getattr(self, "_state", ConvState.IDLE) != ConvState.THINKING:
            logger.info(
                "asr_final watchdog fired after state moved to %s; "
                "clearing stale EOS latch",
                getattr(self, "_state", ConvState.IDLE).value,
            )
            self._eos_sent_this_turn = False
            return
        # The turn is being abandoned: any mid-utterance segments accumulated
        # before the EOS belong to it and must not be glued onto the NEXT
        # utterance's text (the post-EOS final carries the full text when the
        # backend actually emits one; this path only runs when it didn't).
        if getattr(self, "_pending_asr_utterance_text", ""):
            logger.warning(
                "dropping %d accumulated ASR segment char(s) with the abandoned "
                "turn (no post-EOS final within %.1fs)",
                len(self._pending_asr_utterance_text), timeout,
            )
        self._clear_pending_asr_utterance()
        logger.warning(
            "asr_final not received within %.1fs after asr_eos; "
            "assuming empty/dropped final — resetting to IDLE", timeout,
        )
        self._eos_sent_this_turn = False
        if self._wake_command_single_turn_enabled():
            if not await self._speak_wake_command_no_final_feedback():
                await self._return_to_sleep_after_command_turn()
                await self._broadcast("on_assistant_done")
        else:
            self._set_state(ConvState.IDLE)

    async def _update_vad(self, chunk: bytes, chunk_ms: int) -> None:
        """Client-side speech-end detector. Sends asr_eos to SLV after a
        period of silence following speech, so Paraformer has accumulated
        enough audio to produce a non-empty final."""
        assert self._client_vad is not None
        # Gate while SLEEPING: don't update counters or fire eos.
        if getattr(self, "_state", ConvState.IDLE) == ConvState.SLEEPING:
            return
        # PTT mode with explicit-eos-only: skip VAD silence accumulation
        # entirely so the only EOS path is /api/control/ptt/end.
        cfg = getattr(self, "config", None)
        if (
            cfg is not None
            and getattr(cfg, "pipeline_mode", "always_on") == "push_to_talk"
            and getattr(cfg, "push_to_talk_no_vad_silence", True)
        ):
            return
        is_speech = self._client_vad.is_speech(chunk)
        if self._vad_state == "idle":
            if is_speech:
                self._vad_speech_ms += chunk_ms
                if self._vad_speech_ms >= self.config.client_vad_speech_min_ms:
                    self._vad_state = "speech"
                    self._vad_silence_ms = 0
                    self._vad_eos_sent = False
                    self._vad_echo_segment = False
                    logger.info("client VAD: speech started")
                    if (
                        getattr(self, "_state", ConvState.IDLE) == ConvState.THINKING
                        and getattr(self, "_eos_sent_this_turn", False)
                    ):
                        logger.info(
                            "client VAD: new speech while waiting for asr_final; "
                            "starting a fresh ASR turn"
                        )
                        self._eos_sent_this_turn = False
                        self._cancel_asr_watchdog()
                    # If TTS is currently playing *and the FSM agrees that the
                    # assistant is speaking*, this is a barge-in.  Local audio
                    # can remain buffered briefly after TTSDone/state recovery;
                    # treating ``audio.is_playing`` alone as authoritative made
                    # XVF3800 echo/noise abort a completed turn from IDLE or a
                    # pending ASR turn from THINKING.  The ASRPartial path below
                    # already uses the same SPEAKING/BARGED_IN state guard.
                    # Transition straight to BARGED_IN so the dispatch
                    # loop's later ASRPartial check (which races SLV's
                    # ~610ms first-decode latency) doesn't miss the
                    # transition. mic_pump fires first because client
                    # VAD detects speech the moment we send chunks.
                    if (
                        self.audio.is_playing
                        and self._state == ConvState.SPEAKING
                        and self._barge_in_enabled()
                        and self._barge_in_delay_elapsed()[0]
                    ):
                        logger.info("BARGE-IN fired (VAD-driven, state=%s)", self._state.value)
                        self._set_state(ConvState.BARGED_IN)
                        await self._interrupt_current_turn_for_barge_in()
                    elif (
                        self.audio.is_playing
                        and self._state == ConvState.SPEAKING
                        and self._barge_in_enabled()
                    ):
                        _, elapsed_ms, minimum_ms = self._barge_in_delay_elapsed()
                        logger.info(
                            "VAD barge-in skipped %.0fms after playback start "
                            "(need >=%dms; likely echo)",
                            elapsed_ms,
                            minimum_ms,
                        )
                        self._vad_echo_segment = True
                    elif self.audio.is_playing:
                        logger.info(
                            "client VAD speech while playback buffered in state=%s; "
                            "not treating as immediate barge-in",
                            self._state.value,
                        )
                    else:
                        self._set_state(ConvState.LISTENING)
            else:
                self._vad_speech_ms = 0
        elif self._vad_state == "speech":
            if not is_speech:
                self._vad_silence_ms += chunk_ms
                if self._vad_silence_ms >= self.config.client_vad_silence_ms:
                    if (
                        getattr(self, "_vad_echo_segment", False)
                        and getattr(self, "_state", ConvState.IDLE) == ConvState.SPEAKING
                    ):
                        # The segment was rejected as a barge-in (echo guard)
                        # and the assistant is still talking. Ending it used to
                        # send asr_eos and flip SPEAKING → THINKING mid-reply,
                        # which then disabled barge-in for the rest of the reply
                        # (both barge-in paths require SPEAKING) and let the
                        # 3 s asr_final watchdog force IDLE while audio still
                        # played. Measured on rk3588 (2026-09-21): echo VAD at
                        # +230 ms, speech end at +835 ms, speaking → thinking,
                        # and an interruption 1.2 s later was ignored.
                        logger.info(
                            "client VAD: echo segment ended during playback; "
                            "not ending a user turn"
                        )
                        self._vad_eos_sent = True
                    elif not self._vad_eos_sent:
                        import time as _t
                        drove_eos = bool(getattr(self.config, "client_vad_drive_eos", False))
                        if drove_eos:
                            logger.info("client VAD: speech ended -> asr_eos")
                            await self._broadcast(
                                "on_user_speech_end_client",
                                {"ts": int(_t.time() * 1000), "drove_eos": True},
                            )
                            # Dedup: PTT/end may also try to send. Only
                            # one asr_eos per turn — race protection.
                            await self.send_asr_eos_once()
                        else:
                            logger.debug(
                                "client VAD: speech ended (paraformer-endpoint mode, no asr_eos)"
                            )
                            await self._broadcast(
                                "on_user_speech_end_client",
                                {"ts": int(_t.time() * 1000), "drove_eos": False},
                            )
                        self._set_state(ConvState.THINKING)
                        # Arm the thinking-watchdog so a wedged SLV TTS
                        # can't strand the FSM. (Symmetric with the
                        # server-VAD path at _dispatch_one.)
                        self._arm_thinking_watchdog()
                        self._vad_eos_sent = True
                    self._vad_state = "idle"
                    self._vad_speech_ms = 0
                    self._vad_silence_ms = 0
                    self._client_vad.reset()
            else:
                self._vad_silence_ms = 0

    async def _slv_dispatch(self) -> None:
        """Drive SLV events into the FSM. Auto-reconnects whenever the
        events() iterator returns naturally — SLV closes the WS after
        every asr_eos round (even in multi_utterance mode), and an empty
        / dropped final means no ASRFinal session_complete=True ever
        fires the in-band reconnect at line ~768. Without this outer
        loop the dispatch task silently dies after one bad turn and
        every subsequent asr_eos hits a closed WS ("send_json: WS closed
        mid-send" floods the log) — which also kills barge-in because
        TTS never reaches the speaker again.
        """
        backoff = 0.5
        while not getattr(self.slv, "_closed", False):
            try:
                async for evt in self.slv.events():
                    try:
                        await self._dispatch_one(evt)
                    except Exception:
                        logger.exception("dispatch error on %r", evt)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("slv events iterator crashed")
            # events() returned: reader died (SLV closed the WS or net
            # blip). Reconnect and resume — unless we're shutting down.
            if getattr(self.slv, "_closed", False):
                return
            # Break the eviction↔reconnect self-excitation storm. If a
            # concurrent reconnect (e.g. wake()'s) already installed a fresh
            # healthy WS, OR this close was a 1012 admission eviction (our own
            # newer session superseded us / a reconnect is mid-flight), do NOT
            # open yet another WS — that evicts the live holder and re-triggers
            # the loop (each cycle tears the WS down mid-utterance → ASR worker
            # cancel-timeout → forced restart → "卡住"). Just resume consuming
            # events on the live connection. Only a genuine death (no healthy
            # replacement, non-eviction close) falls through to reconnect.
            # Always `continue`, never return (except client shutdown above) —
            # else the dispatch task goes deaf.
            if self.slv.is_healthy():
                continue
            _code = self.slv.last_close_code()
            _reason = (self.slv.last_close_reason() or "").lower()
            _evicted = _code == 1012 and ("superseded" in _reason or "evicted" in _reason)
            if self.slv.is_reconnecting() or _evicted:
                # A reconnect is finishing (or we were superseded by it). Give
                # it a beat to install the new reader, then loop back to
                # events() on the live WS.
                await asyncio.sleep(0.3)
                continue
            try:
                logger.info("slv dispatch: reader exited (code=%s), reconnecting...", _code)
                await asyncio.wait_for(self.slv.reconnect(), timeout=5.0)
                self._slv_reconnect_count = getattr(self, "_slv_reconnect_count", 0) + 1
                self._first_tts_seen = False
                self._eos_sent_this_turn = False
                self._cancel_asr_watchdog()
                # Transport died mid-utterance: half-accumulated segments
                # belong to a session that no longer exists.
                self._clear_pending_asr_utterance()
                if getattr(self, "_state", ConvState.IDLE) in {
                    ConvState.THINKING,
                    ConvState.BARGED_IN,
                }:
                    self._set_state(ConvState.IDLE)
                try:
                    await self._broadcast(
                        "on_slv_reconnect", {"count": self._slv_reconnect_count}
                    )
                except Exception:
                    pass
                await self._readvertise_after_reconnect()  # (#38)
                backoff = 0.5
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("auto-reconnect failed, sleeping %.1fs", backoff)
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                backoff = min(backoff * 2, 5.0)

    def _on_reply_text_sent(self) -> None:
        """The new reply's first text is going out: from here on, TTS events
        belong to it, so stop dropping them and make playback audible."""
        if not getattr(self, "_awaiting_new_reply", False):
            return
        self._awaiting_new_reply = False
        arm = getattr(self.audio, "arm_for_next_turn", None)
        if callable(arm):
            arm()

    async def _dispatch_one(self, evt) -> None:  # noqa: ANN001
        # A superseded reply's tail (audio already in flight, its tts_done,
        # the V2 equivalents) must not play, flip the FSM to SPEAKING, or
        # start a drain that would complete the NEW turn.
        if getattr(self, "_awaiting_new_reply", False) and isinstance(
            evt,
            (
                TTSAudio, TTSStarted, TTSSentenceDone, TTSDone,
                ResponseOutputAudioDone, ResponseDone,
                AssistantTranscriptDelta, AssistantTranscriptDone,
            ),
        ):
            return
        # #3: a send-path connect() (mic pump reviving a dead WS) opens a fresh
        # session without advertising tools, and events() keeps streaming on it
        # without the dispatch guard ever firing. Re-advertise the moment we see
        # an event from an un-advertised generation, before the turn needs tools.
        await self._readvertise_if_session_advanced()

        # ── server-loop remote tool call (#37 Phase 2-product) ──────
        # Handle before any state gate: the server-side LLM loop is
        # blocked waiting for our CLIENT_TOOL_RESULT, so we must always
        # answer (even while SLEEPING) or the server turn stalls until
        # timeout. Only emitted in server-loop mode; the legacy
        # client-loop path never sees this event.
        if isinstance(evt, ServerToolCall):
            # #F4: run the tool handler in a background task so the dispatch
            # loop keeps draining events while the tool executes, instead of
            # blocking the consumer for the tool's duration. The server issues
            # tool calls one at a time (it awaits each CLIENT_TOOL_RESULT before
            # the next), so these tasks don't overlap in practice.
            self._spawn_tool_task(evt)
            return

        # ── pipeline_mode SLEEPING gate ─────────────────────────────
        # SLEEPING means the user explicitly silenced the agent (or it
        # auto-slept). The mic pump already drops audio, but events
        # already queued by SLV (partial / endpoint / final) may still
        # arrive after the sleep call. Honour the gate at the dispatch
        # boundary so a late asr_final can't wake the agent and trigger
        # a new LLM turn.
        if getattr(self, "_state", ConvState.IDLE) == ConvState.SLEEPING:
            if isinstance(evt, (ASRPartial, ASREndpoint)):
                return
            if isinstance(evt, ASRFinal):
                # SLV closed its WS on its side after asr_eos — must
                # still reconnect or the next user turn fails silently
                # with "WS closed mid-send". But DON'T broadcast the
                # utterance or spawn an LLM turn.
                if evt.session_complete:
                    try:
                        await self.slv.reconnect()
                        self._slv_reconnect_count = getattr(self, "_slv_reconnect_count", 0) + 1
                        self._first_tts_seen = False
                        await self._broadcast(
                            "on_slv_reconnect", {"count": self._slv_reconnect_count}
                        )
                        await self._readvertise_after_reconnect()  # (#38)
                    except Exception:
                        logger.exception("SLV reconnect failed (sleeping)")
                return
            # TTS frames during SLEEPING shouldn't normally arrive (we
            # aborted on sleep), but if they do, ignore — playback is
            # stopped anyway. Fall through for SLVError so we still log.

        if isinstance(evt, ASRPartial):
            # SLV's silero VAD can fire spurious empty endpoints from breath/
            # ambient noise; ignore empty partials so we don't trigger
            # bogus barge-ins or noise the dashboard.
            partial_text = (evt.text or "").strip()
            if not partial_text:
                return
            # silero-primary stall fallback: a real partial means silero is
            # actively transcribing — (re)arm the no-partial timer. Scoped to
            # the command-listening phase (IDLE/LISTENING) so a barge-in
            # partial during the assistant's reply doesn't arm it. No-op when
            # vad_stall_eos_ms<=0.
            if self._state in (ConvState.IDLE, ConvState.LISTENING):
                self._arm_stall_watchdog()
            # Barge-in: user spoke (real text) while we were SPEAKING.
            #
            # We gate on FSM state, NOT audio.is_playing alone. Reason: a
            # TTSDone event flips state SPEAKING→IDLE the moment SLV says
            # "no more frames", but audio_io is still draining buffered
            # PCM through the speaker for another 100-500ms. During that
            # window, a fresh ASRPartial from a NEW user utterance (a
            # legitimate new turn, NOT a barge-in over a still-playing
            # reply) would fire barge-in, abort SLV, and cancel a brand
            # new LLM turn that hadn't even started — wedging the agent
            # into a loop of fake barge-ins on garbled partials. By
            # requiring SPEAKING/BARGED_IN here we treat a partial during
            # IDLE/THINKING as the start of a normal utterance.
            #
            # ALSO: require a MINIMUM partial length AND a minimum delay
            # since TTS began. The reSpeaker XVF3800's hardware AEC isn't
            # perfect — the speaker's first 200-500ms of output bleeds
            # into the open mic, server silero triggers, and the FIRST
            # partial we see is the agent's own "好的。" coming back. If
            # we honour that as barge-in we cancel our own turn before
            # the user could possibly speak. Min length (2 chars) +
            # min delay since TTS started (500ms) suppresses the echo
            # blip; a real barge-in of "Hey Jarvis ..." easily meets both.
            barge_min_chars = int(getattr(self.config, "barge_in_min_chars", 2))
            delay_ok, elapsed_ms, barge_min_speaking_ms = (
                self._barge_in_delay_elapsed()
            )
            if len(partial_text) < barge_min_chars:
                logger.debug(
                    "barge-in skipped: partial too short (len=%d < %d): %r",
                    len(partial_text), barge_min_chars, partial_text,
                )
                return
            if not delay_ok:
                logger.debug(
                    "barge-in skipped: elapsed only %.0fms since TTS start (need >=%dms) — "
                    "treating partial %r as echo",
                    elapsed_ms, barge_min_speaking_ms, partial_text,
                )
                return
            if (
                self._state in (ConvState.SPEAKING, ConvState.BARGED_IN)
                and self.audio.is_playing
                and self._barge_in_enabled()
            ):
                logger.info(
                    "BARGE-IN fired (state=%s, partial=%r)",
                    self._state.value, evt.text[:40]
                )
                if self._state == ConvState.SPEAKING:
                    self._set_state(ConvState.BARGED_IN)
                # Cancel any in-flight LLM turn FIRST: otherwise it keeps
                # streaming tokens to SLV which immediately restarts TTS
                # and undoes our barge-in stop_playback below.
                if self._llm_turn_task is not None and not self._llm_turn_task.done():
                    self._llm_turn_task.cancel()
                    try:
                        await self._llm_turn_task
                    except (asyncio.CancelledError, Exception):
                        pass
                await self._interrupt_current_turn_for_barge_in()
            elif self._state in (ConvState.SPEAKING, ConvState.BARGED_IN) and self.audio.is_playing:
                # barge-in disabled (e.g. translation/transcription apps): keep
                # playing the current reply, just surface the partial below.
                logger.debug(
                    "barge-in suppressed (disabled): partial %r during %s",
                    partial_text, self._state.value,
                )
            # Surface the partial to both plugins and the app subclass hook.
            await self._broadcast("on_user_partial", evt.text)
            await self.on_user_partial(evt.text, getattr(evt, "language", None))
            return

        if isinstance(evt, InputAudioSpeechStarted):
            await self._broadcast("on_user_speech_start")
            # Canonical server VAD is an earlier and stronger barge-in signal
            # than waiting for an ASR partial. Stop local playback immediately;
            # the Gateway is responsible for cancelling the remote response.
            if (
                self._state == ConvState.SPEAKING
                and self.audio.is_playing
                and self._barge_in_enabled()
                and self._barge_in_delay_elapsed()[0]
            ):
                self._set_state(ConvState.BARGED_IN)
                if self._llm_turn_task is not None and not self._llm_turn_task.done():
                    self._llm_turn_task.cancel()
                    try:
                        await self._llm_turn_task
                    except (asyncio.CancelledError, Exception):
                        pass
                await self._interrupt_current_turn_for_barge_in()
            elif (
                self._state == ConvState.SPEAKING
                and self.audio.is_playing
                and self._barge_in_enabled()
            ):
                _, elapsed_ms, minimum_ms = self._barge_in_delay_elapsed()
                logger.info(
                    "server VAD barge-in skipped %.0fms after playback start "
                    "(need >=%dms; likely echo)",
                    elapsed_ms,
                    minimum_ms,
                )
            return

        if isinstance(evt, InputAudioSpeechStopped):
            await self._broadcast("on_user_speech_end")
            return

        if isinstance(evt, InputAudioCommitted):
            await self._broadcast(
                "on_user_audio_committed", {"item_id": evt.item_id}
            )
            return

        if isinstance(evt, ConversationItemTruncated):
            await self._broadcast(
                "on_conversation_item_truncated",
                {"item_id": evt.item_id, "audio_end_ms": evt.audio_end_ms},
            )
            return

        if isinstance(evt, ConversationResetDone):
            await self._broadcast("on_conversation_reset")
            return

        if isinstance(evt, ASREndpoint):
            await self._broadcast("on_user_speech_start")
            return

        if isinstance(evt, ASRFinal):
            # A real final arrived — disarm the watchdogs so they don't
            # later reset state / force a stale EOS out from under whatever
            # dispatch we're about to run.
            self._cancel_asr_watchdog()
            self._cancel_stall_watchdog()
            # Snapshot whether the agent had just sent client-side EOS —
            # if a low-signal/empty final follows, that final IS the
            # response to our EOS and there's no LLM turn in flight to
            # protect (race #2: stuck in THINKING).
            _had_pending_eos = self._eos_sent_this_turn
            # Clear the per-turn EOS dedupe flag for ALL final paths
            # (duplicate-of-streamed, empty, and real). Previously it was
            # only reset in the non-empty branch below, so a duplicate or
            # empty final would leave the flag set and the next turn's
            # send_asr_eos_once would early-return → SLV never receives
            # EOS → no final → state stuck THINKING forever (worse than
            # the empty-final bug the watchdog was designed to catch,
            # because the watchdog never even arms).
            self._eos_sent_this_turn = False
            # A close-out final whose text duplicates what was already
            # streamed (server: ``duplicate_of_streamed = final_text ==
            # last_streamed_final``, server/main.py multi_utterance close) carries
            # NO new content — joining it would append the same segment twice once
            # mid-utterance accumulation is in play.
            _final_text = "" if evt.duplicate_of_streamed else (evt.text or "")
            if evt.duplicate_of_streamed and not getattr(
                self, "_pending_asr_utterance_text", ""
            ):
                # A duplicate final means there is no new utterance to route.
                # If the duplicate is the only final after client-driven EOS,
                # cancelling the watchdog and returning here would strand the
                # FSM in THINKING forever.
                if getattr(self, "_state", ConvState.IDLE) == ConvState.THINKING:
                    logger.info(
                        "duplicate asr_final ignored while THINKING; resetting to IDLE"
                    )
                    self._set_state(ConvState.IDLE)
                    self._reset_sleep_timer()
                return
            # SLV closes the WS after every asr_eos-triggered final
            # (session_complete=True), regardless of whether the final
            # text is empty. Reconnect FIRST, then decide whether the
            # text was worth an LLM turn. If we skipped reconnect on
            # empty finals, the next user utterance would silently fail
            # with "send_json: WS closed mid-send, dropping asr_eos".
            if evt.session_complete:
                try:
                    await self.slv.reconnect()
                    logger.debug("SLV reconnected after session_complete final")
                    self._slv_reconnect_count = getattr(self, "_slv_reconnect_count", 0) + 1
                    # Reset first-TTS-frame flag so the next turn re-emits the
                    # THINKING→SPEAKING transition cleanly after reconnect.
                    self._first_tts_seen = False
                    await self._broadcast(
                        "on_slv_reconnect", {"count": self._slv_reconnect_count}
                    )
                    await self._readvertise_after_reconnect()  # (#38)
                except Exception:
                    logger.exception("SLV reconnect failed")
            # Drop empty finals — clawd's proven pattern. SLV's server-side
            # VAD or a too-short utterance produces empty text. Treating
            # those as real utterances would call the LLM with no input.
            # Also drop *low-signal* finals (1 visible char or pure
            # interjection / filler): they're almost always ASR noise on
            # an open mic, and feeding them to the LLM is the canonical
            # trigger for an in-context echo loop — the model emits a
            # short "safe" fallback, that fallback enters history, and
            # after 3-4 such turns the small model latches onto the
            # pattern and replies with the same canned line forever.
            stripped_for_signal = _strip_for_signal(evt.text or "")
            _is_low_signal = (
                not (evt.text or "").strip()
                or len(stripped_for_signal) <= 1
                or stripped_for_signal in _INTERJECTIONS
            )
            # A low-signal final that ANSWERS our client-side EOS still
            # closes the utterance: when earlier backend-endpointed
            # segments were accumulated, fall through and dispatch them
            # (the RK backend commonly returns empty text for the
            # post-EOS final when it already finalized the audio on its
            # own internal endpoint).
            if _is_low_signal and not (
                _had_pending_eos
                and getattr(self, "_pending_asr_utterance_text", "")
            ):
                # Debug only: an open mic emits empty/low-signal finals every
                # 1-4s while idle — at info this floods the log. The state-
                # transition lines (kept at info) still show any FSM effect.
                logger.debug(
                    "low-signal asr_final ignored (text=%r, signal=%r, state=%s)",
                    (evt.text or "")[:30], stripped_for_signal,
                    getattr(self, "_state", ConvState.IDLE).name,
                )
                # Clear the discard latch so a later legitimate turn
                # (including dashboard-typed text with no ASRFinal) gets
                # audible TTS — a prior barge-in / abort may have armed it.
                try:
                    arm = getattr(self.audio, "arm_for_next_turn", None)
                    if callable(arm):
                        arm()
                except Exception:  # pragma: no cover - defensive
                    pass
                # State transitions — be conservative. A noise / silence
                # asr_final must NOT cancel an in-flight LLM/TTS turn
                # belonging to a PREVIOUS real utterance. Symptom: with
                # always-on mic, server VAD frequently emits empty
                # asr_finals while the model is mid-tool-call; forcing
                # state back to IDLE here was cancelling the runner, which
                # then re-fired text-only completions, looped tool
                # invocations, and ultimately stranded the agent in a
                # broken speaking↔idle ping-pong that needed a restart.
                #
                # State transitions on low-signal final:
                #   LISTENING — agent was waiting for THIS final and it
                #     turned out to be noise; recover to IDLE.
                #   BARGED_IN — barge-in fired (cancelled TTS+LLM)
                #     expecting the user's actual command; got noise
                #     instead. Treat as cancelled barge-in: clear back
                #     to IDLE so the next real utterance flows cleanly.
                #     Without this the agent stays in BARGED_IN forever
                #     when the barge-in audio produces only short noise
                #     finals ('Yeah.', '头', etc.).
                #   THINKING / SPEAKING — a noise final mid-turn must
                #     NOT cancel the in-flight LLM/TTS belonging to a
                #     PREVIOUS real utterance.
                #   IDLE / SLEEPING — already terminal; no transition.
                cur_state = getattr(self, "_state", ConvState.IDLE)
                if (
                    cur_state in (ConvState.LISTENING, ConvState.BARGED_IN)
                    and not getattr(self, "_pending_asr_utterance_text", "")
                ):
                    self._set_state(ConvState.IDLE)
                    # noise final → don't extend the hot-mic window (see
                    # _ensure_sleep_timer): ambient room speech must not keep the
                    # device awake forever.
                    self._ensure_sleep_timer()
                elif cur_state == ConvState.THINKING and _had_pending_eos:
                    # Race #2: we sent client-EOS expecting a real
                    # final but SLV returned an empty/low-signal one
                    # (server VAD coalesced silence). There is no
                    # in-flight LLM turn to protect — sit in THINKING
                    # forever otherwise. Force back to IDLE so the
                    # next utterance can flow.
                    logger.info(
                        "empty asr_final after pending EOS while THINKING; "
                        "resetting to IDLE (race #2)"
                    )
                    if self._wake_command_single_turn_enabled():
                        if not await self._speak_wake_command_no_final_feedback():
                            await self._return_to_sleep_after_command_turn()
                            await self._broadcast("on_assistant_done")
                    else:
                        self._set_state(ConvState.IDLE)
                        self._ensure_sleep_timer()  # noise final must not extend
                elif cur_state in (ConvState.THINKING, ConvState.SPEAKING):
                    logger.debug(
                        "low-signal final arrived during %s; FSM left alone "
                        "(in-flight turn keeps running)",
                        cur_state.name,
                    )
                return
            logger.info(
                "asr_final received: %r (language=%r)", evt.text, evt.language
            )
            # Mid-utterance segment? With client-owned endpointing
            # (client_vad_drive_eos) the utterance boundary is the agent's
            # asr_eos, NOT a server final: the RK backend endpoint
            # (400 ms of silence) emits finals while the user is merely
            # pausing mid-sentence. Accumulate such segments and wait for
            # the post-EOS final instead of splitting one spoken sentence
            # into two LLM turns.
            if (
                bool(getattr(self.config, "client_vad_drive_eos", False))
                and not _had_pending_eos
                and getattr(self, "_state", ConvState.IDLE)
                in (ConvState.LISTENING, ConvState.BARGED_IN)
            ):
                self._accumulate_asr_segment(_final_text)
                self._cancel_wake_command_timeout()
                logger.info(
                    "asr_final segment accumulated (pending=%r)",
                    getattr(self, "_pending_asr_utterance_text", ""),
                )
                await self._broadcast("on_user_utterance", evt.text)
                return
            # Utterance end (post-EOS final, or server-VAD mode where
            # every final ends a turn): join any accumulated segments.
            full_text = _join_asr_segment_text(
                self._take_pending_asr_utterance(), _final_text
            )
            if full_text != (evt.text or ""):
                logger.info("asr utterance assembled from segments: %r", full_text)
            self._last_user_utterance_text = full_text
            self._cancel_wake_command_timeout()
            # A new user turn supersedes the previous reply. The LLM task is
            # cancelled further down, but a reply that has already been fully
            # synthesized keeps playing from the local buffer, and the new
            # reply queues behind it. This happens whenever the FSM is not in
            # SPEAKING while that audio plays (e.g. a VAD segment that started
            # just before playback ends mid-reply and moves SPEAKING ->
            # THINKING), because both barge-in paths require SPEAKING.
            # Measured on rk3588 (2026-09-21): five user turns in 15 s while
            # one 20-sentence reply kept playing; "stop" never interrupted it.
            # Done before re-arming playback below: stop_playback latches
            # discard, arm_for_next_turn clears it for the new reply.
            superseded = False
            # Client-loop only: in server-loop the server owns the response
            # (interrupt_response=True) and no reply text is sent from here,
            # so the deferred re-arm below would never fire.
            if (
                getattr(self.audio, "is_playing", False)
                and self._barge_in_enabled()
                and not self._server_loop_enabled()
            ):
                logger.info(
                    "new user turn while the previous reply is still playing; "
                    "interrupting it"
                )
                # Its drain task would otherwise wake on stop_playback and
                # complete the assistant turn — i.e. set the NEW turn to IDLE.
                drain_task = getattr(self, "_playback_drain_task", None)
                if drain_task is not None and not drain_task.done():
                    drain_task.cancel()
                await self._interrupt_current_turn_for_barge_in()
                # Its ResponseDone is dropped as stale below; do not keep
                # pointing at the superseded response.
                self._active_response_id = None
                # Keep the discard latch until the new reply's text goes out
                # (_on_reply_text_sent): audio still in flight from the old
                # reply would otherwise play right after re-arming.
                self._awaiting_new_reply = True
                superseded = True
            # Re-enable speaker playback for the next turn. stop_playback
            # latched discard=True on the prior barge-in / sleep so SLV's
            # tail-end TTS didn't keep playing; clear that now so the new
            # turn's TTS is actually audible.
            if not superseded:
                self._awaiting_new_reply = False
                try:
                    arm = getattr(self.audio, "arm_for_next_turn", None)
                    if callable(arm):
                        arm()
                except Exception:  # pragma: no cover - defensive
                    pass
            # New utterance round about to begin — clear client VAD state so
            # the next speech_start fires fresh. (getattr-guarded so tests
            # that build BaseApp via __new__ don't have to set every field.)
            self._vad_state = "idle"
            self._vad_speech_ms = 0
            self._vad_silence_ms = 0
            self._vad_eos_sent = False
            # Allow the NEXT turn to send asr_eos again.
            self._eos_sent_this_turn = False
            _cv = getattr(self, "_client_vad", None)
            if _cv is not None:
                try:
                    _cv.reset()
                except Exception:  # pragma: no cover
                    pass
            await self._broadcast("on_user_utterance", full_text)
            # (reconnect already happened above, before the empty-text guard)
            # Stop-intent: user said "停下" / "stop" — cancel everything,
            # do NOT route to LLM and do NOT extend session.history (the
            # user asked for quiet, not for more conversation).
            if self._is_stop_intent(full_text):
                logger.info("stop intent matched: %r", full_text)
                if self._llm_turn_task is not None and not self._llm_turn_task.done():
                    self._llm_turn_task.cancel()
                    try:
                        await self._llm_turn_task
                    except (asyncio.CancelledError, Exception):
                        pass
                try:
                    await self.slv.abort()
                except Exception:  # pragma: no cover - best effort
                    pass
                try:
                    await self.audio.stop_playback()
                except Exception:  # pragma: no cover - best effort
                    pass
                self._set_state(ConvState.IDLE)
                self._reset_sleep_timer()
                await self._broadcast("on_user_stop_intent", full_text)
                return
            # Spawn the LLM turn as a tracked task so the dispatch loop
            # stays free to handle queued TTSAudio (playback) and
            # ASRPartial (barge-in) while the model streams.
            if self._llm_turn_task is not None and not self._llm_turn_task.done():
                self._llm_turn_task.cancel()
                try:
                    await self._llm_turn_task
                except (asyncio.CancelledError, Exception):
                    pass
            # Ensure THINKING fires on the server-VAD path (where
            # _update_vad never runs and no client-side transition has
            # set it). Idempotent for client-VAD path which already set
            # THINKING in _update_vad.
            self._set_state(ConvState.THINKING)
            # Arm the thinking watchdog so a wedged SLV TTS pipeline
            # can't strand the FSM here forever (see _thinking_watchdog
            # docstring). Cancelled by the first tts_started/tts_done
            # event or by SLVError, whichever comes first.
            self._arm_thinking_watchdog()
            self._llm_turn_task = asyncio.create_task(
                self._run_user_utterance(full_text, evt.language),
                name="llm-turn",
            )
            return

        if isinstance(evt, TTSStarted):
            # Real TTS started — thinking watchdog can stand down.
            self._cancel_thinking_watchdog()
            sentence = (evt.sentence or "").strip()
            now = time.monotonic()
            # Opt-in: drop a TTS sentence identical to the previous one within
            # this window (collapses double-acks in tool/re-prompt loops). 0 =
            # off (default), so the framework never silently suppresses a
            # legitimate repeat unless an app explicitly enables it.
            duplicate_window_s = float(
                getattr(self.config, "tts_drop_duplicate_window_s", 0.0) or 0.0
            )
            if (
                duplicate_window_s > 0.0
                and sentence
                and sentence == getattr(self, "_last_tts_started_sentence", "")
                and now - float(getattr(self, "_last_tts_started_ts", 0.0)) <= duplicate_window_s
            ):
                self._drop_current_tts_sentence = True
                logger.warning(
                    "dropping duplicate TTS sentence within %.1fs: %r",
                    duplicate_window_s,
                    sentence,
                )
                return
            self._drop_current_tts_sentence = False
            self._last_tts_started_sentence = sentence
            self._last_tts_started_ts = now
            await self._broadcast("on_assistant_sentence_start", evt.sentence)
            return

        if isinstance(evt, ResponseCreated):
            # A response exists but may not have produced audio yet. Keep the
            # FSM in THINKING; the first binary frame remains the authoritative
            # transition to SPEAKING.
            self._active_response_id = evt.response_id
            begin_playback = getattr(self.audio, "begin_response_playback", None)
            if callable(begin_playback):
                begin_playback(evt.response_id)
            self._cancel_thinking_watchdog()
            await self._broadcast(
                "on_response_created",
                {"response_id": evt.response_id, "response": evt.response},
            )
            return

        if isinstance(evt, TTSAudio):
            if getattr(self, "_drop_current_tts_sentence", False):
                return
            # If we're in BARGED_IN, the tail of SLV's prior-turn TTS is
            # still draining over the WS. Don't reset state to SPEAKING or
            # play the audio (audio.play() also drops it via the discard
            # latch, but skip the state flip here too).
            if self._state == ConvState.BARGED_IN:
                return
            first_frame = False
            if not self._first_tts_seen:
                self._first_tts_seen = True
                first_frame = True
                self.audio.set_output_sample_rate(evt.sample_rate)
                self._set_state(ConvState.SPEAKING)
                # Stamp the moment TTS playback actually began so the
                # barge-in gate (see ASRPartial handler) can suppress the
                # echo blip from the speaker's first 200-500ms output
                # leaking back through the open mic.
                self._speaking_since_ts = time.monotonic()
            await self._broadcast(
                "on_tts_audio_frame",
                {
                    "sample_rate": evt.sample_rate,
                    "frame_len": len(evt.pcm),
                    "first": first_frame,
                },
            )
            await self.audio.play(evt.pcm)
            return

        if isinstance(evt, TTSSentenceDone):
            if getattr(self, "_drop_current_tts_sentence", False):
                self._drop_current_tts_sentence = False
                return
            await self._broadcast("on_assistant_sentence", evt.sentence)
            return

        if isinstance(evt, ResponseOutputAudioDone):
            # Remote generation is finished, but buffered PCM may still be
            # playing locally. Do not complete the application turn here.
            mark = getattr(self.audio, "mark_playback_done", None)
            if callable(mark):
                mark()
            await self._broadcast(
                "on_response_output_audio_done",
                {"response_id": evt.response_id},
            )
            return

        if isinstance(evt, AssistantTranscriptDelta):
            await self._broadcast("on_assistant_token", evt.delta)
            return

        if isinstance(evt, AssistantTranscriptDone):
            await self._broadcast(
                "on_assistant_transcript_done",
                {
                    "response_id": evt.response_id,
                    "transcript": evt.transcript,
                },
            )
            return

        if isinstance(evt, ResponseDone):
            logger.debug(
                "ResponseDone received (response_id=%s status=%s)",
                evt.response_id,
                evt.status,
            )
            self._active_response_id = None
            self._first_tts_seen = False
            self._cancel_thinking_watchdog()
            await self._broadcast(
                "on_response_done",
                {
                    "response_id": evt.response_id,
                    "status": evt.status,
                    "response": evt.response,
                },
            )
            output = evt.response.get("output")
            output = output if isinstance(output, list) else []
            if output and all(
                isinstance(item, dict) and item.get("type") == "function_call"
                for item in output
            ):
                # Cloud Realtime providers terminate the function-call
                # response before the device returns its output. This is an
                # intermediate tool boundary, not an audible assistant turn.
                if self._state != ConvState.BARGED_IN:
                    self._set_state(ConvState.THINKING)
                return
            # Don't override BARGED_IN: the user utterance owns the next FSM
            # transition after a cancelled response.
            if self._state != ConvState.BARGED_IN:
                drain_task = getattr(self, "_playback_drain_task", None)
                if drain_task is not None and not drain_task.done():
                    drain_task.cancel()
                drain_enabled = bool(
                    getattr(self.config, "playback_drain_enabled", False)
                )
                if drain_enabled and getattr(self.audio, "is_playing", False):
                    self._playback_drain_task = asyncio.create_task(
                        self._finish_assistant_turn_after_playback(),
                        name="playback-drain",
                    )
                    return
            await self._complete_assistant_turn()
            self._first_tts_seen = False
            self._eos_sent_this_turn = False
            return

        if isinstance(evt, TTSDone):
            # Race #4: preserve and log session_complete so downstream
            # reconnect logic can branch on True (session ends, slot
            # released) vs False (turn done, slot held for continuation).
            session_complete = getattr(evt, "session_complete", True)
            logger.debug(
                "TTSDone received (session_complete=%s)", session_complete
            )
            # Reset first-frame flag so the NEXT turn re-emits SPEAKING.
            self._first_tts_seen = False
            self._cancel_thinking_watchdog()
            # Authoritative is_playing reset (audio_io stopped doing this on
            # transient empty queue to keep barge-in checks reliable).
            mark = getattr(self.audio, "mark_playback_done", None)
            if callable(mark):
                mark()
            # Don't override BARGED_IN: the user is mid-utterance and the
            # VAD silence-end / ASRFinal path will drive state forward.
            # Forcing IDLE here would also kick the auto-sleep timer in
            # push_to_talk mode while the user is still speaking.
            if self._state != ConvState.BARGED_IN:
                drain_task = getattr(self, "_playback_drain_task", None)
                if drain_task is not None and not drain_task.done():
                    drain_task.cancel()
                # Opt-in: defer turn completion until local TTS playback has
                # actually drained (so the turn doesn't end — and the arm
                # doesn't return to sleep — while audio is still playing out the
                # speaker). Default off: complete immediately as before.
                drain_enabled = bool(
                    getattr(self.config, "playback_drain_enabled", False)
                )
                if drain_enabled and getattr(self.audio, "is_playing", False):
                    self._playback_drain_task = asyncio.create_task(
                        self._finish_assistant_turn_after_playback(),
                        name="playback-drain",
                    )
                    return
            await self._complete_assistant_turn()
            # NO proactive reconnect on tts_done. SLV server v1.15+ added
            # an ASR turn wall-clock timeout that force-releases the
            # SessionLimiter slot even when Qwen3-ASR worker wedges, so
            # the worker-stuck workaround that this proactive reconnect
            # was guarding against is now handled server-side. Keeping
            # the reconnect here was actively causing 4429
            # too_many_sessions because the new WS races the
            # just-released slot. The WS stays alive across turns (per
            # SLV multi_utterance protocol where session_complete=False
            # means dialog continues). Real WS death is caught by the
            # outer dispatch loop's events()-returns / reader-exit path,
            # which reconnects on its own.
            self._first_tts_seen = False
            self._eos_sent_this_turn = False
            return

        if isinstance(evt, SLVError):
            # Transport died — any pending asr_final / thinking watchdog
            # is moot; SLVError handling below already drives state
            # back to IDLE.
            self._cancel_asr_watchdog()
            self._cancel_thinking_watchdog()
            self._clear_pending_asr_utterance()
            old_state = getattr(self, "_state", ConvState.IDLE)
            await self._broadcast(
                "on_error",
                TypedLLMError(
                    "slv_error",
                    evt.message,
                    exc_class="SLVError",
                ),
            )
            # Don't leave the FSM stuck in THINKING/SPEAKING after a transport
            # error — cancel any in-flight LLM turn.
            if self._llm_turn_task is not None and not self._llm_turn_task.done():
                self._llm_turn_task.cancel()
                try:
                    await self._llm_turn_task
                except (asyncio.CancelledError, Exception):
                    pass
            # If we were SLEEPING when the transport error fired, stay
            # SLEEPING — a transport hiccup must never wake the agent
            # (would hot-mic in wake_word mode).
            if old_state != ConvState.SLEEPING:
                self._set_state(ConvState.IDLE)
            else:
                logger.info("SLVError while SLEEPING; staying SLEEPING")
            # No proactive reconnect here — same race as TTSDone (the SLV
            # server immediately closes a fresh connection that arrives
            # while it's still tearing the previous session down).
            # ``SLVClient.send_audio`` / ``_send_json`` already auto-call
            # ``connect()`` when ``_ws is None``, and the dead-WS detection
            # in those paths nulls _ws on ``ConnectionClosed``. The next
            # mic chunk (or text send) naturally reopens the transport
            # ~100ms later — enough headroom for SLV to finalize the prior
            # session cleanly.
            return

    async def _run_user_utterance(
        self, text: str, detected_language: str | None = None
    ) -> None:
        """Wrap on_user_utterance so a crashing LLM turn doesn't kill the task silently."""
        try:
            await self.on_user_utterance(text, detected_language=detected_language)
            # Success path: tell the availability plugin so a transient
            # failure that earlier flipped us to DEGRADED gets cleared.
            # Skip if LLM is disabled (noop backend).
            if isinstance(self.llm, NoopLLM):
                return
            avail = getattr(self, "llm_availability", None)
            if avail is not None:
                try:
                    avail.report_request_success()
                except Exception:  # pragma: no cover - defensive
                    pass
        except NotImplementedError:
            logger.error("BaseApp.on_user_utterance not overridden -- text dropped")
            self._set_state(ConvState.IDLE)
        except asyncio.CancelledError:
            # Cancellation happens on barge-in / shutdown / stop-intent;
            # caller already drove the appropriate state transition.
            raise
        except LLMUnavailable as e:
            # Fail-fast path: the availability state machine already
            # decided the LLM is DOWN. Don't bother A3 retry — surface
            # to the dashboard and return to IDLE immediately.
            logger.warning("LLM unavailable, fail-fast: %s", e)
            try:
                await self._broadcast(
                    "on_error",
                    TypedLLMError(
                        "llm_unavailable",
                        f"LLM 不可用：{e}",
                        exc_class=type(e).__name__,
                    ),
                )
            except Exception:
                pass
            self._set_state(ConvState.IDLE)
            try:
                self._reset_sleep_timer()
            except Exception:
                pass
            return
        except LLMTimeoutError as e:
            logger.warning(
                "LLM %s timeout after %.1fs (partial=%r)",
                e.kind, e.timeout_s, e.partial_text[:80],
            )
            # Real-world failure — push the state machine forward without
            # waiting for the next probe.
            avail = getattr(self, "llm_availability", None)
            if avail is not None:
                try:
                    avail.report_request_failure()
                except Exception:  # pragma: no cover - defensive
                    pass
            msg = (
                f"LLM 响应超时（{e.kind}, >{e.timeout_s:.0f}s）。"
                "可能 edge-llm 服务挂了或输入太长。"
            )
            try:
                await self._broadcast(
                    "on_error",
                    TypedLLMError(
                        "llm_timeout",
                        msg,
                        exc_class="LLMTimeoutError",
                        kind=e.kind,
                        timeout_s=e.timeout_s,
                    ),
                )
            except Exception:
                pass
            self._set_state(ConvState.IDLE)
            try:
                self._reset_sleep_timer()
            except Exception:
                pass
        except Exception as e:
            logger.exception("on_user_utterance failed")
            # Real-world failure — feed back to the availability machine
            # (only for LLM-class errors; other exceptions might be local
            # bugs and shouldn't poison the breaker).
            try:
                from openai import APIError as _APIError
                _is_llm_err = isinstance(e, (_APIError, LLMStreamError))
            except Exception:  # pragma: no cover - defensive
                _is_llm_err = isinstance(e, LLMStreamError)
            if _is_llm_err:
                avail = getattr(self, "llm_availability", None)
                if avail is not None:
                    try:
                        avail.report_request_failure()
                    except Exception:  # pragma: no cover - defensive
                        pass
            # A3: surface non-timeout LLM failures to the dashboard so
            # operators see *something* when edge-llm crashes or returns
            # a 4xx. Wrap the original exception's repr into a clean
            # RuntimeError (the on_error contract already accepts a
            # BaseException and prefers str()).
            try:
                exc_class = type(e).__name__
                msg = f"LLM 调用失败（{exc_class}）：{e}"
                err_type = (
                    "llm_stream_error"
                    if isinstance(e, LLMStreamError)
                    else "llm_failure"
                )
                await self._broadcast(
                    "on_error",
                    TypedLLMError(err_type, msg, exc_class=exc_class),
                )
            except Exception:
                pass
            self._set_state(ConvState.IDLE)
            try:
                self._reset_sleep_timer()
            except Exception:
                pass

    async def broadcast(self, hook_name: str, *args) -> None:
        """Public hook broadcaster -- call from subclasses to fan out events.

        Used by DialogueApp.on_user_utterance to fan out per-token deltas
        (`on_assistant_token`) since the dispatch loop has no access to
        the LLM's token stream.
        """
        plugins = getattr(self, "plugins", None)
        if not plugins:
            return
        coros = []
        for p in plugins:
            fn = getattr(p, hook_name, None)
            if fn is None:
                continue
            coros.append(_safe_call(p.name, hook_name, fn, *args))
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)

    # Backwards-compatible alias used internally by the dispatch loop.
    _broadcast = broadcast


async def _safe_call(plugin_name: str, hook: str, fn, *args) -> None:  # noqa: ANN001
    try:
        result = fn(*args)
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        logger.exception("plugin %s.%s failed", plugin_name, hook)


__all__ = ["BaseApp"]

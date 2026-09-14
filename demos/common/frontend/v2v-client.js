/* V2VStreamClient — browser client for `WS /v2v/stream` (native ES module).
 *
 * Wire contract: docs/api/realtime-v2.md (authoritative). Summary:
 *   client → server : `session.update` after `session.created`, then
 *                     binary int16 LE PCM (mic) frames interleaved with JSON
 *                     canonical input-buffer / response control events.
 *   server → client : canonical session / input-buffer / transcription /
 *                     response events plus pure binary PCM. Audio format is
 *                     declared by session.updated; V2 has no binary header.
 *
 * Playback: received PCM is scheduled gap-free on a Web Audio context (same
 * scheduling approach as slv-client.js TTSStreamPlayer). `interrupt()` sends
 * `response.cancel`, reports the actually played duration with
 * `conversation.item.truncate`, and silences local scheduled audio.
 *
 * Unknown JSON frame types are ignored (forwarded to onEvent), as the
 * protocol doc instructs.
 */

/**
 * Recommended session options for full voice-chat clients (v2v-chat and
 * friends). ONE endpoint detector per session — the single most expensive
 * misconfiguration of /v2v/stream is two detectors racing, which silently
 * truncates transcripts with no error on either side
 * (docs/CONFIGURATION.md "Streaming ASR endpointing: pick exactly one
 * detector").
 *
 * Which detector depends on the ASR backend, so pick it from the backend's
 * capability (`fullChatSession({backendEndpointsInternally})`):
 *
 *   backendEndpointsInternally: true  → vad: "none"
 *     Backends that end a turn on their OWN internal VAD (RK Qwen3 true
 *     streaming declares prefer_backend_endpoint_vad). Asking for server VAD
 *     as well would race it. This mirrors the production ovs-agent configs
 *     (slv_config.vad="none" + a client-side VAD).
 *
 *   backendEndpointsInternally: false → vad: "silero"
 *     Offline/accumulate-then-transcribe backends (Whisper, SenseVoice via
 *     OfflineAccumulateStream) never end a turn themselves: with vad:"none"
 *     and no client-side VAD or commits, spoken turns can never complete, so
 *     the server VAD must be the single detector.
 *
 * Trade-off of vad:"none": no server speech_started events, so voice barge-in
 * during TTS playback is unavailable — drive interruption via
 * client.interrupt() (the 打断 button) or run half-duplex.
 */
export function fullChatSession(options = {}) {
  const owns = options.backendEndpointsInternally;
  return Object.freeze({ vad: owns === false ? "silero" : "none" });
}

/**
 * Backend-owns-endpointing probe shared by the demos: the SLV `/health`
 * ``asr_backend`` name is the only capability signal on the wire today.
 * Unknown backends return false so the server VAD stays the detector — the
 * safe default for an offline backend, and merely redundant (never
 * truncating) for one that endpoints internally.
 */
export function backendEndpointsInternally(asrBackendName) {
  const name = String(asrBackendName || "").toLowerCase();
  // RK Qwen3 streaming (qwen3_asr_rk) owns its endpoint VAD in both
  // true_streaming and chunk_confirm modes.
  return name.includes("qwen3_asr_rk");
}

/**
 * Resolve the endpoint detector from an SLV /health payload.
 *
 * Prefers the server's explicit `asr_owns_endpointing` flag (added
 * 2026-09-13); older servers omit it and fall back to the backend-name
 * heuristic. When neither is available this returns `false`, i.e. the SERVER
 * VAD stays the detector: for an offline backend (Whisper/SenseVoice)
 * `vad:"none"` makes spoken turns uncompletable, which is strictly worse than
 * a redundant server VAD on a backend that endpoints internally (that case
 * only risks truncation, and the server logs a single-detector warning).
 */
export function resolveBackendEndpointsInternally(health) {
  if (health && typeof health.asr_owns_endpointing === "boolean") {
    return health.asr_owns_endpointing;
  }
  return backendEndpointsInternally(health && health.asr_backend);
}

export class V2VStreamClient {
  /**
   * @param {object} opts
   * @param {string}  [opts.baseUrl]        http(s) origin of the SLV server (default: page origin)
   * @param {string}  [opts.path]           WS path (default "/v2v/stream")
   * @param {string}  [opts.asrLanguage]    e.g. "auto" | "zh" | "en"; omit to disable ASR
   * @param {string}  [opts.ttsLanguage]    e.g. "zh" | "en" | "auto"; omit to disable TTS
   * @param {number}  [opts.ttsSpeakerId]   speaker ID from /tts/speakers (optional)
   * @param {number}  [opts.ttsSpeed]       optional; some backends only
   * @param {number}  [opts.sampleRate]     mic PCM rate (default 16000)
   * @param {string}  [opts.vad]            "silero" | "webrtcvad" | "none" (optional)
   * @param {number}  [opts.vadSilenceMs]   silence to auto-endpoint (optional)
   * @param {boolean} [opts.multiUtterance] keep session open across utterances (default false)
   * @param {boolean} [opts.playback]       schedule received audio on Web Audio (default true)
   * @param {AudioContext} [opts.audioContext] shared context (created lazily otherwise)
   *
   * Callbacks (all optional):
   *   onAsrPartial(msg)        {"type":"asr_partial","text","is_stable"}
   *   onAsrEndpoint(msg)       {"type":"asr_endpoint"}
   *   onAsrFinal(msg)          {"type":"asr_final","text",...session_complete?}
   *   onVadEvent(event, msg)   event = "speech_start" | "speech_end"
   *   onTtsStarted(msg)        {"type":"tts_started","sentence"}
   *   onTtsSentenceDone(msg)   {"type":"tts_sentence_done","sentence"}
   *   onTtsDone(msg)           {"type":"tts_done"}
   *   onError(msg)             {"type":"error","error"}
   *   onEvent(msg)             any other / unknown JSON frame
   *   onSampleRate(rate)       negotiated session output rate
   *   onAudioChunk(int16)      every non-empty PCM payload, pre-scheduling
   *   onPlaybackStart()        local playback transitioned silent → audible
   *   onPlaybackEnd()          scheduled audio fully played OR stopped/cleared
   *   onClose(ev)              WS closed (ev.code 4429 = session slots full)
   */
  constructor(opts = {}) {
    this.opts = opts;
    this.ws = null;
    this.sampleRate = 0;          // TTS output rate from session.updated
    this._leftover = new Uint8Array(0); // odd trailing byte between chunks
    // playback state
    this.ctx = opts.audioContext || null;
    this._sources = [];
    this._playhead = 0;           // ctx.currentTime-based scheduling cursor
    this._primed = false;         // jitter-buffer lead applied for this response
    this._playing = false;
    this._endTimer = null;
    this._connectResolve = null;
    this._connectReject = null;
    this._activeResponseId = null;
    this._activeOutputItemId = null;
    this._responsePlaybackStartedAt = null;
    this.capabilities = {};
  }

  get _wsUrl() {
    const base = this.opts.baseUrl || window.location.origin;
    const u = new URL(this.opts.path || "/v2v/stream", base);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    return u.toString();
  }

  get connected() {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  get isPlaying() {
    return this._playing;
  }

  /** Canonical Realtime V2 session.update. */
  _configFrame() {
    const o = this.opts;
    const input = {
      format: { type: "audio/pcm", rate: o.sampleRate || 16000,
                channels: 1, endianness: "little" },
      turn_detection: {
        type: o.vad === "none" ? "none" : "server_vad",
        backend: o.vad || "silero",
        silence_duration_ms: o.vadSilenceMs || 400,
        // Auto-create a response at turn end when the server runs the LLM
        // loop (OVS_V2V_SERVER_LOOP=1); callers targeting an ASR-only server
        // can pass createResponse:false to keep the old manual behavior.
        create_response: o.createResponse !== undefined ? !!o.createResponse : false,
        interrupt_response: true,
      },
    };
    if (o.asrLanguage != null) input.transcription = { language: o.asrLanguage };
    const output = {
      format: { type: "audio/pcm", rate: o.sampleRate || 16000,
                channels: 1, endianness: "little" },
    };
    if (o.ttsLanguage != null) output.language = o.ttsLanguage;
    if (o.ttsSpeakerId != null) output.speaker_id = o.ttsSpeakerId;
    if (o.ttsSpeed != null) output.speed = o.ttsSpeed;
    return { type: "session.update", session: {
      type: "realtime", output_modalities: ["audio"], audio: { input, output },
    }};
  }

  /** Open the WS and send the leading config frame. */
  connect() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this._wsUrl, "seeed.realtime.v2");
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        if (ws.protocol !== "seeed.realtime.v2") {
          reject(new Error("server did not accept seeed.realtime.v2"));
          ws.close(1002, "subprotocol required");
        }
      };
      ws.onerror = (e) => reject(e);
      ws.onclose = (ev) => {
        if (this._connectReject) {
          this._connectReject(new Error(`WebSocket closed during handshake (${ev.code})`));
          this._connectResolve = this._connectReject = null;
        }
        this.ws = null;
        if (this.opts.onClose) this.opts.onClose(ev);
      };
      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") this._onJson(ev.data);
        else this._onBinary(ev.data);
      };
      this.ws = ws;
      this._connectResolve = resolve;
      this._connectReject = reject;
    });
  }

  /* ── uplink ─────────────────────────────────────────────────────── */

  /** @param {Int16Array|ArrayBuffer} pcm mic PCM at config sample_rate */
  sendPcm(pcm) {
    if (!this.connected) return;
    this.ws.send(pcm instanceof Int16Array ? pcm.buffer : pcm);
  }

  /** Migration-only incremental TTS input; prefer speak() for exact speech. */
  sendText(text) {
    if (!this.connected) return;
    this.ws.send(JSON.stringify({ type: "text", text }));
  }

  /** Flush the remaining TTS sentence buffer. */
  ttsFlush() {
    if (!this.connected) return;
    this.ws.send(JSON.stringify({ type: "tts_flush" }));
  }

  /** Deterministic, history-free speech. */
  speak(text, conversation = "none") {
    if (!this.connected || !text) return;
    this.ws.send(JSON.stringify({
      type: "x_v2v.response.speak", speech: { text, conversation },
    }));
  }

  /** Replace provider-visible tools and instructions for subsequent turns. */
  updateTools(tools, instructions = null, llmParams = null) {
    if (!this.connected) return;
    const session = { tools: tools || [] };
    if (instructions != null) session.instructions = instructions;
    if (llmParams != null) session.x_v2v = { llm_params: llmParams };
    this.ws.send(JSON.stringify({ type: "session.update", session }));
  }

  sendToolResult(callId, output) {
    if (!this.connected) return;
    this.ws.send(JSON.stringify({
      type: "conversation.item.create",
      item: {
        type: "function_call_output", call_id: callId,
        output: typeof output === "string" ? output : JSON.stringify(output),
      },
    }));
  }

  resetConversation() {
    if (this.connected) {
      this.ws.send(JSON.stringify({ type: "x_v2v.conversation.reset" }));
    }
  }

  /** Manually finalize ASR (overrides VAD). */
  asrEos() {
    if (!this.connected) return;
    this.ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
  }

  /** Cancel, truncate provider history to heard audio, and stop locally. */
  interrupt() {
    if (this.connected) {
      this.ws.send(JSON.stringify({
        type: "response.cancel", response_id: this._activeResponseId,
      }));
      if (this._activeOutputItemId && this.capabilities.conversation_truncate !== false) {
        const now = this.ctx?.currentTime || 0;
        const playedMs = this._responsePlaybackStartedAt == null ? 0
          : Math.max(0, Math.round((now - this._responsePlaybackStartedAt) * 1000));
        this.ws.send(JSON.stringify({
          type: "conversation.item.truncate",
          item_id: this._activeOutputItemId,
          content_index: 0,
          audio_end_ms: playedMs,
        }));
      }
    }
    this.stopPlayback();
  }

  /* ── downlink ───────────────────────────────────────────────────── */

  _onJson(data) {
    let msg;
    try { msg = JSON.parse(data); } catch { return; }
    const cb = this.opts;
    switch (msg.type) {
      case "session.created":
        this.ws.send(JSON.stringify(this._configFrame()));
        break;
      case "session.updated": {
        this.capabilities = msg.session?.capabilities || this.capabilities;
        const rate = msg.session?.audio?.output?.format?.rate
          || msg.session?.audio?.output?.format?.sample_rate;
        if (rate) {
          this.sampleRate = Number(rate);
          if (cb.onSampleRate) cb.onSampleRate(this.sampleRate);
        }
        if (this._connectResolve) this._connectResolve(this);
        this._connectResolve = this._connectReject = null;
        break;
      }
      case "conversation.item.input_audio_transcription.delta":
        cb.onAsrPartial && cb.onAsrPartial({ ...msg, text: msg.delta || "" }); break;
      case "input_audio_buffer.committed":
        cb.onAsrEndpoint && cb.onAsrEndpoint(msg); break;
      case "conversation.item.input_audio_transcription.completed":
        cb.onAsrFinal && cb.onAsrFinal({ ...msg, text: msg.transcript || "" }); break;
      case "input_audio_buffer.speech_started":
        cb.onVadEvent && cb.onVadEvent("speech_start", msg); break;
      case "input_audio_buffer.speech_stopped":
        cb.onVadEvent && cb.onVadEvent("speech_end", msg); break;
      case "response.created":
        this._activeResponseId = msg.response?.id || null;
        this._activeOutputItemId = null;
        this._responsePlaybackStartedAt = null;
        this._primed = false;   // new response: re-apply the jitter-buffer lead once
        cb.onEvent && cb.onEvent(msg); break;
      case "response.output_item.added":
        if (msg.item?.type === "message" && msg.item?.role === "assistant") {
          this._activeOutputItemId = msg.item.id || null;
        }
        cb.onEvent && cb.onEvent(msg); break;
      case "response.function_call_arguments.done":
        cb.onToolCall && cb.onToolCall({
          ...msg,
          id: msg.call_id,
          arguments: (() => { try { return JSON.parse(msg.arguments || "{}"); }
                              catch { return {}; } })(),
        });
        break;
      case "x_v2v.tts_sentence.started":
        cb.onTtsStarted && cb.onTtsStarted(msg); break;
      case "x_v2v.tts_sentence.done":
        cb.onTtsSentenceDone && cb.onTtsSentenceDone(msg); break;
      case "response.done":
        if (msg.response?.id === this._activeResponseId) {
          this._activeResponseId = null;
          this._activeOutputItemId = null;
        }
        cb.onTtsDone && cb.onTtsDone(msg); break;
      case "error": cb.onError && cb.onError(msg); break;
      default: cb.onEvent && cb.onEvent(msg); // unknown frames: ignorable
    }
  }

  _onBinary(buf) {
    let bytes = new Uint8Array(buf);
    if (!this.sampleRate) return; // session.updated must precede audio
    if (this._leftover.length) {
      const merged = new Uint8Array(this._leftover.length + bytes.length);
      merged.set(this._leftover); merged.set(bytes, this._leftover.length);
      bytes = merged;
      this._leftover = new Uint8Array(0);
    }
    const evenLen = bytes.length - (bytes.length % 2);
    if (bytes.length % 2) this._leftover = bytes.slice(evenLen);
    if (!evenLen) return;
    const int16 = new Int16Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + evenLen));
    if (this.opts.onAudioChunk) this.opts.onAudioChunk(int16);
    if (this.opts.playback !== false) this._schedule(int16);
  }

  /* ── gap-free Web Audio playback (TTSStreamPlayer scheduling) ───── */

  _ensureCtx() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  }

  _schedule(int16) {
    if (!int16.length || !this.sampleRate) return;
    const ctx = this._ensureCtx();
    const buf = ctx.createBuffer(1, int16.length, this.sampleRate);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < int16.length; i++) ch[i] = int16[i] / 32768;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    // Jitter buffer: 30ms is fine on loopback but over Tailscale-relayed WS
    // (RTT 80-210ms) the 40ms TTS chunks arrive with heavy jitter and the
    // playback underruns (choppy audio). The lead is applied ONCE per response
    // (or after an underrun) — applying it per chunk would push every chunk
    // 400ms past `now` and insert silence between chunks that had already
    // arrived contiguously. Later chunks stay glued to `_playhead`.
    const now = ctx.currentTime;
    let startAt;
    if (this._playhead > now + 0.01) {
      startAt = this._playhead; // contiguous with already-buffered audio
    } else if (this._primed) {
      startAt = now + 0.03; // underrun mid-response: resume promptly
    } else {
      startAt = now + 0.4; // first chunk of a response: absorb jitter
      this._primed = true;
    }
    if (this._responsePlaybackStartedAt == null) {
      this._responsePlaybackStartedAt = startAt;
    }
    src.start(startAt);
    this._playhead = startAt + buf.duration;
    this._sources.push(src);
    if (!this._playing) {
      this._playing = true;
      if (this.opts.onPlaybackStart) this.opts.onPlaybackStart();
    }
    this._armEndTimer();
  }

  _armEndTimer() {
    if (this._endTimer) clearTimeout(this._endTimer);
    const ctx = this.ctx;
    const waitMs = Math.max(0, (this._playhead - ctx.currentTime) * 1000) + 80;
    this._endTimer = setTimeout(() => {
      this._endTimer = null;
      if (!this._playing) return;
      if (this.ctx && this._playhead - this.ctx.currentTime > 0.05) {
        this._armEndTimer(); // more audio got scheduled meanwhile
        return;
      }
      this._sources = [];
      this._playing = false;
      this._primed = false;   // next response primes the jitter buffer again
      if (this.opts.onPlaybackEnd) this.opts.onPlaybackEnd();
    }, waitMs);
  }

  /** Local-only: silence + clear everything scheduled (no wire frame). */
  stopPlayback() {
    if (this._endTimer) { clearTimeout(this._endTimer); this._endTimer = null; }
    for (const s of this._sources) { try { s.stop(); } catch { /* ended */ } }
    this._sources = [];
    this._playhead = 0;
    this._primed = false;
    if (this._playing) {
      this._playing = false;
      if (this.opts.onPlaybackEnd) this.opts.onPlaybackEnd();
    }
  }

  /** Close the WS (server frees the session slot + cancels in-flight work). */
  close() {
    this.stopPlayback();
    if (this.ws) {
      const ws = this.ws;
      this.ws = null;
      try { ws.close(); } catch { /* already closing */ }
    }
  }
}

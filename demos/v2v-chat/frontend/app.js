/* v2v-chat — full voice conversation with barge-in over WS /v2v/stream.
 *
 * Wire contract: docs/api/v2v-stream.md via /common/v2v-client.js. The
 * protocol carries NO LLM token events (the reply text reaches the client
 * only through tts_started/tts_sentence_done sentences), so the "LLM" bar is
 * derived from documented anchors: asr_final → first tts_started of the turn
 * (server-loop LLM producing its first speakable sentence). The reply itself
 * requires the SLV server to run its server-side loop (OVS_V2V_SERVER_LOOP=1);
 * without it no reply ever arrives and a watchdog surfaces a plain-words hint.
 *
 * Duplex state machine (documented for the report):
 *   session: idle → connecting → live → idle
 *   phase (while live): listening → userSpeaking → thinking → speaking → listening
 *     - listening:     mic streams to server; VAD waits for speech.
 *     - userSpeaking:  vad_event speech_start seen (or first asr_partial).
 *     - thinking:      asr_final committed; waiting for tts_started/audio.
 *     - speaking:      local playback audible (client onPlaybackStart).
 *   HALF duplex (default): during thinking+speaking mic chunks are DROPPED
 *     client-side (level bar grayed) — echo can't fake a barge-in; the only
 *     interruption path is the big ⏹ button → client.interrupt() sends
 *     {"type":"abort"} + stops/clears local playback instantly.
 *   FULL duplex (user opt-in, headset / AEC mic): mic keeps streaming during
 *     speaking; server VAD does the barge-in (cancels TTS on speech_start);
 *     the client mirrors it by dropping local playback the moment
 *     vad_event speech_start arrives while speaking. ⏹ still works too.
 */

import { createStatusPill, createMetricCard, createModelSwitchPanel } from "/common/ui.js";
import { MicCapture } from "/common/mic-capture.js";
import {
  V2VStreamClient,
  fullChatSession,
  resolveBackendEndpointsInternally,
} from "/common/v2v-client.js";

/* ── i18n ──────────────────────────────────────────────────────────── */
const I18N = {
  zh: {
    brandName: "语音对话",
    tagline: "ASR → LLM → TTS · 全程本机推理",
    step1: "选模型", step2: "点击说话", step3: "听回答 · 随时打断",
    start: "开始对话", stop: "结束对话", starting: "连接中…",
    interrupt: "打断",
    hintIdle: "点击麦克风按钮开始一段语音对话。",
    hintStarting: "正在打开麦克风并连接语音服务…",
    hintListening: "正在聆听 —— 直接说话，停顿后自动成句。",
    hintUserSpeaking: "听到了 —— 请继续说。",
    hintThinking: "正在思考回答…",
    hintSpeakingHalf: "正在回答（半双工：此时麦克风静音）。想插话就按 ⏹ 打断。",
    hintSpeakingFull: "正在回答（全双工）—— 直接开口即可打断。",
    chatTitle: "对话",
    chatEmpty: "对话会实时出现在这里 —— 你说的话在右边，设备的回答在左边",
    latencyTitle: "逐阶段延迟",
    latTotal: "语音响应总延迟",
    latAsr: "ASR 定格", latLlm: "LLM 首句", latTts: "TTS 首音频",
    latNote: "总延迟 = 说完话（VAD 判停）到听见第一声回答。协议没有单独的 LLM token 事件，LLM 段按 asr_final → tts_started 推算（含分句缓冲）。",
    duplexTitle: "双工模式",
    duplexHalf: "半双工", duplexFull: "全双工",
    duplexHalfState: "回答播放时麦克风静音，用 ⏹ 按钮打断",
    duplexFullState: "回答播放时麦克风保持开启，开口即打断",
    duplexTip: "戴耳机或使用带 AEC 的 reSpeaker 麦克风可开启全双工语音打断；笔记本外放请保持半双工，避免回声误触发。",
    switchTitle: "模型切换",
    settingsTitle: "模型 / 语言",
    asrLangLabel: "识别语言", ttsLangLabel: "回答语言",
    langAuto: "自动", langZh: "中文", langEn: "英文",
    modelHint: "ASR/TTS 模型热切换在演示门户进行 →",
    retry: "重试",
    userTag: "你", assistantTag: "设备",
    interruptedTag: "已打断",
    statusIdle: "未连接", statusStarting: "连接中…",
    statusListening: "聆听中", statusUserSpeaking: "你在说话",
    statusThinking: "思考中", statusSpeaking: "回答中",
    errConfig: "读取演示配置失败，请刷新页面重试。",
    errWs: "连不上语音服务，请确认设备在线后重试。目标：",
    errClosed: "与语音服务的连接断开了，点击重试恢复。",
    errBusy: "会话已满（同时对话人数达到上限），稍后再试。",
    errServer: "服务端出错了：",
    errNoReply: "识别成功但没有收到语音回答 —— 完整对话需要服务端开启 LLM 回路（OVS_V2V_SERVER_LOOP=1）。可继续测试识别与打断。",
    footer: "seeed-local-voice · 识别、对话与合成全程在本机完成，不经云端",
  },
  en: {
    brandName: "Voice Chat",
    tagline: "ASR → LLM → TTS · fully on-device",
    step1: "Pick model", step2: "Tap & talk", step3: "Hear it · barge in anytime",
    start: "Start chat", stop: "End chat", starting: "Connecting…",
    interrupt: "Interrupt",
    hintIdle: "Tap the mic button to start a voice conversation.",
    hintStarting: "Opening the microphone and connecting to the voice server…",
    hintListening: "Listening — just talk; a pause ends your turn.",
    hintUserSpeaking: "Got it — keep talking.",
    hintThinking: "Thinking about the answer…",
    hintSpeakingHalf: "Answering (half-duplex: mic is muted). Hit ⏹ to interrupt.",
    hintSpeakingFull: "Answering (full duplex) — just start talking to interrupt.",
    chatTitle: "Conversation",
    chatEmpty: "The conversation shows up here live — you on the right, the device on the left",
    latencyTitle: "Per-stage latency",
    latTotal: "Voice-to-voice latency",
    latAsr: "ASR final", latLlm: "LLM first sentence", latTts: "TTS first audio",
    latNote: "Total = end of your speech (VAD) to the first audible reply. The protocol has no LLM token events; the LLM bar is derived from asr_final → tts_started (includes sentence buffering).",
    duplexTitle: "Duplex mode",
    duplexHalf: "Half duplex", duplexFull: "Full duplex",
    duplexHalfState: "Mic muted while the reply plays; interrupt with ⏹",
    duplexFullState: "Mic stays open while the reply plays; speak to barge in",
    duplexTip: "Wear headphones or use an AEC microphone (e.g. reSpeaker) to enable full-duplex voice barge-in. On laptop speakers keep half duplex to avoid echo-triggered false interrupts.",
    switchTitle: "Switch model",
    settingsTitle: "Model / Language",
    asrLangLabel: "Recognition language", ttsLangLabel: "Reply language",
    langAuto: "Auto", langZh: "Chinese", langEn: "English",
    modelHint: "Hot-swap ASR/TTS models in the gallery portal →",
    retry: "Retry",
    userTag: "You", assistantTag: "Device",
    interruptedTag: "interrupted",
    statusIdle: "Disconnected", statusStarting: "Connecting…",
    statusListening: "Listening", statusUserSpeaking: "You're talking",
    statusThinking: "Thinking", statusSpeaking: "Answering",
    errConfig: "Failed to load demo config. Refresh the page and retry.",
    errWs: "Cannot reach the voice server. Check the device is online, then retry. Target: ",
    errClosed: "Lost the connection to the voice server. Hit retry to resume.",
    errBusy: "Session slots are full (too many concurrent chats). Try again in a moment.",
    errServer: "Server error: ",
    errNoReply: "Speech was recognized but no spoken reply arrived — full chat needs the server-side LLM loop (OVS_V2V_SERVER_LOOP=1). ASR and barge-in still work.",
    footer: "seeed-local-voice · recognition, chat and synthesis all happen on this device, no cloud",
  },
};
let lang = localStorage.getItem("slv-demo-lang") || "zh";
const t = (key) => I18N[lang][key];

const $ = (id) => document.getElementById(id);

function applyStaticI18n() {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  for (const node of document.querySelectorAll("[data-i18n]")) {
    node.textContent = t(node.dataset.i18n);
  }
  $("lang-btn").textContent = lang === "zh" ? "English" : "中文";
  renderDuplex();
}

/* ── shared components ─────────────────────────────────────────────── */
const connPill = createStatusPill($("pills"), { state: "idle", text: "…" });
const totalCard = createMetricCard($("metrics"), { label: "", unit: "s", digits: 2 });

$("gallery-link").href = `//${window.location.hostname}:8700/`;

// Inline model switch — v2v runs the full ASR→LLM→TTS loop, so expose both
// kinds (ASR/TTS tabs). Talks to this backend's shared /api/switch.
createModelSwitchPanel($("switch-panel"), { kinds: ["asr", "tts"] });

/* ── state ─────────────────────────────────────────────────────────── */
let state = "idle";        // "idle" | "connecting" | "live"
let phase = "listening";   // while live: "listening" | "userSpeaking" | "thinking" | "speaking"
let duplex = "half";       // "half" | "full"
let mic = null;
let client = null;
let cfg = null;
let hasReply = false;      // any assistant audio heard this session (drives step ③)

let pendingUserEl = null;  // live-updating user bubble (asr_partial)
let assistantEl = null;    // current assistant bubble (per turn)
let replyWatchdog = null;  // fires errNoReply hint when server-loop is off

/* Per-turn latency anchors (all performance.now() ms):
 *   speechEndTs → asr_final       = ASR bar
 *   asr_final   → first tts_started = "LLM first sentence" bar (see header)
 *   tts_started → first PCM chunk = TTS bar
 *   speechEndTs → first PCM chunk = big total number */
let pendingSpeechEndTs = null;
let turn = null;

function beginTurn(now) {
  turn = {
    speechEndTs: pendingSpeechEndTs ?? now,
    asrFinalTs: now,
    ttsStartedTs: null,
    firstAudioTs: null,
  };
  pendingSpeechEndTs = null;
}

/* ── rendering ─────────────────────────────────────────────────────── */
const micBtn = $("mic-btn");
const micLabel = $("mic-label");
const stopBtn = $("stop-btn");
const level = $("level");
const levelBar = $("level-bar");
const chatBox = $("chat");
const asrLangSel = $("asr-lang");
const ttsLangSel = $("tts-lang");
const duplexToggle = $("duplex-toggle");

function micGated() {
  return duplex === "half" && (phase === "thinking" || phase === "speaking");
}

function updateSteps() {
  const current = state !== "live" ? 1 : (hasReply ? 3 : 2);
  for (let i = 1; i <= 3; i++) {
    const el = $(`step-${i}`);
    el.classList.toggle("active", i === current);
    el.classList.toggle("done", i < current);
  }
}

function render() {
  micBtn.classList.toggle("live", state === "live");
  micBtn.disabled = state === "connecting";
  asrLangSel.disabled = state !== "idle";
  ttsLangSel.disabled = state !== "idle";
  const interruptible = state === "live" && (phase === "thinking" || phase === "speaking");
  stopBtn.classList.toggle("visible", interruptible);
  level.classList.toggle("muted", state === "live" && micGated());

  if (state === "idle") {
    micLabel.textContent = t("start");
    $("hint").textContent = t("hintIdle");
    connPill.set("idle", t("statusIdle"));
    levelBar.style.width = "0%";
  } else if (state === "connecting") {
    micLabel.textContent = t("starting");
    $("hint").textContent = t("hintStarting");
    connPill.set("busy", t("statusStarting"));
  } else {
    micLabel.textContent = t("stop");
    if (phase === "listening") {
      $("hint").textContent = t("hintListening");
      connPill.set("ok", t("statusListening"));
    } else if (phase === "userSpeaking") {
      $("hint").textContent = t("hintUserSpeaking");
      connPill.set("ok", t("statusUserSpeaking"));
    } else if (phase === "thinking") {
      $("hint").textContent = t("hintThinking");
      connPill.set("busy", t("statusThinking"));
    } else {
      $("hint").textContent = duplex === "full" ? t("hintSpeakingFull") : t("hintSpeakingHalf");
      connPill.set("busy", t("statusSpeaking"));
    }
  }
  updateSteps();
}

function setPhase(next) {
  phase = next;
  render();
}

function renderDuplex() {
  $("duplex-label").textContent = duplex === "full" ? t("duplexFull") : t("duplexHalf");
  $("duplex-state").textContent = duplex === "full" ? t("duplexFullState") : t("duplexHalfState");
}

function showError(msg) {
  $("error-msg").textContent = msg;
  $("error-box").classList.add("visible");
}
function hideError() {
  $("error-box").classList.remove("visible");
}

/* ── chat bubbles ──────────────────────────────────────────────────── */
function makeBubble(cls, whoText) {
  $("chat-empty")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = whoText;
  const body = document.createElement("span");
  body.className = "body";
  el.append(who, body);
  chatBox.appendChild(el);
  chatBox.scrollTop = chatBox.scrollHeight;
  return el;
}

function upsertPendingUser(text) {
  if (!text) return;
  if (!pendingUserEl) pendingUserEl = makeBubble("user pending", t("userTag"));
  pendingUserEl.querySelector(".body").textContent = text;
  chatBox.scrollTop = chatBox.scrollHeight;
}

function commitUser(text) {
  if (!pendingUserEl) pendingUserEl = makeBubble("user pending", t("userTag"));
  pendingUserEl.classList.remove("pending");
  pendingUserEl.querySelector(".body").textContent = text;
  pendingUserEl = null;
  chatBox.scrollTop = chatBox.scrollHeight;
}

function dropPendingUser() {
  if (pendingUserEl) { pendingUserEl.remove(); pendingUserEl = null; }
}

function appendAssistantSentence(sentence) {
  if (!assistantEl) {
    assistantEl = makeBubble("assistant", t("assistantTag"));
    const eq = document.createElement("span");
    eq.className = "eq";
    eq.innerHTML = "<i></i><i></i><i></i>";
    assistantEl.appendChild(eq);
  }
  if (sentence) {
    const body = assistantEl.querySelector(".body");
    body.textContent = (body.textContent ? body.textContent : "") + sentence;
  }
  chatBox.scrollTop = chatBox.scrollHeight;
}

function settleAssistant(interrupted) {
  if (!assistantEl) return;
  assistantEl.classList.remove("playing");
  if (interrupted && !assistantEl.querySelector(".interrupted-tag")) {
    const tag = document.createElement("span");
    tag.className = "interrupted-tag";
    tag.textContent = `⏹ ${t("interruptedTag")}`;
    assistantEl.appendChild(tag);
  }
  assistantEl = null;
}

/* ── latency bars ──────────────────────────────────────────────────── */
function fmtMs(v) {
  return v == null ? "—" : `${Math.max(0, Math.round(v))} ms`;
}

function updateLatencyUI() {
  if (!turn) return;
  const asrMs = turn.asrFinalTs != null && turn.speechEndTs != null
    ? turn.asrFinalTs - turn.speechEndTs : null;
  const llmMs = turn.ttsStartedTs != null && turn.asrFinalTs != null
    ? turn.ttsStartedTs - turn.asrFinalTs : null;
  const ttsMs = turn.firstAudioTs != null && turn.ttsStartedTs != null
    ? turn.firstAudioTs - turn.ttsStartedTs : null;
  const totalMs = turn.firstAudioTs != null && turn.speechEndTs != null
    ? turn.firstAudioTs - turn.speechEndTs : null;

  $("val-asr").textContent = fmtMs(asrMs);
  $("val-llm").textContent = fmtMs(llmMs);
  $("val-tts").textContent = fmtMs(ttsMs);

  const total = Math.max(1, totalMs ?? ((asrMs ?? 0) + (llmMs ?? 0) + (ttsMs ?? 0)));
  const pct = (v) => `${Math.max(0, Math.min(100, ((v ?? 0) / total) * 100)).toFixed(1)}%`;
  $("seg-asr").style.width = pct(asrMs);
  $("seg-llm").style.width = pct(llmMs);
  $("seg-tts").style.width = pct(ttsMs);

  if (totalMs != null) totalCard.set(totalMs / 1000);
}

/* ── reply watchdog (server-loop OFF degrade, §header) ─────────────── */
function armReplyWatchdog() {
  clearReplyWatchdog();
  replyWatchdog = setTimeout(() => {
    replyWatchdog = null;
    if (state !== "live" || phase !== "thinking") return;
    showError(t("errNoReply"));
    setPhase("listening");
  }, 12000);
}
function clearReplyWatchdog() {
  if (replyWatchdog) { clearTimeout(replyWatchdog); replyWatchdog = null; }
}

/* ── SLV wiring ────────────────────────────────────────────────────── */
function slvBaseUrl() {
  const ws = cfg.ws;
  // Loopback SLV_URL means "SLV runs next to this demo backend" — from the
  // browser's point of view that's the host serving this page.
  const host = ws.loopback ? window.location.hostname : ws.host;
  const scheme = ws.scheme === "wss" ? "https" : "http";
  return `${scheme}://${host}:${ws.port}`;
}

function onPcm(pcm) {
  if (state !== "live" || !client) return;
  if (micGated()) return; // half-duplex: mic muted while the device replies
  client.sendPcm(pcm);
}
function onLevel(_rms, peak) {
  if (state === "idle") return;
  levelBar.style.width = `${Math.min(100, peak * 140).toFixed(1)}%`;
}

function onVadEvent(event) {
  if (state !== "live") return;
  if (event === "speech_start") {
    // Full duplex: the server cancels its in-flight TTS on speech_start (and
    // emits it BEFORE cancelling, per the protocol doc) — drop buffered local
    // playback immediately so the barge-in is audible right away.
    if (phase === "speaking" || phase === "thinking") {
      clearReplyWatchdog();
      client.stopPlayback();
      settleAssistant(true);
    }
    setPhase("userSpeaking");
  } else if (event === "speech_end") {
    pendingSpeechEndTs = performance.now();
  }
}

function onAsrPartial(msg) {
  const text = msg.text || "";
  if (!text) return;
  upsertPendingUser(text);
  if (phase === "listening") setPhase("userSpeaking");
}

function onAsrFinal(msg) {
  // multi_utterance session-end final can duplicate the last streamed one.
  if (msg.session_complete === true && msg.duplicate_of_streamed === true) return;
  const now = performance.now();
  const text = (msg.text || "").trim();
  if (!text) {
    dropPendingUser();
    pendingSpeechEndTs = null;
    if (phase === "userSpeaking") setPhase("listening");
    return;
  }
  commitUser(text);
  settleAssistant(false);   // any leftover bubble from the previous turn
  beginTurn(now);
  updateLatencyUI();
  setPhase("thinking");
  armReplyWatchdog();
}

function onTtsStarted(msg) {
  clearReplyWatchdog();
  hideError();
  const now = performance.now();
  if (turn && turn.ttsStartedTs === null) {
    turn.ttsStartedTs = now;
    updateLatencyUI();
  }
  appendAssistantSentence(msg.sentence || "");
}

function onAudioChunk() {
  if (turn && turn.firstAudioTs === null) {
    turn.firstAudioTs = performance.now();
    updateLatencyUI();
    if (!hasReply) { hasReply = true; updateSteps(); }
  }
}

function onPlaybackStart() {
  if (state !== "live") return;
  if (assistantEl) assistantEl.classList.add("playing");
  setPhase("speaking");
}

function onPlaybackEnd() {
  if (state !== "live") return;
  if (assistantEl) assistantEl.classList.remove("playing");
  if (phase === "speaking") {
    settleAssistant(false);
    setPhase("listening");
  }
}

function onTtsDone() {
  // Network side of the reply is complete; the UI transition is driven by
  // local playback end (onPlaybackEnd) so the bubble animates until audible
  // audio actually stops.
}

function onServerError(msg) {
  showError(t("errServer") + (msg.error || "unknown"));
}

function onClose(ev) {
  clearReplyWatchdog();
  if (state === "idle") return; // our own stop() — expected
  const code = ev && ev.code;
  cleanup();
  state = "idle";
  setPhase("listening");
  if (code === 4429) showError(t("errBusy"));
  else if (code !== 1000) showError(t("errClosed"));
}

/* ── start / stop / interrupt ──────────────────────────────────────── */
async function start() {
  if (state !== "idle") return;
  hideError();
  hasReply = false;
  pendingSpeechEndTs = null;
  turn = null;
  state = "connecting";
  render();

  try {
    const r = await fetch("/api/config");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    cfg = await r.json();
  } catch {
    state = "idle"; render();
    showError(t("errConfig"));
    return;
  }

  // Pick the single endpoint detector for this backend: vad:"none" only when
  // the ASR backend ends turns on its own (RK Qwen3); offline backends need
  // the server VAD or spoken turns never complete. /api/status proxies SLV
  // /health; when the probe fails we keep the SERVER VAD (safe for offline
  // backends) rather than silently disabling turn completion.
  let backendOwnsEndpointing = null;
  try {
    const rs = await fetch("/api/status");
    if (rs.ok) {
      const st = await rs.json();
      backendOwnsEndpointing = resolveBackendEndpointsInternally(st?.slv?.health);
    }
  } catch {
    /* probe is advisory; session still starts with the server VAD */
  }

  mic = new MicCapture({ onPcmChunk: onPcm, onLevel: onLevel });
  try {
    await mic.start();
  } catch (e) {
    mic = null;
    state = "idle"; render();
    showError(e && e.messages ? e.messages[lang] : String(e));
    return;
  }

  client = new V2VStreamClient({
    baseUrl: slvBaseUrl(),
    path: cfg.v2v_path || "/v2v/stream",
    asrLanguage: asrLangSel.value,
    ttsLanguage: ttsLangSel.value,
    sampleRate: 16000,
    // Single endpoint detector chosen from the backend capability
    // (fullChatSession docblock in common/frontend/v2v-client.js).
    vad:
      backendOwnsEndpointing === null
        ? "silero" // probe failed: keep the server VAD (turns must complete)
        : fullChatSession({
            backendEndpointsInternally: backendOwnsEndpointing,
          }).vad,
    multiUtterance: true,
    // This demo is the full-chat page: ask the server to auto-create a
    // response at each turn end (requires OVS_V2V_SERVER_LOOP=1 server-side).
    createResponse: true,
    onAsrPartial, onAsrFinal, onVadEvent,
    onTtsStarted, onTtsDone, onAudioChunk,
    onPlaybackStart, onPlaybackEnd,
    onError: onServerError, onClose,
  });
  try {
    await client.connect();
  } catch {
    const target = slvBaseUrl();
    cleanup();
    state = "idle"; render();
    showError(t("errWs") + target);
    return;
  }

  state = "live";
  setPhase("listening");
}

function stop() {
  if (state === "idle") return;
  state = "idle"; // flip first so onClose knows this close is intentional
  clearReplyWatchdog();
  cleanup();
  dropPendingUser();
  settleAssistant(false);
  setPhase("listening");
}

function cleanup() {
  if (mic) { mic.stop(); mic = null; }
  if (client) { client.close(); client = null; } // WS close frees the server session slot
  levelBar.style.width = "0%";
}

function interrupt() {
  if (state !== "live" || !client) return;
  clearReplyWatchdog();
  client.interrupt(); // {"type":"abort"} + local playback stopped/cleared
  settleAssistant(true);
  setPhase("listening");
}

/* ── events ────────────────────────────────────────────────────────── */
micBtn.addEventListener("click", () => {
  if (state === "live") stop();
  else start();
});
stopBtn.addEventListener("click", interrupt);
$("retry-btn").addEventListener("click", () => { hideError(); start(); });

duplexToggle.addEventListener("change", () => {
  duplex = duplexToggle.checked ? "full" : "half";
  renderDuplex();
  render();
});

$("lang-btn").addEventListener("click", () => {
  lang = lang === "zh" ? "en" : "zh";
  localStorage.setItem("slv-demo-lang", lang);
  applyStaticI18n();
  totalCard.setLabel(t("latTotal"));
  render();
});

// Release the server session slot on refresh / tab close.
function shutdown() { if (state !== "idle") stop(); }
window.addEventListener("pagehide", shutdown);
window.addEventListener("beforeunload", shutdown);

applyStaticI18n();
totalCard.setLabel(t("latTotal"));
render();

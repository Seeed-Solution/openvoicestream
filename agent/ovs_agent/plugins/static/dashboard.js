// OpenVoiceStream Agent Dashboard v2 — vanilla JS, no frameworks.
(function () {
  "use strict";

  // ── DOM refs ─────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);
  const statePill = $("statePill");
  const stateText = $("stateText");
  const chatEl = $("chat");
  const eventsEl = $("events");
  const historyEl = $("history");
  const wsDot = $("wsDot");
  const wsStateEl = $("wsState");
  const rcCount = $("rcCount");
  const uptimeEl = $("uptime");
  const rmsChart = $("rmsChart");
  const rmsThr = $("rmsThr");
  const vadStateEl = $("vadState");
  const ttsRow = $("ttsRow");
  const ttsText = $("ttsText");
  const ttsSentCount = $("ttsSentCount");
  const ttsBytesCur = $("ttsBytesCur");
  const ttsLastDur = $("ttsLastDur");
  const errList = $("errList");
  const errCount = $("errCount");
  const errDot = $("errDot");
  const toastEl = $("toast");

  // ── state ─────────────────────────────────────────────────────────
  let paused = false;
  const MAX_EVENT_ROWS = 500;
  const sessionStartMs = Date.now();

  let pendingPartialBubble = null;
  let pendingAssistantBubble = null;
  let pendingAssistantText = "";

  let rmsSamples = new Array(60).fill(0);
  let rafQueued = false;
  // Latency sparkline buffers (maxlen 30 each).
  const SPARK_MAX = 30;
  const sparkData = { asr: [], ttft: [], ttfa: [], rtt: [] };
  const sparkDirty = { asr: false, ttft: false, ttfa: false, rtt: false };
  let sparkRafQueued = false;

  const filters = {
    utt: true, tok: true, partial: false, stats: false,
    state: false, mic: false, error: true,
  };

  // Latency history (most-recent N)
  const latHist = { asr: [], ttft: [], ttfa: [], rtt: [] };
  const LAT_KEEP = 3;

  // ── helpers ──────────────────────────────────────────────────────
  function fmtTs(ms) {
    const d = new Date(ms);
    const p = (n) => String(n).padStart(2, "0");
    return p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds()) +
      "." + String(d.getMilliseconds()).padStart(3, "0");
  }
  function escape(s) {
    return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  }
  function showToast(msg, kind) {
    toastEl.textContent = msg;
    toastEl.className = "toast " + (kind || "");
    setTimeout(() => { toastEl.classList.add("hidden"); }, 2000);
    toastEl.classList.remove("hidden");
  }
  function shouldRender(ev) {
    if (ev === "stats") return filters.stats;
    if (ev === "on_mic_rms") return filters.mic;
    if (ev === "on_state_change") return filters.state;
    if (ev === "on_user_partial") return filters.partial;
    if (ev === "on_assistant_token" || ev === "assistant_token") return filters.tok;
    if (ev === "on_user_utterance" || ev === "on_user_speech_start" || ev === "on_user_stop_intent" ||
        ev === "on_assistant_sentence" || ev === "on_assistant_done") return filters.utt;
    if (ev === "on_error" || ev === "error") return filters.error;
    return true;
  }

  // ── mode switcher ────────────────────────────────────────────────
  const modeBtn = $("modeBtn");
  const modeMenu = $("modeMenu");
  const modeIcon = $("modeIcon");
  const modeLabel = $("modeLabel");
  let modesCache = [];
  let currentModeName = null;

  function renderModeMenu() {
    modeMenu.innerHTML = "";
    modesCache.forEach((m) => {
      const item = document.createElement("button");
      item.className = "mode-item" + (m.name === currentModeName ? " current" : "");
      item.innerHTML =
        '<span class="label"><span>' + escape(m.icon || "•") + "</span><span>" +
        escape(m.display_name || m.name) + "</span></span>" +
        (m.description ? '<span class="desc">' + escape(m.description) + "</span>" : "");
      item.addEventListener("click", async () => {
        modeMenu.classList.add("hidden");
        if (m.name === currentModeName) return;
        try {
          await post("/api/control/mode", { name: m.name });
          showToast("Switched to " + (m.display_name || m.name), "success");
        } catch (e) { /* post already toasts */ }
      });
      modeMenu.appendChild(item);
    });
  }

  function setCurrentMode(name) {
    currentModeName = name;
    const m = modesCache.find((x) => x.name === name);
    if (m) {
      modeIcon.textContent = m.icon || "•";
      modeLabel.textContent = m.display_name || m.name;
    } else if (name) {
      modeIcon.textContent = "•";
      modeLabel.textContent = name;
    }
    renderModeMenu();
  }

  async function refreshModes() {
    try {
      const r = await fetch("/api/modes");
      if (!r.ok) throw new Error("HTTP " + r.status);
      modesCache = await r.json();
      const cur = modesCache.find((m) => m.current);
      setCurrentMode(cur ? cur.name : currentModeName);
    } catch (e) { /* silent: dashboard works without modes */ }
  }
  modeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    modeMenu.classList.toggle("hidden");
  });
  document.addEventListener("click", (e) => {
    if (!modeMenu.contains(e.target) && e.target !== modeBtn) {
      modeMenu.classList.add("hidden");
    }
  });
  refreshModes();
  // Event-driven now: on `mode_registered` event the backend tells us to
  // refresh. No periodic polling.

  // ── mode-settings panel ─────────────────────────────────────────
  const msPanel = $("modeSettingsPanel");
  const msBtn = $("modeSettingsBtn");
  const msMode = $("msMode");
  const msSystemPrompt = $("msSystemPrompt");
  const msSystemPromptDefault = $("msSystemPromptDefault");
  const msTemperature = $("msTemperature");
  // Track per-field initial values to compute the diff at Save time.
  let msInitial = {};

  function _toFormVal(v) {
    if (v === null || v === undefined) return "";
    if (typeof v === "boolean") return v ? "true" : "false";
    return String(v);
  }

  async function loadModeOverrides(name) {
    if (!name) return;
    try {
      const r = await fetch("/api/modes/" + encodeURIComponent(name) + "/overrides");
      if (!r.ok) throw new Error("HTTP " + r.status);
      const data = await r.json();
      msMode.textContent = (data.icon || "") + " " + (data.display_name || data.name);
      const eff = data.effective || {};
      const def = data.class_default || {};
      const cur = data.current_override || {};
      // Pre-fill with current override only (empty = inherit).
      msSystemPrompt.value = cur.system_prompt != null ? String(cur.system_prompt) : "";
      msSystemPrompt.placeholder = def.system_prompt ? "(继承类默认)" : "(无默认)";
      msSystemPromptDefault.textContent = def.system_prompt || "(无)";
      msTemperature.value = cur.temperature != null ? String(cur.temperature) : "";
      msInitial = {
        system_prompt: msSystemPrompt.value,
        temperature: msTemperature.value,
      };
    } catch (e) {
      showToast("加载失败: " + e.message, "error");
    }
  }

  msBtn.addEventListener("click", async () => {
    if (msPanel.classList.contains("hidden")) {
      msPanel.classList.remove("hidden");
      await loadModeOverrides(currentModeName);
      // Section moved into this panel: load its state too.
      try { if (typeof loadAgentSettings === "function") await loadAgentSettings(); } catch (e) {}
    } else {
      msPanel.classList.add("hidden");
    }
  });
  $("msClose").addEventListener("click", () => msPanel.classList.add("hidden"));

  function _parseTempOrNull(s) {
    if (s === "" || s === null || s === undefined) return null;
    const n = parseFloat(s);
    return Number.isFinite(n) ? n : null;
  }
  $("msSave").addEventListener("click", async () => {
    if (!currentModeName) return;
    // Only send keys the user actually edited.
    const body = {};
    if (msSystemPrompt.value !== msInitial.system_prompt) {
      // Empty string is a legitimate override ("no system message");
      // sending null would reset to class default. The dedicated
      // Reset button below sends null explicitly.
      body.system_prompt = msSystemPrompt.value;
    }
    if (msTemperature.value !== msInitial.temperature) {
      body.temperature = _parseTempOrNull(msTemperature.value);
    }
    if (Object.keys(body).length === 0) {
      showToast("没有改动", "");
      return;
    }
    try {
      const r = await fetch(
        "/api/modes/" + encodeURIComponent(currentModeName) + "/overrides",
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      );
      if (!r.ok) throw new Error("HTTP " + r.status);
      const data = await r.json();
      showToast(
        data.persisted ? "已保存（已写入 yaml）" : "运行时已生效，未持久化",
        "success"
      );
      await loadModeOverrides(currentModeName);
    } catch (e) {
      showToast("保存失败: " + e.message, "error");
    }
  });

  $("msReset").addEventListener("click", async () => {
    if (!currentModeName) return;
    const body = { system_prompt: null, temperature: null };
    try {
      const r = await fetch(
        "/api/modes/" + encodeURIComponent(currentModeName) + "/overrides",
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      );
      if (!r.ok) throw new Error("HTTP " + r.status);
      const data = await r.json();
      showToast(
        data.persisted ? "已恢复默认（已写入 yaml）" : "已恢复默认（未持久化）",
        "success"
      );
      await loadModeOverrides(currentModeName);
    } catch (e) {
      showToast("重置失败: " + e.message, "error");
    }
  });

  // ── mode-settings advanced toggle (persist via localStorage) ────
  const msAdvancedToggle = $("msAdvancedToggle");
  const msAdvancedBody = $("msAdvancedBody");
  function applyAdvancedState() {
    const open = localStorage.getItem("ms_advanced_open") === "1";
    msAdvancedBody.classList.toggle("hidden", !open);
    msAdvancedToggle.textContent = (open ? "▴ 高级" : "▾ 高级");
  }
  msAdvancedToggle.addEventListener("click", () => {
    const wasOpen = !msAdvancedBody.classList.contains("hidden");
    localStorage.setItem("ms_advanced_open", wasOpen ? "0" : "1");
    applyAdvancedState();
  });
  applyAdvancedState();

  // Extend loadModeOverrides to also seed max_history + barge_in fields.
  // (We monkey-patch by wrapping the original implementation via a hook
  // on /api/.../overrides response.)
  const msMaxHistory = $("msMaxHistory");
  const msBargeIn = $("msBargeIn");
  const _origLoadModeOverrides = loadModeOverrides;
  loadModeOverrides = async function (name) {
    await _origLoadModeOverrides(name);
    if (!name) return;
    try {
      const r = await fetch("/api/modes/" + encodeURIComponent(name) + "/overrides");
      if (!r.ok) return;
      const data = await r.json();
      const cur = data.current_override || {};
      msMaxHistory.value = cur.max_history != null ? String(cur.max_history) : "";
      if (cur.barge_in_enabled === true) msBargeIn.value = "true";
      else if (cur.barge_in_enabled === false) msBargeIn.value = "false";
      else msBargeIn.value = "";
      msInitial.max_history = msMaxHistory.value;
      msInitial.barge_in_enabled = msBargeIn.value;
    } catch (e) { /* silent */ }
  };

  // Extend Save to send max_history + barge_in_enabled if changed.
  const _origMsSave = $("msSave").onclick;
  // Replace the existing Save handler with one that includes advanced fields.
  // We use addEventListener already on the button; rather than reorder, hook
  // a fresh handler that swallows the original and rebuilds the diff payload.
  const msSaveBtn = $("msSave");
  // Remove previous registered handler is impractical; instead let original
  // run first, then send a follow-up patch for advanced keys.
  msSaveBtn.addEventListener("click", async () => {
    if (!currentModeName) return;
    const advBody = {};
    if (msMaxHistory.value !== (msInitial.max_history || "")) {
      const s = msMaxHistory.value;
      advBody.max_history = (s === "") ? null : parseInt(s, 10);
      if (advBody.max_history !== null && !Number.isFinite(advBody.max_history)) {
        showToast("max_history 必须是整数", "error"); return;
      }
    }
    if (msBargeIn.value !== (msInitial.barge_in_enabled || "")) {
      advBody.barge_in_enabled = (msBargeIn.value === "") ? null
        : (msBargeIn.value === "true");
    }
    if (Object.keys(advBody).length === 0) return;
    try {
      const r = await fetch(
        "/api/modes/" + encodeURIComponent(currentModeName) + "/overrides",
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(advBody) }
      );
      if (!r.ok) throw new Error("HTTP " + r.status);
    } catch (e) { showToast("高级保存失败: " + e.message, "error"); }
  });

  // ── errors clear button ─────────────────────────────────────────
  $("btnErrClear").addEventListener("click", async (e) => {
    e.stopPropagation();
    try {
      await post("/api/errors/clear");
      errors = []; renderErrors();
      showToast("Errors cleared", "success");
    } catch (_) {}
  });

  // ── agent-settings card (#4) ────────────────────────────────────
  const agentSettingsBody = $("agentSettingsBody");
  const btnAgentSettingsToggle = $("btnAgentSettingsToggle");
  const setPipelineMode = $("setPipelineMode");
  const setSleepTimeout = $("setSleepTimeout");
  const setStopWords = $("setStopWords");

  async function loadAgentSettings() {
    try {
      const r = await fetch("/api/agent/settings");
      if (!r.ok) throw new Error("HTTP " + r.status);
      const d = await r.json();
      setPipelineMode.value = d.pipeline_mode || "always_on";
      setSleepTimeout.value = d.sleep_timeout_s != null ? String(d.sleep_timeout_s) : "30";
      setStopWords.value = (d.stop_words || []).join("\n");
    } catch (e) { showToast("加载设置失败: " + e.message, "error"); }
  }
  function toggleAgentSettings() {
    const hidden = agentSettingsBody.classList.toggle("hidden");
    if (btnAgentSettingsToggle) btnAgentSettingsToggle.classList.toggle("open", !hidden);
    if (!hidden) loadAgentSettings();
  }
  if (btnAgentSettingsToggle) {
    btnAgentSettingsToggle.addEventListener("click", (e) => { e.stopPropagation(); toggleAgentSettings(); });
    const h3 = $("agentSettingsCard")?.querySelector("h3");
    if (h3) {
      h3.addEventListener("click", (e) => {
        if (e.target.id === "btnAgentSettingsToggle") return;
        toggleAgentSettings();
      });
    }
  }
  $("btnAgentSettingsReload").addEventListener("click", loadAgentSettings);
  $("btnAgentSettingsSave").addEventListener("click", async () => {
    const stop_words = setStopWords.value.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
    const sleep_timeout_s = parseFloat(setSleepTimeout.value);
    if (!Number.isFinite(sleep_timeout_s) || sleep_timeout_s < 0) {
      showToast("sleep_timeout_s 无效", "error"); return;
    }
    const body = {
      pipeline_mode: setPipelineMode.value,
      sleep_timeout_s,
      stop_words,
    };
    try {
      const r = await fetch("/api/agent/settings", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        let msg = "HTTP " + r.status;
        try { const j = await r.json(); if (j.error) msg = j.error; } catch (_) {}
        throw new Error(msg);
      }
      const data = await r.json();
      showToast(
        data.persisted ? "已保存（已写入 yaml）" : "运行时已生效，未持久化",
        "success",
      );
    } catch (e) { showToast("保存失败: " + e.message, "error"); }
  });

  // ── translator card (NLLB Phase 2b) ──────────────────────────────
  const translatorBody = $("translatorBody");
  const btnTranslatorToggle = $("btnTranslatorToggle");
  const setTranslatorTgtLang = $("setTranslatorTgtLang");
  const translatorBackendBadge = $("translatorBackendBadge");
  const translatorSrcLangText = $("translatorSrcLangText");
  const btnTranslatorSave = $("btnTranslatorSave");
  const btnTranslatorReload = $("btnTranslatorReload");
  let translatorLoaded = false;

  function applyTranslatorRuntime(d) {
    if (!d || typeof d !== "object") return;
    const backend = String(d.backend || "noop");
    if (translatorBackendBadge) {
      translatorBackendBadge.textContent = backend;
      translatorBackendBadge.className = "muted small " + backend;
    }
    if (translatorSrcLangText) {
      translatorSrcLangText.textContent = String(d.src_lang || "–");
    }
    if (setTranslatorTgtLang) {
      const targets = Array.isArray(d.supported_targets) ? d.supported_targets : [];
      const current = String(d.tgt_lang || "");
      const prevSel = setTranslatorTgtLang.value;
      setTranslatorTgtLang.innerHTML = "";
      targets.forEach((t) => {
        if (!t || !t.code) return;
        const opt = document.createElement("option");
        opt.value = String(t.code);
        opt.textContent = (t.name ? `${t.name} (${t.code})` : String(t.code));
        setTranslatorTgtLang.appendChild(opt);
      });
      // Prefer payload tgt_lang; fall back to previous selection if still present.
      const wanted = current || prevSel;
      if (wanted) setTranslatorTgtLang.value = wanted;
    }
    const noop = backend === "noop";
    if (setTranslatorTgtLang) setTranslatorTgtLang.disabled = noop;
    if (btnTranslatorSave) {
      btnTranslatorSave.disabled = noop;
      btnTranslatorSave.title = noop ? "translator_backend 当前为 noop，不会翻译" : "";
    }
  }

  async function loadTranslatorRuntime() {
    try {
      const r = await fetch("/api/translator/runtime");
      if (!r.ok) throw new Error("HTTP " + r.status);
      const d = await r.json();
      applyTranslatorRuntime(d);
      translatorLoaded = true;
    } catch (e) {
      showToast("加载翻译设置失败: " + e.message, "error");
    }
  }

  async function saveTranslatorRuntime() {
    if (!setTranslatorTgtLang) return;
    const tgt = setTranslatorTgtLang.value;
    if (!tgt) { showToast("请先选择目标语种", "error"); return; }
    try {
      const r = await fetch("/api/translator/runtime", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tgt_lang: tgt }),
      });
      if (!r.ok) {
        let msg = "HTTP " + r.status;
        try { const j = await r.json(); if (j.error) msg = j.error; } catch (_) {}
        throw new Error(msg);
      }
      const data = await r.json();
      const label = data.tgt_lang || tgt;
      const note = data.persisted ? "已写入 yaml" : "运行时已生效，未持久化";
      showToast("已切换到 " + label + "（" + note + "）", "success");
      if (data.persist_error) showToast("持久化失败: " + data.persist_error, "error");
    } catch (e) {
      showToast("保存翻译设置失败: " + e.message, "error");
    }
  }

  function toggleTranslatorCard() {
    if (!translatorBody) return;
    const hidden = translatorBody.classList.toggle("hidden");
    if (btnTranslatorToggle) btnTranslatorToggle.classList.toggle("open", !hidden);
    if (!hidden && !translatorLoaded) loadTranslatorRuntime();
  }
  if (btnTranslatorToggle) {
    btnTranslatorToggle.addEventListener("click", (e) => { e.stopPropagation(); toggleTranslatorCard(); });
    const head = $("translatorCard")?.querySelector(".ms-section-head");
    if (head) {
      head.addEventListener("click", (e) => {
        if (e.target.id === "btnTranslatorToggle") return;
        toggleTranslatorCard();
      });
    }
  }
  if (btnTranslatorReload) btnTranslatorReload.addEventListener("click", loadTranslatorRuntime);
  if (btnTranslatorSave) btnTranslatorSave.addEventListener("click", saveTranslatorRuntime);

  // ── subtitle bubble (NLLB Phase 2b) ──────────────────────────────
  function addSubtitleBubble(payload) {
    const p = payload || {};
    const original = String(p.original || "");
    const translated = String(p.translated || "");
    const srcTag = String(p.detected_language || p.src_lang || "");
    const tgtTag = String(p.tgt_lang || "");
    const b = document.createElement("div");
    b.className = "bubble subtitle";
    // Build via DOM nodes (text-only) to avoid any innerHTML user-input path.
    const roleSpan = document.createElement("span");
    roleSpan.className = "role";
    roleSpan.textContent = "翻译";
    const origDiv = document.createElement("div");
    origDiv.className = "sub-orig muted small";
    origDiv.textContent = original;
    if (srcTag) {
      const tag = document.createElement("span");
      tag.className = "sub-tag";
      tag.textContent = " " + srcTag;
      origDiv.appendChild(tag);
    }
    const transDiv = document.createElement("div");
    transDiv.className = "sub-trans";
    transDiv.textContent = translated;
    if (tgtTag) {
      const tag = document.createElement("span");
      tag.className = "sub-tag";
      tag.textContent = " " + tgtTag;
      transDiv.appendChild(tag);
    }
    b.appendChild(roleSpan);
    b.appendChild(origDiv);
    b.appendChild(transDiv);
    chatEl.appendChild(b);
    chatEl.parentElement.scrollTop = chatEl.parentElement.scrollHeight;
    return b;
  }

  // ── tabs ─────────────────────────────────────────────────────────
  document.querySelectorAll(".tab").forEach((t) => {
    t.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      $("tab-" + t.dataset.tab).classList.add("active");
      if (t.dataset.tab === "history") refreshHistory();
    });
  });

  // ── filter checkboxes ────────────────────────────────────────────
  document.querySelectorAll("#filters input").forEach((cb) => {
    cb.addEventListener("change", () => { filters[cb.dataset.f] = cb.checked; });
  });

  // ── controls ─────────────────────────────────────────────────────
  async function post(path, body) {
    try {
      const opts = { method: "POST", headers: { "Content-Type": "application/json" } };
      if (body !== undefined) opts.body = JSON.stringify(body);
      const r = await fetch(path, opts);
      if (!r.ok) throw new Error("HTTP " + r.status);
      return await r.json();
    } catch (e) {
      showToast("Failed: " + e.message, "error");
      throw e;
    }
  }
  $("btnReconnect").addEventListener("click", async () => {
    await post("/api/control/reconnect");
    showToast("Reconnect requested", "success");
  });
  $("btnRestartMic").addEventListener("click", async () => {
    await post("/api/control/restart_mic");
    showToast("Mic capture restarted", "success");
  });
  $("btnAbort").addEventListener("click", async () => {
    await post("/api/control/abort");
    showToast("TTS aborted", "success");
  });
  $("btnSend").addEventListener("click", sendText);
  $("sendInput").addEventListener("keydown", (e) => { if (e.key === "Enter") sendText(); });
  async function sendText() {
    const inp = $("sendInput");
    const text = inp.value.trim();
    if (!text) return;
    await post("/api/control/send_text", { text });
    inp.value = "";
    showToast("Sent", "success");
  }

  // menu
  $("btnMenu").addEventListener("click", () => { $("menu").classList.toggle("hidden"); });
  $("btnClear").addEventListener("click", () => {
    eventsEl.innerHTML = ""; chatEl.innerHTML = "";
    pendingPartialBubble = null; pendingAssistantBubble = null; pendingAssistantText = "";
    $("menu").classList.add("hidden");
  });
  async function clearSessionContext() {
    if (!window.confirm("清空 LLM 上下文？当前模式和 system prompt 设置会保留。")) return;
    const r = await fetch("/api/session/clear", { method: "POST" });
    let data = {};
    try { data = await r.json(); } catch (_) {}
    if (!r.ok || data.ok === false) throw new Error(data.error || ("HTTP " + r.status));
    await refreshHistory();
    showToast("已清空上下文 (" + (data.cleared || 0) + " 条)", "success");
  }
  $("btnClearContextMenu").addEventListener("click", async () => {
    $("menu").classList.add("hidden");
    try { await clearSessionContext(); } catch (e) { showToast("清空上下文失败: " + e.message, "error"); }
  });
  $("btnPause").addEventListener("click", () => {
    paused = !paused;
    $("btnPause").textContent = paused ? "Resume" : "Pause";
    $("menu").classList.add("hidden");
  });
  $("btnRefreshHist").addEventListener("click", refreshHistory);
  $("btnClearHist").addEventListener("click", async () => {
    try { await clearSessionContext(); } catch (e) { showToast("清空上下文失败: " + e.message, "error"); }
  });

  // pipeline_mode controls — visibility set after snapshot arrives.
  $("btnWake").addEventListener("click", async () => {
    await post("/api/control/wake");
    showToast("Wake requested", "success");
  });
  $("btnSleep").addEventListener("click", async () => {
    await post("/api/control/sleep");
    showToast("Sleep requested", "success");
  });
  const pttBtn = $("btnPTT");
  const pttDown = async () => {
    pttBtn.classList.add("active");
    try { await post("/api/control/ptt/start"); } catch (_) {}
  };
  const pttUp = async () => {
    pttBtn.classList.remove("active");
    try { await post("/api/control/ptt/end"); } catch (_) {}
  };
  pttBtn.addEventListener("mousedown", pttDown);
  pttBtn.addEventListener("mouseup", pttUp);
  pttBtn.addEventListener("mouseleave", (e) => { if (pttBtn.classList.contains("active")) pttUp(); });
  pttBtn.addEventListener("touchstart", (e) => { e.preventDefault(); pttDown(); });
  pttBtn.addEventListener("touchend", (e) => { e.preventDefault(); pttUp(); });

  function applyPipelineMode(mode) {
    const badge = $("pipelineBadge");
    const wakeBtn = $("btnWake");
    const sleepBtn = $("btnSleep");
    const pttB = $("btnPTT");
    $("pipelineModeText").textContent = mode || "always_on";
    // Reset
    wakeBtn.classList.add("hidden");
    sleepBtn.classList.add("hidden");
    pttB.classList.add("hidden");
    badge.classList.add("hidden");
    badge.textContent = "";
    if (mode === "wake_word") {
      wakeBtn.classList.remove("hidden");
      sleepBtn.classList.remove("hidden");
    } else if (mode === "push_to_talk") {
      pttB.classList.remove("hidden");
      sleepBtn.classList.remove("hidden");
    } else {
      // always_on
      badge.classList.remove("hidden");
      badge.textContent = "常驻监听";
    }
  }

  // ── chat rendering ───────────────────────────────────────────────
  function addBubble(kind, text, opts) {
    const b = document.createElement("div");
    b.className = "bubble " + kind;
    const role = (opts && opts.role) || (kind === "user" ? "you" : kind === "assistant" ? "assistant" : kind);
    b.innerHTML = '<span class="role">' + escape(role) + '</span>' + escape(text);
    chatEl.appendChild(b);
    chatEl.parentElement.scrollTop = chatEl.parentElement.scrollHeight;
    return b;
  }
  function updateBubble(bubble, text) {
    const role = bubble.querySelector(".role").outerHTML;
    bubble.innerHTML = role + escape(text);
  }

  // ── event rendering ──────────────────────────────────────────────
  function pushEvent(ts, ev, data) {
    if (!shouldRender(ev)) return;
    const row = document.createElement("div");
    row.className = "event-row";
    const dataStr = (data === null || data === undefined) ? "" :
      (typeof data === "string" ? data : JSON.stringify(data));
    row.innerHTML = '<span class="ts">' + fmtTs(ts) + '</span>' +
      '<span class="ev ev-' + ev + '">' + ev + '</span>' +
      '<span>' + escape(dataStr) + '</span>';
    eventsEl.appendChild(row);
    while (eventsEl.childNodes.length > MAX_EVENT_ROWS) eventsEl.removeChild(eventsEl.firstChild);
    eventsEl.scrollTop = eventsEl.scrollHeight;
  }

  // ── state pill ───────────────────────────────────────────────────
  const STATE_LABEL = {
    idle: "待机",
    listening: "聆听中",
    thinking: "思考中",
    speaking: "说话中",
    barged_in: "被打断",
    sleeping: "休眠中",
  };
  function applyState(s) {
    statePill.className = "state-pill state-" + s;
    stateText.textContent = STATE_LABEL[s] || s.toUpperCase();
  }

  // ── mic chart ────────────────────────────────────────────────────
  function pushRms(v) {
    rmsSamples.push(v); if (rmsSamples.length > 60) rmsSamples.shift();
    if (!rafQueued) { rafQueued = true; requestAnimationFrame(drawRms); }
  }
  function drawRms() {
    rafQueued = false;
    const ctx = rmsChart.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const cssW = rmsChart.clientWidth || rmsChart.width;
    const cssH = rmsChart.clientHeight || rmsChart.height;
    const nextW = Math.max(1, Math.round(cssW * dpr));
    const nextH = Math.max(1, Math.round(cssH * dpr));
    if (rmsChart.width !== nextW || rmsChart.height !== nextH) {
      rmsChart.width = nextW;
      rmsChart.height = nextH;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = cssW, H = cssH;
    ctx.clearRect(0, 0, W, H);
    const bw = W / rmsSamples.length;
    const thr = parseFloat(rmsThr.textContent) || 0.012;
    for (let i = 0; i < rmsSamples.length; i++) {
      const h = Math.min(1, rmsSamples[i] / 0.2) * H;
      ctx.fillStyle = rmsSamples[i] >= thr ? "#82c91e" : "#7a8597";
      ctx.fillRect(i * bw, H - h, Math.max(1, bw - 1), h);
    }
    // threshold line
    const ty = H - Math.min(1, thr / 0.2) * H;
    ctx.strokeStyle = "rgba(92,219,211,0.5)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, ty); ctx.lineTo(W, ty); ctx.stroke();
  }

  // ── latency ──────────────────────────────────────────────────────
  function setLatency(kind, ms) {
    const cell = $("lat-" + kind);
    const hist = $("lath-" + kind);
    if (!cell) return;
    const v = Math.round(ms);
    cell.textContent = v + "ms";
    latHist[kind].push(v);
    if (latHist[kind].length > LAT_KEEP) latHist[kind].shift();
    hist.textContent = "‹ " + latHist[kind].join(" ") + " ›";
    // Sparkline push.
    const buf = sparkData[kind];
    if (buf) {
      buf.push(v);
      if (buf.length > SPARK_MAX) buf.shift();
      sparkDirty[kind] = true;
      if (!sparkRafQueued) { sparkRafQueued = true; requestAnimationFrame(drawSparks); }
    }
  }

  function drawSparks() {
    sparkRafQueued = false;
    Object.keys(sparkData).forEach((kind) => {
      if (!sparkDirty[kind]) return;
      sparkDirty[kind] = false;
      drawSpark(kind);
    });
  }
  function drawSpark(kind) {
    const cv = $("spark-" + kind);
    if (!cv) return;
    const ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const buf = sparkData[kind];
    // The canvas is authored at 2x its CSS box (128x40 -> 64x20), so strokes
    // are doubled here to land on 1px / 1.5px once scaled down.
    if (!buf || buf.length === 0) {
      ctx.strokeStyle = "#3a4258"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
      return;
    }
    const maxV = Math.max(1, ...buf);
    ctx.strokeStyle = "#5cdbd3"; ctx.lineWidth = 3;
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.beginPath();
    const step = buf.length > 1 ? W / (buf.length - 1) : W;
    for (let i = 0; i < buf.length; i++) {
      const x = i * step;
      const y = H - (buf[i] / maxV) * (H - 6) - 3;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  // Paint initial baselines.
  Object.keys(sparkData).forEach(drawSpark);

  // ── errors ───────────────────────────────────────────────────────
  let errors = [];
  // Map typed-error categories to CSS modifier classes that drive colour.
  // Falls back to "error-gray" for unknown / legacy string payloads.
  const ERROR_TYPE_COLOR = {
    llm_timeout:       "error-orange",
    degraded:          "error-orange",
    llm_failure:       "error-red",
    llm_unavailable:   "error-red",
    llm_stream_error:  "error-red",
    slv_error:         "error-red",
    input_too_long:    "error-yellow",
  };
  // Pretty labels for the type pill so operators don't see snake_case.
  const ERROR_TYPE_LABEL = {
    llm_timeout:       "TIMEOUT",
    degraded:          "DEGRADED",
    llm_failure:       "LLM",
    llm_unavailable:   "UNAVAILABLE",
    llm_stream_error:  "STREAM",
    slv_error:         "SLV",
    input_too_long:    "INPUT",
    unknown:           "ERROR",
  };
  function pushError(ts, msg, stack, type, exc_class) {
    errors.push({ ts, msg, stack, type, exc_class });
    if (errors.length > 50) errors.shift();
    renderErrors();
  }
  function renderErrors() {
    errCount.textContent = errors.length;
    errDot.classList.toggle("hidden", errors.length === 0);
    // Red frame only while there is something to look at.
    errCount.closest(".card").classList.toggle("has-errors", errors.length > 0);
    errList.innerHTML = "";
    for (let i = errors.length - 1; i >= 0; i--) {
      const e = errors[i];
      const div = document.createElement("div");
      const colorCls = ERROR_TYPE_COLOR[e.type] || "error-gray";
      div.className = "err-item " + colorCls;
      const label = ERROR_TYPE_LABEL[e.type] || ERROR_TYPE_LABEL.unknown;
      const typeChip = '<span class="err-type">' + escape(label) + '</span>';
      div.innerHTML = '<span class="ts">' + fmtTs(e.ts) + '</span>' +
        typeChip + escape(e.msg) +
        (e.exc_class ? '<span class="err-cls">' + escape(e.exc_class) + '</span>' : "") +
        (e.stack ? '<div class="stack">' + escape(e.stack) + '</div>' : "");
      div.addEventListener("click", () => div.classList.toggle("expanded"));
      errList.appendChild(div);
    }
  }
  // Normalise both legacy (string) and typed (dict) error payloads into
  // a single {msg, type, exc_class} shape for pushError.
  function normaliseErrorPayload(data) {
    if (data && typeof data === "object" && !Array.isArray(data)) {
      return {
        msg: String(data.message || data.msg || JSON.stringify(data)),
        type: String(data.type || "unknown"),
        exc_class: String(data.exc_class || ""),
      };
    }
    return {
      msg: typeof data === "string" ? data : String(data),
      type: "unknown",
      exc_class: "",
    };
  }

  // ── uptime ───────────────────────────────────────────────────────
  setInterval(() => {
    const secs = Math.floor((Date.now() - sessionStartMs) / 1000);
    const m = Math.floor(secs / 60), s = secs % 60;
    uptimeEl.textContent = (m > 0 ? m + "m" : "") + s + "s";
  }, 1000);

  // ── history fetch ────────────────────────────────────────────────
  async function refreshHistory() {
    try {
      const r = await fetch("/api/session/history");
      if (!r.ok) throw new Error("HTTP " + r.status);
      const items = await r.json();
      historyEl.innerHTML = "";
      items.forEach((m) => {
        const div = document.createElement("div");
        div.className = "hist-item " + (m.role || "");
        div.innerHTML = '<div class="role">' + escape(m.role || "") + '</div>' +
          '<div>' + escape(m.content || "") + '</div>';
        historyEl.appendChild(div);
      });
    } catch (e) { showToast("History fetch failed: " + e.message, "error"); }
  }

  // ── handle incoming WS message ───────────────────────────────────
  function handle(msg) {
    if (paused) return;
    const ev = msg.event, data = msg.data, ts = msg.ts;

    pushEvent(ts, ev, data);

    if (ev === "stats") {
      const d = data || {};
      const s = d.slv_ws_state || "unknown";
      wsStateEl.textContent = s;
      wsDot.className = "dot " + (s === "open" ? "open" : (s === "closed" ? "closed" : "unknown"));
      return;
    }

    if (ev === "on_state_change") {
      applyState((data && data.state) || "idle");
      return;
    }

    if (ev === "on_slv_reconnect") {
      rcCount.textContent = (data && data.count) || 0;
      return;
    }

    if (ev === "on_mic_rms") {
      const d = data || {};
      pushRms(d.rms || 0);
      rmsThr.textContent = (d.threshold || 0).toFixed(3);
      vadStateEl.textContent = d.state || "?";
      return;
    }

    if (ev === "on_user_partial") {
      const text = String(data || "");
      if (!pendingPartialBubble) pendingPartialBubble = addBubble("partial", text, { role: "you (partial)" });
      else updateBubble(pendingPartialBubble, text);
      return;
    }

    if (ev === "on_user_utterance") {
      pendingPartialBubble = null;
      addBubble("user", String(data || ""), { role: "you" });
      pendingAssistantBubble = null; pendingAssistantText = "";
      return;
    }

    if (ev === "on_user_stop_intent") {
      pendingPartialBubble = null;
      addBubble("stop", "⏹ " + String(data || ""), { role: "stop" });
      return;
    }

    if (ev === "on_assistant_token" || ev === "assistant_token") {
      const tok = String(data || "");
      pendingAssistantText += tok;
      if (!pendingAssistantBubble) pendingAssistantBubble = addBubble("assistant", pendingAssistantText, { role: "assistant" });
      else updateBubble(pendingAssistantBubble, pendingAssistantText);
      return;
    }

    if (ev === "on_assistant_done") {
      pendingAssistantBubble = null; pendingAssistantText = "";
      ttsRow.classList.add("inactive"); ttsText.textContent = "idle";
      return;
    }

    if (ev === "on_assistant_sentence_start" || ev === "on_assistant_sentence") {
      ttsRow.classList.remove("inactive");
      ttsText.textContent = String(data || "");
      ttsText.classList.remove("muted");
      return;
    }

    if (ev === "on_tts_audio_frame") {
      ttsRow.classList.remove("inactive");
      return;
    }

    if (ev === "on_llm_cache_metrics") {
      // Cache metrics no longer displayed on the dashboard; keep the
      // event reachable in the events tab for debugging only.
      return;
    }

    if (ev === "tts_metrics") {
      const d = data || {};
      if (typeof d.sentence_count === "number") ttsSentCount.textContent = d.sentence_count;
      if (typeof d.bytes_current === "number") ttsBytesCur.textContent = d.bytes_current;
      if (typeof d.last_duration_s === "number") ttsLastDur.textContent = d.last_duration_s.toFixed(1) + " 秒";
      return;
    }

    if (ev === "mode_registered") {
      // Backend says a new mode was registered after start; refresh dropdown.
      refreshModes();
      return;
    }

    if (ev === "on_agent_settings_change") {
      // Refresh agent-settings card if open and we didn't initiate this.
      if (!$("agentSettingsBody").classList.contains("hidden")) {
        loadAgentSettings();
      }
      return;
    }

    if (ev === "on_translator_runtime_change") {
      try { applyTranslatorRuntime(data); translatorLoaded = true; } catch (_) {}
      return;
    }

    if (ev === "on_translation") {
      try { addSubtitleBubble(data || {}); } catch (_) {}
      return;
    }

    if (ev === "errors_cleared") {
      errors = []; renderErrors();
      return;
    }

    if (ev === "on_mode_change") {
      const d = data || {};
      if (Array.isArray(modesCache) && d.name) {
        // Update cache "current" flag too.
        modesCache.forEach((m) => { m.current = (m.name === d.name); });
      }
      setCurrentMode(d.name || currentModeName);
      if (!msPanel.classList.contains("hidden")) {
        loadModeOverrides(currentModeName);
      }
      return;
    }

    if (ev === "snapshot") {
      const d = data || {};
      if (d.state) applyState(d.state);
      if (typeof d.reconnect_count === "number") rcCount.textContent = d.reconnect_count;
      if (Array.isArray(d.modes) && d.modes.length) {
        modesCache = d.modes;
      }
      if (d.mode) setCurrentMode(d.mode);
      applyPipelineMode(d.pipeline_mode || "always_on");
      if (Array.isArray(d.errors) && d.errors.length) {
        // Seed errors panel with anything that happened before this client connected.
        errors = d.errors.slice(-50).map((e) => ({
          ts: e.ts || Date.now(),
          msg: e.message || e.msg || String(e),
          stack: e.stack,
          type: e.type || "unknown",
          exc_class: e.exc_class || "",
        }));
        renderErrors();
      }
      if (d.llm_availability) renderLLMHealth(d.llm_availability);
      setPrefixCacheBadge(!!d.prefix_cache_disabled);
      return;
    }

    if (ev === "on_llm_availability_change") {
      renderLLMHealth(data || {});
      return;
    }

    if (ev === "on_prefix_cache_disabled") {
      setPrefixCacheBadge(true);
      showToast("prefix_cache 已禁用 (服务端不支持)", "");
      return;
    }

    if (ev === "on_session_trimmed") {
      const d = data || {};
      const dropped = (d.dropped_turns != null) ? d.dropped_turns : "?";
      const kept = (d.kept_turns != null) ? d.kept_turns : "?";
      showToast("Session 截断: 丢弃 " + dropped + " 轮 (保留 " + kept + ")", "");
      return;
    }

    if (ev === "latency") {
      const d = data || {};
      if (d.kind && typeof d.ms === "number") setLatency(d.kind, d.ms);
      return;
    }

    if (ev === "on_error" || ev === "error") {
      const n = normaliseErrorPayload(data);
      pushError(ts, n.msg, undefined, n.type, n.exc_class);
      return;
    }
  }

  // ── LLM health card ──────────────────────────────────────────────
  const LLM_STATE_LABEL = {
    healthy:    "✓ 正常",
    degraded:   "⚠ 降级",
    recovering: "◐ 恢复中",
    down:       "✗ 不可用",
    unknown:    "? 未知",
  };
  function renderLLMHealth(d) {
    const dot = $("llm-health-dot");
    const stateEl = $("llm-health-state");
    if (!dot || !stateEl) return;
    const state = (d && d.state) || "unknown";
    dot.className = "health-dot " + state;
    stateEl.textContent = LLM_STATE_LABEL[state] || LLM_STATE_LABEL.unknown;
    const okTs = d && d.last_ok_ts;
    $("llm-last-ok").textContent = okTs
      ? new Date(okTs * 1000).toLocaleTimeString()
      : "-";
    $("llm-failures").textContent = (d && d.consecutive_failures) || 0;
    const interval = d && d.probe_interval_s;
    $("llm-interval").textContent = interval ? interval + "s" : "-";
  }
  function setPrefixCacheBadge(on) {
    const el = $("llm-prefix-cache-badge");
    if (!el) return;
    el.classList.toggle("hidden", !on);
  }
  (function bindProbeBtn() {
    const btn = $("llm-probe-btn");
    if (!btn) return;
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try {
        const r = await fetch("/api/llm/probe", { method: "POST" });
        if (r.ok) {
          const d = await r.json();
          renderLLMHealth(d);
          showToast("已重新探测: " + (LLM_STATE_LABEL[d.state] || d.state), "success");
        } else {
          const e = await r.json().catch(() => ({}));
          showToast("探测失败: " + (e.error || r.status), "error");
        }
      } catch (e) {
        showToast("探测失败: " + e.message, "error");
      } finally {
        btn.disabled = false;
      }
    });
  })();

  // ── TTS 声音卡 + 声音克隆 ────────────────────────────────────────
  (function () {
    const sel = $("voiceSelect");
    const modelEl = $("voiceModel");
    const countEl = $("voiceCount");
    const reloadBtn = $("btnVoiceReload");
    const cloneBtn = $("btnVoiceClone");
    const hintEl = $("voiceHint");
    if (!sel) return;

    const panel = $("voiceClonePanel");
    const vcClose = $("vcClose");
    const vcCancel = $("vcCancel");
    const vcLabel = $("vcLabel");
    const vcRecBtn = $("vcRecBtn");
    const vcRecState = $("vcRecState");
    const vcPreview = $("vcPreview");
    const vcPreviewWrap = $("vcPreviewWrap");
    const vcUpload = $("vcUpload");
    const vcStatus = $("vcStatus");

    let currentSpeakerId = null;
    let speakers = [];
    let modelId = null;
    let supportsVoiceCloning = true;

    function setHint(msg, kind) {
      hintEl.textContent = msg || "";
      hintEl.style.color = kind === "error" ? "var(--err, #ef4444)" : "";
    }

    function applyCloneCapability() {
      // Show the clone button only when the active TTS backend advertises
      // voice cloning. Qwen3-CustomVoice ships built-in speakers and has no
      // speaker_encoder, so cloning is unsupported — grey out + tooltip.
      if (!cloneBtn) return;
      if (supportsVoiceCloning) {
        cloneBtn.classList.remove("hidden");
        cloneBtn.disabled = false;
        cloneBtn.removeAttribute("title");
        cloneBtn.style.opacity = "";
        cloneBtn.style.cursor = "";
      } else {
        // Keep visible but disabled so users see WHY it's gone (a hidden
        // button would be confusing — tooltip explains the gap).
        cloneBtn.classList.remove("hidden");
        cloneBtn.disabled = true;
        cloneBtn.title = "Voice cloning unavailable with current TTS model";
        cloneBtn.style.opacity = "0.5";
        cloneBtn.style.cursor = "not-allowed";
      }
    }

    function renderSpeakers() {
      sel.innerHTML = "";
      if (!speakers.length) {
        const opt = document.createElement("option");
        opt.textContent = "(无)";
        opt.disabled = true;
        sel.appendChild(opt);
        sel.disabled = true;
      } else {
        sel.disabled = false;
        for (const s of speakers) {
          const opt = document.createElement("option");
          opt.value = String(s.id);
          const lbl = s.label ? `${s.id} · ${s.label}` : `speaker ${s.id}`;
          opt.textContent = s.type === "embedding" ? `${lbl} (clone)` : lbl;
          sel.appendChild(opt);
        }
        if (currentSpeakerId != null) sel.value = String(currentSpeakerId);
      }
      countEl.textContent = String(speakers.length);
      modelEl.textContent = modelId || "–";
    }

    async function loadSpeakers() {
      try {
        const r = await fetch("/api/tts/speakers");
        if (r.status === 503) { setHint("TTS 未就绪", "error"); return; }
        if (!r.ok) { setHint("加载失败: HTTP " + r.status, "error"); return; }
        const j = await r.json();
        speakers = Array.isArray(j.speakers) ? j.speakers : [];
        modelId = j.model_id || null;
        // Server flag drives clone-UI gating. Default true so legacy servers
        // without the field keep the old behavior.
        supportsVoiceCloning = j.supports_voice_cloning !== false;
        if (currentSpeakerId == null && j.default_speaker_id != null) {
          currentSpeakerId = j.default_speaker_id;
        }
        renderSpeakers();
        applyCloneCapability();
        setHint("");
      } catch (e) { setHint("加载失败: " + e.message, "error"); }
    }

    async function loadRuntime() {
      try {
        const r = await fetch("/api/tts/runtime");
        if (!r.ok) return;
        const j = await r.json();
        const eff = (j && j.effective) || {};
        if (eff.speaker_id != null) {
          currentSpeakerId = eff.speaker_id;
          if (sel.options.length) sel.value = String(currentSpeakerId);
        }
      } catch (e) { /* ignore */ }
    }

    sel.addEventListener("change", async () => {
      const sid = parseInt(sel.value, 10);
      if (!Number.isFinite(sid)) return;
      try {
        const r = await fetch("/api/tts/runtime", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ speaker_id: sid }),
        });
        if (!r.ok) {
          const j = await r.json().catch(() => ({}));
          showToast("切换失败: " + (j.error || ("HTTP " + r.status)), "error");
          return;
        }
        currentSpeakerId = sid;
        showToast("已切换 speaker " + sid, "success");
      } catch (e) { showToast("切换失败: " + e.message, "error"); }
    });

    reloadBtn.addEventListener("click", loadSpeakers);

    // 克隆面板
    let mediaStream = null, audioCtx = null, procNode = null, srcNode = null;
    let recBuffers = [], recSampleRate = 0, recording = false;
    let wavBlob = null, recStartedAt = 0;

    function resetCloneUI() {
      vcLabel.value = "";
      vcStatus.textContent = "";
      vcStatus.style.color = "";
      vcRecState.textContent = "就绪";
      vcRecBtn.classList.remove("recording");
      vcRecBtn.textContent = "● 开始录音";
      vcPreviewWrap.classList.add("hidden");
      vcPreview.src = "";
      vcUpload.disabled = true;
      wavBlob = null;
      recBuffers = [];
    }
    function openClone() { resetCloneUI(); panel.classList.remove("hidden"); }
    function closeClone() { stopRecording().catch(() => {}); panel.classList.add("hidden"); }

    cloneBtn.addEventListener("click", () => {
      if (!supportsVoiceCloning) {
        showToast("当前 TTS backend 不支持克隆，请切到 MOSS", "error");
        return;
      }
      openClone();
    });
    vcClose.addEventListener("click", closeClone);
    vcCancel.addEventListener("click", closeClone);

    function encodeWAV(buffers, sampleRate) {
      let total = 0;
      for (const b of buffers) total += b.length;
      const flat = new Float32Array(total);
      let off = 0;
      for (const b of buffers) { flat.set(b, off); off += b.length; }
      const pcm = new Int16Array(flat.length);
      for (let i = 0; i < flat.length; i++) {
        let s = Math.max(-1, Math.min(1, flat[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      const byteLen = pcm.length * 2;
      const buf = new ArrayBuffer(44 + byteLen);
      const view = new DataView(buf);
      const ws = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
      ws(0, "RIFF");
      view.setUint32(4, 36 + byteLen, true);
      ws(8, "WAVE"); ws(12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      ws(36, "data");
      view.setUint32(40, byteLen, true);
      new Int16Array(buf, 44).set(pcm);
      return new Blob([buf], { type: "audio/wav" });
    }

    async function startRecording() {
      if (recording) return;
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false },
        });
      } catch (e) {
        vcStatus.style.color = "var(--err, #ef4444)";
        vcStatus.textContent = "麦克风权限被拒: " + e.message;
        return;
      }
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      recSampleRate = audioCtx.sampleRate;
      srcNode = audioCtx.createMediaStreamSource(mediaStream);
      procNode = audioCtx.createScriptProcessor(4096, 1, 1);
      recBuffers = [];
      procNode.onaudioprocess = (ev) => {
        const ch = ev.inputBuffer.getChannelData(0);
        recBuffers.push(new Float32Array(ch));
      };
      srcNode.connect(procNode);
      procNode.connect(audioCtx.destination);
      recording = true;
      recStartedAt = Date.now();
      vcRecBtn.classList.add("recording");
      vcRecBtn.textContent = "■ 停止录音";
      vcRecState.textContent = "录音中…";
      vcUpload.disabled = true;
      vcStatus.textContent = "";
    }

    async function stopRecording() {
      if (!recording) return;
      recording = false;
      try { procNode && procNode.disconnect(); } catch (e) {}
      try { srcNode && srcNode.disconnect(); } catch (e) {}
      try { mediaStream && mediaStream.getTracks().forEach(t => t.stop()); } catch (e) {}
      try { audioCtx && await audioCtx.close(); } catch (e) {}
      procNode = srcNode = mediaStream = audioCtx = null;
      const dur = (Date.now() - recStartedAt) / 1000;
      vcRecBtn.classList.remove("recording");
      vcRecBtn.textContent = "● 重新录音";
      vcRecState.textContent = `已录 ${dur.toFixed(1)}s`;
      if (!recBuffers.length) {
        vcStatus.style.color = "var(--err, #ef4444)";
        vcStatus.textContent = "没有捕获到音频";
        return;
      }
      wavBlob = encodeWAV(recBuffers, recSampleRate);
      vcPreview.src = URL.createObjectURL(wavBlob);
      vcPreviewWrap.classList.remove("hidden");
      vcUpload.disabled = !(vcLabel.value || "").trim();
    }

    vcRecBtn.addEventListener("click", () => {
      if (recording) stopRecording(); else startRecording();
    });
    vcLabel.addEventListener("input", () => {
      vcUpload.disabled = !((vcLabel.value || "").trim() && wavBlob);
    });

    vcUpload.addEventListener("click", async () => {
      const label = (vcLabel.value || "").trim();
      if (!label || !wavBlob) return;
      vcUpload.disabled = true;
      vcStatus.style.color = "";
      vcStatus.textContent = "提取声纹中…";
      try {
        const fd = new FormData();
        fd.append("file", wavBlob, "reference.wav");
        const r1 = await fetch("/api/tts/clone/embedding", { method: "POST", body: fd });
        if (r1.status === 501) {
          cloneBtn.classList.add("hidden");
          throw new Error("当前 TTS 后端不支持声音克隆");
        }
        if (!r1.ok) {
          const j = await r1.json().catch(() => ({}));
          throw new Error(j.error || ("HTTP " + r1.status));
        }
        const emb = await r1.json();
        vcStatus.textContent = "注册 speaker…";
        const r2 = await fetch("/api/tts/speakers/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            speaker_embedding_b64: emb.speaker_embedding_b64,
            label: label,
          }),
        });
        if (!r2.ok) {
          const j = await r2.json().catch(() => ({}));
          throw new Error(j.error || ("HTTP " + r2.status));
        }
        const reg = await r2.json();
        await loadSpeakers();
        sel.value = String(reg.speaker_id);
        sel.dispatchEvent(new Event("change"));
        showToast(`已注册 speaker ${reg.speaker_id}`, "success");
        closeClone();
      } catch (e) {
        vcStatus.style.color = "var(--err, #ef4444)";
        vcStatus.textContent = "失败: " + e.message;
      } finally {
        vcUpload.disabled = false;
      }
    });

    // Initial show — applyCloneCapability() will refine after first speakers
    // payload arrives (default supportsVoiceCloning=true keeps legacy behavior).
    cloneBtn.classList.remove("hidden");
    loadSpeakers().then(loadRuntime);
  })();

  // ── WS connection ────────────────────────────────────────────────
  let ws = null;
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(proto + "://" + location.host + "/ws");
    ws.onopen = () => { wsDot.classList.remove("closed"); /* server stats will fill */ };
    ws.onclose = () => { setTimeout(connect, 2000); };
    ws.onerror = () => { try { ws.close(); } catch (e) {} };
    ws.onmessage = (e) => { try { handle(JSON.parse(e.data)); } catch (err) { /* ignore */ } };
  }
  connect();
})();

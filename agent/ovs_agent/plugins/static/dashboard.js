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
  let toastTimer = null;
  function showToast(msg, kind, ms) {
    toastEl.textContent = msg;
    toastEl.className = "toast " + (kind || "");
    // A newer toast must not be hidden by the previous one's timer.
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toastEl.classList.add("hidden"); }, ms || 2000);
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

  function closeSettingsPanel() {
    if (msPanel.classList.contains("hidden")) return;
    msPanel.classList.add("hidden");
    msBtn.focus();
  }

  msBtn.addEventListener("click", async () => {
    if (msPanel.classList.contains("hidden")) {
      msPanel.classList.remove("hidden");
      activateSettingsTab(desiredTab, { force: true });
      if (translatorProbe === "idle" || translatorProbe === "failed") loadTranslatorRuntime({ silent: true });
      await loadModeOverrides(currentModeName);
      // Section moved into this panel: load its state too.
      try {
        await Promise.all([loadAgentSettings(), loadWakewordRuntime(), loadLlmConfig()]);
      } catch (e) {}
    } else {
      msPanel.classList.add("hidden");
    }
  });
  $("msClose").addEventListener("click", closeSettingsPanel);
  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape" || msPanel.classList.contains("hidden")) return;
    // The MCP and voice-clone modals sit above the panel and handle their own Escape.
    if (!$("mcpModal").classList.contains("hidden") || !$("voiceClonePanel").classList.contains("hidden")) return;
    closeSettingsPanel();
  });

  // ── LLM endpoint & model ────────────────────────────────────────
  // Global endpoint config, saved separately from the per-mode form above.
  // The key never leaves the server: GET returns only a mask, and sending
  // the mask back means "keep the saved key" (the server drops it when the
  // origin changes).
  const LLM_KEY_MASK = "●●●●●●";
  const llmBackend = $("llmBackend");
  const llmBaseUrl = $("llmBaseUrl");
  const llmModel = $("llmModel");
  const llmApiKey = $("llmApiKey");
  const llmTestResult = $("llmTestResult");
  const llmRestartAlert = $("llmRestartAlert");
  const btnLlmTest = $("btnLlmTest");
  const btnLlmSave = $("btnLlmSave");
  const btnLlmApiKeyToggle = $("btnLlmApiKeyToggle");

  function llmSetPending(on) { llmRestartAlert.classList.toggle("hidden", !on); }
  function llmShowResult(kind, text) {
    llmTestResult.className = "llm-test-result" + (kind ? " " + kind : "");
    llmTestResult.textContent = text;
  }
  function llmClearResult() {
    llmTestResult.className = "llm-test-result hidden";
    llmTestResult.textContent = "";
  }

  // POSTs carry Origin/Referer for the server's same-origin CSRF check;
  // referrerPolicy keeps the Referer even under a stricter page policy.
  async function llmPost(path, body, timeoutMs) {
    let r;
    try {
      r = await fetch(path, {
        method: "POST", cache: "no-store", credentials: "same-origin",
        referrerPolicy: "same-origin",
        headers: { "Content-Type": "application/json", Accept: "application/json", "X-Requested-With": "XMLHttpRequest" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs || 10000),
      });
    } catch (e) {
      const timeout = e && (e.name === "TimeoutError" || e.name === "AbortError");
      throw new Error(timeout ? "请求超时，Agent 未响应" : "网络错误，无法连接 Agent");
    }
    let data = null;
    try { data = await r.json(); } catch (_) {}
    return { status: r.status, data: data || {} };
  }

  function llmPayload() {
    return {
      llm_backend: llmBackend.value,
      llm_base_url: llmBaseUrl.value.trim(),
      llm_model: llmModel.value.trim(),
      llm_api_key: llmApiKey.value,
    };
  }

  async function loadLlmConfig() {
    try {
      const r = await fetch("/api/llm/config", { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const d = await r.json();
      const backend = d.llm_backend === "openai" ? "openai_compat" : d.llm_backend;
      llmBackend.value = ["edge_llm", "openai_compat", "noop"].includes(backend) ? backend : "edge_llm";
      llmBaseUrl.value = d.llm_base_url || "";
      llmModel.value = d.llm_model || "";
      llmApiKey.value = d.has_api_key ? LLM_KEY_MASK : "";
      llmApiKey.type = "password";
      btnLlmApiKeyToggle.setAttribute("aria-pressed", "false");
      btnLlmApiKeyToggle.setAttribute("aria-label", "显示密钥");
      llmSetPending(d.pending_restart === true);
    } catch (e) {
      showToast("加载 LLM 端点配置失败: " + e.message, "error");
    }
  }

  btnLlmApiKeyToggle.addEventListener("click", () => {
    const show = llmApiKey.type === "password";
    llmApiKey.type = show ? "text" : "password";
    btnLlmApiKeyToggle.setAttribute("aria-pressed", show ? "true" : "false");
    btnLlmApiKeyToggle.setAttribute("aria-label", show ? "隐藏密钥" : "显示密钥");
  });
  // Typing over the mask replaces it rather than appending to it.
  llmApiKey.addEventListener("focus", () => { if (llmApiKey.value === LLM_KEY_MASK) llmApiKey.select(); });

  btnLlmTest.addEventListener("click", async () => {
    btnLlmTest.disabled = true;
    llmShowResult("", "正在测试连接…");
    try {
      // Server bounds the probe at 5 s; allow headroom for the round trip.
      const { status, data } = await llmPost("/api/llm/test", llmPayload(), 8000);
      const lat = typeof data.latency_ms === "number" ? "（" + Math.round(data.latency_ms) + " ms）" : "";
      const st = data.status ? "[" + data.status + "] " : "";
      if (data.ok) llmShowResult("success", st + (data.message || "连接成功") + lat);
      else llmShowResult("err", st + (data.error || data.message || "测试失败 (HTTP " + status + ")") + lat);
    } catch (e) {
      llmShowResult("err", e.message);
    } finally {
      btnLlmTest.disabled = false;
    }
  });

  btnLlmSave.addEventListener("click", async () => {
    btnLlmSave.disabled = true;
    try {
      const { status, data } = await llmPost("/api/llm/config", llmPayload(), 10000);
      if (!data.ok) {
        showToast("保存端点配置失败: " + (data.error || "HTTP " + status), "error", 5000);
        return;
      }
      const changed = Array.isArray(data.changed) ? data.changed : [];
      if (!changed.length) {
        showToast("未检测到修改，端点配置保持不变", "info");
        llmSetPending(data.pending_restart === true);
        return;
      }
      llmSetPending(true);
      showToast(
        "端点配置已保存，需重启生效" + (data.key_cleared ? "（服务地址已变更，原 API 密钥已清除）" : ""),
        "success", data.key_cleared ? 6000 : 3000
      );
      await loadLlmConfig();
      llmSetPending(true);
    } catch (e) {
      showToast("保存端点配置失败: " + e.message, "error", 5000);
    } finally {
      btnLlmSave.disabled = false;
    }
  });

  $("btnLlmRestartNow").addEventListener("click", () => triggerRestartAgent());

  // ── settings panel tabs ─────────────────────────────────────────
  // Sections are only regrouped into tab panels; every control keeps its id
  // and listeners.
  //
  // desiredTab is the tab the user last chose (persisted); msActiveTab is the
  // tab actually shown. They differ only while the desired tab is hidden,
  // i.e. translator before its probe says the backend is available.
  const MS_TAB_KEY = "ovs.dashboard.settingsTab";
  const msBody = $("msBody");
  const msTabs = Array.from(document.querySelectorAll("#msTabs .ms-tab"));
  const msTabTranslator = $("msTab-translator");
  let desiredTab = "prompt";
  try { desiredTab = localStorage.getItem(MS_TAB_KEY) || "prompt"; } catch (_) {}
  if (!msFindTab(desiredTab)) desiredTab = "prompt";
  let msActiveTab = "prompt";
  // idle | loading | available | noop | failed
  let translatorProbe = "idle";

  function msTabVisible(tab) { return !!tab && !tab.hidden; }
  function msFindTab(name) { return msTabs.find((t) => t.dataset.tab === name) || null; }
  function saveDesiredTab() {
    try { localStorage.setItem(MS_TAB_KEY, desiredTab); } catch (_) {}
  }

  // Shows `name` if it is a visible tab, otherwise "prompt". Does not touch
  // desiredTab.
  function activateSettingsTab(name, opts) {
    opts = opts || {};
    let tab = msFindTab(name);
    if (!msTabVisible(tab)) tab = msFindTab("prompt");
    const changed = tab.dataset.tab !== msActiveTab || opts.force;
    msTabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      const panel = $(t.getAttribute("aria-controls"));
      if (panel) panel.hidden = !on;
    });
    msActiveTab = tab.dataset.tab;
    if (changed) msBody.scrollTop = 0;
    if (!msPanel.classList.contains("hidden")) {
      tab.scrollIntoView({ block: "nearest", inline: "nearest" });
      if (opts.focus) tab.focus();
    }
  }

  function onUserSelectTab(name, opts) {
    desiredTab = name;
    saveDesiredTab();
    activateSettingsTab(name, opts);
  }

  // Applies a translator probe result: "available" | "noop" | "failed".
  function setTranslatorProbe(state) {
    translatorProbe = state;
    if (!msTabTranslator) return;
    msTabTranslator.hidden = state !== "available";
    if (state === "available") {
      if (desiredTab === "translator" && msActiveTab !== "translator") {
        const hadFocus = msPanel.contains(document.activeElement);
        activateSettingsTab("translator", { focus: hadFocus });
      }
      return;
    }
    if (msActiveTab === "translator") {
      const hadFocus = msPanel.contains(document.activeElement);
      activateSettingsTab("prompt", { focus: hadFocus });
    }
    // noop is a definite answer, so drop a translator intent; a failed probe
    // keeps it for the next retry. Other intents are left alone either way.
    if (state === "noop" && desiredTab === "translator") {
      desiredTab = "prompt";
      saveDesiredTab();
    }
  }

  msTabs.forEach((tab) => {
    tab.addEventListener("click", () => onUserSelectTab(tab.dataset.tab));
    tab.addEventListener("keydown", (ev) => {
      const visible = msTabs.filter(msTabVisible);
      const i = visible.indexOf(tab);
      let next = null;
      if (ev.key === "ArrowRight") next = visible[(i + 1) % visible.length];
      else if (ev.key === "ArrowLeft") next = visible[(i - 1 + visible.length) % visible.length];
      else if (ev.key === "Home") next = visible[0];
      else if (ev.key === "End") next = visible[visible.length - 1];
      if (!next) return;
      ev.preventDefault();
      onUserSelectTab(next.dataset.tab);
      next.focus();
    });
  });
  activateSettingsTab(desiredTab, { force: true });

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

  // ── voice & wake settings card ──────────────────────────────────
  const setPipelineMode = $("setPipelineMode");
  const setSleepTimeout = $("setSleepTimeout");
  const setStopWords = $("setStopWords");
  const setExitWords = $("setExitWords");
  const setWakeReply = $("setWakeReply");
  const setExitReply = $("setExitReply");
  const wwPhrases = $("wwPhrases");
  const REPLY_MAX_CHARS = 200;
  // pipeline_mode as last reported by the server: a save that changes it
  // needs a restart, so the toast has to say so.
  let loadedPipelineMode = null;
  let loadedWakewordPhrases = "";
  let kwsAvailable = false;
  let isVoiceSettingsSaving = false;
  let isPartiallyApplied = false;

  const splitLines = (v) => v.split(/\r?\n/).map((x) => x.trim()).filter(Boolean);
  const splitPhrases = (v) => (v || "").split(/[,，、\n]/).map((x) => x.trim()).filter(Boolean);

  async function readJson(r) {
    let data = {};
    try { data = await r.json(); } catch (_) {}
    if (!r.ok || data.ok === false) {
      const msg = (typeof data.error === "string" && data.error) || ("HTTP " + r.status);
      throw new Error(data.request_id ? msg + "（request_id: " + data.request_id + "）" : msg);
    }
    return data;
  }

  async function loadAgentSettings() {
    try {
      const d = await readJson(await fetch("/api/agent/settings"));
      loadedPipelineMode = d.pipeline_mode || "always_on";
      setPipelineMode.value = loadedPipelineMode;
      const activeMode = d.pipeline_mode_active || loadedPipelineMode;
      const pendingRestart = activeMode !== loadedPipelineMode;
      const tag = $("modeRestartTag");
      if (tag) {
        tag.textContent = pendingRestart ? `待重启 (当前生效: ${activeMode})` : "⚠ 修改需重启生效";
      }
      const btnRestart = $("btnRestartAgent");
      if (btnRestart) {
        btnRestart.classList.toggle("hidden", !pendingRestart);
      }
      setSleepTimeout.value = d.sleep_timeout_s != null ? String(d.sleep_timeout_s) : "30";
      setStopWords.value = (d.stop_words || []).join("\n");
      setExitWords.value = (d.exit_words || []).join("\n");
      setWakeReply.value = d.wake_reply_text || "";
      setExitReply.value = d.exit_reply_text || "";
      return true;
    } catch (e) {
      showToast("加载设置失败: " + e.message, "error");
      return false;
    }
  }

  $("btnAgentSettingsReload").addEventListener("click", async () => {
    const ok = await loadAgentSettings();
    if (ok) showToast("已恢复为当前生效值", "success");
  });

  $("btnAgentSettingsSave").addEventListener("click", async () => {
    if (isVoiceSettingsSaving) return;
    const btn = $("btnAgentSettingsSave");
    const btnWw = $("btnWwSave");
    isVoiceSettingsSaving = true;
    btn.disabled = true;
    if (btnWw) btnWw.disabled = true;

    try {
      const rawTimeout = setSleepTimeout.value.trim();
      if (!/^\d+(\.\d+)?$/.test(rawTimeout)) {
        showToast("休眠等待超时必须是 ≥ 0 的数字", "error"); setSleepTimeout.focus(); return;
      }
      const sleep_timeout_s = parseFloat(rawTimeout);
      if (!Number.isFinite(sleep_timeout_s) || sleep_timeout_s < 0 || sleep_timeout_s > 86400) {
        showToast("休眠等待超时必须在 0 到 86400 秒之间", "error"); setSleepTimeout.focus(); return;
      }
      const wake_reply_text = setWakeReply.value.trim();
      const exit_reply_text = setExitReply.value.trim();
      if (wake_reply_text.length > REPLY_MAX_CHARS || exit_reply_text.length > REPLY_MAX_CHARS) {
        showToast("问候语 / 道别语不超过 " + REPLY_MAX_CHARS + " 字", "error"); return;
      }
      const stop_words = splitLines(setStopWords.value);
      const exit_words = splitLines(setExitWords.value);
      const stopWordsLower = stop_words.map((w) => w.toLowerCase());
      const overlap = exit_words.filter((w) => stopWordsLower.includes(w.toLowerCase()));
      if (overlap.length) {
        showToast("同一个词不能既是停下词又是退出词: " + overlap.join("、"), "error"); return;
      }

      const previousWakewordPhrases = loadedWakewordPhrases;
      const currentPhrases = splitPhrases(wwPhrases.value);
      const previousPhrasesList = splitPhrases(previousWakewordPhrases);
      const wwChanged = kwsAvailable && (
        currentPhrases.length !== previousPhrasesList.length ||
        currentPhrases.some((p, i) => p !== previousPhrasesList[i])
      );

      let kwsBodyJson = null;
      if (wwChanged) {
        kwsBodyJson = wakewordBody();
        if (!kwsBodyJson) return;
        try {
          const vRes = await readJson(await fetch("/api/wakeword/validate", {
            method: "POST", headers: { "Content-Type": "application/json" }, body: kwsBodyJson,
          }));
          if (!vRes || vRes.ok !== true) throw new Error(vRes?.error || "验证未通过");
        } catch (err) {
          showToast("唤醒词验证失败: " + err.message, "error", 5000);
          wwPhrases.focus();
          return;
        }
      }

      const pipeline_mode = setPipelineMode.value;
      const modeChanged = loadedPipelineMode !== null && pipeline_mode !== loadedPipelineMode;

      let kwsApplied = false;
      let kwsPersisted = false;
      let kwsPersistError = null;

      // Phase 2: High-risk runtime patch first
      if (wwChanged && kwsBodyJson) {
        try {
          const kwsRes = await readJson(await fetch("/api/wakeword/runtime", {
            method: "PATCH", headers: { "Content-Type": "application/json" }, body: kwsBodyJson,
          }));
          kwsApplied = true;
          kwsPersisted = (kwsRes.persisted === true);
          kwsPersistError = kwsRes.persist_error || null;
        } catch (kwsErr) {
          showToast("唤醒词应用失败: " + kwsErr.message + "（全局设置未提交，请修正后重试）", "error", 6000);
          return;
        }
      }

      // Phase 3: Settings POST
      let data = null;
      try {
        data = await readJson(await fetch("/api/agent/settings", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            pipeline_mode, sleep_timeout_s, stop_words, exit_words,
            wake_reply_text, exit_reply_text,
          }),
        }));
      } catch (settingsErr) {
        // Compensating Rollback
        if (kwsApplied) {
          try {
            const rollbackBody = JSON.stringify({ phrases: previousPhrasesList });
            const rbRes = await readJson(await fetch("/api/wakeword/runtime", {
              method: "PATCH", headers: { "Content-Type": "application/json" }, body: rollbackBody,
            }));
            const recon = await loadWakewordRuntime();
            const verified = recon.ok && (
              recon.phrases.length === previousPhrasesList.length &&
              recon.phrases.every((p, i) => p === previousPhrasesList[i])
            );
            if (verified) {
              isPartiallyApplied = false;
              wwPhrases.value = previousWakewordPhrases;
              showToast("全局设置保存失败: " + settingsErr.message + "（已自动回滚唤醒词修改）", "error", 7000);
            } else {
              isPartiallyApplied = true;
              wwBadge.className = "vw-badge vw-badge-danger";
              wwBadge.textContent = "部分生效(需核对)";
              wwHint.classList.remove("hidden");
              wwHint.textContent = "⚠ 唤醒词与全局配置处于部分生效状态，请刷新页面核对服务端实际配置。";
              showToast("全局设置保存失败: " + settingsErr.message + "，且唤醒词回滚对账未通过，当前处于部分生效状态，请刷新核对", "error", 9000);
            }
          } catch (rbErr) {
            isPartiallyApplied = true;
            wwBadge.className = "vw-badge vw-badge-danger";
            wwBadge.textContent = "部分生效(需核对)";
            wwHint.classList.remove("hidden");
            wwHint.textContent = "⚠ 唤醒词与全局配置处于部分生效状态，请刷新页面核对服务端实际配置。";
            showToast("全局设置保存失败: " + settingsErr.message + "，且唤醒词回滚失败，当前处于部分生效状态，请刷新页面核对", "error", 9000);
          }
        } else {
          showToast("保存失败: " + settingsErr.message, "error");
        }
        return;
      }

      // Full success: commit local state
      isPartiallyApplied = false;
      const settingsPersisted = (data.persisted === true);
      const settingsChanged = Array.isArray(data.changed) ? data.changed : [];
      loadedPipelineMode = data.pipeline_mode || pipeline_mode;
      if (kwsApplied) {
        loadedWakewordPhrases = wwPhrases.value.trim();
      }

      const activeMode = data.pipeline_mode_active || loadedPipelineMode;
      const pendingRestart = activeMode !== loadedPipelineMode;
      const tag = $("modeRestartTag");
      if (tag) {
        tag.textContent = pendingRestart ? `待重启 (当前生效: ${activeMode})` : "⚠ 修改需重启生效";
      }
      const btnRestart = $("btnRestartAgent");
      if (btnRestart) {
        btnRestart.classList.toggle("hidden", !pendingRestart);
      }

      if (!wwChanged && settingsChanged.length === 0) {
        showToast("未检测到修改，配置保持不变", "info");
        return;
      }

      // 2x2 persistence matrix reporting with explicit persist_error support
      let msg, kind;
      if (wwChanged && settingsChanged.length === 0) {
        if (kwsPersisted) {
          msg = "唤醒词已更新并保存"; kind = "success";
        } else {
          msg = "唤醒词已热更新（无配置文件，重启恢复原值）" + (kwsPersistError ? ": " + kwsPersistError : "");
          kind = "warn";
        }
      } else if (wwChanged && settingsChanged.length > 0) {
        if (settingsPersisted && kwsPersisted) {
          msg = "设置与唤醒词已成功保存并持久化"; kind = "success";
        } else if (settingsPersisted && !kwsPersisted) {
          msg = "全局设置已持久化；唤醒词写入失败: " + (kwsPersistError || "无配置文件，重启恢复原值");
          kind = "warn";
        } else if (!settingsPersisted && kwsPersisted) {
          msg = "唤醒词已持久化；全局设置写入失败: " + (data.persist_error || "无配置文件，重启恢复原值");
          kind = "warn";
        } else {
          msg = "运行时已热生效（无配置文件，重启恢复原值）";
          if (data.persist_error || kwsPersistError) {
            msg += " (" + [data.persist_error, kwsPersistError].filter(Boolean).join("; ") + ")";
          }
          kind = "warn";
        }
      } else {
        if (settingsPersisted) {
          msg = "已保存并写入配置文件"; kind = "success";
        } else if (data.persist_error) {
          msg = "运行时已生效，写入配置文件失败: " + data.persist_error; kind = "warn";
        } else {
          msg = "运行时已生效（无配置文件，重启后恢复原值）"; kind = "warn";
        }
      }

      if (modeChanged) {
        msg += "；pipeline_mode 已改为 " + pipeline_mode + "，重启 agent 后生效";
        kind = "warn";
      }
      showToast(msg, kind, modeChanged || kind !== "success" ? 6000 : 2500);
    } catch (e) {
      showToast("保存失败: " + e.message, "error");
    } finally {
      isVoiceSettingsSaving = false;
      btn.disabled = false;
      if (btnWw) btnWw.disabled = !kwsAvailable;
    }
  });

  // Runtime wake-word (runtime_kws plugin). Without the plugin every control
  // in this group is disabled and the hint explains why.
  const wwBadge = $("wwBadge");
  const wwHint = $("wwHint");
  const wwControls = [wwPhrases, $("btnWwValidate"), $("btnWwSave"), $("btnWwTone")];

  function setWakewordState(available, enabled, running) {
    wwControls.forEach((el) => { el.disabled = !available; });
    let badgeClass = "vw-badge-disabled";
    let badgeText = "未挂载";
    if (available && running) {
      badgeClass = "vw-badge-normal";
      badgeText = "运行中";
    } else if (available && !running) {
      badgeClass = "vw-badge-warning";
      badgeText = "未就绪";
    } else if (enabled) {
      badgeClass = "vw-badge-danger";
      badgeText = "加载失败";
    }
    wwBadge.className = "vw-badge " + badgeClass;
    wwBadge.textContent = badgeText;
    wwHint.classList.toggle("hidden", available && running);
    wwHint.textContent = enabled
      ? "runtime_kws 插件已挂载但未就绪（模型未启动或加载失败），请查看 agent 日志。"
      : "未挂载 runtime_kws 插件，唤醒词不可在线修改。";
  }

  async function loadWakewordRuntime() {
    try {
      const d = await readJson(await fetch("/api/wakeword/runtime"));
      kwsAvailable = !!d.available;
      isPartiallyApplied = false;
      setWakewordState(!!d.available, !!d.enabled, !!d.running);
      if (d.available) {
        loadedWakewordPhrases = (d.phrases || []).join(", ");
        wwPhrases.value = loadedWakewordPhrases;
      } else {
        loadedWakewordPhrases = "";
      }
      return { ok: true, phrases: d.phrases || [] };
    } catch (e) {
      kwsAvailable = false;
      loadedWakewordPhrases = "";
      setWakewordState(false, false, false);
      wwHint.textContent = "读取唤醒词状态失败: " + e.message;
      return { ok: false, error: e.message };
    }
  }

  function wakewordBody() {
    const phrases = splitPhrases(wwPhrases.value);
    if (!phrases.length) { showToast("请输入至少一个唤醒短语", "error"); wwPhrases.focus(); return null; }
    for (const p of phrases) {
      if (p.length > 30) {
        showToast("单个唤醒短语不能超过 30 字符: " + p.slice(0, 10) + "…", "error");
        wwPhrases.focus();
        return null;
      }
    }
    return JSON.stringify({ phrases });
  }

  $("btnWwValidate").addEventListener("click", async () => {
    const body = wakewordBody(); if (!body) return;
    const btn = $("btnWwValidate");
    btn.disabled = true;
    try {
      const d = await readJson(await fetch("/api/wakeword/validate", {
        method: "POST", headers: { "Content-Type": "application/json" }, body,
      }));
      showToast("验证通过: " + (d.phrases || []).join(", "), "success");
    } catch (e) { showToast("验证失败: " + e.message, "error", 5000); }
    finally { btn.disabled = false; }
  });

  $("btnWwSave").addEventListener("click", async () => {
    if (isVoiceSettingsSaving) return;
    const body = wakewordBody(); if (!body) return;
    const btn = $("btnWwSave");
    const btnSave = $("btnAgentSettingsSave");
    isVoiceSettingsSaving = true;
    btn.disabled = true;
    if (btnSave) btnSave.disabled = true;
    try {
      const d = await readJson(await fetch("/api/wakeword/runtime", {
        method: "PATCH", headers: { "Content-Type": "application/json" }, body,
      }));
      isPartiallyApplied = false;
      wwPhrases.value = (d.phrases || []).join(", ");
      loadedWakewordPhrases = wwPhrases.value.trim();
      if (d.persisted) showToast("唤醒词已更新并保存", "success");
      else showToast("唤醒词已热更新，持久化失败（重启后恢复）: " + (d.persist_error || "未知原因"), "warn", 6000);
    } catch (e) { showToast("应用唤醒词失败: " + e.message, "error", 5000); }
    finally {
      isVoiceSettingsSaving = false;
      btn.disabled = false;
      if (btnSave) btnSave.disabled = false;
    }
  });

  $("btnWwTone").addEventListener("click", async () => {
    const btn = $("btnWwTone");
    btn.disabled = true;
    try {
      await readJson(await fetch("/api/wakeword/test-tone", { method: "POST" }));
      showToast("已播放提示音", "success");
    } catch (e) { showToast("试听失败: " + e.message, "error"); }
    finally { btn.disabled = false; }
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
    translatorLoaded = true;
    setTranslatorProbe(noop ? "noop" : "available");
    if (setTranslatorTgtLang) setTranslatorTgtLang.disabled = noop;
    if (btnTranslatorSave) {
      btnTranslatorSave.disabled = noop;
      btnTranslatorSave.title = noop ? "translator_backend 当前为 noop，不会翻译" : "";
    }
  }

  // Only the newest request may apply its result, so a slow stale response
  // cannot override a later one.
  let translatorProbeSeq = 0;
  async function loadTranslatorRuntime(opts) {
    const silent = !!(opts && opts.silent);
    const seq = ++translatorProbeSeq;
    translatorProbe = "loading";
    try {
      const r = await fetch("/api/translator/runtime");
      if (!r.ok) throw new Error("HTTP " + r.status);
      const d = await r.json();
      if (seq !== translatorProbeSeq) return;
      applyTranslatorRuntime(d);
    } catch (e) {
      if (seq !== translatorProbeSeq) return;
      setTranslatorProbe("failed");
      if (!silent) showToast("加载翻译设置失败: " + e.message, "error");
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
  if (btnTranslatorReload) btnTranslatorReload.addEventListener("click", () => loadTranslatorRuntime());
  if (btnTranslatorSave) btnTranslatorSave.addEventListener("click", saveTranslatorRuntime);
  loadTranslatorRuntime({ silent: true });

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

  let restartingAgent = false;
  async function triggerRestartAgent() {
    if (restartingAgent) return;
    if (!window.confirm("确认重启 Agent 进程？\n（若配置了 systemd/Docker 守护将自动拉起；若在终端手动运行请手动重启）")) return;
    restartingAgent = true;
    showToast("正在请求重启 Agent...", "info", 5000);
    try {
      // restart_agent sits behind the same admin-token boundary as /api/mcp/*,
      // so it shares the coded error parsing (Chinese text + request_id).
      await mcpRequest("POST", "/api/control/restart_agent", undefined, { timeoutMs: 10000 });
      showToast("Agent 正在重启，等待服务恢复...", "warn", 10000);

      // Wait 1.5s for process termination before starting recovery probes
      await new Promise((r) => setTimeout(r, 1500));

      let attempts = 0;
      const maxAttempts = 25;
      let stopped = false;

      const pollStep = async () => {
        if (stopped) return;
        attempts++;
        if (attempts > maxAttempts) {
          stopped = true;
          restartingAgent = false;
          showToast("等待 Agent 重启超时，请检查终端日志或手动刷新", "error", 10000);
          return;
        }

        try {
          const res = await fetch("/api/agent/settings", { signal: AbortSignal.timeout(2000) });
          if (res.ok) {
            stopped = true;
            showToast("Agent 重启成功，正在刷新页面...", "success", 2000);
            setTimeout(() => { location.reload(); }, 600);
            return;
          }
        } catch (_) {}

        if (!stopped) {
          setTimeout(pollStep, 1000);
        }
      };

      setTimeout(pollStep, 1000);
    } catch (e) {
      restartingAgent = false;
      showToast("触发重启失败：" + mcpErrorText(e), "error", 8000);
    }
  }

  const btnRestartAgent = $("btnRestartAgent");
  if (btnRestartAgent) {
    btnRestartAgent.addEventListener("click", triggerRestartAgent);
  }
  const btnRestartAgentMenu = $("btnRestartAgentMenu");
  if (btnRestartAgentMenu) {
    btnRestartAgentMenu.addEventListener("click", () => {
      $("menu")?.classList.add("hidden");
      triggerRestartAgent();
    });
  }
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
      ctx.fillStyle = rmsSamples[i] >= thr ? "#357f11" : "#64748b";
      ctx.fillRect(i * bw, H - h, Math.max(1, bw - 1), h);
    }
    // threshold line
    const ty = H - Math.min(1, thr / 0.2) * H;
    ctx.strokeStyle = "#475569"; ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(0, ty); ctx.lineTo(W, ty); ctx.stroke();
    ctx.setLineDash([]);
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
      ctx.strokeStyle = "#e2e8f0"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
      return;
    }
    const maxV = Math.max(1, ...buf);
    ctx.strokeStyle = "#357f11"; ctx.lineWidth = 3;
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
      // Refresh the voice & wake card only while the settings panel is open.
      if (!msPanel.classList.contains("hidden")) {
        loadAgentSettings();
      }
      return;
    }

    if (ev === "on_translator_runtime_change") {
      translatorProbeSeq++;
      try { applyTranslatorRuntime(data); } catch (_) {}
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
      hintEl.style.color = kind === "error" ? "var(--error)" : "";
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
        vcStatus.style.color = "var(--error)";
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
        vcStatus.style.color = "var(--error)";
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
        vcStatus.style.color = "var(--error)";
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

  // ── MCP server management (spec v7.3 §5) ─────────────────────────
  // Backend routes (debug_dashboard.py): GET/POST /api/mcp/servers,
  // PATCH/DELETE /api/mcp/servers/{name}, POST /api/mcp/test. Create uses
  // POST; edit uses PATCH, which merges into the stored entry so fields the
  // form does not show (e.g. timeout_s) are kept.
  const MCP_API = {
    list: "/api/mcp/servers",
    create: "/api/mcp/servers",
    item: (name) => "/api/mcp/servers/" + encodeURIComponent(name),
    test: "/api/mcp/test",
  };
  const MCP_MASK = "******";
  const MCP_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
  // stdio whitelist hints (backend validate_stdio_args is authoritative).
  const MCP_PY_CMD_RE = /^(?:.*\/)?python(?:3(?:\.\d+)?)?$/;
  // Client copy per DashboardApiError code. Unknown codes fall back to the
  // server's SAFE_CLIENT_MESSAGES text, which is already sanitised.
  const MCP_ERROR_TEXT = {
    UNAUTHORIZED: "管理令牌无效或已失效，请刷新页面后重试",
    FORBIDDEN: "请求来源校验未通过，请在本页面内操作",
    ETAG_MISMATCH: "配置已被其他会话修改，列表已重新加载，请确认后重试",
    CONCURRENT_MODIFICATION: "操作冲突，请稍后重试",
    TOO_MANY_REQUESTS: "同时进行的测试已达上限（2 个），请稍后重试",
    SERVICE_UNAVAILABLE: "Agent 正在停机或暂不可用",
    CONNECTION_FAILED: "无法连接 MCP 服务，或工具发现失败",
    TEST_TIMEOUT: "测试连接超时（10 秒）",
    INVALID_PARAM: "参数无效，请检查表单内容",
    SECURITY_VIOLATION: "安全校验未通过：命令、参数或目标地址不在允许范围内",
    FILE_TOO_LARGE: "配置超过 2MB 上限",
    CONFIG_PARSE_ERROR: "配置文件语法错误",
    CONFIG_RECOVERY_REQUIRED: "配置处于只读锁定状态，需要人工恢复后重启 Agent",
    DISK_FULL: "设备存储空间已满，配置未写入",
    TMP_CREATE_FAILED: "无法创建临时配置文件，配置未写入，请检查配置目录权限",
    TMP_CLEANUP_FAILED: "临时配置文件清理失败，请检查配置目录",
    FD_CLOSE_FAILED: "配置文件句柄关闭失败，请查看 Agent 日志",
    FCHMOD_FAILED: "无法设置配置文件权限（0600），配置未写入",
    FSYNC_FAILED: "配置文件同步到存储失败，配置可能未保存",
    DIR_FSYNC_FAILED: "配置目录同步到存储失败，配置可能未保存",
    WRITE_FAILED: "写入配置文件失败，配置未保存",
    SHORT_WRITE: "配置文件写入不完整，配置未保存",
    REPLACE_FAILED: "替换配置文件失败，原配置保持不变",
    READ_FAILED: "读取配置文件失败，请检查文件是否存在及权限",
    BAD_REQUEST: "请求格式错误，请刷新页面后重试",
    INTERNAL_ERROR: "服务端内部错误，请查看 Agent 日志",
    MISSING_ARGUMENT: "参数缺失，stdio 模式需要提供非空 args 列表",
    INVALID_MODULE_NAME: "Python 模块名格式不正确，必须为合法的模块标识符",
    INVALID_SCRIPT_PATH: "Python 脚本路径不合法，必须以 .py 结尾且不能以选项参数开头",
    SCRIPT_NOT_FOUND: "指定的 Python 脚本文件不存在，请检查路径是否正确",
    SCRIPT_NOT_READABLE: "Python 脚本文件无读取权限，请检查文件权限",
    ILLEGAL_ARGUMENT: "包含不支持的参数格式，严禁使用 -c 参数",
    FORBIDDEN_ENV: "禁止设置动态链接器相关环境变量",
  };

  function adminToken() {
    const m = document.querySelector('meta[name="ovs-admin-token"]');
    return m ? (m.getAttribute("content") || "").trim() : "";
  }

  class McpApiError extends Error {
    constructor(code, message, requestId, status) {
      super(message);
      this.code = code; this.requestId = requestId || ""; this.status = status || 0;
    }
  }
  function mcpErrorText(e) {
    if (!(e instanceof McpApiError)) return "操作失败：" + (e && e.message ? e.message : "未知错误");
    return e.message + (e.requestId ? "（request_id: " + e.requestId + "）" : "");
  }

  async function mcpRequest(method, path, body, opts) {
    opts = opts || {};
    const headers = { Accept: "application/json" };
    const token = adminToken();
    if (token) headers["X-Admin-Token"] = token;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (opts.ifMatch) headers["If-Match"] = opts.ifMatch;
    let r;
    try {
      r = await fetch(path, {
        method, headers, cache: "no-store", credentials: "same-origin",
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: opts.timeoutMs ? AbortSignal.timeout(opts.timeoutMs) : undefined,
      });
    } catch (e) {
      const timeout = e && (e.name === "TimeoutError" || e.name === "AbortError");
      throw new McpApiError("NETWORK", timeout ? "请求超时，Agent 未响应" : "网络错误，无法连接 Agent", "", 0);
    }
    let data = null;
    try { data = await r.json(); } catch (_) {}
    if (r.ok && data && data.ok !== false) return { data, headers: r.headers };
    // Error envelope: {ok:false, code, error, request_id}; also accept
    // {error:{code, message, request_id}}.
    const errObj = data && data.error && typeof data.error === "object" ? data.error : null;
    const code = (data && data.code) || (errObj && errObj.code) || "";
    const rid = (data && data.request_id) || (errObj && errObj.request_id) || r.headers.get("X-Request-Id") || "";
    // Only trust server text that came through the coded error boundary.
    const serverMsg = code ? (errObj ? errObj.message : data.error) : "";
    let msg = MCP_ERROR_TEXT[code] || (typeof serverMsg === "string" ? serverMsg : "");
    if (!msg) msg = r.status === 404 ? "当前 Agent 未提供该 MCP 接口 (HTTP 404)" : "请求失败 (HTTP " + r.status + ")";
    throw new McpApiError(code || "HTTP_" + r.status, msg, rid, r.status);
  }

  function mkEl(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }
  function mkBtn(text, cls, act) {
    const b = mkEl("button", "ctl-btn" + (cls ? " " + cls : ""), text);
    b.type = "button";
    if (act) b.dataset.act = act;
    return b;
  }

  const mcpListEl = $("mcpList");
  const mcpNotice = $("mcpNotice");
  const mcpPendingBar = $("mcpPendingBar");
  const mcpCount = $("mcpCount");
  const btnMcpAdd = $("btnMcpAdd");
  const btnMcpRefresh = $("btnMcpRefresh");
  const mcpState = {
    etag: "", servers: [], active: [], pending: false,
    expanded: new Set(), testResults: {},
  };

  function mcpTransport(s) {
    const t = String((s && (s.transport || s.type)) || (s && s.url ? "sse" : "stdio")).toLowerCase();
    return t === "sse" ? "sse" : "stdio";
  }
  function mcpStatus(persisted, active) {
    if (!persisted) return { cls: "vw-badge-warning", text: "待移除 · 重启后生效" };
    if (!active) return { cls: "vw-badge-warning", text: "待重启生效" };
    const raw = active.status != null ? active.status : active.state;
    const st = raw == null ? (active.connected === false ? "failed" : "connected") : String(raw).toLowerCase();
    if (["connected", "running", "ok", "ready", "active"].includes(st)) return { cls: "vw-badge-normal", text: "运行中" };
    if (["failed", "error", "disconnected", "crashed"].includes(st)) return { cls: "vw-badge-danger", text: "连接失败" };
    return { cls: "vw-badge-disabled", text: String(raw) };
  }
  function mcpToolSchema(t) {
    return t.inputSchema || t.input_schema || t.parameters || t.schema || null;
  }
  function mcpTargetText(s) {
    if (mcpTransport(s) === "sse") return s.url || "(未设置 url)";
    const args = Array.isArray(s.args) ? s.args : [];
    return [s.command || "python（默认运行时）"].concat(args).join(" ");
  }
  function mcpServerPayload(s) {
    const p = { name: s.name, transport: mcpTransport(s) };
    if (p.transport === "sse") { p.url = s.url || ""; }
    else { if (s.command) p.command = s.command; p.args = Array.isArray(s.args) ? s.args.slice() : []; p.env = Object.assign({}, s.env || {}); }
    return p; // headers omitted: absent keys keep the persisted credential
  }

  function renderMcpTools(box, tools, fromTest) {
    box.textContent = "";
    if (!tools || !tools.length) {
      box.appendChild(mkEl("div", "muted small", fromTest ? "测试成功，但服务未声明任何工具" : "服务未运行，暂无工具信息；可点击「测试」查看工具清单"));
      return;
    }
    if (fromTest) box.appendChild(mkEl("div", "muted small", "以下为最近一次连通性测试返回的工具"));
    tools.forEach((t) => {
      const row = mkEl("div", "mcp-tool");
      row.appendChild(mkEl("div", "mcp-tool-name", t.name || "(unnamed)"));
      if (t.description) row.appendChild(mkEl("div", "mcp-tool-desc", t.description));
      const schema = mcpToolSchema(t);
      if (schema) {
        const det = document.createElement("details");
        det.appendChild(mkEl("summary", "muted small", "参数 Schema"));
        let txt;
        try { txt = JSON.stringify(schema, null, 2); } catch (_) { txt = String(schema); }
        det.appendChild(mkEl("pre", "mcp-schema", txt));
        row.appendChild(det);
      }
      box.appendChild(row);
    });
  }

  function renderMcpPending() {
    mcpPendingBar.classList.toggle("hidden", !mcpState.pending);
    $("msTabPendingBadge-mcp").classList.toggle("hidden", !mcpState.pending);
  }

  function renderMcp() {
    renderMcpPending();
    const activeBy = new Map();
    mcpState.active.forEach((a) => { if (a && a.name) activeBy.set(a.name, a); });
    const rows = mcpState.servers.filter((s) => s && s.name).map((s) => ({ s, a: activeBy.get(s.name) || null, persisted: true }));
    const names = new Set(rows.map((r) => r.s.name));
    mcpState.active.forEach((a) => { if (a && a.name && !names.has(a.name)) rows.push({ s: a, a, persisted: false }); });

    const running = rows.filter((r) => r.persisted && r.a).length;
    mcpCount.textContent = rows.length ? running + "/" + rows.length + " 运行" : "0";
    mcpCount.className = "vw-badge " + (rows.length ? "vw-badge-normal" : "vw-badge-disabled");

    mcpListEl.textContent = "";
    // The empty-state text (CSS :empty) only applies after a successful load.
    mcpListEl.dataset.loaded = "1";
    rows.forEach(({ s, a, persisted }) => {
      const transport = mcpTransport(s);
      const st = mcpStatus(persisted, a);
      const item = mkEl("div", "mcp-item" + (persisted ? "" : " mcp-removing"));
      item.dataset.name = s.name;

      const head = mkEl("div", "mcp-item-head");
      head.appendChild(mkEl("span", "mcp-name", s.name));
      head.appendChild(mkEl("span", "mcp-proto mcp-proto-" + transport, transport));
      head.appendChild(mkEl("span", "vw-badge " + st.cls, st.text));
      item.appendChild(head);
      item.appendChild(mkEl("div", "mcp-target", mcpTargetText(s)));
      if (transport === "sse" && s.has_headers) {
        const keys = Array.isArray(s.configured_header_keys) ? s.configured_header_keys : [];
        item.appendChild(mkEl("div", "mcp-target", "headers: " + (keys.length ? keys.join(", ") : "已配置")));
      }

      const tested = mcpState.testResults[s.name];
      const tools = tested ? tested.tools : (a && Array.isArray(a.tools) ? a.tools : []);
      const expanded = mcpState.expanded.has(s.name);
      const btns = mkEl("div", "mcp-btn-row");
      const bTools = mkBtn((expanded ? "▾" : "▸") + " 工具 (" + tools.length + ")", "", "tools");
      bTools.setAttribute("aria-expanded", String(expanded));
      btns.appendChild(bTools);
      btns.appendChild(mkBtn("测试", "", "test"));
      const bEdit = mkBtn("编辑", "", "edit");
      const bDel = mkBtn("删除", "danger", "delete");
      if (!persisted) { bEdit.disabled = true; bDel.disabled = true; }
      btns.appendChild(bEdit);
      btns.appendChild(bDel);
      item.appendChild(btns);

      const toolsBox = mkEl("div", "mcp-tools" + (expanded ? "" : " hidden"));
      renderMcpTools(toolsBox, tools, !!tested);
      item.appendChild(toolsBox);
      const result = mkEl("div", "mcp-item-result hidden");
      item.appendChild(result);

      btns.addEventListener("click", (ev) => {
        const b = ev.target.closest("button[data-act]");
        if (!b || b.disabled) return;
        const act = b.dataset.act;
        if (act === "tools") {
          if (mcpState.expanded.has(s.name)) mcpState.expanded.delete(s.name); else mcpState.expanded.add(s.name);
          renderMcp();
        } else if (act === "test") {
          testMcpExisting(s, b, result);
        } else if (act === "edit") {
          openMcpModal(s, b);
        } else if (act === "delete") {
          deleteMcp(s, b);
        }
      });
      mcpListEl.appendChild(item);
    });
  }

  function setMcpNotice(text) {
    mcpNotice.textContent = text || "";
    mcpNotice.classList.toggle("hidden", !text);
  }

  async function loadMcpServers(opts) {
    opts = opts || {};
    if (!adminToken()) {
      setMcpNotice("页面未注入管理令牌（ovs-admin-token），当前 Agent 版本不支持在线管理 MCP 服务。");
      btnMcpAdd.disabled = true;
      return false;
    }
    btnMcpRefresh.disabled = true;
    try {
      const { data, headers } = await mcpRequest("GET", MCP_API.list, undefined, { timeoutMs: 8000 });
      mcpState.etag = data.etag || headers.get("ETag") || "";
      mcpState.servers = Array.isArray(data.servers) ? data.servers : [];
      mcpState.active = Array.isArray(data.active_servers) ? data.active_servers : [];
      mcpState.pending = data.pending_restart === true;
      btnMcpAdd.disabled = false;
      setMcpNotice("");
      renderMcp();
      if (opts.toast) showToast("MCP 列表已刷新", "success");
      return true;
    } catch (e) {
      setMcpNotice("加载 MCP 服务失败：" + mcpErrorText(e));
      if (!opts.silent) showToast("加载 MCP 服务失败：" + mcpErrorText(e), "error", 6000);
      return false;
    } finally {
      btnMcpRefresh.disabled = false;
    }
  }

  function mcpTestSummary(data) {
    const tools = Array.isArray(data.tools) ? data.tools : [];
    const n = data.tools_count != null ? data.tools_count : tools.length;
    return { tools, text: "连接成功，发现 " + n + " 个工具" };
  }

  async function testMcpExisting(s, btn, resultEl) {
    const label = btn.textContent;
    btn.disabled = true; btn.textContent = "测试中…";
    resultEl.className = "mcp-item-result muted"; resultEl.textContent = "正在连接（最长 10 秒）…";
    try {
      const { data } = await mcpRequest("POST", MCP_API.test, mcpServerPayload(s), { timeoutMs: 15000 });
      const sum = mcpTestSummary(data);
      mcpState.testResults[s.name] = { tools: sum.tools };
      mcpState.expanded.add(s.name);
      renderMcp();
      const fresh = mcpListEl.querySelector('.mcp-item[data-name="' + CSS.escape(s.name) + '"] .mcp-item-result');
      if (fresh) { fresh.className = "mcp-item-result ok"; fresh.textContent = "✓ " + sum.text; }
    } catch (e) {
      resultEl.className = "mcp-item-result err";
      resultEl.textContent = "✗ " + mcpErrorText(e);
      btn.disabled = false; btn.textContent = label;
    }
  }

  async function deleteMcp(s, btn) {
    if (!window.confirm("删除 MCP 服务「" + s.name + "」？\n删除写入配置后需重启 Agent 才会断开该服务。")) return;
    btn.disabled = true;
    try {
      const { data } = await mcpRequest("DELETE", MCP_API.item(s.name), undefined, { ifMatch: mcpState.etag, timeoutMs: 10000 });
      if (data.etag) mcpState.etag = data.etag;
      delete mcpState.testResults[s.name];
      mcpState.expanded.delete(s.name);
      showToast("已删除「" + s.name + "」，重启后生效", "success", 3000);
      mcpState.pending = data.pending_restart !== false;
      renderMcpPending();
      await loadMcpServers({ silent: true });
    } catch (e) {
      btn.disabled = false;
      showToast("删除失败：" + mcpErrorText(e), "error", 8000);
      if (e.code === "ETAG_MISMATCH") await loadMcpServers({ silent: true });
    }
  }

  // ── MCP add / edit modal ─────────────────────────────────────────
  const mcpModal = $("mcpModal");
  const mcpForm = $("mcpForm");
  const mcpName = $("mcpName");
  const mcpCommand = $("mcpCommand");
  const mcpArgs = $("mcpArgs");
  const mcpEnv = $("mcpEnv");
  const mcpUrl = $("mcpUrl");
  const mcpHeaders = $("mcpHeaders");
  const mcpTestResult = $("mcpTestResult");
  const mcpFormError = $("mcpFormError");
  const btnMcpTest = $("btnMcpTest");
  const btnMcpSave = $("btnMcpSave");
  let mcpEditing = null;
  let mcpReturnFocus = null;

  function mcpFormTransport() {
    const r = mcpForm.querySelector('input[name="mcpTransport"]:checked');
    return r && r.value === "sse" ? "sse" : "stdio";
  }
  function setMcpFormTransport(t) {
    mcpForm.querySelectorAll('input[name="mcpTransport"]').forEach((r) => { r.checked = r.value === t; });
    $("mcpStdioFields").classList.toggle("hidden", t !== "stdio");
    $("mcpSseFields").classList.toggle("hidden", t !== "sse");
  }
  mcpForm.querySelectorAll('input[name="mcpTransport"]').forEach((r) => {
    r.addEventListener("change", () => setMcpFormTransport(mcpFormTransport()));
  });

  function kvToText(obj, sep) {
    return Object.keys(obj || {}).map((k) => k + sep + (obj[k] == null ? "" : String(obj[k]))).join("\n");
  }
  function parseKvText(text, sep, label) {
    const out = {};
    const lines = String(text || "").split("\n");
    for (let i = 0; i < lines.length; i++) {
      // Only the key is trimmed; the value (often a credential) is kept as
      // typed. A trailing "\r" from pasted CRLF text is not part of the value.
      const line = lines[i].replace(/\r$/, "");
      if (!line.trim()) continue;
      const idx = line.indexOf(sep);
      const key = idx > 0 ? line.slice(0, idx).trim() : "";
      if (!key) return { error: label + " 第 " + (i + 1) + " 行格式应为 KEY" + sep + "VALUE" };
      if (Object.prototype.hasOwnProperty.call(out, key)) return { error: label + " 中 " + key + " 重复" };
      let value = line.slice(idx + sep.length);
      // "Key: Value" header lines: the whitespace after ":" is HTTP optional
      // whitespace, not part of the value (and the server strips it anyway).
      if (sep === ":") value = value.replace(/^[ \t]+/, "");
      out[key] = value;
    }
    return { obj: out };
  }

  function clearMcpFieldErrors() {
    mcpForm.querySelectorAll('[aria-invalid="true"]').forEach((n) => n.removeAttribute("aria-invalid"));
    mcpFormError.classList.add("hidden");
    mcpFormError.textContent = "";
  }
  function showMcpFormError(text, field) {
    mcpFormError.textContent = text;
    mcpFormError.classList.remove("hidden");
    if (field) { field.setAttribute("aria-invalid", "true"); field.focus(); }
  }

  // Secret retention (spec §2.4): untouched "******" keeps the stored value,
  // "" deletes the key. A key removed from the textarea while editing is sent
  // as "" so that deleting the line deletes the entry.
  function withDeletions(next, originalKeys) {
    if (!mcpEditing) {
      Object.keys(next).forEach((k) => { if (next[k] === "") delete next[k]; });
      return next;
    }
    (originalKeys || []).forEach((k) => { if (!Object.prototype.hasOwnProperty.call(next, k)) next[k] = ""; });
    return next;
  }

  function buildMcpPayload(forTest) {
    clearMcpFieldErrors();
    const name = mcpEditing ? mcpEditing.name : mcpName.value.trim();
    if (!MCP_NAME_RE.test(name) && !(forTest && !name)) {
      return { error: "名称需以字母或数字开头，仅含字母、数字、- 或 _，最长 64 位", field: mcpName };
    }
    if (!mcpEditing && !forTest && mcpState.servers.some((s) => s && s.name === name)) {
      return { error: "已存在同名服务「" + name + "」", field: mcpName };
    }
    const transport = mcpFormTransport();
    const p = { transport };
    if (name) p.name = name;
    if (transport === "stdio") {
      // The backend only launches its own Python runtime; command may be left
      // empty (default runtime) or name a python interpreter.
      const command = mcpCommand.value.trim();
      if (command && !MCP_PY_CMD_RE.test(command)) {
        return { error: "command 只能留空（使用默认 Python 运行时）或填写 python / python3", field: mcpCommand };
      }
      if (command) p.command = command;
      p.args = mcpArgs.value.split("\n").map((x) => x.trim()).filter(Boolean);
      const a0 = p.args[0] || "";
      const a1 = p.args[1] || "";
      const argsOk = a0 === "-m"
        ? (p.args.length >= 2 && /^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$/.test(a1))
        : (/\.py$/i.test(a0) && !a0.startsWith("-"));
      if (!argsOk) {
        return { error: "args 第一行需为 -m（第二行写合法的模块名），或以 .py 结尾的 Python 脚本路径", field: mcpArgs };
      }
      const env = parseKvText(mcpEnv.value, "=", "env");
      if (env.error) return { error: env.error, field: mcpEnv };
      const origEnv = mcpEditing && mcpTransport(mcpEditing) === "stdio" ? Object.keys(mcpEditing.env || {}) : [];
      p.env = withDeletions(env.obj, origEnv);
    } else {
      const url = mcpUrl.value.trim();
      let u = null;
      try { u = new URL(url); } catch (_) {}
      if (!u || (u.protocol !== "http:" && u.protocol !== "https:")) {
        return { error: "url 需为 http:// 或 https:// 开头的完整地址", field: mcpUrl };
      }
      p.url = url;
      const hdr = parseKvText(mcpHeaders.value, ":", "headers");
      if (hdr.error) return { error: hdr.error, field: mcpHeaders };
      const origHdr = mcpEditing && mcpTransport(mcpEditing) === "sse" && Array.isArray(mcpEditing.configured_header_keys)
        ? mcpEditing.configured_header_keys : [];
      p.headers = withDeletions(hdr.obj, origHdr);
    }
    return { payload: p };
  }

  function openMcpModal(server, returnFocus) {
    mcpEditing = server || null;
    mcpReturnFocus = returnFocus || document.activeElement;
    clearMcpFieldErrors();
    mcpTestResult.classList.add("hidden");
    mcpTestResult.textContent = "";
    $("mcpModalTitle").textContent = server ? "编辑 MCP 服务 · " + server.name : "添加 MCP 服务";
    mcpName.value = server ? server.name : "";
    mcpName.disabled = !!server;
    mcpName.title = server ? "名称是服务标识，编辑时不可修改；如需改名请删除后重新添加" : "";
    const t = server ? mcpTransport(server) : "stdio";
    mcpCommand.value = server && t === "stdio" ? (server.command || "") : "";
    mcpArgs.value = server && t === "stdio" && Array.isArray(server.args) ? server.args.join("\n") : "";
    mcpEnv.value = server && t === "stdio" ? kvToText(server.env, "=") : "";
    mcpUrl.value = server && t === "sse" ? (server.url || "") : "";
    const hdrKeys = server && t === "sse" && Array.isArray(server.configured_header_keys) ? server.configured_header_keys : [];
    mcpHeaders.value = hdrKeys.map((k) => k + ": " + MCP_MASK).join("\n");
    setMcpFormTransport(t);
    btnMcpSave.disabled = false; btnMcpTest.disabled = false;
    mcpModal.classList.remove("hidden");
    (server ? (t === "stdio" ? mcpCommand : mcpUrl) : mcpName).focus();
  }
  function closeMcpModal() {
    mcpModal.classList.add("hidden");
    mcpEditing = null;
    if (mcpReturnFocus && document.contains(mcpReturnFocus)) mcpReturnFocus.focus();
    mcpReturnFocus = null;
  }

  btnMcpTest.addEventListener("click", async () => {
    const b = buildMcpPayload(true);
    if (b.error) { showMcpFormError(b.error, b.field); return; }
    btnMcpTest.disabled = true; btnMcpTest.textContent = "测试中…";
    mcpTestResult.className = "mcp-test-result muted";
    mcpTestResult.textContent = "正在连接（最长 10 秒）…";
    try {
      const { data } = await mcpRequest("POST", MCP_API.test, b.payload, { timeoutMs: 15000 });
      const sum = mcpTestSummary(data);
      mcpTestResult.className = "mcp-test-result";
      mcpTestResult.textContent = "";
      mcpTestResult.appendChild(mkEl("div", "", "✓ " + sum.text));
      if (sum.tools.length) {
        const ul = document.createElement("ul");
        sum.tools.forEach((t) => {
          const li = mkEl("li", "");
          li.appendChild(mkEl("span", "mcp-tool-name", t.name || "(unnamed)"));
          if (t.description) li.appendChild(mkEl("span", "mcp-tool-desc", " — " + t.description));
          ul.appendChild(li);
        });
        mcpTestResult.appendChild(ul);
      }
    } catch (e) {
      mcpTestResult.className = "mcp-test-result err";
      mcpTestResult.textContent = "✗ " + mcpErrorText(e);
    } finally {
      btnMcpTest.disabled = false; btnMcpTest.textContent = "测试连接";
    }
  });

  mcpForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const b = buildMcpPayload(false);
    if (b.error) { showMcpFormError(b.error, b.field); return; }
    if (!mcpState.etag) await loadMcpServers({ silent: true });
    const editing = mcpEditing;
    btnMcpSave.disabled = true; btnMcpSave.textContent = "保存中…";
    // PATCH keeps omitted keys, so a cleared command must be sent as "" for
    // the backend to fall back to the default Python runtime.
    if (editing && b.payload.transport === "stdio" && !("command" in b.payload)) b.payload.command = "";
    try {
      const { data } = await mcpRequest(
        editing ? "PATCH" : "POST",
        editing ? MCP_API.item(editing.name) : MCP_API.create,
        b.payload,
        { ifMatch: mcpState.etag, timeoutMs: 10000 },
      );
      if (data.etag) mcpState.etag = data.etag;
      delete mcpState.testResults[b.payload.name];
      closeMcpModal();
      showToast("MCP 配置已保存", "success", 3000);
      mcpState.pending = data.pending_restart !== false;
      renderMcpPending();
      await loadMcpServers({ silent: true });
    } catch (e) {
      showMcpFormError((editing ? "保存失败：" : "添加失败：") + mcpErrorText(e));
      if (e.code === "ETAG_MISMATCH") await loadMcpServers({ silent: true });
    } finally {
      btnMcpSave.disabled = false; btnMcpSave.textContent = "保存";
    }
  });

  $("mcpModalClose").addEventListener("click", closeMcpModal);
  $("btnMcpCancel").addEventListener("click", closeMcpModal);
  mcpModal.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") { ev.stopPropagation(); closeMcpModal(); }
  });
  btnMcpAdd.addEventListener("click", () => openMcpModal(null, btnMcpAdd));
  btnMcpRefresh.addEventListener("click", () => loadMcpServers({ toast: true }));
  $("btnMcpRestartNow").addEventListener("click", triggerRestartAgent);
  // The card lives in the settings panel; refresh whenever the panel opens.
  msBtn.addEventListener("click", () => {
    if (!msPanel.classList.contains("hidden")) loadMcpServers({ silent: true });
  });
  loadMcpServers({ silent: true });

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

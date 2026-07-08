import { api } from "./api.js";
import { state } from "./state.js";
import { renderWindow, renderScanResults, formatPercent } from "./render.js";
import { renderSessionOptions, sessionMeta } from "./sessions.js";
import { initTheme, toggleTheme } from "./theme.js";

const el = {};

function cacheElements() {
  const ids = [
    "themeToggle", "exportLink", "sessionSelect", "deleteSession", "sessionMeta",
    "uploadForm", "uploadStatus", "progressFill", "progressBar", "progressText",
    "windowPosition", "windowDetail", "labelForm", "saveStatus", "confidence",
    "confidenceText", "useForTraining", "notes", "mockToggle", "prevButton",
    "nextButton", "scanButton", "scanCancel", "scanStatus", "scanProgressBar",
    "scanProgressFill", "scanProgressText", "scanResults",
    "settingsButton", "settingsDialog",
    "settingsForm", "aiBaseUrl", "aiModel", "aiApiKey", "clearKey",
    "testConnection", "settingsCancel", "settingsClose", "settingsStatus",
    "keyStatus", "aiKeyBadge", "exportAiLink",
  ];
  for (const id of ids) {
    el[id] = document.getElementById(id);
  }
}

// --- progress + session meta ---------------------------------------------

function updateProgress(progress) {
  const total = progress?.total || 0;
  const labeled = progress?.labeled || 0;
  const percent = total > 0 ? (labeled / total) * 100 : 0;
  el.progressFill.style.width = `${percent}%`;
  el.progressBar.setAttribute("aria-valuenow", String(Math.round(percent)));
  el.progressText.textContent = `${labeled} / ${total} labeled`;
}

function activeSession() {
  return state.sessions.find((s) => String(s.id) === String(state.activeSessionId)) || null;
}

function refreshExportLink() {
  const active = state.activeSessionId;
  if (active) {
    el.exportLink.setAttribute("href", api.exportUrl(active, "csv"));
    el.exportLink.setAttribute("aria-disabled", "false");
    el.exportAiLink.setAttribute("href", api.exportAiUrl(active, "csv"));
    el.exportAiLink.setAttribute("aria-disabled", "false");
  } else {
    el.exportLink.setAttribute("href", "#");
    el.exportLink.setAttribute("aria-disabled", "true");
    el.exportAiLink.setAttribute("href", "#");
    el.exportAiLink.setAttribute("aria-disabled", "true");
  }
}

function updateSessionMeta() {
  el.sessionMeta.textContent = sessionMeta(activeSession());
  el.deleteSession.disabled = !state.activeSessionId;
}

// --- sessions -------------------------------------------------------------

async function loadSessions() {
  const payload = await api.listSessions();
  state.sessions = payload.sessions || [];
  renderSessionOptions(el.sessionSelect, state.sessions, state.activeSessionId);
  updateSessionMeta();
  refreshExportLink();
  return state.sessions;
}

async function setActiveSession(id) {
  stopScanPolling();
  state.activeSessionId = id ? Number(id) : null;
  renderSessionOptions(el.sessionSelect, state.sessions, state.activeSessionId);
  updateSessionMeta();
  refreshExportLink();
  if (!state.activeSessionId) {
    clearWindow();
    updateProgress({ total: 0, labeled: 0 });
    resetScanPanel();
    return;
  }
  await loadNextUnlabeled();
  // Restore persisted scan results or reattach to a running scan.
  await pollScan();
}

async function deleteActiveSession() {
  if (!state.activeSessionId) {
    return;
  }
  const active = activeSession();
  if (!window.confirm(`Delete session "${active ? active.source_name : ""}" and its labels?`)) {
    return;
  }
  await api.deleteSession(state.activeSessionId);
  state.activeSessionId = null;
  await loadSessions();
  const first = state.sessions[0];
  await setActiveSession(first ? first.id : null);
}

// --- windows --------------------------------------------------------------

function clearWindow() {
  state.currentWindow = null;
  el.windowPosition.textContent = "No window";
  el.windowDetail.innerHTML = `<p class="empty">Select or create a session to start labeling.</p>`;
  clearLabelForm();
}

function renderWindowPayload(payload) {
  state.currentSeq = payload.window.seq;
  state.currentWindow = payload.window;
  state.currentLabel = payload.label;
  state.total = payload.total;
  updateProgress(payload.progress);
  el.windowPosition.textContent = `${payload.index + 1} / ${payload.total}`;
  renderWindow(el.windowDetail, payload.window, payload.index, payload.total, payload.label);
  fillLabelForm(payload.label);
}

async function loadNextUnlabeled() {
  const payload = await api.nextUnlabeled(state.activeSessionId);
  if (payload.done) {
    updateProgress(payload.progress);
    el.windowPosition.textContent = "Done";
    el.windowDetail.innerHTML = `<p class="empty">All windows in this session have labels.</p>`;
    clearLabelForm();
    return;
  }
  renderWindowPayload(payload);
}

async function loadWindow(seq) {
  if (!state.activeSessionId || state.total <= 0) {
    return;
  }
  try {
    const payload = await api.getWindow(state.activeSessionId, seq);
    renderWindowPayload(payload);
  } catch (error) {
    el.saveStatus.textContent = error.message;
  }
}

// --- label form -----------------------------------------------------------

function clearLabelForm() {
  document.querySelectorAll('input[name="label"]').forEach((input) => {
    input.checked = false;
  });
  el.confidence.value = 1;
  el.useForTraining.checked = true;
  el.notes.value = "";
  state.trainingAutoUnchecked = false;
  updateConfidenceText();
}

function fillLabelForm(savedRecord) {
  clearLabelForm();
  if (!savedRecord) {
    return;
  }
  const label = savedRecord.label || {};
  const input = document.querySelector(`input[name="label"][value="${label.label}"]`);
  if (input) {
    input.checked = true;
  }
  el.confidence.value = label.confidence ?? 1;
  el.useForTraining.checked = Boolean(label.use_for_training);
  el.notes.value = label.reason || "";
  updateConfidenceText();
  el.saveStatus.textContent = "Saved label loaded";
}

function applySuggestion(suggestion, source) {
  const input = document.querySelector(`input[name="label"][value="${suggestion.label}"]`);
  if (input) {
    input.checked = true;
  }
  el.confidence.value = suggestion.confidence ?? 0.5;
  el.useForTraining.checked = Boolean(suggestion.use_for_training);

  const evidence = (suggestion.evidence || []).join("; ");
  const quality = (suggestion.data_quality_flags || []).join("; ");
  const reviewText = suggestion.human_review_needed ? "Human review needed: yes" : "Human review needed: no";
  el.notes.value = [
    `AI suggestion source: ${source}`,
    `Reason: ${suggestion.reason || "-"}`,
    evidence ? `Evidence: ${evidence}` : "",
    quality ? `Data quality: ${quality}` : "",
    reviewText,
  ]
    .filter(Boolean)
    .join("\n");
  updateConfidenceText();
}

function updateConfidenceText() {
  el.confidenceText.textContent = formatPercent(el.confidence.value);
  if (Number(el.confidence.value) < 0.55) {
    if (el.useForTraining.checked) {
      el.useForTraining.checked = false;
      state.trainingAutoUnchecked = true;
    }
  } else if (state.trainingAutoUnchecked) {
    el.useForTraining.checked = true;
    state.trainingAutoUnchecked = false;
  }
}

// --- actions --------------------------------------------------------------

async function uploadFile(event) {
  event.preventDefault();
  el.uploadStatus.textContent = "Uploading…";
  el.saveStatus.textContent = "Not saved";

  const formData = new FormData(el.uploadForm);
  if (!formData.get("max_rows")) {
    formData.delete("max_rows");
  }
  if (!formData.get("project")) {
    formData.delete("project");
  }

  try {
    const session = await api.createSession(formData);
    await loadSessions();
    state.activeSessionId = session.id;
    await setActiveSession(session.id);
    el.uploadStatus.textContent = `Loaded ${session.source_name}`;
  } catch (error) {
    el.uploadStatus.textContent = error.message;
  }
}

async function saveLabel(event) {
  event.preventDefault();
  if (!state.activeSessionId || !state.currentWindow) {
    el.saveStatus.textContent = "Select or create a session first";
    return;
  }
  const selected = document.querySelector('input[name="label"]:checked');
  if (!selected) {
    el.saveStatus.textContent = "Choose a label";
    return;
  }

  try {
    const result = await api.saveLabel(state.activeSessionId, {
      seq: state.currentSeq,
      label: selected.value,
      confidence: Number(el.confidence.value),
      use_for_training: el.useForTraining.checked,
      notes: el.notes.value,
    });
    el.saveStatus.textContent = "Saved";
    updateProgress(result.progress);
    await loadSessions();
    await loadNextUnlabeled();
  } catch (error) {
    el.saveStatus.textContent = error.message;
  }
}

// --- AI scan ----------------------------------------------------------------

let scanTimer = null;

function stopScanPolling() {
  if (scanTimer !== null) {
    clearInterval(scanTimer);
    scanTimer = null;
  }
}

function startScanPolling() {
  if (scanTimer === null) {
    scanTimer = setInterval(pollScan, 1500);
  }
}

function scanStatusText(payload) {
  const suffix = payload.source ? ` (${payload.source})` : "";
  switch (payload.status) {
    case "running":
      return `Scanning ${payload.done} / ${payload.total}${suffix}…`;
    case "done":
      return `Done — ${payload.results.length} flagged${payload.errors ? `, ${payload.errors} errors` : ""}${suffix}`;
    case "cancelled":
      return `Cancelled at ${payload.done} / ${payload.total}`;
    case "error":
      return `Scan failed: ${payload.error_detail || "unknown error"}`;
    default:
      return payload.results.length
        ? `${payload.results.length} flagged from a previous scan`
        : "Not scanned";
  }
}

function renderScanState(payload) {
  const percent = payload.total > 0 ? (payload.done / payload.total) * 100 : 0;
  el.scanProgressFill.style.width = `${percent}%`;
  el.scanProgressBar.setAttribute("aria-valuenow", String(Math.round(percent)));
  el.scanProgressText.textContent = `${payload.done} / ${payload.total} scanned`;
  el.scanStatus.textContent = scanStatusText(payload);
  el.scanButton.disabled = payload.status === "running";
  el.scanCancel.disabled = payload.status !== "running";
  renderScanResults(el.scanResults, payload.results, pickScanResult);
  if (payload.status === "running") {
    startScanPolling();
  } else {
    stopScanPolling();
  }
}

function resetScanPanel() {
  stopScanPolling();
  el.scanProgressFill.style.width = "0%";
  el.scanProgressBar.setAttribute("aria-valuenow", "0");
  el.scanProgressText.textContent = "0 / 0 scanned";
  el.scanStatus.textContent = "Not scanned";
  el.scanButton.disabled = false;
  el.scanCancel.disabled = true;
  el.scanResults.innerHTML = `<p class="empty">Run a scan to list windows flagged as potentially abnormal.</p>`;
}

async function pollScan() {
  if (!state.activeSessionId) {
    stopScanPolling();
    return;
  }
  try {
    renderScanState(await api.getScan(state.activeSessionId));
  } catch (error) {
    stopScanPolling();
    el.scanStatus.textContent = error.message;
  }
}

async function startScan() {
  if (!state.activeSessionId) {
    el.scanStatus.textContent = "Select or create a session first";
    return;
  }
  el.scanButton.disabled = true;
  try {
    renderScanState(await api.startScan(state.activeSessionId, { mock: el.mockToggle.checked }));
  } catch (error) {
    el.scanStatus.textContent = error.message;
    el.scanButton.disabled = false;
  }
}

async function cancelScan() {
  if (!state.activeSessionId) {
    return;
  }
  try {
    renderScanState(await api.cancelScan(state.activeSessionId));
  } catch (error) {
    el.scanStatus.textContent = error.message;
  }
}

async function pickScanResult(item) {
  if (!item) {
    return;
  }
  await loadWindow(item.seq);
  applySuggestion(item.suggestion, item.source);
  el.saveStatus.textContent = "AI scan suggestion loaded. Review before saving.";
}

// --- settings dialog --------------------------------------------------------

function renderKeyStatus(s) {
  const sourceText = {
    db: "API key configured — saved in this app.",
    env: "API key configured — from the DEEPSEEK_API_KEY environment variable.",
  };
  el.keyStatus.textContent = s.has_api_key
    ? `✓ ${sourceText[s.api_key_source] || "API key configured."}`
    : "No API key configured — AI scans fall back to the offline mock.";
  el.keyStatus.classList.toggle("ok", s.has_api_key);
}

function updateAiKeyBadge() {
  const configured = Boolean(state.config && state.config.deepseek_available);
  el.aiKeyBadge.textContent = configured ? "API key: configured" : "API key: missing";
  el.aiKeyBadge.classList.toggle("low", configured);
  el.aiKeyBadge.classList.toggle("unclear", !configured);
}

function setSettingsStatus(message, kind) {
  el.settingsStatus.textContent = message || "";
  el.settingsStatus.classList.toggle("ok", kind === "ok");
  el.settingsStatus.classList.toggle("error", kind === "error");
}

async function openSettings() {
  try {
    const s = await api.getSettings();
    el.aiBaseUrl.value = s.base_url || "";
    el.aiModel.value = s.model || "";
    el.aiApiKey.value = "";
    el.aiApiKey.placeholder = s.has_api_key ? "Configured — leave blank to keep" : "sk-…";
    el.aiApiKey.disabled = false;
    el.clearKey.checked = false;
    renderKeyStatus(s);
    setSettingsStatus("");
    el.settingsDialog.showModal();
  } catch (error) {
    el.saveStatus.textContent = error.message;
  }
}

function buildSettingsBody() {
  const body = {
    base_url: el.aiBaseUrl.value.trim(),
    model: el.aiModel.value.trim(),
  };
  if (el.clearKey.checked) {
    body.api_key = "";
  } else if (el.aiApiKey.value.trim()) {
    body.api_key = el.aiApiKey.value.trim();
  }
  return body;
}

async function saveSettings(event) {
  event.preventDefault();
  try {
    await api.saveSettings(buildSettingsBody());
    el.settingsDialog.close();
    state.config = await api.config();
    el.mockToggle.checked = !state.config.deepseek_available;
    updateAiKeyBadge();
  } catch (error) {
    setSettingsStatus(error.message, "error");
  }
}

async function testSettingsConnection() {
  el.testConnection.disabled = true;
  setSettingsStatus("Testing…");
  try {
    const result = await api.testSettings(buildSettingsBody());
    if (result.ok) {
      setSettingsStatus(`Connection OK (model: ${result.model})`, "ok");
    } else {
      setSettingsStatus(result.detail, "error");
    }
  } catch (error) {
    setSettingsStatus(error.message, "error");
  } finally {
    el.testConnection.disabled = false;
  }
}

// --- bootstrap ------------------------------------------------------------

function wireEvents() {
  el.themeToggle.addEventListener("click", toggleTheme);
  el.settingsButton.addEventListener("click", openSettings);
  el.settingsForm.addEventListener("submit", saveSettings);
  el.testConnection.addEventListener("click", testSettingsConnection);
  el.settingsCancel.addEventListener("click", () => el.settingsDialog.close());
  el.settingsClose.addEventListener("click", () => el.settingsDialog.close());
  el.clearKey.addEventListener("change", () => {
    el.aiApiKey.disabled = el.clearKey.checked;
  });
  el.uploadForm.addEventListener("submit", uploadFile);
  el.labelForm.addEventListener("submit", saveLabel);
  el.confidence.addEventListener("input", updateConfidenceText);
  el.useForTraining.addEventListener("change", () => {
    state.trainingAutoUnchecked = false;
  });
  el.prevButton.addEventListener("click", () => loadWindow(Math.max(0, state.currentSeq - 1)));
  el.scanButton.addEventListener("click", startScan);
  el.scanCancel.addEventListener("click", cancelScan);
  el.nextButton.addEventListener("click", () =>
    loadWindow(Math.min(state.total - 1, state.currentSeq + 1)),
  );
  el.sessionSelect.addEventListener("change", (event) => setActiveSession(event.target.value));
  el.deleteSession.addEventListener("click", deleteActiveSession);
  document.querySelectorAll('input[name="label"]').forEach((input) => {
    input.addEventListener("change", () => {
      if (input.value === "unclear") {
        el.useForTraining.checked = false;
      }
    });
  });
  el.exportLink.addEventListener("click", (event) => {
    if (el.exportLink.getAttribute("aria-disabled") === "true") {
      event.preventDefault();
    }
  });
  el.exportAiLink.addEventListener("click", (event) => {
    if (el.exportAiLink.getAttribute("aria-disabled") === "true") {
      event.preventDefault();
    }
  });
}

async function bootstrap() {
  cacheElements();
  initTheme();
  wireEvents();

  try {
    state.config = await api.config();
  } catch {
    state.config = { deepseek_available: false, default_model: "", labels: [] };
  }
  // Default to offline mock when no API key is configured so AI Suggest works.
  el.mockToggle.checked = !state.config.deepseek_available;
  updateAiKeyBadge();

  const sessions = await loadSessions();
  if (sessions.length) {
    await setActiveSession(sessions[0].id);
  } else {
    clearWindow();
  }
}

bootstrap();

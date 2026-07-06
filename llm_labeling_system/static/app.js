const state = {
  currentIndex: 0,
  currentWindow: null,
  currentLabel: null,
  total: 0,
  trainingAutoUnchecked: false,
};

const QUALITY_FLAG_LABELS = {
  missing_speed: "One or more rows have no GPS speed data",
  missing_heading: "One or more rows have no heading/direction data",
  missing_coordinates: "Latitude/longitude are missing or invalid",
  large_time_gap: "A gap over 120 seconds exists between consecutive data points",
};

const elements = {
  uploadForm: document.querySelector("#uploadForm"),
  uploadStatus: document.querySelector("#uploadStatus"),
  sessionText: document.querySelector("#sessionText"),
  progressFill: document.querySelector("#progressFill"),
  progressText: document.querySelector("#progressText"),
  windowPosition: document.querySelector("#windowPosition"),
  windowDetail: document.querySelector("#windowDetail"),
  labelForm: document.querySelector("#labelForm"),
  saveStatus: document.querySelector("#saveStatus"),
  confidence: document.querySelector("#confidence"),
  confidenceText: document.querySelector("#confidenceText"),
  useForTraining: document.querySelector("#useForTraining"),
  notes: document.querySelector("#notes"),
  prevButton: document.querySelector("#prevButton"),
  suggestButton: document.querySelector("#suggestButton"),
  nextButton: document.querySelector("#nextButton"),
};

elements.uploadForm.addEventListener("submit", uploadFile);
elements.labelForm.addEventListener("submit", saveLabel);
elements.confidence.addEventListener("input", updateConfidenceText);
elements.useForTraining.addEventListener("change", () => {
  state.trainingAutoUnchecked = false;
});
elements.prevButton.addEventListener("click", () => loadWindow(Math.max(0, state.currentIndex - 1)));
elements.suggestButton.addEventListener("click", suggestLabel);
elements.nextButton.addEventListener("click", () => loadWindow(Math.min(state.total - 1, state.currentIndex + 1)));

document.querySelectorAll('input[name="label"]').forEach((input) => {
  input.addEventListener("change", () => {
    if (input.value === "unclear") {
      elements.useForTraining.checked = false;
    }
  });
});

async function uploadFile(event) {
  event.preventDefault();
  elements.uploadStatus.textContent = "Uploading...";
  elements.saveStatus.textContent = "Not saved";

  const formData = new FormData(elements.uploadForm);
  if (!formData.get("max_rows")) {
    formData.delete("max_rows");
  }

  const response = await fetch("/api/manual/upload", {
    method: "POST",
    body: formData,
  });
  const payload = await response.json();
  if (!response.ok) {
    elements.uploadStatus.textContent = payload.detail || "Upload failed";
    return;
  }

  updateSession(payload);
  elements.uploadStatus.textContent = `Loaded ${payload.source_name}`;
  await loadNextUnlabeled();
}

async function loadSession() {
  const response = await fetch("/api/manual/session");
  const payload = await response.json();
  updateSession(payload);
  if (payload.has_session && payload.progress.total > 0) {
    await loadNextUnlabeled();
  }
}

async function loadNextUnlabeled() {
  const response = await fetch("/api/manual/next");
  const payload = await response.json();
  if (payload.done) {
    updateProgress(payload.progress);
    elements.windowPosition.textContent = "Done";
    elements.windowDetail.innerHTML = `<p class="empty">All windows in this session have labels.</p>`;
    clearLabelForm();
    return;
  }
  renderWindowPayload(payload);
}

async function loadNextUnlabeledAfter(startIndex) {
  for (let index = startIndex; index < state.total; index += 1) {
    const response = await fetch(`/api/manual/window?index=${index}`);
    const payload = await response.json();
    if (response.ok && !payload.label) {
      renderWindowPayload(payload);
      return;
    }
  }
  await loadNextUnlabeled();
}

async function loadWindow(index) {
  if (state.total <= 0) {
    return;
  }
  const response = await fetch(`/api/manual/window?index=${index}`);
  const payload = await response.json();
  if (!response.ok) {
    elements.saveStatus.textContent = payload.detail || "Could not load window";
    return;
  }
  renderWindowPayload(payload);
}

function renderWindowPayload(payload) {
  state.currentIndex = payload.index;
  state.currentWindow = payload.window;
  state.currentLabel = payload.label;
  state.total = payload.total;
  updateProgress(payload.progress);
  renderWindow(payload.window, payload.index, payload.total, payload.label);
  fillLabelForm(payload.label);
}

async function saveLabel(event) {
  event.preventDefault();
  if (!state.currentWindow) {
    elements.saveStatus.textContent = "Upload a file first";
    return;
  }

  const selected = document.querySelector('input[name="label"]:checked');
  if (!selected) {
    elements.saveStatus.textContent = "Choose a label";
    return;
  }

  const payload = {
    window_id: state.currentWindow.window_id,
    label: selected.value,
    confidence: Number(elements.confidence.value),
    use_for_training: elements.useForTraining.checked,
    notes: elements.notes.value,
  };

  const response = await fetch("/api/manual/labels", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) {
    elements.saveStatus.textContent = result.detail || "Save failed";
    return;
  }

  elements.saveStatus.textContent = "Saved";
  updateProgress(result.progress);
  await loadNextUnlabeledAfter(state.currentIndex + 1);
}

async function suggestLabel() {
  if (!state.currentWindow) {
    elements.saveStatus.textContent = "Upload a file first";
    return;
  }

  elements.suggestButton.disabled = true;
  elements.saveStatus.textContent = "Getting AI suggestion...";
  try {
    const response = await fetch("/api/manual/suggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ window_id: state.currentWindow.window_id }),
    });
    const result = await response.json();
    if (!response.ok) {
      elements.saveStatus.textContent = result.detail || "AI suggestion failed";
      return;
    }
    applySuggestion(result.suggestion, result.source);
    elements.saveStatus.textContent = `AI suggestion loaded from ${result.source}. Review before saving.`;
  } finally {
    elements.suggestButton.disabled = false;
  }
}

function updateSession(payload) {
  if (!payload.has_session) {
    elements.sessionText.textContent = "Waiting for upload";
    updateProgress(payload.progress);
    return;
  }
  elements.sessionText.textContent = `${payload.source_name} - ${payload.row_count} rows - window ${payload.window_size}, stride ${payload.stride}`;
  updateProgress(payload.progress);
}

function updateProgress(progress) {
  const total = progress?.total || 0;
  const labeled = progress?.labeled || 0;
  const percent = total > 0 ? (labeled / total) * 100 : 0;
  elements.progressFill.style.width = `${percent}%`;
  elements.progressText.textContent = `${labeled} / ${total} labeled`;
}

function renderWindow(windowData, index, total, savedLabel) {
  const summary = windowData.summary || {};
  const rows = windowData.rows || [];
  elements.windowPosition.textContent = `${index + 1} / ${total}`;
  elements.windowDetail.innerHTML = `
    <div class="row-top">
      <div>
        <h3>${escapeHtml(windowData.window_id)}</h3>
        <p class="window-id">${escapeHtml(windowData.vehicle_id)} - ${escapeHtml(windowData.start_time)} to ${escapeHtml(windowData.end_time)}</p>
      </div>
      ${savedLabel ? `<span class="pill ${escapeHtml(savedLabel.label.label)}">saved: ${escapeHtml(savedLabel.label.label)}</span>` : `<span class="pill unclear">unlabeled</span>`}
    </div>

    <div class="detail-grid">
      ${metric("Max GPS speed", `${formatValue(summary.max_gps_speed)} km/h`)}
      ${metric("Average GPS speed", `${formatValue(summary.avg_gps_speed)} km/h`, "Mean GPS speed across all points in this window")}
      ${metric("Speed delta", `${formatValue(summary.gps_speed_delta)} km/h`, "Change in GPS speed from the first point to the last point")}
      ${metric("Distance", `${formatValue(summary.distance_km)} km`, "Total distance traveled based on GPS coordinates")}
      ${metric("Heading change", `${formatValue(summary.total_heading_change)}\u00b0`, "Sum of heading changes between consecutive points")}
      ${metric("Brake count", formatValue(summary.brake_count), "Number of points where the brake signal was active")}
      ${metric("Turn signals", formatValue((summary.left_turn_count || 0) + (summary.right_turn_count || 0)), "Total points where left or right turn signal was active")}
      ${metric("Road changes", formatValue(summary.road_id_change_count), "Number of times the road segment ID changed")}
    </div>

    <div class="section">
      <h3>Context</h3>
      <p><strong>Districts:</strong> ${escapeHtml((summary.districts || []).join(", ") || "-")}</p>
      <p><strong>Vehicle states:</strong> ${escapeHtml((summary.vehicle_states || []).join(", ") || "-")}</p>
    </div>

    <div class="section">
      <h3>Data quality</h3>
      ${list(summary.data_quality_flags)}
    </div>

    <div class="section">
      <h3>Raw window rows</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th title="Timestamp of the GPS data point">time</th>
              <th title="GPS speed in km/h">speed</th>
              <th title="Vehicle Speed Sensor (wheel speed from odometer, km/h)">vss</th>
              <th title="Compass heading relative to true north, degrees">heading</th>
              <th title="Brake signal: 1 = applied, 0 = released">brake</th>
              <th title="Left turn signal: 1 = active, 0 = inactive">left</th>
              <th title="Right turn signal: 1 = active, 0 = inactive">right</th>
              <th title="Road segment ID from digital map">road</th>
              <th title="Administrative district name">district</th>
            </tr>
          </thead>
          <tbody>${rows.map(rowTemplate).join("")}</tbody>
        </table>
      </div>
    </div>
  `;
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
  elements.confidence.value = label.confidence ?? 1;
  elements.useForTraining.checked = Boolean(label.use_for_training);
  elements.notes.value = label.reason || "";
  updateConfidenceText();
  elements.saveStatus.textContent = "Saved label loaded";
}

function applySuggestion(suggestion, source) {
  const input = document.querySelector(`input[name="label"][value="${suggestion.label}"]`);
  if (input) {
    input.checked = true;
  }
  elements.confidence.value = suggestion.confidence ?? 0.5;
  elements.useForTraining.checked = Boolean(suggestion.use_for_training);

  const evidence = (suggestion.evidence || []).join("; ");
  const quality = (suggestion.data_quality_flags || []).join("; ");
  const reviewText = suggestion.human_review_needed ? "Human review needed: yes" : "Human review needed: no";
  elements.notes.value = [
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

function clearLabelForm() {
  document.querySelectorAll('input[name="label"]').forEach((input) => {
    input.checked = false;
  });
  elements.confidence.value = 1;
  elements.useForTraining.checked = true;
  elements.notes.value = "";
  state.trainingAutoUnchecked = false;
  updateConfidenceText();
}

function updateConfidenceText() {
  elements.confidenceText.textContent = formatPercent(elements.confidence.value);
  if (Number(elements.confidence.value) < 0.55) {
    if (elements.useForTraining.checked) {
      elements.useForTraining.checked = false;
      state.trainingAutoUnchecked = true;
    }
  } else if (state.trainingAutoUnchecked) {
    // Only restore the checkbox when this code unchecked it; keep manual unchecks.
    elements.useForTraining.checked = true;
    state.trainingAutoUnchecked = false;
  }
}

function rowTemplate(row) {
  return `
    <tr>
      <td>${escapeHtml(row.time || "")}</td>
      <td>${formatValue(row.gps_speed)}</td>
      <td>${formatValue(row.vss_speed)}</td>
      <td>${formatValue(row.heading)}</td>
      <td>${formatValue(row.brake_signal)}</td>
      <td>${formatValue(row.left_turn_signal)}</td>
      <td>${formatValue(row.right_turn_signal)}</td>
      <td>${escapeHtml(row.road_id || "")}</td>
      <td>${escapeHtml(row.district || "")}</td>
    </tr>
  `;
}

function metric(label, value, tooltip) {
  const titleAttr = tooltip ? ` title="${escapeHtml(tooltip)}"` : "";
  return `<div class="metric"${titleAttr}><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function list(items) {
  const safeItems = (items || []).filter(Boolean);
  if (!safeItems.length) {
    return `<p class="empty">None</p>`;
  }
  return `<ul class="evidence-list">${safeItems
    .map((item) => {
      const desc = QUALITY_FLAG_LABELS[item];
      const titleAttr = desc ? ` title="${escapeHtml(desc)}"` : "";
      return `<li${titleAttr}>${escapeHtml(item)}</li>`;
    })
    .join("")}</ul>`;
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  return `${Math.round(number * 100)}%`;
}

function formatValue(value) {
  // Number(null) is 0 in JavaScript; missing data must not look like a real zero.
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  return Number.isInteger(number) ? String(number) : number.toFixed(2);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

loadSession();

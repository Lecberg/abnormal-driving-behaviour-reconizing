const QUALITY_FLAG_LABELS = {
  missing_speed: "One or more rows have no GPS speed data",
  missing_heading: "One or more rows have no heading/direction data",
  missing_coordinates: "Latitude/longitude are missing or invalid",
  large_time_gap: "A gap over 120 seconds exists between consecutive data points",
};

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function formatValue(value) {
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

export function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  return `${Math.round(number * 100)}%`;
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

export function renderScanResults(container, results, onPick) {
  const items = results || [];
  if (!items.length) {
    container.innerHTML = `<p class="empty">No flagged windows. Run a scan to examine this session.</p>`;
    return;
  }
  container.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>AI label</th>
            <th>Risk</th>
            <th>Confidence</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          ${items
            .map((item) => {
              const s = item.suggestion;
              return `
                <tr data-seq="${Number(item.seq)}" tabindex="0" title="Open this window for review">
                  <td>${Number(item.seq) + 1}</td>
                  <td><span class="pill ${escapeHtml(s.label)}">${escapeHtml(s.label)}</span></td>
                  <td><span class="pill ${escapeHtml(s.risk_level || "unclear")}">${escapeHtml(s.risk_level || "-")}</span></td>
                  <td>${formatPercent(s.confidence)}</td>
                  <td class="scan-reason">${escapeHtml(s.reason || "-")}</td>
                </tr>
              `;
            })
            .join("")}
        </tbody>
      </table>
    </div>
  `;
  container.querySelectorAll("tbody tr").forEach((tr) => {
    const seq = Number(tr.dataset.seq);
    const item = items.find((it) => Number(it.seq) === seq);
    tr.addEventListener("click", () => onPick(item));
    tr.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        onPick(item);
      }
    });
  });
}

export function renderWindow(container, windowData, index, total, savedLabel) {
  const summary = windowData.summary || {};
  const rows = windowData.rows || [];
  const savedPill = savedLabel
    ? `<span class="pill ${escapeHtml(savedLabel.label.label)}">saved: ${escapeHtml(savedLabel.label.label)}</span>`
    : `<span class="pill unclear">unlabeled</span>`;

  container.innerHTML = `
    <div class="row-top">
      <div>
        <h3>${escapeHtml(windowData.window_id)}</h3>
        <p class="window-id">${escapeHtml(windowData.vehicle_id)} · ${escapeHtml(windowData.start_time)} → ${escapeHtml(windowData.end_time)}</p>
      </div>
      ${savedPill}
    </div>

    <div class="detail-grid">
      ${metric("Max GPS speed", `${formatValue(summary.max_gps_speed)} km/h`)}
      ${metric("Average GPS speed", `${formatValue(summary.avg_gps_speed)} km/h`, "Mean GPS speed across all points in this window")}
      ${metric("Speed delta", `${formatValue(summary.gps_speed_delta)} km/h`, "Change in GPS speed from the first point to the last point")}
      ${metric("Distance", `${formatValue(summary.distance_km)} km`, "Total distance traveled based on GPS coordinates")}
      ${metric("Heading change", `${formatValue(summary.total_heading_change)}°`, "Sum of heading changes between consecutive points")}
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
              <th title="Vehicle Speed Sensor (wheel speed, km/h)">vss</th>
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

import { escapeHtml } from "./render.js";

function sessionOptionText(session) {
  const p = session.progress || { labeled: 0, total: 0 };
  const project = session.project && session.project !== "default" ? `[${session.project}] ` : "";
  return `${project}${session.source_name} — ${p.labeled}/${p.total} labeled`;
}

export function renderSessionOptions(selectEl, sessions, activeId) {
  if (!sessions.length) {
    selectEl.innerHTML = `<option value="">No sessions yet — create one</option>`;
    selectEl.disabled = true;
    return;
  }
  selectEl.disabled = false;
  selectEl.innerHTML = sessions
    .map(
      (s) =>
        `<option value="${s.id}"${String(s.id) === String(activeId) ? " selected" : ""}>${escapeHtml(
          sessionOptionText(s),
        )}</option>`,
    )
    .join("");
}

export function sessionMeta(session) {
  if (!session) {
    return "No session selected";
  }
  return `${escapeHtml(session.source_name)} · ${session.row_count} rows · window ${session.window_size}, stride ${session.stride}`;
}

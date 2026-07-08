async function request(url, options) {
  const response = await fetch(url, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const detail = (payload && payload.detail) || `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return payload;
}

export const api = {
  config: () => request("/api/config"),
  listSessions: () => request("/api/sessions"),
  getSession: (id) => request(`/api/sessions/${id}`),
  createSession: (formData) => request("/api/sessions", { method: "POST", body: formData }),
  deleteSession: (id) => request(`/api/sessions/${id}`, { method: "DELETE" }),
  getWindow: (id, seq) => request(`/api/sessions/${id}/windows/${seq}`),
  nextUnlabeled: (id) => request(`/api/sessions/${id}/next-unlabeled`),
  saveLabel: (id, body) =>
    request(`/api/sessions/${id}/labels`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  suggest: (id, body) =>
    request(`/api/sessions/${id}/suggest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  exportUrl: (id, fmt = "csv") => `/api/sessions/${id}/export?fmt=${fmt}`,
  exportAiUrl: (id, fmt = "csv") => `/api/sessions/${id}/export?fmt=${fmt}&source=ai`,
  startScan: (id, body) =>
    request(`/api/sessions/${id}/scan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  getScan: (id) => request(`/api/sessions/${id}/scan`),
  cancelScan: (id) => request(`/api/sessions/${id}/scan`, { method: "DELETE" }),
  getSettings: () => request("/api/settings"),
  saveSettings: (body) =>
    request("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  testSettings: (body) =>
    request("/api/settings/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};

const STORAGE_KEY = "labeling-theme";

function systemPrefersDark() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export function currentTheme() {
  const explicit = document.documentElement.getAttribute("data-theme");
  if (explicit === "light" || explicit === "dark") {
    return explicit;
  }
  return systemPrefersDark() ? "dark" : "light";
}

function updateToggle() {
  const button = document.querySelector("#themeToggle");
  if (!button) {
    return;
  }
  const dark = currentTheme() === "dark";
  button.textContent = dark ? "☀ Light" : "☾ Dark";
  button.setAttribute("aria-pressed", String(dark));
}

export function initTheme() {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved === "light" || saved === "dark") {
    document.documentElement.setAttribute("data-theme", saved);
  }
  updateToggle();
}

export function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem(STORAGE_KEY, next);
  updateToggle();
}

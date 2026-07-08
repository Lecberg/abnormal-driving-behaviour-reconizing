export const state = {
  config: { deepseek_available: false, default_model: "", labels: [] },
  sessions: [],
  activeSessionId: null,
  currentSeq: 0,
  currentWindow: null,
  currentLabel: null,
  total: 0,
  trainingAutoUnchecked: false,
};
